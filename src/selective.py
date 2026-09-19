"""Diagnosis-specific calibration and sequential selective prediction.

This command consumes validation-fold prediction artifacts produced by
``src.evaluate``. It never reads test predictions. Development performance is
estimated with patient-grouped outer cross-fitting. Inside each outer training
partition, independent patient groups fit the probability calibrator and the
risk thresholds. A final deployable policy is then fit with the same split
discipline on all validation patients.

The three final actions are:

* ``trust_reduced``: act on a controlled-confidence reduced-lead prediction;
* ``trust_12_lead``: acquire a 12-lead ECG and act on its prediction;
* ``expert_review``: abstain because even the 12-lead prediction is uncontrolled.

Example:
    python -m src.selective \
      --evaluation-dir artifacts/evaluation/<run> \
      --diagnosis-group rhythm --diagnosis AFIB
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
import sklearn

from .config import LEAD_SUBSETS
from .evaluate import atomic_csv_gz, atomic_json_dump, sha256_file, sigmoid


FULL_LEAD_SET = "12-lead"
REDUCED_LEAD_SETS = [name for name in LEAD_SUBSETS if name != FULL_LEAD_SET]
SEQUENTIAL_METHOD = "fixed-grid-clopper-pearson-bonferroni-routed-v2"
METADATA_COLUMNS = [
    "patient_id", "age", "sex", "baseline_drift", "static_noise",
    "burst_noise", "electrodes_problems", "extra_beats", "pacemaker",
]


def fit_calibrator(logits: np.ndarray, targets: np.ndarray) -> Dict[str, object]:
    """Fit a monotone affine (Platt) calibrator by minimizing binary NLL.

    The positive slope preserves score ranking. The intercept is essential here:
    class-weighted BCE shifts the model's probability scale, which temperature
    scaling alone cannot correct.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int8)
    if logits.ndim != 1 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("logits and targets must be aligned one-dimensional arrays")
    if len(logits) == 0 or np.unique(targets).size < 2:
        return {
            "method": "platt",
            "slope": 1.0,
            "intercept": 0.0,
            "fitted": False,
            "reason": "single_class_or_empty",
        }
    if not np.isfinite(logits).all():
        raise ValueError("calibrator fitting received non-finite logits")

    def objective(parameters: np.ndarray) -> float:
        slope = math.exp(float(parameters[0]))
        intercept = float(parameters[1])
        probabilities = sigmoid(slope * logits + intercept)
        return float(log_loss(targets, probabilities, labels=[0, 1]))

    result = minimize(
        objective,
        x0=np.asarray([0.0, 0.0]),
        bounds=[(math.log(0.05), math.log(20.0)), (-20.0, 20.0)],
        method="L-BFGS-B",
    )
    slope = float(math.exp(result.x[0])) if result.success else 1.0
    intercept = float(result.x[1]) if result.success else 0.0
    return {
        "method": "platt",
        "slope": slope,
        "intercept": intercept,
        "fitted": bool(result.success),
        "reason": None if result.success else str(result.message),
        "objective": float(result.fun) if result.success else None,
    }


def calibrated_probability(
    logits: np.ndarray, calibrator: Mapping[str, object],
) -> np.ndarray:
    slope = float(calibrator["slope"])
    intercept = float(calibrator["intercept"])
    if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
        raise ValueError("calibrator slope/intercept are invalid")
    return sigmoid(slope * np.asarray(logits, dtype=np.float64) + intercept)


def expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, n_bins: int = 10,
) -> float:
    """Equal-width expected calibration error."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(targets) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(probabilities, edges[1:-1]), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bins == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(targets[mask].mean())
            )
    return float(error)


def calibration_metrics(
    targets: np.ndarray, probabilities: np.ndarray, n_bins: int = 10,
) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return {
        "log_loss": float(log_loss(targets, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(targets, probabilities)),
        "ece": expected_calibration_error(targets, probabilities, n_bins=n_bins),
    }


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    """One-sided exact binomial upper confidence limit."""
    if n <= 0:
        return float("nan")
    if errors < 0 or errors > n:
        raise ValueError("errors must be between zero and n")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if errors == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


def threshold_grid(direction: str, size: int) -> np.ndarray:
    if size < 2:
        raise ValueError("threshold grid size must be at least two")
    if direction == "positive":
        return np.linspace(0.5, 0.999, size)
    if direction == "negative":
        return np.linspace(0.001, 0.5, size)
    raise ValueError(f"unknown direction: {direction}")


def risk_coverage_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    direction: str,
    max_risk: float | None,
    family_delta: float,
    min_trusted: int,
    grid_size: int,
) -> pd.DataFrame:
    """Evaluate a fixed threshold family with simultaneous exact risk bounds."""
    if max_risk is not None and not 0 < max_risk < 1:
        raise ValueError("max_risk must be in (0, 1)")
    if min_trusted <= 0:
        raise ValueError("min_trusted must be positive")
    thresholds = threshold_grid(direction, grid_size)
    per_threshold_delta = family_delta / grid_size
    rows = []
    for threshold in thresholds:
        if direction == "positive":
            trusted = probabilities >= threshold
            errors = int(((targets == 0) & trusted).sum())
        else:
            # At exactly 0.5 stage_prediction makes a positive call.
            trusted = (probabilities <= threshold) & (probabilities < 0.5)
            errors = int(((targets == 1) & trusted).sum())
        n_trusted = int(trusted.sum())
        empirical_risk = errors / n_trusted if n_trusted else None
        upper = (
            clopper_pearson_upper(errors, n_trusted, per_threshold_delta)
            if n_trusted else None
        )
        rows.append({
            "direction": direction,
            "threshold": float(threshold),
            "n_trusted": n_trusted,
            "errors": errors,
            "coverage": float(n_trusted / len(targets)) if len(targets) else 0.0,
            "empirical_risk": empirical_risk,
            "risk_upper": upper,
            "eligible": bool(
                max_risk is not None and n_trusted >= min_trusted
                and upper is not None and upper <= max_risk
            ),
        })
    return pd.DataFrame(rows)


def select_threshold(curve: pd.DataFrame, direction: str) -> Dict[str, object]:
    eligible = curve[curve.eligible].copy()
    if eligible.empty:
        return {
            "direction": direction,
            "feasible": False,
            "threshold": None,
            "n_trusted": 0,
            "errors": 0,
            "coverage": 0.0,
            "empirical_risk": None,
            "risk_upper": None,
        }
    chosen = eligible.sort_values(
        ["n_trusted", "risk_upper"], ascending=[False, True],
    ).iloc[0]
    return {
        "direction": direction,
        "feasible": True,
        "threshold": float(chosen.threshold),
        "n_trusted": int(chosen.n_trusted),
        "errors": int(chosen.errors),
        "coverage": float(chosen.coverage),
        "empirical_risk": float(chosen.empirical_risk),
        "risk_upper": float(chosen.risk_upper),
    }


def fit_stage_policy(
    calibration_frame: pd.DataFrame,
    risk_frame: pd.DataFrame,
    max_positive_risk: float | None,
    max_negative_risk: float | None,
    confidence: float,
    min_trusted: int,
    grid_size: int,
) -> tuple[Dict[str, object], pd.DataFrame]:
    """Fit probability calibration on one partition and risk thresholds on another."""
    calibration_fit = fit_calibrator(
        calibration_frame.logit.to_numpy(), calibration_frame.target.to_numpy(),
    )
    risk_probabilities = calibrated_probability(
        risk_frame.logit.to_numpy(), calibration_fit,
    )
    # Four threshold families are selected by the sequential policy: positive and
    # negative at both reduced and 12-lead stages. This Bonferroni allocation is
    # persisted so the statistical contract is explicit.
    family_delta = (1.0 - confidence) / 4.0
    curves = []
    thresholds = {}
    for direction in ("positive", "negative"):
        risk_target = (
            max_positive_risk if direction == "positive" else max_negative_risk
        )
        curve = risk_coverage_curve(
            risk_frame.target.to_numpy(), risk_probabilities, direction,
            max_risk=risk_target, family_delta=family_delta,
            min_trusted=min_trusted, grid_size=grid_size,
        )
        curve["max_risk"] = risk_target
        curves.append(curve)
        thresholds[direction] = select_threshold(curve, direction)
    policy = {
        "calibration": calibration_fit,
        "thresholds": thresholds,
        "calibration_fit_records": int(len(calibration_frame)),
        "risk_calibration_records": int(len(risk_frame)),
        "risk_control": {
            "method": "fixed-grid-clopper-pearson-bonferroni",
            "max_positive_risk": max_positive_risk,
            "max_negative_risk": max_negative_risk,
            "confidence": confidence,
            "family_delta": family_delta,
            "grid_size_per_direction": grid_size,
            "min_trusted": min_trusted,
        },
    }
    return policy, pd.concat(curves, ignore_index=True)


def stage_prediction(
    probabilities: np.ndarray, policy: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary predictions and whether each prediction is trusted."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int8)
    trusted = np.zeros(len(probabilities), dtype=bool)
    positive = policy["thresholds"]["positive"]
    negative = policy["thresholds"]["negative"]
    if positive["feasible"]:
        trusted |= probabilities >= float(positive["threshold"])
    if negative["feasible"]:
        trusted |= (probabilities <= float(negative["threshold"])) & (probabilities < 0.5)
    return predictions, trusted


def apply_sequential_policy(
    paired: pd.DataFrame,
    reduced_policy: Mapping[str, object],
    twelve_policy: Mapping[str, object],
) -> pd.DataFrame:
    """Apply reduced-lead triage, followed by 12-lead rescue or expert review."""
    result = paired.copy()
    result["reduced_probability"] = calibrated_probability(
        result.reduced_logit.to_numpy(), reduced_policy["calibration"],
    )
    result["twelve_probability"] = calibrated_probability(
        result.twelve_logit.to_numpy(), twelve_policy["calibration"],
    )
    reduced_prediction, reduced_trusted = stage_prediction(
        result.reduced_probability.to_numpy(), reduced_policy,
    )
    twelve_prediction, twelve_trusted = stage_prediction(
        result.twelve_probability.to_numpy(), twelve_policy,
    )
    result["reduced_prediction"] = reduced_prediction
    result["twelve_prediction"] = twelve_prediction
    result["initial_action"] = np.where(reduced_trusted, "trust_reduced", "obtain_12_lead")
    result["final_action"] = np.where(
        reduced_trusted,
        "trust_reduced",
        np.where(twelve_trusted, "trust_12_lead", "expert_review"),
    )
    final_prediction = np.where(
        reduced_trusted,
        reduced_prediction.astype(float),
        np.where(twelve_trusted, twelve_prediction.astype(float), np.nan),
    )
    result["final_prediction"] = pd.array(final_prediction, dtype="Int8")
    correctness = np.where(
        np.isnan(final_prediction),
        np.nan,
        (final_prediction == result.target.to_numpy()).astype(float),
    )
    result["correct"] = pd.array(correctness, dtype="boolean")
    return result


def split_calibration_and_risk(
    frame: pd.DataFrame, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Random 50/50 patient split without consulting risk outcomes.

    Single-class calibration uses fit_calibrator's explicit identity fallback;
    risk partitions may legitimately have zero observed errors or positives.
    """
    if frame.patient_id.isna().any():
        raise ValueError("complete patient IDs are required for risk partitioning")
    patients = frame.patient_id.unique()
    if len(patients) < 4:
        raise ValueError("need at least four patients for independent calibration and risk partitions")
    patients = np.random.default_rng(seed).permutation(patients)
    calibration = frame.patient_id.isin(patients[:len(patients) // 2]).to_numpy()
    return np.flatnonzero(calibration), np.flatnonzero(~calibration)


def load_prediction_file(
    evaluation_dir: Path, lead_set: str, diagnosis_group: str, diagnosis: str,
) -> tuple[pd.DataFrame, Path]:
    path = evaluation_dir / "predictions" / f"val__{lead_set}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    required = {"ecg_id", "split", "lead_set", "diagnosis_group", "diagnosis", "target", "logit"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame[
        (frame.split == "val")
        & (frame.lead_set == lead_set)
        & (frame.diagnosis_group == diagnosis_group)
        & (frame.diagnosis == diagnosis)
    ].copy()
    if frame.empty:
        raise ValueError(f"no {diagnosis_group}/{diagnosis} validation rows in {path}")
    if frame.ecg_id.duplicated().any():
        raise ValueError(f"duplicate ECG IDs in {path}")
    if "patient_id" not in frame or frame.patient_id.isna().any():
        raise ValueError(f"patient_id is required and must be complete in {path}")
    if frame.target.nunique() < 2:
        raise ValueError(f"{diagnosis} has only one target class in {path}")
    if not np.isfinite(frame.logit.to_numpy()).all():
        raise ValueError(f"non-finite logits in {path}")
    return frame.sort_values("ecg_id").reset_index(drop=True), path


def pair_lead_predictions(reduced: pd.DataFrame, twelve: pd.DataFrame) -> pd.DataFrame:
    metadata = [column for column in METADATA_COLUMNS if column in reduced.columns]
    left = reduced[["ecg_id", "target", "logit", *metadata]].rename(
        columns={"logit": "reduced_logit"},
    )
    right = twelve[["ecg_id", "patient_id", "target", "logit"]].rename(
        columns={"target": "twelve_target", "logit": "twelve_logit", "patient_id": "twelve_patient_id"},
    )
    paired = left.merge(right, on="ecg_id", how="inner", validate="one_to_one")
    if len(paired) != len(reduced) or len(paired) != len(twelve):
        raise ValueError("reduced and 12-lead prediction files have different ECG IDs")
    if not np.array_equal(paired.target.to_numpy(), paired.twelve_target.to_numpy()):
        raise ValueError("reduced and 12-lead targets disagree")
    if not paired.patient_id.equals(paired.twelve_patient_id):
        raise ValueError("reduced and 12-lead patient IDs disagree")
    return paired.drop(columns=["twelve_target", "twelve_patient_id"])


def fit_sequential_policy(
    paired: pd.DataFrame,
    seed: int,
    max_positive_risk: float | None,
    max_negative_risk: float | None,
    confidence: float,
    min_trusted: int,
    grid_size: int,
) -> tuple[Dict[str, object], pd.DataFrame]:
    """Fit disjoint branches with independent patient risk partitions.

    Conditional on calibration and the frozen reduced policy, the rescue grid
    is evaluated only on referred records from a previously unused partition.
    The four direction/branch bounds have total failure probability <= alpha.
    Each final direction is a mixture of two disjoint bounded branches. Bounds
    remain record-level (repeat ECG dependence is a separate known limitation).
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    calibration_idx, risk_idx = split_calibration_and_risk(paired, seed)
    calibration = paired.iloc[calibration_idx]
    risk = paired.iloc[risk_idx]
    # Split by patient without consulting outcomes; the rescue observations
    # cannot participate in selection of the reduced threshold/routing rule.
    patients = risk.patient_id.unique()
    if len(patients) < 2:
        raise ValueError("need two independent patient risk partitions")
    patients = np.random.default_rng(seed + 1).permutation(patients)
    reduced_patients = patients[:len(patients) // 2]
    reduced_risk = risk[risk.patient_id.isin(reduced_patients)]
    rescue_risk = risk[~risk.patient_id.isin(reduced_patients)]
    reduced_policy, reduced_curve = fit_stage_policy(
        calibration.rename(columns={"reduced_logit": "logit"}),
        reduced_risk.rename(columns={"reduced_logit": "logit"}),
        max_positive_risk, max_negative_risk, confidence, min_trusted, grid_size,
    )
    _, already_trusted = stage_prediction(
        calibrated_probability(rescue_risk.reduced_logit.to_numpy(), reduced_policy["calibration"]),
        reduced_policy,
    )
    referred_risk = rescue_risk.loc[~already_trusted]
    twelve_policy, twelve_curve = fit_stage_policy(
        calibration.rename(columns={"twelve_logit": "logit"}),
        referred_risk.rename(columns={"twelve_logit": "logit"}),
        max_positive_risk, max_negative_risk, confidence, min_trusted, grid_size,
    )
    reduced_curve.insert(0, "stage", "reduced")
    twelve_curve.insert(0, "stage", "12-lead")
    policy = {
        "method": SEQUENTIAL_METHOD,
        "reduced": reduced_policy,
        "12-lead": twelve_policy,
        "partition": {
            "seed": seed,
            "calibration_patients": int(calibration.patient_id.nunique()),
            "calibration_records": int(len(calibration)),
            "risk_patients": int(risk.patient_id.nunique()),
            "risk_records": int(len(risk)),
            "reduced_risk_patients": int(reduced_risk.patient_id.nunique()),
            "reduced_risk_records": int(len(reduced_risk)),
            "rescue_risk_patients": int(rescue_risk.patient_id.nunique()),
            "rescue_risk_records": int(len(rescue_risk)),
            "referred_risk_patients": int(referred_risk.patient_id.nunique()),
            "referred_risk_records": int(len(referred_risk)),
        },
    }
    return policy, pd.concat([reduced_curve, twelve_curve], ignore_index=True)


def cross_fitted_decisions(
    paired: pd.DataFrame,
    n_splits: int,
    seed: int,
    max_positive_risk: float | None,
    max_negative_risk: float | None,
    confidence: float,
    min_trusted: int,
    grid_size: int,
) -> tuple[pd.DataFrame, List[Dict[str, object]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    decisions = []
    fold_policies = []
    seen = np.zeros(len(paired), dtype=np.int8)
    for fold, (development_idx, audit_idx) in enumerate(splitter.split(
        paired, paired.target.to_numpy(), paired.patient_id.to_numpy(),
    )):
        development = paired.iloc[development_idx]
        audit = paired.iloc[audit_idx]
        if set(development.patient_id) & set(audit.patient_id):
            raise RuntimeError("patient leakage in outer cross-fitting split")
        policy, _ = fit_sequential_policy(
            development,
            seed=seed + 1000 + fold,
            max_positive_risk=max_positive_risk,
            max_negative_risk=max_negative_risk,
            confidence=confidence,
            min_trusted=min_trusted,
            grid_size=grid_size,
        )
        fold_result = apply_sequential_policy(
            audit, policy["reduced"], policy["12-lead"],
        )
        fold_result["fold"] = fold
        fold_result["development_records"] = len(development)
        decisions.append(fold_result)
        seen[audit_idx] += 1
        fold_policies.append({
            "fold": fold,
            "development_records": int(len(development)),
            "audit_records": int(len(audit)),
            "development_patients": int(development.patient_id.nunique()),
            "audit_patients": int(audit.patient_id.nunique()),
            "policy": policy,
        })
    if not np.all(seen == 1):
        raise RuntimeError("cross-fitting did not audit every record exactly once")
    return pd.concat(decisions, ignore_index=True).sort_values("ecg_id"), fold_policies


def decision_metrics(frame: pd.DataFrame) -> Dict[str, float | int | None]:
    n = len(frame)
    if n == 0:
        raise ValueError("cannot summarize empty decisions")
    reduced = frame.final_action == "trust_reduced"
    twelve = frame.final_action == "trust_12_lead"
    expert = frame.final_action == "expert_review"
    automated = ~expert
    positive = automated & (frame.final_prediction == 1)
    negative = automated & (frame.final_prediction == 0)
    actual_positive = frame.target == 1
    actual_negative = frame.target == 0

    def error(mask: pd.Series) -> float | None:
        if not mask.any():
            return None
        return float((frame.loc[mask, "final_prediction"] != frame.loc[mask, "target"]).mean())

    return {
        "n_records": int(n),
        "n_patients": int(frame.patient_id.nunique()),
        "reduced_trust_rate": float(reduced.mean()),
        "twelve_lead_acquisition_rate": float((~reduced).mean()),
        "twelve_lead_rescue_rate": float(twelve.mean()),
        "expert_referral_rate": float(expert.mean()),
        "automated_coverage": float(automated.mean()),
        "automated_error": error(automated),
        "trusted_positive_records": int(positive.sum()),
        "trusted_positive_errors": int((positive & actual_negative).sum()),
        "trusted_negative_records": int(negative.sum()),
        "trusted_negative_errors": int((negative & actual_positive).sum()),
        "trusted_positive_error": error(positive),
        "trusted_negative_error": error(negative),
        "automated_sensitivity": (
            float(((frame.final_prediction == 1) & (frame.target == 1)).sum()
                  / max(1, ((frame.target == 1) & automated).sum()))
            if ((frame.target == 1) & automated).any() else None
        ),
        "automated_specificity": (
            float(((frame.final_prediction == 0) & (frame.target == 0)).sum()
                  / max(1, ((frame.target == 0) & automated).sum()))
            if ((frame.target == 0) & automated).any() else None
        ),
        "positive_automation_recall": float(
            (positive & actual_positive).sum() / actual_positive.sum()
        ) if actual_positive.any() else None,
        "negative_automation_recall": float(
            (negative & actual_negative).sum() / actual_negative.sum()
        ) if actual_negative.any() else None,
        "positive_referral_rate": float(
            (expert & actual_positive).sum() / actual_positive.sum()
        ) if actual_positive.any() else None,
        "negative_referral_rate": float(
            (expert & actual_negative).sum() / actual_negative.sum()
        ) if actual_negative.any() else None,
    }


def patient_bootstrap_intervals(
    decisions: pd.DataFrame, n_bootstrap: int, seed: int,
) -> Dict[str, Dict[str, float | None]]:
    """Percentile intervals from efficient patient-level resampling."""
    if n_bootstrap <= 0:
        return {}
    expert = decisions.final_action == "expert_review"
    automated = decisions.final_action != "expert_review"
    final_positive = decisions.final_prediction == 1
    final_negative = decisions.final_prediction == 0
    target_positive = decisions.target == 1
    target_negative = decisions.target == 0
    counts = pd.DataFrame({
        "patient_id": decisions.patient_id,
        "records": 1,
        "reduced": (decisions.final_action == "trust_reduced").astype(int),
        "acquired": (decisions.final_action != "trust_reduced").astype(int),
        "rescued": (decisions.final_action == "trust_12_lead").astype(int),
        "expert": (decisions.final_action == "expert_review").astype(int),
        "automated": automated.astype(int),
        "automated_errors": (automated & ~decisions.correct.fillna(True)).astype(int),
        "trusted_positive": (automated & final_positive).astype(int),
        "trusted_positive_errors": (
            automated & final_positive & target_negative
        ).astype(int),
        "trusted_negative": (automated & final_negative).astype(int),
        "trusted_negative_errors": (
            automated & final_negative & target_positive
        ).astype(int),
        "automated_actual_positive": (automated & target_positive).astype(int),
        "automated_true_positive": (
            automated & target_positive & final_positive
        ).astype(int),
        "automated_actual_negative": (automated & target_negative).astype(int),
        "automated_true_negative": (
            automated & target_negative & final_negative
        ).astype(int),
        "actual_positive": target_positive.astype(int),
        "actual_negative": target_negative.astype(int),
        "expert_positive": (expert & target_positive).astype(int),
        "expert_negative": (expert & target_negative).astype(int),
    }).groupby("patient_id", observed=True).sum()
    columns = list(counts.columns)
    values = counts.to_numpy(dtype=np.float64)
    column = {name: index for index, name in enumerate(columns)}
    n_patients = len(counts)
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {}

    def ratio(totals: np.ndarray, numerator: str, denominator: str) -> float | None:
        denominator_value = totals[column[denominator]]
        if denominator_value <= 0:
            return None
        return float(totals[column[numerator]] / denominator_value)

    for _ in range(n_bootstrap):
        weights = rng.multinomial(n_patients, np.full(n_patients, 1.0 / n_patients))
        totals = weights @ values
        metrics = {
            "reduced_trust_rate": ratio(totals, "reduced", "records"),
            "twelve_lead_acquisition_rate": ratio(totals, "acquired", "records"),
            "twelve_lead_rescue_rate": ratio(totals, "rescued", "records"),
            "expert_referral_rate": ratio(totals, "expert", "records"),
            "automated_coverage": ratio(totals, "automated", "records"),
            "automated_error": ratio(totals, "automated_errors", "automated"),
            "trusted_positive_error": ratio(
                totals, "trusted_positive_errors", "trusted_positive",
            ),
            "trusted_negative_error": ratio(
                totals, "trusted_negative_errors", "trusted_negative",
            ),
            "automated_sensitivity": ratio(
                totals, "automated_true_positive", "automated_actual_positive",
            ),
            "automated_specificity": ratio(
                totals, "automated_true_negative", "automated_actual_negative",
            ),
            "positive_automation_recall": ratio(
                totals, "automated_true_positive", "actual_positive",
            ),
            "negative_automation_recall": ratio(
                totals, "automated_true_negative", "actual_negative",
            ),
            "positive_referral_rate": ratio(
                totals, "expert_positive", "actual_positive",
            ),
            "negative_referral_rate": ratio(
                totals, "expert_negative", "actual_negative",
            ),
        }
        for key, value in metrics.items():
            if value is not None and np.isfinite(value):
                samples.setdefault(key, []).append(value)
    intervals = {}
    for key, values in samples.items():
        if values:
            low, high = np.percentile(values, [2.5, 97.5])
            intervals[key] = {"lower": float(low), "upper": float(high)}
    return intervals


def crossfit_calibration_summary(decisions: pd.DataFrame) -> Dict[str, object]:
    target = decisions.target.to_numpy()
    return {
        "reduced": {
            "raw": calibration_metrics(target, sigmoid(decisions.reduced_logit.to_numpy())),
            "calibrated": calibration_metrics(target, decisions.reduced_probability.to_numpy()),
        },
        "12-lead": {
            "raw": calibration_metrics(target, sigmoid(decisions.twelve_logit.to_numpy())),
            "calibrated": calibration_metrics(target, decisions.twelve_probability.to_numpy()),
        },
    }


def reliability_points(
    targets: pd.Series, probabilities: pd.Series, n_bins: int = 10,
) -> pd.DataFrame:
    """Return equal-frequency reliability points for visualization."""
    if len(targets) == 0:
        return pd.DataFrame(columns=["mean_probability", "observed", "n"])
    bins = pd.qcut(probabilities, q=n_bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"target": targets, "probability": probabilities, "bin": bins})
    return frame.groupby("bin", observed=True).agg(
        mean_probability=("probability", "mean"),
        observed=("target", "mean"),
        n=("target", "size"),
    ).reset_index(drop=True)


def plot_selective_artifacts(
    decisions: pd.DataFrame, curves: pd.DataFrame, output_dir: Path,
) -> List[Path]:
    """Render cross-fitted calibration, risk–coverage, and action summaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    lead_sets = list(dict.fromkeys(decisions.lead_set.tolist()))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="black", alpha=0.5, label="ideal")
    for lead_set in lead_sets:
        subset = decisions[decisions.lead_set == lead_set]
        points = reliability_points(subset.target, subset.reduced_probability)
        ax.plot(
            points.mean_probability, points.observed, marker="o", linewidth=1.4,
            label=lead_set,
        )
    twelve = decisions[decisions.lead_set == lead_sets[0]]
    twelve_points = reliability_points(twelve.target, twelve.twelve_probability)
    ax.plot(
        twelve_points.mean_probability, twelve_points.observed,
        marker="s", linewidth=2.0, color="black", label="12-lead",
    )
    ax.set(xlabel="Cross-fitted calibrated probability", ylabel="Observed frequency",
           title="Diagnosis-specific calibration")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "calibration.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, direction in zip(axes, ("negative", "positive")):
        selected = curves[curves.direction == direction]
        for (lead_set, stage), group in selected.groupby(["lead_set", "stage"], sort=False):
            linestyle = "-" if stage == "reduced" else "--"
            axis.plot(
                group.coverage, group.risk_upper, linestyle=linestyle, linewidth=1.2,
                label=f"{lead_set} / {stage}",
            )
        for risk_target in sorted(selected.max_risk.unique()):
            axis.axhline(risk_target, color="black", linestyle=":", alpha=0.5)
        axis.set(
            xlabel="Coverage on risk-calibration partition",
            title=f"Trusted {direction} predictions",
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("One-sided simultaneous risk upper bound")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = output_dir / "risk_coverage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    action_counts = (
        decisions.groupby(["lead_set", "final_action"], observed=True).size()
        .unstack(fill_value=0).reindex(lead_sets)
    )
    action_rates = action_counts.div(action_counts.sum(axis=1), axis=0)
    action_rates = action_rates.reindex(
        columns=["trust_reduced", "trust_12_lead", "expert_review"], fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    action_rates.plot(
        kind="bar", stacked=True, ax=ax,
        color=["#4c78a8", "#f2cf5b", "#e45756"],
    )
    ax.set(xlabel="Reduced lead set", ylabel="Cross-fitted fraction of records",
           title="Sequential triage actions")
    ax.legend(title="Final action", fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = output_dir / "actions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def resolve_evaluation_dir(
    requested: Path | None, root: Path, required_lead_sets: Sequence[str],
) -> Path:
    if requested is not None:
        return requested.resolve()
    candidates = sorted(
        (path.parent for path in root.glob("*/manifest.json")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            manifest = json.loads((candidate / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        required = [FULL_LEAD_SET, *required_lead_sets]
        if manifest.get("status") == "complete" and all(
            (candidate / "predictions" / f"val__{lead}.csv.gz").exists()
            for lead in required
        ):
            return candidate.resolve()
    raise FileNotFoundError(
        "no complete validation evaluation contains all requested lead sets; "
        "pass --evaluation-dir"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument("--evaluation-root", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/selective"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--diagnosis-group", choices=["superclass", "subcode", "rhythm"], default="rhythm")
    parser.add_argument("--diagnosis", default="AFIB")
    parser.add_argument("--lead-sets", nargs="+", choices=REDUCED_LEAD_SETS, default=REDUCED_LEAD_SETS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-positive-risk", type=float, default=0.10,
        help="maximum error among trusted positive predictions",
    )
    parser.add_argument(
        "--max-negative-risk", type=float, default=0.02,
        help="maximum error among trusted negative predictions",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-trusted", type=int, default=25)
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("folds must be at least two")
    if not 0 < args.max_positive_risk < 1 or not 0 < args.max_negative_risk < 1:
        raise ValueError("positive and negative risk targets must be in (0, 1)")
    if not 0 < args.confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if args.min_trusted <= 0 or args.threshold_grid_size < 2:
        raise ValueError("min-trusted must be positive and threshold-grid-size at least two")
    if args.bootstrap < 0:
        raise ValueError("bootstrap cannot be negative")
    if len(set(args.lead_sets)) != len(args.lead_sets):
        raise ValueError("lead-sets must not contain duplicates")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    evaluation_dir = resolve_evaluation_dir(
        args.evaluation_dir, args.evaluation_root.resolve(), args.lead_sets,
    )
    source_manifest_path = evaluation_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("status") != "complete":
        raise ValueError("source evaluation manifest is not complete")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    default_name = f"{timestamp}-{args.diagnosis_group}-{args.diagnosis}"
    run_name = args.run_name or default_name
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run-name must be a single directory name")
    run_dir = args.out_dir.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    twelve, twelve_path = load_prediction_file(
        evaluation_dir, FULL_LEAD_SET, args.diagnosis_group, args.diagnosis,
    )
    input_files = {FULL_LEAD_SET: {"path": str(twelve_path), "sha256": sha256_file(twelve_path)}}
    manifest: Dict[str, object] = {
        "schema_version": 1,
        "status": "in_progress",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_evaluation": {
            "path": str(evaluation_dir),
            "manifest_sha256": sha256_file(source_manifest_path),
            "checkpoint": source_manifest.get("checkpoint"),
            "test_predictions_read": False,
        },
        "target": {"diagnosis_group": args.diagnosis_group, "diagnosis": args.diagnosis},
        "configuration": {
            "lead_sets": args.lead_sets,
            "outer_folds": args.folds,
            "seed": args.seed,
            "max_positive_risk": args.max_positive_risk,
            "max_negative_risk": args.max_negative_risk,
            "confidence": args.confidence,
            "min_trusted": args.min_trusted,
            "threshold_grid_size": args.threshold_grid_size,
            "bootstrap_replicates": args.bootstrap,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": input_files,
        "artifacts": [],
    }
    manifest_path = run_dir / "manifest.json"
    atomic_json_dump(manifest, manifest_path)

    all_decisions = []
    all_curves = []
    policies = []
    summaries = []
    for lead_index, lead_set in enumerate(args.lead_sets):
        print(f"[selective] diagnosis={args.diagnosis} lead_set={lead_set}")
        reduced, reduced_path = load_prediction_file(
            evaluation_dir, lead_set, args.diagnosis_group, args.diagnosis,
        )
        input_files[lead_set] = {"path": str(reduced_path), "sha256": sha256_file(reduced_path)}
        paired = pair_lead_predictions(reduced, twelve)
        positive_groups = paired.loc[paired.target == 1, "patient_id"].nunique()
        negative_groups = paired.loc[paired.target == 0, "patient_id"].nunique()
        if min(positive_groups, negative_groups) < args.folds:
            raise ValueError(
                f"{args.diagnosis}/{lead_set} has too few patient groups per class "
                f"for {args.folds} folds"
            )
        decisions, fold_policies = cross_fitted_decisions(
            paired,
            n_splits=args.folds,
            seed=args.seed,
            max_positive_risk=args.max_positive_risk,
            max_negative_risk=args.max_negative_risk,
            confidence=args.confidence,
            min_trusted=args.min_trusted,
            grid_size=args.threshold_grid_size,
        )
        decisions.insert(2, "diagnosis_group", args.diagnosis_group)
        decisions.insert(3, "diagnosis", args.diagnosis)
        decisions.insert(4, "lead_set", lead_set)
        all_decisions.append(decisions)

        final_policy, curves = fit_sequential_policy(
            paired,
            seed=args.seed + 100000 + lead_index,
            max_positive_risk=args.max_positive_risk,
            max_negative_risk=args.max_negative_risk,
            confidence=args.confidence,
            min_trusted=args.min_trusted,
            grid_size=args.threshold_grid_size,
        )
        curves.insert(0, "diagnosis_group", args.diagnosis_group)
        curves.insert(1, "diagnosis", args.diagnosis)
        curves.insert(2, "lead_set", lead_set)
        all_curves.append(curves)
        policies.append({
            "diagnosis_group": args.diagnosis_group,
            "diagnosis": args.diagnosis,
            "lead_set": lead_set,
            "policy": final_policy,
            "crossfit_fold_policies": fold_policies,
        })

        point = decision_metrics(decisions)
        summaries.append({
            "diagnosis_group": args.diagnosis_group,
            "diagnosis": args.diagnosis,
            "lead_set": lead_set,
            "crossfit": {
                "point": point,
                "patient_bootstrap_95_ci": patient_bootstrap_intervals(
                    decisions, args.bootstrap, args.seed + lead_index,
                ),
                "calibration": crossfit_calibration_summary(decisions),
            },
        })

    decisions_path = run_dir / "decisions.csv.gz"
    curves_path = run_dir / "risk_coverage.csv.gz"
    policy_path = run_dir / "policy.json"
    summary_path = run_dir / "summary.json"
    combined_decisions = pd.concat(all_decisions, ignore_index=True)
    combined_curves = pd.concat(all_curves, ignore_index=True)
    atomic_csv_gz(combined_decisions, decisions_path)
    atomic_csv_gz(combined_curves, curves_path)
    atomic_json_dump({
        "schema_version": 1,
        "status": "frozen_on_validation",
        "source_checkpoint_sha256": source_manifest.get("checkpoint", {}).get("sha256"),
        "risk_control_note": (
            "Exact one-sided binomial bounds are simultaneous over the fixed threshold "
            "grid via Bonferroni correction. Final test performance must be evaluated "
            "without refitting this policy."
        ),
        "policies": policies,
    }, policy_path)
    atomic_json_dump({"schema_version": 1, "results": summaries}, summary_path)
    plot_paths = plot_selective_artifacts(combined_decisions, combined_curves, run_dir)
    manifest["inputs"] = input_files
    manifest["artifacts"] = [
        str(path.relative_to(run_dir))
        for path in (decisions_path, curves_path, policy_path, summary_path, *plot_paths)
    ]
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json_dump(manifest, manifest_path)
    print(f"[selective] complete -> {run_dir}")


if __name__ == "__main__":
    main()
