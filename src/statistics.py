"""Patient-aware evaluation statistics for frozen ECG model audits.

The functions in this module calculate descriptive evaluation statistics only.
They do not fit model weights, probability calibrators, or decision thresholds.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


MetricValue = float | int | None


def _as_binary_arrays(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(targets, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("targets and probabilities must be aligned non-empty vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("targets must be binary")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    return y, p


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _calibration_coefficients(
    targets: np.ndarray, probabilities: np.ndarray,
) -> tuple[float | None, float | None]:
    """Return calibration-in-the-large and logistic calibration slope."""
    if np.unique(targets).size < 2:
        return None, None
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    scores = np.log(clipped / (1.0 - clipped))
    if not np.isfinite(scores).all():
        return None, None

    def nll(linear: np.ndarray) -> float:
        return float(np.sum(np.logaddexp(0.0, linear) - targets * linear))

    intercept_fit = minimize(
        lambda value: nll(scores + float(value[0])),
        x0=np.asarray([0.0]),
        method="L-BFGS-B",
    )
    intercept = float(intercept_fit.x[0]) if intercept_fit.success else None

    if np.allclose(scores, scores[0]):
        return intercept, None
    slope_fit = minimize(
        lambda values: nll(values[0] + values[1] * scores),
        x0=np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
    )
    slope = float(slope_fit.x[1]) if slope_fit.success else None
    return intercept, slope


def binary_performance(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
    include_calibration_coefficients: bool = True,
) -> dict[str, MetricValue]:
    """Discrimination, calibration, and locked-threshold binary metrics."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    y, p = _as_binary_arrays(targets, probabilities)
    predicted = p >= threshold
    positive = y == 1
    negative = ~positive
    tp = int((predicted & positive).sum())
    fp = int((predicted & negative).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & negative).sum())
    auc = None
    average_precision = None
    if positive.any() and negative.any():
        auc = float(roc_auc_score(y, p))
        average_precision = float(average_precision_score(y, p))
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    calibration_intercept, calibration_slope = (
        _calibration_coefficients(y, p)
        if include_calibration_coefficients else (None, None)
    )
    return {
        "n_records": int(len(y)),
        "positives": int(positive.sum()),
        "negatives": int(negative.sum()),
        "prevalence": float(positive.mean()),
        "auroc": auc,
        "average_precision": average_precision,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "threshold": float(threshold),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "positive_predictive_value": _ratio(tp, tp + fp),
        "negative_predictive_value": _ratio(tn, tn + fn),
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, fn + tp),
    }


def selective_performance(decisions: pd.DataFrame) -> dict[str, MetricValue]:
    """Summarize the fixed three-action policy on an audit population."""
    required = {"patient_id", "target", "final_action", "final_prediction"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"decisions are missing required columns: {missing}")
    if decisions.empty:
        raise ValueError("cannot summarize empty decisions")
    reduced = decisions.final_action == "trust_reduced"
    twelve = decisions.final_action == "trust_12_lead"
    expert = decisions.final_action == "expert_review"
    automated = ~expert
    predicted_positive = automated & (decisions.final_prediction == 1)
    predicted_negative = automated & (decisions.final_prediction == 0)
    actual_positive = decisions.target == 1
    actual_negative = decisions.target == 0
    correct = decisions.final_prediction == decisions.target
    true_automated_positive = predicted_positive & actual_positive
    true_automated_negative = predicted_negative & actual_negative

    return {
        "n_records": int(len(decisions)),
        "n_patients": int(decisions.patient_id.nunique()),
        "positive_records": int(actual_positive.sum()),
        "positive_patients": int(decisions.loc[actual_positive, "patient_id"].nunique()),
        "reduced_trust_rate": float(reduced.mean()),
        "twelve_lead_acquisition_rate": float((~reduced).mean()),
        "twelve_lead_rescue_rate": float(twelve.mean()),
        "expert_referral_rate": float(expert.mean()),
        "automated_coverage": float(automated.mean()),
        "automated_error": _ratio(int((automated & ~correct.fillna(False)).sum()), int(automated.sum())),
        "trusted_positive_records": int(predicted_positive.sum()),
        "trusted_positive_errors": int((predicted_positive & actual_negative).sum()),
        "trusted_negative_records": int(predicted_negative.sum()),
        "trusted_negative_errors": int((predicted_negative & actual_positive).sum()),
        "trusted_positive_error": _ratio(
            int((predicted_positive & actual_negative).sum()), int(predicted_positive.sum()),
        ),
        "trusted_negative_error": _ratio(
            int((predicted_negative & actual_positive).sum()), int(predicted_negative.sum()),
        ),
        "automated_sensitivity": _ratio(
            int((automated & actual_positive & (decisions.final_prediction == 1)).sum()),
            int((automated & actual_positive).sum()),
        ),
        "automated_specificity": _ratio(
            int((automated & actual_negative & (decisions.final_prediction == 0)).sum()),
            int((automated & actual_negative).sum()),
        ),
        "positive_automation_recall": _ratio(
            int(true_automated_positive.sum()), int(actual_positive.sum()),
        ),
        "negative_automation_recall": _ratio(
            int(true_automated_negative.sum()), int(actual_negative.sum()),
        ),
        "positive_referral_rate": _ratio(
            int((expert & actual_positive).sum()), int(actual_positive.sum()),
        ),
        "negative_referral_rate": _ratio(
            int((expert & actual_negative).sum()), int(actual_negative.sum()),
        ),
    }


def patient_level_selective_performance(
    decisions: pd.DataFrame,
) -> dict[str, MetricValue]:
    """Aggregate repeated ECG decisions into conservative patient-level events."""
    required = {"patient_id", "target", "final_action", "final_prediction"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"decisions are missing required columns: {missing}")
    if decisions.empty:
        raise ValueError("cannot summarize empty decisions")
    cluster_column = (
        "__bootstrap_cluster" if "__bootstrap_cluster" in decisions else "patient_id"
    )
    automated = decisions.final_action != "expert_review"
    predicted_positive = automated & (decisions.final_prediction == 1)
    predicted_negative = automated & (decisions.final_prediction == 0)
    errors = automated & (decisions.final_prediction != decisions.target).fillna(False)
    patient_events = pd.DataFrame({
        "cluster": decisions[cluster_column],
        "automated": automated,
        "automated_error": errors,
        "trusted_positive": predicted_positive,
        "trusted_positive_error": predicted_positive & (decisions.target == 0),
        "trusted_negative": predicted_negative,
        "trusted_negative_error": predicted_negative & (decisions.target == 1),
    }).groupby("cluster", observed=True).max()

    def event_rate(event: str, eligible: str) -> float | None:
        mask = patient_events[eligible].astype(bool)
        return _ratio(int(patient_events.loc[mask, event].sum()), int(mask.sum()))

    return {
        "n_patients": int(len(patient_events)),
        "patients_with_automated_decision_rate": float(patient_events.automated.mean()),
        "patient_any_automated_error_rate": event_rate("automated_error", "automated"),
        "patient_any_trusted_positive_error_rate": event_rate(
            "trusted_positive_error", "trusted_positive",
        ),
        "patient_any_trusted_negative_error_rate": event_rate(
            "trusted_negative_error", "trusted_negative",
        ),
    }


def flat_audit_metrics(decisions: pd.DataFrame) -> dict[str, MetricValue]:
    """Flatten the metrics used for patient-cluster bootstrap intervals."""
    target = decisions.target.to_numpy(dtype=np.int8)
    metrics: dict[str, MetricValue] = {}
    for prefix, column in (
        ("reduced", "reduced_probability"),
        ("twelve_lead", "twelve_probability"),
    ):
        stage = binary_performance(
            target,
            decisions[column].to_numpy(),
            include_calibration_coefficients=False,
        )
        for name, value in stage.items():
            if name not in {
                "n_records", "positives", "negatives", "threshold",
                "true_positive", "false_positive", "false_negative", "true_negative",
            }:
                metrics[f"{prefix}.{name}"] = value
    for name, value in selective_performance(decisions).items():
        if not name.startswith("n_") and not name.endswith("_records") and not name.endswith("_patients"):
            metrics[f"policy.{name}"] = value
    for name, value in patient_level_selective_performance(decisions).items():
        if not name.startswith("n_"):
            metrics[f"patient_policy.{name}"] = value
    return metrics


def resample_patient_clusters(
    frame: pd.DataFrame, rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample patients with replacement and carry all of each selected patient's ECGs."""
    if "patient_id" not in frame or frame.patient_id.isna().any():
        raise ValueError("complete patient_id values are required for cluster resampling")
    grouped_positions = frame.groupby("patient_id", sort=False, observed=True).indices
    patients = np.asarray(list(grouped_positions), dtype=object)
    if len(patients) == 0:
        raise ValueError("cannot resample an empty patient set")
    sampled = rng.choice(patients, size=len(patients), replace=True)
    position_chunks = [np.asarray(grouped_positions[patient_id]) for patient_id in sampled]
    return _assemble_patient_clusters(frame, position_chunks)


def _assemble_patient_clusters(
    frame: pd.DataFrame, position_chunks: Sequence[np.ndarray],
) -> pd.DataFrame:
    """Build one resampled frame from an ordered set of patient row positions."""
    positions = np.concatenate(position_chunks)
    cluster_ids = np.repeat(
        np.arange(len(position_chunks), dtype=np.int64),
        [len(chunk) for chunk in position_chunks],
    )
    result = frame.iloc[positions].copy().reset_index(drop=True)
    result["__bootstrap_cluster"] = cluster_ids
    return result


def _resample_patient_clusters_from_groups(
    frame: pd.DataFrame,
    patient_position_chunks: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Resample from precomputed row positions for repeated bootstrap draws."""
    sampled_indices = rng.integers(
        0, len(patient_position_chunks), size=len(patient_position_chunks),
    )
    position_chunks = [patient_position_chunks[index] for index in sampled_indices]
    return _assemble_patient_clusters(frame, position_chunks)


def patient_cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], Mapping[str, MetricValue]],
    n_bootstrap: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Percentile intervals from patient-cluster rather than record resampling."""
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap cannot be negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_bootstrap == 0:
        return {}
    if "patient_id" not in frame or frame.patient_id.isna().any():
        raise ValueError("complete patient_id values are required for cluster resampling")
    grouped_positions = frame.groupby("patient_id", sort=False, observed=True).indices
    position_chunks = [np.asarray(positions) for positions in grouped_positions.values()]
    if not position_chunks:
        raise ValueError("cannot resample an empty patient set")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        replicate = _resample_patient_clusters_from_groups(frame, position_chunks, rng)
        for name, value in metric_fn(replicate).items():
            if value is not None and isinstance(value, (int, float, np.integer, np.floating)):
                numeric = float(value)
                if math.isfinite(numeric):
                    samples.setdefault(name, []).append(numeric)
    tail = (1.0 - confidence) / 2.0
    intervals = {}
    for name, values in samples.items():
        if values:
            lower, upper = np.quantile(values, [tail, 1.0 - tail])
            intervals[name] = {"lower": float(lower), "upper": float(upper)}
    return intervals


def reliability_table(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Fixed-width reliability table; empty bins are omitted."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    y, p = _as_binary_arrays(targets, probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    rows = []
    for index in range(n_bins):
        selected = bins == index
        if selected.any():
            rows.append({
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "n_records": int(selected.sum()),
                "mean_probability": float(p[selected].mean()),
                "observed_prevalence": float(y[selected].mean()),
            })
    return pd.DataFrame(rows)


def empirical_risk_coverage_curve(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    coverage_grid: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Descriptive error-coverage curve ordered by distance from 0.5.

    This function never selects or updates a deployed threshold.
    """
    y, p = _as_binary_arrays(targets, probabilities)
    grid = np.asarray(
        coverage_grid if coverage_grid is not None else np.linspace(0.05, 1.0, 20),
        dtype=np.float64,
    )
    if grid.ndim != 1 or np.any((grid <= 0.0) | (grid > 1.0)):
        raise ValueError("coverage grid values must be in (0, 1]")
    confidence = np.abs(p - 0.5)
    order = np.argsort(-confidence, kind="stable")
    predicted = p >= 0.5
    rows = []
    for requested in grid:
        n_selected = max(1, int(math.ceil(float(requested) * len(y))))
        selected = order[:n_selected]
        rows.append({
            "requested_coverage": float(requested),
            "coverage": float(n_selected / len(y)),
            "n_records": n_selected,
            "confidence_cutoff": float(confidence[selected].min()),
            "empirical_error": float((predicted[selected] != y[selected]).mean()),
        })
    return pd.DataFrame(rows)
