"""Fit-free audit of a frozen selective policy on untouched test predictions.

This module intentionally imports policy-application helpers, but no fitting
helper is called. It fails closed when checkpoint, schema, split, or protocol
contracts disagree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn

from .evaluate import atomic_csv_gz, atomic_json_dump, sha256_file, sigmoid
from .selective import apply_sequential_policy
from .statistics import (
    binary_performance,
    empirical_risk_coverage_curve,
    flat_audit_metrics,
    patient_cluster_bootstrap_intervals,
    patient_level_selective_performance,
    reliability_table,
    selective_performance,
)


FULL_LEAD_SET = "12-lead"
QUALITY_COLUMNS = [
    "baseline_drift", "static_noise", "burst_noise", "electrodes_problems",
    "extra_beats", "pacemaker",
]
REQUIRED_PREDICTION_COLUMNS = {
    "ecg_id", "patient_id", "split", "lead_set", "diagnosis_group",
    "diagnosis", "target", "logit",
}


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _atomic_text_dump(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, object]:
    protocol = _load_json(path)
    required = {
        "schema_version", "protocol_id", "status", "split", "target",
        "required_lead_sets", "bootstrap_replicates", "bootstrap_seed",
        "calibration_bins", "minimum_subgroup_records",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"protocol is missing required fields: {missing}")
    target = protocol["target"]
    if not isinstance(target, dict) or not {"diagnosis_group", "diagnosis"} <= set(target):
        raise ValueError("protocol target must define diagnosis_group and diagnosis")
    if protocol["split"] != "test":
        raise ValueError("locked policy audit only permits split='test'")
    lead_sets = protocol["required_lead_sets"]
    if not isinstance(lead_sets, list) or not lead_sets or len(set(lead_sets)) != len(lead_sets):
        raise ValueError("required_lead_sets must be a non-empty unique list")
    if FULL_LEAD_SET in lead_sets:
        raise ValueError("required_lead_sets must contain reduced-lead sets only")
    if int(protocol["bootstrap_replicates"]) < 0:
        raise ValueError("bootstrap_replicates cannot be negative")
    if int(protocol["calibration_bins"]) < 2:
        raise ValueError("calibration_bins must be at least two")
    if int(protocol["minimum_subgroup_records"]) <= 0:
        raise ValueError("minimum_subgroup_records must be positive")
    return protocol


def load_test_prediction(
    evaluation_dir: Path,
    split: str,
    lead_set: str,
    diagnosis_group: str,
    diagnosis: str,
) -> tuple[pd.DataFrame, Path]:
    path = evaluation_dir / "predictions" / f"{split}__{lead_set}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    selected = frame[
        (frame.split == split)
        & (frame.lead_set == lead_set)
        & (frame.diagnosis_group == diagnosis_group)
        & (frame.diagnosis == diagnosis)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"no {diagnosis_group}/{diagnosis} {split} rows for {lead_set} in {path}"
        )
    if selected.ecg_id.duplicated().any():
        raise ValueError(f"duplicate ECG IDs in {path}")
    if selected.patient_id.isna().any():
        raise ValueError(f"patient_id must be complete in {path}")
    if selected.target.nunique() < 2:
        raise ValueError(f"{diagnosis} has only one target class in {path}")
    if not np.isfinite(selected.logit.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"non-finite logits in {path}")
    if not selected.target.isin([0, 1]).all():
        raise ValueError(f"non-binary targets in {path}")
    return selected.sort_values("ecg_id").reset_index(drop=True), path


def pair_test_predictions(reduced: pd.DataFrame, twelve: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        column for column in ["patient_id", "age", "sex", *QUALITY_COLUMNS]
        if column in reduced.columns
    ]
    left = reduced[["ecg_id", "target", "logit", *metadata]].rename(
        columns={"logit": "reduced_logit"},
    )
    right_columns = ["ecg_id", "patient_id", "target", "logit"]
    right = twelve[right_columns].rename(columns={
        "patient_id": "twelve_patient_id",
        "target": "twelve_target",
        "logit": "twelve_logit",
    })
    paired = left.merge(right, on="ecg_id", how="inner", validate="one_to_one")
    if len(paired) != len(reduced) or len(paired) != len(twelve):
        raise ValueError("reduced and 12-lead files contain different ECG IDs")
    if not np.array_equal(paired.target.to_numpy(), paired.twelve_target.to_numpy()):
        raise ValueError("reduced and 12-lead targets disagree")
    if not np.array_equal(
        paired.patient_id.to_numpy(), paired.twelve_patient_id.to_numpy(), equal_nan=False,
    ):
        raise ValueError("reduced and 12-lead patient IDs disagree")
    return paired.drop(columns=["twelve_target", "twelve_patient_id"])


def _policy_index(policy_payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if policy_payload.get("status") != "frozen_on_validation":
        raise ValueError("policy status must be frozen_on_validation")
    entries = policy_payload.get("policies")
    if not isinstance(entries, list) or not entries:
        raise ValueError("policy must contain at least one lead-specific policy")
    indexed: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each policy entry must be an object")
        lead_set = entry.get("lead_set")
        policy = entry.get("policy")
        if not isinstance(lead_set, str) or not isinstance(policy, dict):
            raise ValueError("each policy entry must define lead_set and policy")
        if lead_set in indexed:
            raise ValueError(f"duplicate policy for {lead_set}")
        if not {"reduced", FULL_LEAD_SET} <= set(policy):
            raise ValueError(f"{lead_set} policy must define reduced and 12-lead stages")
        for stage_name in ("reduced", FULL_LEAD_SET):
            stage = policy[stage_name]
            if not isinstance(stage, dict):
                raise ValueError(f"{lead_set}/{stage_name} stage must be an object")
            calibration = stage.get("calibration")
            thresholds = stage.get("thresholds")
            if not isinstance(calibration, dict) or not {"slope", "intercept"} <= set(calibration):
                raise ValueError(f"{lead_set}/{stage_name} has an invalid calibration")
            if not np.isfinite([
                float(calibration["slope"]), float(calibration["intercept"]),
            ]).all():
                raise ValueError(f"{lead_set}/{stage_name} calibration is non-finite")
            if not isinstance(thresholds, dict) or not {"positive", "negative"} <= set(thresholds):
                raise ValueError(f"{lead_set}/{stage_name} has invalid thresholds")
            for direction in ("positive", "negative"):
                threshold = thresholds[direction]
                if not isinstance(threshold, dict) or "feasible" not in threshold:
                    raise ValueError(f"{lead_set}/{stage_name}/{direction} threshold is invalid")
                if threshold["feasible"]:
                    value = threshold.get("threshold")
                    if value is None or not 0.0 <= float(value) <= 1.0:
                        raise ValueError(
                            f"{lead_set}/{stage_name}/{direction} threshold is out of range"
                        )
        indexed[lead_set] = entry
    return indexed


def validate_audit_contract(
    evaluation_manifest: Mapping[str, object],
    selective_manifest: Mapping[str, object],
    policy_payload: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Validate all immutable boundaries before reading any test prediction."""
    if evaluation_manifest.get("status") != "complete":
        raise ValueError("evaluation manifest status must be complete")
    evaluation = evaluation_manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation manifest is missing evaluation metadata")
    if "test" not in evaluation.get("splits", []) or not evaluation.get("test_evaluated"):
        raise ValueError("evaluation manifest does not declare a test export")
    if evaluation.get("max_n") is not None:
        raise ValueError("locked audit refuses a max_n-limited test export")
    if selective_manifest.get("status") != "complete":
        raise ValueError("selective manifest status must be complete")

    evaluation_checkpoint = evaluation_manifest.get("checkpoint")
    if not isinstance(evaluation_checkpoint, dict) or not evaluation_checkpoint.get("sha256"):
        raise ValueError("evaluation manifest is missing checkpoint SHA-256")
    checkpoint_hash = evaluation_checkpoint["sha256"]
    if policy_payload.get("source_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("policy and evaluation checkpoint SHA-256 values differ")
    selective_source = selective_manifest.get("source_evaluation")
    selective_checkpoint = (
        selective_source.get("checkpoint") if isinstance(selective_source, dict) else None
    )
    if not isinstance(selective_checkpoint, dict) or selective_checkpoint.get("sha256") != checkpoint_hash:
        raise ValueError("selective manifest checkpoint SHA-256 does not match evaluation")

    target = protocol["target"]
    assert isinstance(target, dict)
    diagnosis_group = str(target["diagnosis_group"])
    diagnosis = str(target["diagnosis"])
    schema = evaluation_manifest.get("label_schema")
    if not isinstance(schema, dict) or diagnosis not in schema.get(diagnosis_group, []):
        raise ValueError("protocol target is absent from the evaluation label schema")
    selective_target = selective_manifest.get("target")
    if selective_target != target:
        raise ValueError("selective manifest target does not match the protocol")

    indexed = _policy_index(policy_payload)
    required_leads = protocol["required_lead_sets"]
    assert isinstance(required_leads, list)
    primary = protocol.get("primary_analysis")
    if isinstance(primary, dict) and primary.get("lead_set") not in required_leads:
        raise ValueError("primary analysis lead set is absent from required_lead_sets")
    missing_policies = sorted(set(required_leads) - set(indexed))
    if missing_policies:
        raise ValueError(f"policy is missing required lead sets: {missing_policies}")
    exported_leads = evaluation.get("lead_sets")
    if not isinstance(exported_leads, dict):
        raise ValueError("evaluation manifest does not list exported lead sets")
    missing_exports = sorted(({FULL_LEAD_SET, *required_leads}) - set(exported_leads))
    if missing_exports:
        raise ValueError(f"evaluation is missing required lead exports: {missing_exports}")
    for lead_set in required_leads:
        entry = indexed[lead_set]
        if entry.get("diagnosis_group") != diagnosis_group or entry.get("diagnosis") != diagnosis:
            raise ValueError(f"{lead_set} policy target does not match the protocol")

    configuration = selective_manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("selective manifest is missing its frozen configuration")
    configured_leads = configuration.get("lead_sets")
    if not isinstance(configured_leads, list) or not set(required_leads) <= set(configured_leads):
        raise ValueError("selective manifest does not contain all protocol lead sets")
    if isinstance(primary, dict) and primary.get("endpoint") == "trusted_negative_error":
        frozen_limit = configuration.get("max_negative_risk")
        if frozen_limit is None or float(primary["risk_limit"]) != float(frozen_limit):
            raise ValueError("protocol negative-risk limit differs from the frozen policy")
    secondary = protocol.get("secondary_analysis")
    if isinstance(secondary, dict) and "trusted_positive_error_limit" in secondary:
        frozen_limit = configuration.get("max_positive_risk")
        if frozen_limit is None or float(secondary["trusted_positive_error_limit"]) != float(frozen_limit):
            raise ValueError("protocol positive-risk limit differs from the frozen policy")
    return indexed


def _quality_present(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return series.notna() & ~normalized.isin(["", "0", "0.0", "false", "none", "nan"])


def subgroup_masks(frame: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    masks: list[tuple[str, str, pd.Series]] = [
        ("overall", "all", pd.Series(True, index=frame.index)),
    ]
    if "sex" in frame:
        for value in sorted(frame.sex.dropna().unique(), key=str):
            masks.append(("sex", str(value), frame.sex == value))
        if frame.sex.isna().any():
            masks.append(("sex", "missing", frame.sex.isna()))
    if "age" in frame:
        age = pd.to_numeric(frame.age, errors="coerce")
        bands = pd.cut(
            age,
            bins=[-np.inf, 39, 64, 79, np.inf],
            labels=["<40", "40-64", "65-79", "80+"],
        )
        for value in bands.dropna().unique():
            masks.append(("age_band", str(value), bands == value))
        if bands.isna().any():
            masks.append(("age_band", "missing", bands.isna()))
    quality_masks = []
    for column in QUALITY_COLUMNS:
        if column in frame:
            present = _quality_present(frame[column])
            quality_masks.append(present)
            masks.append(("quality_flag", f"{column}:present", present))
            masks.append(("quality_flag", f"{column}:absent", ~present))
    if quality_masks:
        any_quality = pd.concat(quality_masks, axis=1).any(axis=1)
        masks.append(("quality", "any_flag", any_quality))
        masks.append(("quality", "no_flags", ~any_quality))
    return masks


def _subgroup_rows(
    decisions: pd.DataFrame, lead_set: str, minimum_records: int,
) -> list[dict[str, object]]:
    rows = []
    for group, value, mask in subgroup_masks(decisions):
        subset = decisions.loc[mask]
        if len(subset) < minimum_records and group != "overall":
            continue
        policy = selective_performance(subset)
        target = subset.target.to_numpy(dtype=np.int8)
        reduced = binary_performance(target, subset.reduced_probability.to_numpy())
        twelve = binary_performance(target, subset.twelve_probability.to_numpy())
        rows.append({
            "lead_set": lead_set,
            "subgroup": group,
            "value": value,
            **policy,
            "reduced_auroc": reduced["auroc"],
            "reduced_average_precision": reduced["average_precision"],
            "reduced_brier": reduced["brier"],
            "twelve_lead_auroc": twelve["auroc"],
            "twelve_lead_average_precision": twelve["average_precision"],
            "twelve_lead_brier": twelve["brier"],
        })
    return rows


def _paired_contrast_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    target = frame.target.to_numpy(dtype=np.int8)
    reduced = binary_performance(
        target,
        frame.reduced_probability.to_numpy(),
        include_calibration_coefficients=False,
    )
    twelve = binary_performance(
        target,
        frame.twelve_probability.to_numpy(),
        include_calibration_coefficients=False,
    )
    return {
        "delta_auroc": (
            float(reduced["auroc"] - twelve["auroc"])
            if reduced["auroc"] is not None and twelve["auroc"] is not None else None
        ),
        "delta_average_precision": (
            float(reduced["average_precision"] - twelve["average_precision"])
            if reduced["average_precision"] is not None
            and twelve["average_precision"] is not None else None
        ),
        "delta_brier": float(reduced["brier"] - twelve["brier"]),
    }


def _report_markdown(
    protocol: Mapping[str, object], results: Sequence[Mapping[str, object]],
) -> str:
    target = protocol["target"]
    assert isinstance(target, dict)
    lines = [
        f"# Frozen test audit: {target['diagnosis']}",
        "",
        f"Protocol: `{protocol['protocol_id']}`  ",
        f"Evidence status: `{protocol['status']}`  ",
        "Split: `test`",
        "",
        "This report applies validation-fitted policies without refitting. It is an",
        "internal retrospective evaluation and not a clinical-safety claim.",
        "",
        "## Lead-set results",
        "",
        "| Lead set | Records | Patients | Positive patients | Automated coverage | Positive automation recall | Positive-call error | Negative-call error | Expert referral |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        policy = result["policy"]
        assert isinstance(policy, dict)
        def percent(value: object) -> str:
            return "NA" if value is None else f"{100.0 * float(value):.1f}%"
        lines.append(
            f"| {result['lead_set']} | {policy['n_records']} | {policy['n_patients']} | "
            f"{policy['positive_patients']} | {percent(policy['automated_coverage'])} | "
            f"{percent(policy['positive_automation_recall'])} | "
            f"{percent(policy['trusted_positive_error'])} | "
            f"{percent(policy['trusted_negative_error'])} | "
            f"{percent(policy['expert_referral_rate'])} |"
        )
    lines.extend([
        "",
        "See `metrics.json` for discrimination, calibration, threshold metrics, and",
        "patient-cluster bootstrap intervals. `risk_coverage.csv.gz` is descriptive",
        "only and was not used to select or modify the frozen policy.",
        "",
        "Methods and evidence rationale: `AFIB_EVALUATION_PROTOCOL.md` and `REFERENCES.md`.",
        "",
    ])
    return "\n".join(lines)


def run_audit(
    evaluation_dir: Path,
    policy_path: Path,
    protocol_path: Path,
    out_dir: Path,
    run_name: str | None = None,
) -> Path:
    evaluation_dir = evaluation_dir.resolve()
    policy_path = policy_path.resolve()
    protocol_path = protocol_path.resolve()
    evaluation_manifest_path = evaluation_dir / "manifest.json"
    selective_manifest_path = policy_path.parent / "manifest.json"
    evaluation_manifest = _load_json(evaluation_manifest_path)
    selective_manifest = _load_json(selective_manifest_path)
    policy_payload = _load_json(policy_path)
    protocol = load_protocol(protocol_path)
    policy_entries = validate_audit_contract(
        evaluation_manifest, selective_manifest, policy_payload, protocol,
    )

    target = protocol["target"]
    assert isinstance(target, dict)
    diagnosis_group = str(target["diagnosis_group"])
    diagnosis = str(target["diagnosis"])
    split = str(protocol["split"])
    lead_sets = [str(value) for value in protocol["required_lead_sets"]]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    selected_name = run_name or f"{timestamp}-{diagnosis}-locked-test"
    if Path(selected_name).name != selected_name or selected_name in {".", ".."}:
        raise ValueError("run-name must be a single directory name")
    run_dir = out_dir.resolve() / selected_name
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "in_progress",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_free": True,
        "split": split,
        "target": target,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "protocol_id": protocol["protocol_id"],
            "status": protocol["status"],
        },
        "inputs": {
            "evaluation_manifest": {
                "path": str(evaluation_manifest_path),
                "sha256": sha256_file(evaluation_manifest_path),
            },
            "selective_manifest": {
                "path": str(selective_manifest_path),
                "sha256": sha256_file(selective_manifest_path),
            },
            "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
            "predictions": {},
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": [],
    }
    manifest_path = run_dir / "manifest.json"
    atomic_json_dump(manifest, manifest_path)

    twelve, twelve_path = load_test_prediction(
        evaluation_dir, split, FULL_LEAD_SET, diagnosis_group, diagnosis,
    )
    prediction_inputs = manifest["inputs"]["predictions"]
    assert isinstance(prediction_inputs, dict)
    prediction_inputs[FULL_LEAD_SET] = {
        "path": str(twelve_path), "sha256": sha256_file(twelve_path),
    }

    all_decisions = []
    metric_results = []
    subgroup_rows = []
    contrast_rows = []
    calibration_frames = []
    risk_frames = []
    n_bootstrap = int(protocol["bootstrap_replicates"])
    bootstrap_seed = int(protocol["bootstrap_seed"])
    calibration_bins = int(protocol["calibration_bins"])
    minimum_subgroup_records = int(protocol["minimum_subgroup_records"])

    for lead_index, lead_set in enumerate(lead_sets):
        reduced, reduced_path = load_test_prediction(
            evaluation_dir, split, lead_set, diagnosis_group, diagnosis,
        )
        prediction_inputs[lead_set] = {
            "path": str(reduced_path), "sha256": sha256_file(reduced_path),
        }
        paired = pair_test_predictions(reduced, twelve)
        policy = policy_entries[lead_set]["policy"]
        assert isinstance(policy, dict)
        decisions = apply_sequential_policy(
            paired, policy["reduced"], policy[FULL_LEAD_SET],
        )
        decisions.insert(2, "split", split)
        decisions.insert(3, "diagnosis_group", diagnosis_group)
        decisions.insert(4, "diagnosis", diagnosis)
        decisions.insert(5, "lead_set", lead_set)
        all_decisions.append(decisions)

        target_values = decisions.target.to_numpy(dtype=np.int8)
        def bootstrap_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
            values = dict(flat_audit_metrics(frame))
            reduced_auroc = values["reduced.auroc"]
            twelve_auroc = values["twelve_lead.auroc"]
            reduced_ap = values["reduced.average_precision"]
            twelve_ap = values["twelve_lead.average_precision"]
            values["contrast.delta_auroc"] = (
                float(reduced_auroc - twelve_auroc)
                if reduced_auroc is not None and twelve_auroc is not None else None
            )
            values["contrast.delta_average_precision"] = (
                float(reduced_ap - twelve_ap)
                if reduced_ap is not None and twelve_ap is not None else None
            )
            values["contrast.delta_brier"] = float(
                values["reduced.brier"] - values["twelve_lead.brier"],
            )
            return values

        combined_intervals = patient_cluster_bootstrap_intervals(
            decisions,
            bootstrap_metrics,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + lead_index,
        )
        point = {
            "lead_set": lead_set,
            "reduced": binary_performance(
                target_values, decisions.reduced_probability.to_numpy(),
            ),
            FULL_LEAD_SET: binary_performance(
                target_values, decisions.twelve_probability.to_numpy(),
            ),
            "policy": selective_performance(decisions),
            "patient_level_policy": patient_level_selective_performance(decisions),
            "patient_cluster_bootstrap_95_ci": {
                name: interval for name, interval in combined_intervals.items()
                if not name.startswith("contrast.")
            },
        }
        metric_results.append(point)
        subgroup_rows.extend(_subgroup_rows(decisions, lead_set, minimum_subgroup_records))

        contrast_point = _paired_contrast_metrics(decisions)
        contrast_ci = {
            name.removeprefix("contrast."): interval
            for name, interval in combined_intervals.items()
            if name.startswith("contrast.")
        }
        for metric, value in contrast_point.items():
            interval = contrast_ci.get(metric, {})
            contrast_rows.append({
                "lead_set": lead_set,
                "reference": FULL_LEAD_SET,
                "metric": metric,
                "point": value,
                "ci_lower": interval.get("lower"),
                "ci_upper": interval.get("upper"),
                "resampling_unit": "patient",
            })

        probability_sets = [
            ("reduced", "raw", sigmoid(decisions.reduced_logit.to_numpy())),
            ("reduced", "calibrated", decisions.reduced_probability.to_numpy()),
            (FULL_LEAD_SET, "raw", sigmoid(decisions.twelve_logit.to_numpy())),
            (FULL_LEAD_SET, "calibrated", decisions.twelve_probability.to_numpy()),
        ]
        for model, scale, probabilities in probability_sets:
            calibration = reliability_table(
                target_values, probabilities, n_bins=calibration_bins,
            )
            calibration.insert(0, "lead_set", lead_set)
            calibration.insert(1, "model", model)
            calibration.insert(2, "scale", scale)
            calibration_frames.append(calibration)
        for model, column in (
            ("reduced", "reduced_probability"),
            (FULL_LEAD_SET, "twelve_probability"),
        ):
            curve = empirical_risk_coverage_curve(
                target_values, decisions[column].to_numpy(),
            )
            curve.insert(0, "lead_set", lead_set)
            curve.insert(1, "model", model)
            curve.insert(2, "kind", "descriptive_confidence_ranking")
            risk_frames.append(curve)

    decisions_path = run_dir / "decisions.csv.gz"
    metrics_path = run_dir / "metrics.json"
    subgroup_path = run_dir / "subgroup_metrics.csv.gz"
    contrasts_path = run_dir / "paired_contrasts.csv.gz"
    calibration_path = run_dir / "calibration.csv.gz"
    risk_path = run_dir / "risk_coverage.csv.gz"
    report_path = run_dir / "report.md"
    atomic_csv_gz(pd.concat(all_decisions, ignore_index=True), decisions_path)
    primary_spec = protocol.get("primary_analysis")
    primary_result = None
    if isinstance(primary_spec, dict):
        primary_lead = primary_spec.get("lead_set")
        primary_endpoint = primary_spec.get("endpoint")
        selected = next(
            (result for result in metric_results if result["lead_set"] == primary_lead), None,
        )
        if selected is not None and isinstance(primary_endpoint, str):
            policy_metrics = selected["policy"]
            assert isinstance(policy_metrics, dict)
            value = policy_metrics.get(primary_endpoint)
            risk_limit = primary_spec.get("risk_limit")
            primary_result = {
                "lead_set": primary_lead,
                "endpoint": primary_endpoint,
                "point": value,
                "risk_limit": risk_limit,
                "observed_within_limit": (
                    None if value is None or risk_limit is None
                    else bool(float(value) <= float(risk_limit))
                ),
                "claim_boundary": (
                    "Descriptive locked-test comparison; it is not a new finite-sample "
                    "guarantee and must not be used to retune this policy"
                ),
            }
    atomic_json_dump({
        "schema_version": 1,
        "split": split,
        "fit_free": True,
        "bootstrap": {
            "unit": "patient",
            "replicates": n_bootstrap,
            "confidence": 0.95,
            "seed": bootstrap_seed,
        },
        "primary_analysis": primary_result,
        "results": metric_results,
    }, metrics_path)
    atomic_csv_gz(pd.DataFrame(subgroup_rows), subgroup_path)
    atomic_csv_gz(pd.DataFrame(contrast_rows), contrasts_path)
    atomic_csv_gz(pd.concat(calibration_frames, ignore_index=True), calibration_path)
    atomic_csv_gz(pd.concat(risk_frames, ignore_index=True), risk_path)
    _atomic_text_dump(_report_markdown(protocol, metric_results), report_path)

    artifacts = [
        decisions_path, metrics_path, subgroup_path, contrasts_path,
        calibration_path, risk_path, report_path,
    ]
    manifest["artifacts"] = [
        {"path": str(path.relative_to(run_dir)), "sha256": sha256_file(path)}
        for path in artifacts
    ]
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json_dump(manifest, manifest_path)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("protocol/afib_v1.json"),
    )
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/audit"))
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_protocol(args.protocol.resolve())
    if args.split != protocol["split"]:
        raise ValueError("CLI split does not match the locked protocol")
    run_dir = run_audit(
        evaluation_dir=args.evaluation_dir,
        policy_path=args.policy,
        protocol_path=args.protocol,
        out_dir=args.out_dir,
        run_name=args.run_name,
    )
    print(f"[audit] complete -> {run_dir}")


if __name__ == "__main__":
    main()
