"""Fit and audit selective ECG policies across every exported diagnosis label.

The ``fit`` phase reads validation predictions only and freezes a policy bundle.
The ``audit`` phase verifies that bundle before reading test predictions. Labels
with too few validation patients remain in the outputs as performance-only
targets rather than receiving unstable selective thresholds.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
import sklearn

from .audit_policy import subgroup_masks, _policy_index
from .evaluate import atomic_csv_gz, atomic_json_dump, sha256_file
from .selective import (
    FULL_LEAD_SET,
    REDUCED_LEAD_SETS,
    SEQUENTIAL_METHOD,
    apply_sequential_policy,
    cross_fitted_decisions,
    decision_metrics,
    fit_sequential_policy,
    pair_lead_predictions,
    patient_bootstrap_intervals,
)
from .statistics import (
    binary_performance,
    empirical_risk_coverage_curve,
    patient_level_selective_performance,
    reliability_table,
    selective_performance,
)


REQUIRED_PREDICTION_COLUMNS = {
    "ecg_id", "patient_id", "split", "lead_set", "diagnosis_group",
    "diagnosis", "target", "logit",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_text_dump(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def load_all_label_protocol(path: Path, lead_sets: Sequence[str] | None = None) -> dict[str, Any]:
    protocol = _load_json(path)
    if "risk_limits_placeholders" in protocol:
        from .use_contract import execution_protocol, load_use_contract
        protocol = execution_protocol(load_use_contract(path), lead_sets)
    elif lead_sets is not None and list(lead_sets) != protocol.get("lead_sets"):
        raise ValueError("lead-set overrides are only supported by the use contract")
    required = {
        "schema_version", "protocol_id", "status", "splits", "diagnosis_groups",
        "lead_sets", "policy_eligibility", "shared_policy_constraints",
        "uncertainty", "calibration_bins", "minimum_subgroup_records",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"all-label protocol is missing fields: {missing}")
    splits = protocol["splits"]
    if not isinstance(splits, dict) or splits.get("policy_fit") != "val" or splits.get("audit") != "test":
        raise ValueError("all-label protocol must fit on val and audit on test")
    groups = protocol["diagnosis_groups"]
    if not isinstance(groups, list) or not groups or len(groups) != len(set(groups)):
        raise ValueError("diagnosis_groups must be a non-empty unique list")
    leads = protocol["lead_sets"]
    if not isinstance(leads, list) or not leads or len(leads) != len(set(leads)):
        raise ValueError("lead_sets must be a non-empty unique list")
    unknown_leads = sorted(set(leads) - set(REDUCED_LEAD_SETS))
    if unknown_leads:
        raise ValueError(f"unknown reduced lead sets: {unknown_leads}")
    eligibility = protocol["policy_eligibility"]
    constraints = protocol["shared_policy_constraints"]
    uncertainty = protocol["uncertainty"]
    if not isinstance(eligibility, dict) or not isinstance(constraints, dict) or not isinstance(uncertainty, dict):
        raise ValueError("eligibility, constraints, and uncertainty must be objects")
    if min(
        int(eligibility.get("minimum_positive_patients", 0)),
        int(eligibility.get("minimum_negative_patients", 0)),
        int(eligibility.get("outer_folds", 0)),
    ) <= 0:
        raise ValueError("policy eligibility counts must be positive")
    for name in ("max_positive_risk", "max_negative_risk", "confidence"):
        value = float(constraints.get(name, -1))
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1)")
    if int(constraints.get("minimum_trusted_records", 0)) <= 0:
        raise ValueError("minimum_trusted_records must be positive")
    if int(constraints.get("threshold_grid_size", 0)) < 2:
        raise ValueError("threshold_grid_size must be at least 2")
    if int(uncertainty.get("patient_bootstrap_replicates", -1)) < 0:
        raise ValueError("patient_bootstrap_replicates cannot be negative")
    if int(eligibility["outer_folds"]) < 2:
        raise ValueError("outer_folds must be at least two")
    if float(uncertainty.get("confidence", 0.95)) != 0.95:
        raise ValueError("patient bootstrap currently supports confidence=0.95 only")
    return protocol


def _targets(schema: Mapping[str, object], protocol: Mapping[str, Any]) -> list[tuple[str, str]]:
    targets = targets_from_schema(schema, protocol["diagnosis_groups"])
    if "use_contract" in protocol:
        from .use_contract import COMPOSITE_COMPONENTS, COMPOSITE_DIAGNOSIS, COMPOSITE_GROUP
        if not set(COMPOSITE_COMPONENTS) <= set(targets):
            raise ValueError("label schema lacks composite components")
        if (COMPOSITE_GROUP, COMPOSITE_DIAGNOSIS) in targets:
            raise ValueError("exported schema must not contain a synthetic composite")
        targets.append((COMPOSITE_GROUP, COMPOSITE_DIAGNOSIS))
    return targets


def _constraints(protocol: Mapping[str, Any], group: str, diagnosis: str) -> dict[str, Any]:
    constraints = dict(protocol["shared_policy_constraints"])
    if "use_contract" in protocol:
        from .use_contract import resolve_limits
        positive, negative = resolve_limits(protocol["use_contract"], group, diagnosis)
        constraints.update(max_positive_risk=positive, max_negative_risk=negative)
    return constraints


def _interpretation(group: str, diagnosis: str) -> str:
    return "assert_normal" if (group, diagnosis) in {
        ("superclass", "NORM"), ("subcode", "NORM"), ("rhythm", "SR"),
    } else "assert_disease"


def _add_composites(frames: dict[str, pd.DataFrame], protocol: Mapping[str, Any]) -> None:
    if "use_contract" in protocol:
        from .use_contract import build_composite_frame
        for lead, frame in frames.items():
            frames[lead] = pd.concat([frame, build_composite_frame(frame)], ignore_index=True)


def targets_from_schema(
    schema: Mapping[str, object], diagnosis_groups: Sequence[str],
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for group in diagnosis_groups:
        diagnoses = schema.get(group)
        if not isinstance(diagnoses, list):
            raise ValueError(f"label schema is missing diagnosis group {group}")
        targets.extend((group, str(diagnosis)) for diagnosis in diagnoses)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("label schema targets must be non-empty and unique")
    return targets


def target_eligibility(
    frame: pd.DataFrame,
    minimum_positive_patients: int,
    minimum_negative_patients: int,
) -> dict[str, object]:
    positive = frame.target == 1
    negative = frame.target == 0
    positive_patients = int(frame.loc[positive, "patient_id"].nunique())
    negative_patients = int(frame.loc[negative, "patient_id"].nunique())
    reasons = []
    if positive_patients < minimum_positive_patients:
        reasons.append(
            f"positive_patients={positive_patients}<{minimum_positive_patients}",
        )
    if negative_patients < minimum_negative_patients:
        reasons.append(
            f"negative_patients={negative_patients}<{minimum_negative_patients}",
        )
    return {
        "n_records": int(len(frame)),
        "n_patients": int(frame.patient_id.nunique()),
        "positive_records": int(positive.sum()),
        "negative_records": int(negative.sum()),
        "positive_patients": positive_patients,
        "negative_patients": negative_patients,
        "policy_eligible": not reasons,
        "reason": ";".join(reasons) if reasons else None,
    }


def _validate_evaluation_manifest(
    manifest: Mapping[str, object], split: str, protocol: Mapping[str, object],
) -> None:
    if manifest.get("status") != "complete":
        raise ValueError("evaluation manifest status must be complete")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation manifest is missing evaluation metadata")
    splits = evaluation.get("splits")
    if not isinstance(splits, list) or split not in splits:
        raise ValueError(f"evaluation does not contain split {split}")
    if evaluation.get("max_n") is not None:
        raise ValueError("all-label evaluation refuses max_n-limited exports")
    if split == "val" and ("test" in splits or evaluation.get("test_evaluated")):
        raise ValueError("policy fitting requires a validation-only export")
    if split == "test" and not evaluation.get("test_evaluated"):
        raise ValueError("test audit requires a complete test export")
    schema = manifest.get("label_schema")
    if not isinstance(schema, dict):
        raise ValueError("evaluation manifest is missing label_schema")
    _targets(schema, protocol)
    exported_leads = evaluation.get("lead_sets")
    if not isinstance(exported_leads, dict):
        raise ValueError("evaluation manifest is missing lead sets")
    required_leads = {FULL_LEAD_SET, *protocol["lead_sets"]}
    if not required_leads <= set(exported_leads):
        raise ValueError("evaluation export is missing required lead sets")


def _validate_bundle_population(
    evaluation_dir: Path, split: str, frames: Mapping[str, pd.DataFrame],
    schema: Mapping[str, object], protocol: Mapping[str, Any],
) -> None:
    """Require every exported target/lead to cover the complete record inventory."""
    records_path = evaluation_dir / f"records__{split}.csv.gz"
    records = pd.read_csv(records_path, low_memory=False)
    if records.empty or records.ecg_id.duplicated().any() or records.patient_id.isna().any():
        raise ValueError("record inventory must have unique ECG IDs and complete patient IDs")
    records = records.set_index("ecg_id").sort_index()
    expected = set(targets_from_schema(schema, protocol["diagnosis_groups"]))
    reference = frames[FULL_LEAD_SET].set_index(["diagnosis_group", "diagnosis", "ecg_id"]).sort_index()
    for lead, frame in frames.items():
        actual = set(zip(frame.diagnosis_group, frame.diagnosis))
        if not expected <= actual:
            raise ValueError(f"{lead} is missing requested targets")
        for (group, diagnosis), target in frame.groupby(["diagnosis_group", "diagnosis"]):
            target = target.set_index("ecg_id").sort_index()
            if not target.index.equals(records.index):
                raise ValueError(f"incomplete ECG population for {lead}/{group}/{diagnosis}")
            if not target.patient_id.equals(records.patient_id):
                raise ValueError("patient IDs disagree with record inventory")
            if lead != FULL_LEAD_SET:
                try:
                    full = reference.loc[(group, diagnosis)].sort_index()
                except KeyError as exc:
                    raise ValueError("lead label populations disagree") from exc
                if not target.target.equals(full.target):
                    raise ValueError("targets disagree across leads")


def _load_prediction_bundle(
    evaluation_dir: Path,
    split: str,
    lead_sets: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    frames: dict[str, pd.DataFrame] = {}
    inputs: dict[str, dict[str, str]] = {}
    for lead_set in (FULL_LEAD_SET, *lead_sets):
        path = evaluation_dir / "predictions" / f"{split}__{lead_set}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        if not ((frame.split == split) & (frame.lead_set == lead_set)).all():
            raise ValueError(f"{path} contains rows from a different split or lead set")
        if frame.empty:
            raise ValueError(f"{path} has no {split}/{lead_set} rows")
        if frame[["diagnosis_group", "diagnosis", "ecg_id"]].duplicated().any():
            raise ValueError(f"{path} contains duplicate target/ECG rows")
        if frame.patient_id.isna().any() or not np.isfinite(frame.logit.to_numpy()).all():
            raise ValueError(f"{path} has incomplete patient IDs or non-finite logits")
        if not frame.target.isin([0, 1]).all():
            raise ValueError(f"{path} contains non-binary targets")
        frames[lead_set] = frame
        inputs[lead_set] = {"path": str(path), "sha256": sha256_file(path)}
    return frames, inputs


def _target_frame(
    frame: pd.DataFrame, diagnosis_group: str, diagnosis: str,
) -> pd.DataFrame:
    selected = frame[
        (frame.diagnosis_group == diagnosis_group) & (frame.diagnosis == diagnosis)
    ].copy()
    if selected.empty:
        raise ValueError(f"no rows for {diagnosis_group}/{diagnosis}")
    return selected.sort_values("ecg_id").reset_index(drop=True)


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _new_run_dir(out_dir: Path, run_name: str | None, suffix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    selected = run_name or f"{timestamp}-{suffix}"
    if Path(selected).name != selected or selected in {".", ".."}:
        raise ValueError("run-name must be a single directory name")
    run_dir = out_dir.resolve() / selected
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _artifact_entries(run_dir: Path, paths: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.relative_to(run_dir)), "sha256": sha256_file(path)}
        for path in paths
    ]


def _verify_frozen_bundle(
    policy_path: Path, payload: Mapping[str, Any], manifest: Mapping[str, Any],
    protocol: Mapping[str, Any], evaluation_manifest: Mapping[str, Any],
) -> None:
    """Check provenance and integrity before opening any test prediction file."""
    entries = [item for item in manifest.get("artifacts", [])
               if isinstance(item, dict) and item.get("path") == policy_path.name]
    if len(entries) != 1 or entries[0].get("sha256") != sha256_file(policy_path):
        raise ValueError("frozen policy artifact SHA-256 mismatch or missing hash")
    source = manifest.get("source_evaluation", {})
    if source.get("test_predictions_read") is not False:
        raise ValueError("policy provenance must explicitly exclude test predictions")
    source_path = Path(source.get("path", "")) / "manifest.json"
    if not source_path.is_file() or sha256_file(source_path) != source.get("manifest_sha256"):
        raise ValueError("validation source manifest SHA-256 mismatch")
    source_manifest = _load_json(source_path)
    _validate_evaluation_manifest(source_manifest, "val", protocol)
    checkpoint = payload.get("source_checkpoint_sha256")
    if any(value != checkpoint for value in (
        source.get("checkpoint", {}).get("sha256"),
        source_manifest.get("checkpoint", {}).get("sha256"),
    )):
        raise ValueError("validation provenance checkpoint SHA-256 mismatch")
    if source_manifest.get("label_schema") != payload.get("label_schema"):
        raise ValueError("validation provenance label schema mismatch")
    for lead in (FULL_LEAD_SET, *protocol["lead_sets"]):
        if source_manifest["evaluation"]["lead_sets"][lead] != evaluation_manifest["evaluation"]["lead_sets"][lead]:
            raise ValueError("validation and test lead definitions differ")
    inputs = manifest.get("inputs", {})
    for key in (FULL_LEAD_SET, *protocol["lead_sets"], "records"):
        item = inputs.get(key, {})
        expected = source_path.parent / (
            "records__val.csv.gz" if key == "records" else f"predictions/val__{key}.csv.gz"
        )
        if item.get("path") != str(expected) or not expected.is_file() or item.get("sha256") != sha256_file(expected):
            raise ValueError(f"validation input SHA-256 mismatch: {key}")
    config = payload.get("configuration", {})
    if config.get("execution_protocol") != protocol or config.get("method") != SEQUENTIAL_METHOD:
        raise ValueError("unsupported or changed execution contract; refit on validation with the current method")
    if config.get("lead_sets") != protocol["lead_sets"]:
        raise ValueError("policy lead configuration differs from protocol")
    targets = _targets(payload["label_schema"], protocol)
    entries = payload.get("targets", [])
    if not isinstance(entries, list) or len(entries) != len(targets):
        raise ValueError("policy bundle must enumerate every target exactly once")
    identities = [(entry.get("diagnosis_group"), entry.get("diagnosis")) for entry in entries]
    if len(set(identities)) != len(entries) or set(identities) != set(targets):
        raise ValueError("policy targets are duplicated or incomplete")
    for entry in entries:
        constraints = _constraints(protocol, entry["diagnosis_group"], entry["diagnosis"])
        if entry.get("risk_limits") != constraints:
            raise ValueError("target risk limits differ from contract")
        eligibility = entry.get("eligibility", {})
        eligible = (
            eligibility.get("positive_patients", -1) >= protocol["policy_eligibility"]["minimum_positive_patients"]
            and eligibility.get("negative_patients", -1) >= protocol["policy_eligibility"]["minimum_negative_patients"]
        )
        if eligibility.get("policy_eligible") != eligible:
            raise ValueError("target eligibility disagrees with patient counts")
        policies = entry.get("lead_policies", [])
        failures = entry.get("lead_failures", [])
        policy_leads = [p["lead_set"] for p in policies]
        failure_leads = [p["lead_set"] for p in failures]
        if not eligible:
            if policies or failures or entry.get("status") != "insufficient_data":
                raise ValueError("ineligible targets cannot contain fitted policies")
            continue
        if len(set(policy_leads + failure_leads)) != len(policy_leads + failure_leads) or set(policy_leads + failure_leads) != set(protocol["lead_sets"]):
            raise ValueError("eligible target does not account for every lead exactly once")
        if entry.get("status") != ("partial" if failures else "policy_fitted"):
            raise ValueError("target status disagrees with fitted lead policies")
        if policies:
            _policy_index({"status": "frozen_on_validation", "policies": policies})
        for lead in policies:
            if lead["policy"].get("method") != SEQUENTIAL_METHOD:
                raise ValueError("policy lacks independent routed risk calibration")
            for stage in ("reduced", FULL_LEAD_SET):
                risk = lead["policy"][stage].get("risk_control", {})
                for key, expected in (
                    ("max_positive_risk", constraints["max_positive_risk"]),
                    ("max_negative_risk", constraints["max_negative_risk"]),
                    ("confidence", constraints["confidence"]),
                    ("grid_size_per_direction", constraints["threshold_grid_size"]),
                    ("min_trusted", constraints["minimum_trusted_records"]),
                ):
                    if key not in risk or risk[key] != expected:
                        raise ValueError("stage risk control differs from target contract")
                for direction in ("positive", "negative"):
                    if constraints[f"max_{direction}_risk"] is None and lead["policy"][stage]["thresholds"][direction]["feasible"]:
                        raise ValueError("disabled contract direction has a trusted threshold")


def fit_all_label_policies(
    evaluation_dir: Path,
    protocol_path: Path,
    out_dir: Path,
    run_name: str | None = None,
    *, lead_sets: Sequence[str] | None = None,
) -> Path:
    evaluation_dir = evaluation_dir.resolve()
    protocol_path = protocol_path.resolve()
    protocol = load_all_label_protocol(protocol_path, lead_sets)
    source_manifest_path = evaluation_dir / "manifest.json"
    source_manifest = _load_json(source_manifest_path)
    _validate_evaluation_manifest(source_manifest, "val", protocol)
    checkpoint = source_manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
        raise ValueError("validation manifest is missing checkpoint SHA-256")
    schema = source_manifest["label_schema"]
    assert isinstance(schema, dict)
    targets = _targets(schema, protocol)
    lead_sets = [str(value) for value in protocol["lead_sets"]]
    run_dir = _new_run_dir(out_dir, run_name, "all-labels-policy")
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "in_progress",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_evaluation": {
            "path": str(evaluation_dir),
            "manifest_sha256": sha256_file(source_manifest_path),
            "checkpoint": checkpoint,
            "test_predictions_read": False,
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "protocol_id": protocol["protocol_id"],
        },
        "environment": _environment(),
        "inputs": {},
        "artifacts": [],
    }
    atomic_json_dump(manifest, manifest_path)
    frames, inputs = _load_prediction_bundle(evaluation_dir, "val", lead_sets)
    _validate_bundle_population(evaluation_dir, "val", frames, schema, protocol)
    records_path = evaluation_dir / "records__val.csv.gz"
    inputs["records"] = {"path": str(records_path), "sha256": sha256_file(records_path)}
    _add_composites(frames, protocol)
    manifest["inputs"] = inputs

    eligibility_config = protocol["policy_eligibility"]
    constraints = protocol["shared_policy_constraints"]
    uncertainty = protocol["uncertainty"]
    assert isinstance(eligibility_config, dict)
    assert isinstance(constraints, dict)
    assert isinstance(uncertainty, dict)
    minimum_positive = int(eligibility_config["minimum_positive_patients"])
    minimum_negative = int(eligibility_config["minimum_negative_patients"])
    folds = int(eligibility_config["outer_folds"])
    bootstrap = int(uncertainty["patient_bootstrap_replicates"])
    seed = int(constraints["seed"])

    eligibility_rows: list[dict[str, object]] = []
    policy_targets: list[dict[str, object]] = []
    validation_results: list[dict[str, object]] = []
    decision_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []

    twelve_bundle = frames[FULL_LEAD_SET]
    for diagnosis_group, diagnosis in targets:
        print(f"[all-labels:fit] {diagnosis_group}/{diagnosis}")
        twelve = _target_frame(twelve_bundle, diagnosis_group, diagnosis)
        constraints = _constraints(protocol, diagnosis_group, diagnosis)
        eligibility = target_eligibility(
            twelve, minimum_positive, minimum_negative,
        )
        eligibility_rows.append({
            "diagnosis_group": diagnosis_group,
            "diagnosis": diagnosis,
            **eligibility,
        })
        target_entry: dict[str, object] = {
            "diagnosis_group": diagnosis_group,
            "diagnosis": diagnosis,
            "eligibility": eligibility,
            "status": "insufficient_data",
            "lead_policies": [],
            "lead_failures": [],
            "risk_limits": constraints,
            "interpretation": _interpretation(diagnosis_group, diagnosis),
        }
        if not eligibility["policy_eligible"]:
            policy_targets.append(target_entry)
            continue

        # Match the single-target seed convention for reproducible comparisons.
        target_seed = seed
        lead_policies: list[dict[str, object]] = []
        lead_failures: list[dict[str, str]] = []
        for lead_index, lead_set in enumerate(lead_sets):
            reduced = _target_frame(frames[lead_set], diagnosis_group, diagnosis)
            paired = pair_lead_predictions(reduced, twelve)
            try:
                decisions, fold_policies = cross_fitted_decisions(
                    paired,
                    n_splits=folds,
                    seed=target_seed,
                    max_positive_risk=constraints["max_positive_risk"],
                    max_negative_risk=constraints["max_negative_risk"],
                    confidence=float(constraints["confidence"]),
                    min_trusted=int(constraints["minimum_trusted_records"]),
                    grid_size=int(constraints["threshold_grid_size"]),
                )
                final_policy, curves = fit_sequential_policy(
                    paired,
                    seed=target_seed + 100000 + lead_index,
                    max_positive_risk=constraints["max_positive_risk"],
                    max_negative_risk=constraints["max_negative_risk"],
                    confidence=float(constraints["confidence"]),
                    min_trusted=int(constraints["minimum_trusted_records"]),
                    grid_size=int(constraints["threshold_grid_size"]),
                )
            except ValueError as error:
                lead_failures.append({"lead_set": lead_set, "reason": str(error)})
                continue

            decisions.insert(2, "diagnosis_group", diagnosis_group)
            decisions.insert(3, "diagnosis", diagnosis)
            decisions.insert(4, "lead_set", lead_set)
            decision_frames.append(decisions)
            curves.insert(0, "diagnosis_group", diagnosis_group)
            curves.insert(1, "diagnosis", diagnosis)
            curves.insert(2, "lead_set", lead_set)
            curve_frames.append(curves)
            lead_policies.append({
                "lead_set": lead_set,
                "policy": final_policy,
                "crossfit_fold_policies": fold_policies,
            })
            validation_results.append({
                "diagnosis_group": diagnosis_group,
                "diagnosis": diagnosis,
                "lead_set": lead_set,
                "point": decision_metrics(decisions),
                "patient_bootstrap_95_ci": patient_bootstrap_intervals(
                    decisions, bootstrap, target_seed + lead_index,
                ),
            })
        target_entry["lead_policies"] = lead_policies
        target_entry["lead_failures"] = lead_failures
        target_entry["status"] = (
            "policy_fitted" if len(lead_policies) == len(lead_sets) else "partial"
        )
        policy_targets.append(target_entry)

    eligibility_path = run_dir / "eligibility.csv.gz"
    decisions_path = run_dir / "validation_decisions.csv.gz"
    curves_path = run_dir / "validation_risk_coverage.csv.gz"
    policy_path = run_dir / "policy.json"
    validation_metrics_path = run_dir / "validation_metrics.json"
    report_path = run_dir / "report.md"
    atomic_csv_gz(pd.DataFrame(eligibility_rows), eligibility_path)
    atomic_csv_gz(
        pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame(columns=["ecg_id", "patient_id", "diagnosis_group", "diagnosis", "lead_set", "target", "final_action", "final_prediction"]),
        decisions_path,
    )
    atomic_csv_gz(
        pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame(columns=["diagnosis_group", "diagnosis", "lead_set", "stage", "direction", "threshold", "risk_upper", "eligible"]),
        curves_path,
    )
    atomic_json_dump({
        "schema_version": 2,
        "status": "frozen_on_validation",
        "source_checkpoint_sha256": checkpoint["sha256"],
        "protocol": manifest["protocol"],
        "label_schema": schema,
        "configuration": {
            "lead_sets": lead_sets,
            "policy_eligibility": eligibility_config,
            "shared_policy_constraints": protocol["shared_policy_constraints"],
            "execution_protocol": protocol,
            "method": SEQUENTIAL_METHOD,
        },
        "targets": policy_targets,
    }, policy_path)
    atomic_json_dump({
        "schema_version": 1,
        "bootstrap": {"unit": "patient", "replicates": bootstrap, "confidence": 0.95},
        "results": validation_results,
    }, validation_metrics_path)
    eligible_count = sum(bool(row["policy_eligible"]) for row in eligibility_rows)
    fitted_count = sum(entry["status"] == "policy_fitted" for entry in policy_targets)
    _atomic_text_dump(
        "\n".join([
            "# All-label validation policy bundle",
            "",
            f"Protocol: `{protocol['protocol_id']}`  ",
            f"Targets in schema: {len(targets)}  ",
            f"Targets meeting patient-count eligibility: {eligible_count}  ",
            f"Targets with every lead policy fitted: {fitted_count}",
            "",
            "Policies use validation predictions only. Ineligible targets remain listed in",
            "`eligibility.csv.gz`; they receive test model metrics but no selective policy.",
            "Each target's limits are recorded in policy.json; they are research placeholders.",
            "",
        ]),
        report_path,
    )
    artifacts = [
        eligibility_path, decisions_path, curves_path, policy_path,
        validation_metrics_path, report_path,
    ]
    manifest["artifacts"] = _artifact_entries(run_dir, artifacts)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json_dump(manifest, manifest_path)
    print(f"[all-labels:fit] complete -> {run_dir}")
    return run_dir


def _flatten_model_metrics(
    diagnosis_group: str,
    diagnosis: str,
    lead_set: str,
    frame: pd.DataFrame,
    policy_eligible: bool,
) -> dict[str, object]:
    metrics = binary_performance(frame.target.to_numpy(), expit(frame.logit.to_numpy()))
    return {
        "diagnosis_group": diagnosis_group,
        "diagnosis": diagnosis,
        "lead_set": lead_set,
        "policy_eligible": policy_eligible,
        **metrics,
    }


def _flatten_policy_metrics(
    diagnosis_group: str,
    diagnosis: str,
    lead_set: str,
    decisions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    point = selective_performance(decisions)
    patient = patient_level_selective_performance(decisions)
    intervals = patient_bootstrap_intervals(decisions, n_bootstrap, seed)
    row: dict[str, object] = {
        "diagnosis_group": diagnosis_group,
        "diagnosis": diagnosis,
        "lead_set": lead_set,
        **point,
        **{f"patient_{name}": value for name, value in patient.items()},
    }
    for name, interval in intervals.items():
        row[f"{name}_ci_lower"] = interval.get("lower")
        row[f"{name}_ci_upper"] = interval.get("upper")
    return row


def _batch_subgroup_rows(
    decisions: pd.DataFrame,
    diagnosis_group: str,
    diagnosis: str,
    lead_set: str,
    minimum_records: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subgroup, value, mask in subgroup_masks(decisions):
        subset = decisions.loc[mask]
        if subgroup != "overall" and len(subset) < minimum_records:
            continue
        target = subset.target.to_numpy(dtype=np.int8)
        reduced = binary_performance(
            target, subset.reduced_probability.to_numpy(),
            include_calibration_coefficients=False,
        )
        twelve = binary_performance(
            target, subset.twelve_probability.to_numpy(),
            include_calibration_coefficients=False,
        )
        rows.append({
            "diagnosis_group": diagnosis_group,
            "diagnosis": diagnosis,
            "lead_set": lead_set,
            "subgroup": subgroup,
            "value": value,
            **selective_performance(subset),
            "reduced_auroc": reduced["auroc"],
            "reduced_average_precision": reduced["average_precision"],
            "reduced_brier": reduced["brier"],
            "twelve_lead_auroc": twelve["auroc"],
            "twelve_lead_average_precision": twelve["average_precision"],
            "twelve_lead_brier": twelve["brier"],
        })
    return rows


def _audit_report(
    protocol: Mapping[str, object],
    label_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    policy_metrics: pd.DataFrame,
) -> str:
    eligible = int(label_summary.policy_eligible.sum())
    risk_lines = []
    for direction in ("positive", "negative"):
        error = f"trusted_{direction}_error"
        upper = f"{error}_ci_upper"
        limit = f"max_{direction}_risk"
        routes = policy_metrics[policy_metrics[error].notna()] if error in policy_metrics else pd.DataFrame()
        checked = routes.dropna(subset=[upper, limit]) if not routes.empty and upper in routes else pd.DataFrame()
        uncertain = int((checked[upper] > checked[limit]).sum()) if not checked.empty else 0
        risk_lines.append(
            f"- {direction.title()} routes evaluated: {len(routes)}; intervals available: {len(checked)}; "
            f"bootstrap upper interval above the target-specific limit: {uncertain}."
        )
    group_rows = []
    for (group, lead_set), subset in model_metrics.groupby(
        ["diagnosis_group", "lead_set"], sort=False,
    ):
        supported = subset[subset.policy_eligible]
        group_rows.append(
            f"| {group} | {lead_set} | {len(subset)} | "
            f"{subset.auroc.mean():.3f} | {subset.average_precision.mean():.3f} | "
            f"{supported.auroc.mean():.3f} | {supported.average_precision.mean():.3f} |",
        )
    policy_lines = []
    if not policy_metrics.empty:
        for (group, interpretation), subset in policy_metrics[
            policy_metrics.lead_set == "2-lead"
        ].groupby(["diagnosis_group", "interpretation"], sort=False):
            policy_lines.append(
                f"| {group} / {interpretation} | {len(subset)} | {subset.automated_coverage.median():.1%} | "
                f"{subset.expert_referral_rate.median():.1%} |",
            )
    return "\n".join([
        "# All-label test audit",
        "",
        f"Protocol: `{protocol['protocol_id']}`  ",
        f"Evidence status: `{protocol['status']}`  ",
        f"Labels evaluated for model performance: {len(label_summary)}  ",
        f"Labels eligible for selective policies: {eligible}",
        "",
        "Every exported label receives test discrimination, calibration, and",
        "risk-coverage results. Selective-policy results are emitted only where the",
        "validation patient-count boundary was met. Results are post-hoc research",
        "evidence; all risk limits are research placeholders, not clinical guarantees.",
        "",
        "## Test-time risk check",
        "",
        *risk_lines,
        "",
        "Validation-side risk control does not guarantee the same bound on a finite test",
        "sample. Test intervals are descriptive stability checks and must not be used to",
        "retune the frozen policy.",
        "",
        "## Model summary",
        "",
        "Macro averages over rare labels can be unstable. Supported-label columns",
        "include only diagnoses meeting the selective-policy patient-count boundary.",
        "",
        "| Group | Lead set | Labels | AUROC, all | AP, all | AUROC, supported | AP, supported |",
        "|---|---|---:|---:|---:|---:|---:|",
        *group_rows,
        "",
        "## Two-lead policy summary",
        "",
        "Values are medians across heterogeneous diagnoses and are descriptive only.",
        "",
        "| Group | Policies | Automated coverage | Expert referral |",
        "|---|---:|---:|---:|",
        *policy_lines,
        "",
        "See `label_summary.csv.gz`, `model_metrics.csv.gz`,",
        "`policy_metrics.csv.gz`, and `subgroup_metrics.csv.gz` for label-level results.",
        "",
    ])


def audit_all_label_policies(
    evaluation_dir: Path,
    policy_path: Path,
    protocol_path: Path,
    out_dir: Path,
    run_name: str | None = None,
) -> Path:
    evaluation_dir = evaluation_dir.resolve()
    policy_path = policy_path.resolve()
    protocol_path = protocol_path.resolve()
    policy_payload = _load_json(policy_path)
    protocol = load_all_label_protocol(
        protocol_path, policy_payload.get("configuration", {}).get("lead_sets"),
    )
    policy_manifest_path = policy_path.parent / "manifest.json"
    policy_manifest = _load_json(policy_manifest_path)
    evaluation_manifest_path = evaluation_dir / "manifest.json"
    evaluation_manifest = _load_json(evaluation_manifest_path)
    _validate_evaluation_manifest(evaluation_manifest, "test", protocol)
    if policy_manifest.get("status") != "complete":
        raise ValueError("all-label policy manifest status must be complete")
    if policy_payload.get("status") != "frozen_on_validation":
        raise ValueError("all-label policy bundle must be frozen_on_validation")
    checkpoint = evaluation_manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
        raise ValueError("test manifest is missing checkpoint SHA-256")
    if policy_payload.get("source_checkpoint_sha256") != checkpoint["sha256"]:
        raise ValueError("policy and test checkpoint SHA-256 values differ")
    frozen_protocol = policy_payload.get("protocol")
    if not isinstance(frozen_protocol, dict) or frozen_protocol.get("sha256") != sha256_file(protocol_path):
        raise ValueError("policy bundle and audit protocol hashes differ")
    if policy_payload.get("label_schema") != evaluation_manifest.get("label_schema"):
        raise ValueError("policy and test label schemas differ")

    _verify_frozen_bundle(policy_path, policy_payload, policy_manifest, protocol, evaluation_manifest)

    lead_sets = [str(value) for value in protocol["lead_sets"]]
    schema = evaluation_manifest["label_schema"]
    assert isinstance(schema, dict)
    targets = _targets(schema, protocol)
    run_dir = _new_run_dir(out_dir, run_name, "all-labels-test")
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "in_progress",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_free": True,
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
            "policy_manifest": {
                "path": str(policy_manifest_path),
                "sha256": sha256_file(policy_manifest_path),
            },
            "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
            "predictions": {},
        },
        "environment": _environment(),
        "artifacts": [],
    }
    atomic_json_dump(manifest, manifest_path)
    frames, prediction_inputs = _load_prediction_bundle(
        evaluation_dir, "test", lead_sets,
    )
    _validate_bundle_population(evaluation_dir, "test", frames, schema, protocol)
    _add_composites(frames, protocol)
    manifest_inputs = manifest["inputs"]
    assert isinstance(manifest_inputs, dict)
    manifest_inputs["predictions"] = prediction_inputs

    target_entries = {
        (entry["diagnosis_group"], entry["diagnosis"]): entry
        for entry in policy_payload.get("targets", [])
        if isinstance(entry, dict)
    }
    if set(targets) != set(target_entries):
        raise ValueError("policy bundle does not enumerate every protocol target")
    uncertainty = protocol["uncertainty"]
    assert isinstance(uncertainty, dict)
    bootstrap = int(uncertainty["patient_bootstrap_replicates"])
    seed = int(protocol["shared_policy_constraints"]["seed"])
    calibration_bins = int(protocol["calibration_bins"])
    minimum_subgroup_records = int(protocol["minimum_subgroup_records"])

    label_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    calibration_frames: list[pd.DataFrame] = []
    risk_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []

    for target_index, (diagnosis_group, diagnosis) in enumerate(targets):
        print(f"[all-labels:audit] {diagnosis_group}/{diagnosis}")
        entry = target_entries[(diagnosis_group, diagnosis)]
        eligibility = entry.get("eligibility")
        if not isinstance(eligibility, dict):
            raise ValueError(f"policy target {diagnosis_group}/{diagnosis} lacks eligibility")
        policy_eligible = bool(eligibility.get("policy_eligible"))
        twelve = _target_frame(frames[FULL_LEAD_SET], diagnosis_group, diagnosis)
        test_eligibility = target_eligibility(twelve, 1, 1)
        lead_policy_entries = entry.get("lead_policies")
        if not isinstance(lead_policy_entries, list):
            raise ValueError("lead_policies must be a list")
        lead_policy_index = {
            item["lead_set"]: item["policy"]
            for item in lead_policy_entries if isinstance(item, dict)
        }
        label_rows.append({
            "diagnosis_group": diagnosis_group,
            "diagnosis": diagnosis,
            "validation_policy_status": entry.get("status"),
            "policy_eligible": policy_eligible,
            "eligibility_reason": eligibility.get("reason"),
            "available_lead_policies": len(lead_policy_index),
            **{
                f"test_{key}": value
                for key, value in test_eligibility.items()
                if key not in {"reason", "policy_eligible"}
            },
        })

        target_model_rows: dict[str, dict[str, object]] = {}
        for lead_set in (FULL_LEAD_SET, *lead_sets):
            frame = _target_frame(frames[lead_set], diagnosis_group, diagnosis)
            row = _flatten_model_metrics(
                diagnosis_group, diagnosis, lead_set, frame, policy_eligible,
            )
            model_rows.append(row)
            target_model_rows[lead_set] = row
            probabilities = expit(frame.logit.to_numpy())
            calibration = reliability_table(
                frame.target.to_numpy(), probabilities, n_bins=calibration_bins,
            )
            calibration.insert(0, "diagnosis_group", diagnosis_group)
            calibration.insert(1, "diagnosis", diagnosis)
            calibration.insert(2, "lead_set", lead_set)
            calibration_frames.append(calibration)
            risk = empirical_risk_coverage_curve(
                frame.target.to_numpy(), probabilities,
            )
            risk.insert(0, "diagnosis_group", diagnosis_group)
            risk.insert(1, "diagnosis", diagnosis)
            risk.insert(2, "lead_set", lead_set)
            risk_frames.append(risk)

        reference = target_model_rows[FULL_LEAD_SET]
        for lead_set in lead_sets:
            reduced_row = target_model_rows[lead_set]
            for metric in ("auroc", "average_precision", "brier"):
                reduced_value = reduced_row.get(metric)
                reference_value = reference.get(metric)
                contrast_rows.append({
                    "diagnosis_group": diagnosis_group,
                    "diagnosis": diagnosis,
                    "lead_set": lead_set,
                    "reference": FULL_LEAD_SET,
                    "metric": f"delta_{metric}",
                    "point": (
                        None if reduced_value is None or reference_value is None
                        else float(reduced_value) - float(reference_value)
                    ),
                    "inference": "descriptive_point_only",
                })

        for lead_index, lead_set in enumerate(lead_sets):
            policy = lead_policy_index.get(lead_set)
            if not isinstance(policy, dict):
                continue
            reduced = _target_frame(frames[lead_set], diagnosis_group, diagnosis)
            paired = pair_lead_predictions(reduced, twelve)
            decisions = apply_sequential_policy(
                paired, policy["reduced"], policy[FULL_LEAD_SET],
            )
            decisions.insert(2, "split", "test")
            decisions.insert(3, "diagnosis_group", diagnosis_group)
            decisions.insert(4, "diagnosis", diagnosis)
            decisions.insert(5, "lead_set", lead_set)
            decision_frames.append(decisions)
            policy_row = _flatten_policy_metrics(
                diagnosis_group,
                diagnosis,
                lead_set,
                decisions,
                bootstrap,
                seed + target_index * 1009 + lead_index,
            )
            limits = _constraints(protocol, diagnosis_group, diagnosis)
            policy_row.update(
                max_positive_risk=limits["max_positive_risk"],
                max_negative_risk=limits["max_negative_risk"],
                interpretation=_interpretation(diagnosis_group, diagnosis),
            )
            policy_rows.append(policy_row)
            subgroup_rows.extend(_batch_subgroup_rows(
                decisions,
                diagnosis_group,
                diagnosis,
                lead_set,
                minimum_subgroup_records,
            ))

    label_summary = pd.DataFrame(label_rows)
    model_metrics = pd.DataFrame(model_rows)
    policy_metrics = pd.DataFrame(policy_rows) if policy_rows else pd.DataFrame(columns=[
        "diagnosis_group", "diagnosis", "lead_set", "interpretation",
        "trusted_positive_error", "trusted_negative_error", "max_positive_risk", "max_negative_risk",
        "automated_coverage", "expert_referral_rate",
    ])
    label_path = run_dir / "label_summary.csv.gz"
    model_path = run_dir / "model_metrics.csv.gz"
    policy_metrics_path = run_dir / "policy_metrics.csv.gz"
    subgroup_path = run_dir / "subgroup_metrics.csv.gz"
    contrasts_path = run_dir / "paired_contrasts.csv.gz"
    calibration_path = run_dir / "calibration.csv.gz"
    risk_path = run_dir / "risk_coverage.csv.gz"
    decisions_path = run_dir / "decisions.csv.gz"
    report_path = run_dir / "report.md"
    atomic_csv_gz(label_summary, label_path)
    atomic_csv_gz(model_metrics, model_path)
    atomic_csv_gz(policy_metrics, policy_metrics_path)
    atomic_csv_gz(pd.DataFrame(subgroup_rows) if subgroup_rows else pd.DataFrame(columns=["diagnosis_group", "diagnosis", "lead_set", "subgroup", "value"]), subgroup_path)
    atomic_csv_gz(pd.DataFrame(contrast_rows), contrasts_path)
    atomic_csv_gz(pd.concat(calibration_frames, ignore_index=True), calibration_path)
    atomic_csv_gz(pd.concat(risk_frames, ignore_index=True), risk_path)
    atomic_csv_gz(
        pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame(columns=["ecg_id", "patient_id", "diagnosis_group", "diagnosis", "lead_set", "target", "final_action", "final_prediction"]),
        decisions_path,
    )
    _atomic_text_dump(
        _audit_report(protocol, label_summary, model_metrics, policy_metrics),
        report_path,
    )
    artifacts = [
        label_path, model_path, policy_metrics_path, subgroup_path, contrasts_path,
        calibration_path, risk_path, decisions_path, report_path,
    ]
    manifest["artifacts"] = _artifact_entries(run_dir, artifacts)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json_dump(manifest, manifest_path)
    print(f"[all-labels:audit] complete -> {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", help="fit every eligible policy on validation")
    fit.add_argument("--evaluation-dir", type=Path, required=True)
    fit.add_argument("--protocol", type=Path, default=Path("protocol/all_labels_v1.json"))
    fit.add_argument("--out-dir", type=Path, default=Path("artifacts/selective-all"))
    fit.add_argument("--run-name", type=str, default=None)
    audit = subparsers.add_parser("audit", help="apply a frozen all-label policy bundle")
    audit.add_argument("--evaluation-dir", type=Path, required=True)
    audit.add_argument("--policy", type=Path, required=True)
    audit.add_argument("--protocol", type=Path, default=Path("protocol/all_labels_v1.json"))
    audit.add_argument("--out-dir", type=Path, default=Path("artifacts/audit-all"))
    audit.add_argument("--run-name", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fit":
        fit_all_label_policies(
            args.evaluation_dir, args.protocol, args.out_dir, args.run_name,
        )
    else:
        audit_all_label_policies(
            args.evaluation_dir, args.policy, args.protocol, args.out_dir,
            args.run_name,
        )


if __name__ == "__main__":
    main()
