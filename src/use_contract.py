"""Wire frozen use-contract v1 to validation-only policy fitting.

Implements the two pieces deferred by USE_CONTRACT.md:

1. Composite AFIB-or-AFLT scorer, frozen as max-logit v1.
2. Per-group risk-limit plumbing read from protocol/use_contract_v1.json.

Fitting refuses exports containing test predictions. The shared all-label engine
fits eligible exported labels and the composite, and audits them without refitting.
The primary remains 2-lead composite rule-out; other results are descriptive.

Run:
    python -m src.use_contract fit \
      --evaluation-dir artifacts/evaluation/<val-only-export> \
      --protocol protocol/use_contract_v1.json \
      --run-name use-contract-v1-afib-aflt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from .selective import REDUCED_LEAD_SETS

# Frozen v1 scoring rule. Do not change without protocol/use_contract_v2.json
# plus a new untouched cohort.
COMPOSITE_SCORING_V1 = "max_logit"
COMPOSITE_GROUP = "rhythm"
COMPOSITE_DIAGNOSIS = "AFIB-or-AFLT"
COMPOSITE_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("rhythm", "AFIB"),
    ("rhythm", "AFLT"),
)


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_use_contract(path: Path) -> Dict[str, Any]:
    contract = _load_json(path)
    for key in (
        "protocol_id",
        "targets",
        "risk_limits_placeholders",
        "primary_analysis",
        "policy_eligibility",
        "uncertainty",
        "statistical_method",
    ):
        if key not in contract:
            raise ValueError(f"use contract is missing '{key}'")
    primary = contract["primary_analysis"]
    if primary.get("target") != COMPOSITE_DIAGNOSIS:
        raise ValueError("use contract primary target must be AFIB-or-AFLT")
    return contract


def _flatten_applies_to(contract: Dict[str, Any]) -> Dict[str, tuple[float, float | None]]:
    """Map exact group/diagnosis identities to (max_pos, max_neg)."""
    limits = contract["risk_limits_placeholders"]
    mapping: Dict[str, tuple[float, float | None]] = {}

    def _add(label: str, value: tuple[float, float | None]) -> None:
        mapping[label] = value

    mi = limits["MI_STTC_ischemia"]
    for label in mi["applies_to"]:
        _add(label, (
            float(mi["trusted_positive_error_max"]),
            float(mi["trusted_negative_error_max"]),
        ))
    cond = limits["conduction"]
    for label in cond["applies_to"]:
        _add(label, (
            float(cond["trusted_positive_error_max"]),
            float(cond["trusted_negative_error_max"]),
        ))
    normal = limits["assert_normal"]
    for label in normal["applies_to"]:
        # Trusted positive asserts normality -> rule-out stringency.
        # The contract defines no automated negative call for assert-normal.
        _add(label, (float(normal["trusted_positive_error_max"]), None))
    return mapping


def resolve_limits(
    contract: Dict[str, Any], diagnosis_group: str, diagnosis: str
) -> tuple[float, float | None]:
    """Return (max_positive_risk, max_negative_risk) for a label."""
    if diagnosis_group == COMPOSITE_GROUP and diagnosis == COMPOSITE_DIAGNOSIS:
        rule_out = contract["risk_limits_placeholders"]["AFIB-or-AFLT_rule_out"]
        rule_in = contract["risk_limits_placeholders"]["AFIB-or-AFLT_rule_in"]
        return (float(rule_in["comparability_anchor"]), float(rule_out["max_risk"]))
    mapping = _flatten_applies_to(contract)
    key = f"{diagnosis_group}/{diagnosis}"
    if key in mapping:
        return mapping[key]
    # Fallback generic anchor for labels outside the contract groups.
    # Callers treat non-primary targets as descriptive only.
    return (0.10, 0.02)


def build_composite_frame(
    frame: pd.DataFrame,
    components: Sequence[tuple[str, str]] = COMPOSITE_COMPONENTS,
    scoring: str = COMPOSITE_SCORING_V1,
) -> pd.DataFrame:
    """Combine component rows into one composite row per ECG.

    Frozen v1: composite_logit = max(component logits),
    composite_target = OR(component targets). Metadata must agree
    across components for the same ECG.
    """
    if scoring != COMPOSITE_SCORING_V1:
        raise ValueError(f"only frozen scoring '{COMPOSITE_SCORING_V1}' is allowed in v1")
    parts = []
    for group, diagnosis in components:
        selected = frame[
            (frame.diagnosis_group == group) & (frame.diagnosis == diagnosis)
        ].copy()
        if selected.empty:
            raise ValueError(f"no rows for {group}/{diagnosis}")
        parts.append(selected)
    # Align explicitly; never combine different ECG populations or metadata.
    metadata = [c for c in parts[0].columns if c not in {
        "diagnosis_group", "diagnosis", "target", "logit", "probability",
    }]
    for part in parts:
        if part.ecg_id.duplicated().any():
            raise ValueError("duplicate ECG IDs within a component")
        if not part.target.isin([0, 1]).all():
            raise ValueError("composite targets must be binary")
    meta_source = parts[0].set_index("ecg_id").sort_index()
    aligned = meta_source
    logits, targets = [], []
    for part in parts:
        indexed = part.set_index("ecg_id").sort_index()
        if not indexed.index.equals(meta_source.index):
            raise ValueError("ECG IDs differ across composite components")
        for column in metadata:
            if column == "ecg_id":
                continue
            if column not in indexed or not indexed[column].equals(meta_source[column]):
                raise ValueError(f"{column} mismatch across composite components")
        logits.append(indexed.logit.to_numpy(dtype=np.float64))
        targets.append(indexed.target.to_numpy())
    logit_frame = np.column_stack(logits)
    target_frame = np.column_stack(targets)
    if not np.isfinite(logit_frame).all():
        raise ValueError("non-finite logits in composite components")
    composite_logit = logit_frame.max(axis=1)
    composite_target = (target_frame.max(axis=1) > 0).astype(np.int8)
    # Carry metadata from the first component (verified consistent patient_id).
    out = pd.DataFrame({
        "ecg_id": aligned.index.to_numpy(),
        "split": meta_source["split"].to_numpy(),
        "lead_set": meta_source["lead_set"].to_numpy(),
        "diagnosis_group": COMPOSITE_GROUP,
        "diagnosis": COMPOSITE_DIAGNOSIS,
        "target": composite_target,
        "logit": composite_logit,
        "patient_id": meta_source["patient_id"].to_numpy(),
    })
    for column in (
        "age", "sex", "baseline_drift", "static_noise", "burst_noise",
        "electrodes_problems", "extra_beats", "pacemaker",
    ):
        if column in meta_source.columns:
            out[column] = meta_source[column].to_numpy()
    return out.sort_values("ecg_id").reset_index(drop=True)


def execution_protocol(contract: Dict[str, Any], lead_sets: Sequence[str] | None = None) -> Dict[str, Any]:
    """Adapt the frozen research document to the shared fit/audit engine.

    The original document and hash remain unchanged. Scoring and the execution
    method are also persisted in policy.json so audits can reject drift.
    """
    selected = list(REDUCED_LEAD_SETS if lead_sets is None else lead_sets)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(REDUCED_LEAD_SETS):
        raise ValueError("lead sets must be non-empty, unique reduced lead sets")
    if contract["primary_analysis"]["lead_set"] not in selected:
        raise ValueError("lead sets must include the contract primary lead set")
    rule_out = contract["risk_limits_placeholders"]["AFIB-or-AFLT_rule_out"]
    stats = contract["statistical_method"]
    uncertainty = contract["uncertainty"]
    if float(rule_out["max_risk"]) != float(contract["primary_analysis"]["risk_limit"]):
        raise ValueError("primary risk limit differs from composite rule-out limit")
    if contract["primary_analysis"]["endpoint"] != "trusted_negative_error":
        raise ValueError("unsupported primary endpoint")
    return {
        "schema_version": 2,
        "protocol_id": contract["protocol_id"],
        "status": contract["status"],
        "splits": {"policy_fit": "val", "audit": "test"},
        "diagnosis_groups": ["superclass", "subcode", "rhythm"],
        "lead_sets": selected,
        "policy_eligibility": {**contract["policy_eligibility"], "outer_folds": 5},
        "shared_policy_constraints": {
            "max_positive_risk": 0.10, "max_negative_risk": 0.02,
            "confidence": float(rule_out["confidence"]),
            "minimum_trusted_records": int(rule_out["minimum_trusted_records"]),
            "threshold_grid_size": int(stats["threshold_grid_size"]),
            "seed": int(stats["seed"]),
        },
        "uncertainty": {
            "patient_bootstrap_replicates": int(uncertainty.get("patient_bootstrap_replicates", uncertainty["replicates"])),
            "confidence": float(uncertainty["confidence"]),
        },
        "calibration_bins": int(uncertainty["calibration_bins"]),
        "minimum_subgroup_records": int(uncertainty["minimum_subgroup_records"]),
        "composite_scoring": COMPOSITE_SCORING_V1,
        "use_contract": contract,
    }


def fit_contract_policies(
    evaluation_dir: Path, protocol_path: Path, out_dir: Path,
    run_name: str | None = None, lead_sets: Sequence[str] | None = None,
) -> Path:
    from .audit_all_labels import fit_all_label_policies

    load_use_contract(protocol_path)
    return fit_all_label_policies(
        evaluation_dir, protocol_path, out_dir, run_name, lead_sets=lead_sets,
    )


def audit_contract_policies(
    evaluation_dir: Path, policy_path: Path, protocol_path: Path, out_dir: Path,
    run_name: str | None = None,
) -> Path:
    from .audit_all_labels import audit_all_label_policies

    load_use_contract(protocol_path)
    return audit_all_label_policies(evaluation_dir, policy_path, protocol_path, out_dir, run_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit composite and all exported targets on val only")
    fit.add_argument("--evaluation-dir", type=Path, default=None)
    fit.add_argument("--evaluation-root", type=Path, default=Path("artifacts/evaluation"))
    fit.add_argument("--protocol", type=Path, default=Path("protocol/use_contract_v1.json"))
    fit.add_argument("--out-dir", type=Path, default=Path("artifacts/selective-all"))
    fit.add_argument("--run-name", type=str, default=None)
    fit.add_argument("--lead-sets", nargs="+", choices=REDUCED_LEAD_SETS, default=None)
    audit = sub.add_parser("audit", help="audit frozen composite and per-label policies without fitting")
    audit.add_argument("--evaluation-dir", type=Path, required=True)
    audit.add_argument("--policy", type=Path, required=True)
    audit.add_argument("--protocol", type=Path, default=Path("protocol/use_contract_v1.json"))
    audit.add_argument("--out-dir", type=Path, default=Path("artifacts/audit-all"))
    audit.add_argument("--run-name", type=str, default=None)
    return parser


def _resolve_evaluation_dir(requested: Path | None, root: Path) -> Path:
    if requested is not None:
        return requested.resolve()
    candidates = sorted(
        (p.parent for p in root.glob("*/manifest.json")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for candidate in candidates:
        try:
            manifest = json.loads((candidate / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        evaluation = manifest.get("evaluation", {})
        if manifest.get("status") == "complete" and "val" in evaluation.get("splits", []):
            if "test" not in evaluation.get("splits", []) and not evaluation.get("test_evaluated"):
                return candidate.resolve()
    raise FileNotFoundError("no val-only evaluation found; pass --evaluation-dir")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fit":
        evaluation_dir = _resolve_evaluation_dir(args.evaluation_dir, args.evaluation_root.resolve())
        fit_contract_policies(
            evaluation_dir, args.protocol, args.out_dir, args.run_name, args.lead_sets,
        )
    else:
        audit_contract_policies(
            args.evaluation_dir, args.policy, args.protocol, args.out_dir, args.run_name,
        )


if __name__ == "__main__":
    main()
