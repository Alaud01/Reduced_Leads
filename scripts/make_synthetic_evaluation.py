"""Generate a small synthetic evaluation export for demos and documentation.

The output mirrors the file layout and column schema written by
``python -m src.evaluate`` (manifest.json, metrics.json, records__<split>.csv.gz
and predictions/<split>__<lead-set>.csv.gz), but every identifier, demographic
value, target and logit is randomly generated. Nothing is derived from PTB-XL
or from a trained model, so the files can be shared freely and used to try the
policy-fitting and audit commands without downloading data or weights.

Usage (from the repository root):
    python scripts/make_synthetic_evaluation.py                 # writes examples/synthetic-evaluation
    python scripts/make_synthetic_evaluation.py --out-dir /tmp/synthetic

The generator is deterministic for a given --seed.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LEAD_SUBSETS  # noqa: E402
from src.evaluate import PATIENT_COLUMNS, QUALITY_COLUMNS, diagnosis_metrics  # noqa: E402
from src.io_utils import atomic_csv_gz, atomic_json_dump, sigmoid  # noqa: E402

# A compact label schema: enough to exercise the AFIB-or-AFLT composite, an
# assert-normal label and an ischemia label without the full 61-label export.
LABEL_SCHEMA: Dict[str, List[str]] = {
    "superclass": ["NORM", "MI"],
    "subcode": [],
    "rhythm": ["AFIB", "AFLT", "SR"],
}
PREVALENCE = {"NORM": 0.42, "MI": 0.22, "AFIB": 0.14, "AFLT": 0.02, "SR": 0.70}
# Separation of positive and negative logits; the two-lead input is noisier.
SIGNAL = {"12-lead": 3.2, "2-lead": 2.4}
LEAD_SETS = ("12-lead", "2-lead")
SPLIT_OFFSETS = {"val": 900_000, "test": 950_000}
CHECKPOINT_SHA256 = hashlib.sha256(b"reduced-leads synthetic example; no model").hexdigest()


def synthetic_records(rng: np.random.Generator, split: str, n_patients: int) -> pd.DataFrame:
    # About 15% of patients contribute a repeat ECG, as in real cohorts.
    repeats = rng.random(n_patients) < 0.15
    patient_ids = np.repeat(np.arange(n_patients), np.where(repeats, 2, 1)) + SPLIT_OFFSETS[split]
    n = len(patient_ids)
    patient_age = rng.integers(25, 90, size=n_patients).astype(float)
    patient_sex = rng.integers(0, 2, size=n_patients)
    index = patient_ids - SPLIT_OFFSETS[split]
    frame = pd.DataFrame({
        "ecg_id": np.arange(n) + SPLIT_OFFSETS[split] + 1,
        "split": split,
        "patient_id": patient_ids.astype(float),
        "age": patient_age[index],
        "sex": patient_sex[index],
    })
    for column in QUALITY_COLUMNS:
        flagged = rng.random(n) < 0.05
        frame[column] = np.where(flagged, "synthetic-flag", None)
    return frame[["ecg_id", "split", *PATIENT_COLUMNS, *QUALITY_COLUMNS]]


def synthetic_targets(rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
    targets = {name: (rng.random(n) < p).astype(np.int8) for name, p in PREVALENCE.items()}
    # Keep the rhythm labels plausible: atrial fibrillation/flutter excludes
    # sinus rhythm, and normal ECGs exclude myocardial infarction.
    targets["SR"][(targets["AFIB"] == 1) | (targets["AFLT"] == 1)] = 0
    targets["MI"][targets["NORM"] == 1] = 0
    return targets


def export_split(
    rng: np.random.Generator, out_dir: Path, split: str, n_patients: int,
) -> None:
    run_dir = out_dir / split
    records = synthetic_records(rng, split, n_patients)
    targets = synthetic_targets(rng, len(records))
    atomic_csv_gz(records, run_dir / f"records__{split}.csv.gz")
    artifacts = [f"records__{split}.csv.gz"]
    metric_results = []
    latent = {name: rng.normal(size=len(records)) for name in targets}
    for lead_set in LEAD_SETS:
        pieces = []
        groups: Dict[str, object] = {}
        for group, names in LABEL_SCHEMA.items():
            if not names:
                groups[group] = diagnosis_metrics(np.zeros((len(records), 0)), np.zeros((len(records), 0)), [])
                continue
            y = np.column_stack([targets[name] for name in names])
            noise = np.column_stack([
                0.6 * latent[name] + rng.normal(scale=1.0, size=len(records)) for name in names
            ])
            baseline = np.log(np.asarray([PREVALENCE[name] for name in names]))
            # Round so the written values do not depend on platform-specific
            # last-digit differences in exp/log.
            logits = np.round(baseline + SIGNAL[lead_set] * (y - 0.5) + noise, 4)
            probabilities = np.round(sigmoid(logits), 6)
            groups[group] = diagnosis_metrics(y, probabilities, names)
            for column, name in enumerate(names):
                pieces.append(records.assign(
                    lead_set=lead_set,
                    diagnosis_group=group,
                    diagnosis=name,
                    target=y[:, column],
                    logit=logits[:, column],
                    probability=probabilities[:, column],
                ))
        long_frame = pd.concat(pieces, ignore_index=True)[[
            "ecg_id", "split", "lead_set", "diagnosis_group", "diagnosis", "target",
            "logit", "probability", *PATIENT_COLUMNS, *QUALITY_COLUMNS,
        ]]
        relative = f"predictions/{split}__{lead_set}.csv.gz"
        atomic_csv_gz(long_frame, run_dir / relative)
        artifacts.append(relative)
        metric_results.append({
            "split": split, "lead_set": lead_set, "n_records": len(records), "groups": groups,
        })

    atomic_json_dump({"schema_version": 1, "results": metric_results}, run_dir / "metrics.json")
    artifacts.append("metrics.json")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "synthetic": True,
        "description": (
            "Synthetic demonstration export generated by scripts/make_synthetic_evaluation.py. "
            "Identifiers, demographics, targets and logits are random; no PTB-XL data or "
            "trained model was used."
        ),
        "run_name": f"synthetic-{split}",
        "created_at_utc": None,
        "checkpoint": {
            "path": None,
            "sha256": CHECKPOINT_SHA256,
            "epoch": None,
            "training_run_id": None,
            "training_seed": None,
            "selection_fold": 8,
            "calibration_independent": True,
            "training_lead_set": "random",
            "selection_lead_set": "2-lead",
        },
        "dataset": {"root": None, "version": "synthetic"},
        "evaluation": {
            "splits": [split],
            "test_evaluated": split == "test",
            "lead_sets": {name: list(LEAD_SUBSETS[name]) for name in LEAD_SETS},
            "max_n": None,
        },
        "label_schema_source": "synthetic",
        "label_schema": LABEL_SCHEMA,
        "artifacts": artifacts,
    }
    atomic_json_dump(manifest, run_dir / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("examples/synthetic-evaluation"))
    parser.add_argument("--patients", type=int, default=450, help="patients per split")
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    for split in ("val", "test"):
        export_split(rng, args.out_dir, split, args.patients)
    print(f"synthetic exports written to {args.out_dir}/val and {args.out_dir}/test")


if __name__ == "__main__":
    main()
