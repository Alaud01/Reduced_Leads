"""Frozen-checkpoint prediction export for diagnosis-specific evaluation.

Validation fold 9 is the default.  Fold 10 is only read when ``test`` is
explicitly included in ``--splits`` so calibration and policy development can
remain isolated from the final test set.

Example:
    python -m src.evaluate --checkpoint checkpoints/best.pt
    python -m src.evaluate --checkpoint checkpoints/best.pt --splits val test
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import sklearn
import torch
from torch.utils.data import DataLoader

from .config import (
    DATA_ROOT, LEAD_SUBSETS, ModelCfg, RHYTHMS, SUBCODES, SUPERCLASSES,
    TEST_FOLD, TRAIN_FOLDS, VAL_FOLD,
)
from .data import PTBXLDataset, compute_train_norm, make_collate
from .labels import build_label_frame, load_metadata
from .model import LeadAwareTransformer


QUALITY_COLUMNS = [
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
    "extra_beats",
    "pacemaker",
]
PATIENT_COLUMNS = ["patient_id", "age", "sex"]
LABEL_GROUPS: Dict[str, List[str]] = {
    "superclass": SUPERCLASSES,
    "subcode": SUBCODES,
    "rhythm": RHYTHMS,
}
OUTPUT_KEYS = {
    "superclass": ("cls_super", "y_super"),
    "subcode": ("cls_sub", "y_sub"),
    "rhythm": ("aux_rhythm", "y_rhythm"),
}


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for exported NumPy logits."""
    clipped = np.clip(logits, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def _model_cfg_from_checkpoint(checkpoint: Mapping) -> ModelCfg:
    raw = checkpoint.get("model_cfg")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint does not contain a model_cfg mapping")
    allowed = {field.name for field in fields(ModelCfg)}
    return ModelCfg(**{key: value for key, value in raw.items() if key in allowed})


def validate_label_schema(checkpoint: Mapping) -> str:
    expected = {name: list(labels) for name, labels in LABEL_GROUPS.items()}
    stored = checkpoint.get("label_schema")
    if stored is None:
        return "runtime-config-fallback"
    if not isinstance(stored, Mapping):
        raise ValueError("checkpoint label_schema must be a mapping")
    normalized = {name: list(stored.get(name, [])) for name in expected}
    if normalized != expected:
        raise ValueError(
            "checkpoint label ordering does not match the current config; "
            "refusing to export mislabelled predictions"
        )
    return "checkpoint"


def normalization_from_checkpoint(
    checkpoint: Mapping,
    label_frame: pd.DataFrame,
    fallback_max_load: int,
) -> tuple[Dict[str, np.ndarray], str]:
    stored = checkpoint.get("norm")
    if isinstance(stored, Mapping) and "mean" in stored and "std" in stored:
        norm = {
            "mean": np.asarray(stored["mean"], dtype=np.float32),
            "std": np.asarray(stored["std"], dtype=np.float32),
        }
        if norm["mean"].shape != (12,) or norm["std"].shape != (12,):
            raise ValueError("checkpoint normalization arrays must each have 12 values")
        if not np.isfinite(norm["mean"]).all() or not np.isfinite(norm["std"]).all():
            raise ValueError("checkpoint normalization contains non-finite values")
        if (norm["std"] <= 0).any():
            raise ValueError("checkpoint normalization standard deviations must be positive")
        return norm, "checkpoint"

    # Backward compatibility for checkpoints produced before normalization was
    # persisted.  This is deterministic for a fixed PTB-XL release, but the
    # manifest records the fallback so it cannot be mistaken for exact metadata.
    norm = compute_train_norm(label_frame, max_load=fallback_max_load)
    return norm, f"recomputed-from-train-first-{fallback_max_load}"


def load_frozen_model(
    checkpoint_path: Path, device: torch.device,
) -> tuple[LeadAwareTransformer, Mapping, ModelCfg]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    cfg = _model_cfg_from_checkpoint(checkpoint)
    model = LeadAwareTransformer(cfg)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, cfg


@torch.inference_mode()
def predict_subset(
    model: LeadAwareTransformer,
    frame: pd.DataFrame,
    norm: Dict[str, np.ndarray],
    keep_leads: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_n: int | None = None,
    num_workers: int = 0,
) -> Dict[str, object]:
    selected = frame.iloc[:max_n] if max_n is not None else frame
    if selected.empty:
        raise ValueError("prediction split is empty")
    dataset = PTBXLDataset(selected, norm=norm, keep_leads=keep_leads)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=make_collate,
    )

    ids: List[np.ndarray] = []
    logits: Dict[str, List[np.ndarray]] = {name: [] for name in LABEL_GROUPS}
    targets: Dict[str, List[np.ndarray]] = {name: [] for name in LABEL_GROUPS}
    for batch in loader:
        x = batch["x"].to(device)
        lead_mask = batch["lead_mask"].to(device)
        outputs = model(x, lead_mask)
        ids.append(batch["ecg_id"].numpy())
        for group, (output_key, target_key) in OUTPUT_KEYS.items():
            if not torch.isfinite(outputs[output_key]).all():
                bad_ids = batch["ecg_id"].tolist()
                raise ValueError(
                    f"non-finite {group} logits for batch containing ECG IDs {bad_ids}"
                )
            logits[group].append(outputs[output_key].float().cpu().numpy())
            targets[group].append(batch[target_key].numpy())

    merged_logits = {name: np.concatenate(parts) for name, parts in logits.items()}
    merged_targets = {name: np.concatenate(parts) for name, parts in targets.items()}
    return {
        "ecg_id": np.concatenate(ids),
        "logits": merged_logits,
        "targets": merged_targets,
    }


def diagnosis_metrics(
    targets: np.ndarray, probabilities: np.ndarray, names: Sequence[str],
) -> Dict[str, object]:
    classes = []
    auc_values: List[float] = []
    ap_values: List[float] = []
    for index, name in enumerate(names):
        y_true = targets[:, index]
        y_score = probabilities[:, index]
        positives = int(y_true.sum())
        negatives = int(y_true.shape[0] - positives)
        auc = None
        average_precision = None
        if positives > 0 and negatives > 0:
            auc = float(roc_auc_score(y_true, y_score))
            average_precision = float(average_precision_score(y_true, y_score))
            auc_values.append(auc)
            ap_values.append(average_precision)
        classes.append({
            "diagnosis": name,
            "n": int(y_true.shape[0]),
            "positives": positives,
            "negatives": negatives,
            "prevalence": float(positives / y_true.shape[0]),
            "auroc": auc,
            "average_precision": average_precision,
        })
    return {
        "n_classes": len(names),
        "n_evaluable_classes": len(auc_values),
        "macro_auroc": float(np.mean(auc_values)) if auc_values else None,
        "macro_average_precision": float(np.mean(ap_values)) if ap_values else None,
        "classes": classes,
    }


def metrics_from_predictions(predictions: Mapping[str, object]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    logits = predictions["logits"]
    targets = predictions["targets"]
    for group, names in LABEL_GROUPS.items():
        group_logits = logits[group]
        probabilities = sigmoid(group_logits)
        result[group] = diagnosis_metrics(targets[group], probabilities, names)
    return result


def long_prediction_frame(
    predictions: Mapping[str, object],
    metadata: pd.DataFrame,
    split: str,
    lead_set: str,
) -> pd.DataFrame:
    ids = np.asarray(predictions["ecg_id"])
    pieces = []
    for group, names in LABEL_GROUPS.items():
        logits = predictions["logits"][group]
        targets = predictions["targets"][group]
        probabilities = sigmoid(logits)
        pieces.append(pd.DataFrame({
            "ecg_id": np.repeat(ids, len(names)),
            "split": split,
            "lead_set": lead_set,
            "diagnosis_group": group,
            "diagnosis": np.tile(np.asarray(names, dtype=object), len(ids)),
            "target": targets.reshape(-1).astype(np.int8),
            "logit": logits.reshape(-1).astype(np.float32),
            "probability": probabilities.reshape(-1).astype(np.float32),
        }))
    result = pd.concat(pieces, ignore_index=True)
    export_columns = [
        column for column in PATIENT_COLUMNS + QUALITY_COLUMNS
        if column in metadata.columns
    ]
    meta = metadata.loc[ids, export_columns].copy()
    meta.index.name = "ecg_id"
    return result.merge(
        meta.reset_index(), on="ecg_id", how="left", validate="many_to_one",
    )


def record_frame(metadata: pd.DataFrame, ids: Iterable[int], split: str) -> pd.DataFrame:
    selected_columns = [
        column for column in PATIENT_COLUMNS + QUALITY_COLUMNS
        if column in metadata.columns
    ]
    result = metadata.loc[list(ids), selected_columns].copy()
    result.insert(0, "split", split)
    result.index.name = "ecg_id"
    return result.reset_index()


def split_frame(label_frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in {"val", "test"}:
        raise ValueError(f"unsupported split: {split}")
    return label_frame[label_frame.fold_role == split]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--splits", nargs="+", choices=["val", "test"], default=["val"],
        help="fold 9 validation by default; include test explicitly to access fold 10",
    )
    parser.add_argument(
        "--lead-sets", nargs="+", choices=list(LEAD_SUBSETS),
        default=list(LEAD_SUBSETS),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=None, help="smoke-test limit per split")
    parser.add_argument(
        "--norm-max-load", type=int, default=2000,
        help="train records used only when an older checkpoint lacks saved normalization",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num-workers cannot be negative")
    if args.max_n is not None and args.max_n <= 0:
        raise ValueError("max-n must be positive")
    if args.norm_max_load <= 0:
        raise ValueError("norm-max-load must be positive")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("splits must not contain duplicates")
    if len(set(args.lead_sets)) != len(args.lead_sets):
        raise ValueError("lead-sets must not contain duplicates")

    checkpoint_hash = sha256_file(checkpoint_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = args.run_name or f"{timestamp}-{checkpoint_path.stem}-{checkpoint_hash[:8]}"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run-name must be a single directory name")
    run_dir = args.out_dir.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    device = resolve_device(args.device)
    print(f"[evaluate] device={device} checkpoint={checkpoint_path}")
    metadata = load_metadata()
    label_frame = build_label_frame(metadata)
    model, checkpoint, model_cfg = load_frozen_model(checkpoint_path, device)
    label_schema_source = validate_label_schema(checkpoint)
    norm, norm_source = normalization_from_checkpoint(
        checkpoint, label_frame, args.norm_max_load,
    )
    if norm_source != "checkpoint":
        print(f"[evaluate] warning: normalization {norm_source}")

    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "epoch": checkpoint.get("epoch"),
            "training_run_id": checkpoint.get("run_id"),
            "training_seed": checkpoint.get("seed"),
        },
        "dataset": {
            "root": DATA_ROOT,
            "version": Path(DATA_ROOT).name.rsplit("-", 1)[-1],
            "train_folds": list(TRAIN_FOLDS),
            "validation_fold": VAL_FOLD,
            "test_fold": TEST_FOLD,
        },
        "evaluation": {
            "splits": args.splits,
            "test_evaluated": "test" in args.splits,
            "lead_sets": {name: list(LEAD_SUBSETS[name]) for name in args.lead_sets},
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_n": args.max_n,
            "device": str(device),
        },
        "model_cfg": {field.name: getattr(model_cfg, field.name) for field in fields(ModelCfg)},
        "normalization": {
            "source": norm_source,
            "mean": norm["mean"].tolist(),
            "std": norm["std"].tolist(),
        },
        "label_schema_source": label_schema_source,
        "label_schema": {name: list(labels) for name, labels in LABEL_GROUPS.items()},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": [],
    }
    manifest_path = run_dir / "manifest.json"
    atomic_json_dump(manifest, manifest_path)

    metric_results = []
    for split in args.splits:
        split_data = split_frame(label_frame, split)
        if args.max_n is not None:
            split_data = split_data.iloc[:args.max_n]
        records = record_frame(metadata, split_data.index, split)
        records_path = run_dir / f"records__{split}.csv.gz"
        atomic_csv_gz(records, records_path)
        manifest["artifacts"].append(str(records_path.relative_to(run_dir)))

        for lead_set in args.lead_sets:
            print(f"[evaluate] split={split} lead_set={lead_set} n={len(split_data)}")
            predictions = predict_subset(
                model=model,
                frame=split_data,
                norm=norm,
                keep_leads=LEAD_SUBSETS[lead_set],
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            long_frame = long_prediction_frame(
                predictions, metadata, split=split, lead_set=lead_set,
            )
            output_path = run_dir / "predictions" / f"{split}__{lead_set}.csv.gz"
            atomic_csv_gz(long_frame, output_path)
            manifest["artifacts"].append(str(output_path.relative_to(run_dir)))
            metric_results.append({
                "split": split,
                "lead_set": lead_set,
                "n_records": len(split_data),
                "groups": metrics_from_predictions(predictions),
            })

    metrics_payload = {"schema_version": 1, "results": metric_results}
    metrics_path = run_dir / "metrics.json"
    atomic_json_dump(metrics_payload, metrics_path)
    manifest["artifacts"].append(str(metrics_path.relative_to(run_dir)))
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json_dump(manifest, manifest_path)
    print(f"[evaluate] complete -> {run_dir}")


if __name__ == "__main__":
    main()
