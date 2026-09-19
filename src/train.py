"""
Training entrypoint for the lead-aware selective-prediction backbone.

Usage:
    python -m src.train                  # full run (CPU/MPS)
    python -m src.train --epochs 1 --quick  # quick smoke test on a small subset
    python -m src.train --device cpu      # force CPU

The script:
  1. builds the label frame + train/val/test splits
  2. computes per-lead normalization stats on the train split
  3. computes inverse-frequency class weights for super/sub/rhythm heads
  4. trains with random lead dropping or one fixed canonical lead set
  5. evaluates on the validation subset(s) appropriate to that regime
  6. checkpoints by superclass macro-AUC on the training regime's lead set
  7. writes an epoch-boundary ``last.pt`` that can resume interrupted runs
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, get_worker_info
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    DATA_ROOT, LEAD_SUBSETS, ModelCfg, N_LEADS, RHYTHMS,
    SUBCODES, SUPERCLASSES, TrainCfg,
)
from .labels import load_metadata, build_label_frame, class_weights, split_indices
from .data import PTBXLDataset, compute_train_norm, make_collate
from .model import LeadAwareTransformer
from .losses import MultiTaskLoss
from .training_state import capture_rng, restore_rng, prepare_epoch, resume_contract, validate_resume, apply_research_split, best_inference_artifact


def plot_training_curves(
    log_path: str, out_dir: str, run_id: str | None = None,
) -> str | None:
    """
    Read the JSONL training log and produce a 3-panel figure:
      (1) train loss components vs. step (raw + EMA smoothing)
      (2) val macro-AUC vs. epoch for each lead subset (superclass head)
      (3) val macro-AUC by head (super/sub/rhythm) at the 12-lead subset

    Returns the saved figure path, or None if the log is empty/missing.
    """
    if not os.path.exists(log_path):
        print(f"[plot] no log at {log_path}")
        return None
    log = pd.read_json(log_path, lines=True)
    if run_id is not None and "run_id" in log.columns:
        log = log[log["run_id"] == run_id].copy()
    if log.empty:
        print("[plot] log is empty")
        return None

    batch = log[log.get("type", pd.Series([None]*len(log))) != "epoch"].copy()
    epoch_rows = log[log.get("type", pd.Series([None]*len(log))) == "epoch"].copy()
    if epoch_rows.empty and batch.empty:
        print("[plot] no usable rows in log")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- (1) train loss components vs. step ----------------------------
    if not batch.empty and "step" in batch.columns:
        loss_cols = {
            "loss": ("Total loss", "#222222"),
            "loss_super": ("Superclass", "#4c72b0"),
            "loss_sub": ("Subcode", "#dd8452"),
            "loss_rhythm": ("Rhythm", "#55a868"),
            "loss_lp": ("Lead presence", "#c44e52"),
        }
        for col, (label, color) in loss_cols.items():
            if col in batch.columns:
                y = batch[col].to_numpy()
                x = batch["step"].to_numpy()
                axes[0].plot(x, y, alpha=0.25, color=color, lw=0.7)
                # EMA smoothing
                alpha = 0.3
                ema = np.zeros_like(y)
                ema[0] = y[0]
                for i in range(1, len(y)):
                    ema[i] = alpha * y[i] + (1 - alpha) * ema[i-1]
                axes[0].plot(x, ema, color=color, lw=1.6, label=label)
        axes[0].set_xlabel("global step")
        axes[0].set_ylabel("BCE loss")
        axes[0].set_title("Training loss components")
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].grid(True, alpha=0.3)

    # ---- (2) val super-AUC vs. epoch by lead subset --------------------
    if not epoch_rows.empty and "val" in epoch_rows.columns:
        # Each epoch row's "val" is a dict of subset -> {super_auc, sub_auc, ...}
        # Gather subset names from the first epoch row that has val data.
        subset_names = []
        for v in epoch_rows["val"]:
            if isinstance(v, dict) and v:
                subset_names = list(v.keys())
                break
        subset_colors = {
            "12-lead": "#222222",
            "6-lead-limb": "#4c72b0",
            "4-lead": "#dd8452",
            "3-lead": "#55a868",
            "2-lead": "#c44e52",
            "1-lead-I": "#8172b3",
            "1-lead-II": "#937860",
            # Backward-compatible names used by pre-manifest logs.
            "I+II+III+aVR+aVL+aVF": "#4c72b0",
            "I+II+III+V2": "#dd8452",
            "I+II+V2": "#55a868",
            "I+II": "#c44e52",
            "I": "#8172b3",
        }
        epochs = epoch_rows["epoch"].to_numpy()
        for sn in subset_names:
            aucs = []
            for v in epoch_rows["val"]:
                if isinstance(v, dict) and sn in v and "super_auc" in v[sn]:
                    a = v[sn]["super_auc"]
                    aucs.append(a if a == a else np.nan)  # NaN guard
                else:
                    aucs.append(np.nan)
            axes[1].plot(epochs, aucs, "-o", ms=3, lw=1.4,
                         color=subset_colors.get(sn, None),
                         label=sn)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("macro AUC (superclass)")
        axes[1].set_title("Val AUC by lead subset")
        axes[1].legend(fontsize=7, ncol=2, loc="lower right")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0.45, 1.0)

        # ---- (3) val AUC by head at model-selection lead set -----------
        selection_subset = "12-lead"
        if "selection_lead_set" in epoch_rows.columns:
            configured = epoch_rows["selection_lead_set"].dropna()
            if not configured.empty:
                selection_subset = str(configured.iloc[0])
        head_cols = {
            "super_auc": ("Superclass (5)", "#4c72b0"),
            "sub_auc": ("Subcode (44)", "#dd8452"),
            "rhythm_auc": ("Rhythm (12)", "#55a868"),
        }
        for key, (label, color) in head_cols.items():
            vals = []
            for v in epoch_rows["val"]:
                if (isinstance(v, dict) and selection_subset in v
                        and key in v[selection_subset]):
                    a = v[selection_subset][key]
                    vals.append(a if a == a else np.nan)
                else:
                    vals.append(np.nan)
            axes[2].plot(epochs, vals, "-o", ms=3, lw=1.4,
                         color=color, label=label)
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("macro AUC")
        axes[2].set_title(f"Val AUC by head ({selection_subset})")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0.45, 1.0)

    fig.suptitle("Lead-aware transformer — training curves", y=1.01,
                 fontsize=13, weight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {path}")
    return path


def get_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        print("[train] MPS not available, falling back to CPU")
        name = "cpu"
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Seed model initialization, shuffling, and host-side randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    """Give each DataLoader worker an independent deterministic NumPy RNG."""
    info = get_worker_info()
    if info is None:
        return
    worker_seed = int(info.seed % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    if hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(worker_seed)


def lead_subset_name(leads: List[str]) -> str:
    key = tuple(leads)
    for name, configured in LEAD_SUBSETS.items():
        if key == configured:
            return name
    return "+".join(leads)


def training_leads(cfg: TrainCfg) -> List[str] | None:
    """Resolve the configured training regime to fixed leads or augmentation."""
    if cfg.training_lead_set == "random":
        return None
    if cfg.training_lead_set not in LEAD_SUBSETS:
        raise ValueError(
            "training_lead_set must be 'random' or a canonical lead-set name"
        )
    return list(LEAD_SUBSETS[cfg.training_lead_set])


def selection_lead_set(cfg: TrainCfg) -> str:
    """Lead set used for model selection in a training regime."""
    return cfg.selection_lead or ("12-lead" if cfg.training_lead_set == "random" else cfg.training_lead_set)


def configure_training_regime(cfg: TrainCfg) -> str:
    """Apply fixed-regime evaluation/loss settings and return selection leads."""
    training_leads(cfg)  # validate the name
    selected = selection_lead_set(cfg)
    if selected not in LEAD_SUBSETS:
        raise ValueError("invalid selection lead set")
    if cfg.training_lead_set != "random" and selected != cfg.training_lead_set:
        raise ValueError("fixed models must select on their training lead set")
    if cfg.training_lead_set != "random":
        cfg.eval_subsets = [list(LEAD_SUBSETS[selected])]
        # Lead-presence prediction is useful only when masks vary. With fixed
        # inputs its target is constant and adds no meaningful comparator task.
        cfg.w_lead_presence = 0.0
    return selected


def build_dataloaders(
    df: pd.DataFrame,
    norm: Dict[str, np.ndarray],
    cfg: TrainCfg,
    seed: int,
    quick: bool = False,
):
    """Build loaders using random dropping or a fixed direct-training subset."""
    rng = np.random.default_rng(seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_idx, val_idx, _ = split_indices(df)
    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]

    if quick:
        train_df = train_df.iloc[:256]
        val_df = val_df.iloc[:128]

    fixed_training_leads = training_leads(cfg)
    train_dataset_kwargs = (
        {"drop_min": cfg.drop_min, "drop_max": cfg.drop_max, "rng": rng}
        if fixed_training_leads is None
        else {"keep_leads": fixed_training_leads}
    )
    train_ds = PTBXLDataset(train_df, norm=norm, **train_dataset_kwargs)
    val_ds = PTBXLDataset(
        val_df, norm=norm,
        keep_leads=LEAD_SUBSETS[selection_lead_set(cfg)],
    )

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=make_collate, drop_last=True,
        persistent_workers=False, worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=make_collate,
        persistent_workers=False,
    )
    return train_dl, val_dl, train_df, val_df


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate_subset(
    model: nn.Module,
    df: pd.DataFrame,
    norm: Dict[str, np.ndarray],
    keep_leads: List[str],
    device: torch.device,
    batch_size: int = 64,
    max_n: int | None = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    Evaluate the model at a single fixed lead subset.

    Returns dict with per-head macro AUC (skipped if a class has only one
    value in the targets).
    """
    from .config import SUPERCLASSES, SUBCODES, RHYTHMS
    sub_df = df
    if max_n is not None:
        sub_df = df.iloc[:max_n]
    ds = PTBXLDataset(sub_df, norm=norm, keep_leads=keep_leads)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate)

    model.eval()
    sup_preds, sub_preds, rhy_preds = [], [], []
    sup_y, sub_y, rhy_y = [], [], []
    amp_device = device.type if device.type in ("mps", "cuda") else "cpu"
    use_eval_amp = bool(use_amp and device.type in ("mps", "cuda"))
    for batch in dl:
        batch = move_batch(batch, device)
        with torch.amp.autocast(amp_device, dtype=torch.float16, enabled=use_eval_amp):
            out = model(batch["x"], batch["lead_mask"])
        sup_preds.append(torch.sigmoid(out["cls_super"]).cpu().numpy())
        sub_preds.append(torch.sigmoid(out["cls_sub"]).cpu().numpy())
        rhy_preds.append(torch.sigmoid(out["aux_rhythm"]).cpu().numpy())
        sup_y.append(batch["y_super"].cpu().numpy())
        sub_y.append(batch["y_sub"].cpu().numpy())
        rhy_y.append(batch["y_rhythm"].cpu().numpy())

    sup_preds = np.concatenate(sup_preds); sup_y = np.concatenate(sup_y)
    sub_preds = np.concatenate(sub_preds); sub_y = np.concatenate(sub_y)
    rhy_preds = np.concatenate(rhy_preds); rhy_y = np.concatenate(rhy_y)

    def _macro_auc(preds, y, names):
        aucs = []
        per_class = {}
        for i, name in enumerate(names):
            if y[:, i].sum() == 0 or y[:, i].sum() == y.shape[0]:
                continue
            try:
                a = roc_auc_score(y[:, i], preds[:, i])
                aucs.append(a)
                per_class[name] = a
            except Exception:
                continue
        return (float(np.mean(aucs)) if aucs else float("nan")), per_class

    sup_auc, sup_pc = _macro_auc(sup_preds, sup_y, SUPERCLASSES)
    sub_auc, sub_pc = _macro_auc(sub_preds, sub_y, SUBCODES)
    rhy_auc, rhy_pc = _macro_auc(rhy_preds, rhy_y, RHYTHMS)
    return {
        "super_auc": sup_auc, "sub_auc": sub_auc, "rhythm_auc": rhy_auc,
        "super_per_class": sup_pc, "sub_per_class": sub_pc, "rhythm_per_class": rhy_pc,
    }


def train_one_epoch(
    model: nn.Module, loss_fn: MultiTaskLoss, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_dl: DataLoader, device: torch.device, grad_clip: float,
    epoch: int, global_step: int, log_every: int = 25,
    use_amp: bool = False, scaler: torch.amp.GradScaler | None = None,
) -> tuple[Dict[str, float], int, List[Dict]]:
    model.train()
    totals = {"loss": 0.0, "loss_super": 0.0, "loss_sub": 0.0,
              "loss_rhythm": 0.0, "loss_lead_presence": 0.0}
    n = 0
    batch_log: List[Dict] = []
    amp_device = device.type if device.type in ("mps", "cuda") else "cpu"
    pbar = tqdm(train_dl, desc=f"epoch {epoch}", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    for bi, batch in enumerate(pbar):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(amp_device, dtype=torch.float16, enabled=use_amp):
            out = model(batch["x"], batch["lead_mask"])
            losses = loss_fn(out, batch)
            loss = losses["loss"]
        optimizer_updated = True
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            # GradScaler lowers its scale when non-finite gradients cause it
            # to skip the update. Successful steps retain or grow the scale.
            optimizer_updated = scaler.get_scale() >= scale_before
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if optimizer_updated:
            scheduler.step()
        # Keep the existing batch-based logging axis, including skipped steps.
        global_step += 1
        bs = batch["x"].shape[0]
        n += bs
        for k in totals:
            totals[k] += float(losses[k].detach()) * bs

        if bi % log_every == 0 or bi == len(train_dl) - 1:
            # running averages so far this epoch
            run = {k: v / max(n, 1) for k, v in totals.items()}
            lr_now = optimizer.param_groups[0]["lr"]
            n_leads_mean = float(batch["lead_mask"].sum(1).mean())
            pbar.set_postfix({
                "loss": f"{run['loss']:.3f}",
                "sup": f"{run['loss_super']:.3f}",
                "sub": f"{run['loss_sub']:.3f}",
                "rhy": f"{run['loss_rhythm']:.3f}",
                "lp": f"{run['loss_lead_presence']:.3f}",
                "lr": f"{lr_now:.1e}",
                "nL": f"{n_leads_mean:.1f}",
            })
            batch_log.append({
                "epoch": epoch, "step": global_step, "batch": bi,
                "optimizer_updated": optimizer_updated,
                "loss": run["loss"], "loss_super": run["loss_super"],
                "loss_sub": run["loss_sub"], "loss_rhythm": run["loss_rhythm"],
                "loss_lp": run["loss_lead_presence"], "lr": lr_now,
                "n_leads_mean": n_leads_mean,
            })
    return {k: v / max(n, 1) for k, v in totals.items()}, global_step, batch_log


def make_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # cosine decay to 0
        import math
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def atomic_torch_save(payload: Mapping[str, object], path: str | Path) -> None:
    """Save a checkpoint without exposing a partially written final path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def resumable_checkpoint(
    *, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None, model_cfg: ModelCfg,
    train_cfg: TrainCfg, norm: Mapping[str, np.ndarray],
    label_schema: Mapping[str, List[str]], seed: int, run_id: str,
    global_step: int, best_metric: float, selection_lead: str,
    current_metric: float,
    contract: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Build the common payload used by best and epoch-boundary checkpoints."""
    return {
        "checkpoint_schema_version": 3,
        "checkpoint_role": "last_for_resume",
        "rng_state": capture_rng(),
        "resume_contract": dict(contract or {}),
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "norm": {"mean": norm["mean"].tolist(), "std": norm["std"].tolist()},
        "label_schema": dict(label_schema),
        "seed": seed,
        "run_id": run_id,
        "global_step": global_step,
        "best_metric": best_metric,
        "selection_lead_set": selection_lead,
        "selection_super_auc": current_metric,
    }


def _dataclass_from_mapping(cls, raw: object):
    if not isinstance(raw, Mapping):
        return cls()
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in allowed})


def load_resume_state(
    checkpoint: Mapping[str, object], model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
) -> tuple[int, int, float]:
    """Restore a schema-v3 checkpoint and return start epoch, step, and best."""
    if checkpoint.get("checkpoint_role") == "best_for_inference":
        raise ValueError("resume last.pt; best.pt is an inference-only snapshot")
    required = {"model_state", "optimizer_state", "scheduler_state", "global_step"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            "checkpoint predates resumable training state; missing: "
            + ", ".join(missing)
        )
    if checkpoint.get("checkpoint_schema_version") != 3 or "rng_state" not in checkpoint:
        raise ValueError("checkpoint predates resumable schema-v3 random state")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    stored_scaler = checkpoint.get("scaler_state")
    if scaler is not None:
        if not isinstance(stored_scaler, Mapping):
            raise ValueError("AMP resume requires a saved scaler_state")
        scaler.load_state_dict(stored_scaler)
    elif stored_scaler is not None:
        raise ValueError("cannot resume an AMP checkpoint with AMP disabled")
    restore_rng(checkpoint["rng_state"])
    completed_epoch = int(checkpoint.get("epoch", 0))
    return (
        completed_epoch + 1,
        int(checkpoint["global_step"]),
        float(checkpoint.get("best_metric", -1.0)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lead", choices=list(LEAD_SUBSETS), default=None)
    parser.add_argument("--lead-presence-weight", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="smoke test on small subset")
    parser.add_argument("--eval-max-n", type=int, default=None,
                        help="limit val eval samples (useful for quick)")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true",
                        help="disable MPS/CUDA fp16 autocast (default: enabled)")
    parser.add_argument("--eval-full-every", type=int, default=None,
                        help="full 7-subset val eval cadence (default from TrainCfg)")
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="re-enable gradient checkpointing (default: off; only needed on OOM)")
    parser.add_argument(
        "--train-lead-set", choices=["random", *LEAD_SUBSETS], default=None,
        help=("random lead dropping (default) or a fixed direct-training "
              "comparator such as 12-lead or 2-lead"),
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="resume a schema-v3 last.pt at the next epoch",
    )
    return parser


def main():
    args = build_parser().parse_args()
    resume_checkpoint: Mapping[str, object] | None = None
    if args.resume is not None:
        resume_path = args.resume.resolve()
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if not isinstance(resume_checkpoint, Mapping):
            raise ValueError("resume checkpoint must contain a mapping")
        if (resume_checkpoint.get('checkpoint_schema_version') != 3
                or resume_checkpoint.get('checkpoint_role') != 'last_for_resume'):
            raise ValueError('resume requires schema-v3 last.pt; other checkpoints are inference-only')
        if resume_checkpoint.get('label_schema') != {
            'superclass': SUPERCLASSES, 'subcode': SUBCODES, 'rhythm': RHYTHMS,
        }:
            raise ValueError('resume label schema/order differs')

    mcfg = _dataclass_from_mapping(
        ModelCfg, resume_checkpoint.get("model_cfg") if resume_checkpoint else None,
    )
    tcfg = _dataclass_from_mapping(
        TrainCfg, resume_checkpoint.get("train_cfg") if resume_checkpoint else None,
    )
    seed = int(
        args.seed if args.seed is not None
        else (resume_checkpoint.get("seed", 42) if resume_checkpoint else 42)
    )
    if args.selection_lead is not None: tcfg.selection_lead = args.selection_lead
    if args.lead_presence_weight is not None: tcfg.w_lead_presence = args.lead_presence_weight
    if args.epochs is not None: tcfg.epochs = args.epochs
    if args.batch_size is not None: tcfg.batch_size = args.batch_size
    if args.num_workers is not None: tcfg.num_workers = args.num_workers
    if args.device is not None: tcfg.device = args.device
    if args.lr is not None: tcfg.lr = args.lr
    if args.out_dir is not None: tcfg.out_dir = args.out_dir
    elif args.resume is not None: tcfg.out_dir = str(args.resume.resolve().parent)
    if args.no_amp: tcfg.use_amp = False
    if args.eval_full_every is not None: tcfg.eval_full_every = args.eval_full_every
    if args.grad_ckpt: mcfg.use_grad_ckpt = True
    if args.train_lead_set is not None: tcfg.training_lead_set = args.train_lead_set
    training_leads(tcfg)  # validate before touching data or output files
    if tcfg.batch_size <= 0 or tcfg.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers cannot be negative")
    if args.eval_max_n is not None and args.eval_max_n <= 0:
        raise ValueError("eval-max-n must be positive")
    if resume_checkpoint is not None:
        saved_contract = resume_checkpoint.get("resume_contract", {})
        if args.eval_max_n is None:
            args.eval_max_n = saved_contract.get("eval_max_n")
        if args.quick != saved_contract.get("quick", False):
            raise ValueError("quick mode cannot change on resume")
    if args.quick:
        tcfg.epochs = 1
        tcfg.batch_size = 16
        if tcfg.training_lead_set == "random":
            tcfg.drop_min = 0
            tcfg.drop_max = 6
    if resume_checkpoint is not None:
        stored_train = _dataclass_from_mapping(TrainCfg, resume_checkpoint.get("train_cfg"))
        stored_model = _dataclass_from_mapping(ModelCfg, resume_checkpoint.get("model_cfg"))
        if asdict(stored_model) != asdict(mcfg):
            raise ValueError("model configuration overrides are not allowed when resuming")
        immutable = tuple(f.name for f in fields(TrainCfg) if f.name not in ("out_dir", "eval_subsets"))
        changed = [name for name in immutable if getattr(stored_train, name) != getattr(tcfg, name)]
        if changed:
            raise ValueError(
                "resume must keep the original schedule/training regime; changed: "
                + ", ".join(changed)
            )
        if args.seed is not None and seed != int(resume_checkpoint.get("seed", seed)):
            raise ValueError("seed cannot be changed when resuming")
    seed_everything(seed)
    if resume_checkpoint is not None and not resume_checkpoint.get("run_id"):
        raise ValueError("resume checkpoint is missing run_id")
    run_id = str(resume_checkpoint.get("run_id")) if resume_checkpoint else (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-seed{seed}"
    )
    selected_lead = configure_training_regime(tcfg)
    os.makedirs(tcfg.out_dir, exist_ok=True)

    device = get_device(tcfg.device)
    print(f"[train] device = {device}")
    use_amp = bool(tcfg.use_amp and device.type in ("mps", "cuda"))
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    print(f"[train] amp(fp16) = {use_amp} | batch_size = {tcfg.batch_size} "
          f"| grad_ckpt = {mcfg.use_grad_ckpt} | train_leads = {tcfg.training_lead_set} "
          f"| selection_leads = {selected_lead} | eval_full_every = {tcfg.eval_full_every}")

    print("[train] loading metadata ...")
    Y = load_metadata()
    print(f"[train] building label frame (n={len(Y)}) ...")
    df = apply_research_split(build_label_frame(Y), Y, tcfg.selection_fold)
    train_idx, val_idx, test_idx = split_indices(df)
    print(f"[train] splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    if resume_checkpoint is None:
        print("[train] computing train normalization stats ...")
        norm = compute_train_norm(df, max_load=200 if args.quick else 2000)
        norm_source = (
            "PTB-XL training folds excluding model-selection fold, first "
            f"{200 if args.quick else 2000} records"
        )
    else:
        print("[train] restoring normalization stats from checkpoint ...")
        stored_norm = resume_checkpoint.get("norm")
        if not isinstance(stored_norm, Mapping):
            raise ValueError("resume checkpoint is missing normalization statistics")
        norm = {
            "mean": np.asarray(stored_norm.get("mean"), dtype=np.float32),
            "std": np.asarray(stored_norm.get("std"), dtype=np.float32),
        }
        if norm["mean"].shape != (N_LEADS,) or norm["std"].shape != (N_LEADS,):
            raise ValueError("resume normalization arrays must each have 12 values")
        if not np.isfinite(norm["mean"]).all() or not np.isfinite(norm["std"]).all():
            raise ValueError("resume normalization contains non-finite values")
        if (norm["std"] <= 0).any():
            raise ValueError("resume normalization standard deviations must be positive")
        norm_source = "checkpoint"
    print(f"[train]   per-lead mean: {norm['mean'].round(4).tolist()}")
    print(f"[train]   per-lead std : {norm['std'].round(4).tolist()}")

    train_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "data_root": DATA_ROOT,
        "model_cfg": asdict(mcfg),
        "train_cfg": asdict(tcfg),
        "normalization": {
            "mean": norm["mean"].tolist(),
            "std": norm["std"].tolist(),
            "source": norm_source,
        },
        "label_schema": {
            "superclass": SUPERCLASSES,
            "subcode": SUBCODES,
            "rhythm": RHYTHMS,
        },
        "torch_version": str(torch.__version__),
    }
    print("[train] computing class weights ...")
    weights_df = df[df.fold_role == "train"].iloc[:256] if args.quick else df
    super_w = class_weights(weights_df, "superclass", tau=0.5)
    sub_w = class_weights(weights_df, "subcode", tau=0.5)
    rhy_w = class_weights(weights_df, "rhythm", tau=0.5)
    print(f"[train]   super weights: {super_w.round(3).tolist()}")
    print(f"[train]   sub   weights: min={sub_w.min():.3f} max={sub_w.max():.3f}")
    print(f"[train]   rhythm weights: {rhy_w.round(3).tolist()}")

    print("[train] building dataloaders ...")
    train_dl, val_dl, train_df, val_df = build_dataloaders(df, norm, tcfg, seed, quick=args.quick)
    print(f"[train]   train batches: {len(train_dl)}  val batches: {len(val_dl)}")

    print("[train] building model ...")
    model = LeadAwareTransformer(mcfg).to(device)
    n_params = model.num_parameters()
    print(f"[train]   parameters: {n_params:,}  ({n_params/1e6:.2f}M)")

    loss_fn = MultiTaskLoss(
        w_super=tcfg.w_super, w_sub=tcfg.w_sub,
        w_rhythm=tcfg.w_rhythm, w_lead_presence=tcfg.w_lead_presence,
        super_weights=torch.from_numpy(super_w),
        sub_weights=torch.from_numpy(sub_w),
        rhythm_weights=torch.from_numpy(rhy_w),
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay,
    )
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * tcfg.epochs
    scheduler = make_warmup_scheduler(optimizer, tcfg.warmup_steps, total_steps)
    print(f"[train] steps/epoch={steps_per_epoch} total_steps={total_steps}")

    best_metric = -1.0
    eval_max_n = args.eval_max_n if args.eval_max_n else (128 if args.quick else None)
    contract = resume_contract(
        train_df, val_df, eval_max_n=eval_max_n, quick=args.quick,
        total_steps=total_steps, data_root=DATA_ROOT,
    )
    contract["effective_device"] = str(device)
    contract["split_design"] = "train1-7_select8_calibrate9_audit10"
    train_manifest['resume_contract'] = contract
    manifest_path = Path(tcfg.out_dir) / f"train_manifest_{run_id}.json"
    if resume_checkpoint is not None:
        validate_resume(resume_checkpoint, train_manifest['label_schema'], contract)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"resume manifest missing: {manifest_path}")
        saved_manifest = json.loads(manifest_path.read_text())
        if (saved_manifest.get('resume_contract') != contract
                or saved_manifest.get('label_schema') != train_manifest['label_schema']
                or saved_manifest.get('run_id') != run_id
                or saved_manifest.get('model_cfg') != train_manifest['model_cfg']
                or any(saved_manifest.get('train_cfg', {}).get(k) != v for k, v in train_manifest['train_cfg'].items() if k != 'out_dir')
                or any(saved_manifest.get('normalization', {}).get(k) != train_manifest['normalization'][k]
                       for k in ('mean', 'std'))):
            raise ValueError('resume manifest identity mismatch')
        # last.pt owns the committed best snapshot; interrupted best.pt writes
        # and migration with only last.pt can be recovered without losing selection.
        if float(resume_checkpoint.get('best_metric', -1)) > -1:
            atomic_torch_save(best_inference_artifact(resume_checkpoint), Path(tcfg.out_dir) / 'best.pt')
    else:
        manifest_path.write_text(json.dumps(train_manifest, indent=2))
    global_step = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        start_epoch, global_step, best_metric = load_resume_state(
            resume_checkpoint, model, optimizer, scheduler, scaler,
        )
        if start_epoch > tcfg.epochs:
            raise ValueError(
                f"checkpoint already completed epoch {start_epoch - 1} of {tcfg.epochs}"
            )
        print(f"[train] resumed run={run_id} at epoch={start_epoch} step={global_step}")
    log_path = os.path.join(tcfg.out_dir, "train_log.jsonl")
    print(f"[train] logging batch metrics -> {log_path}")

    # Discard only this run's logs beyond the committed epoch boundary.
    if resume_checkpoint is not None and Path(log_path).exists():
        lines = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
        retained = [row for row in lines if row.get('run_id') != run_id
                    or row.get('epoch', 0) < start_epoch]
        temporary = Path(log_path + '.tmp')
        temporary.write_text(''.join(json.dumps(row) + '\n' for row in retained))
        os.replace(temporary, log_path)
    best_snapshot = resume_checkpoint.get("best_snapshot") if resume_checkpoint else None
    for epoch in range(start_epoch, tcfg.epochs + 1):
        prepare_epoch(train_dl, seed, epoch)
        t0 = time.time()
        train_totals, global_step, batch_log = train_one_epoch(
            model, loss_fn, optimizer, scheduler, train_dl, device,
            tcfg.grad_clip, epoch, global_step,
            use_amp=use_amp, scaler=scaler,
        )
        for entry in batch_log:
            entry["run_id"] = run_id
        t_train = time.time() - t0

        # append batch-level logs to JSONL
        with open(log_path, "a") as f:
            for entry in batch_log:
                f.write(json.dumps(entry) + "\n")

        # The random-lead model evaluates all subsets periodically and its
        # selection subset otherwise. Fixed comparators have one eval subset.
        eval_t0 = time.time()
        print(f"\r[train] [epoch {epoch}/{tcfg.epochs}] train_loss={train_totals['loss']:.4f} "
              f"super={train_totals['loss_super']:.4f} sub={train_totals['loss_sub']:.4f} "
              f"rhy={train_totals['loss_rhythm']:.4f} lp={train_totals['loss_lead_presence']:.4f} "
              f"({t_train:.1f}s)            ")
        cadence = max(1, int(tcfg.eval_full_every))
        is_full_eval = (epoch == 1) or (epoch == tcfg.epochs) or (epoch % cadence == 0)
        if is_full_eval:
            eval_list = tcfg.eval_subsets
        else:
            # Selection subset every epoch; full evaluation is periodic only
            # for the random-lead model.
            eval_list = [s for s in tcfg.eval_subsets
                         if lead_subset_name(s) == selected_lead] or tcfg.eval_subsets[:1]
            print(f"[train]   ({selected_lead}-only eval; full eval every {cadence} epochs)")
        subset_metrics = {}
        for subset in eval_list:
            m = evaluate_subset(
                model, val_df, norm, subset, device,
                batch_size=tcfg.batch_size, max_n=eval_max_n,
                use_amp=use_amp,
            )
            tag = lead_subset_name(subset)
            subset_metrics[tag] = m
            print(f"[train]   val[{tag:20s}] super={m['super_auc']:.3f} "
                  f"sub={m['sub_auc']:.3f} rhy={m['rhythm_auc']:.3f}")
        t_eval = time.time() - eval_t0

        # epoch-level summary to JSONL
        epoch_summary = {
            "type": "epoch", "run_id": run_id,
            "epoch": epoch, "step": global_step,
            "train_loss": train_totals["loss"],
            "train_loss_super": train_totals["loss_super"],
            "train_loss_sub": train_totals["loss_sub"],
            "train_loss_rhythm": train_totals["loss_rhythm"],
            "train_loss_lp": train_totals["loss_lead_presence"],
            "train_time_s": t_train, "eval_time_s": t_eval,
            "training_lead_set": tcfg.training_lead_set,
            "selection_lead_set": selected_lead,
            "val": {tag: {k: v for k, v in m.items() if not isinstance(v, dict)}
                    for tag, m in subset_metrics.items()},
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(epoch_summary) + "\n")

        # Compare models on the inputs they were trained to consume.
        selected_metrics = subset_metrics.get(selected_lead, {})
        cur = selected_metrics.get("super_auc", float("nan"))
        if not np.isnan(cur) and cur > best_metric:
            best_metric = cur
            best_snapshot = {
                'epoch': epoch, 'metric': cur,
                'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            print(f"[train]   new best super_auc@{selected_lead}={cur:.4f}")
        last_payload = resumable_checkpoint(
            epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, model_cfg=mcfg,
            train_cfg=tcfg, norm=norm,
            label_schema=train_manifest["label_schema"], seed=seed,
            run_id=run_id, global_step=global_step,
            best_metric=best_metric, selection_lead=selected_lead,
            current_metric=cur, contract=contract,
        )
        last_payload['best_snapshot'] = best_snapshot
        atomic_torch_save(last_payload, os.path.join(tcfg.out_dir, "last.pt"))
        if best_snapshot is not None:
            atomic_torch_save(best_inference_artifact(last_payload), os.path.join(tcfg.out_dir, 'best.pt'))
        print(f"[train]   eval {t_eval:.1f}s | best super_auc@{selected_lead}={best_metric:.4f}")

    final = os.path.join(tcfg.out_dir, "last.pt")
    print(f"[train] saved final -> {final}")
    print(f"[train] done. best super_auc@{selected_lead}={best_metric:.4f}")

    # Plot training curves from the JSONL log
    plot_training_curves(
        os.path.join(tcfg.out_dir, "train_log.jsonl"), tcfg.out_dir, run_id=run_id,
    )


if __name__ == "__main__":
    main()
