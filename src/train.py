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
  4. trains with random lead-dropping augmentation
  5. evaluates on the val split at every configured lead subset
  6. checkpoints the best model by 12-lead superclass macro-AUC
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import random
import time
from typing import Dict, List
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
    DATA_ROOT, LEAD_NAMES, LEAD_SUBSETS, ModelCfg, N_LEADS, RHYTHMS,
    SUBCODES, SUPERCLASSES, TrainCfg,
)
from .labels import load_metadata, build_label_frame, class_weights, split_indices
from .data import PTBXLDataset, compute_train_norm, make_collate
from .model import LeadAwareTransformer
from .losses import MultiTaskLoss


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

        # ---- (3) val AUC by head at 12-lead ----------------------------
        head_cols = {
            "super_auc": ("Superclass (5)", "#4c72b0"),
            "sub_auc": ("Subcode (44)", "#dd8452"),
            "rhythm_auc": ("Rhythm (12)", "#55a868"),
        }
        for key, (label, color) in head_cols.items():
            vals = []
            for v in epoch_rows["val"]:
                if isinstance(v, dict) and "12-lead" in v and key in v["12-lead"]:
                    a = v["12-lead"][key]
                    vals.append(a if a == a else np.nan)
                else:
                    vals.append(np.nan)
            axes[2].plot(epochs, vals, "-o", ms=3, lw=1.4,
                         color=color, label=label)
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("macro AUC")
        axes[2].set_title("Val AUC by head (12-lead)")
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


def build_dataloaders(
    df: pd.DataFrame,
    norm: Dict[str, np.ndarray],
    cfg: TrainCfg,
    seed: int,
    quick: bool = False,
):
    """Build train + val dataloaders.  Val uses the full 12-lead subset."""
    rng = np.random.default_rng(seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_idx, val_idx, _ = split_indices(df)
    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]

    if quick:
        train_df = train_df.iloc[:256]
        val_df = val_df.iloc[:128]

    train_ds = PTBXLDataset(
        train_df, norm=norm,
        drop_min=cfg.drop_min, drop_max=cfg.drop_max, rng=rng,
    )
    val_ds = PTBXLDataset(
        val_df, norm=norm,
        keep_leads=LEAD_NAMES,  # full 12-lead eval
    )

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, collate_fn=make_collate, drop_last=True,
        persistent_workers=True, worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=1, collate_fn=make_collate,
        persistent_workers=True,
    )
    return train_dl, val_dl, train_df, val_df


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate_subset(
    model: nn.Module,
    df: pd.DataFrame,
    norm: Dict[str, np.ndarray],
    keep_leads: List[str],
    device: torch.device,
    batch_size: int = 64,
    max_n: int | None = None,
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
    for batch in dl:
        batch = move_batch(batch, device)
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
) -> tuple[Dict[str, float], int, List[Dict]]:
    model.train()
    totals = {"loss": 0.0, "loss_super": 0.0, "loss_sub": 0.0,
              "loss_rhythm": 0.0, "loss_lead_presence": 0.0}
    n = 0
    batch_log: List[Dict] = []
    pbar = tqdm(train_dl, desc=f"epoch {epoch}", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    for bi, batch in enumerate(pbar):
        batch = move_batch(batch, device)
        out = model(batch["x"], batch["lead_mask"])
        losses = loss_fn(out, batch)
        loss = losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="smoke test on small subset")
    p.add_argument("--eval-max-n", type=int, default=None,
                   help="limit val eval samples (useful for quick)")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-seed{args.seed}"
    )

    mcfg = ModelCfg()
    tcfg = TrainCfg()
    if args.epochs is not None: tcfg.epochs = args.epochs
    if args.batch_size is not None: tcfg.batch_size = args.batch_size
    if args.device is not None: tcfg.device = args.device
    if args.lr is not None: tcfg.lr = args.lr
    if args.out_dir is not None: tcfg.out_dir = args.out_dir
    if args.quick:
        tcfg.epochs = 1
        tcfg.batch_size = 16
        tcfg.drop_min = 0
        tcfg.drop_max = 6
    os.makedirs(tcfg.out_dir, exist_ok=True)

    device = get_device(tcfg.device)
    print(f"[train] device = {device}")

    print("[train] loading metadata ...")
    Y = load_metadata()
    print(f"[train] building label frame (n={len(Y)}) ...")
    df = build_label_frame(Y)
    train_idx, val_idx, test_idx = split_indices(df)
    print(f"[train] splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    print("[train] computing train normalization stats ...")
    norm = compute_train_norm(df, max_load=200 if args.quick else 2000)
    print(f"[train]   per-lead mean: {norm['mean'].round(4).tolist()}")
    print(f"[train]   per-lead std : {norm['std'].round(4).tolist()}")

    train_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "data_root": DATA_ROOT,
        "model_cfg": asdict(mcfg),
        "train_cfg": asdict(tcfg),
        "normalization": {
            "mean": norm["mean"].tolist(),
            "std": norm["std"].tolist(),
            "source": (
                "PTB-XL train folds 1-8, first "
                f"{200 if args.quick else 2000} records"
            ),
        },
        "label_schema": {
            "superclass": SUPERCLASSES,
            "subcode": SUBCODES,
            "rhythm": RHYTHMS,
        },
        "torch_version": str(torch.__version__),
    }
    manifest_path = os.path.join(tcfg.out_dir, f"train_manifest_{run_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(train_manifest, f, indent=2)
    print(f"[train] manifest -> {manifest_path}")

    print("[train] computing class weights ...")
    super_w = class_weights(df, "superclass", tau=0.5)
    sub_w = class_weights(df, "subcode", tau=0.5)
    rhy_w = class_weights(df, "rhythm", tau=0.5)
    print(f"[train]   super weights: {super_w.round(3).tolist()}")
    print(f"[train]   sub   weights: min={sub_w.min():.3f} max={sub_w.max():.3f}")
    print(f"[train]   rhythm weights: {rhy_w.round(3).tolist()}")

    print("[train] building dataloaders ...")
    train_dl, val_dl, train_df, val_df = build_dataloaders(df, norm, tcfg, args.seed, quick=args.quick)
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
    global_step = 0
    log_path = os.path.join(tcfg.out_dir, "train_log.jsonl")
    print(f"[train] logging batch metrics -> {log_path}")

    for epoch in range(1, tcfg.epochs + 1):
        t0 = time.time()
        train_totals, global_step, batch_log = train_one_epoch(
            model, loss_fn, optimizer, scheduler, train_dl, device,
            tcfg.grad_clip, epoch, global_step,
        )
        for entry in batch_log:
            entry["run_id"] = run_id
        t_train = time.time() - t0

        # append batch-level logs to JSONL
        with open(log_path, "a") as f:
            for entry in batch_log:
                f.write(json.dumps(entry) + "\n")

        # Evaluate across lead subsets
        eval_t0 = time.time()
        print(f"\r[train] [epoch {epoch}/{tcfg.epochs}] train_loss={train_totals['loss']:.4f} "
              f"super={train_totals['loss_super']:.4f} sub={train_totals['loss_sub']:.4f} "
              f"rhy={train_totals['loss_rhythm']:.4f} lp={train_totals['loss_lead_presence']:.4f} "
              f"({t_train:.1f}s)            ")
        subset_metrics = {}
        for subset in tcfg.eval_subsets:
            m = evaluate_subset(
                model, val_df, norm, subset, device,
                batch_size=tcfg.batch_size, max_n=eval_max_n,
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
            "val": {tag: {k: v for k, v in m.items() if not isinstance(v, dict)}
                    for tag, m in subset_metrics.items()},
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(epoch_summary) + "\n")

        # track 12-lead super AUC as the main metric
        m12 = subset_metrics.get("12-lead", {})
        cur = m12.get("super_auc", float("nan"))
        if not np.isnan(cur) and cur > best_metric:
            best_metric = cur
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_cfg": mcfg.__dict__,
                "train_cfg": asdict(tcfg),
                "norm": {"mean": norm["mean"].tolist(), "std": norm["std"].tolist()},
                "label_schema": train_manifest["label_schema"],
                "seed": args.seed,
                "run_id": run_id,
                "val_super_auc_12": cur,
            }
            path = os.path.join(tcfg.out_dir, "best.pt")
            torch.save(ckpt, path)
            print(f"[train]   new best super_auc@12lead={cur:.4f} -> {path}")
        print(f"[train]   eval {t_eval:.1f}s | best super_auc@12lead={best_metric:.4f}")

    # final save
    final = os.path.join(tcfg.out_dir, "last.pt")
    torch.save({
        "epoch": tcfg.epochs,
        "model_state": model.state_dict(),
        "model_cfg": mcfg.__dict__,
        "train_cfg": asdict(tcfg),
        "norm": {"mean": norm["mean"].tolist(), "std": norm["std"].tolist()},
        "label_schema": train_manifest["label_schema"],
        "seed": args.seed,
        "run_id": run_id,
    }, final)
    print(f"[train] saved final -> {final}")
    print(f"[train] done. best super_auc@12lead={best_metric:.4f}")

    # Plot training curves from the JSONL log
    plot_training_curves(
        os.path.join(tcfg.out_dir, "train_log.jsonl"), tcfg.out_dir, run_id=run_id,
    )


if __name__ == "__main__":
    main()
