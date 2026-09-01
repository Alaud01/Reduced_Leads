"""
PTB-XL dataset for the lead-aware selective-prediction backbone.

Key features:
- Loads the 10 s @ 100 Hz low-resolution waveform per ecg_id (1000 x 12 samples).
- Per-lead normalization (z-score per lead computed from the TRAIN split stats,
  cached, applied at load time so the model sees standardized signals).
- Random lead-dropping augmentation: every batch sample has its own random
  per-lead drop mask drawn so the model sees the full lead-availability
  spectrum (12 down to 2 leads).  Dropped leads are zeroed AND their lead
  presence bit is set to 0 so the encoder's missing-lead token pathway is
  trained.
- At eval time a fixed lead subset can be requested (no randomness) so we
  can benchmark 12/6/4/3/2/1-lead performance with one trained model.
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    DATA_ROOT, LEAD_NAMES, N_LEADS, SIGNAL_LEN, SIGNAL_HZ, SAMPLE_RATE_SUFFIX,
    SUPERCLASSES, SUBCODES, RHYTHMS,
)

_LEAD_IDX = {name: i for i, name in enumerate(LEAD_NAMES)}


def load_signal(filename_lr: str) -> np.ndarray:
    """Load the 1000x12 low-resolution signal for one record using wfdb."""
    import wfdb
    path = os.path.join(DATA_ROOT, filename_lr)
    sig, _ = wfdb.rdsamp(path)
    sig = np.asarray(sig, dtype=np.float32)  # (1000, 12)
    if sig.shape[0] != SIGNAL_LEN:
        # pad/truncate to 1000 if needed
        if sig.shape[0] < SIGNAL_LEN:
            pad = np.zeros((SIGNAL_LEN - sig.shape[0], sig.shape[1]), dtype=np.float32)
            sig = np.concatenate([sig, pad], axis=0)
        else:
            sig = sig[:SIGNAL_LEN]
    return sig


def compute_train_norm(df: pd.DataFrame, max_load: int = 2000) -> Dict[str, float]:
    """
    Compute per-lead mean/std from a subset of TRAIN-split signals.

    Returns dict with 'mean' and 'std' arrays of shape (N_LEADS,).
    """
    train = df[df.fold_role == "train"]
    n = min(max_load, len(train))
    sample_ids = train.index.to_numpy()[:n]
    sigs = []
    for eid in sample_ids:
        try:
            sigs.append(load_signal(train.loc[eid, "filename_lr"]))
        except Exception:
            continue
    if not sigs:
        # fall back to dataset-wide zero mean / unit std
        return {"mean": np.zeros(N_LEADS, dtype=np.float32),
                "std": np.ones(N_LEADS, dtype=np.float32)}
    arr = np.stack(sigs, axis=0)  # (n, 1000, 12)
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    std = np.maximum(std, 1e-6)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


class PTBXLDataset(Dataset):
    """
    Multi-task dataset.

    Each item returns a dict:
        x              : (N_LEADS, SIGNAL_LEN) float32, normalized, dropped leads zeroed
        lead_mask      : (N_LEADS,) bool, True = lead present
        y_super        : (N_SUPER,) float32 multi-hot
        y_sub          : (N_SUB,)   float32 multi-hot
        y_rhythm       : (N_RHYTHM,) float32 multi-hot
        ecg_id         : int
    """

    def __init__(
        self,
        df: pd.DataFrame,
        norm: Optional[Dict[str, np.ndarray]] = None,
        keep_leads: Optional[Sequence[str]] = None,
        drop_min: int = 0,
        drop_max: int = 0,
        rng: Optional[np.random.Generator] = None,
        return_ids: bool = True,
    ):
        self.df = df
        self.norm = norm or {"mean": np.zeros(N_LEADS, dtype=np.float32),
                              "std": np.ones(N_LEADS, dtype=np.float32)}
        self.rng = rng if rng is not None else np.random.default_rng()
        self.return_ids = return_ids

        # fixed evaluation subset: only these leads are present, others always dropped
        self.keep_mask = np.ones(N_LEADS, dtype=bool)
        if keep_leads is not None:
            self.keep_mask[:] = False
            for name in keep_leads:
                if name in _LEAD_IDX:
                    self.keep_mask[_LEAD_IDX[name]] = True
        self.fixed_subset = keep_leads is not None

        self.drop_min = int(drop_min)
        self.drop_max = int(drop_max)

        # pre-extract label arrays for speed
        self._super_cols = [f"superclass_{c}" for c in SUPERCLASSES]
        self._sub_cols = [f"subcode_{c}" for c in SUBCODES]
        self._rhy_cols = [f"rhythm_{c}" for c in RHYTHMS]
        self._y_super = df[self._super_cols].to_numpy(dtype=np.float32)
        self._y_sub = df[self._sub_cols].to_numpy(dtype=np.float32)
        self._y_rhy = df[self._rhy_cols].to_numpy(dtype=np.float32)
        self._fn_lr = df["filename_lr"].to_numpy()
        self._ids = df.index.to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def _sample_drop_mask(self) -> np.ndarray:
        """
        Returns (N_LEADS,) bool array: True = present, False = dropped.
        """
        if self.fixed_subset:
            return self.keep_mask.copy()

        # random drop: choose how many to drop from [drop_min, drop_max]
        n_drop = int(self.rng.integers(self.drop_min, self.drop_max + 1))
        n_drop = min(n_drop, N_LEADS - 1)  # always keep >=1 lead
        if n_drop <= 0:
            return np.ones(N_LEADS, dtype=bool)

        mask = np.ones(N_LEADS, dtype=bool)
        drop_idx = self.rng.choice(N_LEADS, size=n_drop, replace=False)
        mask[drop_idx] = False
        return mask

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sig = load_signal(self._fn_lr[i])  # (1000, 12) float32

        # normalize per-lead
        sig = (sig - self.norm["mean"]) / self.norm["std"]

        # lead dropping
        lead_mask = self._sample_drop_mask()  # (12,) bool
        sig = sig * lead_mask.astype(np.float32)[None, :]  # zero dropped leads

        # transpose -> (N_LEADS, SIGNAL_LEN) for patch tokenizer
        sig = sig.T.astype(np.float32)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(sig)),
            "lead_mask": torch.from_numpy(np.ascontiguousarray(lead_mask.astype(np.float32))),
            "y_super": torch.from_numpy(self._y_super[i].copy()),
            "y_sub": torch.from_numpy(self._y_sub[i].copy()),
            "y_rhythm": torch.from_numpy(self._y_rhy[i].copy()),
            "ecg_id": int(self._ids[i]),
        }


def make_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack list-of-dicts into batched tensors."""
    out = {}
    for k in batch[0]:
        if k == "ecg_id":
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out