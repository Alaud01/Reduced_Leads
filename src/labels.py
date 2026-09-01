"""
Build multi-label targets and class weights from PTB-XL metadata.

Outputs module-level constants used by the dataset and trainer:
- SUPERCLASS_IDX, SUBCODE_IDX, RHYTHM_IDX  : dict[str,int]
- build_label_frame(Y) -> pd.DataFrame with per-ecg multi-hot arrays
   plus boolean masks and per-class prevalence for weighting.
"""
from __future__ import annotations
import ast
from typing import Dict, List
import numpy as np
import pandas as pd

from .config import (
    DATA_ROOT, SUPERCLASSES, SUBCODES, RHYTHMS, TRAIN_FOLDS, VAL_FOLD, TEST_FOLD,
)
import os

SUPERCLASS_IDX: Dict[str, int] = {c: i for i, c in enumerate(SUPERCLASSES)}
SUBCODE_IDX: Dict[str, int] = {c: i for i, c in enumerate(SUBCODES)}
RHYTHM_IDX: Dict[str, int] = {c: i for i, c in enumerate(RHYTHMS)}

# Map subcode -> superclass using scp_statements (built lazily / cached)
_SUB_TO_SUPER: Dict[str, str] | None = None


def _load_scp():
    return pd.read_csv(os.path.join(DATA_ROOT, "scp_statements.csv"), index_col=0)


def sub_to_super() -> Dict[str, str]:
    """Map each diagnostic subcode to its diagnostic_class superclass."""
    global _SUB_TO_SUPER
    if _SUB_TO_SUPER is not None:
        return _SUB_TO_SUPER
    agg = _load_scp()
    diag = agg[agg.diagnostic == 1]
    _SUB_TO_SUPER = {code: cls for code, cls in zip(diag.index, diag.diagnostic_class)}
    return _SUB_TO_SUPER


def load_metadata() -> pd.DataFrame:
    """Load ptbxl_database.csv with parsed scp_codes dict + fold_role."""
    Y = pd.read_csv(
        os.path.join(DATA_ROOT, "ptbxl_database.csv"), index_col="ecg_id",
    )
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

    def _fold_role(f: int) -> str:
        return {TEST_FOLD: "test", VAL_FOLD: "val"}.get(int(f), "train")
    Y["fold_role"] = Y.strat_fold.apply(_fold_role)
    return Y


def build_label_frame(Y: pd.DataFrame) -> pd.DataFrame:
    """
    Build multi-hot targets per ecg_id for superclasses, subcodes, rhythms,
    plus per-class prevalence (computed on the train split only so weights
    reflect the training distribution).

    Returns DataFrame indexed by ecg_id with columns:
        superclass_<NAME>  (0/1)
        subcode_<NAME>     (0/1)
        rhythm_<NAME>      (0/1)
    Also adds `filename_lr`, `fold_role` passthrough columns.
    """
    rows = []
    for ecg_id, row in Y.iterrows():
        d = row.scp_codes
        sup_hot = np.zeros(len(SUPERCLASSES), dtype=np.int8)
        sub_hot = np.zeros(len(SUBCODES), dtype=np.int8)
        rhy_hot = np.zeros(len(RHYTHMS), dtype=np.int8)

        s2s = sub_to_super()
        for code in d:
            if code in SUBCODE_IDX:
                sub_hot[SUBCODE_IDX[code]] = 1
            if code in s2s and s2s[code] in SUPERCLASS_IDX:
                sup_hot[SUPERCLASS_IDX[s2s[code]]] = 1
            if code in RHYTHM_IDX:
                rhy_hot[RHYTHM_IDX[code]] = 1

        rows.append({
            "ecg_id": ecg_id,
            "filename_lr": row.filename_lr,
            "filename_hr": row.filename_hr,
            "fold_role": row.fold_role,
            **{f"superclass_{c}": int(sup_hot[i]) for i, c in enumerate(SUPERCLASSES)},
            **{f"subcode_{c}": int(sub_hot[i]) for i, c in enumerate(SUBCODES)},
            **{f"rhythm_{c}": int(rhy_hot[i]) for i, c in enumerate(RHYTHMS)},
        })
    df = pd.DataFrame(rows).set_index("ecg_id")
    return df


def class_weights(df: pd.DataFrame, prefix: str, tau: float = 1.0) -> np.ndarray:
    """
    Inverse-frequency class weights computed on the TRAIN split.

    weight_i = (N_train / (K * count_i)) ** tau
    tau=1 -> standard inv-freq; tau<1 -> dampens extreme weights.
    """
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    train = df[df.fold_role == "train"]
    n = len(train)
    counts = train[cols].sum(axis=0).to_numpy(dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (n / (len(cols) * counts)) ** tau
    return weights.astype(np.float32)


def split_indices(df: pd.DataFrame):
    """Return (train_idx, val_idx, test_idx) numpy arrays of ecg_ids."""
    return (
        df.index[df.fold_role == "train"].to_numpy(),
        df.index[df.fold_role == "val"].to_numpy(),
        df.index[df.fold_role == "test"].to_numpy(),
    )