"""Shared artifact helpers: hashing, atomic writes, and logit transforms.

These helpers are used by the evaluator, policy fitting, audits, and evaluation
assembly. They live here rather than in ``src.evaluate`` so that the
policy/audit modules do not need to import PyTorch just to write files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(os.path.abspath(__file__)).parent.parent


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for exported NumPy logits."""
    clipped = np.clip(logits, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


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


def atomic_text_dump(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def portable_path(path: str | os.PathLike) -> str:
    """Return ``path`` relative to the repository root when it lies inside it.

    Manifests record this form so exported artifacts do not embed
    machine-specific absolute paths. Paths outside the repository are kept
    absolute because they cannot be reconstructed otherwise.
    """
    # abspath normalizes without following symlinks, so a symlinked dataset
    # directory inside the repository is still recorded relative to it.
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(absolute)
