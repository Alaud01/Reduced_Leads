"""Versioned research-run identity and reproducible epoch-boundary resume."""
from __future__ import annotations
import hashlib
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def capture_rng() -> dict:
    state = np.random.get_state()
    result = {
        'python': random.getstate(),
        'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result['mps'] = torch.mps.get_rng_state()
    return result


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state:
        torch.mps.set_rng_state(state['mps'].cpu())


def prepare_epoch(loader, seed: int, epoch: int) -> None:
    """Workers restart each epoch; their seeds and masks depend on epoch, not uptime."""
    epoch_seed = seed + 1000003 * epoch
    random.seed(epoch_seed)
    np.random.seed(epoch_seed % 2**32)
    torch.manual_seed(epoch_seed)
    loader.generator.manual_seed(epoch_seed)
    loader.dataset.rng = np.random.default_rng(epoch_seed)


def fingerprint_population(frame: pd.DataFrame, data_root: str | None = None) -> str:
    """Ordered metadata/labels plus actual waveform bytes; portable across paths."""
    digest = hashlib.sha256(frame.to_json(orient='split', double_precision=15).encode())
    if data_root is not None:
        for name in sorted(set(frame.filename_lr)):
            for suffix in ('.hea', '.dat'):
                path = Path(data_root) / (str(name) + suffix)
                digest.update((str(name) + suffix).encode())
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
    return digest.hexdigest()


def resume_contract(train, val, *, eval_max_n, quick, total_steps, data_root=None) -> dict:
    return {
        'training_population_sha256': fingerprint_population(train, data_root),
        'validation_population_sha256': fingerprint_population(val, data_root),
        'validation_ecg_ids': [int(x) for x in val.iloc[:eval_max_n].index],
        'eval_max_n': eval_max_n, 'quick': quick, 'total_steps': total_steps,
        'rng_scheme': 'epoch-seeded-workers-v1',
        'torch_version': str(torch.__version__),
    }


def validate_resume(checkpoint, schema, contract) -> None:
    if checkpoint.get('checkpoint_schema_version') != 3:
        raise ValueError('exact resume requires schema-v3; older checkpoints remain inference-only')
    if checkpoint.get('label_schema') != schema:
        raise ValueError('resume label schema/order differs')
    if checkpoint.get('resume_contract') != contract:
        raise ValueError('resume data, validation population, runtime or schedule differs')


def apply_research_split(frame: pd.DataFrame, metadata: pd.DataFrame, selection_fold: int = 8) -> pd.DataFrame:
    """Use patient-disjoint fold 8 for selection; leave fold 9 untouched for risk."""
    if selection_fold != 8:
        raise ValueError('research training requires selection fold 8')
    result = frame.copy()
    result['patient_id'] = metadata.patient_id
    if result.patient_id.isna().any():
        raise ValueError('training requires complete patient IDs')
    result.loc[result.fold_role == 'val', 'fold_role'] = 'calibration'
    result.loc[metadata.strat_fold == selection_fold, 'fold_role'] = 'val'
    if (result.groupby('patient_id').fold_role.nunique() > 1).any():
        raise ValueError('patient leakage across training/selection/calibration/audit splits')
    return result


def best_inference_artifact(last_checkpoint: dict) -> dict:
    """Recover the best model atomically committed inside last.pt, without stale optimizer state."""
    snapshot = last_checkpoint.get('best_snapshot')
    if (not isinstance(snapshot, dict)
            or snapshot['epoch'] > last_checkpoint['epoch']
            or snapshot['metric'] != last_checkpoint['best_metric']):
        raise ValueError('last checkpoint is missing a consistent best-model snapshot')
    keys = ('checkpoint_schema_version', 'model_cfg', 'train_cfg', 'norm',
            'label_schema', 'seed', 'run_id', 'selection_lead_set', 'resume_contract')
    return {**{k: last_checkpoint[k] for k in keys},
            'checkpoint_role': 'best_for_inference',
            'model_state': snapshot['model_state'], 'epoch': snapshot['epoch'],
            'selection_super_auc': snapshot['metric'], 'best_metric': snapshot['metric']}
