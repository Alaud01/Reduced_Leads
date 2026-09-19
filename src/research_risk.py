"""Patient-independent selective inference and monotone conformal sensitivity.

Conditional selective risk uses Learn-then-Test (exact binomial tests and a
fixed-grid Bonferroni family). CRC here controls a DIFFERENT loss: any erroneous
automation in a new patient's observed ECG cluster, unconditionally. It must
never replace the conditional false-reassurance limits in the use contract.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
import pandas as pd
from scipy.stats import beta
from .evaluate import sigmoid

PATIENT_METHOD = 'patient-representative-ltt-routed-v3'


def representative_ecgs(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """One outcome-blind pseudorandom ECG per patient, stable under row reorder.

    The seed is protocol-fixed before outcomes are inspected. Apply to complete
    patients BEFORE routing, never select a new representative after referral.
    """
    if frame.patient_id.isna().any() or frame.ecg_id.duplicated().any():
        raise ValueError('representatives require unique ECGs and complete patient IDs')
    result = frame.copy()
    result['_rank'] = [hashlib.sha256(f'{seed}:{int(p)}:{int(e)}'.encode()).hexdigest()
                       for p, e in zip(result.patient_id, result.ecg_id)]
    return (result.sort_values(['_rank', 'ecg_id']).drop_duplicates('patient_id')
            .drop(columns='_rank').sort_values('ecg_id').reset_index(drop=True))


def fit_patient_crc(frame: pd.DataFrame, alpha: float, grid_size: int = 101,
                    *, allow_positive: bool = True, allow_negative: bool = True) -> dict:
    """CRC for monotone patient-cluster any-error loss, with B=1 correction.

    Fixed model, raw sigmoid scores, no fitting on these labels before CRC.
    Increasing confidence threshold only removes automated calls. For each
    patient the loss is max(error AND accepted) over their observed ECGs.
    Choose the least restrictive t with (sum_i L_i(t)+1)/(n+1) <= alpha.
    If none exists, abstain everywhere (known loss zero).
    """
    if not 0 < alpha < 1 or grid_size < 2:
        raise ValueError('invalid CRC alpha/grid')
    if frame.empty or frame.patient_id.isna().any() or frame.ecg_id.duplicated().any():
        raise ValueError('CRC requires complete independent patient clusters')
    if not frame.target.isin([0, 1]).all() or not np.isfinite(frame.logit).all():
        raise ValueError('invalid CRC outcomes/scores')
    probability = sigmoid(frame.logit.to_numpy())
    prediction = probability >= .5
    confidence = np.maximum(probability, 1 - probability)
    enabled = np.where(prediction, allow_positive, allow_negative)
    wrong = prediction != frame.target.to_numpy()
    n = frame.patient_id.nunique()
    rows = []
    chosen = None
    for threshold in np.linspace(.5, 1., grid_size):
        accepted = (confidence >= threshold) & enabled
        errors = pd.Series(wrong & accepted).groupby(frame.patient_id.to_numpy()).max().sum()
        corrected = (int(errors) + 1) / (n + 1)
        rows.append({'threshold': float(threshold), 'patient_errors': int(errors),
                     'corrected_risk': corrected, 'record_coverage': float(accepted.mean())})
        if chosen is None and corrected <= alpha:
            chosen = float(threshold)
    return {'method': 'monotone-patient-cluster-crc-v1', 'alpha': alpha,
            'endpoint': 'unconditional_probability_any_automated_error_per_patient_cluster',
            'guarantee': 'marginal_expectation_under_exchangeable_patient_clusters',
            'conditional_selective_risk_control': False, 'n_patients': int(n),
            'allow_positive': allow_positive, 'allow_negative': allow_negative,
            'threshold': chosen, 'fallback': 'abstain_all' if chosen is None else None,
            'curve': rows}


def exact_binomial_interval(errors: int, n: int) -> tuple[float | None, float | None]:
    """Descriptive two-sided 95% interval; no denominator means undefined."""
    if n < 0 or errors < 0 or errors > n:
        raise ValueError('invalid binomial counts')
    if n == 0:
        return None, None
    return (float(beta.ppf(.025, errors, n-errors+1)) if errors else 0.,
            float(beta.ppf(.975, errors+1, n-errors)) if errors < n else 1.)


def apply_patient_crc(frame: pd.DataFrame, policy: dict) -> dict:
    p = sigmoid(frame.logit.to_numpy())
    pred = p >= .5
    enabled = np.where(pred, policy['allow_positive'], policy['allow_negative'])
    accepted = (np.maximum(p, 1-p) >= policy['threshold']) & enabled if policy['threshold'] is not None else np.zeros(len(p), dtype=bool)
    errors = accepted & (pred != frame.target.to_numpy())
    cluster_errors = pd.Series(errors).groupby(frame.patient_id.to_numpy()).max()
    n, k = len(cluster_errors), int(cluster_errors.sum())
    lower, upper = exact_binomial_interval(k, n)
    return {'n_patients': len(cluster_errors),
            'unconditional_patient_error_ci_lower': lower,
            'unconditional_patient_error_ci_upper': upper,
            'interval_method': 'descriptive_patient_binomial_exact_95', 'n_records': len(frame),
            'automated_coverage': float(accepted.mean()),
            'unconditional_patient_error': float(cluster_errors.mean()),
            'conditional_record_error_descriptive': float(errors.sum()/accepted.sum()) if accepted.any() else None,
            'independent_calibration_verified': policy.get('independent_calibration_verified', False),
            'endpoint': policy['endpoint'],
            'status': 'no_automated_decisions' if not accepted.any() else 'automated_decisions_observed'}


def feasibility_rows(curves: pd.DataFrame, confidence: float, grid_size: int) -> list[dict]:
    """Explain infeasibility without adapting the grid/limits to audit outcomes."""
    rows = []
    for (stage, direction), group in curves.groupby(['stage', 'direction']):
        limits = group.max_risk.dropna()
        limit = float(limits.iloc[0]) if len(limits) else None
        rows.append({'stage': stage, 'direction': direction, 'max_risk': limit,
                     'minimum_error_free_independent_patients':
                         math.ceil(math.log((1-confidence)/(4*grid_size))/math.log1p(-limit)) if limit else None,
                     'maximum_observed_trusted': int(group.n_trusted.max()),
                     'best_risk_upper': float(group.risk_upper.min()) if group.risk_upper.notna().any() else None,
                     'status': 'direction_disabled' if limit is None else
                         ('feasible' if group.eligible.any() else 'insufficient_evidence_for_automation')})
    return rows


def repeat_sensitivity(decisions: pd.DataFrame, seed: int) -> list[dict]:
    from .selective import decision_metrics
    counts = decisions.groupby('patient_id').ecg_id.transform('size')
    populations = {'all_records_descriptive': decisions,
                   'one_ecg_per_patient_primary': representative_ecgs(decisions, seed),
                   'repeat_patient_records_descriptive': decisions[counts > 1],
                   'single_record_patients_descriptive': decisions[counts == 1]}
    rows = []
    for name, subset in populations.items():
        if not subset.empty:
            metrics = decision_metrics(subset)
            automated = subset.final_action != 'expert_review'
            errors = (automated & subset.final_prediction.ne(subset.target)).fillna(False)
            metrics['patient_any_automated_error'] = float(errors.groupby(subset.patient_id).max().mean())
            rows.append({'population': name, 'n_patients': int(subset.patient_id.nunique()),
                         'status': 'no_automated_decisions' if not automated.any() else 'automated_decisions_observed', **metrics})
    return rows
