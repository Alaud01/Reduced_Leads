from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.evaluate import sigmoid
from src.selective import (
    apply_sequential_policy,
    calibrated_probability,
    cross_fitted_decisions,
    fit_calibrator,
    fit_sequential_policy,
    fit_stage_policy,
    decision_metrics,
    clopper_pearson_upper,
    patient_bootstrap_intervals,
    risk_coverage_curve,
    select_threshold,
    split_calibration_and_risk,
)


def logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def stage_policy(positive: float | None, negative: float | None):
    return {
        "calibration": {
            "method": "platt", "slope": 1.0, "intercept": 0.0, "fitted": True,
        },
        "thresholds": {
            "positive": {
                "feasible": positive is not None,
                "threshold": positive,
            },
            "negative": {
                "feasible": negative is not None,
                "threshold": negative,
            },
        },
    }


class CalibrationTest(unittest.TestCase):
    def test_platt_scaling_does_not_worsen_fit_loss(self):
        targets = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
        logits = np.asarray([-12.0, -6.0, -1.0, 1.0, 6.0, 12.0])
        fitted = fit_calibrator(logits, targets)
        calibrated = calibrated_probability(logits, fitted)
        self.assertTrue(fitted["fitted"])
        self.assertLessEqual(
            log_loss(targets, calibrated, labels=[0, 1]),
            log_loss(targets, sigmoid(logits), labels=[0, 1]) + 1e-9,
        )

    def test_single_class_calibration_is_explicit_fallback(self):
        fitted = fit_calibrator(np.asarray([1.0, 2.0]), np.asarray([1, 1]))
        self.assertFalse(fitted["fitted"])
        self.assertEqual(fitted["slope"], 1.0)
        self.assertEqual(fitted["intercept"], 0.0)


class RiskControlTest(unittest.TestCase):
    def test_empty_or_disabled_risk_direction_abstains(self):
        for targets, probabilities, limit in (
            (np.array([]), np.array([]), .1),
            (np.zeros(1000), np.full(1000, .01), None),
        ):
            curve = risk_coverage_curve(targets, probabilities, "negative", limit, .01, 1, 3)
            self.assertFalse(select_threshold(curve, "negative")["feasible"])

    def test_half_probability_is_never_certified_as_negative(self):
        curve = risk_coverage_curve(np.zeros(1000), np.full(1000, .5), "negative", .1, .01, 1, 3)
        self.assertFalse(curve.eligible.any())

    def test_coverage_is_monotone_over_fixed_grid(self):
        targets = np.asarray([0, 0, 1, 1, 1])
        probabilities = np.asarray([0.05, 0.2, 0.7, 0.9, 0.9])
        positive = risk_coverage_curve(
            targets, probabilities, "positive", 0.2, 0.05, 1, 11,
        )
        negative = risk_coverage_curve(
            targets, probabilities, "negative", 0.2, 0.05, 1, 11,
        )
        self.assertTrue((positive.n_trusted.diff().fillna(0) <= 0).all())
        self.assertTrue((negative.n_trusted.diff().fillna(0) >= 0).all())

    def test_infeasible_risk_target_abstains(self):
        targets = np.asarray([0, 1, 0, 1])
        probabilities = np.asarray([0.9, 0.9, 0.1, 0.1])
        curve = risk_coverage_curve(
            targets, probabilities, "positive", 0.01, 0.05, 2, 11,
        )
        chosen = select_threshold(curve, "positive")
        self.assertFalse(chosen["feasible"])
        self.assertIsNone(chosen["threshold"])
        self.assertEqual(chosen["coverage"], 0.0)


class SequentialPolicyTest(unittest.TestCase):
    def test_rescue_is_calibrated_only_on_independent_referred_patients(self):
        # Marginal stage risks are 8%, but naive union risk is 13.79%.
        y = np.r_[np.ones(8400), np.ones(800), np.zeros(800), np.ones(800), np.zeros(800)]
        r = np.r_[np.full(10000, 5.), np.zeros(1600)]
        t = np.r_[np.full(8400, 5.), np.zeros(1600), np.full(1600, 5.)]
        frame = pd.DataFrame({"ecg_id": np.arange(len(y)), "patient_id": np.arange(len(y)),
                              "target": y.astype(int), "reduced_logit": r, "twelve_logit": t})
        old = stage_policy(.9, None)
        self.assertLess(clopper_pearson_upper(800, 10000, .05 / 4 / 101), .1)
        self.assertGreater(decision_metrics(apply_sequential_policy(frame, old, old))["trusted_positive_error"], .1)
        # Repeat the population on distinct patients for enough risk evidence.
        paired = pd.concat([frame.assign(ecg_id=frame.ecg_id + k * len(frame),
                                        patient_id=frame.patient_id + k * len(frame))
                            for k in range(4)], ignore_index=True)
        with patch("src.selective.fit_calibrator", return_value={"slope": 1., "intercept": 0.}), \
             patch("src.selective.fit_stage_policy", wraps=fit_stage_policy) as stages:
            policy, _ = fit_sequential_policy(paired, 7, .1, None, .95, 25, 101)
        reduced_cal, reduced_risk = stages.call_args_list[0].args[:2]
        twelve_cal, twelve_risk = stages.call_args_list[1].args[:2]
        self.assertFalse(set(reduced_risk.patient_id) & set(twelve_risk.patient_id))
        self.assertFalse(set(reduced_cal.patient_id) & (set(reduced_risk.patient_id) | set(twelve_risk.patient_id)))
        self.assertEqual(set(reduced_cal.patient_id), set(twelve_cal.patient_id))
        self.assertTrue((twelve_risk.reduced_logit == 0).all())
        self.assertTrue(policy["reduced"]["thresholds"]["positive"]["feasible"])
        self.assertFalse(policy["12-lead"]["thresholds"]["positive"]["feasible"])
        result = apply_sequential_policy(frame, policy["reduced"], policy["12-lead"])
        self.assertLessEqual(decision_metrics(result)["trusted_positive_error"], .1)
        self.assertNotIn("trust_12_lead", set(result.final_action))

    def test_no_referred_risk_records_disables_rescue_without_crashing(self):
        n = 2000
        frame = pd.DataFrame({"ecg_id": np.arange(n), "patient_id": np.arange(n),
                              "target": np.arange(n) % 2,
                              "reduced_logit": np.where(np.arange(n) % 2, 5., -5.),
                              "twelve_logit": np.where(np.arange(n) % 2, 5., -5.)})
        policy, curves = fit_sequential_policy(frame, 4, .1, .1, .9, 5, 3)
        self.assertEqual(policy["partition"]["referred_risk_records"], 0)
        self.assertFalse(policy["12-lead"]["thresholds"]["positive"]["feasible"])
        self.assertFalse(policy["12-lead"]["thresholds"]["negative"]["feasible"])
        self.assertTrue((curves.loc[curves.stage == "12-lead", "coverage"] == 0).all())

    def test_routes_to_all_three_final_actions(self):
        paired = pd.DataFrame({
            "ecg_id": [1, 2, 3],
            "patient_id": [11, 12, 13],
            "target": [1, 1, 0],
            "reduced_logit": [logit(0.9), logit(0.5), logit(0.5)],
            "twelve_logit": [logit(0.5), logit(0.9), logit(0.5)],
        })
        policy = stage_policy(positive=0.8, negative=0.2)
        result = apply_sequential_policy(paired, policy, policy)
        self.assertEqual(
            result.final_action.tolist(),
            ["trust_reduced", "trust_12_lead", "expert_review"],
        )
        self.assertEqual(result.final_prediction.tolist(), [1, 1, pd.NA])
        intervals = patient_bootstrap_intervals(result, n_bootstrap=20, seed=3)
        self.assertIn("expert_referral_rate", intervals)
        self.assertGreaterEqual(intervals["expert_referral_rate"]["lower"], 0.0)

    def test_grouped_inner_split_has_no_patient_overlap(self):
        rows = []
        for patient in range(20):
            target = patient % 2
            rows.extend([
                {"patient_id": patient, "target": target},
                {"patient_id": patient, "target": target},
            ])
        frame = pd.DataFrame(rows)
        calibration, risk = split_calibration_and_risk(frame, seed=7)
        self.assertFalse(
            set(frame.iloc[calibration].patient_id) & set(frame.iloc[risk].patient_id)
        )
        self.assertEqual(frame.iloc[calibration].target.nunique(), 2)
        self.assertEqual(frame.iloc[risk].target.nunique(), 2)
        changed = frame.assign(target=1 - frame.target)
        changed_calibration, changed_risk = split_calibration_and_risk(changed, seed=7)
        np.testing.assert_array_equal(calibration, changed_calibration)
        np.testing.assert_array_equal(risk, changed_risk)

    def test_crossfit_audits_every_record_once(self):
        probabilities = np.linspace(0.02, 0.98, 60)
        targets = (probabilities >= 0.5).astype(np.int8)
        paired = pd.DataFrame({
            "ecg_id": np.arange(60),
            "patient_id": np.arange(60),
            "target": targets,
            "reduced_logit": np.log(probabilities / (1.0 - probabilities)),
            "twelve_logit": np.log(probabilities / (1.0 - probabilities)),
        })
        decisions, folds = cross_fitted_decisions(
            paired,
            n_splits=3,
            seed=9,
            max_positive_risk=0.2,
            max_negative_risk=0.2,
            confidence=0.8,
            min_trusted=2,
            grid_size=5,
        )
        self.assertEqual(len(decisions), len(paired))
        self.assertFalse(decisions.ecg_id.duplicated().any())
        self.assertEqual(len(folds), 3)


if __name__ == "__main__":
    unittest.main()
