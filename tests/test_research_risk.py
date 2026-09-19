import unittest
import numpy as np
import pandas as pd
from src.research_risk import (representative_ecgs, fit_patient_crc, apply_patient_crc,
                               repeat_sensitivity, feasibility_rows)
from src.selective import risk_coverage_curve, pair_lead_predictions


class PatientResearchTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'ecg_id': range(100), 'patient_id': np.repeat(range(50), 2),
                             'target': np.zeros(100, dtype=int), 'logit': np.full(100, -5.)})

    def test_selection_is_outcome_blind_and_order_independent(self):
        frame = self.frame()
        chosen = representative_ecgs(frame, 42).ecg_id.tolist()
        frame['target'] = 1
        frame['logit'] = np.arange(len(frame))
        self.assertEqual(chosen, representative_ecgs(frame.sample(frac=1), 42).ecg_id.tolist())
        self.assertEqual(len(chosen), 50)

    def test_crc_counts_patients_and_controls_monotone_loss(self):
        frame = self.frame()
        frame.loc[::10, 'target'] = 1
        policy = fit_patient_crc(frame, .05)
        self.assertEqual(policy['n_patients'], 50)
        risks = np.array([r['corrected_risk'] for r in policy['curve']])
        self.assertTrue((np.diff(risks) <= 0).all())
        result = apply_patient_crc(frame, policy)
        self.assertEqual(result['automated_coverage'], 0.)
        self.assertIsNone(result['conditional_record_error_descriptive'])
        self.assertFalse(policy['conditional_selective_risk_control'])

    def test_zero_errors_still_need_independent_patients(self):
        frame = self.frame()
        policy = fit_patient_crc(frame, .01)
        self.assertIsNone(policy['threshold'])  # 1/51 > .01 even with no errors
        repeated = pd.concat([frame.assign(ecg_id=frame.ecg_id+100*i) for i in range(5)])
        self.assertEqual(policy, fit_patient_crc(repeated, .01))
        enabled = fit_patient_crc(frame, .02)
        self.assertEqual(enabled['threshold'], .5)
        self.assertEqual(apply_patient_crc(frame, enabled)['automated_coverage'], 1.)

    def test_disabled_direction_cannot_automate(self):
        frame = self.frame()
        policy = fit_patient_crc(frame, .1, allow_negative=False)
        self.assertEqual(apply_patient_crc(frame, policy)['automated_coverage'], 0.)

    def test_repeat_analysis_separates_denominators(self):
        frame = self.frame()
        frame['final_prediction'] = pd.array([pd.NA]*100, dtype='Int8')
        frame['final_action'] = 'expert_review'
        frame['initial_action'] = 'obtain_12_lead'
        rows = {r['population']: r for r in repeat_sensitivity(frame, 42)}
        self.assertEqual(rows['one_ecg_per_patient_primary']['n_records'], 50)
        self.assertIsNone(rows['all_records_descriptive']['trusted_negative_error'])

    def test_feasibility_reports_required_sample_size(self):
        curve = risk_coverage_curve(np.zeros(50), np.zeros(50), 'negative', .02, .05/4, 25, 101)
        curve['stage'] = 'reduced'
        curve['max_risk'] = .02
        row = feasibility_rows(curve, .95, 101)[0]
        self.assertEqual(row['minimum_error_free_independent_patients'], 446)
        self.assertEqual(row['status'], 'insufficient_evidence_for_automation')

    def test_pair_rejects_patient_identity_disagreement(self):
        frame = self.frame()
        other = frame.copy()
        other.loc[0, 'patient_id'] = 999
        with self.assertRaisesRegex(ValueError, 'patient IDs'):
            pair_lead_predictions(frame, other)

    def test_zero_observed_errors_have_nonzero_exact_upper_bound(self):
        from src.research_risk import exact_binomial_interval
        lower, upper = exact_binomial_interval(0, 20)
        self.assertEqual(lower, 0.)
        self.assertGreater(upper, .1)
        self.assertEqual(exact_binomial_interval(0, 0), (None, None))
