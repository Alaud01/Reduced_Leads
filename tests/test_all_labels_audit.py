from __future__ import annotations

import unittest

import pandas as pd

from src.audit_all_labels import target_eligibility, targets_from_schema


class AllLabelAuditTest(unittest.TestCase):
    def test_schema_expansion_preserves_group_and_label(self):
        targets = targets_from_schema(
            {"superclass": ["NORM", "MI"], "rhythm": ["AFIB"]},
            ["superclass", "rhythm"],
        )
        self.assertEqual(
            targets,
            [("superclass", "NORM"), ("superclass", "MI"), ("rhythm", "AFIB")],
        )

    def test_sparse_target_remains_visible_but_policy_ineligible(self):
        frame = pd.DataFrame({
            "patient_id": [1, 2, 3, 4, 5, 6],
            "target": [1, 0, 0, 0, 0, 0],
        })
        result = target_eligibility(frame, 2, 2)
        self.assertFalse(result["policy_eligible"])
        self.assertIn("positive_patients=1<2", result["reason"])
        self.assertEqual(result["positive_records"], 1)

    def test_supported_target_is_policy_eligible(self):
        frame = pd.DataFrame({
            "patient_id": [1, 2, 3, 4],
            "target": [1, 1, 0, 0],
        })
        result = target_eligibility(frame, 2, 2)
        self.assertTrue(result["policy_eligible"])
        self.assertIsNone(result["reason"])

# End-to-end fixtures exercise the actual exported schema and record inventory.
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from src.audit_all_labels import fit_all_label_policies, audit_all_label_policies
from src.config import LEAD_SUBSETS
from src.evaluate import sha256_file
from src.use_contract import fit_contract_policies, audit_contract_policies


class PolicyWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'output'
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def export(self, split, targets):
        directory = self.root / split
        (directory / 'predictions').mkdir(parents=True, exist_ok=True)
        n = len(next(iter(targets.values())))
        offset = 0 if split == 'val' else 100000
        records = pd.DataFrame({'ecg_id': np.arange(n) + offset,
                                'patient_id': np.arange(n) + offset})
        records.to_csv(directory / f'records__{split}.csv.gz', index=False)
        schema = {'superclass': [], 'subcode': [], 'rhythm': []}
        for group, label in targets:
            schema[group].append(label)
        for lead in ['12-lead', '2-lead']:
            frames = []
            for (group, label), y in targets.items():
                frame = records.assign(split=split, lead_set=lead, diagnosis_group=group,
                                       diagnosis=label, target=y, logit=np.where(y, 4., -4.))
                frames.append(frame)
            pd.concat(frames).to_csv(directory / 'predictions' / f'{split}__{lead}.csv.gz', index=False)
        manifest = {'status': 'complete', 'checkpoint': {'sha256': 'a' * 64},
                    'label_schema': schema, 'evaluation': {
                        'splits': [split], 'test_evaluated': split == 'test', 'max_n': None,
                        'lead_sets': {lead: LEAD_SUBSETS[lead] for lead in ['12-lead', '2-lead']}}}
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return directory

    def protocol(self, contract=False, bootstrap=0):
        path = Path('protocol/use_contract_v1.json' if contract else 'protocol/all_labels_v1.json')
        payload = json.loads(path.read_text())
        if contract:
            payload['uncertainty']['replicates'] = bootstrap
        else:
            payload['lead_sets'] = ['2-lead']
            payload['uncertainty']['patient_bootstrap_replicates'] = bootstrap
        output = self.root / ('contract.json' if contract else 'generic.json')
        output.write_text(json.dumps(payload))
        return output

    def fit_generic(self):
        y = np.arange(120) % 2
        targets = {('rhythm', 'AFIB'): y}
        self.export('test', targets)
        val = self.export('val', targets)
        protocol = self.protocol()
        run = fit_all_label_policies(val, protocol, self.output, 'fit')
        return run, protocol

    def assert_rejected_before_test_read(self, run, protocol, pattern):
        with patch('src.audit_all_labels._load_prediction_bundle', side_effect=AssertionError('test read')):
            with self.assertRaisesRegex(ValueError, pattern):
                audit_all_label_policies(self.root / 'test', run / 'policy.json', protocol, self.output, 'audit')

    def rehash_policy(self, run):
        path = run / 'manifest.json'
        manifest = json.loads(path.read_text())
        for entry in manifest['artifacts']:
            if entry['path'] == 'policy.json':
                entry['sha256'] = sha256_file(run / 'policy.json')
        path.write_text(json.dumps(manifest))

    def test_contract_fits_every_target_with_group_limits_and_audits_fit_free(self):
        y = np.arange(120) % 2
        targets = {('superclass', 'MI'): y, ('superclass', 'CD'): y,
                   ('superclass', 'NORM'): 1-y, ('subcode', 'IMI'): y,
                   ('subcode', 'NORM'): 1-y, ('rhythm', 'AFIB'): y,
                   ('rhythm', 'AFLT'): np.zeros(120, dtype=int), ('rhythm', 'SR'): 1-y}
        val = self.export('val', targets)
        test = self.export('test', targets)
        protocol = self.protocol(contract=True, bootstrap=3)
        run = fit_contract_policies(val, protocol, self.output, 'fit', ['2-lead'])
        policy_path = run / 'policy.json'
        policy = json.loads(policy_path.read_text())
        entries = {(t['diagnosis_group'], t['diagnosis']): t for t in policy['targets']}
        self.assertEqual(len(entries), len(targets) + 1)
        expected = {('superclass', 'MI'): (.05, .01), ('superclass', 'CD'): (.1, .02),
                    ('superclass', 'NORM'): (.02, None), ('subcode', 'IMI'): (.05, .01),
                    ('subcode', 'NORM'): (.02, None), ('rhythm', 'SR'): (.02, None),
                    ('rhythm', 'AFIB-or-AFLT'): (.1, .02)}
        for key, limits in expected.items():
            entry = entries[key]
            self.assertEqual(entry['status'], 'policy_fitted')
            for stage in ['reduced', '12-lead']:
                risk = entry['lead_policies'][0]['policy'][stage]['risk_control']
                self.assertEqual((risk['max_positive_risk'], risk['max_negative_risk']), limits)
        self.assertEqual(entries[('rhythm', 'AFLT')]['status'], 'insufficient_data')
        before = sha256_file(policy_path)
        with patch('src.audit_all_labels.fit_sequential_policy', side_effect=AssertionError('refit')), \
             patch('src.audit_all_labels.cross_fitted_decisions', side_effect=AssertionError('refit')), \
             patch('src.selective.fit_calibrator', side_effect=AssertionError('refit')):
            audit = audit_contract_policies(test, policy_path, protocol, self.output, 'audit')
        self.assertEqual(before, sha256_file(policy_path))
        manifest = json.loads((audit / 'manifest.json').read_text())
        self.assertTrue(manifest['fit_free'])
        self.assertEqual(manifest['status'], 'complete')
        labels = pd.read_csv(audit / 'label_summary.csv.gz')
        self.assertEqual(len(labels), len(targets) + 1)
        metrics = pd.read_csv(audit / 'policy_metrics.csv.gz')
        self.assertEqual(set(metrics.interpretation), {'assert_normal', 'assert_disease'})
        self.assertEqual(metrics.query("diagnosis=='MI'").iloc[0].max_negative_risk, .01)
        self.assertIn('target-specific limit', (audit / 'report.md').read_text())

    def test_contract_enforces_positive_patient_floor(self):
        y = (np.arange(100) < 20).astype(int)
        targets = {('rhythm', 'AFIB'): y, ('rhythm', 'AFLT'): np.zeros(100, dtype=int)}
        run = fit_contract_policies(self.export('val', targets), self.protocol(True), self.output, 'fit', ['2-lead'])
        policy = json.loads((run / 'policy.json').read_text())
        composite = next(t for t in policy['targets'] if t['diagnosis'] == 'AFIB-or-AFLT')
        self.assertEqual(composite['status'], 'insufficient_data')
        self.assertEqual(composite['eligibility']['positive_patients'], 20)
        self.assertEqual(composite['lead_policies'], [])

    def test_contract_enforces_negative_patient_floor(self):
        y = (np.arange(100) >= 20).astype(int)
        targets = {('rhythm', 'AFIB'): y, ('rhythm', 'AFLT'): np.zeros(100, dtype=int)}
        run = fit_contract_policies(self.export('val', targets), self.protocol(True), self.output, 'fit', ['2-lead'])
        policy = json.loads((run / 'policy.json').read_text())
        composite = next(t for t in policy['targets'] if t['diagnosis'] == 'AFIB-or-AFLT')
        self.assertEqual(composite['status'], 'insufficient_data')
        self.assertEqual(composite['eligibility']['negative_patients'], 20)

    def test_all_sparse_and_single_class_labels_complete_with_readable_empty_outputs(self):
        targets = {('rhythm', 'ZERO'): np.zeros(50, dtype=int),
                   ('rhythm', 'ONE'): np.ones(50, dtype=int)}
        protocol = self.protocol()
        val = self.export('val', targets)
        test = self.export('test', targets)
        run = fit_all_label_policies(val, protocol, self.output, 'fit')
        audit = audit_all_label_policies(test, run / 'policy.json', protocol, self.output, 'audit')
        self.assertEqual(json.loads((audit / 'manifest.json').read_text())['status'], 'complete')
        model = pd.read_csv(audit / 'model_metrics.csv.gz')
        self.assertEqual(len(model), 4)
        self.assertTrue(model.auroc.isna().all())
        self.assertTrue(model.brier.notna().all())
        for filename in ['policy_metrics.csv.gz', 'subgroup_metrics.csv.gz', 'decisions.csv.gz']:
            self.assertTrue(pd.read_csv(audit / filename).empty)
        self.assertTrue(pd.read_csv(run / 'validation_decisions.csv.gz').empty)

    def test_audit_handles_single_class_test_for_fitted_target_and_zero_bootstrap(self):
        run, protocol = self.fit_generic()
        self.export('test', {('rhythm', 'AFIB'): np.zeros(120, dtype=int)})
        audit = audit_all_label_policies(self.root / 'test', run / 'policy.json', protocol, self.output, 'audit')
        self.assertEqual(json.loads((audit / 'manifest.json').read_text())['status'], 'complete')
        self.assertIn('intervals available: 0', (audit / 'report.md').read_text())

    def test_modified_threshold_is_rejected_before_test_read(self):
        run, protocol = self.fit_generic()
        path = run / 'policy.json'; policy = json.loads(path.read_text())
        policy['targets'][0]['lead_policies'][0]['policy']['reduced']['thresholds']['positive'].update(feasible=True, threshold=.5)
        path.write_text(json.dumps(policy))
        self.assert_rejected_before_test_read(run, protocol, 'artifact SHA-256')

    def test_test_exposed_provenance_is_rejected_before_test_read(self):
        run, protocol = self.fit_generic()
        path = run / 'manifest.json'; manifest = json.loads(path.read_text())
        manifest['source_evaluation']['test_predictions_read'] = True
        path.write_text(json.dumps(manifest))
        self.assert_rejected_before_test_read(run, protocol, 'exclude test')

    def test_changed_validation_inputs_are_rejected_before_test_read(self):
        run, protocol = self.fit_generic()
        path = self.root / 'val' / 'predictions' / 'val__2-lead.csv.gz'
        frame = pd.read_csv(path); frame.loc[0, 'logit'] = 0.; frame.to_csv(path, index=False)
        self.assert_rejected_before_test_read(run, protocol, 'validation input SHA-256')

    def test_duplicate_targets_are_rejected_even_with_updated_hash(self):
        run, protocol = self.fit_generic()
        path = run / 'policy.json'; policy = json.loads(path.read_text())
        policy['targets'].append(policy['targets'][0]); path.write_text(json.dumps(policy))
        self.rehash_policy(run)
        self.assert_rejected_before_test_read(run, protocol, 'exactly once')

    def test_stage_limit_drift_is_rejected_even_with_updated_hash(self):
        run, protocol = self.fit_generic()
        path = run / 'policy.json'; policy = json.loads(path.read_text())
        policy['targets'][0]['lead_policies'][0]['policy']['reduced']['risk_control']['max_positive_risk'] = .4
        path.write_text(json.dumps(policy)); self.rehash_policy(run)
        self.assert_rejected_before_test_read(run, protocol, 'stage risk control')

    def test_incomplete_predictions_do_not_fit(self):
        val = self.export('val', {('rhythm', 'AFIB'): np.arange(120) % 2})
        path = val / 'predictions' / 'val__2-lead.csv.gz'
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'incomplete ECG population'):
            fit_all_label_policies(val, self.protocol(), self.output, 'fit')


if __name__ == "__main__":
    unittest.main()
