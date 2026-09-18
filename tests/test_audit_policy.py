from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.audit_policy import run_audit, validate_audit_contract
from src.statistics import (
    binary_performance,
    empirical_risk_coverage_curve,
    resample_patient_clusters,
    selective_performance,
)


def logit(probabilities: np.ndarray) -> np.ndarray:
    return np.log(probabilities / (1.0 - probabilities))


def stage_policy() -> dict[str, object]:
    return {
        "calibration": {
            "method": "platt",
            "slope": 1.0,
            "intercept": 0.0,
            "fitted": True,
        },
        "thresholds": {
            "positive": {"feasible": True, "threshold": 0.8},
            "negative": {"feasible": True, "threshold": 0.2},
        },
    }


class FixedChoice:
    def choice(self, values, size, replace):
        self.values = values
        self.size = size
        self.replace = replace
        return np.asarray([1, 1, 2])


class StatisticsTest(unittest.TestCase):
    def test_patient_resampling_carries_every_record_in_each_cluster_draw(self):
        frame = pd.DataFrame({
            "patient_id": [1, 1, 2, 3],
            "ecg_id": [10, 11, 20, 30],
        })
        sampled = resample_patient_clusters(frame, FixedChoice())
        first = sampled.loc[sampled["__bootstrap_cluster"] == 0, "ecg_id"].tolist()
        second = sampled.loc[sampled["__bootstrap_cluster"] == 1, "ecg_id"].tolist()
        self.assertEqual(first, [10, 11])
        self.assertEqual(second, [10, 11])
        self.assertEqual(
            sampled.loc[sampled["__bootstrap_cluster"] == 2, "ecg_id"].tolist(), [20],
        )

    def test_binary_metrics_and_descriptive_curve_are_finite(self):
        targets = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
        metrics = binary_performance(targets, probabilities)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        curve = empirical_risk_coverage_curve(
            targets, probabilities, coverage_grid=[0.5, 1.0],
        )
        self.assertEqual(curve.n_records.tolist(), [2, 4])
        self.assertTrue((curve.empirical_error == 0.0).all())

    def test_selective_summary_distinguishes_conditional_and_overall_recall(self):
        decisions = pd.DataFrame({
            "patient_id": [1, 2, 3, 4, 5, 6],
            "target": [1, 1, 1, 0, 0, 0],
            "final_action": [
                "trust_12_lead", "trust_reduced", "expert_review",
                "trust_12_lead", "trust_reduced", "expert_review",
            ],
            "final_prediction": [1, 0, np.nan, 1, 0, np.nan],
        })
        metrics = selective_performance(decisions)
        self.assertEqual(metrics["automated_sensitivity"], 0.5)
        self.assertEqual(metrics["positive_automation_recall"], 1.0 / 3.0)
        self.assertEqual(metrics["positive_referral_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["negative_automation_recall"], 1.0 / 3.0)
        self.assertEqual(metrics["negative_referral_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["trusted_positive_records"], 2)
        self.assertEqual(metrics["trusted_positive_errors"], 1)


class AuditPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.evaluation = self.root / "evaluation"
        self.selective = self.root / "selective"
        self.output = self.root / "audit"
        (self.evaluation / "predictions").mkdir(parents=True)
        self.selective.mkdir(parents=True)
        checkpoint_hash = "a" * 64
        self.evaluation_manifest = {
            "schema_version": 1,
            "status": "complete",
            "checkpoint": {"sha256": checkpoint_hash},
            "evaluation": {
                "splits": ["val", "test"],
                "test_evaluated": True,
                "max_n": None,
                "lead_sets": {"12-lead": list(range(12)), "2-lead": ["I", "II"]},
            },
            "label_schema": {"rhythm": ["AFIB"]},
        }
        self.selective_manifest = {
            "schema_version": 1,
            "status": "complete",
            "target": {"diagnosis_group": "rhythm", "diagnosis": "AFIB"},
            "source_evaluation": {"checkpoint": {"sha256": checkpoint_hash}},
            "configuration": {
                "lead_sets": ["2-lead"],
                "max_positive_risk": 0.10,
                "max_negative_risk": 0.02,
            },
        }
        policy = {
            "schema_version": 1,
            "status": "frozen_on_validation",
            "source_checkpoint_sha256": checkpoint_hash,
            "policies": [{
                "diagnosis_group": "rhythm",
                "diagnosis": "AFIB",
                "lead_set": "2-lead",
                "policy": {"reduced": stage_policy(), "12-lead": stage_policy()},
            }],
        }
        self.protocol = {
            "schema_version": 1,
            "protocol_id": "test-afib-audit",
            "status": "exploratory_research",
            "split": "test",
            "target": {"diagnosis_group": "rhythm", "diagnosis": "AFIB"},
            "required_lead_sets": ["2-lead"],
            "bootstrap_replicates": 8,
            "bootstrap_seed": 4,
            "calibration_bins": 4,
            "minimum_subgroup_records": 2,
        }
        (self.evaluation / "manifest.json").write_text(json.dumps(self.evaluation_manifest))
        (self.selective / "manifest.json").write_text(json.dumps(self.selective_manifest))
        self.policy_path = self.selective / "policy.json"
        self.policy_path.write_text(json.dumps(policy))
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text(json.dumps(self.protocol))
        self._write_predictions()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_predictions(self):
        reduced_probability = np.asarray([
            0.05, 0.12, 0.25, 0.45, 0.55, 0.72, 0.82, 0.90, 0.10, 0.40, 0.60, 0.95,
        ])
        twelve_probability = np.asarray([
            0.04, 0.08, 0.10, 0.50, 0.85, 0.90, 0.94, 0.97, 0.12, 0.18, 0.88, 0.92,
        ])
        targets = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
        patient_ids = np.asarray([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        for lead_set, probabilities in (
            ("12-lead", twelve_probability), ("2-lead", reduced_probability),
        ):
            frame = pd.DataFrame({
                "ecg_id": np.arange(100, 112),
                "patient_id": patient_ids,
                "split": "test",
                "lead_set": lead_set,
                "diagnosis_group": "rhythm",
                "diagnosis": "AFIB",
                "target": targets,
                "logit": logit(probabilities),
                "age": np.arange(30, 90, 5),
                "sex": [0, 1] * 6,
                "static_noise": [np.nan] * 10 + ["I", "II"],
            })
            frame.to_csv(
                self.evaluation / "predictions" / f"test__{lead_set}.csv.gz",
                index=False,
                compression="gzip",
            )

    def test_fit_free_audit_writes_hashed_artifacts(self):
        policy_before = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        run_dir = run_audit(
            self.evaluation,
            self.policy_path,
            self.protocol_path,
            self.output,
            run_name="fixture",
        )
        self.assertEqual(policy_before, hashlib.sha256(self.policy_path.read_bytes()).hexdigest())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["fit_free"])
        self.assertEqual(len(manifest["artifacts"]), 7)
        for artifact in manifest["artifacts"]:
            path = run_dir / artifact["path"]
            self.assertTrue(path.exists())
            self.assertEqual(artifact["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        self.assertEqual(metrics["split"], "test")
        self.assertEqual(metrics["results"][0]["lead_set"], "2-lead")
        decisions = pd.read_csv(run_dir / "decisions.csv.gz")
        self.assertEqual(len(decisions), 12)
        self.assertEqual(set(decisions.final_action), {
            "trust_reduced", "trust_12_lead", "expert_review",
        })

    def test_contract_rejects_checkpoint_mismatch(self):
        policy = json.loads(self.policy_path.read_text())
        policy["source_checkpoint_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
            validate_audit_contract(
                self.evaluation_manifest,
                self.selective_manifest,
                policy,
                self.protocol,
            )

    def test_contract_rejects_limited_test_export(self):
        manifest = json.loads(json.dumps(self.evaluation_manifest))
        manifest["evaluation"]["max_n"] = 16
        policy = json.loads(self.policy_path.read_text())
        with self.assertRaisesRegex(ValueError, "max_n"):
            validate_audit_contract(
                manifest,
                self.selective_manifest,
                policy,
                self.protocol,
            )


if __name__ == "__main__":
    unittest.main()
