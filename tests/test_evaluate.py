from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import (
    LABEL_GROUPS,
    build_parser,
    diagnosis_metrics,
    long_prediction_frame,
    normalization_from_checkpoint,
    sigmoid,
    validate_label_schema,
)
from src.config import DATA_ROOT
from src.labels import load_metadata


class EvaluateHelpersTest(unittest.TestCase):
    def test_validation_is_default_and_both_single_leads_are_available(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.splits, ["val"])
        self.assertIn("1-lead-I", args.lead_sets)
        self.assertIn("1-lead-II", args.lead_sets)

    def test_diagnosis_metrics_handles_evaluable_and_single_class_targets(self):
        targets = np.asarray([
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 1],
        ], dtype=np.float32)
        scores = np.asarray([
            [0.1, 0.8],
            [0.2, 0.7],
            [0.8, 0.9],
            [0.9, 0.6],
        ], dtype=np.float32)
        result = diagnosis_metrics(targets, scores, ["A", "B"])
        self.assertEqual(result["n_evaluable_classes"], 1)
        self.assertAlmostEqual(result["macro_auroc"], 1.0)
        self.assertIsNone(result["classes"][1]["auroc"])

    def test_long_predictions_preserve_ids_labels_and_metadata(self):
        ids = np.asarray([101, 102])
        logits = {
            group: np.zeros((2, len(names)), dtype=np.float32)
            for group, names in LABEL_GROUPS.items()
        }
        targets = {
            group: np.zeros((2, len(names)), dtype=np.float32)
            for group, names in LABEL_GROUPS.items()
        }
        metadata = pd.DataFrame({
            "age": [50, 60],
            "sex": [0, 1],
            "patient_id": [1, 2],
            "static_noise": [np.nan, "I"],
        }, index=ids)
        frame = long_prediction_frame(
            {"ecg_id": ids, "logits": logits, "targets": targets},
            metadata,
            split="val",
            lead_set="2-lead",
        )
        expected_rows = len(ids) * sum(len(names) for names in LABEL_GROUPS.values())
        self.assertEqual(len(frame), expected_rows)
        self.assertEqual(set(frame.ecg_id), {101, 102})
        self.assertEqual(set(frame.split), {"val"})
        self.assertEqual(set(frame.lead_set), {"2-lead"})
        self.assertIn("age", frame.columns)
        self.assertIn("static_noise", frame.columns)

    def test_checkpoint_normalization_and_label_order_are_validated(self):
        schema = {group: list(names) for group, names in LABEL_GROUPS.items()}
        checkpoint = {
            "label_schema": schema,
            "norm": {"mean": [0.0] * 12, "std": [1.0] * 12},
        }
        self.assertEqual(validate_label_schema(checkpoint), "checkpoint")
        norm, source = normalization_from_checkpoint(
            checkpoint, pd.DataFrame(), fallback_max_load=1,
        )
        self.assertEqual(source, "checkpoint")
        self.assertEqual(norm["mean"].shape, (12,))

        bad = {"label_schema": {**schema, "rhythm": list(reversed(schema["rhythm"]))}}
        with self.assertRaisesRegex(ValueError, "label ordering"):
            validate_label_schema(bad)

    def test_sigmoid_is_finite_for_extreme_logits(self):
        result = sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result[1], 0.5)


class PTBXLSplitIntegrityTest(unittest.TestCase):
    @unittest.skipUnless(
        Path(DATA_ROOT, "ptbxl_database.csv").exists(),
        "local PTB-XL metadata is not available",
    )
    def test_official_split_counts_and_patient_isolation(self):
        metadata = load_metadata()
        counts = metadata.fold_role.value_counts().to_dict()
        self.assertEqual(
            counts,
            {"train": 17418, "test": 2198, "val": 2183},
        )
        patients_crossing_splits = (
            metadata.groupby("patient_id").fold_role.nunique() > 1
        ).sum()
        self.assertEqual(int(patients_crossing_splits), 0)


if __name__ == "__main__":
    unittest.main()
