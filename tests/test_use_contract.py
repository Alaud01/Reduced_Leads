import unittest

import pandas as pd

from src.use_contract import (
    COMPOSITE_DIAGNOSIS,
    build_composite_frame,
    load_use_contract,
    resolve_limits,
)
from pathlib import Path


def _frame() -> pd.DataFrame:
    rows = []
    for ecg, afib, aflt, la, lf in [
        (1, 1, 0, 2.0, -1.0),
        (2, 0, 1, -2.0, 1.5),
        (3, 0, 0, -1.0, -2.0),
        (4, 1, 1, 0.5, 0.8),
    ]:
        for diag, target, logit in (("AFIB", afib, la), ("AFLT", aflt, lf)):
            rows.append({
                "ecg_id": ecg, "split": "val", "lead_set": "2-lead",
                "diagnosis_group": "rhythm", "diagnosis": diag,
                "target": target, "logit": logit, "patient_id": 100 + ecg,
            })
    return pd.DataFrame(rows)


class TestUseContract(unittest.TestCase):
    def test_composite_or_and_max(self):
        composite = build_composite_frame(_frame())
        by_ecg = dict(zip(composite.ecg_id, zip(composite.target, composite.logit)))
        self.assertEqual(by_ecg[1], (1, 2.0))
        self.assertEqual(by_ecg[2], (1, 1.5))
        self.assertEqual(by_ecg[3], (0, -1.0))
        self.assertEqual(by_ecg[4], (1, 0.8))
        self.assertTrue((composite.diagnosis == COMPOSITE_DIAGNOSIS).all())

    def test_resolve_limits(self):
        contract = load_use_contract(Path("protocol/use_contract_v1.json"))
        pos, neg = resolve_limits(contract, "rhythm", COMPOSITE_DIAGNOSIS)
        self.assertAlmostEqual(pos, 0.10)
        self.assertAlmostEqual(neg, 0.02)
        pos_mi, neg_mi = resolve_limits(contract, "superclass", "MI")
        self.assertAlmostEqual(pos_mi, 0.05)
        self.assertAlmostEqual(neg_mi, 0.01)
        pos_n, _ = resolve_limits(contract, "rhythm", "SR")
        self.assertAlmostEqual(pos_n, 0.02)
        self.assertIsNone(resolve_limits(contract, "rhythm", "SR")[1])
        # Bare diagnosis names must not inherit another group's contract.
        self.assertEqual(resolve_limits(contract, "subcode", "MI"), (.1, .02))

    def test_composite_aligns_shuffled_rows(self):
        expected = build_composite_frame(_frame())
        actual = build_composite_frame(_frame().sample(frac=1, random_state=2))
        pd.testing.assert_frame_equal(expected, actual)

    def test_composite_rejects_missing_ecgs_and_metadata_mismatch(self):
        frame = _frame()
        with self.assertRaisesRegex(ValueError, "ECG IDs differ"):
            build_composite_frame(frame.iloc[1:])
        for column, value in (("patient_id", 999), ("split", "test"), ("lead_set", "12-lead")):
            changed = frame.copy()
            changed.loc[1, column] = value
            with self.assertRaisesRegex(ValueError, f"{column} mismatch"):
                build_composite_frame(changed)

    def test_composite_all_negative_is_valid_for_performance_only(self):
        frame = _frame().assign(target=0)
        self.assertTrue((build_composite_frame(frame).target == 0).all())


if __name__ == "__main__":
    unittest.main()
