"""The committed synthetic example must match its generator and run end to end."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from src.use_contract import DEFAULT_PROTOCOL, audit_contract_policies, fit_contract_policies

EXAMPLE = Path("examples/synthetic-evaluation")


def _assert_close(test: unittest.TestCase, expected, actual, where: str) -> None:
    """Compare JSON values, allowing tiny platform-dependent float differences."""
    if isinstance(expected, dict):
        test.assertEqual(expected.keys(), actual.keys(), where)
        for key in expected:
            _assert_close(test, expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        test.assertEqual(len(expected), len(actual), where)
        for index, (a, b) in enumerate(zip(expected, actual)):
            _assert_close(test, a, b, f"{where}[{index}]")
    elif isinstance(expected, float):
        test.assertAlmostEqual(expected, actual, places=6, msg=where)
    else:
        test.assertEqual(expected, actual, where)


class SyntheticExampleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_committed_example_matches_generator(self):
        generated = self.root / "generated"
        subprocess.run(
            [sys.executable, "scripts/make_synthetic_evaluation.py", "--out-dir", str(generated)],
            check=True, capture_output=True,
        )
        committed = sorted(p.relative_to(EXAMPLE) for p in EXAMPLE.rglob("*") if p.is_file() and p.name != "README.md")
        produced = sorted(p.relative_to(generated) for p in generated.rglob("*") if p.is_file())
        self.assertEqual(committed, produced)
        for relative in committed:
            if relative.suffix == ".gz":
                pd.testing.assert_frame_equal(
                    pd.read_csv(EXAMPLE / relative), pd.read_csv(generated / relative),
                    check_exact=False, atol=1e-6, obj=str(relative),
                )
            else:
                _assert_close(self, json.loads((EXAMPLE / relative).read_text()),
                              json.loads((generated / relative).read_text()), str(relative))

    def test_example_contains_no_absolute_paths(self):
        for manifest in EXAMPLE.glob("*/manifest.json"):
            payload = json.loads(manifest.read_text())
            self.assertTrue(payload["synthetic"])
            self.assertNotIn("/Users/", manifest.read_text())

    def test_default_protocol_fits_and_audits_example(self):
        config = json.loads(DEFAULT_PROTOCOL.read_text())
        config["uncertainty"]["replicates"] = 0
        protocol = self.root / "protocol.json"
        protocol.write_text(json.dumps(config))
        with contextlib.redirect_stdout(io.StringIO()):
            run = fit_contract_policies(EXAMPLE / "val", protocol, self.root / "fit", "demo", ["2-lead"])
            audit = audit_contract_policies(EXAMPLE / "test", run / "policy.json", protocol, self.root / "audit", "demo")
        self.assertEqual(json.loads((audit / "manifest.json").read_text())["status"], "complete")
        self.assertTrue((audit / "policy_metrics.csv.gz").is_file())


if __name__ == "__main__":
    unittest.main()
