# Development

## Environments

| File | Purpose |
|---|---|
| [`requirements.txt`](../requirements.txt) | Compatible dependency ranges. The lower bounds were verified on Python 3.10 by installing the lowest allowed versions (`uv pip install --resolution lowest-direct`) and running the test suite. |
| [`requirements-lock.txt`](../requirements-lock.txt) | Exact pins for the CI environment: Python 3.12, Linux x86_64, CPU-only PyTorch. |
| [`pyproject.toml`](../pyproject.toml) | Project metadata, `requires-python >= 3.10`, the same dependency ranges, and ruff settings. Installing the package is optional; commands run from the repository root. |

Python 3.10 is the minimum because the code relies on PEP 604 union types at
runtime. PyTorch 2.4 is the minimum because training uses
`torch.amp.GradScaler(device)` and checkpoints are loaded with
`weights_only=True`.

Tested configurations:

| Python | Platform | Dependencies | Result |
|---|---|---|---|
| 3.10 | macOS arm64 | lowest allowed versions (torch 2.4.0, numpy 1.24, pandas 2.0) | tests pass |
| 3.10 | macOS arm64 | newest compatible versions | tests pass |
| 3.12 | macOS arm64 | `requirements-lock.txt` pins (torch 2.14.1) | tests pass |
| 3.14 | macOS arm64 | local development environment (torch 2.13) | tests pass |
| 3.12 / 3.10 | Linux x86_64 | CI: lock file / ranges, CPU torch | see GitHub Actions |

The published seed-42 models were trained on Runpod with
`torch 2.4.1+cu124`; the historical model on Apple Silicon MPS. The exact
versions are in each training manifest under `results/training/`.

Bitwise reproducibility across hardware or library versions is not promised;
see the resume contract in [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md#resume-contract).

## Tests and lint

```bash
python -m unittest discover -s tests -q
ruff check src tests scripts
```

The suite covers evaluation helpers, calibration, threshold risk bounds,
infeasible-policy abstention, the three routing actions, cross-fit
completeness, calibration-split patient isolation, the patient-level v2 fit
and audit (including rejection of tampered or test-exposed inputs before test
predictions are read), training resume, and the synthetic example. When the
local PTB-XL metadata is present it also checks the official split counts and
patient isolation across folds; without the dataset that one test is skipped.

[`.github/workflows/tests.yml`](../.github/workflows/tests.yml) runs ruff and
the unit tests on every push to `master` and on pull requests, using the
locked Python 3.12 environment and Python 3.10 with dependency ranges.

### Regenerating the lock file

```bash
uv pip compile requirements.txt --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --no-emit-index-url \
  -o requirements-lock.txt
```

Then restore the explanatory header at the top of the file.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/explore_ptbxl.py` | Dataset exploration figures and summary (writes to `figures/`) |
| `scripts/make_synthetic_evaluation.py` | Regenerates `examples/synthetic-evaluation/` |
| `scripts/build_release_bundles.py` | Builds release weights and bundles with checksums ([`ARTIFACTS.md`](ARTIFACTS.md)) |
| `docs/visuals/assets/*.py` | Re-render the poster figures |

## Code organization notes

Shared hashing, atomic-write and logit helpers live in `src/io_utils.py`, so
policy and audit modules do not import PyTorch. `src.evaluate` re-exports them
for backwards compatibility.

Possible follow-up refactors (not required for correctness):

- `src/audit_all_labels.py` (~1,150 lines) mixes protocol loading, fitting,
  auditing, verification and Markdown reporting; reporting and input
  verification could become separate modules.
- `src/selective.py` (~1,030 lines) combines the statistical engine with its
  CLI, plotting and output writing.
- `src/train.py` (~900 lines) combines the training loop, validation,
  plotting and CLI.
- Policy and audit manifests still record absolute input paths, because the
  verifiers compare those stored strings when checking provenance. Making them
  repository-relative needs a coordinated change to writers and verifiers.
