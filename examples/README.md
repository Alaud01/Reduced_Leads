# Examples

## `synthetic-evaluation/`

A synthetic validation (`val/`) and test (`test/`) export in exactly the format
written by `python -m src.evaluate`: `manifest.json`, `metrics.json`,
`records__<split>.csv.gz` and `predictions/<split>__<lead-set>.csv.gz` for the
`12-lead` and `2-lead` inputs.

All identifiers, demographics, targets and logits are random. Nothing comes
from PTB-XL or from a trained model, so the files can be used to try the
policy-fitting and audit commands without downloading data or weights:

```bash
python -m src.use_contract fit \
  --evaluation-dir examples/synthetic-evaluation/val \
  --lead-sets 2-lead --run-name synthetic-demo

python -m src.use_contract audit \
  --evaluation-dir examples/synthetic-evaluation/test \
  --policy artifacts/selective-all/synthetic-demo/policy.json \
  --run-name synthetic-demo-audit
```

Outputs go to the gitignored `artifacts/` directory. With only 450 synthetic
patients per split, every route abstains (`no_automated_decisions`): a 2%
negative-risk bound needs at least 446 error-free patients in a single
validation branch. That is the intended behavior; `feasibility.csv.gz` in the
fitted policy shows the sample sizes involved.

The example uses a compact label schema (`NORM`, `MI`, `AFIB`, `AFLT`, `SR`)
rather than all 61 labels. It is regenerated deterministically by
`python scripts/make_synthetic_evaluation.py`, and
`tests/test_synthetic_example.py` checks that the committed files match the
generator and run through fit and audit.

Frozen originals of the earlier smoke exports, which contained real PTB-XL
rows and local paths, are preserved in the `legacy-smoke-exports.tar.gz`
release asset.
