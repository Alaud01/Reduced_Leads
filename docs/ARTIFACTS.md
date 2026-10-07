# Artifacts: what is tracked, released, or ignored

Every file the project produces falls into exactly one of three classes.

| Class | Where | What belongs there | Rule |
|---|---|---|---|
| **Curated evidence** (tracked) | [`results/`](../results/README.md), [`examples/`](../examples/README.md) | Reports, aggregate summary tables, figures, analysis plans, verification files, report-building scripts, and training logs/manifests/curves for published checkpoints; synthetic demonstration exports | Small, aggregate, no per-record or per-patient rows, no local absolute paths. Copies are byte-identical to the originals and listed in `results/SHA256SUMS`. |
| **Release assets** (downloaded) | [GitHub Releases](https://github.com/Alaud01/Reduced_Leads/releases) | Model weights (`best.pt`) and complete per-record bundles: evaluation exports, fitted policies, audit decisions, logs | Frozen originals, never edited. Each release has `SHA256SUMS` and `release-manifest.json`. |
| **Disposable outputs** (ignored) | `checkpoints/`, `artifacts/`, `runpod-results/`, `figures/`, `release/` | Anything a command writes locally: new training runs, evaluation exports, policies, audits, plots | Regenerable; never committed. See [`.gitignore`](../.gitignore). |

Protocols in `protocol/` are inputs rather than outputs: they are tracked and
frozen once used (a changed protocol gets a new file name).

## Promoting a new result

1. Run the workflow so outputs land in `artifacts/` and `checkpoints/`.
2. Copy only the report, aggregate tables, figure and analysis plan into a new
   `results/<name>/` directory, unchanged. Leave out files that contain
   per-record rows or absolute paths.
3. Regenerate `results/SHA256SUMS`:
   `cd results && find . -type f ! -name README.md ! -name SHA256SUMS | sort | sed 's|^\./||' | xargs shasum -a 256 > SHA256SUMS`.
4. Add the full directories and weights to a new release with
   `scripts/build_release_bundles.py` (edit its asset lists for the new run).

## Release `results-s42-v1`

| Asset | Size | Contents | Extracts to |
|---|---:|---|---|
| `fixed12-s42.best.pt` | 39 MB | Fixed 12-lead model (also the 12-lead rescue stage) | `runpod-results/checkpoints/fixed12-s42/best.pt` |
| `fixed2-s42.best.pt` | 39 MB | Fixed I+II model | `runpod-results/checkpoints/fixed2-s42/best.pt` |
| `random2-s42.best.pt` | 39 MB | Random-lead model selected on I+II | `runpod-results/checkpoints/random2-s42/best.pt` |
| `historical-random12-s42.best.pt` | 117 MB | Historical random-lead model (selected on fold 9) | `checkpoints/best.pt` |
| `runpod-s42-comparison-full.tar.gz` | 49 MB | Seed-42 comparison: per-record exports, policies, audits, logs | `artifacts/comparison/runpod-s42/` |
| `runpod-s42-relaxed-risk-full.tar.gz` | 99 MB | Relaxed-risk sensitivity: policies, audits, logs | `artifacts/comparison/runpod-s42-relaxed-risk/` |
| `historical-model-full.tar.gz` | 105 MB | Historical-model validation/test exports and the final AFIB, all-label (`all-labels-test-v3`), use-contract v1 and legacy patient-level (`patient-v2-final-audit`) policies and audits | `artifacts/evaluation/`, `artifacts/selective*/`, `artifacts/audit*/` |
| `legacy-smoke-exports.tar.gz` | 11 KB | Original smoke exports formerly tracked in Git | `artifacts/evaluation/` |

SHA-256 of the weights (also recorded in
`results/runpod-s42/training_verification.json`):

```text
8d030e41ba046df9416a1da7c60a31a259bcda61c2fa800fe12e1b48b30d8db2  fixed12-s42.best.pt
2dd7620d2d96f251f3fc64797c0abc3ff9cbaf78d46022c0f58f01f1c87c58ee  fixed2-s42.best.pt
095a424958eb6a6a212518fe0e7ded7aa6585b291773b12513813079f0b47ff0  random2-s42.best.pt
b7fb8247d03d15fbc73ee08fee1923467f0605832d4b4854615e7d38797d857a  historical-random12-s42.best.pt
```

### Downloading

From the repository root, with the [GitHub CLI](https://cli.github.com/):

```bash
mkdir -p release/results-s42-v1
gh release download results-s42-v1 --repo Alaud01/Reduced_Leads --dir release/results-s42-v1
(cd release/results-s42-v1 && shasum -a 256 -c SHA256SUMS)

# Weights, placed where the documented commands expect them
for run in fixed12-s42 fixed2-s42 random2-s42; do
  mkdir -p runpod-results/checkpoints/$run
  cp release/results-s42-v1/$run.best.pt runpod-results/checkpoints/$run/best.pt
done

# Full bundles restore their original artifacts/ paths
tar -xzf release/results-s42-v1/runpod-s42-comparison-full.tar.gz
```

Without the CLI, download the same files from the release page in a browser.

### Data and privacy notes

All evaluation outputs derive from PTB-XL 1.0.3, which is distributed under
CC BY 4.0 with pseudonymized patient identifiers. The per-record bundles keep
those identifiers, ages, sexes and noise flags because the patient-level
analyses depend on them. When you use or redistribute them, keep the PTB-XL
attribution in the [README](../README.md#data-attribution): credit the PTB-XL
authors, link the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/),
and note that the files are derived from PTB-XL. Logs and manifests
inside the bundles are unedited and may record absolute paths from the
machines that produced them.

### Rebuilding the release

The assets are produced from a checkout that still holds the original local
outputs:

```bash
python scripts/build_release_bundles.py --source-root /path/to/checkout
gh release create results-s42-v1 release/results-s42-v1/* \
  --title "Seed-42 weights and evidence bundles" \
  --notes-file release/results-s42-v1/RELEASE_NOTES.md
```

`legacy-smoke-exports.tar.gz` was built with `git archive` from the last
commit that tracked those files, so it is identical to the former Git contents.
That commit belongs to the pre-publication history, which is archived
privately, so the script skips this asset when run from the public repository.
