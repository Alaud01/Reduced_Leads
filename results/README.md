# Curated results

This directory holds the evidence behind [`docs/RESULTS.md`](../docs/RESULTS.md):
human-readable reports, aggregate summary tables, figures, analysis plans and
the scripts that produced them. Everything here is small, aggregate (no
per-record or per-patient rows) and tracked in Git. Model weights and complete
per-record bundles are release downloads; see
[`docs/ARTIFACTS.md`](../docs/ARTIFACTS.md).

Every file is a byte-identical copy of the original output. `SHA256SUMS`
lists their hashes (`shasum -a 256 -c SHA256SUMS` from this directory), and
`runpod-s42/comparison_manifest.json` independently records the hash of each
comparison output at the time it was produced.

| Directory | What it contains | Original location |
|---|---|---|
| [`runpod-s42/`](runpod-s42/report.md) | Seed-42 comparison of fixed I+II, random-lead and fixed 12-lead training against the historical model: report, figure, model/policy/subgroup/CRC/repeat-ECG tables, analysis plan, manifest and scripts. | `artifacts/comparison/runpod-s42/` |
| [`runpod-s42-relaxed-risk/`](runpod-s42-relaxed-risk/report.md) | Post-hoc AFIB-or-AFLT risk-limit sensitivity (2%, 2.5%, 3% negative limits): report, scenario table, analysis plan, sensitivity protocols and verification. | `artifacts/comparison/runpod-s42-relaxed-risk/` |
| [`historical/all-labels-test-v3/`](historical/all-labels-test-v3/report.md) | Historical all-label test audit cited by [`docs/USE_CONTRACT.md`](../docs/USE_CONTRACT.md): aggregate model, policy, calibration, subgroup and paired-contrast tables. | `artifacts/audit-all/all-labels-test-v3/` |
| [`training/`](training/) | Training logs, manifests and learning curves for each published checkpoint. | `runpod-results/checkpoints/<run>/`, `checkpoints/` |

Run names such as `fixed2-s42` are explained in
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

## What is not here

- `pipeline.log`, `run.log`, per-record evaluation exports, fitted policy
  bundles, audit decisions and `manifest.json` files that embed local
  absolute paths. They are preserved unchanged in the release bundles.
- The relaxed-risk `policies/`, `audits/` and `interrupted/` directories.
  The two `neg05-*` protocols are kept for completeness; the 5% scenarios were
  retired before the reported sensitivity analysis (see the report).

## Reproducing the reports

The scripts in each directory were run from the repository root against the
original `artifacts/` layout, so they read and write paths such as
`artifacts/comparison/runpod-s42/...`. To rerun them, download and extract the
matching full bundle from the repository root (see `docs/ARTIFACTS.md`), which
restores that layout, and then run the script named in the report.

## Training records

Each `training/<run>/` directory contains the JSONL training log, the training
manifest (configuration, normalization statistics, label schema and resume
contract) and `training_curves.png`. Manifests are frozen originals: the
Runpod runs record the container path `/workspace/Reduced_Leads/...`, and the
historical run records the local path of the machine that trained it. Those
paths are informational only; new manifests store repository-relative paths.

`historical-random12-s42/` has two manifests created about 30 seconds apart;
only `20260901T221441103670Z-seed42` appears in the log. The log also begins
with 483 untagged entries (epochs 1-20 of an earlier attempt, written before
run IDs were logged). Filter on `run_id` to read the completed 50-epoch run.
