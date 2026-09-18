# Reduced Leads — Lead-Aware Selective Prediction for ECG Diagnosis

A research framework for answering a clinical deployment question that average
accuracy cannot: **for this recording and this diagnosis, is reduced-lead ECG
interpretation reliable enough to act on, or should the system abstain?**

The core idea is a single **lead-aware transformer backbone** trained once on
12-lead PTB-XL waveforms with random lead-dropping augmentation. The same
weights are then evaluated under fixed reduced-lead subsets (12 → 1 lead), and a
frozen-checkpoint evaluation pipeline exports per-diagnosis predictions plus
patient, demographic, and signal-quality metadata as the evidence base for
calibration, selective prediction, and later conformal risk control.

See `PROJECT.md` for the research scope and `DATASET.md` for dataset details.

## Repository layout

```
├── PROJECT.md            Research scope and methods
├── DATASET.md            PTB-XL provenance, splits, label taxonomy, caveats
├── ptb-xl-...-1.0.3/     PTB-XL 1.0.3 release (gitignored, ~1 GB; see DATASET.md)
├── explore_ptbxl.py      Standalone dataset exploration script
├── requirements.txt      Python dependencies
├── checkpoints/          Training outputs (gitignored): best.pt, last.pt, logs
├── figures/              Generated figures (gitignored)
├── artifacts/            Evaluation exports (gitignored)
├── src/
│   ├── config.py         All constants: leads, lead subsets, labels, model/train cfg
│   ├── labels.py         Metadata loading, multi-hot label frame, class weights, splits
│   ├── data.py           WFDB loading, per-lead normalization, lead-dropping dataset
│   ├── model.py          LeadAwareTransformer (lead + time transformer encoder)
│   ├── losses.py         Multi-task BCE loss with inverse-frequency class weights
│   ├── train.py          Training entrypoint with manifests, JSONL logs, plots
│   ├── evaluate.py       Frozen-checkpoint prediction export
│   ├── selective.py      Calibration, risk control, and sequential triage policy
│   ├── statistics.py     Patient-aware audit metrics and clustered bootstrap
│   └── audit_policy.py   Fit-free locked-test policy audit
├── protocol/             Versioned, machine-readable evaluation protocols
└── tests/                Unit and split-integrity tests
```

## Setup

Python 3.10+ (developed with the repo's `.venv`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the PTB-XL 1.0.3 release from
[PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and unzip it in the repo
root so the path `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv`
exists. Quick check that everything resolves:

```bash
python -m unittest discover -s tests -q
```

## Data flow: from CSV to predictions

### 1. Metadata and labels — `src/labels.py`

- `load_metadata()` reads `ptbxl_database.csv`, maps the official strat folds to
  roles (`fold_role`: folds 1–8 → `train`, 9 → `val`, 10 → `test`) and adds a
  `filename_lr` column for the 100 Hz waveforms.
- `build_label_frame()` parses the SCP-ECG code lists into multi-hot targets for
  three heads: **superclass** (5: NORM/MI/STTC/CD/HYP), **subcode** (44), and
  **rhythm** (12). Class order comes from `src/config.py` and is treated as a
  frozen contract everywhere downstream.
- `class_weights()` computes inverse-frequency weights (tau=0.5) used by the
  loss; `split_indices()` returns index arrays for the three roles.

### 2. Signals and lead handling — `src/data.py`

- `load_signal()` reads the 1000×12 (10 s @ 100 Hz) waveform via `wfdb` and
  pads/truncates to a fixed length. NaN values in the raw data are **not**
  imputed (see `DATASET.md` caveats).
- `compute_train_norm()` z-scores per lead using the first N train records
  (2000 full runs, 200 in `--quick`); this is an approximation, and checkpoints
  persist the exact arrays used.
- `PTBXLDataset` applies per-item **random lead dropping** during training
  (0–10 leads dropped, ≥2 always kept). Dropped leads are zeroed and flagged in
  `lead_mask`, which routes them to a learned missing-lead token in the model.
  At evaluation, a fixed `keep_leads` list is used instead (no randomness).

### 3. Model — `src/model.py`

`LeadAwareTransformer`: patch-tokenizes each lead (50-sample = 0.5 s patches →
20 tokens/lead), adds per-lead embeddings (plus one learned "missing" token for
dropped leads), runs 6 lead-attention and 6 time-attention transformer blocks,
pools across leads and time, and outputs four heads:

| Head | Output | Purpose |
|---|---|---|
| `cls_super` | (B, 5) | Superclass diagnosis |
| `cls_sub` | (B, 44) | Diagnostic subcodes |
| `aux_rhythm` | (B, 12) | Rhythm labels |
| `aux_lead_presence` | (B, 12) | Auxiliary: predict which leads were present |

Gradient checkpointing keeps training within 24 GB Apple Silicon MPS memory.

### 4. Training — `src/train.py`

```bash
python -m src.train                  # full run (50 epochs)
python -m src.train --quick          # smoke test (1 epoch, 256 records)
python -m src.train --device cpu     # force CPU
```

Per run it:

1. seeds everything (`--seed`, default 42) for reproducible init, shuffling, and
   per-worker augmentation RNGs;
2. writes `checkpoints/train_manifest_<run_id>.json` (config, normalization
   stats and provenance, label schema, versions);
3. trains with the multi-task loss under lead-dropping augmentation, logging
   batch metrics to `train_log.jsonl` (entries tagged with `run_id`);
4. evaluates 12-lead validation every epoch, and all canonical subsets on the
   first, last, and every fifth epoch (`--eval-full-every` controls the cadence);
5. tracks the best checkpoint by **12-lead superclass macro-AUC** and saves
   `best.pt` / `last.pt`, both embedding `model_cfg`, `train_cfg`, normalization
   arrays, and `label_schema`;
6. renders `training_curves.png` from the JSONL log for the current run.

Automatic mixed precision (AMP) uses FP16 for suitable GPU operations while
keeping numerically sensitive operations in FP32, reducing memory use and often
improving speed. It is enabled by default on MPS/CUDA; `--no-amp` disables it for
both training and validation. CPU runs use FP32. Gradient scaling helps preserve
small gradients and skips a weight update when gradients are non-finite. The
learning-rate schedule advances only after a successful update. Logged `step`
still counts processed batches; `optimizer_updated` identifies whether the
logged batch updated weights. Use `--grad-ckpt` to enable gradient checkpointing
when additional memory savings are needed.

## Evaluation — `src/evaluate.py`

The frozen-checkpoint evaluator is the repo's main CLI. It **only exports
evidence**: per-diagnosis targets, logits, and probabilities plus metadata. It
does not fit thresholds or calibration models — those come later and must be
built on these artifacts.

### Split policy

Validation fold 9 is the default. Use it for model selection, calibration, and
policy development:

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt
```

Fold 10 (test) is only read when `test` is explicitly requested, so the final
test set stays untouched until the calibration and triage policy are frozen:

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --splits val test
```

Smoke test with limits:

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt \
  --splits val --lead-sets 2-lead 1-lead-II --max-n 16 --norm-max-load 32
```

Other flags: `--device auto|cpu|mps|cuda`, `--batch-size`, `--num-workers`,
`--run-name`, `--out-dir`.

### Safety checks

- **Label schema guard**: refuses to run if the checkpoint's stored label order
  differs from the current `src/config.py`, preventing mislabelled exports.
  Checkpoints without a schema fall back to runtime config, recorded as
  `label_schema_source: runtime-config-fallback` in the manifest.
- **Normalization provenance**: new checkpoints carry the exact per-lead
  mean/std used at training time. Older checkpoints trigger a recomputation from
  train records (`--norm-max-load`), flagged in the manifest as
  `recomputed-from-train-first-<n>`.
- **Non-finite logit guard**: prediction aborts if any logit is non-finite,
  naming the affected ECG IDs.

### Lead subsets

Evaluation runs over the canonical named sets in `src/config.py` (`LEAD_SUBSETS`):
`12-lead`, `6-lead-limb`, `4-lead`, `3-lead`, `2-lead`, `1-lead-I`, `1-lead-II`.
These names are stable and are persisted in prediction artifacts; select them
with `--lead-sets`.

### Artifacts

Each invocation creates a unique directory under `artifacts/evaluation/`
(`<timestamp>-<checkpoint>-<hash8>/`, or `--run-name`) containing:

- `manifest.json` — checkpoint SHA-256 and training run linkage, split-access
  record (`test_evaluated`), model and label schema, normalization provenance,
  lead sets, runtime versions, artifact inventory, and a `status` field set to
  `complete` only after all outputs are written (all writes are atomic
  tmp-then-rename).
- `records__<split>.csv.gz` — one row per ECG with patient, demographic, and
  available signal-quality metadata (`patient_id`, `age`, `sex`, plus noise
  flags like `static_noise`, `burst_noise`).
- `predictions/<split>__<lead-set>.csv.gz` — one row per ECG × diagnosis with
  `target`, `logit`, `probability`, lead set, and the same metadata columns.
  Diagnosis order follows `src.config`.
- `metrics.json` — diagnosis-level AUROC / average precision (plus prevalence
  and positivity counts) and macro averages for every evaluated split, lead
  set, and output family. Classes without both positive and negative examples
  are reported but excluded from the macro means.

## Selective prediction — `src/selective.py`

The selective-prediction CLI consumes **validation-fold files only**. It starts
with AFIB by default but supports any exported superclass, subcode, or rhythm:

```bash
python -m src.selective \
  --evaluation-dir artifacts/evaluation/<complete-run> \
  --diagnosis-group rhythm \
  --diagnosis AFIB
```

If `--evaluation-dir` is omitted, the command finds the newest complete
validation export containing all requested lead sets.

For every diagnosis × reduced-lead set, it:

1. estimates development performance with patient-grouped outer cross-fitting;
2. splits each development partition again by patient, fitting a monotone affine
   (Platt) calibrator on one half; the other half is randomly split by patient
   into independent reduced-stage and rescue-stage risk partitions;
3. controls trusted-positive and trusted-negative errors separately using
   one-sided exact binomial bounds, simultaneously corrected over a fixed
   threshold grid and the four reduced/12-lead decision families. The rescue
   threshold uses only records referred by the frozen reduced-stage policy;
4. applies the sequential policy: `trust_reduced` → `trust_12_lead` after
   acquisition → `expert_review` when neither stage meets its bound;
5. fits and serializes a final policy on fold 9 only, ready for one-time fold-10
   evaluation without refitting.

Defaults are deliberately asymmetric: `--max-positive-risk 0.10` and
`--max-negative-risk 0.02`. These are research defaults, not clinical operating
requirements; they must be prespecified for the intended use case. An infeasible
direction gets no threshold and therefore abstains—risk targets are never silently
relaxed.

Outputs under `artifacts/selective/<run>/`:

- `policy.json` — frozen calibrators and diagnosis/lead-specific thresholds,
  including the statistical contract and all cross-fit fold policies;
- `decisions.csv.gz` — cross-fitted probabilities, predictions, all three actions,
  correctness, demographics, and signal-quality metadata;
- `risk_coverage.csv.gz` — empirical risks, exact upper bounds, and coverage over
  the complete fixed threshold grid;
- `summary.json` — action rates, selective error, sensitivity/specificity,
  raw-versus-calibrated metrics, and patient-bootstrap 95% intervals;
- `calibration.png`, `risk_coverage.png`, and `actions.png`.

The exact binomial bounds assume independent records. Patient-grouped partitions
and cluster bootstrap intervals do not remove the known repeat-ECG limitation.
This is not yet the project’s final conformal method or a clinical guarantee under
distribution shift. External validation and explicit conformal sensitivity
analyses remain on the roadmap.

## Locked test audit — `src/audit_policy.py`

After model selection and policy fitting are complete, export fold 10 and apply
the frozen policy without refitting:

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --splits val test
python -m src.audit_policy \
  --evaluation-dir artifacts/evaluation/<val-and-test-export> \
  --policy artifacts/selective/<run>/policy.json \
  --protocol protocol/afib_v1.json
```

The audit refuses incomplete or `--max-n`-limited test exports and verifies the
checkpoint hash, label schema, target, lead sets, policy state, and risk limits
before reading test predictions. It writes fit-free test decisions, discrimination
and calibration metrics, conservative patient-level summaries, patient-cluster
bootstrap intervals, subgroup tables, paired reduced-vs-12-lead contrasts,
descriptive risk-coverage curves, and a hashed artifact manifest under
`artifacts/audit/`.

The v1 AFIB protocol is explicitly exploratory research. A result within its
risk limits is not a new finite-sample guarantee, a clinical diagnosis claim, or
authorization to retune against fold 10. See `AFIB_EVALUATION_PROTOCOL.md` and
`REFERENCES.md` for the evidence ladder and literature rationale.

### Post-hoc positive-call sensitivity policy

The original locked policy found no positive threshold satisfying its 10%
simultaneous upper-risk constraint. A separate, non-confirmatory sensitivity
policy keeps the 2% negative-risk constraint and 25-case minimum, but relaxes
the positive-risk upper-bound constraint to 40%. This value is deliberately
descriptive: it is the smallest round validation-only setting that makes the
2-lead pathway estimable after 12-lead confirmation, not a clinically acceptable
error target.

```bash
python -m src.selective \
  --evaluation-dir artifacts/evaluation/<validation-export> \
  --run-name afib-positive-exploratory-risk40-v2 \
  --max-positive-risk 0.40 \
  --max-negative-risk 0.02 \
  --confidence 0.95 \
  --min-trusted 25 \
  --bootstrap 2000

python -m src.audit_policy \
  --evaluation-dir artifacts/evaluation/<test-export> \
  --policy artifacts/selective/afib-positive-exploratory-risk40-v2/policy.json \
  --protocol protocol/afib_positive_exploratory_v1.json \
  --run-name afib-positive-exploratory-test-v2
```

Because this policy was proposed after reviewing the original test audit, its
test results are hypothesis-generating even though its thresholds are fitted
using validation data only. Any confirmation requires new untouched external
data.

## All-label research audit — `src/audit_all_labels.py`

The all-label workflow expands the same validation-fit/test-audit boundary to
every superclass, diagnostic subcode, and rhythm label in the evaluation
manifest. Every label receives discrimination, calibration, prevalence,
risk-coverage, and reduced-vs-12-lead comparisons. Selective policies are fitted
only for labels with at least 25 positive and 25 negative validation patients;
sparser labels remain explicitly marked `insufficient_data` and receive model
metrics only.

```bash
python -m src.audit_all_labels fit \
  --evaluation-dir artifacts/evaluation/<validation-export> \
  --protocol protocol/all_labels_v1.json \
  --run-name all-labels-baseline-v2

python -m src.audit_all_labels audit \
  --evaluation-dir artifacts/evaluation/<test-export> \
  --policy artifacts/selective-all/all-labels-baseline-v2/policy.json \
  --protocol protocol/all_labels_v1.json \
  --run-name all-labels-test-v3
```

The shared 10% positive and 2% negative risk limits are comparison anchors, not
diagnosis-specific clinical choices. In particular, the meaning and harm of a
positive decision differs for `NORM` versus a disease label. The test fold has
also already been inspected during AFIB development, so this complete-label
audit is post-hoc; new external data is required for confirmation.
See `ALL_LABEL_EVALUATION_PROTOCOL.md` for the evidence tiers, action semantics,
uncertainty rules, and advancement criteria.

## Testing

```bash
python -m unittest discover -s tests -q
```

Covers evaluation helpers, calibration, threshold risk bounds, infeasible-policy
abstention, all three routing actions, cross-fit completeness, calibration split
patient isolation, and—when local PTB-XL metadata is present—the official split
counts and patient isolation across folds.


## Use-contract fitting and audit

The shared all-label engine now fits every exported label plus the frozen
`max_logit` AFIB-or-AFLT composite. It applies the ischemia, conduction, and
assert-normal limits from `protocol/use_contract_v1.json`. Labels outside these
groups retain the descriptive 10%/2% anchors. Assert-normal negative automation
is disabled because the contract defines no negative-call limit for those labels.
Every target must meet both the positive- and negative-patient eligibility floors.
Single-class and sparse targets still receive model-performance outputs.

```bash
python -m src.use_contract fit \
  --evaluation-dir artifacts/evaluation/<validation-only-export> \
  --run-name contract-routed-v2

python -m src.use_contract audit \
  --evaluation-dir artifacts/evaluation/<test-export> \
  --policy artifacts/selective-all/contract-routed-v2/policy.json \
  --run-name contract-routed-v2-audit
```

Use `--lead-sets 2-lead` during fitting for a primary-lead-only run. Audit reads
the frozen lead selection. Both commands also work through `src.audit_all_labels`
with `--protocol protocol/use_contract_v1.json` (all leads by default).

New bundles record execution method
`fixed-grid-clopper-pearson-bonferroni-routed-v2`, the normalized execution
contract, per-target limits, and hashed artifacts. Audit checks the saved policy
hash, validation-only source manifest, checkpoint, source input hashes, schema,
lead definitions, eligibility, and target/lead enumeration before reading test
predictions. Keep the original validation export available for these checks.
All prediction populations must match the exported `records__<split>.csv.gz`.
Normality coverage is reported separately from disease coverage; risk checks
use each target's limits. Zero-bootstrap runs explicitly report no intervals.

Historical protocol JSON files and existing fitted artifacts are unchanged.
Older all-label/composite bundles must be refitted on validation into a new
output directory to use the corrected method; they are not silently upgraded.
The additional independent risk partition can reduce coverage. Existing test
fold reuse remains exploratory, and the correction does not create an untouched
confirmatory cohort or resolve the repeat-ECG limitation of record-level bounds.
