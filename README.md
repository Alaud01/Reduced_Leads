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
│   └── evaluate.py       Frozen-checkpoint prediction export (this repo's CLI)
└── tests/                Unit tests for evaluation helpers and split integrity
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
4. evaluates the val split at every canonical lead subset after each epoch;
5. tracks the best checkpoint by **12-lead superclass macro-AUC** and saves
   `best.pt` / `last.pt`, both embedding `model_cfg`, `train_cfg`, normalization
   arrays, and `label_schema`;
6. renders `training_curves.png` from the JSONL log for the current run.

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

## Downstream use (roadmap)

The prediction exports are designed as the input to:

1. **Calibration** — per-diagnosis reliability curves from exported
   probabilities (e.g. on val fold 9).
2. **Selective prediction / triage** — a three-action policy per diagnosis:
   trust the reduced-lead output, obtain a full 12-lead, or refer for expert
   review, expressed as accuracy–coverage curves.
3. **Conformal risk control** — distribution-free guarantees built on the
   exported per-diagnosis scores.
4. **Subgroup and noise analysis** — reliability stratified by age/sex
   (`patient_id`, `age`, `sex` columns) and by signal-quality flags.

## Testing

```bash
python -m unittest discover -s tests -q
```

Covers evaluation helpers (metrics, schema/normalization validation, sigmoid
stability) and — when local PTB-XL metadata is present — the official split
counts and patient isolation across folds.