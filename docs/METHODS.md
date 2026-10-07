# Methods and pipeline reference

This is the detailed reference for each stage of the pipeline. The
recommended end-to-end commands are in the [README](../README.md#recommended-workflow-patient-level-v2);
the statistical design is in [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md).

## Fold roles

PTB-XL ships ten patient-disjoint stratified folds. Under the current
patient-level v2 design they are used as follows:

| Folds | Role | Used by |
|---|---|---|
| 1-7 | gradient fitting | `src.train` |
| 8 | checkpoint selection | `src.train` (best epoch by superclass macro-AUROC on the selection lead set) |
| 9 | probability calibration and policy fitting (`val` split) | `src.evaluate --splits val`, then `src.use_contract fit` |
| 10 | one-time locked audit (`test` split) | `src.evaluate --splits test`, then `src.use_contract audit` |

`labels.load_metadata()` labels folds 1-8 `train`, fold 9 `val` and fold 10
`test`. Training then moves fold 8 into its own selection role
(`training_state.apply_research_split`), and normalization statistics and class
weights use folds 1-7 only. Evaluation manifests record
`model_selection_fold` and `calibration_independent`, and the v2 protocol
refuses exports whose checkpoint was selected on fold 9.

The historical model (`historical-random12-s42`) predates this design: it was
trained on folds 1-8 and selected on fold 9, so its fold-9 predictions cannot
provide independent calibration. See [`HISTORICAL_WORKFLOWS.md`](HISTORICAL_WORKFLOWS.md).

## 1. Metadata and labels — `src/labels.py`

- `load_metadata()` reads `ptbxl_database.csv`, assigns the `fold_role` column
  above and adds `filename_lr` for the 100 Hz waveforms.
- `build_label_frame()` parses the SCP-ECG code lists into multi-hot targets for
  three heads: **superclass** (5: NORM/MI/STTC/CD/HYP), **subcode** (44) and
  **rhythm** (12). Class order comes from `src/config.py` and is treated as a
  frozen contract everywhere downstream.
- `class_weights()` computes inverse-frequency weights (tau=0.5) used by the
  loss; `split_indices()` returns index arrays for each role.

## 2. Signals and lead handling — `src/data.py`

- `load_signal()` reads the 1000×12 (10 s @ 100 Hz) waveform via `wfdb` and
  pads/truncates to a fixed length. NaN values in the raw data are **not**
  imputed (see [`DATASET.md`](DATASET.md) caveats).
- `compute_train_norm()` z-scores each lead using the first N training records
  (2000 for full runs, 200 with `--quick`). This is an approximation;
  checkpoints persist the exact arrays used.
- `PTBXLDataset` applies per-item **random lead dropping** during random-lead
  training (0-10 leads dropped, at least 2 kept). Dropped leads are zeroed and
  flagged in `lead_mask`, which routes them to a learned missing-lead token.
  Fixed-lead training and all evaluation use a fixed `keep_leads` list.

## 3. Model — `src/model.py`

`LeadAwareTransformer` patch-tokenizes each lead (50-sample = 0.5 s patches,
20 tokens per lead), adds per-lead embeddings (plus a learned "missing" token
for dropped leads), runs 6 lead-attention and 6 time-attention transformer
blocks, pools across leads and time, and outputs four heads:

| Head | Output | Purpose |
|---|---|---|
| `cls_super` | (B, 5) | Superclass diagnosis |
| `cls_sub` | (B, 44) | Diagnostic subcodes |
| `aux_rhythm` | (B, 12) | Rhythm labels |
| `aux_lead_presence` | (B, 12) | Auxiliary: predict which leads were present |

The backbone has about 9.8M parameters. `--grad-ckpt` enables gradient
checkpointing when memory is tight.

## 4. Training — `src/train.py`

```bash
python -m src.train --quick                      # smoke test (1 epoch, small subset)
python -m src.train --train-lead-set 2-lead --seed 42 --out-dir checkpoints/fixed2-s42
python -m src.train --train-lead-set random --selection-lead 2-lead \
  --lead-presence-weight 0 --seed 42 --out-dir checkpoints/random2-s42
```

Without `--train-lead-set`, training uses random lead dropping with the
auxiliary lead-presence loss and selects on 12 leads. Use `--num-workers 0` on
hosts without multiprocessing/shared-memory support, and `--device cpu|mps|cuda`
to force a device.

Per run, training:

1. seeds everything (`--seed`, default 42) for reproducible initialization,
   shuffling and per-worker augmentation;
2. writes `train_manifest_<run_id>.json` (configuration, normalization
   statistics and provenance, label schema, versions, resume contract);
3. trains the multi-task loss under random lead dropping, or directly on one
   fixed canonical subset selected with `--train-lead-set`; the lead-presence
   loss is disabled for fixed inputs because its target would be constant;
4. evaluates the selection fold every epoch on the selection lead set (all
   subsets periodically for random-lead runs);
5. keeps the best checkpoint by superclass macro-AUROC on the selection lead
   set: `--selection-lead` for random-lead runs, the training subset for
   fixed runs. `best.pt` and the epoch-boundary `last.pt` contain model,
   optimizer, scheduler, AMP scaler, normalization, configuration and label
   schema state;
6. renders `training_curves.png` from the JSONL log.

Automatic mixed precision (FP16 autocast with gradient scaling) is enabled by
default on MPS/CUDA; `--no-amp` disables it for training and validation, and
CPU runs use FP32. The learning-rate schedule advances only after a successful
optimizer update. Logged `step` counts processed batches; `optimizer_updated`
records whether that batch updated weights.

### Resume

Interrupted schema-v3 runs resume at the next epoch without restarting the
learning-rate schedule:

```bash
python -m src.train --resume checkpoints/fixed2-s42/last.pt --device cuda
```

Resume preserves random state and refuses changes to the seed, schedule,
validation IDs/limit, label order, waveform-content fingerprints,
architecture, device or worker count. Copy the complete run directory,
including `last.pt`, `best.pt` and the training manifest. Older checkpoint
schemas remain usable for inference but are refused for faithful resume. See
[`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md#resume-contract).

## 5. Frozen-checkpoint export — `src/evaluate.py`

The evaluator **only exports evidence**: per-diagnosis targets, logits and
probabilities plus metadata. It fits nothing.

```bash
python -m src.evaluate --checkpoint checkpoints/fixed2-s42/best.pt \
  --splits val --lead-sets 2-lead --run-name fixed2-s42-val
```

`--splits` defaults to `val` (fold 9). Fold 10 is read only when `test` is
requested explicitly, after policies are frozen. Other flags: `--lead-sets`,
`--device auto|cpu|mps|cuda`, `--batch-size`, `--num-workers`, `--run-name`,
`--out-dir`, and `--max-n` / `--norm-max-load` for smoke tests.

Export a fixed-lead model only on the lead set it was trained on. The fixed
12-lead model is exported on both `12-lead` (rescue stage) and `2-lead` (the
masked baseline).

### Safety checks

- **Label schema guard**: refuses to run if the checkpoint's stored label order
  differs from `src/config.py`. Checkpoints without a schema fall back to the
  runtime config, recorded as `label_schema_source: runtime-config-fallback`.
- **Normalization provenance**: new checkpoints carry the exact per-lead
  mean/std used in training. Older checkpoints trigger recomputation from
  training records (`--norm-max-load`), flagged as
  `recomputed-from-train-first-<n>`.
- **Non-finite logit guard**: prediction aborts if any logit is non-finite,
  naming the affected ECG IDs.

### Lead subsets

Evaluation uses the canonical named sets in `LEAD_SUBSETS`: `12-lead`,
`6-lead-limb`, `4-lead`, `3-lead`, `2-lead`, `1-lead-I`, `1-lead-II` (see
[`EXPERIMENTS.md`](EXPERIMENTS.md#lead-set-names)).

### Export files

Each invocation creates `artifacts/evaluation/<run-name>/` (or
`<timestamp>-<checkpoint>-<hash8>/`) containing:

- `manifest.json`: checkpoint SHA-256 and training-run linkage, selection
  fold and lead regime, split-access record (`test_evaluated`), model and label
  schema, normalization provenance, lead sets, runtime versions, artifact
  inventory, and `status: complete` once every output is written (all writes
  are atomic tmp-then-rename). Paths are stored relative to the repository.
- `records__<split>.csv.gz`: one row per ECG with patient, demographic and
  available signal-quality metadata (`patient_id`, `age`, `sex`, noise flags).
- `predictions/<split>__<lead-set>.csv.gz`: one row per ECG × diagnosis with
  `target`, `logit`, `probability`, lead set and the same metadata.
- `metrics.json`: per-diagnosis AUROC / average precision with prevalence and
  counts, plus macro averages for every split, lead set and output family.
  Classes without both positives and negatives are reported but excluded from
  macro means.

[`examples/synthetic-evaluation/`](../examples/README.md) contains a synthetic
export in this format.

## 6. Two-stage assembly — `src/assemble_evaluation.py`

The routed policy starts on a reduced lead set and escalates to a 12-lead
rescue model. When those stages come from different checkpoints, combine
their exports:

```bash
python -m src.assemble_evaluation \
  --reduced-dir artifacts/evaluation/fixed2-s42-val \
  --twelve-dir artifacts/evaluation/fixed12-s42-val \
  --lead-set 2-lead --split val \
  --out-dir artifacts/evaluation/fixed2-s42-paired-val
```

Assembly verifies records, patient IDs, targets, schemas, lead definitions and
source hashes. The combined manifest records both checkpoint hashes; its SHA
is a digest of that stage mapping, not of a checkpoint file. Keep the source
exports for later provenance checks.

## 7. Patient-level policy fitting and audit — `src/use_contract.py`

`src.use_contract` fits every exported label plus the frozen `max_logit`
AFIB-or-AFLT composite, using the diagnosis-specific limits of
`protocol/use_contract_v2.json` (the default). Labels outside the contract
groups keep the descriptive 10% positive / 2% negative anchors; assert-normal
labels (NORM, SR) have negative automation disabled.

```bash
python -m src.use_contract fit \
  --evaluation-dir artifacts/evaluation/fixed2-s42-paired-val \
  --lead-sets 2-lead --run-name fixed2-s42

python -m src.use_contract audit \
  --evaluation-dir artifacts/evaluation/fixed2-s42-paired-test \
  --policy artifacts/selective-all/fixed2-s42/policy.json \
  --run-name fixed2-s42
```

Fitting:

1. selects one outcome-blind ECG per patient (shared across labels, leads and
   models) before any partitioning;
2. fits Platt calibration on one patient group, selects the reduced-stage
   threshold on a second, and selects rescue thresholds only on referred
   patients from a third;
3. controls trusted-positive and trusted-negative error separately with
   one-sided exact binomial bounds, Bonferroni-corrected over a fixed
   101-point threshold grid and four stage/direction families;
4. routes each case to `trust_reduced`, `obtain_12_lead` → `trust_12_lead`, or
   `expert_review` when no stage meets its bound;
5. writes `policy.json`, threshold curves, `feasibility.csv.gz` (the error-free
   sample each branch would need) and a hashed manifest under
   `artifacts/selective-all/<run>/`.

Policies require at least 25 positive and 25 negative validation patients;
sparser labels still receive model-performance outputs. An infeasible
direction gets no threshold and abstains: limits are never relaxed to create
coverage.

The audit verifies the saved policy hash, validation-only source manifest,
checkpoint, source input hashes, schema, lead definitions, eligibility and
target enumeration **before reading test predictions**, then applies the frozen
policy without refitting. It writes decisions, model and policy metrics,
calibration, subgroup tables, paired reduced-vs-12-lead contrasts,
risk-coverage curves, `repeat_ecg_sensitivity.csv.gz`,
`crc_sensitivity.csv.gz` (separate patient-cluster conformal risk control at
alpha 0.02) and a report under `artifacts/audit-all/<run>/`. Zero automated
coverage is reported explicitly; its conditional error is undefined, not zero.

## 8. Lower-level selective-prediction engine — `src/selective.py`

`src.selective` is the record-level engine underneath the policy code
(calibration, threshold search, exact bounds, sequential routing,
cross-fitting and bootstrap intervals). Its CLI fits a single diagnosis on a
validation export; it was the primary interface for the historical v1 AFIB
analysis and is documented in [`HISTORICAL_WORKFLOWS.md`](HISTORICAL_WORKFLOWS.md).

## Limitations

Record-level exact bounds assume independent records; the v2 workflow uses one
ECG per patient for its primary estimand and reports repeat-ECG and conformal
analyses separately. The PTB-XL test fold has been inspected during
development, so results on it are exploratory. Masked hospital ECGs are not
recordings from a two-lead device, and none of these methods guarantees
performance under distribution shift. External validation on an untouched
cohort is required before any stronger claim.
