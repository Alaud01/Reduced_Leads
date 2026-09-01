# Data Card — PTB-XL 1.0.3 (as used by Reduced_Leads)

Schema and content reference for the data consumed by this project. Everything here
reflects what `src/config.py`, `src/labels.py`, and `src/data.py` actually load and emit.

## Dataset identity

| | |
|---|---|
| Name | PTB-XL: A Large Publicly Available Electrocardiography Dataset |
| Version | 1.0.3 (use this version; earlier releases had duplicates/label fixes) |
| Source | PhysioNet — Wagner et al., 2020 (PhysioNet/Computing in Cardiology Challenge lineage) |
| License | CC-BY 4.0 (`LICENSE.txt` in the dataset folder) |
| Contents | 21,799 twelve-lead ECG recordings from 18,869 patients |
| Recording specs | 10 seconds, 12 standard leads, 100 Hz (`_lr`) and 500 Hz (`_hr`) versions |
| Format | WFDB: binary waveform `.dat` + text header `.hea` per record |
| Local location | `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/` (= `DATA_ROOT`) |

## On-disk layout

| Path | Content |
|---|---|
| `ptbxl_database.csv` | Master metadata: 21,799 rows (one per ECG), 26 columns, indexed by `ecg_id` |
| `scp_statements.csv` | Lookup of 71 SCP-ECG statement codes → diagnostic class / subclass / category |
| `records100/00000/…` | 100 Hz waveforms, 22 subfolders, one `.dat` + `.hea` pair per record |
| `records500/00000/…` | 500 Hz waveforms, same structure (unused by current config: `SIGNAL_HZ = 100`) |
| `RECORDS` | Plain list of all record paths (WFDB tooling) |
| `SHA256SUMS.txt` | Integrity checksums |

## Key metadata columns (`ptbxl_database.csv`)

Columns consumed by this codebase:

| Column | Type | Used for |
|---|---|---|
| `ecg_id` | int (index) | Sample identifier throughout the pipeline |
| `scp_codes` | stringified dict `{code: likelihood}` | Parsed in `labels.load_metadata`; source of all labels |
| `strat_fold` | int 1–10 | Split assignment → `fold_role` |
| `filename_lr` | path string | Passed to `wfdb.rdsamp` (e.g. `records100/00000/00001_lr`) |
| `filename_hr` | path string | 500 Hz counterpart (passthrough only) |

Derived column added by `labels.load_metadata()`:

| Column | Values | Rule |
|---|---|---|
| `fold_role` | `train` / `val` / `test` | folds 1–8 → train, 9 → val, 10 → test (`config.TRAIN_FOLDS`, `VAL_FOLD`, `TEST_FOLD`) |

The frozen-checkpoint evaluator exports `patient_id`, `age`, `sex`, and the available
noise/quality flags alongside diagnosis-level predictions. Other columns such as
`height`, `weight`, `nurse`, `site`, `device`, `recording_date`, and `validated_by`
remain unused. Samples are evaluated per recording; PTB-XL folds are patient-wise,
so no patient leaks across splits, but a patient may appear multiple times within a split.

## Label taxonomy

Built in `labels.build_label_frame()`; mapping subcode → superclass via
`scp_statements.csv` (`diagnostic == 1`, `diagnostic_class`).

| Head | # classes | Order defined in `config.py` |
|---|---|---|
| Superclass | 5 | NORM, MI, STTC, CD, HYP |
| Subcode | 44 | alphabetical: 1AVB … WPW (`config.SUBCODES`) |
| Rhythm | 12 | AFIB, AFLT, BIGU, PACE, PSVT, SARRH, SBRAD, SR, STACH, SVARR, SVTAC, TRIGU |

Labels are **multi-hot** (0/1): one recording can carry several diagnoses; an empty
`scp_codes` entry yields all-zero vectors (no explicit "no finding" class).

## Processed sample schema (`PTBXLDataset.__getitem__`)

Per-sample dict returned by `src/data.py`:

| Key | Shape (per sample) | Dtype | Meaning |
|---|---|---|---|
| `x` | (12, 1000) | float32 | 12 leads × 1000 time points (10 s @ 100 Hz, one sample / 10 ms); per-lead z-scored; dropped leads zeroed |
| `lead_mask` | (12,) | float32 | 1.0 = lead present, 0.0 = dropped/absent |
| `y_super` | (5,) | float32 | multi-hot superclass targets |
| `y_sub` | (44,) | float32 | multi-hot subcode targets |
| `y_rhythm` | (12,) | float32 | multi-hot rhythm targets |
| `ecg_id` | scalar int | — | Record identifier (batched to `(B,)` long) |

Batched by `make_collate`: `x` → (B, 12, 1000), `lead_mask` → (B, 12),
labels → (B, 5) / (B, 44) / (B, 12), `ecg_id` → (B,).

## Signal processing pipeline

1. `wfdb.rdsamp(filename_lr)` decodes `.dat` + `.hea` → ADC counts converted to
   physical units (µV), per-lead baseline applied → (1000, 12).
2. Pad with zeros / truncate to exactly 1000 samples if a record deviates.
3. Per-lead z-score with mean/std from `compute_train_norm` — computed on the
   **train split only**, first 2000 train records, cached and reused for all splits.
4. Lead-dropping: 0–10 leads randomly dropped per training sample (`drop_min=0`,
   `drop_max=10`; ≥2 leads always kept). Dropped leads are zeroed and flagged in
   `lead_mask`. At eval, fixed subsets instead (`TrainCfg.eval_subsets`):
   12 / 6 (limb) / 4 / 3 / 2 / 1 leads, with separate I-only and II-only sets.

## Caveats

- Normalization stats are an approximation: first 2000 train records in index
  order, not the full train split and not a random sample. New checkpoints persist
  the exact arrays; evaluation of older checkpoints recomputes them and records that
  fallback in the evaluation manifest.
- WFDB decodes invalid samples as NaN; `load_signal` does not impute or mask them.
- Reduced-lead inputs are simulated by masking 12-lead recordings — they are not
  recordings from actual reduced-lead devices (see PROJECT.md).
