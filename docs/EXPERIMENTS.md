# Experiments and naming conventions

## Project name

The project is called **Reduced Leads** in prose. `Reduced_Leads` is only the
GitHub repository slug (and default clone directory), and `reduced-leads` is
the Python distribution name in `pyproject.toml`.

## Lead-set names

Lead sets always use the canonical keys from `LEAD_SUBSETS` in
[`src/config.py`](../src/config.py). These names are persisted in prediction
files, manifests and policies, so they never change:

| Lead-set key | Leads | Short form in run names | Prose |
|---|---|---|---|
| `12-lead` | all twelve | `12` | 12-lead |
| `6-lead-limb` | I, II, III, aVR, aVL, aVF | `6limb` | 6-lead limb |
| `4-lead` | I, II, III, V2 | `4` | 4-lead |
| `3-lead` | I, II, V2 | `3` | 3-lead |
| `2-lead` | I, II | `2` | I+II (two-lead) |
| `1-lead-I` | I | `1I` | lead I |
| `1-lead-II` | II | `1II` | lead II |

## Run names

Training runs are named `<regime><lead-set>-s<seed>`:

- **`fixed<lead-set>`**: the model only ever sees that lead set, during
  training *and* checkpoint selection
  (`--train-lead-set <lead-set-key>`). `fixed2` is trained and selected on
  I+II; `fixed12` on all 12 leads.
- **`random<lead-set>`**: the model is trained with random lead dropping
  (any 2-12 leads per sample, `--train-lead-set random`), and the
  **lead set in the name is the one used to choose the checkpoint** on the
  selection fold (`--selection-lead <lead-set-key>`). `random2` therefore
  means "trained on random subsets, best epoch chosen by I+II validation
  AUROC", not "trained on two leads".
- **`-s<seed>`**: the training seed (`--seed`). The prespecified paired seeds
  are 42, 43 and 44.
- **`historical-` prefix**: runs that predate the patient-level v2 split
  design and are kept only as a reference (see below).

This distinction between *training input* and *checkpoint-selection input*
matters: two models can see identical inputs at test time but differ in what
they were trained on (`fixed2` versus `random2`), or share a training regime
but differ in which validation input picked the epoch (`random2` versus
`random12`).

Derived outputs append a suffix to the run name:

| Pattern | Meaning | Example |
|---|---|---|
| `<run>-val`, `<run>-test` | `src.evaluate` export of one split | `fixed2-s42-val` |
| `<run>-paired-<split>` | `src.assemble_evaluation` bundle: this run's I+II predictions plus `fixed12-s42` 12-lead predictions for the rescue stage | `random2-s42-paired-test` |
| `neg<limit>-pos<limit>--<run>` | relaxed-risk sensitivity policy; limits in percent with the decimal point dropped (`025` = 2.5%) | `neg025-pos10--random2-s42` |

Older output names such as `all-labels-test-v3` or
`afib-locked-test-audit-v2` carry a version suffix because each rerun after a
method correction was written to a new directory instead of overwriting the
previous one.

## Run registry

| Run | Training input | Selection input / fold | Training folds | Lead-presence loss | Status |
|---|---|---|---|---|---|
| `fixed12-s42` | 12-lead | 12-lead / fold 8 | 1-7 | 0 (always off for fixed input) | current; 12-lead rescue model for all v2 comparisons |
| `fixed2-s42` | I+II | I+II / fold 8 | 1-7 | 0 (always off for fixed input) | current |
| `random2-s42` | random lead dropping | I+II / fold 8 | 1-7 | 0 (`--lead-presence-weight 0`) | current |
| `historical-random12-s42` | random lead dropping | 12-lead / fold 9 | 1-8 | 0.2 | historical; fold 9 was used for selection, so it cannot have independent calibration |

The current runs were trained on Runpod (CUDA, `torch 2.4.1+cu124`, batch
size 64, 50 epochs). The historical run was trained locally on Apple Silicon
MPS with batch size 32. Training records for all four runs are in
[`results/training/`](../results/training/), and the weights are release
assets ([`ARTIFACTS.md`](ARTIFACTS.md)).

Commands for the current runs (repeat with `--seed 43` and `--seed 44`):

```bash
python -m src.train --train-lead-set 12-lead --seed 42 --out-dir checkpoints/fixed12-s42
python -m src.train --train-lead-set 2-lead  --seed 42 --out-dir checkpoints/fixed2-s42
python -m src.train --train-lead-set random --selection-lead 2-lead \
  --lead-presence-weight 0 --seed 42 --out-dir checkpoints/random2-s42
```

The comparison in [`RESULTS.md`](RESULTS.md) uses these labels:

| Label in reports | Reduced (I+II) stage | 12-lead rescue stage |
|---|---|---|
| Fixed I+II | `fixed2-s42` | `fixed12-s42` |
| Random lead training | `random2-s42` | `fixed12-s42` |
| Twelve-lead model masked to I+II | `fixed12-s42` with ten leads zeroed | `fixed12-s42` |
| Previous model, patient analysis | `historical-random12-s42` | `historical-random12-s42` |
