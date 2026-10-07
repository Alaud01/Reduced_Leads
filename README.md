# Reduced Leads

[![tests](https://github.com/Alaud01/Reduced_Leads/actions/workflows/tests.yml/badge.svg)](https://github.com/Alaud01/Reduced_Leads/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Diagnosis-specific selective prediction for reduced-lead ECG AI.**

Average accuracy cannot answer the clinical deployment question: *for this
recording and this diagnosis, is a reduced-lead ECG reliable enough to act on,
or should the system ask for a full 12-lead ECG or an expert?* This project
trains lead-aware transformer models on [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/),
evaluates them on reduced lead sets, and turns calibrated scores into one of
three actions per diagnosis (trust the reduced-lead call, acquire 12 leads,
or refer for expert review) with patient-level statistical risk control and
locked, fit-free test audits.

## Project status

Research prototype; not a medical device. Seed-42 results
([`docs/RESULTS.md`](docs/RESULTS.md)):

- **Reduced-lead quality depends on the diagnosis.** A model trained directly
  on leads I+II improves two-lead superclass macro-AUROC (0.859 → 0.867),
  MI and conduction disorders, but loses AFIB-or-AFLT ranking (0.958 → 0.929).
- **High AUROC did not yield safe automation.** No model met the prespecified
  2% accepted-negative error limit for two-lead AFIB-or-AFLT rule-out, so every
  case is escalated. Abstention is the intended safe failure, not a bug.
- **Limitations.** One training seed; the PTB-XL test fold has been inspected,
  so results are exploratory; masked hospital ECGs are not a portable two-lead
  device. Seeds 43/44 and an untouched external cohort are the next steps.

## Repository layout

```text
├── README.md, LICENSE, CITATION.cff
├── pyproject.toml, requirements.txt, requirements-lock.txt
├── docs/                    Documentation (index: docs/README.md) and visuals/
├── protocol/                Frozen, machine-readable evaluation protocols
├── results/                 Curated reports and aggregate tables (tracked)
├── examples/                Synthetic evaluation export for trying the pipeline
├── scripts/                 Dataset exploration, synthetic data, release bundles
├── src/
│   ├── config.py            Leads, lead subsets, label schema, model/training config
│   ├── labels.py            Metadata loading, multi-hot labels, class weights, folds
│   ├── data.py              WFDB loading, normalization, lead-dropping dataset
│   ├── model.py             LeadAwareTransformer (lead + time attention)
│   ├── losses.py            Multi-task weighted BCE loss
│   ├── training_state.py    Fold-8 selection split, resume contract, RNG state
│   ├── train.py             Training CLI (manifests, logs, curves, resume)
│   ├── evaluate.py          Frozen-checkpoint prediction export CLI
│   ├── assemble_evaluation.py  Combine reduced and 12-lead exports for routing
│   ├── selective.py         Calibration, exact risk bounds, sequential routing
│   ├── research_risk.py     Patient representatives, feasibility, conformal risk control
│   ├── statistics.py        Patient-aware metrics and clustered bootstrap
│   ├── audit_all_labels.py  All-label policy fitting and fit-free audit engine
│   ├── use_contract.py      Use-contract CLI: AFIB-or-AFLT composite + per-group limits
│   ├── audit_policy.py      Historical single-target locked audit
│   └── io_utils.py          Hashing, atomic writes, portable paths
└── tests/                   Unit, workflow and split-integrity tests
```

Local outputs (`checkpoints/`, `artifacts/`, `figures/`) and the dataset are
gitignored; see [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) for what is tracked,
released or ignored.

## Setup

Python 3.10 or newer:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements-lock.txt for exact CI pins
python -m unittest discover -s tests -q
```

Download [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) and unzip
it in the repository root so that
`ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv`
exists ([`docs/DATASET.md`](docs/DATASET.md)). Training defaults to Apple
Silicon MPS and falls back to CPU; pass `--device cuda` on NVIDIA GPUs.

### Try it without the dataset

The synthetic export in [`examples/`](examples/README.md) runs through policy
fitting and the locked audit in seconds:

```bash
python -m src.use_contract fit --evaluation-dir examples/synthetic-evaluation/val \
  --lead-sets 2-lead --run-name synthetic-demo
python -m src.use_contract audit --evaluation-dir examples/synthetic-evaluation/test \
  --policy artifacts/selective-all/synthetic-demo/policy.json --run-name synthetic-demo-audit
```

Pretrained seed-42 weights and complete per-record result bundles are release
downloads ([`docs/ARTIFACTS.md`](docs/ARTIFACTS.md#downloading)).

## Recommended workflow (patient-level v2)

Folds 1-7 fit weights, fold 8 selects the checkpoint, fold 9 calibrates and
fits the decision policy, and fold 10 is audited once
([`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md)). Run names follow
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): `fixed2` is trained and selected
on leads I+II, `random2` is trained on random lead subsets and *selected* on
I+II, and `fixed12` is the 12-lead model that also serves as the rescue stage.

**1. Train** the matched comparators (repeat with seeds 43 and 44):

```bash
python -m src.train --train-lead-set 12-lead --seed 42 --out-dir checkpoints/fixed12-s42
python -m src.train --train-lead-set 2-lead  --seed 42 --out-dir checkpoints/fixed2-s42
python -m src.train --train-lead-set random --selection-lead 2-lead \
  --lead-presence-weight 0 --seed 42 --out-dir checkpoints/random2-s42
```

**2. Export validation predictions** (fold 9 only) and pair each two-lead model
with the 12-lead rescue model:

```bash
python -m src.evaluate --checkpoint checkpoints/fixed12-s42/best.pt \
  --splits val --lead-sets 12-lead 2-lead --run-name fixed12-s42-val
for run in fixed2-s42 random2-s42; do
  python -m src.evaluate --checkpoint checkpoints/$run/best.pt \
    --splits val --lead-sets 2-lead --run-name $run-val
  python -m src.assemble_evaluation --reduced-dir artifacts/evaluation/$run-val \
    --twelve-dir artifacts/evaluation/fixed12-s42-val --lead-set 2-lead --split val \
    --out-dir artifacts/evaluation/$run-paired-val
done
```

**3. Fit and freeze policies** on validation (the default protocol is
`protocol/use_contract_v2.json`):

```bash
python -m src.use_contract fit --evaluation-dir artifacts/evaluation/fixed2-s42-paired-val \
  --lead-sets 2-lead --run-name fixed2-s42
```

Repeat for `random2-s42-paired-val` and `fixed12-s42-val`.

**4. Only after every policy is frozen**, export the test fold the same way
(`--splits test`, `-test` / `-paired-test` names) and audit without refitting:

```bash
python -m src.use_contract audit --evaluation-dir artifacts/evaluation/fixed2-s42-paired-test \
  --policy artifacts/selective-all/fixed2-s42/policy.json --run-name fixed2-s42
```

The audit verifies hashes, schemas and validation-only provenance before it
reads any test prediction. [`results/runpod-s42/run_pipeline.py`](results/runpod-s42/run_pipeline.py)
is the exact script that produced the published comparison. Stage-by-stage
details are in [`docs/METHODS.md`](docs/METHODS.md).

## Documentation

| Document | Contents |
|---|---|
| [`docs/PROJECT.md`](docs/PROJECT.md) | Research question, scope and planned analyses |
| [`docs/DATASET.md`](docs/DATASET.md) | PTB-XL provenance, label taxonomy, processing, caveats |
| [`docs/METHODS.md`](docs/METHODS.md) | Pipeline reference: folds, model, training, export, policies, audits |
| [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md) | Current patient-level v2 statistical protocol |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Naming conventions and run registry |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Results and comparison with the literature |
| [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) | What is tracked, released or ignored; downloads |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Environments, tests, CI, scripts |
| [`docs/HISTORICAL_WORKFLOWS.md`](docs/HISTORICAL_WORKFLOWS.md) | Superseded v1 commands, kept for reproducibility |
| [`docs/visuals/`](docs/README.md#visuals) | Research poster, pipeline explainer and dataset overview pages |

## Citation and license

The code is released under the [MIT License](LICENSE). If you use this
project, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this
repository" button) and cite PTB-XL:

> Wagner P, Strodthoff N, Bousseljot R-D, et al. PTB-XL, a large publicly
> available electrocardiography dataset. *Scientific Data*. 2020;7:154.
> [doi:10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)

### Data attribution

This project uses **PTB-XL 1.0.3** by Patrick Wagner, Nils Strodthoff,
Ralf-Dieter Bousseljot, Wojciech Samek and Tobias Schaeffter, published on
PhysioNet ([doi:10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)) under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
The dataset itself is not redistributed here. The released model weights were
trained on PTB-XL waveforms, and the result tables and release bundles are
derived from PTB-XL: they contain its labels, pseudonymous patient identifiers
and metadata alongside model predictions, aggregated or reformatted by this
project. The PTB-XL authors and PhysioNet do not endorse this project.

The ECG electrode-placement illustration in `docs/visuals/assets/` was created
with OpenAI image generation.
