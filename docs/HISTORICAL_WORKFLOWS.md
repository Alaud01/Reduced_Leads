# Historical workflows (v1)

> **Historical.** These commands reproduce analyses that predate the
> patient-level v2 design. New work should follow the
> [recommended workflow](../README.md#recommended-workflow-patient-level-v2)
> and [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md). The commands below still
> run, and their protocols and outputs are kept unchanged for provenance.

Every analysis on this page used the historical checkpoint
(`historical-random12-s42`, local path `checkpoints/best.pt`), which was
trained on folds 1-8 and **selected on fold 9**. Fold 9 was therefore not
independent of the model when these policies were fitted on it, the exact
binomial bounds were record-level (repeat ECGs from one patient were treated as
independent), and fold 10 was inspected repeatedly during development. All
results are exploratory. The weights are the `historical-random12-s42.best.pt`
release asset ([`ARTIFACTS.md`](ARTIFACTS.md)).

The evidence ladder and rationale for these steps are in
[`AFIB_EVALUATION_PROTOCOL.md`](AFIB_EVALUATION_PROTOCOL.md),
[`ALL_LABEL_EVALUATION_PROTOCOL.md`](ALL_LABEL_EVALUATION_PROTOCOL.md) and
[`USE_CONTRACT.md`](USE_CONTRACT.md).

## Historical model training and export

```bash
python -m src.train --device mps          # random lead dropping, 12-lead selection
python -m src.evaluate --checkpoint checkpoints/best.pt               # fold 9 only
python -m src.evaluate --checkpoint checkpoints/best.pt --splits val test
```

The current code selects checkpoints on fold 8, so retraining with this command
does **not** recreate the historical fold-9 selection; it produces a new
random-lead run with 12-lead selection (`random12` in
[`EXPERIMENTS.md`](EXPERIMENTS.md) terms) and the lead-presence auxiliary loss.

## Single-target AFIB selective prediction — `src/selective.py`

The selective-prediction CLI consumes validation-fold files only. It starts with
AFIB by default but supports any exported superclass, subcode or rhythm:

```bash
python -m src.selective \
  --evaluation-dir artifacts/evaluation/<complete-run> \
  --diagnosis-group rhythm \
  --diagnosis AFIB
```

If `--evaluation-dir` is omitted, the command finds the newest complete
validation export containing all requested lead sets. For every diagnosis ×
reduced-lead set it:

1. estimates development performance with patient-grouped outer cross-fitting;
2. splits each development partition by patient, fitting a monotone affine
   (Platt) calibrator on one half; the other half is split by patient into
   independent reduced-stage and rescue-stage risk partitions;
3. controls trusted-positive and trusted-negative errors separately using
   one-sided exact binomial bounds, corrected over a fixed threshold grid and
   the four reduced/12-lead decision families; the rescue threshold uses only
   records referred by the frozen reduced-stage policy;
4. applies the sequential policy: `trust_reduced` → `trust_12_lead` after
   acquisition → `expert_review` when neither stage meets its bound;
5. fits and serializes a final policy on fold 9 for one-time fold-10
   evaluation without refitting.

Defaults are `--max-positive-risk 0.10` and `--max-negative-risk 0.02`. These
are research defaults, not clinical requirements. An infeasible direction gets
no threshold and abstains.

Outputs under `artifacts/selective/<run>/`: `policy.json`,
`decisions.csv.gz`, `risk_coverage.csv.gz`, `summary.json`, and
`calibration.png`, `risk_coverage.png`, `actions.png`.

## Locked AFIB test audit — `src/audit_policy.py`

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --splits val test
python -m src.audit_policy \
  --evaluation-dir artifacts/evaluation/<val-and-test-export> \
  --policy artifacts/selective/<run>/policy.json \
  --protocol protocol/afib_v1.json
```

The audit refuses incomplete or `--max-n`-limited test exports and verifies the
checkpoint hash, label schema, target, lead sets, policy state and risk limits
before reading test predictions. It writes fit-free decisions, discrimination
and calibration metrics, conservative patient-level summaries, patient-cluster
bootstrap intervals, subgroup tables, paired reduced-vs-12-lead contrasts,
descriptive risk-coverage curves and a hashed manifest under
`artifacts/audit/`. A result within the v1 risk limits is not a finite-sample
guarantee, a clinical claim, or permission to retune on fold 10.

## Post-hoc positive-call sensitivity policy

The original locked policy found no positive threshold satisfying its 10%
simultaneous upper-risk constraint. This separate, non-confirmatory policy
keeps the 2% negative-risk constraint and 25-case minimum but relaxes the
positive-risk constraint to 40%: the smallest round validation-only setting
that makes the two-lead pathway estimable after 12-lead confirmation, not a
clinically acceptable error target.

```bash
python -m src.selective \
  --evaluation-dir artifacts/evaluation/<validation-export> \
  --run-name afib-positive-exploratory-risk40-v2 \
  --max-positive-risk 0.40 --max-negative-risk 0.02 \
  --confidence 0.95 --min-trusted 25 --bootstrap 2000

python -m src.audit_policy \
  --evaluation-dir artifacts/evaluation/<test-export> \
  --policy artifacts/selective/afib-positive-exploratory-risk40-v2/policy.json \
  --protocol protocol/afib_positive_exploratory_v1.json \
  --run-name afib-positive-exploratory-test-v2
```

Because it was proposed after reviewing the original test audit, its test
results are hypothesis-generating even though thresholds were fitted on
validation data only.

## All-label research audit — `src/audit_all_labels.py`

Extends the validation-fit / test-audit boundary to every superclass,
subcode and rhythm label. Every label gets discrimination, calibration,
prevalence, risk-coverage and reduced-vs-12-lead comparisons; selective
policies are fitted only for labels with at least 25 positive and 25 negative
validation patients (others are marked `insufficient_data`).

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

The shared 10% positive / 2% negative limits are comparison anchors, not
diagnosis-specific clinical choices. The aggregate tables from
`all-labels-test-v3` are tracked in
[`results/historical/all-labels-test-v3/`](../results/historical/all-labels-test-v3/report.md).

## Use contract v1

`protocol/use_contract_v1.json` introduced the diagnosis-specific limits (see
[`USE_CONTRACT.md`](USE_CONTRACT.md)) and the AFIB-or-AFLT composite. The
`src.use_contract` CLI now defaults to v2, so pass the v1 protocol explicitly:

```bash
python -m src.use_contract fit \
  --evaluation-dir artifacts/evaluation/<validation-only-export> \
  --protocol protocol/use_contract_v1.json \
  --run-name contract-routed-v2

python -m src.use_contract audit \
  --evaluation-dir artifacts/evaluation/<test-export> \
  --policy artifacts/selective-all/contract-routed-v2/policy.json \
  --protocol protocol/use_contract_v1.json \
  --run-name contract-routed-v2-audit
```

Bundles record execution method
`fixed-grid-clopper-pearson-bonferroni-routed-v2`. Older all-label/composite
bundles must be refitted on validation into a new directory to use the
corrected method; they are not silently upgraded.

To run the patient-level v2 *code path* on the historical checkpoint for
software or sensitivity checks, use
`--protocol protocol/use_contract_v2_legacy_exploratory.json`. The default v2
protocol rejects that checkpoint because it was selected on fold 9; the
legacy protocol accepts it explicitly without formal guarantees. The
historical row in the seed-42 comparison (`patient-v2-final-audit`) was
produced this way.
