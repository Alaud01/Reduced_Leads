# AFIB evaluation protocol

Status: proposed study and implementation plan. This is a research protocol,
not a claim of clinical safety or regulatory readiness. References use the keys
defined in [`REFERENCES.md`](REFERENCES.md).

## Executive recommendation

The next useful build is not another validation-set AFIB score. It is a staged,
locked evaluation:

1. develop the model without using the policy-calibration or test patients;
2. fit calibration and the three-action selective policy once on a dedicated
   calibration split;
3. apply that frozen policy once to PTB-XL fold 10, with patient-level confidence
   intervals and no refitting;
4. repeat the frozen evaluation on source-held-out ECG cohorts; and
5. only then test the workflow prospectively on recordings acquired with the
   intended reduced-lead device.

This ordering follows the separation of development and evaluation emphasized
by TRIPOD+AI and PROBAST+AI [collins2024_tripod_ai; moons2025_probast_ai]. It also
matches the core lesson from multi-source ECG challenges: internal results can
fall materially when the data source changes [perezalday2020_multisource].

## What the repository establishes today

The existing pipeline already provides several useful foundations:

- patient-disjoint official PTB-XL folds [wagner2020_ptbxl];
- a frozen-checkpoint exporter for validation and test predictions;
- paired predictions for 12-lead and each masked reduced-lead input;
- monotone affine (Platt) calibration;
- a three-action policy: trust reduced lead, acquire 12 leads, or refer;
- patient-grouped cross-fitting within validation fold 9; and
- patient-bootstrap intervals for cross-fitted development summaries.

The present AFIB result is nevertheless a development result. `src/train.py`
selects the checkpoint on fold 9 using 12-lead superclass macro-AUROC, while
`src/selective.py` also develops and audits the AFIB policy inside fold 9. The
policy file is described as ready for fold 10, but there is no command that
loads that file and produces a locked fold-10 policy audit.

The local PTB-XL 1.0.3 copy contains the following AFIB counts:

| Role | Records | Patients | AFIB-positive records | AFIB-positive patients |
|---|---:|---:|---:|---:|
| Train, folds 1-8 | 17,418 | 15,023 | 1,211 | 991 |
| Validation, fold 9 | 2,183 | 1,942 | 151 | 124 |
| Test, fold 10 | 2,198 | 1,904 | 152 | 130 |

These counts are enough for a useful internal test, but 130 positive patients
will not support very precise estimates for every lead set, subgroup, noise
stratum, and operating threshold. The study should therefore choose one primary
comparison and calculate sample size from desired confidence-interval precision,
not a generic events-per-variable rule [riley2021_external_validation_size].

## 1. Write the clinical-use contract first

Freeze a machine-readable protocol before opening fold 10. At minimum, specify:

- **Intended population:** for example, adults receiving a 10-second resting ECG
  in primary care. Do not silently generalize this to ambulatory screening.
- **Input:** exact device, lead placement, sampling rate, duration, filtering,
  gain, and missing-signal behavior.
- **Target:** primary target `AFIB` one-vs-rest; secondary targets should include
  `AFIB-or-AFLT` and explicit error analysis against AFLT, sinus rhythm, SVT,
  ectopy, and paced rhythm.
- **Reference standard:** the dataset label for retrospective experiments; for
  new data, blinded adjudication by at least two qualified readers with a
  prespecified disagreement process.
- **Unit of analysis:** patient for inference and formal guarantees; ECG record
  may remain the operational prediction unit.
- **Actions:** what `trust_positive`, `trust_negative`, `acquire_12_lead`, and
  `expert_review` actually trigger in the proposed workflow.
- **Primary harm:** prespecify separate tolerances for false reassurance and
  false alert. The current 2% trusted-negative and 10% trusted-positive error
  limits are research placeholders, not clinically justified thresholds.

This system should be described as triage or decision support. Current AF
guidance requires clinician visual confirmation of a first AF diagnosis,
regardless of the monitoring device [joglar2024_af_guideline].

## 2. Separate model tuning, policy calibration, and evaluation

For a clean rerun, use the official folds as follows:

| Data | Permitted use | Forbidden use |
|---|---|---|
| Folds 1-8 | Model fitting; inner patient-grouped cross-validation for architecture, loss weights, epoch selection, and AFIB-specific model choices | Threshold or performance claims on fold 10 |
| Fold 9 | Fit Platt parameters and selective thresholds after the model is frozen | Further model or feature selection |
| Fold 10 | One locked internal evaluation | Refitting, threshold adjustment, choosing a preferred lead set after inspecting results |
| External cohorts | Transportability evaluation, initially without recalibration | Model selection against the external answer key |

The current checkpoint was selected using fold 9. It can still be used for an
exploratory fold-10 analysis, but a confirmatory run should move epoch and
hyperparameter selection into patient-grouped inner cross-validation on folds
1-8. Then refit on all folds 1-8, calibrate on fold 9, hash the checkpoint and
policy, and unlock fold 10 exactly once.

Record all candidate architectures, seeds, and lead sets considered. A single
reported seed understates training variability; run at least three to five
prespecified training seeds and either evaluate a frozen ensemble or report the
distribution of results. Do not choose the best seed on fold 10.

## 3. Build comparators that answer the actual lead question

The reduced-lead literature shows that useful information can remain in fewer
leads, but it does not establish that masked hospital 12-lead recordings are
equivalent to a real portable acquisition [reyna2021_reduced_leads]. Compare:

1. the present lead-aware model evaluated with each lead mask;
2. a dedicated model trained for each lead set;
3. a strong, simpler ECG baseline such as a 1-D ResNet/Inception-style model,
   which performed well in PTB-XL benchmarking [strodthoff2021_ptbxl_benchmark];
4. the full 12-lead model as a reference, not as an infallible gold standard;
5. an unselective fixed-threshold classifier; and
6. the selective policy with acquisition and referral.

Hannun et al. demonstrate that single-lead ambulatory ECG can support strong
arrhythmia classification, while Ribeiro et al. provide a large 12-lead deep
learning example [hannun2019_single_lead; ribeiro2020_12lead]. Their acquisition,
population, labels, and reference standards differ from this project, so they
are methodological precedents rather than performance targets.

## 4. Evaluate more than rank discrimination

For every lead set, report patient-clustered 95% confidence intervals for:

- **Discrimination:** AUROC and area under the precision-recall curve. Always
  report AFIB prevalence beside precision-recall results.
- **Locked operating point:** sensitivity, specificity, positive and negative
  predictive value, false-negative and false-positive rates, and the full
  confusion matrix.
- **Calibration:** calibration-in-the-large, calibration slope, Brier score,
  log loss, and a smooth calibration plot with uncertainty. Neural-network
  scores commonly need post-hoc calibration [guo2017_calibration], and clinical
  evaluation should not collapse calibration to one scalar
  [vancalster2019_calibration].
- **Selective behavior:** coverage, selective error, trusted-positive error,
  trusted-negative error, referral rate, 12-lead acquisition rate, and area
  under the risk-coverage curve. Selective prediction is explicitly a
  risk-versus-coverage problem [geifman2017_selective].
- **Incremental value:** paired differences between reduced lead, 12 lead, and
  the sequential policy. Use a patient-cluster bootstrap for the paired
  difference. DeLong's test is appropriate for correlated ROC curves when its
  observation assumptions fit, but it does not by itself solve repeated ECGs
  per patient [delong1988_correlated_auc].
- **Clinical utility:** if defensible action costs or threshold probabilities
  can be set with clinicians, report decision curves/net benefit rather than
  claiming utility from AUROC alone [vickers2006_decision_curve].

Choose one primary endpoint, such as fold-10 trusted-negative error at the
prespecified minimum coverage for 2-lead ECG. Treat other lead sets and subgroup
tests as secondary and control multiplicity or present adjusted intervals.

## 5. Make the risk guarantee match the data unit

The current threshold search uses one-sided Clopper-Pearson binomial bounds over
a fixed, Bonferroni-corrected threshold grid [clopper1934_binomial]. That is a
reasonable conservative baseline if calibration errors are independent
Bernoulli trials. In the current code, however, trials are ECG records and some
patients contribute multiple records. Patient-grouped splitting prevents a
patient appearing in both fit and audit partitions, but records within the
calibration partition can still be dependent.

Implement and label two analyses:

- **Descriptive record-level analysis:** retain all ECGs; use a patient-cluster
  bootstrap for uncertainty; make no exact finite-sample patient guarantee.
- **Formal patient-level sensitivity analysis:** deterministically select one
  eligible ECG per patient before calibration, or define one bounded loss per
  patient (for example, any wrong automated decision for that patient) and
  calibrate the risk bound over patients.

Add conformal risk control as a prespecified sensitivity analysis only after the
loss, monotonicity, calibration unit, and exchangeability claim are explicit.
Conformal risk control can bound the expected value of bounded monotone losses,
but the guarantee depends on its assumptions and does not automatically survive
dataset or device shift [angelopoulos2024_crc]. Report both achieved coverage
and the bound; abstaining on nearly everything is not a useful success.

## 6. Stress tests and hidden stratification

Run the frozen model and policy through prespecified strata:

- age bands, sex, site, and ECG device;
- clean versus each PTB-XL quality flag;
- AFIB with/without other rhythm or diagnostic labels;
- paced rhythm and ectopy, reported separately;
- heart-rate bands and high/low AFIB confidence; and
- one versus multiple ECGs per patient.

For each stratum, report prevalence, patient count, positive-patient count,
coverage, sensitivity, specificity, predictive values, calibration, and
selective errors with intervals. Emphasize effect sizes and interactions, not a
collection of underpowered within-group significance tests. Suppress or clearly
mark estimates below a prespecified minimum count.

Synthetic perturbations are useful engineering tests: baseline wander, mains
interference, burst noise, amplitude scaling, temporal shift, channel dropout,
lead reversal, and clipping. Their severity must be defined in physical units
and reviewed for label preservation. They are robustness tests, not substitutes
for recordings from the target device.

## 7. External evaluation

Start with public source-held-out 12-lead cohorts that contain AF labels, such as
the Chapman-Shaoxing dataset and usable sources from the PhysioNet Challenge
collections [zheng2020_chapman; perezalday2020_multisource]. For each source:

1. write a versioned adapter for lead names, sampling rate, duration, units,
   filters, and labels;
2. publish the exact AFIB/AFLT label mapping and an unmapped-label table;
3. prevent any source used for model selection from being called external;
4. apply the original normalization, model, calibration, and thresholds first;
5. report source-specific results rather than only a pooled average; and
6. if recalibration is necessary, split that source by patient into recalibration
   and untouched audit sets and label the result as locally updated.

The multicenter Challenge data show why this matters: data and labels differ by
institution, and hidden-source performance can drop despite strong development
results [perezalday2020_multisource].

## 8. Prospective evidence ladder

External retrospective accuracy still does not establish workflow benefit.
Advance in stages:

1. **Device-equivalence study:** simultaneously acquire the intended reduced
   leads and a reference 12-lead ECG; quantify waveform and prediction agreement.
2. **Silent prospective study:** run the frozen policy without exposing output
   to clinicians; measure calibration, failure modes, latency, and missingness.
3. **Assisted workflow study:** expose output, capture overrides, acquisition and
   referral rates, time to decision, and automation bias; use DECIDE-AI reporting
   [vasey2022_decide_ai].
4. **Impact trial:** if clinical use is intended, compare the workflow against
   usual care using patient-relevant and safety outcomes, not only model scores.

The initial AF diagnosis must remain visually confirmed by a clinician unless
the governing clinical standard changes [joglar2024_af_guideline].

## Implementation blueprint

### A. Locked internal audit (implemented)

`src/audit_policy.py` now provides a deliberately fit-free interface:

```text
python -m src.audit_policy \
  --evaluation-dir artifacts/evaluation/<val-and-test-export> \
  --policy artifacts/selective/<run>/policy.json \
  --split test \
  --protocol protocol/afib_v1.json
```

It should:

- verify the policy status is `frozen_on_validation`;
- verify checkpoint SHA-256, diagnosis, lead sets, label schema, and split;
- reject any policy whose source manifest lacks fold-10 predictions;
- call only policy-application functions, never fit functions;
- pair test predictions by `ecg_id` and validate patient consistency;
- emit point estimates, patient-cluster bootstrap intervals, subgroup tables,
  paired lead-set contrasts, and plots; and
- write an immutable manifest containing hashes of all inputs and the protocol.

Recommended outputs:

```text
artifacts/audit/<run>/
  manifest.json
  metrics.json
  decisions.csv.gz
  subgroup_metrics.csv.gz
  paired_contrasts.csv.gz
  calibration.csv.gz
  risk_coverage.csv.gz
  report.md
```

### B. Statistics module (initial implementation)

`src/statistics.py` now provides reusable patient-cluster bootstrap resampling,
calibration intercept/slope, Brier/log loss, threshold metrics, paired contrasts,
reliability tables, descriptive risk-coverage curves, and conservative
patient-level selective summaries. Fixed fixtures verify that a sampled patient
brings all of their ECGs into a bootstrap replicate. Multiplicity-adjusted
confirmatory intervals remain future work; the v1 protocol labels secondary
comparisons descriptive.

### C. Post-hoc positive-call sensitivity analysis

The original 10% positive-error policy produced no feasible positive threshold.
For a transparent sensitivity analysis, `protocol/afib_positive_exploratory_v1.json`
defines a separate policy with a 40% simultaneous positive-risk upper-bound
constraint while retaining the 2% negative constraint, 95% confidence, fixed
101-threshold grid, and minimum of 25 trusted records.

The 40% limit is not an expected error rate or clinical target. On validation,
the best 2-lead pathway's 12-lead confirmation threshold had 1 error among 27
trusted positive records (3.7% empirical error), but its multiplicity-adjusted
one-sided upper bound was 35.3%. Forty percent is the smallest round value that
makes that pathway feasible without weakening the minimum sample or negative
constraint. Since this analysis was specified after the original test audit,
all resulting test estimates are post-hoc and hypothesis-generating; the
original locked result remains primary.

Observed on the existing test set, the 2-lead pathway automatically and
correctly identified 66 of 152 AFIB records (43.4%; patient-bootstrap 95% CI
35.2%-52.2%). It made 80 trusted positive calls, including 14 false positives,
for a 17.5% positive-call error (95% CI 9.1%-26.9%). Expert referral fell from
13.6% under the original policy to 10.0%, while total automated error increased
from 0.7% to 1.4%. The negative-call error was unchanged at 0.7% because the
negative constraint and threshold were unchanged.

These estimates describe the tradeoff the sensitivity policy creates; they do
not validate the 40% constraint. Subgroup positive-call counts are often small
and must be treated as unstable. New untouched external data is required before
choosing or confirming any positive-call operating point.

### C. External dataset contract

Add `src/external/base.py` with a canonical record schema and one adapter per
cohort. Every adapter should produce a data-quality report before inference and
fail closed on missing units, duplicate identifiers, unknown lead order, or an
unapproved label map.

### D. Protocol and model cards

Version the prespecified primary endpoint, operating limits, inclusion rules,
subgroups, perturbations, and analysis code. Report the final work with
TRIPOD+AI and assess it with PROBAST+AI
[collins2024_tripod_ai; moons2025_probast_ai].

## Promotion gates

Do not advance evidence levels merely because a mean metric looks good.

| Gate | Minimum evidence |
|---|---|
| Internal test complete | Frozen fold-10 audit; all primary estimates and patient-level intervals; no test-driven changes |
| External transport supported | Acceptable prespecified performance on at least two independent sources, with source-specific calibration and failure analysis |
| Device claim supported | Direct recordings from the intended lead placement/device, not masked hospital ECGs |
| Workflow claim supported | Prospective silent and assisted evaluation, including human factors and overrides |
| Clinical benefit claim supported | Appropriately designed comparative impact study and safety monitoring |

Any change prompted by fold 10 or external results creates a new model version
and requires a new untouched evaluation cohort. That rule is more important than
the choice of any single neural-network architecture.
