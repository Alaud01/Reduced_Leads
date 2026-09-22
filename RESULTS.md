# Results and literature comparison

This document summarizes the frozen Runpod seed-42 evaluation and places it in
the context of related ECG and selective-prediction research. The underlying
project results are in
[`artifacts/comparison/runpod-s42/report.md`](artifacts/comparison/runpod-s42/report.md).
The complete bibliography and source links are in [`REFERENCES.md`](REFERENCES.md).

These are exploratory results from an already inspected PTB-XL test fold and a
single training seed. They must not be used to retune thresholds on fold 10.

## Overall assessment

The results are strong enough for a thesis or capstone, a reproducible methods
report, and potentially a feasibility preprint after the prespecified repeated
seeds are completed. They are not sufficient for clinical deployment, a claim
that two leads are equivalent to twelve leads, or a claim that the system can
safely automate AFIB-or-AFLT triage at the specified risk limit.

The clearest scientific result is diagnosis-specific: fixed I+II training helps
some targets but harms others. The clearest operational result is negative: high
AUROC did not produce any automated decisions under the prespecified 2% trusted-
negative error limit.

## Model comparison

### Broad discrimination

| Approach | Evaluation input | Superclass macro-AUROC | Subcode macro-AUROC | Rhythm macro-AUROC |
|---|---|---:|---:|---:|
| Historical model | 12 leads | 0.9128 | 0.8774 | 0.9148 |
| Historical model | I+II | 0.8586 | 0.8246 | 0.8921 |
| New fixed I+II model | I+II | **0.8673** | **0.8268** | 0.8784 |
| New random-lead model | I+II | 0.8516 | 0.8209 | **0.8940** |
| New fixed 12-lead model | 12 leads | 0.8954 | 0.8440 | 0.8667 |
| Fixed 12-lead model masked after training | I+II | 0.6409 | 0.6447 | 0.7655 |

Fixed I+II improves the primary two-lead superclass summary from 0.8586 to
0.8673. Random-lead training gives the strongest new rhythm summary, but does
not broadly improve the superclass or subcode summaries. The new 12-lead model
also underperforms the historical 12-lead model, so the new training approach is
not a general improvement.

### Diagnosis-specific two-lead results

| Approach | AFIB-or-AFLT | MI | STTC | Conduction disorders | Normal superclass |
|---|---:|---:|---:|---:|---:|
| Historical model | **0.9582** | 0.8190 | 0.8962 | 0.8351 | 0.9089 |
| New fixed I+II model | 0.9287 | **0.8409** | **0.8991** | **0.8496** | **0.9112** |
| New random-lead model | 0.9483 | 0.8053 | 0.8918 | 0.8359 | 0.9023 |
| Fixed 12-lead model masked to I+II | 0.7599 | 0.6073 | 0.6740 | 0.5149 | 0.7070 |

Fixed I+II improves MI, STTC, conduction-disorder, and normal-superclass AUROC,
but AFIB-or-AFLT falls from 0.9582 to 0.9287. The patient-paired AFIB-or-AFLT
difference for fixed I+II versus the historical model is -0.0319, with a
descriptive 95% bootstrap interval of [-0.0490, -0.0168]. That interval excludes
zero for this trained pair, but it does not include variation from retraining
with different random seeds.

Random-lead training retains more AFIB-or-AFLT ranking performance than fixed
I+II and improves average precision from 0.6690 to 0.6799. Its calibrated
patient-level Brier score also improves from 0.0338 to 0.0309. This means its
probability estimates are slightly better overall even though its AUROC is lower.

### Primary triage policy

| Approach | Automated coverage | 12-lead acquisition | Expert referral |
|---|---:|---:|---:|
| Historical model | 0.00% | 100.00% | 100.00% |
| New fixed I+II model | 0.00% | 100.00% | 100.00% |
| New random-lead model | 0.00% | 100.00% | 100.00% |
| Fixed 12-lead model masked to I+II | 0.00% | 100.00% | 100.00% |

The best validation upper bounds for trusted-negative AFIB-or-AFLT error are
2.60% for fixed I+II, 2.40% for random-lead training, and 3.35% for 12-lead
rescue. All exceed the frozen 2% limit. Therefore every case is escalated. Zero
accepted cases means there is no demonstrated automation; it does not mean the
model demonstrated zero conditional error.

## Exploratory relaxation of AFIB-or-AFLT risk limits

On 2026-09-21, four additional sensitivity settings were defined and fitted using
the existing patient-level policy engine. The set was revised the same day from
3%/5% to 2.5%/3% negative limits to probe nearer the 2% certification boundary;
the 5% policies remain on disk as history but are no longer part of the reported
set. All 12 policies (four settings across three new models) were frozen on
validation before the sensitivity test audits.
The models were not retrained. The original 2% negative / 10% positive policies
remain unchanged. This is a post-hoc analysis on the previously inspected cohort,
not independent confirmation or a newly validated clinical risk limit.

Only AFIB-or-AFLT risk limits changed. The 95% validation confidence level,
Bonferroni correction, 101-point threshold grids, minimum 25 accepted patients,
patient selection, and other diagnosis limits were retained. The separate CRC
analysis remains at 2%. This sensitivity disables bootstrap computation but
retains the audit's exact patient-binomial error intervals. These test intervals
are descriptive and pointwise, not simultaneous across models or scenarios.

The audit includes 1,904 patient representatives, of whom 118 have AFIB-or-AFLT.
All automated outputs in these scenarios are negative calls: the model clears
the patient for this target. They do not clear the ECG of other diagnoses.

| Negative / positive risk limit | Model using I+II | Automated coverage | Expert referral | Wrong / accepted negative calls | Negative-call error (exact 95% CI) | Fraction of actual AFIB/AFLT cases incorrectly cleared |
|---|---|---:|---:|---:|---|---:|
| 2% / 10% (original) | Fixed I+II or random-lead | 0% | 100% | 0/0 | Undefined | 0/118 (all referred) |
| 2% / 40% | Fixed I+II or random-lead | 0% | 100% | 0/0 | Undefined | 0/118 (all referred) |
| 2.5% / 10% | Fixed I+II | 0% | 100% | 0/0 | Undefined | 0/118 (all referred) |
| 2.5% / 10% | Random-lead | 77.15% | 22.85% | 7/1,469 | 0.48% (0.19–0.98%) | 5.93% |
| 3% / 10% | Fixed I+II | 73.53% | 26.47% | 10/1,400 | 0.71% (0.34–1.31%) | 8.47% |
| 3% / 10% | Random-lead | 84.24% | 15.76% | 13/1,604 | 0.81% (0.43–1.38%) | 11.02% |

For these purpose-trained two-lead models, all accepted calls occur at the I+II
stage; twelve-lead acquisition equals the expert-referral percentage. The routed
twelve-lead stage cannot certify further calls among the remaining harder cases.

Raising the positive limit to 40% alongside the 3% negative limit produces exactly
the same primary decisions as 3% / 10% for every model. No scenario automates
positive calls.

The fixed twelve-lead model masked to I+II remains at zero coverage at 2%,
2.5%, and 3%: it accepts no I+II calls, and unlike the earlier 5% sensitivity,
its full twelve-lead rescue stage also automates nothing at these limits. All
patients require twelve-lead acquisition, and all still require expert review.
The absence of rescue automation here is a property of these tighter limits,
not evidence about full-lead performance.

The previously available relaxation of positive-call error to 40% does not
address the binding negative-call constraint. It also does not produce positive
automation: the current validation stages have too few positive calls at the
permitted thresholds to satisfy the minimum accepted count and risk bounds.

The cliff between 2.5% and 3% illustrates a certification bottleneck. Validation
has 1,942 patient representatives: 971 for probability calibration, 485 for
reduced-stage risk fitting, and 486 for rescue-stage risk fitting. With the
current correction, certifying a 2% bound requires at least 446 error-free
accepted patients in a branch. The selected fixed-I+II 3% threshold has 342
accepted validation patients and zero errors, but its corrected upper bound is
2.60% — so it clears 3% while still failing 2.5%. Zero observed errors in that
sample therefore cannot certify the original 2% requirement, and the fixed
model stays fully abstained at 2.5% while the random-lead model already
automates 77.15% there.

Increasing the negative limit from 2.5% to 3% adds 1,400 automated clearances
for fixed I+II (from zero) with 10 additional missed AFIB/AFLT cases, and 135
clearances for random-lead (1,469 to 1,604) with 6 additional misses (7 to 13).
Thus, coverage improves substantially, but the extra clearances have a
meaningful cost. The negative-call error percentage uses all cleared patients
as its denominator; the missed-case percentage uses the 118 actual cases. A low
value for the former does not establish adequate disease detection.

These results support further study of the 2.5–3% boundary and more efficient
use of independent calibration data; they do not establish that 2.5% or 3% is
clinically acceptable or that the original 2% target should be replaced. Any
selected revision needs clinical justification and evaluation on untouched data.

Full scenario results, frozen protocols, validation policies, provenance checks,
and reproduction commands are in
[`artifacts/comparison/runpod-s42-relaxed-risk/report.md`](artifacts/comparison/runpod-s42-relaxed-risk/report.md).

## Comparison with published literature

### Reduced leads can work, but adequacy depends on the diagnosis

The PhysioNet/Computing in Cardiology Challenge evaluated models across 12-,
6-, 4-, 3-, and 2-lead inputs. Its main analysis found that average Challenge-
metric changes between 12 and 2 leads were generally below 2%, while changes
varied substantially by diagnosis and source database
([`reyna2021_reduced_leads`](REFERENCES.md#reyna2021_reduced_leads),
[`reyna2022_multilead_challenge`](REFERENCES.md#reyna2022_multilead_challenge)).

This project agrees with the diagnosis-specific part of that evidence: fixed
I+II improves MI and conduction-disorder ranking but reduces AFIB-or-AFLT
ranking. The numerical changes are not directly comparable because the
Challenge score and this project's macro-AUROC measure different things.

### The two-lead results are plausible but not state of the art evidence

The fixed I+II superclass macro-AUROC of 0.8673 is slightly above the historical
two-lead result of 0.8586. Published PTB-XL diagnostic benchmarks report values
around 0.93 in some strong 12-lead configurations
([`strodthoff2021_ptbxl_benchmark`](REFERENCES.md#strodthoff2021_ptbxl_benchmark)).
That value is useful context rather than a head-to-head target because the label
sets, architectures, folds, preprocessing, and averaging schemes differ.

The results therefore support the feasibility of reduced-lead diagnosis, but
they do not establish a new state of the art. The direct within-project
comparisons are more trustworthy than comparing isolated AUROC values between
papers.

### Masking a 12-lead model is an input mismatch

The new fixed 12-lead model falls from 0.8954 superclass macro-AUROC with its
intended input to 0.6409 after masking it to I+II. Reduced-lead literature that
reports useful performance trains or fine-tunes models for the available input
([`saglietto2024_single_twelve`](REFERENCES.md#saglietto2024_single_twelve),
[`reyna2022_multilead_challenge`](REFERENCES.md#reyna2022_multilead_challenge)).

The masked result consequently measures robustness to ten unexpectedly missing
leads. It does not estimate how well a properly trained two-lead model can work.
The fixed I+II result of 0.8673 is the relevant estimate for that question.

### High AUROC does not guarantee safe automation

AFIB-or-AFLT AUROC remains between 0.9287 and 0.9483 for the purpose-trained new
reduced-lead models, yet neither model satisfies the 2% validation risk bound at
nonzero coverage. AUROC measures ranking across all thresholds; the policy asks
a harder question: whether a particular subset can be accepted with a
statistically supported error limit.

Selective-prediction research treats abstention as a valid safety response but
requires reporting both the error among accepted cases and the accepted
coverage ([`geifman2017_selective`](REFERENCES.md#geifman2017_selective),
[`feng2023_selective_prediction_sets`](REFERENCES.md#feng2023_selective_prediction_sets)).
The current 0% coverage is therefore a safe failure to automate, not a successful
automation result.

## Evidence limitations

- Only seed 42 has been evaluated. Seeds 43 and 44 are needed to measure
  training instability.
- The historical and new runs differ in development folds, checkpoint
  selection, batch size, and auxiliary loss. Their difference cannot be assigned
  to one change without controlled ablations.
- PTB-XL fold 10 has already been inspected repeatedly, so these comparisons are
  exploratory. Confirmation requires an untouched external cohort.
- Masked hospital ECG leads do not reproduce the filtering, electrode placement,
  noise, or acquisition context of a real portable two-lead device.
- Some subgroup estimates are too sparse for safety claims. The under-40
  AFIB-or-AFLT test subgroup, for example, has only one positive record.
- The conformal sensitivity analysis controls a different, unconditional
  any-error quantity. It must not be presented as validation of the primary
  conditional trusted-negative policy.

## Research conclusion and next steps

The current package supports a rigorous feasibility result: reduced-lead model
quality depends on the diagnosis and training input, and strong discrimination
does not guarantee useful selective coverage at a stringent clinical risk
limit. The prespecified primary automation objective was not achieved.

Before a stronger comparative or clinical claim:

1. Complete seeds 43 and 44 and report across-seed variation.
2. Run controlled ablations for the fold split, checkpoint rule, auxiliary loss,
   and other training differences.
3. Freeze the complete analysis before evaluating an untouched external cohort.
4. Justify the external-validation sample size for calibration, discrimination,
   and utility precision
   ([`riley2021_external_validation_size`](REFERENCES.md#riley2021_external_validation_size)).
5. Test recordings from the intended two-lead device before making a device or
   form-factor claim.
6. Define action thresholds or error costs with clinical stakeholders before
   changing the 2% limit
   ([`vickers2006_decision_curve`](REFERENCES.md#vickers2006_decision_curve)).
7. Follow prospective clinical workflow guidance before any deployment study
   ([`vasey2022_decide_ai`](REFERENCES.md#vasey2022_decide_ai)).
