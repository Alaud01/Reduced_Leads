# Use contract v1 — frozen research contract (item 1)

Status: frozen research contract. Not a clinical-safety, device-equivalence, or regulatory claim.
Machine-readable: `protocol/use_contract_v1.json`.
Prior generic contract it replaces for future work: `protocol/all_labels_v1.json` (`10%` positive / `2%` negative for all 61 labels).
Historical protocols `protocol/afib_v1.json` and `protocol/all_labels_v1.json` remain frozen and unchanged.

## 1. Why a new contract was needed

Re-analysis of frozen artifacts (no refitting, CPU-only):

* `artifacts/audit-all/all-labels-test-v3/policy_metrics.csv.gz`, 2-lead under generic `10%/2%`:
  * `MI (476 pos pts): 0.0% coverage / 100% referral`
  * `STTC (475): 0.0% / 100%`
  * `CD (439): 0.0% / 100%`
  * `HYP (249): 0.0% / 100%`
  * `IMI (240), ASMI (199), LAFB (150), IRBBB (106), LVH (202), AMI (34): all 0.0% / 100%`
  * This is not sample size — these are the largest subcodes. It is limit infeasibility.
* Same run:
  * `AFIB (130 pos pts): 86.4% coverage, 0.7% error, 0 trusted positives, sensitivity 0.0` — rule-out only.
  * `SR (1506 pos pts): 63.2% coverage at 8.35% error (1389 trusted positives, 116 errors)`.
  * `NORM superclass: 20.0% at 5.23% error; subcode NORM: 20.1% at 5.90% error`.
  * Model: `MI 0.909→0.819 AUROC 12→2-lead; CD 0.897→0.835; AFIB 0.967→0.957` (`model_metrics.csv.gz`).
* Count boundary (`label_summary.csv.gz`): 29/61 eligible at `≥25` pos/neg validation patients. `AFLT` alone ineligible (test 7 records / 6 patients; val 7 records). Composite below fixes eligibility without test fitting.

Interpretation: generic limits conflate three different harms — miss disease, false alarm for disease, and falsely asserting normality — and force abstention on exactly the `PROJECT.md` emphasis targets (MI/ST-T, conduction).

## 2. Intended use (frozen)

* Population: adults represented by PTB-XL receiving a 10-s resting ECG in primary-care-like triage. No ambulatory, pediatric, exercise, or wearable generalization.
* Input: 10 s @ 100 Hz, standard placement masked to `LEAD_SUBSETS` in `src/config.py`. Masked hospital ECGs ≠ portable device. New devices must prespecify filtering, gain, missing-signal handling.
* Reference: PTB-XL SCP label retrospectively; ≥2 blinded readers with disagreement process prospectively.
* Units: ECG record operates; patient infers (bootstrap and guarantees by patient).
* Role: triage decision support. First AF requires visual confirmation.
* Actions:
  * `trust_reduced` — show reduced-lead call, log, no human for that diagnosis.
  * `obtain_12_lead` — require full 12-lead, re-run 12-lead stage.
  * `trust_12_lead` — show 12-lead-stage call.
  * `expert_review` — queue to reader, block automation. Referral is cost, not failure.
  * `trust_positive` for disease = assert disease. `trust_positive` for NORM/SR = assert normality. Not interchangeable.

## 3. Targets

* Primary: `AFIB-or-AFLT` composite, `target = 1 if AFIB=1 OR AFLT=1`.
  * Test: AFIB 152 rec / 130 pts + AFLT 7 / 6 → composite 157 / 134. Val: 151 + 7 → 156 / 125. Captures 2 dual-labeled records.
  * Confounders for error analysis: AFLT, SR, SVTAC/PSVT, SARRH/SVARR, extra beats, pacemaker. AFIB∩SR = 0 records on test (mutually exclusive here).
  * Composite probability scoring (e.g. max of calibrated probs) must be frozen before any validation fit. This contract freezes the target, not the scorer.
* Secondary: MI-ischemia, conduction, other rhythms — reported per-label, never pooled.
* Assert-normal separate: `superclass/NORM`, `subcode/NORM`, `rhythm/SR`. Apply rule-out stringency to their trusted positives. Never aggregate with disease coverage. SR/NORM current error (8.35% / ~5-6%) FAILS the bar below.

## 4. Diagnosis-specific risk limits (research placeholders)

Require clinical harm sign-off before any deployment claim.

| Group | Applies to | Trusted-neg max | Trusted-pos max |
|---|---|---:|---:|
| AFIB-or-AFLT rule-out (primary) | composite | 0.02 | — |
| AFIB-or-AFLT rule-in (secondary) | composite | — | 0.10 anchor; 0.40 val/external sensitivity only |
| MI/STTC ischemia | MI, STTC, IMI, ASMI, ALMI, ILMI, AMI, ISC_, ISCAL | 0.01 | 0.05 |
| Conduction | CD, 1AVB, CLBBB, CRBBB, IRBBB, IVCD, LAFB | 0.02 | 0.10 |
| Assert-normal | NORM (super+sub), SR | — (neg N/A) | 0.02 on trusted positives |

Rationale: ischemia stricter both ways (miss and false cath-lab both severe); conduction retains generic as anchor but must not be pooled (CLBBB 98% vs IRBBB 0% under same limits); rule-in stays descriptive (10% gave 0 auto positives; 40% gave 80 auto positives at 17.5% error on `afib-positive-exploratory-test-v2` — neither validated).

## 5. Single primary

2-lead `AFIB-or-AFLT` `trusted_negative_error ≤ 0.02` (patient-cluster bootstrap). Secondaries: coverage, referral, acquisition rate, AUROC/AP, Brier, calibration. All other leads/diagnoses/subgroups/rule-in are descriptive, no significance.

## 6. Fitting / audit boundary

* Fit calibration + scorer + thresholds on fold 9 (or external-recalibration split) only. Five-fold patient-grouped cross-fit for development; freeze.
* Audit on fold 10 or external source fit-free: verify checkpoint SHA, schema, complete exports, lead sets, protocol hash, full target enumeration.
* `≥25` pos/neg validation patients to fit (stability floor, not clinical adequacy). External N via Riley precision.
* Record-level Clopper-Pearson-Bonferroni over 101-grid is current baseline with known repeat-ECG limitation — require one-ECG-per-patient and CRC sensitivities (roadmap items 3) before confirmatory claims.
* Any audit-driven change → `use_contract_v2.json` + new untouched cohort. This file is now frozen.

## Reproduction (descriptive only, no refit)

```text
python3 -c "import pandas as pd; print(pd.read_csv('artifacts/audit-all/all-labels-test-v3/policy_metrics.csv.gz').query(\"lead_set=='2-lead'\")[['diagnosis','positive_patients','automated_coverage','expert_referral_rate']].to_string())"
```

No `src/selective.py` or `src/audit_all_labels.py` rerun was performed for this contract. Next code change (roadmap): implement composite scorer + per-group limit plumbing on validation only.
