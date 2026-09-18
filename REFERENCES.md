# References

This file tracks the evidence used in `AFIB_EVALUATION_PROTOCOL.md`,
`ALL_LABEL_EVALUATION_PROTOCOL.md`, and `USE_CONTRACT.md`. Links point to the
paper, publisher, PubMed record, or official dataset page. Accessed 2026-09-07.
Citation keys are stable identifiers for project documents.

## ECG datasets, reduced leads, and model studies

### `wagner2020_ptbxl`

Wagner P, Strodthoff N, Bousseljot R-D, et al. PTB-XL, a large publicly
available electrocardiography dataset. *Scientific Data*. 2020;7:154.
[doi:10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6).

Used for: PTB-XL cohort provenance, signal characteristics, labels, and the
official patient-respecting fold structure.

### `strodthoff2021_ptbxl_benchmark`

Strodthoff N, Wagner P, Schaeffter T, Samek W. Deep Learning for ECG Analysis:
Benchmarks and Insights from PTB-XL. *IEEE Journal of Biomedical and Health
Informatics*. 2021;25(5):1519-1528.
[doi:10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989).

Used for: PTB-XL benchmarking conventions, strong convolutional baselines,
hidden stratification, uncertainty, and the need for structured comparison.

### `reyna2021_reduced_leads`

Reyna MA, Sadr N, Perez Alday EA, et al. Will Two Do? Varying Dimensions in
Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021.
*Computing in Cardiology*. 2021;48:1-4.
[doi:10.23919/CinC53138.2021.9662687](https://doi.org/10.23919/CinC53138.2021.9662687).
Official dataset: [PhysioNet Challenge 2021 v1.0.3](https://physionet.org/content/challenge-2021/1.0.3/).

Used for: canonical 12-, 6-, 4-, 3-, and 2-lead comparisons and evidence that
reduced-lead performance must be evaluated separately across diagnoses and data
sources.

### `perezalday2020_multisource`

Perez Alday EA, Gu A, Shah AJ, et al. Classification of 12-lead ECGs: the
PhysioNet/Computing in Cardiology Challenge 2020. *Physiological Measurement*.
2020;41(12):124003.
[doi:10.1088/1361-6579/abc960](https://doi.org/10.1088/1361-6579/abc960).

Used for: multi-institution ECG evaluation, source heterogeneity, label
harmonization, reproducibility, and observed degradation on hidden test data.

### `zheng2020_chapman`

Zheng J, Zhang J, Danioko S, et al. A 12-lead electrocardiogram database for
arrhythmia research covering more than 10,000 patients. *Scientific Data*.
2020;7:48.
[doi:10.1038/s41597-020-0386-x](https://doi.org/10.1038/s41597-020-0386-x).

Used for: a candidate source-held-out external rhythm/AF evaluation cohort.

### `hannun2019_single_lead`

Hannun AY, Rajpurkar P, Haghpanahi M, et al. Cardiologist-level arrhythmia
detection and classification in ambulatory electrocardiograms using a deep
neural network. *Nature Medicine*. 2019;25(1):65-69.
[doi:10.1038/s41591-018-0268-3](https://doi.org/10.1038/s41591-018-0268-3).

Used for: precedent for deep rhythm classification from single-lead ambulatory
ECGs and independent expert-adjudicated testing. It is not treated as a directly
comparable benchmark because the device, duration, population, and labels differ.

### `ribeiro2020_12lead`

Ribeiro AH, Ribeiro MH, Paixao GMM, et al. Automatic diagnosis of the 12-lead
ECG using a deep neural network. *Nature Communications*. 2020;11:1760.
[doi:10.1038/s41467-020-15432-4](https://doi.org/10.1038/s41467-020-15432-4).

Used for: precedent for large-scale 12-lead deep-learning development and
independent evaluation.

## AFIB target and clinical-use boundary

### `joglar2024_af_guideline`

Joglar JA, Chung MK, Armbruster AL, et al. 2023 ACC/AHA/ACCP/HRS Guideline for
the Diagnosis and Management of Atrial Fibrillation. *Circulation*.
2024;149:e1-e156.
[doi:10.1161/CIR.0000000000001193](https://doi.org/10.1161/CIR.0000000000001193).

Used for: the ECG characteristics and documentation of AF, monitoring options,
and the requirement that an initial AF diagnosis be visually confirmed by a
clinician. This is a clinical guideline, not a model-validation study.

## Calibration, selective prediction, and risk control

### `guo2017_calibration`

Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks.
In: *Proceedings of the 34th International Conference on Machine Learning*.
PMLR 70; 2017:1321-1330.
[Paper](https://proceedings.mlr.press/v70/guo17a.html).

Used for: the need to evaluate neural-network calibration and post-hoc scaling
on data separate from model fitting.

### `vancalster2019_calibration`

Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration:
the Achilles heel of predictive analytics. *BMC Medicine*. 2019;17:230.
[doi:10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7).

Used for: calibration plots, calibration-in-the-large and slope, external
calibration assessment, and avoiding reliance on a single calibration summary.

### `geifman2017_selective`

Geifman Y, El-Yaniv R. Selective Classification for Deep Neural Networks. In:
*Advances in Neural Information Processing Systems 30*. 2017.
[Paper](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).

Used for: selective classification, rejection/abstention, and risk-coverage
evaluation of a trained neural network.

### `angelopoulos2024_crc`

Angelopoulos AN, Bates S, Fisch A, Lei L, Schuster T. Conformal Risk Control.
In: *International Conference on Learning Representations*. 2024.
[Paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html).

Used for: conformal control of expected bounded monotone loss and the explicit
assumptions behind a finite-sample risk-control sensitivity analysis.

### `clopper1934_binomial`

Clopper CJ, Pearson ES. The Use of Confidence or Fiducial Limits Illustrated in
the Case of the Binomial. *Biometrika*. 1934;26(4):404-413.
[doi:10.1093/biomet/26.4.404](https://doi.org/10.1093/biomet/26.4.404).

Used for: the conservative exact binomial bound in the current fixed-grid
threshold baseline, and for clarifying that its sampling unit must be defensible.

## Performance comparison, utility, and sample size

### `delong1988_correlated_auc`

DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more
correlated receiver operating characteristic curves: a nonparametric approach.
*Biometrics*. 1988;44(3):837-845.
[PubMed](https://pubmed.ncbi.nlm.nih.gov/3203132/).

Used for: paired comparison of correlated ROC curves, with the caveat that
patient clustering must still be handled when patients have repeated ECGs.

### `vickers2006_decision_curve`

Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating
prediction models. *Medical Decision Making*. 2006;26(6):565-574.
[doi:10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361).

Used for: net-benefit evaluation when clinically defensible action thresholds
or relative harms are available.

### `riley2021_external_validation_size`

Riley RD, Debray TPA, Collins GS, et al. Minimum sample size for external
validation of a clinical prediction model with a binary outcome. *Statistics
in Medicine*. 2021;40(19):4230-4251.
[doi:10.1002/sim.9025](https://doi.org/10.1002/sim.9025).

Used for: precision-based external-validation sample-size planning across
calibration, discrimination, and clinical-utility measures.

## Reporting, bias appraisal, and clinical evaluation

### `collins2024_tripod_ai`

Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance
for reporting clinical prediction models that use regression or machine
learning methods. *BMJ*. 2024;385:e078378.
[doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378).

Used for: transparent reporting of development and evaluation data, model
specification, fairness/subgroups, protocols, and open-science artifacts.

### `moons2025_probast_ai`

Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of
bias, and applicability assessment tool for prediction models using regression
or artificial intelligence methods. *BMJ*. 2025;388:e082505.
[doi:10.1136/bmj-2024-082505](https://doi.org/10.1136/bmj-2024-082505).

Used for: prospective quality checks on participants/data sources, predictors,
outcome labels, analysis, applicability, and fairness.

### `vasey2022_decide_ai`

Vasey B, Nagendran M, Campbell B, et al. Reporting guideline for the early-stage
clinical evaluation of decision support systems driven by artificial
intelligence: DECIDE-AI. *Nature Medicine*. 2022;28:924-933.
[doi:10.1038/s41591-022-01772-9](https://doi.org/10.1038/s41591-022-01772-9).

Used for: prospective silent and assisted clinical workflow evaluation,
including human factors, safety, and system-level performance.

## Historical / legacy evaluations

The following are frozen history. Cite them as background only. Do not refit,
retune, or upgrade them in place. Future fits use `protocol/use_contract_v1.json`
and `USE_CONTRACT.md`:

- Protocols: `protocol/afib_v1.json`, `protocol/all_labels_v1.json`,
  `protocol/afib_positive_exploratory_v1.json` (frozen; `afib_v1` remains the
  primary AFIB result, the positive-40% run is hypothesis-generating only).
- Policies: `artifacts/selective/afib-validation-policy`,
  `artifacts/selective/afib-positive-exploratory-risk40-v2`,
  `artifacts/selective-all/all-labels-baseline-v1`,
  `artifacts/selective-all/all-labels-baseline-v2`.
- Audits: `artifacts/audit/afib-locked-test-audit-v1`,
  `artifacts/audit/afib-locked-test-audit-v2`,
  `artifacts/audit/afib-positive-exploratory-test-v1`,
  `artifacts/audit/afib-positive-exploratory-test-v2`,
  `artifacts/audit-all/all-labels-test-v1`,
  `artifacts/audit-all/all-labels-test-v2`,
  `artifacts/audit-all/all-labels-test-v3` (post-hoc; test fold already inspected).
- To use the corrected routed method, refit on validation into a new output
  directory; never edit a frozen bundle.

## Maintenance notes

- Add a citation key here before using a new source in a project document.
- Record the specific project decision supported by each source; do not list
  papers merely because they are related.
- Prefer original studies, official datasets, consensus guidelines, and primary
  methods papers over secondary summaries.
- Recheck clinical guidelines and reporting standards before a protocol is
  registered or a manuscript is submitted.
