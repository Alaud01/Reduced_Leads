from pathlib import Path
import pandas as pd
import json
R=Path('artifacts/comparison/runpod-s42')
m=pd.read_csv(R/'all_model_metrics.csv');p=pd.read_csv(R/'all_policy_metrics.csv');c=pd.read_csv(R/'crc_comparison.csv')
lines=['# Runpod seed-42 comparison','',
'All three new models completed 50 epochs. Best checkpoints, label schemas, validation-log selection scores, and local waveform fingerprints were verified. All new validation policies were frozen before exporting the new test predictions. Risk limits were unchanged.','',
'Comparisons use 2,198 test ECGs for descriptive model metrics and 1,904 protocol-selected patient ECGs for primary policy metrics. These are exploratory results on an already inspected cohort, with one training seed.','',
'## Two-lead AFIB-or-AFLT discrimination','',
'Raw-score AUROC and average precision assess ranking. Brier assesses raw probability error (lower is better).','',
'| Approach | AUROC | Average precision | Raw Brier |','|---|---:|---:|---:|']
for _,r in m[(m.diagnosis=='AFIB-or-AFLT')&(m.lead_set=='2-lead')].iterrows():
 lines.append(f'| {r.approach} | {r.auroc:.4f} | {r.average_precision:.4f} | {r.brier:.4f} |')
lines+=['','## Primary frozen triage policy','',
'All new approaches use the same newly trained fixed twelve-lead model for rescue. The previous model uses its historical twelve-lead predictions.','',
'| Approach | Automated | Twelve-lead acquisition | Expert referral | Accepted negatives | Wrong negatives | Negative error upper 95% |',
'|---|---:|---:|---:|---:|---:|---:|']
for _,r in p[p.diagnosis=='AFIB-or-AFLT'].iterrows():
 upper='Undefined' if pd.isna(r.trusted_negative_error_exact95_upper) else f'{r.trusted_negative_error_exact95_upper:.2%}'
 lines.append(f'| {r.approach} | {r.automated_coverage:.2%} | {r.twelve_lead_acquisition_rate:.2%} | {r.expert_referral_rate:.2%} | {r.trusted_negative_records} | {r.trusted_negative_errors} | {upper} |')
lines+=['','Zero accepted predictions means no demonstrated automation, not zero conditional error. Test intervals are descriptive and do not select or modify policies.','',
'## Key diagnosis AUROCs with I+II','',
'| Approach | AFIB-or-AFLT | MI | STTC | CD | NORM (superclass) |','|---|---:|---:|---:|---:|---:|']
for name,g in m[m.lead_set=='2-lead'].groupby('approach',sort=False):
 vals=[]
 for group,label in [('rhythm','AFIB-or-AFLT'),('superclass','MI'),('superclass','STTC'),('superclass','CD'),('superclass','NORM')]:
  vals.append(f'{g[(g.diagnosis_group==group)&(g.diagnosis==label)].iloc[0].auroc:.4f}')
 lines.append('| '+name+' | '+' | '.join(vals)+' |')
lines+=['','## Broad model performance','',
'Unweighted diagnosis macro means; rare-label estimates are unstable and these are descriptive comparisons.','',
'| Approach | Lead set | Superclass AUROC | Subcode AUROC | Rhythm AUROC (includes composite) |','|---|---|---:|---:|---:|']
for (name,lead),g in m.groupby(['approach','lead_set'],sort=False):
 means=g.groupby('diagnosis_group').auroc.mean()
 lines.append(f'| {name} | {lead} | {means.superclass:.4f} | {means.subcode:.4f} | {means.rhythm:.4f} |')
lines+=['','## Patient-paired primary AUROC contrasts','',
'One ECG per patient, 2,000 paired bootstrap replicates, descriptive percentile intervals. These do not include training-seed variability or multiplicity correction.','',
'| Approach | Reference | AUROC difference | 95% interval |','|---|---|---:|---:|']
for _,r in pd.read_csv(R/'primary_paired_auroc.csv').iterrows():
 lines.append(f'| {r.approach} | {r.reference} | {r.delta:+.4f} | [{r.lower:+.4f}, {r.upper:+.4f}] |')
lines+=['','## Frozen calibration on patient representatives','',
'| Approach | Raw Brier | Calibrated Brier |','|---|---:|---:|']
for _,r in pd.read_csv(R/'primary_calibration.csv').iterrows():
 lines.append(f'| {r.approach} | {r.raw_brier:.4f} | {r.calibrated_brier:.4f} |')
lines+=['','## Separate conformal sensitivity','',
'This is standalone-model unconditional patient any-error risk, not conditional error among accepted predictions and not the two-stage policy. The historical checkpoint lacks independent calibration, so its conformal calculation is nominal only.','',
'| Approach | Lead set | Record coverage | Patient any-error rate | Independent calibration |','|---|---|---:|---:|---|']
for _,r in c[c.diagnosis=='AFIB-or-AFLT'].iterrows():
 lines.append(f'| {r.approach} | {r.lead_set} | {r.automated_coverage:.2%} | {r.unconditional_patient_error:.2%} | {r.independent_calibration_verified} |')
lines+=['','## Interpretation boundaries and next steps','',
'- The historical and new experiments differ in development split, checkpoint selection, auxiliary loss and training; their differences cannot be attributed to one change alone.',
'- Fixed I+II versus the new random-lead model is the closest matched training comparison: same data, architecture, seed, objective and selection lead.',
'- The masked twelve-lead baseline was selected on twelve-lead performance, whereas two-lead models were selected on two-lead performance.',
'- Retain the frozen limits and all results, including zero coverage. Do not choose new thresholds using test results.',
'- Repeat the already specified comparison with seeds 43 and 44 and use an untouched external cohort for confirmation.',
'- Per-target metrics, calibration, subgroup and repeat-ECG sensitivity tables are included as companion CSV files. Sparse subgroup results are descriptive, not subgroup safety guarantees.','']
# Add the interpretation after the prespecified comparison has completed.
summary = [
'## Main findings', '',
'- Fixed I+II improves the two-lead superclass macro-AUROC from 0.8586 to 0.8673, with improvements for MI (0.8190 to 0.8409) and CD (0.8351 to 0.8496). These are descriptive, single-seed gains.',
'- AFIB-or-AFLT ranking regresses: previous 0.9582, fixed I+II 0.9287, new random-lead model 0.9483. The patient-paired interval for fixed I+II versus the previous model excludes zero, but does not account for training-seed variation.',
'- The new random-lead model slightly improves AFIB-or-AFLT average precision (0.6690 to 0.6799) and calibrated patient-level Brier error (0.0338 to 0.0309), despite lower AUROC.',
'- No approach achieves nonzero automation for the primary conditional-error endpoint. The best validation negative-error upper bounds are 2.60% for fixed I+II, 2.40% for random-lead training and 3.35% for full-lead rescue, above the 2% limit.',
'- Among the 29 eligible diagnosis policies, the previous patient-level run automated at least some cases for three targets (PACE, CRBBB, STACH). The two new reduced-lead approaches do so only for PACE; the masked twelve-lead approach does so for none. This is a per-diagnosis count, not patient-wide automation.',
'- The new full-lead rescue model also regresses in superclass macro-AUROC: 0.9128 to 0.8954. Merely masking this model is a poor reduced-lead strategy.',
'- Overall: useful diagnosis-specific improvements, but no broad improvement and no improvement in the primary triage endpoint. Keep the historical model as a reference, not as an independently calibrated deployment candidate.', '',
'![Comparison figure](comparison.png)', '',
]
lines = lines[:5] + [''] + summary + lines[5:]
lines += ['## Subgroup and repeat-record checks', '',
          'Age/sex and signal-quality tables are in `subgroup_comparison.csv`; repeat-record analyses are in `repeat_ecg_comparison.csv`. For AFIB-or-AFLT in patients aged 80+, AUROC is 0.8824 previously, 0.8277 for fixed I+II and 0.8595 for random-lead training (340 ECGs, 71 positive records). These are descriptive subgroup results. The under-40 group has only one positive record, so its apparent AUROC differences are not reliable.', '',
          'Twelve-lead rows under each new approach refer to the same shared fixed12 rescue model, not to the fixed-I+II model receiving twelve leads. The conformal CSV includes exact descriptive patient-error intervals; point estimates near 2% must not be interpreted as certain satisfaction of that limit.', '']
(R/'report.md').write_text('\n'.join(lines))
print(R/'report.md')
