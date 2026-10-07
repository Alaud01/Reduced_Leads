# Exploratory AFIB-or-AFLT risk-limit sensitivity

Date: 2026-09-21. Seed 42; previously inspected PTB-XL test fold.

All 12 new policies were fitted on validation and frozen before this audit. The original 2%/10% policies are unchanged. These scenarios relax only AFIB-or-AFLT limits; other diagnosis limits, 95% validation confidence, 101-point grids, Bonferroni correction, patient representatives, minimum 25 accepted patients, and model checkpoints are unchanged.

Test summaries use 1,904 patient representatives, including 118 positive AFIB-or-AFLT cases. Bootstrap was disabled for this descriptive analysis; the existing audit still computes exact 95% binomial accepted-error intervals. Intervals are pointwise descriptive, not simultaneous over scenarios. No scenario is selected for deployment.

Negative-call risk means wrong negative calls / all accepted negative calls; missed-positive fraction means wrong negative calls / all 118 true cases. These denominators answer different questions.

| Model | Negative / positive limit | Automated | 12-lead acquisition | Expert referral | Wrong / accepted negatives | Negative error (exact 95% CI) | Wrong / accepted positives | Missed / all positive cases |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Fixed I+II | 2% / 10% | 0.00% | 100.00% | 100.00% | 0/0 | Undefined | 0/0 | 0/118 (0.00%) |
| Random-lead | 2% / 10% | 0.00% | 100.00% | 100.00% | 0/0 | Undefined | 0/0 | 0/118 (0.00%) |
| 12-lead model masked to I+II | 2% / 10% | 0.00% | 100.00% | 100.00% | 0/0 | Undefined | 0/0 | 0/118 (0.00%) |
| Fixed I+II | 2% / 40% | 0.00% | 100.00% | 100.00% | 0/0 | Undefined | 0/0 | 0/118 (0.00%) |
| Random-lead | 2% / 40% | 0.00% | 100.00% | 100.00% | 0/0 | Undefined | 0/0 | 0/118 (0.00%) |
| 12-lead model masked to I+II | 2% / 40% | 0.00% | 100.00% | 100.00% | 0/0 | Undefined | 0/0 | 0/118 (0.00%) |
| Fixed I+II | 2.5% / 10% | 0.00% | 100.00% | 100.00% | 0/0 | Undefined | 0/0 | 0/118 (0.00%) |
| Random-lead | 2.5% / 10% | 77.15% | 22.85% | 22.85% | 7/1469 | 0.48% (0.19%–0.98%) | 0/0 | 7/118 (5.93%) |
| 12-lead model masked to I+II | 2.5% / 10% | 0.00% | 100.00% | 100.00% | 0/0 | Undefined | 0/0 | 0/118 (0.00%) |
| Fixed I+II | 3% / 10% | 73.53% | 26.47% | 26.47% | 10/1400 | 0.71% (0.34%–1.31%) | 0/0 | 10/118 (8.47%) |
| Random-lead | 3% / 10% | 84.24% | 15.76% | 15.76% | 13/1604 | 0.81% (0.43%–1.38%) | 0/0 | 13/118 (11.02%) |
| 12-lead model masked to I+II | 3% / 10% | 0.00% | 100.00% | 100.00% | 0/0 | Undefined | 0/0 | 0/118 (0.00%) |
| Fixed I+II | 3% / 40% | 73.53% | 26.47% | 26.47% | 10/1400 | 0.71% (0.34%–1.31%) | 0/0 | 10/118 (8.47%) |
| Random-lead | 3% / 40% | 84.24% | 15.76% | 15.76% | 13/1604 | 0.81% (0.43%–1.38%) | 0/0 | 13/118 (11.02%) |
| 12-lead model masked to I+II | 3% / 40% | 0.00% | 100.00% | 100.00% | 0/0 | Undefined | 0/0 | 0/118 (0.00%) |

## Reproduction and artifacts

Run from repository root:

```bash
.venv/bin/python artifacts/comparison/runpod-s42-relaxed-risk/run_analysis.py
.venv/bin/python artifacts/comparison/runpod-s42-relaxed-risk/summarize.py
```

- `analysis_plan.json`: complete scenario list and exploratory status.
- `protocols/`: separately versioned sensitivity contracts.
- `policies/`: validation fits, threshold curves, feasibility and manifests.
- `audits/`: frozen-policy test decisions, metrics and source-hash verification.
- `primary_comparison.csv`: all scenarios and exact test error intervals.
- `verification.json`: all nonprimary policies and selected test metrics match baseline.

The separate unconditional CRC sensitivity remains at 2% and does not replace these conditional-risk results. All new approaches use the same fixed 12-lead rescue model, fitted on the population routed to that stage.
