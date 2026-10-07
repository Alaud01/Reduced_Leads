"""Summarize all frozen sensitivity scenarios; never select on test results."""
import json
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent / 'runpod-s42'
MODELS = ['fixed2-s42', 'random2-s42', 'fixed12-s42']
SCENARIOS = ['neg02-pos40', 'neg025-pos10', 'neg03-pos10', 'neg03-pos40']
rows = []
checks = []
for scenario in ['baseline', *SCENARIOS]:
    for model in MODELS:
        root = BASE if scenario == 'baseline' else OUT
        name = model if scenario == 'baseline' else f'{scenario}--{model}'
        for folder in ['policies', 'audits']:
            manifest = json.loads((root / folder / name / 'manifest.json').read_text())
            assert manifest['status'] == 'complete', (scenario, model, folder)
        metrics = pd.read_csv(root / 'audits' / name / 'policy_metrics.csv.gz')
        primary = metrics[metrics.diagnosis.eq('AFIB-or-AFLT') & metrics.lead_set.eq('2-lead')]
        assert len(primary) == 1
        row = primary.iloc[0].to_dict()
        row.update(scenario=scenario, model=model)
        assert row['n_records'] == row['n_patients'] == 1904
        row['missed_fraction_of_all_positive_patients'] = row['trusted_negative_errors'] / row['positive_records']
        rows.append(row)
        if scenario == 'baseline':
            continue
        original = json.loads((BASE / 'policies' / model / 'policy.json').read_text())
        current = json.loads((OUT / 'policies' / name / 'policy.json').read_text())
        unchanged = lambda p: [t for t in p['targets'] if t['diagnosis'] != 'AFIB-or-AFLT']
        assert unchanged(original) == unchanged(current), f'Unexpected nonprimary policy change: {name}'
        base_metrics = pd.read_csv(BASE / 'audits' / model / 'policy_metrics.csv.gz')
        cols = ['diagnosis_group', 'diagnosis', 'lead_set', 'automated_coverage',
                'trusted_negative_records', 'trusted_negative_errors',
                'trusted_positive_records', 'trusted_positive_errors']
        pd.testing.assert_frame_equal(
            base_metrics[base_metrics.diagnosis.ne('AFIB-or-AFLT')][cols].reset_index(drop=True),
            metrics[metrics.diagnosis.ne('AFIB-or-AFLT')][cols].reset_index(drop=True))
        checks.append({'run': name, 'all_nonprimary_policies_unchanged': True,
                       'all_nonprimary_selected_test_metrics_unchanged': True})

summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'primary_comparison.csv', index=False)
(OUT / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
cols = ['scenario', 'model', 'automated_coverage', 'twelve_lead_acquisition_rate',
        'expert_referral_rate', 'trusted_negative_records', 'trusted_negative_errors',
        'trusted_negative_error', 'trusted_negative_error_exact95_lower',
        'trusted_negative_error_exact95_upper', 'trusted_positive_records',
        'trusted_positive_errors', 'positive_records', 'missed_fraction_of_all_positive_patients']
print(summary[cols].to_string(index=False))

lines = ['# Exploratory AFIB-or-AFLT risk-limit sensitivity', '',
         'Date: 2026-09-21. Seed 42; previously inspected PTB-XL test fold.', '',
         'All 12 new policies were fitted on validation and frozen before this audit. '
         'The original 2%/10% policies are unchanged. These scenarios relax only '
         'AFIB-or-AFLT limits; other diagnosis limits, 95% validation confidence, '
         '101-point grids, Bonferroni correction, patient representatives, minimum '
         '25 accepted patients, and model checkpoints are unchanged.', '',
         'Test summaries use 1,904 patient representatives, including 118 positive '
         'AFIB-or-AFLT cases. Bootstrap was disabled for this descriptive analysis; '
         'the existing audit still computes exact 95% binomial accepted-error intervals. '
         'Intervals are pointwise descriptive, not simultaneous over scenarios. '
         'No scenario is selected for deployment.', '',
         'Negative-call risk means wrong negative calls / all accepted negative calls; '
         'missed-positive fraction means wrong negative calls / all 118 true cases. '
         'These denominators answer different questions.', '',
         '| Model | Negative / positive limit | Automated | 12-lead acquisition | Expert referral | Wrong / accepted negatives | Negative error (exact 95% CI) | Wrong / accepted positives | Missed / all positive cases |',
         '|---|---|---:|---:|---:|---:|---|---:|---:|']
labels = {'fixed2-s42': 'Fixed I+II', 'random2-s42': 'Random-lead',
          'fixed12-s42': '12-lead model masked to I+II'}
pct = lambda v: f'{v * 100:g}%'
for row in rows:
    neg = int(row['trusted_negative_records']); missed = int(row['trusted_negative_errors'])
    pos = int(row['trusted_positive_records']); wrong_pos = int(row['trusted_positive_errors'])
    ci = 'Undefined' if neg == 0 else (
        f"{row['trusted_negative_error']:.2%} "
        f"({row['trusted_negative_error_exact95_lower']:.2%}–{row['trusted_negative_error_exact95_upper']:.2%})")
    lines.append(f"| {labels[row['model']]} | {pct(row['max_negative_risk'])} / {pct(row['max_positive_risk'])} | "
                 f"{row['automated_coverage']:.2%} | {row['twelve_lead_acquisition_rate']:.2%} | "
                 f"{row['expert_referral_rate']:.2%} | {missed}/{neg} | {ci} | {wrong_pos}/{pos} | "
                 f"{missed}/118 ({row['missed_fraction_of_all_positive_patients']:.2%}) |")
lines += ['', '## Reproduction and artifacts', '',
          'Run from repository root:', '', '```bash',
          '.venv/bin/python artifacts/comparison/runpod-s42-relaxed-risk/run_analysis.py',
          '.venv/bin/python artifacts/comparison/runpod-s42-relaxed-risk/summarize.py', '```', '',
          '- `analysis_plan.json`: complete scenario list and exploratory status.',
          '- `protocols/`: separately versioned sensitivity contracts.',
          '- `policies/`: validation fits, threshold curves, feasibility and manifests.',
          '- `audits/`: frozen-policy test decisions, metrics and source-hash verification.',
          '- `primary_comparison.csv`: all scenarios and exact test error intervals.',
          '- `verification.json`: all nonprimary policies and selected test metrics match baseline.',
          '', 'The separate unconditional CRC sensitivity remains at 2% and does not '
          'replace these conditional-risk results. All new approaches use the same '
          'fixed 12-lead rescue model, fitted on the population routed to that stage.', '']
(OUT / 'report.md').write_text('\n'.join(lines))
