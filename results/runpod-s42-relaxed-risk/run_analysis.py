"""Exploratory AFIB/AFLT risk sensitivity using the unchanged fit/audit engine."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/comparison/runpod-s42'
MODELS = ['fixed2-s42', 'random2-s42', 'fixed12-s42']
SCENARIOS = [('neg02-pos40', .02, .40), ('neg025-pos10', .025, .10),
             ('neg03-pos10', .03, .10), ('neg03-pos40', .03, .40)]

def source(model, split):
    paired = '-paired' if model != 'fixed12-s42' else ''
    return BASE / 'evaluation' / f'{model}{paired}-{split}'

def run(*args):
    print('RUN', *map(str, args), flush=True)
    subprocess.run([sys.executable, '-m', 'src.use_contract', *map(str, args)],
                   cwd=ROOT, check=True,
                   env={**os.environ, 'MPLCONFIGDIR': '/tmp/reduced-leads-mpl',
                        'PYTHONUNBUFFERED': '1'})

if __name__ == '__main__':
    baseline = json.loads((ROOT / 'protocol/use_contract_v2.json').read_text())
    (OUT / 'protocols').mkdir(exist_ok=True)
    plan = {'status': 'posthoc_exploratory_no_confirmatory_claims',
            'models': MODELS, 'scenarios': SCENARIOS,
            'scope': 'AFIB-or-AFLT only; all other diagnosis limits unchanged',
            'selection': 'All scenarios defined before this sensitivity audit; report all, select no winner.',
            'cohort': 'Previously inspected PTB-XL fold 10; no untouched validation claim.',
            'uncertainty': 'Exact patient-binomial test error intervals retained; bootstrap disabled for this sensitivity.',
            'crc': 'Unchanged at 2%; not the conditional risk endpoint.'}
    plan_path = OUT / 'analysis_plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == json.loads(json.dumps(plan))
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    for name, negative, positive in SCENARIOS:
        protocol = copy.deepcopy(baseline)
        protocol['protocol_id'] = f'ptbxl-use-contract-v2-posthoc-{name}'
        protocol['status'] = 'posthoc_risk_sensitivity_no_formal_clinical_claim'
        protocol['risk_limits_placeholders']['AFIB-or-AFLT_rule_out']['max_risk'] = negative
        protocol['risk_limits_placeholders']['AFIB-or-AFLT_rule_out']['rationale'] = 'Post-hoc sensitivity placeholder; not a clinical recommendation.'
        protocol['risk_limits_placeholders']['AFIB-or-AFLT_rule_in']['comparability_anchor'] = positive
        protocol['risk_limits_placeholders']['AFIB-or-AFLT_rule_in']['rationale'] = 'Post-hoc sensitivity placeholder; not a clinical recommendation.'
        protocol['primary_analysis']['risk_limit'] = negative
        protocol['uncertainty']['replicates'] = 0
        protocol['sensitivity_analysis'] = plan
        path = OUT / 'protocols' / f'{name}.json'
        if path.exists():
            assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
        else:
            path.write_text(json.dumps(protocol, indent=2) + '\n')
    # Freeze every validation policy before any new sensitivity audit.
    for name, _, _ in SCENARIOS:
        for model in MODELS:
            run_name = f'{name}--{model}'
            manifest = OUT / 'policies' / run_name / 'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text())['status'] == 'complete':
                continue
            run('fit', '--evaluation-dir', source(model, 'val'), '--protocol',
                OUT / 'protocols' / f'{name}.json', '--lead-sets', '2-lead',
                '--out-dir', OUT / 'policies', '--run-name', run_name)
    for name, _, _ in SCENARIOS:
        for model in MODELS:
            run_name = f'{name}--{model}'
            manifest = OUT / 'audits' / run_name / 'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text())['status'] == 'complete':
                continue
            run('audit', '--evaluation-dir', source(model, 'test'), '--protocol',
                OUT / 'protocols' / f'{name}.json', '--policy',
                OUT / 'policies' / run_name / 'policy.json',
                '--out-dir', OUT / 'audits', '--run-name', run_name)
    print('SENSITIVITY COMPLETE', flush=True)
