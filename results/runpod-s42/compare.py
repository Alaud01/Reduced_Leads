from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.evaluate import sha256_file
from src.research_risk import representative_ecgs
from src.use_contract import build_composite_frame

ROOT=Path('artifacts/comparison/runpod-s42')
RUNS={'Previous model, patient analysis':Path('artifacts/audit-all/patient-v2-final-audit'),
      'Fixed I+II':ROOT/'audits/fixed2-s42',
      'Random lead training':ROOT/'audits/random2-s42',
      'Twelve-lead model masked to I+II':ROOT/'audits/fixed12-s42'}
models=[];policies=[];crc=[];subgroups=[];repeats=[]
for name,path in RUNS.items():
 manifest=json.loads((path/'manifest.json').read_text());assert manifest['status']=='complete'
 for item in manifest['artifacts']:
  assert sha256_file(path/item['path'])==item['sha256'],(name,item['path'])
 for file,collector in [('model_metrics',models),('policy_metrics',policies),('crc_sensitivity',crc),('subgroup_metrics',subgroups),('repeat_ecg_sensitivity',repeats)]:
  data=pd.read_csv(path/f'{file}.csv.gz');data.insert(0,'approach',name);collector.append(data)
model=pd.concat(models,ignore_index=True);policy=pd.concat(policies,ignore_index=True)
model.to_csv(ROOT/'all_model_metrics.csv',index=False);policy.to_csv(ROOT/'all_policy_metrics.csv',index=False)
for file,frames in [('crc_comparison',crc),('subgroup_comparison',subgroups),('repeat_ecg_comparison',repeats)]:
 pd.concat(frames,ignore_index=True).to_csv(ROOT/f'{file}.csv',index=False)
keys=['diagnosis_group','diagnosis','lead_set']
old=model[model.approach=='Previous model, patient analysis'][keys+['auroc','average_precision','brier']]
deltas=model.merge(old,on=keys,suffixes=('','_previous'))
for metric in ['auroc','average_precision','brier']:
 deltas[f'delta_{metric}']=deltas[metric]-deltas[f'{metric}_previous']
deltas.to_csv(ROOT/'per_target_changes.csv',index=False)
# Paired one-patient-one-ECG comparison avoids overweighting repeat recordings.
predpaths={'Previous model, patient analysis':Path('artifacts/evaluation/afib-locked-test-export-v1/predictions/test__2-lead.csv.gz'),
           'Fixed I+II':ROOT/'evaluation/fixed2-s42-test/predictions/test__2-lead.csv.gz',
           'Random lead training':ROOT/'evaluation/random2-s42-test/predictions/test__2-lead.csv.gz',
           'Twelve-lead model masked to I+II':ROOT/'evaluation/fixed12-s42-test/predictions/test__2-lead.csv.gz'}
frames={}
for name,path in predpaths.items():
 frame=build_composite_frame(pd.read_csv(path,low_memory=False))
 frames[name]=representative_ecgs(frame,42).set_index('ecg_id').sort_index()
ref=frames['Previous model, patient analysis']
for name,frame in frames.items():
 assert frame[['patient_id','target']].equals(ref[['patient_id','target']]),name
pairs=[(name,'Previous model, patient analysis') for name in list(frames)[1:]]+ [('Fixed I+II','Random lead training')]
ci=[]
y=ref.target.to_numpy();rng=np.random.default_rng(42)
indices=rng.integers(0,len(y),size=(2000,len(y)))
for left,right in pairs:
 a=frames[left].logit.to_numpy();b=frames[right].logit.to_numpy()
 differences=[roc_auc_score(y[idx],a[idx])-roc_auc_score(y[idx],b[idx]) for idx in indices if np.unique(y[idx]).size==2]
 ci.append({'approach':left,'reference':right,'n_patients':len(y),'auroc':roc_auc_score(y,a),'reference_auroc':roc_auc_score(y,b),'delta':roc_auc_score(y,a)-roc_auc_score(y,b),'lower':float(np.quantile(differences,.025)),'upper':float(np.quantile(differences,.975))})
pd.DataFrame(ci).to_csv(ROOT/'primary_paired_auroc.csv',index=False)
print('COMPARISON READY')
print(model[(model.diagnosis=='AFIB-or-AFLT')&(model.lead_set=='2-lead')][['approach','auroc','average_precision','brier']].to_string(index=False))
print(policy[policy.diagnosis=='AFIB-or-AFLT'][['approach','automated_coverage','trusted_negative_records','trusted_negative_errors','trusted_negative_error_exact95_upper']].to_string(index=False))
# Frozen calibration quality, on the same patient representatives as primary policy metrics.
calibration=[]
for name,path in RUNS.items():
 d=pd.read_csv(path/'decisions.csv.gz',low_memory=False)
 d=d[(d.diagnosis=='AFIB-or-AFLT')&(d.lead_set=='2-lead')]
 d=representative_ecgs(d,42)
 calibration.append({'approach':name,'n_patients':len(d),
                     'raw_brier':float(np.mean((1/(1+np.exp(-d.reduced_logit))-d.target)**2)),
                     'calibrated_brier':float(np.mean((d.reduced_probability-d.target)**2))})
pd.DataFrame(calibration).to_csv(ROOT/'primary_calibration.csv',index=False)
# Preserve historical record-level method separately; never treat a denominator change as model improvement.
historical=pd.read_csv('artifacts/audit-all/use-contract-routed-v2-audit/policy_metrics.csv.gz')
historical[historical.lead_set=='2-lead'].to_csv(ROOT/'historical_record_level_policy.csv',index=False)
# Hash the final analysis inputs/outputs for reproducibility.
manifest={'status':'complete','training_seed':42,'analysis_plan_sha256':sha256_file(ROOT/'analysis_plan.json'),
          'sources':{name:{'path':str(path),'manifest_sha256':sha256_file(path/'manifest.json')} for name,path in RUNS.items()},
          'outputs':{p.name:sha256_file(p) for p in ROOT.glob('*.csv')}}
(ROOT/'comparison_manifest.json').write_text(json.dumps(manifest,indent=2))
