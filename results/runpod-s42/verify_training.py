import json
from pathlib import Path
import torch
from src.config import DATA_ROOT
from src.labels import load_metadata,build_label_frame
from src.training_state import apply_research_split,fingerprint_population
from src.evaluate import sha256_file
out=Path('artifacts/comparison/runpod-s42')
meta=load_metadata();frame=apply_research_split(build_label_frame(meta),meta)
actual={role:fingerprint_population(frame[frame.fold_role==role],DATA_ROOT) for role in ('train','val')}
results=[]
for root in sorted(Path('runpod-results/checkpoints').iterdir()):
 manifest=json.loads(next(root.glob('train_manifest*')).read_text())
 best=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
 last=torch.load(root/'last.pt',map_location='cpu',weights_only=True)
 logs=[json.loads(x) for x in (root/'train_log.jsonl').read_text().splitlines()]
 epochs=[x for x in logs if x.get('type')=='epoch']
 selection=best['selection_lead_set']
 winner=max(epochs,key=lambda x:x['val'][selection]['super_auc'])
 assert best['epoch']==winner['epoch']
 assert best['best_metric']==winner['val'][selection]['super_auc']
 assert last['epoch']==manifest['train_cfg']['epochs']==50
 assert best['label_schema']==manifest['label_schema']
 assert best['resume_contract']==manifest['resume_contract']
 assert actual['train']==best['resume_contract']['training_population_sha256']
 assert actual['val']==best['resume_contract']['validation_population_sha256']
 results.append({'model':root.name,'best_sha256':sha256_file(root/'best.pt'),'best_epoch':best['epoch'],'selection_auc':best['best_metric'],'last_epoch':last['epoch'],'skipped_updates':sum(x.get('optimizer_updated') is False for x in logs),'logged_updates':sum('optimizer_updated' in x for x in logs),'local_waveforms_match':True,'selection_lead':selection,'split_design':best['resume_contract']['split_design']})
(out/'training_verification.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
