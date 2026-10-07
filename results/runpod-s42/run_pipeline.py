from pathlib import Path
import subprocess, sys, json, os
ROOT=Path.cwd()
OUT=ROOT/'artifacts/comparison/runpod-s42'
EVAL=OUT/'evaluation'
ENV={**os.environ,'MPLCONFIGDIR':'/tmp/reduced-leads-mpl','PYTHONUNBUFFERED':'1'}
MODELS=['fixed12-s42','fixed2-s42','random2-s42']
PROTOCOL='protocol/use_contract_v2.json'
def run(*args):
 print('RUN',*args,flush=True)
 subprocess.run([sys.executable,'-m',*args],check=True,env=ENV)
def export(split):
 for model in MODELS:
  destination=EVAL/f'{model}-{split}'
  if (destination/'manifest.json').exists() and json.loads((destination/'manifest.json').read_text()).get('status')=='complete':
   continue
  leads=['12-lead','2-lead'] if model=='fixed12-s42' else ['2-lead']
  run('src.evaluate','--checkpoint',f'runpod-results/checkpoints/{model}/best.pt','--splits',split,'--lead-sets',*leads,'--device','cpu','--batch-size','32','--out-dir',str(EVAL),'--run-name',destination.name)
 for model in MODELS[1:]:
  destination=EVAL/f'{model}-paired-{split}'
  if not destination.exists():
   run('src.assemble_evaluation','--reduced-dir',str(EVAL/f'{model}-{split}'),'--twelve-dir',str(EVAL/f'fixed12-s42-{split}'),'--lead-set','2-lead','--split',split,'--out-dir',str(destination))
def source(model,split):
 return EVAL/f'{model}{"-paired" if model!="fixed12-s42" else ""}-{split}'
export('val')
for model in MODELS:
 destination=OUT/'policies'/model
 if not (destination/'policy.json').exists():
  run('src.use_contract','fit','--evaluation-dir',str(source(model,'val')),'--protocol',PROTOCOL,'--lead-sets','2-lead','--out-dir',str(OUT/'policies'),'--run-name',model)
# No test predictions are read/exported until all three policy bundles complete.
for model in MODELS:
 assert json.loads((OUT/'policies'/model/'manifest.json').read_text())['status']=='complete'
export('test')
for model in MODELS:
 destination=OUT/'audits'/model
 if not (destination/'manifest.json').exists():
  run('src.use_contract','audit','--evaluation-dir',str(source(model,'test')),'--policy',str(OUT/'policies'/model/'policy.json'),'--protocol',PROTOCOL,'--out-dir',str(OUT/'audits'),'--run-name',model)
print('PIPELINE COMPLETE',flush=True)
