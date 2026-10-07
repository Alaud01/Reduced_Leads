from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('artifacts/comparison/runpod-s42')
m=pd.read_csv(R/'all_model_metrics.csv');ci=pd.read_csv(R/'primary_paired_auroc.csv')
names=['Previous model, patient analysis','Fixed I+II','Random lead training','Twelve-lead model masked to I+II']
short=['Previous model','Fixed I+II','Random-lead training','12-lead model, masked I+II']
colors=['#6b7280','#0072B2','#009E73','#D55E00']
targets=[('rhythm','AFIB-or-AFLT'),('superclass','MI'),('superclass','STTC'),('superclass','CD'),('superclass','NORM')]
fig,(a,b)=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1.3,1]})
for i,(name,label,color) in enumerate(zip(names,short,colors)):
 vals=[m[(m.approach==name)&(m.lead_set=='2-lead')&(m.diagnosis_group==g)&(m.diagnosis==d)].iloc[0].auroc for g,d in targets]
 a.plot(vals,np.arange(len(targets))+(i-1.5)*.13,'o',color=color,label=label,markersize=6)
a.set_yticks(range(len(targets)),[x[1] for x in targets]);a.invert_yaxis();a.set_xlabel('Test AUROC (all ECG records)');a.set_title('Two-lead discrimination');a.grid(axis='x',alpha=.2);a.legend(loc='lower left',fontsize=8)
for i,r in ci.iterrows():
 b.errorbar(r.delta,i,xerr=[[r.delta-r.lower],[r.upper-r.delta]],fmt='o',color=colors[[1,2,3,1][i]],capsize=4)
b.axvline(0,color='#999',linestyle='--',linewidth=1)
b.set_yticks(range(4),['Fixed I+II vs previous','Random leads vs previous','Masked 12-lead vs previous','Fixed I+II vs random'])
b.invert_yaxis();b.set_xlabel('Patient-paired AUROC difference (95% interval)');b.set_title('AFIB-or-AFLT, one ECG per patient');b.grid(axis='x',alpha=.2)
fig.suptitle('Runpod seed 42: frozen test comparison',fontsize=14)
fig.text(.5,.015,'Exploratory, one training seed. Primary triage automation remains 0% under unchanged risk limits.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(R/'comparison.png',dpi=180);fig.savefig(R/'comparison.pdf');plt.close(fig)
