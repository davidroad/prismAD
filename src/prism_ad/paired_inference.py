"""Recompute paired AP tests and current-state comparison intervals."""
import json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from .runtime import workspace,input_path
from . import metrics as metrics_module
from .significance import bootstrap
def main():
    metrics_module.HERE=input_path('prediction_directory')
    data,_=metrics_module.load()
    out=workspace();out.mkdir(exist_ok=True,parents=True)
    if (out/'paired_AP_tests.csv').exists():raise RuntimeError('Choose a new output directory')
    rows=[];current_rows=[]
    current=[('PRISM_current',x) for x in ['LR_current','HGB_current','TabPFN_current']]
    history=[('PRISM_history',x) for x in metrics_module.CONTRASTS]
    for task in ('CN_MCI','MCI_AD'):
        for horizon in (3,5):
            cell=[data[(task,f,horizon)] for f in (1,2,3)]
            m=pd.concat([x[0] for x in cell],ignore_index=True)
            pred={n:np.concatenate([x[1][n] for x in cell]) for n in metrics_module.MODELS}
            for subset,pairs in [('first_eligible_landmark',current+history),('all_landmarks',history)]:
                keep=m.first_eligible.to_numpy(bool) if subset.startswith('first') else np.ones(len(m),bool)
                q=m.loc[keep].reset_index(drop=True)
                result=bootstrap(q,{n:v[keep] for n,v in pred.items()},pairs)
                for record in result:
                    a,b=record['model'],record['reference']
                    interval=metrics_module.bootstrap_pair(q,pred[a][keep],pred[b][keep],500)
                    record.update(ci_low_original500=interval['auprc']['ci_low'],ci_high_original500=interval['auprc']['ci_high'])
                    if (a,b) in current:
                        for metric in ['auroc','auprc','brier_hajek']:
                            aa=metrics_module.scores(q.y,pred[a][keep],q.ipcw,len(q))[metric]
                            bb=metrics_module.scores(q.y,pred[b][keep],q.ipcw,len(q))[metric]
                            current_rows.append(dict(task=task,horizon=horizon,model=a,reference=b,metric=metric,
                                difference=aa-bb,**interval[metric]))
                rows.extend(dict(task=task,horizon=horizon,subset=subset,**r) for r in result)
    d=pd.DataFrame(rows)
    for subset,n in [('first_eligible_landmark',28),('all_landmarks',16)]:
        keep=d.subset.eq(subset);assert keep.sum()==n
        d.loc[keep,'q']=multipletests(d.loc[keep,'p'],method='fdr_bh')[1]
        d.loc[keep,'family_size']=n
    d.to_csv(out/'paired_AP_tests.csv',index=False)
    pd.DataFrame(current_rows).to_csv(out/'current_model_paired_comparisons.csv',index=False)
    print(json.dumps(dict(paired_AP_tests=len(d),bootstrap_draws=9999,percentile_interval_draws=500)))
if __name__=='__main__':main()
