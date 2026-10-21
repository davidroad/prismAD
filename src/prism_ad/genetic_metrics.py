"""Paired genetic-information comparison from independently fitted predictions."""
import json
import numpy as np
import pandas as pd
from .runtime import workspace,input_path
from .metrics import scores,bootstrap_pair
HERE=workspace()
MODELS=('LR','HGB','TabPFN','PRISM')
def metric_values(m, p):
    s = scores(m.y, p, m.ipcw, len(m))
    return dict(n_risk=len(m), n_subjects=int(m.subject_id.nunique()), n_known=s['n_known'],
                n_target=s['n_target'], weight_mass=s['weight_mass'],
                weighted_AUROC=s['auroc'], weighted_AP=s['auprc'], Hajek_Brier=s['brier_hajek'],
                mean_prediction=s['mean_predicted_risk'], prevalence=s['weighted_prevalence'])

def summarize(cells):
    rows, bins, pairs = [], [], []
    for task in ('CN_MCI', 'MCI_AD'):
        for horizon in (3, 5):
            chosen = [c for c in cells if c['task']==task and c['horizon']==horizon]
            for fold in ('1', '2', '3', 'pooled_123'):
                group = chosen if fold=='pooled_123' else [c for c in chosen if str(c['fold'])==fold]
                m = pd.concat([c['meta'] for c in group], ignore_index=True)
                first = m.first_eligible.to_numpy(bool)
                m = m.loc[first].reset_index(drop=True)
                assert m.subject_id.is_unique
                for name in MODELS:
                    plus = np.concatenate([c['plus'][name] for c in group])[first]
                    minus = np.concatenate([np.load((c['path']/(name+'_current') if (c['path']/(name+'_current')).exists() else c['path']/name)/'probability.npy') for c in group])[first]
                    base = dict(task=task, horizon=horizon, fold=fold, model=name,
                                subset='first_eligible_landmark')
                    values = {}
                    for condition, p in (('plusPRS', plus), ('noPRS', minus)):
                        values[condition] = metric_values(m, p)
                        rows.append(dict(**base, condition=condition, **values[condition]))
                        for k in range(10):
                            lo, hi = k/10, (k+1)/10
                            ix = (p>=lo) & ((p<hi) if k<9 else (p<=1)) & (m.ipcw.to_numpy()>0)
                            w = m.ipcw.to_numpy()[ix]
                            bins.append(dict(**base, condition=condition, risk_lo=lo, risk_hi=hi,
                                n_known=int(ix.sum()), weight_mass=float(w.sum()),
                                mean_prediction=float(np.average(p[ix], weights=w)) if ix.any() else np.nan,
                                observed_prevalence=float(np.average(m.y.to_numpy()[ix], weights=w)) if ix.any() else np.nan))
                    ci = bootstrap_pair(m, plus, minus, replicates=500)
                    for col, legacy in [('weighted_AUROC','auroc'), ('weighted_AP','auprc'), ('Hajek_Brier','brier_hajek')]:
                        pairs.append(dict(**base, metric=col, contrast='plusPRS_minus_noPRS',
                            plusPRS=values['plusPRS'][col], noPRS=values['noPRS'][col],
                            difference=values['plusPRS'][col]-values['noPRS'][col],
                            n_subjects=len(m), n_risk=len(m), n_known=values['plusPRS']['n_known'],
                            n_target=values['plusPRS']['n_target'], weight_mass=values['plusPRS']['weight_mass'],
                            primary=(col=='weighted_AP'), **ci[legacy]))
    out = HERE/'results'; out.mkdir()
    for name, records in [('metrics', rows), ('calibration_bins', bins), ('paired_contrasts', pairs)]:
        pd.DataFrame(records).to_csv(out/(name+'.csv'), index=False)
    assert len(rows)==128 and len(pairs)==192 and len(bins)==1280
    assert all(r['n_bootstrap']==500 for r in pairs)
    return dict(metrics_rows=len(rows), paired_contrast_rows=len(pairs), calibration_rows=len(bins),
                bootstrap_draws_per_contrast=500, first_eligible_unique_subjects=True,
                fold_metrics_not_averaged=True)
def main():
    plus=input_path('prediction_directory');minus=input_path('no_prs_directory')
    cells=[]
    for task in ('CN_MCI','MCI_AD'):
        for fold in (1,2,3):
            for horizon in (3,5):
                src=plus/'LOCAL_RESTRICTED'/task/f'fold{fold}_h{horizon}'
                dst=minus/'LOCAL_RESTRICTED'/task/src.name
                m=pd.read_csv(src/'validation.csv',dtype={'subject_id':str})
                n=pd.read_csv(dst/'validation.csv',dtype={'subject_id':str})
                pd.testing.assert_frame_equal(m,n,check_exact=False,atol=1e-12,rtol=1e-12)
                pp={}
                for name in MODELS:
                    folder=src/(name+'_current')
                    corrected=folder/'seeded_inference_correction/probability.npy'
                    pp[name]=np.load(corrected if corrected.exists() else folder/'probability.npy')
                cells.append(dict(task=task,fold=fold,horizon=horizon,meta=m,plus=pp,path=dst))
    print(json.dumps(summarize(cells)))
if __name__=='__main__':main()
