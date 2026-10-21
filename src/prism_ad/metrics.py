#!/usr/bin/env python3
"""Aggregate held-out clinical prognosis; no fitting or model selection."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .runtime import workspace
HERE=workspace()
MODELS=('LR_current','HGB_current','HGB_history','TabPFN_current','TabPFN_history',
        'PRISM_current','PRISM_history','HGB_history_native_ipcw')
CONTRASTS=('PRISM_current','HGB_history','TabPFN_history','HGB_history_native_ipcw')


def scores(y,p,w,n_risk):
    ok=w>0;y=np.asarray(y)[ok];p=np.asarray(p)[ok];w=np.asarray(w)[ok]
    if not len(y) or not w.sum(): return {}
    q=np.clip(p,1e-7,1-1e-7)
    out=dict(n_known=len(y),n_target=int(y.sum()),weight_mass=float(w.sum()),
        weighted_prevalence=float(np.average(y,weights=w)),mean_predicted_risk=float(np.average(p,weights=w)),
        brier_hajek=float(np.average((y-p)**2,weights=w)),brier_ipcw=float(np.dot(w,(y-p)**2)/n_risk),
        logloss_hajek=float(-np.average(y*np.log(q)+(1-y)*np.log1p(-q),weights=w)),
        known_prevalence=float(y.mean()),brier_unweighted=float(np.mean((y-p)**2)))
    two=len(np.unique(y))==2
    out.update(auroc=roc_auc_score(y,p,sample_weight=w) if two else np.nan,
        auprc=average_precision_score(y,p,sample_weight=w) if two else np.nan,
        auroc_unweighted=roc_auc_score(y,p) if two else np.nan,
        auprc_unweighted=average_precision_score(y,p) if two else np.nan)
    return out


def load():
    data={};fits=[]
    for task in ('CN_MCI','MCI_AD'):
        for fold in (1,2,3):
            for horizon in (3,5):
                cell=HERE/'LOCAL_RESTRICTED'/task/f'fold{fold}_h{horizon}'
                if not (cell/'validation.csv').exists(): continue
                m=pd.read_csv(cell/'validation.csv',dtype={'subject_id':str})
                assert m.fold.eq(fold).all() and m.task.eq(task).all()
                assert not m.duplicated(['subject_id','visit']).any()
                preds={}
                for name in MODELS:
                    path=cell/name/'FIT.json'
                    if not path.exists(): continue
                    fit=json.loads(path.read_text())
                    prediction_path=cell/name/'probability.npy'
                    correction_path=cell/name/'seeded_inference_correction/CORRECTION.json'
                    if correction_path.exists():
                        correction=json.loads(correction_path.read_text())
                        assert fit['status']=='FAILED' and correction['status']=='CORRECTED_EXACT_REPLAY'
                        assert correction['exact_replay'] and correction['max_abs_delta']==0
                        fit.update(initial_status='FAILED',status='COMPLETED',technical_seed_correction=True,
                                   correction_seconds=correction['seconds'],exact_seeded_replay=True)
                        prediction_path=correction_path.parent/'probability.npy'
                    fits.append(fit)
                    if fit['status']!='COMPLETED': continue
                    p=np.load(prediction_path)
                    assert len(p)==len(m) and np.isfinite(p).all()
                    preds[name]=p
                if preds: data[(task,fold,horizon)]=(m,preds)
    return data,fits


def subsets(m):
    yield 'all_landmarks',np.ones(len(m),bool)
    yield 'first_eligible_landmark',m.first_eligible.to_numpy(bool)
    yield 'original_index_only',m.is_original_index.to_numpy(bool)
    yield 'with_past_history',m.has_history.to_numpy(bool)
    yield 'without_past_history',~m.has_history.to_numpy(bool)
    for study in sorted(m.study.unique()): yield 'study_'+study,m.study.eq(study).to_numpy()
    if m.task.iloc[0]=='MCI_AD':
        for origin in ('baseline_MCI','original_CN'):
            yield 'origin_'+origin,m.origin.eq(origin).to_numpy()


def bootstrap_pair(m,a,b,replicates=500):
    """Resample participants inside each validation fold; retain all their rows."""
    keys=pd.MultiIndex.from_frame(m[['fold','subject_id']])
    ix,unique=pd.factorize(keys,sort=True)
    fold_subjects={f:np.flatnonzero(np.asarray([u[0] for u in unique])==f) for f in sorted(m.fold.unique())}
    rng=np.random.default_rng(42);y=m.y.to_numpy(int);base=m.ipcw.to_numpy(float)
    draws=[]
    for _ in range(replicates):
        count=np.zeros(len(unique),int)
        for index in fold_subjects.values():
            chosen=rng.choice(index,size=len(index),replace=True)
            count+=np.bincount(chosen,minlength=len(unique))
        weight=base*count[ix]
        if len(np.unique(y[weight>0]))!=2: continue
        ok=weight>0;yy=y[ok];ww=weight[ok]
        def core(p):
            pp=p[ok]
            return np.asarray([roc_auc_score(yy,pp,sample_weight=ww),
                average_precision_score(yy,pp,sample_weight=ww),np.average((yy-pp)**2,weights=ww)])
        draws.append(core(a)-core(b))
    draws=np.asarray(draws)
    return {metric:dict(ci_low=float(np.quantile(draws[:,j],.025)),
                        ci_high=float(np.quantile(draws[:,j],.975)),n_bootstrap=len(draws))
            for j,metric in enumerate(('auroc','auprc','brier_hajek'))} if len(draws) else {}


def summarize(bootstrap):
    data,fits=load();rows=[];bins=[];pairrows=[];horizon_rows=[];explanations=[]
    contexts=[]
    for (task,fold,horizon),(m,preds) in data.items():
        context=json.loads((HERE/'LOCAL_RESTRICTED'/task/f'fold{fold}_h{horizon}'/'CONTEXT.json').read_text())
        contexts.append({k:v for k,v in context.items() if not isinstance(v,dict)})
        for name,p in preds.items():
            for sub,ix in subsets(m):
                if not ix.any(): continue
                rows.append(dict(task=task,fold=str(fold),horizon=horizon,model=name,subset=sub,
                    n_risk=int(ix.sum()),n_subjects=m.loc[ix,'subject_id'].nunique(),
                    **scores(m.y.to_numpy()[ix],p[ix],m.ipcw.to_numpy()[ix],ix.sum())))
            if name.startswith('PRISM'):
                path=HERE/'LOCAL_RESTRICTED'/task/f'fold{fold}_h{horizon}'/name/'native_explanations.npz'
                e=np.load(path);ledger=e['ledger'];w=m.ipcw.to_numpy();ok=w>0
                for j,group in enumerate(('intercept','clinical_main','clinical_pair','clinical_residual','PRS','trajectory')):
                    explanations.append(dict(task=task,fold=fold,horizon=horizon,model=name,group=group,
                        mean_abs_logit=float(np.average(np.abs(ledger[ok,j]),weights=w[ok])),
                        mean_signed_logit=float(np.average(ledger[ok,j],weights=w[ok])),
                        max_abs_ledger_closure_error=float(np.max(np.abs(ledger.sum(1)-e['logit'])))))
    for task in ('CN_MCI','MCI_AD'):
        for horizon in (3,5):
            # Comparisons pool exactly the same validation folds, not method-specific support.
            available=[(f,data[(task,f,horizon)]) for f in (1,2,3) if (task,f,horizon) in data]
            complete=[(f,x) for f,x in available if all(name in x[1] for name in MODELS)]
            if not complete: continue
            m=pd.concat([x[0] for _,x in complete],ignore_index=True)
            pred={n:np.concatenate([x[1][n] for _,x in complete]) for n in MODELS}
            assert m.groupby('subject_id').fold.nunique().max()==1
            fold_label='pooled_'+''.join(str(f) for f,_ in complete)
            for name,p in pred.items():
                for sub,ix in subsets(m):
                    if not ix.any(): continue
                    rows.append(dict(task=task,fold=fold_label,horizon=horizon,model=name,subset=sub,
                        n_risk=int(ix.sum()),n_subjects=m.loc[ix,'subject_id'].nunique(),
                        **scores(m.y.to_numpy()[ix],p[ix],m.ipcw.to_numpy()[ix],ix.sum())))
                # Descriptive fixed risk bins, not optimized calibration fitting.
                for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.01,.1)):
                    ix=(p>=lo)&(p<hi if hi<.9999 else p<=1)&m.known.to_numpy(bool)
                    if not ix.any(): continue
                    bins.append(dict(task=task,horizon=horizon,model=name,fold=fold_label,
                        risk_lo=lo,risk_hi=hi,n_known=int(ix.sum()),n_subjects=m.loc[ix,'subject_id'].nunique(),
                        mean_risk=float(np.average(p[ix],weights=m.ipcw.to_numpy()[ix])),
                        observed_ipcw=float(np.average(m.y.to_numpy()[ix],weights=m.ipcw.to_numpy()[ix]))))
            for sub in ('all_landmarks','first_eligible_landmark'):
                ix=np.ones(len(m),bool) if sub=='all_landmarks' else m.first_eligible.to_numpy(bool)
                q=m.loc[ix].reset_index(drop=True);a=pred['PRISM_history'][ix]
                for name in CONTRASTS:
                    b=pred[name][ix]
                    aa=scores(q.y,a,q.ipcw,len(q));bb=scores(q.y,b,q.ipcw,len(q))
                    ci=bootstrap_pair(q,a,b) if bootstrap else {}
                    for metric in ('auroc','auprc','brier_hajek'):
                        pairrows.append(dict(task=task,horizon=horizon,fold=fold_label,subset=sub,
                            reference=name,model='PRISM_history',metric=metric,difference=aa[metric]-bb[metric],
                            n_subjects=q.subject_id.nunique(),n_risk=len(q),
                            **ci.get(metric,{})))
        for fold in (1,2,3):
            if not all((task,fold,h) in data for h in (3,5)): continue
            m3,p3=data[(task,fold,3)];m5,p5=data[(task,fold,5)]
            assert m3[['subject_id','visit']].equals(m5[['subject_id','visit']])
            for name in set(p3)&set(p5):
                violation=p3[name]>p5[name]+1e-8
                horizon_rows.append(dict(task=task,fold=fold,model=name,n_predictions=len(m3),
                    n_p3_gt_p5=int(violation.sum()),fraction_p3_gt_p5=float(violation.mean()),
                    maximum_inversion=float(np.maximum(p3[name]-p5[name],0).max())))
    out=HERE/'results';out.mkdir(exist_ok=True)
    for name,table in (('metrics',rows),('calibration_bins',bins),('paired_contrasts',pairrows),
                       ('horizon_ordering',horizon_rows),('native_explanation_summary',explanations),('training_support',contexts)):
        pd.DataFrame(table).to_csv(out/f'{name}.csv',index=False)
    fits_public=[{k:v for k,v in f.items() if k not in ('train_loss_by_epoch','censor_stratum_fallback') and not isinstance(v,dict)} for f in fits]
    pd.DataFrame(fits_public).to_csv(out/'fit_summary.csv',index=False)
    # Follow-up measurement resolution is separate from model discrimination.
    full=pd.read_csv(HERE/'LOCAL_RESTRICTED/metadata.csv',dtype={'subject_id':str})
    support=[]
    for (task,fold),m in full.groupby(['task','fold']):
        q=m[m.first_eligible & m.event.eq(1)]
        support.append(dict(task=task,fold=fold,n_first_landmark_target_subjects=len(q),
            event_bracket_median=float(q.interval_width.median()),
            event_bracket_q75=float(q.interval_width.quantile(.75)),
            events_bracket_le_halfyear=int(q.interval_width.le(.5).sum()),
            events_bracket_le_two_year=int(q.interval_width.le(2).sum()),
            n_uncertain_event_censored=int(m.uncertain_event.sum())))
    pd.DataFrame(support).to_csv(out/'timing_resolution.csv',index=False)
    completed=sum(f['status']=='COMPLETED' for f in fits);failed=sum(f['status']=='FAILED' for f in fits)
    receipt=dict(status='AGGREGATED_COMPLETE_GRID' if completed==96 else 'PARTIAL_GRID',
        completed_main_fits=completed,failed_main_fits=failed,planned_main_fits=96,
        completed_fresh_repro_fits=sum('fresh_fit_replay' in f for f in fits),
        initial_failed_cells=sum(f.get('initial_status')=='FAILED' or f['status']=='FAILED' for f in fits),
        seed_corrected_cells=sum(f.get('technical_seed_correction',False) for f in fits),
        additional_unseeded_diagnostic_contexts=int((HERE/'LOCAL_RESTRICTED/CN_MCI/fold1_h3/TabPFN_history/replay_diagnostic/DIAGNOSTIC.json').exists()),
        bootstrap_replicates=500 if bootstrap else 0,
        ci_scope='subject cluster bootstrap within fold, conditional on fitted models and censor curves; no training/search uncertainty',
        baseline_control='first eligible or original index evaluation of the same dynamic-landmark-trained models; not a separately baseline-trained cohort',
        no_reserved_test_used=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'SUMMARY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    if rows:
        d=pd.DataFrame(rows);q=d[d.fold.eq('pooled_123')&d.subset.eq('all_landmarks')]
        if len(q): print(q[['task','horizon','model','auroc','auprc','brier_hajek']].round(4).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bootstrap',action='store_true');args=p.parse_args()
    summarize(args.bootstrap)
