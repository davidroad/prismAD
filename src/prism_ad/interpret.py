"""Frozen-model shape stability and independent outcome-defined group application."""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import sys
import json
import signal
import time
import unittest
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from .contributions import COMPONENTS,account,centered,contrasts,correct_family,exact_join
from .runtime import workspace,input_path,sha
from . import training as runner
from .features import Transform,CURRENT
HERE=workspace();PILOT=input_path('prediction_directory')
GROUP_FILE=input_path('cn_course_groups')


def dump(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def main():
    start=time.time();os.umask(0o077)
    if (HERE/'VALIDATION.json').exists():raise RuntimeError('Completed analysis; no overwrite')
    def timeout(*_):raise TimeoutError('900-second runtime ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    runner.prohibit_outbound();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.10,device=0)
    runner.reset_seed()
    raw=np.load(PILOT/'LOCAL_RESTRICTED/features.npz',allow_pickle=False)
    allmeta=pd.read_csv(PILOT/'LOCAL_RESTRICTED/metadata.csv',dtype={'subject_id':str})
    ix=np.flatnonzero(allmeta.task.eq('CN_MCI'))
    m=allmeta.iloc[ix].reset_index(drop=True);a={k:raw[k][ix] for k in raw.files}
    assert m.fold.isin(range(1,10)).all() and m.groupby('subject_id').fold.nunique().max()==1
    groups=pd.read_csv(GROUP_FILE,dtype={'subject_id':str})
    groups=groups.rename(columns={'PRS':'raw_PRS'})
    configs={}; grids={}; support=[]
    for fold in (1,2,3):
        tr=np.flatnonzero(m.fold.ne(fold));va=np.flatnonzero(m.fold.eq(fold))
        assert not set(m.iloc[tr].subject_id)&set(m.iloc[va].subject_id)
        pre=Transform().fit(a['current'][tr],a['history'][tr],a['prs'][tr],a['events'][tr],a['mask'][tr])
        configs[fold]=(tr,va,pre)
    for j,name in enumerate(CURRENT):
        ranges=[]
        for fold,(tr,va,pre) in configs.items():
            values=a['current'][tr,j]; values=values[np.isfinite(values)]
            low,high=np.quantile(values,[.1,.9]);ranges.append((low,high))
            support.append(dict(feature=name,fold=fold,n_training_observed=len(values),p10=low,p90=high))
        low=max(x[0] for x in ranges);high=min(x[1] for x in ranges)
        if high<=low:continue
        grid=np.arange(np.ceil(low),np.floor(high)+1) if name in ('PHC_Sex','n_domains') else np.linspace(low,high,41)
        if len(grid)>=2:grids[name]=grid
    pieces=[];curves=[];replay=[]
    private=HERE/'LOCAL_RESTRICTED';private.mkdir(exist_ok=True,mode=0o700);private.chmod(0o700)
    for fold,(tr,va,pre) in configs.items():
        def bundle(rows,hist):
            return pre.apply(a['current'][rows],a['history'][rows],a['prs'][rows],a['events'][rows],
                a['mask'][rows],m.study.to_numpy()[rows],hist)
        for horizon in (3,5):
            cell=PILOT/'LOCAL_RESTRICTED/CN_MCI'/f'fold{fold}_h{horizon}'
            assert pre.record()==json.loads((cell/'transform.json').read_text())
            draw=np.load(cell/'shared_bootstrap_local_indices.npy')
            v=pd.read_csv(cell/'validation.csv',dtype={'subject_id':str})
            assert v[['subject_id','visit','fold']].reset_index(drop=True).equals(m.iloc[va][['subject_id','visit','fold']].reset_index(drop=True))
            for name in ('PRISM_current','PRISM_history'):
                hist=name.endswith('history');train=bundle(tr,hist);valid=bundle(va,hist)
                runner.reset_seed()
                model=runner.StructuralPrismAD(train['names'],train['static'][draw],'relative_history',pre).cuda()
                model.load_state_dict(torch.load(cell/name/'model_state.pt',map_location='cuda:0',weights_only=True))
                t=runner.tensors(valid);p,extras=runner.prism_prediction(model,t)
                expected=np.load(cell/name/'probability.npy');assert np.array_equal(p,expected)
                mains=[];pairs=[]
                with torch.no_grad():
                    for j in range(0,len(v),256):
                        c=model.clinical_anchor(t['static'][j:j+256])
                        mains.append(c['clinical_main_contribution'].cpu().numpy())
                        pairs.append(c['clinical_pair_contribution'].cpu().numpy())
                mains=np.concatenate(mains);pairs=np.concatenate(pairs)
                ledger=account(valid['names'],mains,extras['ledger'])
                block=v[['subject_id','visit','study','age','fold','has_history']].copy()
                for feature,values in ledger.items():block[feature]=values
                block['risk_logit']=extras['logit'];block['predicted_risk']=p
                for j,feature in enumerate(valid['names']):block['native_main__'+feature]=mains[:,j]
                for j,pair in enumerate(model.pair_names):block['native_pair__'+'__x__'.join(pair)]=pairs[:,j]
                joined=exact_join(groups,block)
                joined['model']=name;joined['horizon']=horizon
                pieces.append(joined)
                with torch.no_grad():
                    for feature,grid in grids.items():
                        j=valid['names'].index(feature);raw_j=pre.raw_names.index(feature)
                        x=np.zeros((len(grid),len(valid['names'])),dtype=np.float32)
                        x[:,j]=(grid-pre.median[raw_j])/pre.scale[raw_j]
                        f=model.clinical_anchor.main(torch.as_tensor(x,device='cuda:0'))[:,j].cpu().numpy()
                        for raw_x,y in zip(grid,centered(f)):
                            curves.append(dict(model=name,horizon=horizon,fold=fold,feature=feature,
                                raw_value=float(raw_x),centered_main_logit=float(y),
                                reference_raw_value=float(grid[len(grid)//2])))
                replay.append(dict(fold=fold,horizon=horizon,model=name,exact_replay=True,n_validation=len(v),n_group_join=len(joined)))
                print(json.dumps(replay[-1]),flush=True)
                del model,t;torch.cuda.empty_cache()
    individuals=pd.concat(pieces,ignore_index=True)
    assert not individuals.duplicated(['subject_id','horizon','model']).any()
    dest=private/'fixed_groups_OOF_contributions.parquet';individuals.to_parquet(dest,index=False);dest.chmod(0o600)
    cf=pd.DataFrame(curves);cf.to_csv(HERE/'training_supported_shape_curves.csv',index=False)
    pd.DataFrame(support).to_csv(HERE/'shape_training_support.csv',index=False)
    stability=[]
    for (name,horizon,feature),b in cf.groupby(['model','horizon','feature']):
        matrix=b.pivot(index='raw_value',columns='fold',values='centered_main_logit').sort_index()
        delta=(matrix.iloc[-1]-matrix.iloc[0]).to_numpy()
        rho=[float(spearmanr(matrix[f],matrix[g]).statistic) for f,g in combinations((1,2,3),2)
             if np.ptp(matrix[f])>1e-9 and np.ptp(matrix[g])>1e-9]
        stability.append(dict(model=name,horizon=horizon,feature=feature,
            raw_low=float(matrix.index[0]),raw_high=float(matrix.index[-1]),
            delta_fold1=float(delta[0]),delta_fold2=float(delta[1]),delta_fold3=float(delta[2]),
            all_high_minus_low_negative=bool((delta < -1e-6).all()),
            all_high_minus_low_positive=bool((delta > 1e-6).all()),
            min_pairwise_shape_spearman=min(rho) if rho else np.nan,
            max_pairwise_curve_gap=float(np.ptp(matrix.to_numpy(),axis=1).max())))
    pd.DataFrame(stability).to_csv(HERE/'shape_stability.csv',index=False)
    effect=[];description=[];coverage=[]
    for (name,horizon),d in individuals.groupby(['model','horizon']):
        for scope in ('primary_observed','persistent_CN'):
            if scope=='persistent_CN':q=d[d.group.eq('CN_to_AD')|d.all_later_classified_CN].copy()
            else:q=d.copy()
            for group,b in q.groupby('group'):
                coverage.append(dict(model=name,horizon=horizon,scope=scope,group=group,n_subjects=len(b),
                    n_with_history=int(b.has_history.sum()),n_intermediate_MCI=int(b.intermediate_MCI.sum())))
                for feature in [*ledger,'risk_logit','predicted_risk','education','index_age','raw_PRS']:
                    values=b[feature].dropna()
                    description.append(dict(model=name,horizon=horizon,scope=scope,group=group,
                        feature=feature,n_group=len(b),n_observed=len(values),mean=float(values.mean()),
                        sd=float(values.std()),median=float(values.median())))
            for feature in COMPONENTS:
                effect.extend(dict(model=name,horizon=horizon,scope=scope,component=feature,**r)
                              for r in contrasts(q,feature))
    effects=correct_family(effect);effects.to_csv(HERE/'adjusted_component_contrasts.csv',index=False)
    pd.DataFrame(description).to_csv(HERE/'all_group_component_summaries.csv',index=False)
    pd.DataFrame(coverage).to_csv(HERE/'application_scope_support.csv',index=False)
    pd.DataFrame(replay).to_csv(HERE/'checkpoint_and_join_validation.csv',index=False)
    dump('VALIDATION.json',dict(status='COMPLETED',checkpoint_cells=len(replay),
        all_predictions_exact=True,distinct_group_participants=int(individuals.subject_id.nunique()),
        planned_contrasts=len(effects),estimable_contrasts=int(effects.status.eq('ok').sum()),
        model_training=False,elapsed_seconds=time.time()-start,
        scope='Conditional model interpretation and retrospective clinical-course groups'))
    print('INTERPRETATION APPLICATION PASS',flush=True)


if __name__=='__main__':main()
