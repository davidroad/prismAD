#!/usr/bin/env python3
"""Independent, matched-input clinical prognosis models with fixed settings."""
from __future__ import annotations
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
    CUBLAS_WORKSPACE_CONFIG=':4096:8', DO_NOT_TRACK='1', TABPFN_DISABLE_TELEMETRY='1')
import argparse
import hashlib
import importlib.metadata
import json
import pickle
import random
import resource
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from .features import Transform, Censoring, shared_bootstrap

from .runtime import workspace,input_path
from .model import StructuralPrismAD
from .inference import seeded_predict
HERE=workspace()
FEATURES=Path(os.environ.get('PRISM_AD_FEATURES',str(HERE))).resolve()
CHECKPOINT=input_path('tabpfn_checkpoint')
INCLUDE_PRS=os.environ.get('PRISM_AD_INCLUDE_PRS','1')=='1'
if not INCLUDE_PRS:
    from .genetic_control import NoPRSPrism,no_prs
    StructuralPrismAD=NoPRSPrism
CHECKPOINT_SHA='d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988'
MODELS=('LR_current','HGB_current','HGB_history','TabPFN_current','TabPFN_history',
        'PRISM_current','PRISM_history','HGB_history_native_ipcw')
if not INCLUDE_PRS: MODELS=('LR_current','HGB_current','TabPFN_current','PRISM_current')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''): h.update(x)
    return h.hexdigest()


def digest(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def prohibit_outbound():
    old=socket.socket.connect; old_ex=socket.socket.connect_ex
    def guarded(self,address):
        if self.family in (socket.AF_INET,socket.AF_INET6):
            raise PermissionError('Pilot prohibits outgoing Internet/IP sockets')
        return old(self,address)
    def guarded_ex(self,address):
        if self.family in (socket.AF_INET,socket.AF_INET6):
            raise PermissionError('Pilot prohibits outgoing Internet/IP sockets')
        return old_ex(self,address)
    socket.socket.connect=guarded; socket.socket.connect_ex=guarded_ex


def initialize():
    os.umask(0o077)
    prohibit_outbound()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.8,device=0)
    assert torch.cuda.is_available()
    assert importlib.metadata.version('tabpfn')=='8.2.0'
    assert sha(CHECKPOINT)==CHECKPOINT_SHA


def reset_seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)


def tab_model(fold):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(n_estimators=8,auto_scale_n_estimators=False,
        model_path=str(CHECKPOINT),device='cuda:0',random_state=42+fold,
        fit_mode='fit_preprocessors',n_preprocessing_jobs=1,show_progress_bar=False)


def tensors(bundle):
    return {k:torch.as_tensor(bundle[k],device='cuda:0') for k in
        ('static','prs','event_numeric','event_domain','event_mask','severity')}


def forward(model,t,idx): return model(**{k:v[idx] for k,v in t.items()})


def prism_prediction(model,t):
    model.eval(); probabilities=[]; ledgers=[]; logits=[]; attention=[]; signed=[]; embedding=[]
    with torch.no_grad():
        for j in range(0,len(t['static']),256):
            out=forward(model,t,slice(j,j+256))
            probabilities.append(torch.sigmoid(out['risk_logit']).cpu().numpy())
            logits.append(out['risk_logit'].cpu().numpy())
            ledgers.append(out['risk_group_contribution'].cpu().numpy())
            attention.append(out['event_attention'].cpu().numpy())
            signed.append(out['event_signed_contribution'].cpu().numpy())
            embedding.append(out['trajectory_embedding'].cpu().numpy())
    extra={k:np.concatenate(v) for k,v in dict(logit=logits,ledger=ledgers,
           attention=attention,signed=signed,embedding=embedding).items()}
    assert np.allclose(extra['logit'],extra['ledger'].sum(1),rtol=1e-5,atol=2e-6)
    assert np.allclose(extra['ledger'][:,-1],extra['signed'].sum(1),rtol=1e-5,atol=2e-6)
    no_history=~t['event_mask'].any(1).cpu().numpy()
    assert np.all(extra['ledger'][no_history,-1]==0)
    return np.concatenate(probabilities).astype(float),extra


def fit_prism(train,valid,y,pre,path):
    reset_seed()
    model=StructuralPrismAD(train['names'],train['static'],'relative_history',pre).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    tt=tensors(train); vt=tensors(valid); yt=torch.as_tensor(y,dtype=torch.float32,device='cuda:0')
    losses=[]; rng=torch.Generator().manual_seed(42)
    for epoch in range(60):
        model.train(); order=torch.randperm(len(y),generator=rng).cuda(); total=0.
        for j in range(0,len(y),256):
            ix=order[j:j+256]; optimizer.zero_grad(set_to_none=True)
            out=forward(model,tt,ix)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(out['risk_logit'],yt[ix])
            objective=loss+.0001*model.regularization_penalty()
            if not torch.isfinite(objective): raise RuntimeError('Nonfinite PRISM loss')
            objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step(); total+=float(loss.detach())*len(ix)
        losses.append(total/len(y))
        if (epoch+1)%20==0:
            print(json.dumps(dict(model=path.name,epoch=epoch+1,training_loss=losses[-1])),flush=True)
    p,extra=prism_prediction(model,vt)
    torch.save({k:v.cpu() for k,v in model.state_dict().items()},path/'model_state.pt')
    model.load_state_dict(torch.load(path/'model_state.pt',map_location='cuda:0',weights_only=True))
    p2,_=prism_prediction(model,vt)
    assert np.array_equal(p,p2), 'Neural checkpoint inference replay differs'
    np.savez_compressed(path/'native_explanations.npz',**extra)
    info=dict(parameters=model.parameter_count(),train_loss_by_epoch=losses,
              exact_checkpoint_replay=True,ledger_max_abs_error=float(np.max(np.abs(extra['logit']-extra['ledger'].sum(1)))),
              no_history_exact_zero=True,prs_coefficient=float(model.effective_prs_coefficient().detach().cpu()))
    del optimizer,model,tt,vt,yt
    torch.cuda.empty_cache()
    return p,info


def subset(bundle,ix):
    return {k:v if k=='names' else v[ix] for k,v in bundle.items()}


def preflight():
    initialize(); reset_seed()
    from tabpfn import save_fitted_tabpfn_model,load_fitted_tabpfn_model
    import inspect
    from tabpfn import TabPFNClassifier
    assert 'sample_weight' not in inspect.signature(TabPFNClassifier.fit).parameters
    from .features import CURRENT,HISTORY
    n=64; rng=np.random.default_rng(42)
    c=rng.normal(size=(n,len(CURRENT))); h=rng.normal(size=(n,len(HISTORY)))
    prs=rng.normal(size=n); ev=rng.normal(size=(n,16,7)).astype(np.float32)
    mask=rng.random((n,16))>.3; mask[0]=False; studies=np.array(['ADNI']*n)
    pre=Transform().fit(c[:48],h[:48],prs[:48],ev[:48],mask[:48])
    bundle=pre.apply(c,h,prs,ev,mask,studies,True)
    model=StructuralPrismAD(bundle['names'],bundle['static'][:48],'relative_history',pre).cuda()
    t=tensors(bundle);model.train();out=forward(model,t,slice(0,32))
    loss=out['risk_logit'].square().mean();loss.backward()
    p,extra=prism_prediction(model,t)
    assert extra['ledger'][0,-1]==0
    # Inverse transform check through the native relative-history module.
    rel=model.relative_inputs(t['static'],t['event_numeric'],t['event_domain'],t['event_mask']).detach().cpu().numpy()
    expected=(c[:,4:8][:,np.tile(np.arange(4),4)]-ev[:,:,0])/pre.event_value_scale
    np.testing.assert_allclose(rel[:,:,7][mask],expected[mask],atol=2e-5,rtol=1e-5)
    del model,t;torch.cuda.empty_cache()
    tab=tab_model(1);x=bundle['tabular'][:,:20];y=(c[:,0]+c[:,1]>0).astype(int)
    tab.fit(x[:48],y[:48]);pred=tab.predict_proba(x[48:])
    pred2=tab.predict_proba(x[48:]);assert np.array_equal(pred,pred2)
    # Native supported serialization tested only on synthetic records.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='stage_pilot_tabpfn_',dir='/tmp') as td:
        dest=Path(td)/'synthetic.tabpfn_fit'
        save_fitted_tabpfn_model(tab,dest)
        restored=load_fitted_tabpfn_model(dest,device='cuda:0')
        pred3=restored.predict_proba(x[48:])
        assert np.array_equal(pred,pred3)
    dump(HERE/'PREFLIGHT.json',dict(status='PASS',native_neural_backward=True,
        relative_raw_units_verified=True,ledger_verified=True,no_history_zero=True,
        tabpfn_offline_fit=True,tabpfn_synthetic_serialization_exact=True,
        tabpfn='8.2.0',checkpoint_sha256=CHECKPOINT_SHA,
        python=sys.version,torch=torch.__version__,sklearn=importlib.metadata.version('scikit-learn'),
        gpu=torch.cuda.get_device_name(0),network='Python IP sockets blocked plus offline flags'))
    print('PREFLIGHT PASS',flush=True)


def train(task):
    initialize()
    assert json.loads((HERE/'PREFLIGHT.json').read_text())['status']=='PASS'
    receipt=json.loads((FEATURES/'DATA_RECEIPT.json').read_text())
    for name,h in receipt['output_sha256'].items(): assert sha(FEATURES/name)==h
    root=HERE/'LOCAL_RESTRICTED'/task;root.mkdir(mode=0o700,exist_ok=True)
    worker_file=HERE/f'WORKER_{task}.json'
    if worker_file.exists(): raise RuntimeError('Task worker has already been invoked; no automatic reruns')
    started=time.time(); results=[]
    dump(worker_file,dict(status='RUNNING',pid=os.getpid(),task=task,started_unix=started))
    def timeout(_signum,_frame): raise SystemExit('Fixed3600s task-worker budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(3600)
    meta=pd.read_csv(FEATURES/'LOCAL_RESTRICTED/metadata.csv',dtype={'subject_id':str})
    assert meta.fold.isin(range(1,10)).all()
    assert meta.groupby('subject_id').fold.nunique().max()==1
    raw=np.load(FEATURES/'LOCAL_RESTRICTED/features.npz',allow_pickle=False)
    select=np.flatnonzero(meta.task.eq(task));meta=meta.iloc[select].reset_index(drop=True)
    data={k:raw[k][select] for k in raw.files}
    def transformed(ix,pre,history):
        bundle=pre.apply(data['current'][ix],data['history'][ix],data['prs'][ix],
                         data['events'][ix],data['mask'][ix],meta.study.to_numpy()[ix],history)
        return bundle if INCLUDE_PRS else no_prs(bundle)
    status='COMPLETED'
    try:
        for fold in (1,2,3):
            tr=np.flatnonzero(meta.fold.ne(fold));va=np.flatnonzero(meta.fold.eq(fold))
            assert not set(meta.iloc[tr].subject_id)&set(meta.iloc[va].subject_id)
            pre=Transform().fit(data['current'][tr],data['history'][tr],data['prs'][tr],data['events'][tr],data['mask'][tr])
            censor=Censoring().fit(meta.iloc[tr])
            for horizon in (3,5):
                cell=root/f'fold{fold}_h{horizon}';cell.mkdir()
                dump(cell/'transform.json',pre.record())
                lt=censor.label(meta.iloc[tr],horizon);lv=censor.label(meta.iloc[va],horizon)
                ix=shared_bootstrap(lt,42+1000*(task=='MCI_AD')+100*fold+horizon)
                np.save(cell/'shared_bootstrap_local_indices.npy',ix)
                validation=meta.iloc[va].copy()
                validation['y']=lv['y'];validation['known']=lv['known'];validation['ipcw']=lv['weight']
                validation['censor_survival']=lv['g'];validation['weight_clipped']=lv['clipped']
                validation.to_csv(cell/'validation.csv',index=False)
                context=dict(task=task,fold=fold,horizon=horizon,n_training_risk_rows=len(tr),
                    n_validation_risk_rows=len(va),n_known_training=int(lt['known'].sum()),
                    n_bootstrap=len(ix),n_unique_bootstrap=len(np.unique(ix)),
                    bootstrap_hash=digest(ix),training_row_hash=digest(tr),validation_row_hash=digest(va),
                    training_subjects=meta.iloc[tr].subject_id.nunique(),validation_subjects=meta.iloc[va].subject_id.nunique(),
                    n_known_validation=int(lv['known'].sum()),known_validation_target=int(lv['y'][lv['known']].sum()),
                    validation_weight_mass=float(lv['weight'].sum()),validation_weight_clipped=int(lv['clipped'].sum()),
                    training_weight_clipped=int(lt['clipped'].sum()),censor_stratum_fallback=censor.fallback)
                dump(cell/'CONTEXT.json',context)
                for name in MODELS:
                    path=cell/name;path.mkdir();start=time.time()
                    print(json.dumps(dict(event='START',task=task,fold=fold,horizon=horizon,model=name)),flush=True)
                    rec=dict(**context,model=name,status='RUNNING')
                    dump(path/'FIT.json',rec)
                    try:
                        hist='history' in name;tb=transformed(tr,pre,hist);vb=transformed(va,pre,hist)
                        native=name.endswith('native_ipcw')
                        sampled=np.flatnonzero(lt['known']) if native else ix
                        training=subset(tb,sampled);y=lt['y'][sampled]
                        assert len(np.unique(y))==2
                        reset_seed()
                        if name.startswith('PRISM'):
                            p,info=fit_prism(training,vb,y,pre,path)
                        elif name.startswith('TabPFN'):
                            model=tab_model(fold)
                            model.fit(training['tabular'],y)
                            p=seeded_predict(model,vb['tabular'],fold).astype(float)
                            # Same query batching for repeatability; context rebuilt only in prescribed cell.
                            p2=seeded_predict(model,vb['tabular'],fold).astype(float)
                            info=dict(exact_same_context_replay=bool(np.array_equal(p,p2)),
                                      replay_max_abs_error=float(np.max(np.abs(p-p2))),checkpoint_sha256=CHECKPOINT_SHA,
                                      fit_context_stored='rebuild using cache, split, transform and shared indices; no duplicated weights')
                            assert np.array_equal(p,p2), 'TabPFN same-context inference replay differs'
                            del model;torch.cuda.empty_cache()
                        else:
                            if name.startswith('LR'):
                                model=LogisticRegression(C=1.,max_iter=2000,random_state=42)
                            else:
                                model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=250,max_leaf_nodes=15,
                                    min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=42)
                            kwargs={'sample_weight':lt['weight'][sampled]} if native else {}
                            model.fit(training['tabular'],y,**kwargs)
                            p=model.predict_proba(vb['tabular'])[:,1].astype(float)
                            with (path/'model.pkl').open('wb') as f: pickle.dump(model,f)
                            with (path/'model.pkl').open('rb') as f: replay=pickle.load(f)
                            assert np.array_equal(p,replay.predict_proba(vb['tabular'])[:,1])
                            info=dict(exact_checkpoint_replay=True)
                            del model,replay
                        assert len(p)==len(va) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                        np.save(path/'probability.npy',p)
                        rec.update(status='COMPLETED',seconds=time.time()-start,
                                   prediction_sha256=sha(path/'probability.npy'),**info)
                        # Exactly three optional, prespecified fresh-model reproducibility repeats.
                        if INCLUDE_PRS and task=='CN_MCI' and fold==1 and horizon==3 and name in ('PRISM_history','TabPFN_current','TabPFN_history'):
                            rp=path/'reproducibility_repeat';rp.mkdir()
                            if name.startswith('PRISM'):
                                repeat,repeatinfo=fit_prism(training,vb,y,pre,rp)
                            else:
                                reset_seed();again=tab_model(fold);again.fit(training['tabular'],y)
                                repeat=seeded_predict(again,vb['tabular'],fold).astype(float)
                                repeatinfo={};del again;torch.cuda.empty_cache()
                            np.save(rp/'probability.npy',repeat)
                            repeatinfo.update(exact_fresh_fit_replay=bool(np.array_equal(p,repeat)),max_abs_error=float(np.max(np.abs(p-repeat))))
                            dump(rp/'REPLAY.json',repeatinfo)
                            rec['fresh_fit_replay']=repeatinfo['exact_fresh_fit_replay']
                            rec['fresh_fit_max_abs_error']=repeatinfo['max_abs_error']
                            assert np.array_equal(p,repeat), 'Prespecified fresh-fit reproducibility differs'
                    except Exception as exc:
                        rec.update(status='FAILED',seconds=time.time()-start,error_type=type(exc).__name__,error=str(exc))
                        # Trace contains code locations; input arrays/participant records are never printed.
                        (path/'ERROR.txt').write_text(traceback.format_exc())
                        print(json.dumps(dict(event='FAILED',task=task,model=name,error_type=type(exc).__name__,error=str(exc))),flush=True)
                        torch.cuda.empty_cache()
                    dump(path/'FIT.json',rec);results.append(rec)
                    dump(worker_file,dict(status='RUNNING',task=task,completed=sum(r['status']=='COMPLETED' for r in results),
                        failed=sum(r['status']=='FAILED' for r in results),elapsed_seconds=time.time()-started))
                    print(json.dumps(dict(event='END',task=task,fold=fold,horizon=horizon,model=name,status=rec['status'],seconds=rec['seconds'])),flush=True)
    except SystemExit as exc:
        status='TIMED_OUT';print(str(exc),flush=True)
    except Exception:
        status='WORKER_FAILED';(root/'WORKER_ERROR.txt').write_text(traceback.format_exc())
    finally:
        signal.alarm(0)
        dump(worker_file,dict(status=status,task=task,completed=sum(r['status']=='COMPLETED' for r in results),
            failed=sum(r['status']=='FAILED' for r in results),scheduled_main_fits=6*len(MODELS),
            elapsed_seconds=time.time()-started,peak_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            code_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('features.py'))},
            maximum_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
        print(json.dumps(dict(task=task,status=status,attempted=len(results))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=('CN_MCI','MCI_AD'));parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    preflight() if args.preflight else train(args.task)
