#!/usr/bin/env python3
"""Retrospective diagnostics; original data, predictions and checkpoints remain read-only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import sys,json,hashlib,inspect,signal,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from .runtime import workspace,input_path
OUT=workspace();OUT.mkdir(parents=True,exist_ok=True)
ORIGINAL=input_path('prediction_directory')
from . import training as pilot

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def log(value):
 message=json.dumps(value);print(message,flush=True)
 with (OUT/'replay.log').open('a') as f:f.write(message+'\n')
def logits(model,t):
 return torch.cat([pilot.forward(model,t,slice(j,j+256))['risk_logit'] for j in range(0,len(t['static']),256)])

CONTEXT={}
def monitor(model,tt,vt,yt,epoch,online):
 c=CONTEXT
 assert online==c['fit']['train_loss_by_epoch'][epoch],(c['case'],epoch+1,online,c['fit']['train_loss_by_epoch'][epoch])
 cpu_state,cuda_states=torch.get_rng_state(),torch.cuda.get_rng_state_all()
 model.eval()
 with torch.no_grad():
  training=float(F.binary_cross_entropy_with_logits(logits(model,tt).double(),yt.double()))
  each=F.binary_cross_entropy_with_logits(logits(model,vt).double(),c['vy'],reduction='none')
  validation=float((each*c['vw']).sum()/c['vw'].sum())
 torch.set_rng_state(cpu_state);torch.cuda.set_rng_state_all(cuda_states)
 c['rows'].append(dict(**c['case'],epoch=epoch+1,training_BCE=training,
  validation_IPCW_BCE=validation,original_online_training_BCE=online))
 if (epoch+1)%20==0:log(dict(event='DIAGNOSTIC',**c['case'],epoch=epoch+1,training_BCE=training,validation_IPCW_BCE=validation))

def verify_final(model,vt,path,losses):
 reference=torch.load(path/'model_state.pt',map_location='cpu',weights_only=True)
 current={k:v.detach().cpu() for k,v in model.state_dict().items()}
 assert current.keys()==reference.keys()
 assert all(torch.equal(v,reference[k]) for k,v in current.items()),CONTEXT['case']
 np.testing.assert_array_equal(losses,CONTEXT['fit']['train_loss_by_epoch'])
 p,_=pilot.prism_prediction(model,vt)
 np.testing.assert_array_equal(p,np.load(path/'probability.npy',allow_pickle=False))

def main():
 if (OUT/'replay_status.json').exists():raise RuntimeError('Replay already invoked; no automatic retries.')
 def timeout(*_):raise TimeoutError('2400-second replay budget exhausted')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(2400)
 started=time.time();dump(OUT/'replay_status.json',dict(status='RUNNING',completed=0))
 pilot.initialize()
 assert torch.__version__=='2.6.0+cu124'
 receipt=json.loads((ORIGINAL/'DATA_RECEIPT.json').read_text())
 for file,expected in receipt['output_sha256'].items():assert sha(ORIGINAL/file)==expected,file
 # Preserve the exact original optimizer/training body; insert only read-only monitoring.
 original_source=inspect.getsource(pilot.fit_prism)
 modified=original_source.replace('def fit_prism(','def fit_prism_diagnostics(',1)
 needle='        losses.append(total/len(y))'
 assert modified.count(needle)==1
 modified=modified.replace(needle,needle+'\n        epoch_monitor(model,tt,vt,yt,epoch,losses[-1])',1)
 marker='\n    p,extra=prism_prediction(model,vt)'
 assert modified.count(marker)==1
 modified=modified.split(marker)[0]+'\n    final_monitor(model,vt,path,losses)\n    del optimizer,model,tt,vt,yt\n    torch.cuda.empty_cache()\n'
 assert 'torch.save' not in modified and 'np.save' not in modified
 (OUT/'original_fit_function.py').write_text(original_source)
 (OUT/'monitored_fit_function.py').write_text(modified)
 namespace=dict(pilot.__dict__,epoch_monitor=monitor,final_monitor=verify_final)
 exec(compile(modified,str(OUT/'monitored_fit_function.py'),'exec'),namespace)
 replay=namespace['fit_prism_diagnostics']
 provenance=dict(original_directory=str(ORIGINAL),python=sys.version,torch=torch.__version__,
  gpu=torch.cuda.get_device_name(0),original_source_sha256={str(Path(m.__file__)):sha(m.__file__) for m in (pilot,sys.modules['prism_ad.features'],sys.modules['prism_ad.model'])},
  input_sha256=receipt['output_sha256'],run_audits=[])
 meta_all=pd.read_csv(ORIGINAL/'LOCAL_RESTRICTED/metadata.csv',dtype={'subject_id':str})
 with np.load(ORIGINAL/'LOCAL_RESTRICTED/features.npz',allow_pickle=False) as archive:data_all={k:archive[k] for k in archive.files}
 rows=[];audits=[]
 for task in ('MCI_AD','CN_MCI'):
  select=np.flatnonzero(meta_all.task.eq(task));meta=meta_all.iloc[select].reset_index(drop=True)
  data={k:v[select] for k,v in data_all.items()}
  for fold in (1,2,3):
   tr=np.flatnonzero(meta.fold.ne(fold));va=np.flatnonzero(meta.fold.eq(fold))
   assert not set(meta.iloc[tr].subject_id)&set(meta.iloc[va].subject_id)
   pre=pilot.Transform().fit(data['current'][tr],data['history'][tr],data['prs'][tr],data['events'][tr],data['mask'][tr])
   censor=pilot.Censoring().fit(meta.iloc[tr])
   for horizon in (3,5):
    cell=ORIGINAL/'LOCAL_RESTRICTED'/task/f'fold{fold}_h{horizon}'
    lt=censor.label(meta.iloc[tr],horizon);lv=censor.label(meta.iloc[va],horizon)
    sample=np.load(cell/'shared_bootstrap_local_indices.npy',allow_pickle=False)
    np.testing.assert_array_equal(sample,pilot.shared_bootstrap(lt,42+1000*(task=='MCI_AD')+100*fold+horizon))
    saved=pd.read_csv(cell/'validation.csv')
    np.testing.assert_array_equal(saved['known'].to_numpy(),lv['known'])
    np.testing.assert_array_equal(saved['y'].to_numpy(),lv['y'])
    np.testing.assert_allclose(saved['ipcw'],lv['weight'],rtol=2e-15,atol=2e-15)
    for name in ('PRISM_current','PRISM_history'):
     directory=cell/name;fit=json.loads((directory/'FIT.json').read_text())
     assert fit['status']=='COMPLETED' and len(fit['train_loss_by_epoch'])==60
     assert pilot.digest(sample)==fit['bootstrap_hash']
     assert pilot.digest(tr)==fit['training_row_hash'] and pilot.digest(va)==fit['validation_row_hash']
     assert sha(directory/'probability.npy')==fit['prediction_sha256']
     case=dict(task=task,fold=fold,horizon=horizon,model=name)
     log(dict(event='START',**case));case_started=time.time()
     hist=name.endswith('history')
     def transformed(indices):
      return pre.apply(data['current'][indices],data['history'][indices],data['prs'][indices],
       data['events'][indices],data['mask'][indices],meta.study.to_numpy()[indices],hist)
     training=pilot.subset(transformed(tr),sample);validation=transformed(va)
     CONTEXT.clear();CONTEXT.update(case=case,fit=fit,rows=[],
      vy=torch.as_tensor(lv['y'],dtype=torch.float64,device='cuda:0'),
      vw=torch.as_tensor(lv['weight'],dtype=torch.float64,device='cuda:0'))
     assert torch.isfinite(CONTEXT['vw']).all() and float(CONTEXT['vw'].sum())>0
     replay(training,validation,lt['y'][sample],pre,directory)
     audit=dict(**case,original_fit=str(directory/'FIT.json'),fit_sha256=sha(directory/'FIT.json'),
      original_checkpoint_sha256=sha(directory/'model_state.pt'),original_probability_sha256=sha(directory/'probability.npy'),
      all_60_original_training_losses_exact=True,all_final_state_tensors_exact=True,
      all_final_validation_probabilities_exact=True,n_training_resampled_visits=len(sample),
      n_validation_visits=len(va),n_known_validation=int(lv['known'].sum()),seconds=time.time()-case_started)
     rows.extend(CONTEXT['rows']);audits.append(audit)
     case_name=f'{task}_fold{fold}_h{horizon}_{name}'
     pd.DataFrame(CONTEXT['rows']).to_csv(OUT/f'{case_name}.csv',index=False)
     dump(OUT/f'{case_name}_verification.json',audit)
     pd.DataFrame(rows).to_csv(OUT/'learning_curves.csv',index=False)
     dump(OUT/'replay_status.json',dict(status='RUNNING',completed=len(audits),seconds=time.time()-started))
     log(dict(event='EXACT_REPLAY_PASS',**case,seconds=audit['seconds']))
     del training,validation
 assert len(audits)==24 and len(rows)==1440
 provenance.update(run_audits=audits,status='PASS',seconds=time.time()-started)
 dump(OUT/'replay_verification.json',provenance)
 dump(OUT/'replay_status.json',dict(status='PASS',completed=24,epochs=1440,seconds=time.time()-started))
 log(dict(event='ALL_24_EXACT_REPLAYS_PASS',seconds=time.time()-started))
if __name__=='__main__':
 try:main()
 except Exception as error:
  dump(OUT/'replay_status.json',dict(status='FAILED',error_type=type(error).__name__,error=str(error)))
  raise
