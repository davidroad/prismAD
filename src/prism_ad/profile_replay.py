#!/usr/bin/env python3
"""Replay frozen CN-to-first-MCI current-state checkpoints; no fitting or tests."""
from pathlib import Path
import os,sys,json,hashlib,time,signal,random
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
from .runtime import workspace,input_path,settings
OUT=workspace()
PILOT=input_path('prediction_directory')
GROUPS=input_path('cn_course_indices')
OLD=input_path('prior_targeted_contributions')
import numpy as np,pandas as pd,torch
from . import training as runner
from .features import Transform,CURRENT
START=time.time()
os.umask(0o077)
for d in ['restricted_local','audit']:(OUT/d).mkdir(parents=True,exist_ok=True,mode=0o700)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');p.chmod(0o600)
def timeout(*_):raise TimeoutError('600-second bounded checkpoint replay')
signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
runner.prohibit_outbound()
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.backends.mha.set_fastpath_enabled(False)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.cuda.set_per_process_memory_fraction(.10,device=0)
runner.reset_seed()
contract=dict(unit='one first-eligible CN landmark per held-out participant, folds 1–3',
 checkpoint='PRISM_current CN_MCI, three-year horizon, one frozen checkpoint per fold',
 predictors='Cognition and all remaining clinical inputs are measured at CN prediction. Later CN/MCI/AD labels are retrospective grouping variables, never predictors.',
 components='Cognitive-test main functions only: global_z, PHC_MEM_z, PHC_EXF_z, PHC_LAN_z, PHC_VSP_z. PRS term from native ledger column4. Both in log-odds units.',
 group_definition={'CN':'Final explicitly classified status CN and no recorded AD','MCI':'Final explicitly classified status MCI and no recorded AD','AD':'Final status AD and first recorded AD dated strictly after the CN index'},
 grouping_window='Later status is defined over all available recorded follow-up, not restricted to the three-year forecast horizon.',
 group_CNFU='No added ≥3-year non-AD follow-up restriction; therefore this exploratory set is broader than the old 688-person clinical-course set.',
 interpretation='C and G are signed contributions to predicted log odds. Their exponentials are multiplicative odds factors, not probability shares. Fold-specific scales and post hoc grouping limit interpretation.',
 raw_data='Read only; output restricted to this new analysis directory.',model_fitting=False,new_statistical_tests=False)
dump(OUT/'audit'/'analysis_contract.json',contract)
sources=[PILOT/'LOCAL_RESTRICTED/features.npz',PILOT/'LOCAL_RESTRICTED/metadata.csv',GROUPS,OLD]
raw=np.load(sources[0],allow_pickle=False)
meta=pd.read_csv(sources[1],dtype={'subject_id':str})
ix=np.flatnonzero(meta.task.eq('CN_MCI'))
m=meta.iloc[ix].reset_index(drop=True)
a={k:raw[k][ix] for k in raw.files}
g=pd.read_parquet(GROUPS);g.subject_id=g.subject_id.astype(str)
assert g.subject_id.is_unique
old=pd.read_parquet(OLD);old=old[old.model.eq('PRISM_current')&old.horizon.eq(3)].copy()
assert old.subject_id.is_unique
blocks=[];replay=[]
for fold in [1,2,3]:
 tr=np.flatnonzero(m.fold.ne(fold));va=np.flatnonzero(m.fold.eq(fold))
 assert not set(m.iloc[tr].subject_id)&set(m.iloc[va].subject_id)
 pre=Transform().fit(a['current'][tr],a['history'][tr],a['prs'][tr],a['events'][tr],a['mask'][tr])
 cell=PILOT/'LOCAL_RESTRICTED/CN_MCI'/f'fold{fold}_h3';path=cell/'PRISM_current'
 assert pre.record()==json.loads((cell/'transform.json').read_text())
 v=pd.read_csv(cell/'validation.csv',dtype={'subject_id':str})
 assert v[['subject_id','visit','fold']].reset_index(drop=True).equals(m.iloc[va][['subject_id','visit','fold']].reset_index(drop=True))
 def bundle(rows):return pre.apply(a['current'][rows],a['history'][rows],a['prs'][rows],a['events'][rows],a['mask'][rows],m.study.to_numpy()[rows],False)
 train=bundle(tr);valid=bundle(va)
 draw=np.load(cell/'shared_bootstrap_local_indices.npy')
 runner.reset_seed()
 model=runner.StructuralPrismAD(train['names'],train['static'][draw],'relative_history',pre).cuda()
 model.load_state_dict(torch.load(path/'model_state.pt',map_location='cuda:0',weights_only=True))
 t=runner.tensors(valid);p,extras=runner.prism_prediction(model,t)
 expected=np.load(path/'probability.npy');native=np.load(path/'native_explanations.npz')
 assert np.array_equal(p,expected)
 assert np.array_equal(extras['logit'],native['logit'])
 assert np.array_equal(extras['ledger'],native['ledger'])
 mains=[]
 with torch.no_grad():
  for j in range(0,len(v),256):mains.append(model.clinical_anchor(t['static'][j:j+256])['clinical_main_contribution'].cpu().numpy())
 mains=np.concatenate(mains)
 np.testing.assert_allclose(mains.sum(1),extras['ledger'][:,1],atol=3e-6,rtol=1e-5)
 cogcols=[valid['names'].index(k) for k in ['global_z','PHC_MEM_z','PHC_EXF_z','PHC_LAN_z','PHC_VSP_z']]
 block=v[['subject_id','visit','study','age','fold','first_eligible']].copy()
 cognitive=np.zeros(len(v),dtype=np.float64)
 for col in cogcols:cognitive+=mains[:,col]
 block['cognition_main']=cognitive
 block['PRS']=extras['ledger'][:,4]
 block['risk_logit']=extras['logit'];block['predicted_risk']=p
 block['raw_PRS']=a['prs'][va]
 prior=old[old.fold.eq(fold)].merge(block,on=['subject_id','visit'],suffixes=('_old','_new'),validate='one_to_one')
 assert len(prior)==int(old.fold.eq(fold).sum())
 errors={}
 for c in ['cognition_main','PRS','risk_logit','predicted_risk']:
  x=prior[c+'_old'].to_numpy();y=prior[c+'_new'].to_numpy()
  np.testing.assert_array_equal(x,y)
  errors[c]=float(np.max(np.abs(x-y)))
 blocks.append(block)
 sources.extend([cell/'validation.csv',cell/'transform.json',cell/'shared_bootstrap_local_indices.npy',path/'model_state.pt',path/'probability.npy',path/'native_explanations.npz'])
 replay.append(dict(fold=fold,n_validation_rows=len(v),n_first_eligible=int(v.first_eligible.sum()),targeted_profile_overlap=len(prior),old_component_max_abs_errors=errors,probability_array_exact=True,native_logit_exact=True,native_ledger_exact=True,clinical_main_ledger_max_abs_error=float(np.abs(mains.sum(1)-extras['ledger'][:,1]).max())))
 print(json.dumps(replay[-1]),flush=True)
 del model,t;torch.cuda.empty_cache()
pred=pd.concat(blocks,ignore_index=True)
first=pred[pred.first_eligible].copy()
assert first.subject_id.is_unique and len(first)==908
joined=g.merge(first,on=['subject_id','visit'],suffixes=('_group',''),how='inner',validate='one_to_one')
alignment=joined.Study.eq(joined.study)&np.isclose(joined.cognition_age,joined.age,atol=1e-6,rtol=0)
assert alignment.all()
np.testing.assert_allclose(joined.PRS_group,joined.raw_PRS,atol=1e-12,rtol=1e-12)
no_prior=~joined.prior_MCI_AD
cn=joined.final_status.eq('CN')&~joined.first_AD_recorded&no_prior
mci=joined.final_status.eq('MCI')&~joined.first_AD_recorded&no_prior
ad=joined.final_status.eq('AD')&joined.first_AD_recorded&joined.first_AD_elapsed.gt(0)&joined.course_status.eq('first_AD_dated_after_CN_index')&no_prior
keep=cn|mci|ad
selected=joined[keep].copy()
selected['clinical_group']=np.select([cn[keep],mci[keep],ad[keep]],['CN','MCI','AD'],default='ERROR')
assert not selected.clinical_group.eq('ERROR').any()
selected=selected.sort_values(['fold','study','age','visit']).reset_index(drop=True)
selected['display_alias']=['P'+str(i+1).zfill(4) for i in range(len(selected))]
cols=['display_alias','fold','clinical_group','PRS','cognition_main','risk_logit','predicted_risk','raw_PRS']
minimal=selected[cols].copy()
assert not minimal.isna().any().any()
csv=OUT/'restricted_local/relative_components_controlled.csv';minimal.to_csv(csv,index=False);csv.chmod(0o600)
ids=selected[['display_alias','subject_id','visit','Study','study','cognition_age','age','final_status','course_status','classified_followup_span','first_AD_elapsed','first_AD_recorded','intermediate_MCI','prior_MCI_AD']].copy()
ids.to_parquet(OUT/'restricted_local/private_join_keys.parquet',index=False)
summary=minimal.groupby(['fold','clinical_group']).agg(n=('display_alias','size'),signed_PRS_mean=('PRS','mean'),signed_cognition_mean=('cognition_main','mean'),raw_PRS_min=('raw_PRS','min'),raw_PRS_max=('raw_PRS','max'))
summary.to_csv(OUT/'audit/fold_group_summary.csv')
codefiles=[Path(x.__file__) for x in list(sys.modules.values()) if getattr(x,'__file__',None) and 'prism_ad' in str(getattr(x,'__file__','')) and str(getattr(x,'__file__','')).endswith('.py')]
sources+=codefiles+[Path(__file__).resolve()]
inputs={str(p):sha(p) for p in sorted(set(sources))}
outputs={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'restricted_local').glob('*'))}
manifest=dict(status='PASS_FROZEN_CHECKPOINT_REPLAY_EXPLORATORY_DESCRIPTIVE_ONLY',elapsed_seconds=time.time()-START,
 contract=contract,clinical_group_counts={k:int(v) for k,v in minimal.clinical_group.value_counts().items()},
 fold_group_counts={str(k):{a:int(b) for a,b in v.items()} for k,v in pd.crosstab(minimal.fold,minimal.clinical_group).to_dict(orient='index').items()},
 selection=dict(first_eligible_validation_subjects=len(first),exact_original_CN_index_joins=len(joined),study_age_mismatches=int((~alignment).sum()),missing_original_index_join=len(first)-len(joined),excluded_after_join=int((~keep).sum()),excluded_final_status_counts=joined.loc[~keep,'final_status'].value_counts().to_dict(),included=len(minimal)),
 replay=replay,targeted_exact_overlap=sum(x['targeted_profile_overlap'] for x in replay),minimal_file=str(csv),minimal_columns=cols,
 raw_PRS='Same unstandardized frozen score across all folds; checkpoint standardized PRS terms have fold-specific mean/scale.',
 no_subject_identifiers_in_minimal_csv=True,no_model_training=True,no_new_prediction_task=True,no_new_statistical_tests=True,
 sources_sha256=inputs,outputs_sha256=outputs,python=sys.version,torch=torch.__version__,device=torch.cuda.get_device_name(0))
dump(OUT/'audit/manifest.json',manifest)
signal.alarm(0)
print(json.dumps({k:manifest[k] for k in ['status','elapsed_seconds','clinical_group_counts','fold_group_counts','selection','minimal_file','targeted_exact_overlap']},indent=2),flush=True)
