"""Registered exploratory tests. Inputs read only; no model fitting."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd,scipy,statsmodels,sklearn
from sklearn.metrics import roc_auc_score,average_precision_score
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from .runtime import workspace,input_path,sha
A=workspace();A.mkdir(exist_ok=True,parents=True)
cfg={k:str(input_path(k))for k in ['prediction_directory','no_prs_directory']}
sources={}
def track(p):sources[str(p)]=sha(p);return p
def batch_scores(y,p,weights):
 order=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[order])!=0),len(p)-1]
 ww=weights[:,order];tp=np.cumsum(ww*y[order],axis=1)[:,ends];fp=np.cumsum(ww*(1-y[order]),axis=1)[:,ends]
 precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)
 dt=np.diff(np.c_[np.zeros(len(weights)),tp],axis=1);ap=(precision*dt).sum(axis=1)/tp[:,-1]
 tpr=np.c_[np.zeros(len(weights)),tp/tp[:,-1,None]];fpr=np.c_[np.zeros(len(weights)),fp/fp[:,-1,None]]
 auc=((tpr[:,1:]+tpr[:,:-1])*.5*np.diff(fpr,axis=1)).sum(axis=1)
 return np.c_[auc,ap]
genetic=pd.read_csv(track(input_path('source_data_directory')/'plotting/genetic_contrasts.csv'))
records=[];B=9999
for task in ['CN_MCI','MCI_AD']:
 for horizon in [3,5]:
  metas=[];plus=[];minus=[]
  for fold in [1,2,3]:
   cell=f'{task}/fold{fold}_h{horizon}';p=Path(cfg['prediction_directory'])/'LOCAL_RESTRICTED'/cell;n=Path(cfg['no_prs_directory'])/'LOCAL_RESTRICTED'/cell
   m=pd.read_csv(track(p/'validation.csv'),dtype={'subject_id':str});other=pd.read_csv(track(n/'validation.csv'),dtype={'subject_id':str})
   pd.testing.assert_frame_equal(m,other,check_exact=False,atol=1e-12,rtol=1e-12)
   pp=p/'PRISM_current/probability.npy';nn=n/'PRISM/probability.npy'
   assert not (pp.parent/'seeded_inference_correction/probability.npy').exists()
   a=np.load(track(pp));b=np.load(track(nn));assert len(a)==len(b)==len(m) and m.fold.eq(fold).all()
   keep=m.first_eligible.to_numpy(bool);metas.append(m.loc[keep]);plus.append(a[keep]);minus.append(b[keep])
  m=pd.concat(metas,ignore_index=True);a=np.concatenate(plus);b=np.concatenate(minus)
  assert m.subject_id.is_unique and np.isfinite(a).all() and np.isfinite(b).all()
  ix,unique=pd.factorize(pd.MultiIndex.from_frame(m[['fold','subject_id']]),sort=True)
  folds=np.array([u[0] for u in unique]);groups=[np.flatnonzero(folds==f) for f in [1,2,3]]
  y=m.y.to_numpy(int);base=m.ipcw.to_numpy(float);ok=base>0
  assert np.isfinite(base).all() and (base>=0).all()
  def score(pred,w):
   k=w>0;return np.array([roc_auc_score(y[k],pred[k],sample_weight=w[k]),average_precision_score(y[k],pred[k],sample_weight=w[k])])
  point=[score(pred,base) for pred in [a,b]]
  for pred,s in zip([a,b],point):np.testing.assert_allclose(batch_scores(y,pred,base[None,:])[0],s,atol=1e-12,rtol=0)
  delta=point[0]-point[1];rng=np.random.default_rng(42);draws=[]
  for start in range(0,B,64):
   k=min(64,B-start);counts=np.zeros((k,len(unique)),int)
   for j in range(k):
    for g in groups:counts[j]+=np.bincount(rng.choice(g,len(g),replace=True),minlength=len(unique))
   weights=counts[:,ix]*base;assert ((weights*y).sum(axis=1)>0).all() and ((weights*(1-y)).sum(axis=1)>0).all()
   da=batch_scores(y,a,weights);db=batch_scores(y,b,weights)
   if start==0:
    for j in range(3):
     for pred,v in [(a,da),(b,db)]:np.testing.assert_allclose(v[j],score(pred,weights[j]),atol=1e-12,rtol=0)
   draws.append(da-db)
  draws=np.concatenate(draws)
  for j,metric in enumerate(['weighted_AUROC','weighted_AP']):
   old=genetic[(genetic.task==task)&(genetic.horizon==horizon)&(genetic.model=='PRISM')&(genetic.fold=='pooled_123')&(genetic.metric==metric)&(genetic.subset=='first_eligible_landmark')]
   assert len(old)==1;old=old.iloc[0]
   np.testing.assert_allclose([point[0][j],point[1][j],delta[j]],[old.plusPRS,old.noPRS,old.difference],atol=1e-12,rtol=0)
   np.testing.assert_allclose(np.quantile(draws[:500,j],[.025,.975]),[old.ci_low,old.ci_high],atol=1e-12,rtol=0)
   exceed=int((np.abs(draws[:,j]-delta[j])>=abs(delta[j])).sum());pval=(exceed+1)/(B+1)
   records.append(dict(task=task,horizon=horizon,model='PRISM',metric=metric,contrast='plusPRS_minus_noPRS',plusPRS=point[0][j],noPRS=point[1][j],difference=delta[j],ci_low_original500=old.ci_low,ci_high_original500=old.ci_high,ci_low_9999=np.quantile(draws[:,j],.025),ci_high_9999=np.quantile(draws[:,j],.975),n_subjects=len(m),n_known=int(ok.sum()),n_target=int(y[ok].sum()),p=pval,p_mc_se=np.sqrt(pval*(1-pval)/(B+1)),bootstrap_replicates=B,exceedances=exceed,family_size=8))
  print('Completed',task,horizon,'n',len(m),flush=True)
def stars(q):return '***' if q<.001 else '**' if q<.01 else '*' if q<.05 else ''
d=pd.DataFrame(records);d['q']=multipletests(d.p,method='fdr_bh')[1];d['star']=d.q.map(stars);d.to_csv(A/'Figure2_PRS_increment_tests.csv',index=False)
raw=pd.read_csv(track(input_path('current_profile_components')))
assert len(raw)==893 and raw.display_alias.is_unique and raw.clinical_group.value_counts().to_dict()=={'CN':701,'MCI':97,'AD':95}
raw['prs_quintile']=pd.qcut(raw.raw_PRS,5,labels=False)+1
summ=pd.read_csv(track(input_path('source_data_directory')/'tables/overall_components_fold_prs_quintile_summary.csv'))
for (fold,group,bin_),cell in raw.groupby(['fold','clinical_group','prs_quintile']):
 v=summ[(summ.fold==fold)&(summ.group==group)&(summ.bin==bin_)];assert len(v)==1 and v.iloc[0].n==len(cell)
 np.testing.assert_allclose([cell.cognition_main.mean(),cell.PRS.mean()],[v.iloc[0].mean_signed_C,v.iloc[0].mean_signed_G],atol=1e-12,rtol=0)
omni=[];coeffs=[]
for panel,group in zip('abcd',['All','CN','MCI','AD']):
 cell=raw if group=='All' else raw[raw.clinical_group==group]
 fit=ols('cognition_main ~ C(prs_quintile) + C(fold)',data=cell).fit(cov_type='HC3',use_t=True)
 keys=[k for k in fit.params.index if k.startswith('C(prs_quintile)')];assert len(keys)==4 and np.linalg.matrix_rank(fit.model.exog)==7
 R=np.zeros((4,len(fit.params)))
 for i,k in enumerate(keys):R[i,list(fit.params.index).index(k)]=1
 test=fit.f_test(R)
 omni.append(dict(panel=panel,group=group,n_subjects=len(cell),F=float(test.fvalue),df_num=float(test.df_num),df_denom=float(test.df_denom),p=float(test.pvalue),family_size=4,formula='cognition_main ~ C(prs_quintile) + C(fold)',covariance='HC3',min_quintile_n=int(cell.groupby('prs_quintile').size().min()),max_leverage=float(fit.get_influence().hat_matrix_diag.max())))
 for k in fit.params.index:coeffs.append(dict(panel=panel,group=group,term=k,beta=fit.params[k],se_HC3=fit.bse[k],p_nominal=fit.pvalues[k]))
f=pd.DataFrame(omni);f['q']=multipletests(f.p,method='fdr_bh')[1];f['star']=f.q.map(stars);f.to_csv(A/'Figure3_cognitive_quintile_omnibus_tests.csv',index=False)
pd.DataFrame(coeffs).to_csv(A/'Figure3_OLS_coefficients.csv',index=False)
for p,h in sources.items():assert sha(p)==h,'Input modified: '+p
receipt=dict(script_sha256=sha(__file__),input_sha256=sources,raw_inputs_unchanged=True,models_refitted=False,restricted_rows_exported=False,first_500_draws_match_all_original_CIs=True,score_functions_match_sklearn=True,Figure3_group_bin_means_match_frozen_figure=True,Figure2_tests=len(d),Figure2_significant=int((d.q<.05).sum()),Figure3_tests=len(f),Figure3_significant=int((f.q<.05).sum()),versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__,sklearn=sklearn.__version__))
(A/'test_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(d[['task','horizon','metric','difference','p','q','star']].to_string(index=False));print(f[['panel','group','F','p','q','star']].to_string(index=False))
