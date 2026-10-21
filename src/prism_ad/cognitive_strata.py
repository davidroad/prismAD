"""Post hoc AD--CN genetic-factor comparisons within shared cognitive-risk strata.
Input rows are controlled participant-level model contributions, not public data.
All C deciles are defined in the complete initially-CN profile sample first.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from .runtime import input_path,workspace,sha
def design(z,global_model=False,expanded=False):
 x=pd.DataFrame({'Intercept':np.ones(len(z)),'AD':z.clinical_group.eq('AD').astype(float)},index=z.index)
 c=z.cognition_main.to_numpy();x['C']=(c-c.mean())/c.std()
 if global_model:x=pd.concat([x,pd.get_dummies(z.C_decile.astype(str),prefix='C_decile',drop_first=True,dtype=float)],axis=1)
 for name in ['fold']+(['sex','Study'] if expanded else []):
  x=pd.concat([x,pd.get_dummies(z[name].astype(str),prefix=name,drop_first=True,dtype=float)],axis=1)
 if expanded:
  age=z.index_age-75
  for name,v in dict(age=age,age2=age**2,**{f'PC{i}':z[f'PC{i}'] for i in range(1,6)}).items():
   if v.std()>1e-12:x[name]=(v-v.mean())/v.std()
 # Preserve target first; eliminate only linearly redundant nuisance columns deterministically.
 keep=[];basis=[]
 for name in x:
  v=x[name].to_numpy(float).copy();norm0=np.linalg.norm(v)
  for _ in range(2):
   for q in basis:v-=q*np.dot(q,v)
  norm=np.linalg.norm(v)
  if norm>1e-9*max(1,norm0):keep.append(name);basis.append(v/norm)
 assert 'AD' in keep
 return x[keep]

def fit(z,name,global_model=False,expanded=False):
 input_n=len(z)
 if expanded:
  need=['index_age','sex','Study']+[f'PC{i}' for i in range(1,6)]
  z=z.dropna(subset=need)
  ct=pd.crosstab(z.Study,z.clinical_group).reindex(columns=['AD','CN'],fill_value=0)
  supported=ct.index[(ct>=5).all(axis=1)]
  z=z[z.Study.isin(supported)]
 counts=z.clinical_group.value_counts()
 row=dict(test=name,n_input=input_n,n=len(z),n_AD=int(counts.get('AD',0)),n_CN=int(counts.get('CN',0)),status='ok',beta=np.nan,ci_low=np.nan,ci_high=np.nan,factor_ratio=np.nan,ratio_ci_low=np.nan,ratio_ci_high=np.nan,se=np.nan,t=np.nan,df=np.nan,p=np.nan,max_leverage=np.nan)
 if min(row['n_AD'],row['n_CN'])<5:row['status']='group_n_below_5';return row
 x=design(z,global_model,expanded);X=x.to_numpy(float);y=z.PRS.to_numpy()
 inv=np.linalg.inv(X.T@X);B=inv@X.T@y;r=y-X@B
 h=np.einsum('ij,jk,ik->i',X,inv,X);df=len(y)-X.shape[1]
 row.update(df=df,max_leverage=float(h.max()),design_rank=X.shape[1],studies=int(z.Study.nunique()))
 if df<20 or h.max()>=.99:row['status']='insufficient_df_or_high_leverage';return row
 if np.max(np.abs(r))<1e-9:row['status']='zero_residual_variance';return row
 V=inv@(X.T@((r/(1-h))[:,None]**2*X))@inv
 j=list(x.columns).index('AD');beta=B[j];se=np.sqrt(V[j,j]);T=beta/se
 low=beta-t.ppf(.975,df)*se;high=beta+t.ppf(.975,df)*se
 row.update(beta=beta,ci_low=low,ci_high=high,factor_ratio=np.exp(beta),ratio_ci_low=np.exp(low),ratio_ci_high=np.exp(high),se=se,t=T,p=2*t.sf(abs(T),df))
 return row

def analyze(frame):
    d=frame.copy()
    assert d.clinical_group.isin(['CN','MCI','AD']).all()
    d['C_decile']=pd.qcut(d.cognition_main,10,labels=False)+1
    d=d[d.clinical_group.isin(['CN','AD'])].copy()
    rows=[fit(d,'global',True)]
    rows +=[fit(d[d.C_decile.eq(i)],'C_decile_'+str(i))for i in range(1,11)]
    rows +=[fit(d,'expanded_sensitivity',True,True)]
    out=pd.DataFrame(rows)
    p=out.p.fillna(1).to_numpy();ix=np.argsort(p);q=np.empty(len(p))
    q[ix]=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1)
    out['q_family12']=q
    out['marker']=out.apply(lambda r:''if r.status!='ok'else
        '***'if r.q_family12<.001 else'**'if r.q_family12<.01 else
        '*'if r.q_family12<.05 else'(*)'if r.p<.05 else'',axis=1)
    return out

def main():
    rawpath=input_path('current_profile_components')
    keypath=input_path('profile_join_keys')
    covpath=input_path('cn_course_indices')
    original={str(p):sha(p)for p in [rawpath,keypath,covpath]}
    raw=pd.read_csv(rawpath);key=pd.read_parquet(keypath);cov=pd.read_parquet(covpath)
    for block in [key,cov]:block['subject_id']=block.subject_id.astype(str)
    cols=['subject_id','visit','Study','index_age','sex']+[f'PC{i}'for i in range(1,6)]
    d=raw.merge(key,on='display_alias',validate='one_to_one').merge(
        cov[cols],on=['subject_id','visit'],validate='one_to_one',suffixes=('','_cov'))
    assert len(d)==893 and d.display_alias.is_unique and(d.Study==d.Study_cov).all()
    assert np.allclose(d.age,d.index_age,atol=1e-6,rtol=0)
    out=analyze(d);dest=workspace();dest.mkdir(exist_ok=True,parents=True)
    out.to_csv(dest/'Figure5_C_decile_AD_CN_tests.csv',index=False)
    assert all(sha(p)==h for p,h in original.items())
    (dest/'execution.json').write_text(json.dumps(dict(input_sha256=original,
        script_sha256=sha(__file__),n=893,AD_CN_n=796,test_slots=12,
        models_refitted=False,participant_rows_exported=False),indent=2)+'\n')
if __name__=='__main__':main()
