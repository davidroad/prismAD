"""Pure alignment, disjoint attribution accounting and fixed contrasts."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

GROUPS = ['resilience_CN','other_CN_no_AD','CN_to_AD']
COMPONENTS = ['education_main','cognition_main','past_summary_main','missingness_main',
              'pair_interactions','PRS','trajectory','risk_logit']
COGNITION = {'global_z','PHC_MEM_z','PHC_EXF_z','PHC_LAN_z','PHC_VSP_z'}


def main_group(name):
    if name.endswith('__missing'): return 'missingness_main'
    if name.startswith('study='): return 'study_main'
    if name.startswith('strict_history_'): return 'past_summary_main'
    if name in COGNITION: return 'cognition_main'
    return {'visit_age':'age_main','PHC_Sex':'sex_main','PHC_Education':'education_main',
            'n_domains':'domain_coverage_main'}[name]


def account(names, mains, ledger):
    out = {k: np.zeros(len(mains)) for k in ('age_main','sex_main','education_main','cognition_main',
          'domain_coverage_main','missingness_main','study_main','past_summary_main')}
    for j,name in enumerate(names): out[main_group(name)] += mains[:,j]
    for j,name in [(0,'intercept'),(2,'pair_interactions'),(3,'residual'),(4,'PRS'),(5,'trajectory')]:
        out[name] = ledger[:,j].copy()
    np.testing.assert_allclose(sum(out.values()),ledger.sum(1),atol=3e-6,rtol=1e-5)
    return out


def exact_join(groups, predictions):
    assert groups.subject_id.is_unique
    assert not predictions.duplicated(['subject_id','visit']).any()
    j = groups.merge(predictions,on=['subject_id','visit'],validate='one_to_one',how='inner')
    keep = j.Study.eq(j.study) & np.isclose(j.cognition_age,j.age,rtol=0,atol=1e-6)
    return j.loc[keep].copy()


def centered(values):
    a = np.asarray(values,float)
    return a-a[len(a)//2]


def correct_family(rows):
    d = pd.DataFrame(rows)
    assert len(d)==128
    d['q_all_128'] = multipletests(d.p.fillna(1).to_numpy(),method='fdr_bh')[1]
    return d


def contrasts(d,feature):
    needed = ['index_age','sex','Study','fold']+[f'PC{i}' for i in range(1,6)]+[feature]
    a = d.dropna(subset=needed).copy()
    count = pd.crosstab(a.Study,a.group).reindex(columns=GROUPS,fill_value=0)
    studies = count.index[(count>=5).all(axis=1)]
    a = a[a.Study.isin(studies)]
    n = a.group.value_counts().reindex(GROUPS,fill_value=0)
    base = dict(n_input=len(d),n_supported=len(a),studies='|'.join(studies),
                **{f'n_{g}':int(n[g]) for g in GROUPS})
    rows = [dict(**base,left=GROUPS[0],right=g,beta=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,
                 status='insufficient_support') for g in GROUPS[1:]]
    if n.min()<20: return rows
    if np.ptp(a[feature])<1e-10:
        for r in rows: r.update(status='constant_readout_no_inference')
        return rows
    x = pd.DataFrame({'Intercept':1.,'resilience_CN':a.group.eq(GROUPS[0]).astype(float),
                      'CN_to_AD':a.group.eq(GROUPS[2]).astype(float)},index=a.index)
    age = a.index_age-75
    for name,value in {'age_c':age,'age2':age**2,**{f'PC{i}':a[f'PC{i}'] for i in range(1,6)}}.items():
        if value.std()>1e-12: x[name]=(value-value.mean())/value.std()
    for name in ('sex','Study','fold'):
        x=pd.concat([x,pd.get_dummies(a[name].astype(str),prefix=name,drop_first=True,dtype=float)],axis=1)
    keep=[]; basis=[]
    for name in x:
        v=x[name].to_numpy(float).copy(); norm0=np.linalg.norm(v)
        for _ in range(2):
            for q in basis: v-=q*np.dot(q,v)
        norm=np.linalg.norm(v)
        if norm>1e-9*max(1,norm0): keep.append(name);basis.append(v/norm)
    if not {'resilience_CN','CN_to_AD'}<=set(keep): return rows
    fit=sm.OLS(a[feature],x[keep]).fit()
    leverage=float(fit.get_influence().hat_matrix_diag.max())
    if fit.df_resid<20 or leverage>=.99:
        for r in rows:r['status']='insufficient_df_or_high_leverage'
        return rows
    if np.max(np.abs(fit.resid))<1e-9:
        for r in rows:r['status']='readout_explained_by_adjustment_no_inference'
        return rows
    robust=fit.get_robustcov_results(cov_type='HC3',use_t=True)
    for r in rows:
        v=np.zeros(len(keep));v[keep.index('resilience_CN')]=1
        if r['right']=='CN_to_AD':v[keep.index('CN_to_AD')]=-1
        t=robust.t_test(v);lo,hi=t.conf_int()[0]
        r.update(status='ok',beta=float(np.asarray(t.effect).item()),ci_low=float(lo),ci_high=float(hi),
                 p=float(np.asarray(t.pvalue).item()),residual_df=float(fit.df_resid),max_leverage=leverage)
    return rows
