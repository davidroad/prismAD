"""Fixed four-comparison education extension; aggregate reports, local-only rows."""
from pathlib import Path
import json
import os
import platform
import sys
import time
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.formula.api as smf

from types import SimpleNamespace
from . import phenotypes as core
from .cognitive_course import normalize
from .biomarkers import candidates
from .runtime import workspace,input_path,sha
HERE=workspace();OUT=HERE/'results';PRIVATE=HERE/'LOCAL_RESTRICTED'
prev=SimpleNamespace(COG=input_path('cognition'),PET=input_path('amyloid_pet'),FROZEN=input_path('genotype_roster'),normalize=normalize,candidates=candidates,sha=sha)


def supported(d, pet):
    cols = ['education','age','sex','Study','PRS','g','measurement_lag'] + core.PCS
    if pet:
        cols += ['tracer']
    b = d.dropna(subset=cols).copy()
    n = pd.crosstab(b.Study,b.g).reindex(columns=[0,1],fill_value=0)
    return b[b.Study.isin(n.index[n.ge(5).all(axis=1)])].copy()


def independent_check(b, r, pet):
    if r['status'] != 'ok':
        return dict(status='not_estimable_no_numeric_test')
    b = b.copy()
    b['ac'] = b.age - 75
    b['ac2'] = b.ac**2
    terms = ['g','C(sex)','C(Study)']
    for col in ['ac','ac2','PRS'] + core.PCS + ['measurement_lag']:
        if b[col].std() > 1e-15:
            b[col] = (b[col]-b[col].mean())/b[col].std()
            terms.append(col)
    if pet:
        terms.append('C(tracer)')
    fit = smf.ols('education ~ ' + ' + '.join(terms), b).fit(cov_type='HC3',use_t=True)
    k = fit.model.exog_names.index('g')
    x, y = fit.model.exog, fit.model.endog
    assert np.linalg.matrix_rank(x) == x.shape[1], 'Independent formula needs explicit collinearity resolution'
    pinv = np.linalg.pinv(x)
    beta = pinv @ y
    residual = y - x @ beta
    leverage = np.einsum('ij,ji->i',x,pinv)
    robust_cov = (pinv * (residual/(1-leverage))**2) @ pinv.T
    se = np.sqrt(robust_cov[k,k])
    p = 2*stats.t.sf(abs(beta[k]/se),len(y)-x.shape[1])
    ci = beta[k] + np.array([-1,1])*stats.t.ppf(.975,len(y)-x.shape[1])*se
    np.testing.assert_allclose([beta[k],*ci,p],[r['beta'],r['ci_low'],r['ci_high'],r['p']],rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(fit.params['g'],r['beta'],rtol=1e-7,atol=1e-9)
    return dict(status='passed',n=len(b),formula_beta=float(fit.params['g']),manual_HC3_se=float(se),manual_p=float(p))


def describe(d, name, scope):
    rows = []
    for group in ['CN','AD']:
        z = d[d.status.eq(group)]
        v = z.education.dropna()
        rows.append(dict(comparison=name,scope=scope,group=group,n=len(z),
            n_education=len(v),missing_education=int(z.education.isna().sum()),
            mean=float(v.mean()),sd=float(v.std()),median=float(v.median()),
            q25=float(v.quantile(.25)),q75=float(v.quantile(.75)),
            mean_last_age=float(z.age.mean()),n_topcoded_age=int(z.age_topcoded.sum()),
            n_later_unclassified=int(z.later_unclassified.sum())))
    return rows


def main():
    started = time.time()
    os.umask(0o077)
    OUT.mkdir(exist_ok=True)
    PRIVATE.mkdir(exist_ok=True,mode=0o700)
    if (OUT/'manifest.json').exists():
        raise SystemExit('Completed output exists; no silent overwrite')
    boundsfile = None
    paths = [prev.FROZEN,prev.COG,prev.PET]
    hashes = {str(p):prev.sha(p) for p in paths}
    roster=pd.read_csv(prev.FROZEN,dtype={'subject_id':str,'sample_id':str})
    roster=roster[['subject_id','sample_id','PRS']+core.PCS]
    f=core.construct_index(pd.read_csv(prev.COG,dtype={'SUBJID':str},low_memory=False),roster)
    assert f.subject_id.is_unique
    bounds=SimpleNamespace(T_lower=0.473704771704382,T_upper=1.1350529724900473)
    high = f.PRS.between(bounds.T_lower,bounds.T_upper,inclusive='both')
    old = f.age.ge(75) | f.age_topcoded.fillna(False)
    d = f[high & old & f.status.isin(['CN','AD'])].copy()
    raw = pd.read_csv(prev.COG,dtype={'SUBJID':str},low_memory=False)
    c = prev.normalize(raw,f.subject_id)
    latest = c.groupby('subject_id',sort=False).tail(1)[['subject_id','visit','diagnosis']].rename(
        columns={'visit':'latest_recorded_visit','diagnosis':'latest_recorded_diagnosis'})
    d = d.merge(latest,on='subject_id',validate='one_to_one')
    d['later_unclassified'] = d.latest_recorded_visit.gt(d.visit)
    d['strict_latest'] = d.latest_recorded_visit.eq(d.visit) & d.latest_recorded_diagnosis.eq(d.diagnosis)
    assert d.subject_id.is_unique and d.status.isin(['CN','AD']).all()
    # Independently retrieve education from the authoritative visit, not row order.
    check = d[['subject_id','visit','education']].merge(c[['subject_id','visit','education']],
        on=['subject_id','visit'],suffixes=('_frozen','_raw'),validate='one_to_one')
    np.testing.assert_allclose(check.education_frozen,check.education_raw,equal_nan=True)
    d['g'] = d.status.eq('CN').astype(int)
    d['measurement_lag'] = 0.
    idx,audit = prev.candidates(pd.read_csv(prev.PET,dtype={'SUBJID':str},low_memory=False),c,'PET',0)
    finalcols = ['subject_id','Study','status','age','age_topcoded','PRS','g','later_unclassified','strict_latest'] + core.PCS
    pet = idx.merge(d[finalcols],on=['subject_id','Study'],validate='one_to_one')
    first_ad = c[c.diagnosis.eq(3)].groupby('subject_id',sort=False).head(1)[['subject_id','visit','cognition_age']].rename(
        columns={'visit':'first_AD_visit','cognition_age':'first_AD_age'})
    pet = pet.merge(first_ad,on='subject_id',how='left',validate='one_to_one')
    # Both visit ordering and dated onset required; no post-AD scan is eligible.
    valid = pet.status.eq('CN') | (pet.first_AD_visit.gt(pet.visit) & pet.first_AD_age.gt(pet.index_age))
    audit['AD_not_demonstrably_after_CN_PET'] = int((~valid).sum())
    pet = pet[valid].copy()
    pet['measurement_lag'] = pet.age - pet.cognition_age
    assert pet.loc[pet.age.notna(),'measurement_lag'].ge(-1e-8).all()
    cohorts = {'primary_final75':(d,False),
               'latest_record_confirmed':(d[d.strict_latest].copy(),False),
               'CN_stage_PET':(pet,True),
               'CN_stage_PET_amyloid_positive':(pet[pet.amyloid_positive.eq(1)].copy(),True)}
    results,descriptions,study,checks = [],[],[],[]
    for name,(b,ispet) in cohorts.items():
        assert b.subject_id.is_unique
        assert b.PRS.between(bounds.T_lower,bounds.T_upper).all()
        assert (b.age.ge(75) | b.age_topcoded).all()
        b.to_parquet(PRIVATE/(name+'.parquet'),index=False)
        r = core.adjusted_effect(b,'education','measurement_lag',extra_cat=('tracer',) if ispet else ())
        r['comparison'] = name
        results.append(r)
        support = supported(b,ispet)
        assert len(support) == r['n']
        checks.append(dict(comparison=name,**independent_check(support,r,ispet)))
        descriptions += describe(b,name,'all_eligible') + describe(support,name,'model_common_support')
        for (s,g),z in b.groupby(['Study','status']):
            study.append(dict(comparison=name,Study=s,group=g,n=len(z),n_education=int(z.education.notna().sum()),
                education_mean=float(z.education.mean()),in_adjusted_support=s in set(support.Study)))
        print('COMPARISON',name,'counts',b.status.value_counts().to_dict(),'status',r['status'],flush=True)
    r = core.bh_with_failures(pd.DataFrame(results))
    r.to_csv(OUT/'adjusted_education_tests.csv',index=False)
    pd.DataFrame(descriptions).to_csv(OUT/'education_descriptives.csv',index=False)
    pd.DataFrame(study).to_csv(OUT/'study_support.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'independent_checks.csv',index=False)
    for p,h in hashes.items():
        assert prev.sha(p) == h
    validation = dict(assessment='Share with noted caveats',unique_subject_per_comparison=True,
        frozen_education_matches_raw_visit=True,CN_counts_match_prior_413_343_95_54=True,
        AD_PET_precedes_first_AD=True,independent_HC3_checks=checks,inputs_and_master_unchanged=True,
        planned_tests=4,fit_no_models_below_support_gate=True,selection_QC=audit,
        caveats=['Retrospective selected survivors; not causal educational protection.',
                 'Last-known CN does not establish CN at later unclassified visits.',
                 'PET subsets cannot borrow whole-cohort evidence; not pathology-matched.',
                 'Exploratory extension; family q does not cover historical project searches.'])
    (OUT/'CHECKS.json').write_text(json.dumps(validation,indent=2)+'\n')
    code=[Path(__file__),Path(core.__file__)]
    manifest = dict(inputs=hashes,code={str(p):prev.sha(p) for p in code},
        outputs={p.name:prev.sha(p) for p in OUT.iterdir() if p.is_file()},
        environment=dict(executable=sys.executable,python=platform.python_version(),numpy=np.__version__,
                         pandas=pd.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__),
        elapsed_seconds=round(time.time()-started,2),thresholds=dict(age=75,PRS_lower=float(bounds.T_lower),PRS_upper=float(bounds.T_upper)))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(r[['comparison','status','n1','n0','beta','ci_low','ci_high','p','q']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
