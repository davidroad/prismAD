"""Fixed CN-origin groups. No effect-direction-dependent selection."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

GROUPS = ['resilience_CN', 'other_CN_no_AD', 'CN_to_AD']
PCS = [f'PC{i}' for i in range(1, 6)]
PAIRS = [('resilience_CN', 'other_CN_no_AD'),
         ('resilience_CN', 'CN_to_AD'), ('CN_to_AD', 'other_CN_no_AD')]


def exact_age(x):
    a = pd.to_numeric(x, errors='coerce')
    return a.where(a.between(0, 90, inclusive='neither'))


def normalize(raw, ids):
    c = raw[raw.SUBJID.isin(ids)].copy().rename(columns={
        'SUBJID':'subject_id', 'PHC_Visit':'visit', 'PHC_Diagnosis':'diagnosis'})
    c['visit'] = pd.to_numeric(c.visit, errors='coerce')
    c['diagnosis'] = pd.to_numeric(c.diagnosis, errors='coerce')
    c['cognition_age'] = exact_age(c.PHC_Age_Cognition)
    c['education'] = pd.to_numeric(c.PHC_Education, errors='coerce')
    c['education'] = c.education.where(c.education.ge(0))
    c['sex'] = pd.to_numeric(c.PHC_Sex, errors='coerce').where(c.PHC_Sex.isin([1,2]))
    assert c.visit.notna().all() and not c.duplicated(['subject_id','visit']).any()
    assert not (c.groupby('subject_id').Study.nunique()>1).any()
    c = c.sort_values(['subject_id','visit'], kind='stable')
    disease = c.diagnosis.isin([2,3]).astype(int)
    c['prior_MCI_AD'] = disease.groupby(c.subject_id).cumsum().sub(disease).gt(0)
    return c


def make_clinical_indices(c):
    first = c[c.diagnosis.isin([1,2,3])].groupby('subject_id',sort=False).head(1)
    d = first[first.diagnosis.eq(1) & first.cognition_age.notna()].copy()
    d['index_age'] = d.cognition_age
    d['pet_age'] = np.nan
    d['centiloids'] = np.nan
    d['amyloid_positive'] = np.nan
    d['tracer'] = pd.NA
    d['pet_cognition_gap'] = 0.
    return d


def make_pet_indices(raw, c):
    raw = raw[raw.SUBJID.isin(c.subject_id)].copy()
    audit = dict(linked_rows=len(raw), linked_subjects=int(raw.SUBJID.nunique()))
    qc = raw.PHC_QC_IMAGE.eq(1) & raw.PHC_QC_TIMING.eq(1)
    audit['qc_excluded_rows'] = int((~qc).sum())
    p = raw[qc].copy().rename(columns={'SUBJID':'subject_id', 'PHC_Visit':'visit'})
    p['pet_age'] = exact_age(p.PHC_Age_PET)
    p['centiloids'] = pd.to_numeric(p.PHC_CENTILOIDS, errors='coerce')
    p['amyloid_positive'] = pd.to_numeric(p.PHC_AMYLOID_STATUS,errors='coerce').where(p.PHC_AMYLOID_STATUS.isin([0,1]))
    p['tracer'] = p.PHC_TRACER.astype('string')
    p['pet_source_diagnosis'] = pd.to_numeric(p.PHC_Diagnosis,errors='coerce')
    columns = ['subject_id','Study','visit','pet_age','centiloids','amyloid_positive','tracer','pet_source_diagnosis']
    p = p[columns].merge(c, on=['subject_id','Study','visit'],how='inner',validate='many_to_one')
    audit['cognition_joined_rows'] = len(p)
    mismatch = p.pet_source_diagnosis.notna() & p.diagnosis.notna() & p.pet_source_diagnosis.ne(p.diagnosis)
    audit['source_diagnosis_disagreement_rows'] = int(mismatch.sum())
    p['pet_cognition_gap'] = (p.pet_age-p.cognition_age).abs()
    valid = (~mismatch & p.diagnosis.eq(1) & ~p.prior_MCI_AD
             & p.pet_age.notna() & p.cognition_age.notna()
             & np.isfinite(p.centiloids) & p.pet_cognition_gap.le(90/365.25+1e-8))
    p = p[valid].copy()
    audit['eligible_CN_scan_rows_before_dedup'] = len(p)
    cols = ['subject_id','Study','visit','pet_age','cognition_age','education','sex',
            'centiloids','amyloid_positive','tracer','pet_cognition_gap','diagnosis','prior_MCI_AD']
    p = p[cols].drop_duplicates()
    ties = p.duplicated(['subject_id','visit','pet_age'],keep=False)
    audit['discordant_tied_scan_rows'] = int(ties.sum())
    p = p[~ties].sort_values(['subject_id','visit','pet_age'],kind='stable')
    d = p.groupby('subject_id',sort=False).head(1).copy()
    d['index_age'] = np.maximum(d.pet_age,d.cognition_age)
    audit['earliest_CN_PET_subjects'] = len(d)
    assert not d.subject_id.duplicated().any()
    assert (d.pet_age <= d.index_age).all()
    return d,audit


def add_course(indices, visits, frozen):
    roster = frozen[['subject_id','sample_id','PRS','status','T_20']+PCS].rename(
        columns={'status':'final_status','T_20':'frozen_T20'})
    d = indices.merge(roster,on='subject_id',how='left',validate='one_to_one')
    histories = {sid:g for sid,g in visits.groupby('subject_id',sort=False)}
    rows=[]
    for b in d.itertuples(index=False):
        history=histories[b.subject_id]
        later=history[(history.visit>b.visit)&history.diagnosis.isin([1,2,3])]
        dated=later[later.cognition_age.notna()]
        prior=history[history.visit<b.visit]
        ad=later[later.diagnosis.eq(3)]
        span=float(dated.cognition_age.max()-b.index_age) if len(dated) else np.nan
        first_time=float(ad.iloc[0].cognition_age-b.index_age) if len(ad) else np.nan
        group='not_in_three_groups'
        reason='final_MCI_unknown_or_other_state'
        if prior.diagnosis.isin([2,3]).any():
            reason='prior_MCI_or_AD'
        elif b.final_status=='CN' and not len(ad):
            group='resilience_CN' if b.frozen_T20==1 else 'other_CN_no_AD'
            reason='CN_at_index_and_last_no_observed_AD'
        elif b.final_status=='AD' and len(ad):
            if np.isfinite(first_time) and first_time>0:
                group='CN_to_AD'; reason='first_AD_dated_after_CN_index'
            else:
                reason='first_AD_time_unknown_or_not_positive'
        rows.append(dict(subject_id=b.subject_id,group=group,course_status=reason,
            classified_followup_span=max(span,0) if np.isfinite(span) else np.nan,
            first_AD_elapsed=first_time,n_later_classified=len(later),
            all_later_classified_CN=bool(len(later) and later.diagnosis.eq(1).all()),
            intermediate_MCI=bool(later.diagnosis.eq(2).any()),
            first_AD_recorded=bool(len(ad))))
    return d.merge(pd.DataFrame(rows),on='subject_id',validate='one_to_one')


def scope_mask(d,scope):
    cn=d.group.isin(GROUPS[:2]); event=d.group.eq('CN_to_AD')
    if scope=='availability_only':
        return cn|event
    horizon=5 if scope=='fixed5' else 3
    controls=cn & d.classified_followup_span.ge(horizon)
    if scope in ['fixed3','fixed5']:
        event &= d.first_AD_elapsed.le(horizon)
    if scope=='persistent_CN':
        controls &= d.all_later_classified_CN
    return controls|event


def summaries(d,cohort,scope):
    rows=[]
    for g in GROUPS:
        block=d[d.group.eq(g)]
        for feat in ['education','centiloids','index_age','classified_followup_span','first_AD_elapsed','PRS']:
            v=block[feat].dropna()
            rows.append(dict(cohort=cohort,scope=scope,group=g,feature=feat,n_group=len(block),
                n_observed=len(v),mean=float(v.mean()),sd=float(v.std()),median=float(v.median()),
                q25=float(v.quantile(.25)),q75=float(v.quantile(.75)),
                n_intermediate_MCI=int(block.intermediate_MCI.sum()),
                n_amyloid_positive=int(block.amyloid_positive.eq(1).sum()),
                n_amyloid_status_observed=int(block.amyloid_positive.notna().sum())))
    return rows


def joint_model(d,feature,pet=False):
    cols=[feature,'index_age','sex','Study']+PCS
    if pet:
        cols+=['tracer','pet_cognition_gap']
    a=d.dropna(subset=cols).copy()
    counts=pd.crosstab(a.Study,a.group).reindex(columns=GROUPS,fill_value=0)
    common=counts.index[(counts>=5).all(axis=1)]
    a=a[a.Study.isin(common)].copy()
    n=a.group.value_counts().reindex(GROUPS,fill_value=0)
    meta=dict(n_candidate=len(d),n_complete_common_study=len(a),n_studies=len(common),
              studies='|'.join(map(str,common)),**{f'n_{g}':int(n[g]) for g in GROUPS})
    failed=[dict(**meta,left=l,right=r,beta=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,
                 status='insufficient_three_group_support') for l,r in PAIRS]
    if n.min()<20:
        return failed,a
    x=pd.DataFrame({'Intercept':np.ones(len(a)),
        'resilience_CN':a.group.eq('resilience_CN').to_numpy(float),
        'CN_to_AD':a.group.eq('CN_to_AD').to_numpy(float)},index=a.index)
    ac=a.index_age-75
    numeric={'age_c':ac,'age_squared':ac*ac,**{p:a[p] for p in PCS}}
    if pet:
        numeric['PET_cognition_gap']=a.pet_cognition_gap
    for name,value in numeric.items():
        sd=value.std()
        if sd>1e-12:
            x[name]=(value-value.mean())/sd
    for col in ['Study','sex']+(['tracer'] if pet else []):
        x=pd.concat([x,pd.get_dummies(a[col].astype(str),prefix=col,drop_first=True,dtype=float)],axis=1)
    # Fixed-order full-rank basis, preserving the two focal group indicators.
    keep=[];basis=[]
    for col in x:
        v=x[col].to_numpy(float).copy(); norm0=np.linalg.norm(v)
        for _ in range(2):
            for q in basis:
                v-=q*np.dot(q,v)
        norm=np.linalg.norm(v)
        if norm>1e-9*max(norm0,1):
            keep.append(col);basis.append(v/norm)
    if not all(g in keep for g in ['resilience_CN','CN_to_AD']):
        return failed,a
    x=x[keep]; plain=sm.OLS(a[feature],x).fit()
    leverage=plain.get_influence().hat_matrix_diag.max()
    if plain.df_resid<20 or leverage>=.99:
        for row in failed: row['status']='insufficient_df_or_high_leverage'
        return failed,a
    fit=plain.get_robustcov_results(cov_type='HC3',use_t=True)
    output=[]
    for left,right in PAIRS:
        v=np.zeros(len(keep))
        if left!='other_CN_no_AD':v[keep.index(left)]+=1
        if right!='other_CN_no_AD':v[keep.index(right)]-=1
        result=fit.t_test(v); ci=result.conf_int()[0]
        output.append(dict(**meta,left=left,right=right,status='ok',
            beta=float(np.asarray(result.effect).item()),ci_low=float(ci[0]),ci_high=float(ci[1]),
            p=float(np.asarray(result.pvalue).item()),residual_df=float(fit.df_resid),
            max_leverage=float(leverage)))
    return output,a
