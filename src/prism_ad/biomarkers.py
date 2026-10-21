"""Select the earliest eligible CN-stage amyloid PET record."""
import numpy as np
import pandas as pd
from .cognitive_course import exact_age

def candidates(raw,c,modality,age):
    r=raw[raw.SUBJID.isin(c.subject_id)].copy();audit={'linked_rows':len(r)}
    if modality=='PET':
        good=r.PHC_QC_IMAGE.eq(1)&r.PHC_QC_TIMING.eq(1)
        audit['QC_excluded']=int((~good).sum());r=r[good].copy()
    p=r[['SUBJID','Study','PHC_Visit','PHC_Diagnosis']].rename(columns={'SUBJID':'subject_id','PHC_Visit':'visit','PHC_Diagnosis':'scan_diagnosis'}).copy()
    p['scan_age']=exact_age(r['PHC_Age_PET' if modality=='PET' else 'PHC_Age_T1'])
    if modality=='PET':
        p['centiloids']=pd.to_numeric(r.PHC_CENTILOIDS,errors='coerce').replace([np.inf,-np.inf],np.nan)
        p['amyloid_positive']=pd.to_numeric(r.PHC_AMYLOID_STATUS,errors='coerce').where(r.PHC_AMYLOID_STATUS.isin([0,1]))
        p['tracer']=r.PHC_TRACER.astype('string')
        fields=['centiloids','amyloid_positive','tracer']
    else:
        raise ValueError('This workflow uses PET inputs only')
        p=p[['subject_id','Study','visit','scan_diagnosis','scan_age']+fields]
    p=p.merge(c,on=['subject_id','Study','visit'],validate='many_to_one')
    p['scan_cognition_gap']=(p.scan_age-p.cognition_age).abs()
    valid=(p.scan_age.ge(age)&p.cognition_age.ge(age)&p.diagnosis.eq(1)&~p.prior_MCI_AD
        &p.scan_diagnosis.eq(1)&p.scan_cognition_gap.le(90/365.25+1e-8))
    valid &= p.centiloids.notna() if modality=='PET' else p[['hippocampus_pct_icv','entorhinal_thickness']].notna().any(axis=1)
    p=p[valid].copy();audit['eligible_rows_before_ties']=len(p)
    cols=['subject_id','Study','visit','scan_age','cognition_age','education','sex','diagnosis','prior_MCI_AD','scan_cognition_gap']+fields
    p=p[cols].drop_duplicates()
    ties=p.duplicated(['subject_id','visit','scan_age'],keep=False);audit['discordant_tie_rows']=int(ties.sum());p=p[~ties].copy()
    p['index_age']=np.maximum(p.scan_age,p.cognition_age)
    # No disease visit may precede joint availability, even when scan is later
    # than its matched cognition record. Screen every candidate, then select.
    disease={sid:b.cognition_age.dropna().to_numpy() for sid,b in c[c.diagnosis.isin([2,3])].groupby('subject_id')}
    between=np.array([np.any((disease.get(row.subject_id,np.array([]))>=row.cognition_age)&
        (disease.get(row.subject_id,np.array([]))<=row.index_age)) for row in p.itertuples()],dtype=bool)
    audit['disease_before_joint_availability']=int(between.sum());p=p[~between]
    # Chronological joint availability, then deterministic visit/scan ties.
    p=p.sort_values(['subject_id','index_age','visit','scan_age'],kind='stable').groupby('subject_id',sort=False).head(1).copy()
    assert p.subject_id.is_unique
    audit['selected_subjects']=len(p)
    return p,audit
