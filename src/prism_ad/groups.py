"""Construct controlled clinical-course groups and CN-stage PET analysis inputs."""
from pathlib import Path
import os,json
import numpy as np
import pandas as pd
from .runtime import input_path,workspace,sha
from .phenotypes import construct_index,PCS
from .cognitive_course import normalize,make_clinical_indices,make_pet_indices,add_course,scope_mask
from .biomarkers import candidates
LOW,HIGH=0.473704771704382,1.1350529724900473
def main():
    os.umask(0o077)
    out=workspace();dest=out/'restricted_local'
    if (out/'groups.json').exists():raise RuntimeError('Choose a new output directory')
    dest.mkdir(mode=0o700,parents=True,exist_ok=True)
    roster=pd.read_csv(input_path('genotype_roster'),dtype={'subject_id':str,'sample_id':str})
    roster=roster[['subject_id','sample_id','PRS']+PCS]
    assert roster.subject_id.is_unique
    PRS_MEAN,PRS_SD=float(roster.PRS.mean()),float(roster.PRS.std(ddof=1))
    raw=pd.read_csv(input_path('cognition'),dtype={'SUBJID':str},low_memory=False)
    frozen=construct_index(raw,roster)
    frozen['T_20']=frozen.PRS.between(LOW,HIGH,inclusive='both').astype(float)
    c=normalize(raw,frozen.subject_id)
    pet=pd.read_csv(input_path('amyloid_pet'),dtype={'SUBJID':str},low_memory=False)
    pi,_=make_pet_indices(pet,c)
    cohorts={'clinical_CN_index':add_course(make_clinical_indices(c),c,frozen),
             'PET_CN_index':add_course(pi,c,frozen)}
    counts=[]
    for name,d in cohorts.items():
        d.to_parquet(dest/(name+'_indices_and_course.parquet'),index=False)
        primary=d[scope_mask(d,'primary_observed')].copy()
        primary.to_csv(dest/(name+'_three_groups_primary.csv'),index=False)
        counts += [dict(cohort=name,group=str(g),n=int(n)) for g,n in primary.group.value_counts().items()]
    # Older PET group is defined by age at last classified cognition, not scan age.
    idx,_=candidates(pet,c,'PET',0)
    f=frozen[frozen.status.eq('CN') & (frozen.age.ge(75)|frozen.age_topcoded.fillna(False))].copy()
    keep=['subject_id','Study','age','age_topcoded','PRS','visit']+PCS
    f=f[keep].rename(columns={'visit':'last_classified_visit','age':'last_classified_age'})
    older=idx.merge(f,on=['subject_id','Study'],validate='one_to_one')
    latest=c.groupby('subject_id',sort=False).tail(1)[['subject_id','visit']].rename(columns={'visit':'last_recorded_visit'})
    older=older.merge(latest,on='subject_id',validate='one_to_one')
    older['later_unclassified_visits']=older.last_recorded_visit.gt(older.last_classified_visit)
    older['high']=older.PRS.between(LOW,HIGH,inclusive='both').astype(float)
    older['PRS_sd']=(older.PRS-PRS_MEAN)/PRS_SD
    older['female']=older.sex.map({1:0.,2:1.})
    older['CN_lag']=older.last_classified_age-older.scan_age
    older.to_parquet(dest/'older_CN_PET.parquet',index=False)
    inputs={k:sha(input_path(k)) for k in ['genotype_roster','cognition','amyloid_pet']}
    receipt=dict(inputs_sha256=inputs,cohort_counts=counts,older_CN_PET=len(older),
                 older_high_PRS_CN_PET=int(older.high.eq(1).sum()),
                 scope='Restricted intermediate records; release aggregate summaries only')
    (out/'groups.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='inputs_sha256'}))
if __name__=='__main__':main()
