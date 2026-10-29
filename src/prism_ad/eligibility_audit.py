"""Separate pre-prediction exclusion from subsequent AD followed by reversion."""
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
from .features import endpoint
from .transitions import normalize_visits
from .model_scale_checks import private_output
from .runtime import input_path,sha

def prior_flags(visits):
    v=visits.sort_values(["subject_id","visit"],kind="stable").copy()
    for name,codes in [("prior_MCI_AD",[2,3]),("prior_AD",[3])]:
        flag=v.diagnosis.isin(codes).astype(int)
        v[name]=(flag.groupby(v.subject_id).cumsum()-flag).gt(0)
    return v

def audit(meta,visits,course):
    v=prior_flags(visits)
    if v.duplicated(["subject_id","visit"]).any():raise ValueError("Diagnostic visit keys must be unique")
    q=meta.merge(v[["subject_id","visit","diagnosis","prior_MCI_AD","prior_AD"]],
                 on=["subject_id","visit"],how="left",validate="many_to_one")
    bad=((q.task.eq("CN_MCI")&(q.diagnosis.ne(1)|q.prior_MCI_AD))|
         (q.task.eq("MCI_AD")&(q.diagnosis.ne(2)|q.prior_AD)))
    histories={sid:g.reset_index(drop=True)for sid,g in v.groupby("subject_id",sort=False)}
    mismatches=0;max_error=0.
    for row in meta.itertuples(index=False):
        h=histories[row.subject_id];pos=np.flatnonzero(h.visit.to_numpy()==row.visit)
        if len(pos)!=1:mismatches+=1;continue
        rec,_=endpoint(h.age.to_numpy(float),h.diagnosis.to_numpy(float),int(pos[0]),
                       1 if row.task=="CN_MCI"else 2)
        if rec is None:mismatches+=1;continue
        error=abs(rec["end_time"]-row.end_time);max_error=max(max_error,error)
        mismatches+=int(error>1e-8 or rec["event"]!=row.event or
                        bool(rec["uncertain_event"])!=bool(row.uncertain_event))
    first=meta[meta.first_eligible&meta.fold.isin([1,2,3])&meta.task.eq("CN_MCI")]
    joined=course.merge(first,on=["subject_id","visit"],how="inner",
                        suffixes=("_group",""),validate="one_to_one")
    prior=joined.prior_MCI_AD
    cn=joined.final_status.eq("CN")&~joined.first_AD_recorded&~prior
    mci=joined.final_status.eq("MCI")&~joined.first_AD_recorded&~prior
    ad=(joined.final_status.eq("AD")&joined.first_AD_recorded&~prior&
        joined.first_AD_elapsed.gt(0)&joined.course_status.eq("first_AD_dated_after_CN_index"))
    reverted=joined.final_status.eq("excluded_prior_AD_reversion")&joined.first_AD_recorded
    prior_in_reverters=0
    for sid,visit in joined.loc[reverted,["subject_id","visit"]].itertuples(index=False,name=None):
        h=histories[sid];prior_in_reverters+=int(h.loc[h.visit.lt(visit),"diagnosis"].isin([2,3]).any())
    report=dict(prediction_rows=len(meta),prediction_subjects=int(meta.subject_id.nunique()),
                prior_disqualifying_rows=int(bad.sum()),endpoint_mismatches=mismatches,
                max_end_time_error=max_error,initial_CN_evaluation=len(first),
                original_CN_index_matches=len(joined),index_nonmatches=len(first)-len(joined),
                retained_profiles=int((cn|mci|ad).sum()),
                profile_group_counts=dict(CN=int(cn.sum()),MCI=int(mci.sum()),AD=int(ad.sum())),
                AD_followed_by_reversion=int(reverted.sum()),
                reversion_cases_with_prior_MCI_AD_at_prediction=prior_in_reverters,
                nonreversion_profile_exclusions=int((~(cn|mci|ad|reverted)).sum()))
    report["status"]="PASS"if not any([report["prior_disqualifying_rows"],mismatches,prior_in_reverters])else"FAIL"
    return report

def main():
    os.umask(0o077)
    rawpath=input_path("cognition")
    metapath=input_path("prediction_directory")/"LOCAL_RESTRICTED/metadata.csv"
    coursepath=input_path("cn_course_indices")
    hashes={str(p):sha(p)for p in [rawpath,metapath,coursepath]}
    meta=pd.read_csv(metapath,dtype={"subject_id":str})
    columns=["SUBJID","Study","PHC_Visit","PHC_Age_Cognition","PHC_Diagnosis"]
    columns+=["PHC_"+x for x in ["MEM","EXF","LAN","VSP"]]
    raw=pd.read_csv(rawpath,usecols=columns,dtype={"SUBJID":str},low_memory=False)
    visits=normalize_visits(raw[raw.SUBJID.isin(set(meta.subject_id))])
    course=pd.read_parquet(coursepath);course.subject_id=course.subject_id.astype(str)
    report=audit(meta,visits,course)
    report.update(input_sha256=hashes,source_sha256=sha(__file__),participant_rows_exported=False)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    out=private_output()
    if (out/"eligibility_audit.json").exists():raise RuntimeError("Choose a new private output directory")
    (out/"eligibility_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    if report["status"]!="PASS":raise RuntimeError("Prediction eligibility audit failed; inspect private report")
    print(json.dumps(dict(status="PASS",rows_checked=len(meta),participant_rows_exported=False)))
if __name__=="__main__":main()
