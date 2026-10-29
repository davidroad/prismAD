"""Four review-requested sensitivity checks; all empirical output stays private."""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
from .cognitive_strata import fit
from .runtime import input_path, workspace, sha

def private_output():
    out=workspace();code=Path(__file__).resolve().parents[2]
    if out==code or code in out.parents:
        raise RuntimeError("Use private output storage outside the code checkout")
    for p in [out]+list(out.parents):
        if p.name in ("PRISM-AD_Public","PRISM-AD_Local") or (p/".git").exists():
            raise RuntimeError("Use private output storage outside code packages and Git repositories")
    out.mkdir(mode=0o700,parents=True,exist_ok=True);out.chmod(0o700)
    return out

def analyze(frame,roster_mean,roster_sd):
    d=frame.copy()
    fields=["fold","clinical_group","cognition_main","PRS","raw_PRS","Study"]
    if d[fields].isna().any().any() or not np.isfinite(roster_sd) or roster_sd<=0:
        raise ValueError("Complete contributions and a positive reference PRS SD are required")
    if not d.clinical_group.isin(["CN","MCI","AD"]).all():
        raise ValueError("Unexpected clinical-course group")
    d["C_decile"]=pd.qcut(d.cognition_main,10,labels=False)+1
    pair=d[d.clinical_group.isin(["CN","AD"])].copy();rows=[];affine=[]
    for fold in [1,2,3]:
        z=pair[pair.fold.eq(fold)].copy()
        row=fit(z,"within_fold_"+str(fold),global_model=True)
        row.update(outcome="native_G",units="log_odds",fold=fold);rows.append(row)
        q=d[d.fold.eq(fold)]
        if len(q)>2 and q.raw_PRS.std()>0:
            x=np.column_stack([np.ones(len(q)),q.raw_PRS])
            coef=np.linalg.lstsq(x,q.PRS,rcond=None)[0]
            affine.append(dict(fold=fold,n=len(q),intercept=float(coef[0]),slope=float(coef[1]),
                               max_abs_error=float(np.abs(q.PRS-x@coef).max())))
    raw=pair.copy();raw["PRS"]=(raw.raw_PRS-roster_mean)/roster_sd
    row=fit(raw,"pooled_raw_PRS_SD",global_model=True)
    row.update(outcome="raw_PRS_standardized",units="full_roster_SD",fold="pooled")
    for col in ["factor_ratio","ratio_ci_low","ratio_ci_high"]:row[col]=np.nan
    rows.append(row);result=pd.DataFrame(rows)
    p=result.p.fillna(1).to_numpy();ix=np.argsort(p);q=np.empty(len(p))
    q[ix]=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1)
    result["q_sensitivity4"]=q
    return result,affine

def main():
    os.umask(0o077)
    source=input_path("current_profile_components");roster=input_path("genotype_roster")
    keys=input_path("profile_join_keys")
    hashes={str(p):sha(p)for p in [source,roster,keys]}
    d=pd.read_csv(source).merge(pd.read_parquet(keys)[["display_alias","Study"]],on="display_alias",validate="one_to_one")
    prs=pd.read_csv(roster,usecols=["PRS"]).PRS.to_numpy(float)
    if not np.isfinite(prs).all():raise ValueError("Reference roster PRS must be complete")
    mean,sd=float(prs.mean()),float(prs.std(ddof=0))
    result,affine=analyze(d,mean,sd);out=private_output()
    if (out/"model_scale_sensitivity.csv").exists():raise RuntimeError("Choose a new private output directory")
    result.to_csv(out/"model_scale_sensitivity.csv",index=False)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    receipt=dict(status="PASS",input_sha256=hashes,source_sha256=sha(__file__),profile_n=len(d),
                 test_family=4,reference_mean=mean,reference_sd=sd,affine_G_rawPRS_checks=affine,
                 prediction_models_refitted=False,independent_replication=False,participant_rows_exported=False)
    (out/"execution.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(status="PASS",sensitivity_comparisons=4,models_refitted=False)))
if __name__=="__main__":main()
