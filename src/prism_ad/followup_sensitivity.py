"""Exploratory follow-up sensitivities; controlled inputs and outputs stay private."""
from pathlib import Path
import json, os, hashlib
import numpy as np
import pandas as pd
from .features import endpoint, Censoring
from .transitions import normalize_visits
from .runtime import input_path, settings, sha
from .model_scale_checks import private_output

def ad_endpoints(first, visits):
    """First recorded AD after CN; MCI is intermediate, not competing."""
    if first.duplicated(["subject_id","visit"]).any() or first.subject_id.duplicated().any():
        raise ValueError("One first-eligible CN visit per participant is required")
    histories={str(s):g.reset_index(drop=True)for s,g in visits.groupby("subject_id",sort=False)}
    rows=[]
    for row in first.itertuples(index=False):
        g=histories[str(row.subject_id)]
        ix=np.flatnonzero(g.visit.to_numpy()==row.visit)
        if len(ix)!=1:raise ValueError("First-CN history linkage failed")
        j=int(ix[0])
        if g.diagnosis.iloc[j]!=1 or g.diagnosis.iloc[:j].isin([2,3]).any():
            raise ValueError("CN entry requires no prior MCI or AD")
        rec,reason=endpoint(g.age.to_numpy(float),g.diagnosis.to_numpy(float),j,2)
        info=dict(subject_id=str(row.subject_id),visit=row.visit,endpoint_eligible=rec is not None,
                  endpoint_reason=reason)
        info.update(rec if rec is not None else dict(end_time=np.nan,event=np.nan,uncertain_event=False,
                                                     interval_lower=np.nan,interval_width=np.nan))
        rows.append(info)
    return pd.DataFrame(rows)

def make_design(frame, include_ad=False):
    """Shared specified adjustment; C slopes remain separate by fitted fold."""
    d=frame.copy();fold=pd.to_numeric(d.fold).astype(int)
    x=pd.DataFrame(index=d.index)
    if include_ad:
        x["Intercept"]=1.
        x["AD_within_horizon"]=d.AD_within_horizon.astype(float)
    else:
        x["raw_PRS_SD"]=d.raw_PRS_SD.astype(float)
    for f in sorted(fold.unique()):
        ix=fold.eq(f)
        x["C_fold"+str(f)]=np.where(ix,d.cognition_main-d.loc[ix,"cognition_main"].mean(),0.)
    x["age_centered"]=(d.age.astype(float)-70.)/10.
    x["age_centered_squared"]=x.age_centered**2
    x["sex"]=d.sex.astype(float)
    for s in sorted(d.study.unique())[1:]:x["study="+str(s)]=d.study.eq(s).astype(float)
    if include_ad:
        for f in sorted(fold.unique())[1:]:x["fold="+str(f)]=fold.eq(f).astype(float)
    if np.linalg.matrix_rank(x.to_numpy(float))!=len(x.columns):
        raise ValueError("Specified follow-up design is not identifiable")
    return x

def prepare_frame(cfg):
    root=Path(cfg["inputs"]["prediction_directory"])/"LOCAL_RESTRICTED"
    firstpath=Path(cfg["inputs"]["first_CN_components"])
    rawpath=Path(cfg["inputs"]["cognition"]);rosterpath=Path(cfg["inputs"]["genotype_roster"])
    paths=[root/"metadata.csv",root/"features.npz",firstpath,rawpath,rosterpath]
    hashes={str(p):sha(p)for p in paths}
    meta=pd.read_csv(paths[0],dtype={"subject_id":str})
    arrays=np.load(paths[1],allow_pickle=False)
    firstmeta=meta[meta.task.eq("CN_MCI")&meta.first_eligible&meta.fold.isin([1,2,3])].copy()
    firstmeta["sex"]=arrays["current"][firstmeta.index,1]
    if firstmeta.subject_id.duplicated().any():raise ValueError("Duplicate CN first entry")
    first=pd.read_parquet(firstpath);first.subject_id=first.subject_id.astype(str)
    keys=["subject_id","visit","fold"]
    d=first.merge(firstmeta[keys+["sex"]],on=keys,validate="one_to_one")
    if len(d)!=len(firstmeta)or len(d)!=len(first):raise ValueError("Full CN forecast linkage failed")
    cols=["SUBJID","Study","PHC_Visit","PHC_Age_Cognition","PHC_Diagnosis"]
    cols+=["PHC_"+x for x in ["MEM","EXF","LAN","VSP"]]
    raw=pd.read_csv(rawpath,usecols=cols,dtype={"SUBJID":str},low_memory=False)
    visits=normalize_visits(raw[raw.SUBJID.isin(set(d.subject_id))])
    ep=ad_endpoints(d,visits)
    d=d.merge(ep,on=["subject_id","visit"],validate="one_to_one")
    prs=pd.read_csv(rosterpath,usecols=["PRS"]).PRS.to_numpy(float)
    if not np.isfinite(prs).all():raise ValueError("Reference PRS must be complete")
    reference=dict(mean=float(prs.mean()),population_sd=float(prs.std(ddof=0)),n=len(prs))
    d["raw_PRS_SD"]=(d.raw_PRS-reference["mean"])/reference["population_sd"]
    fields=["raw_PRS_SD","cognition_main","age","sex","study","fold"]
    d["complete_adjustment"]=~d[fields].isna().any(axis=1)
    complete=d[d.endpoint_eligible&d.complete_adjustment].copy()
    if not np.isfinite(complete[["raw_PRS_SD","cognition_main","age","sex"]].to_numpy(float)).all():
        raise ValueError("Nonfinite follow-up covariate")
    make_design(complete)
    censor=Censoring().fit(d[d.endpoint_eligible])
    lab=censor.label(d[d.endpoint_eligible],5.)
    known=d.loc[d.endpoint_eligible].copy()
    known["AD_within_horizon"]=lab["y"]
    known["known_horizon"]=lab["known"]
    known["AD_ipcw"]=lab["weight"]
    support={}
    for h in [3.,5.]:
        z=censor.label(d[d.endpoint_eligible],h)
        support[str(int(h))]=dict(AD_events=int((z["known"]&(z["y"]==1)).sum()),
             no_AD_observed=int((z["known"]&(z["y"]==0)).sum()),
             unknown=int((~z["known"]).sum()),weights_clipped=int(z["clipped"].sum()),
             min_censoring_survival=float(z["g"][z["known"]].min()),
             effective_sample_size=float(z["weight"].sum()**2/(z["weight"]**2).sum()))
    report=dict(n_entries=len(d),eligible_endpoints=int(d.endpoint_eligible.sum()),
                endpoint_reasons=d.endpoint_reason.value_counts().to_dict(),
                AD_events=int(d.event.eq(1).sum()),right_censored=int(d.event.eq(0).sum()),
                uncertain_first_AD=int(d.uncertain_event.sum()),complete_covariates=int(d.complete_adjustment.sum()),
                missing_adjustment={c:int(d[c].isna().sum())for c in fields},
                studies=d.study.value_counts().to_dict(),sex_values=d.sex.value_counts(dropna=False).to_dict(),
                horizon_support=support,cox_design_columns=list(make_design(complete).columns),
                original_prediction_endpoints_reused=False,censoring_model="new AD-specific ReverseKM by study/ageband with original fallback and cap10",
                reference=reference,input_sha256=hashes,source_sha256=sha(__file__))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    return d,known,report

def bh_family(pvalues):
    p=np.asarray(pvalues,float)
    if not np.isfinite(p).all()or np.any((p<0)|(p>1)):raise ValueError("P values must be finite")
    order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1)
    return out

def five_year_contrast(known):
    """IPCW WLS mean-score difference; HC3 inference conditional on fitted weights."""
    import statsmodels.api as sm
    d=known[known.known_horizon&known.complete_adjustment].copy()
    if min(d.AD_within_horizon.value_counts().reindex([0,1],fill_value=0))<5:
        return dict(test="fixed_5y_raw_PRS_SD",status="insufficient_group_support",p=1.)
    x=make_design(d,include_ad=True)
    a=x.to_numpy(float);w=d.AD_ipcw.to_numpy(float)
    leverage=w*np.sum(a*(a@np.linalg.pinv(a.T@(w[:,None]*a))),axis=1)
    if np.max(leverage)>=.99:raise ValueError("Excess weighted-model leverage")
    fit=sm.WLS(d.raw_PRS_SD,x,weights=d.AD_ipcw).fit(cov_type="HC3",use_t=True)
    name="AD_within_horizon";ci=fit.conf_int().loc[name]
    if fit.df_resid<20:raise ValueError("Insufficient weighted-model residual df")
    return dict(test="fixed_5y_raw_PRS_SD",status="ok",n=len(d),
                events=int(d.AD_within_horizon.sum()),nonevents=int((1-d.AD_within_horizon).sum()),
                beta=float(fit.params[name]),se=float(fit.bse[name]),
                ci_low=float(ci.iloc[0]),ci_high=float(ci.iloc[1]),p=float(fit.pvalues[name]),
                df=float(fit.df_resid),max_leverage=float(leverage.max()),units="population_SD",weights="new_AD_specific_IPCW",
                interval="HC3_residual_df_t",fitted_weights_treated_fixed=True)

R_CODE = r"""
args<-commandArgs(trailingOnly=TRUE)
d<-read.csv(args[1],check.names=FALSE)
library(survival)
options(warn=2)
xs<-grep("^X[0-9]+$",names(d),value=TRUE)
form<-as.formula(paste("Surv(end_time,event) ~",paste(xs,collapse="+"),"+strata(fold)"))
fit<-coxph(form,data=d,ties="efron",robust=TRUE,x=TRUE,y=TRUE,
           control=coxph.control(iter.max=100,eps=1e-09))
b<-coef(fit);se<-sqrt(diag(vcov(fit)));z<-b/se
if(any(!is.finite(c(b,se,z)))||any(se<=0))stop("Nonfinite Cox coefficients or standard errors")
o<-data.frame(term=names(b),beta=unname(b),se=unname(se),z=unname(z),
              p=unname(2*pnorm(-abs(z))),ci_low=unname(b-qnorm(.975)*se),
              ci_high=unname(b+qnorm(.975)*se),row.names=NULL)
write.csv(o,paste0(args[2],"_coefficients.csv"),row.names=FALSE)
write.csv(vcov(fit),paste0(args[2],"_covariance.csv"),row.names=TRUE)
diag<-cox.zph(fit,transform="rank",terms=TRUE,global=TRUE)
write.csv(data.frame(term=rownames(diag$table),diag$table,check.names=FALSE),
          paste0(args[2],"_PH_diagnostics.csv"),row.names=FALSE)
writeLines(c(R.version.string,paste("survival",packageVersion("survival")),
             paste("n",fit$n),paste("events",fit$nevent),
             paste("loglik",paste(fit$loglik,collapse=" ")),
             paste("iterations",fit$iter)),paste0(args[2],"_runtime.txt"))
"""

def cox_association(frame, rscript, output):
    """Fold-stratified time-to-first-recorded-AD association using R survival."""
    import subprocess
    d=frame[frame.endpoint_eligible&frame.complete_adjustment].copy()
    x=make_design(d);out=Path(output);out.mkdir(mode=0o700,parents=True,exist_ok=True)
    (out/"COX_DESIGN_COLUMNS.json").write_text(json.dumps({"X"+str(i):c for i,c in enumerate(x.columns)},indent=2))
    data=x.rename(columns={c:"X"+str(i)for i,c in enumerate(x.columns)}).copy()
    data["end_time"]=d.end_time.to_numpy();data["event"]=d.event.to_numpy(int);data["fold"]=d.fold.to_numpy(int)
    path=out/"cox_input_PRIVATE.csv";data.to_csv(path,index=False)
    code=out/"cox_driver.R";code.write_text(R_CODE)
    prefix=out/"cox"
    proc=subprocess.run([str(rscript),str(code),str(path),str(prefix)],capture_output=True,text=True)
    (out/"cox_execution.log").write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError("Cox fit/diagnostics failed; inspect private execution log")
    coef=pd.read_csv(out/"cox_coefficients.csv");z=coef[coef.term.eq("X0")].iloc[0]
    ph=pd.read_csv(out/"cox_PH_diagnostics.csv")
    # The diagnostic is reported as an assumption check, not an effect-discovery test.
    result=dict(test="time_to_AD_raw_PRS_SD",status="ok",n=len(d),events=int(d.event.sum()),
        beta=float(z.beta),se=float(z.se),ci_low=float(z.ci_low),ci_high=float(z.ci_high),
        hazard_ratio=float(np.exp(z.beta)),HR_low=float(np.exp(z.ci_low)),HR_high=float(np.exp(z.ci_high)),
        p=float(z.p),units="hazard_ratio_per_population_SD",ties="efron",strata="fold",
        interval="robust_sandwich_normal",PH_transform="rank",
        PH_raw_PRS_P=float(ph.loc[ph.term.eq("X0"),"p"].iloc[0]),
        PH_global_P=float(ph.loc[ph.term.eq("GLOBAL"),"p"].iloc[0]),
        design_columns=list(x.columns))
    return result

def main():
    os.umask(0o077)
    cfg=settings();out=private_output()
    if (out/"followup_sensitivity.csv").exists():raise RuntimeError("Use a new private output directory")
    # A private, frozen plan is required before estimating clinical associations.
    planpath=input_path("followup_analysis_plan");plan=json.loads(planpath.read_text())
    if plan.get("frozen_before_effect_estimates")is not True or "primary"not in plan or "secondary"not in plan:
        raise RuntimeError("The follow-up analysis plan must be frozen before estimation")
    plan_sha=sha(planpath)
    amendmentpath=input_path("followup_plan_amendment")
    amendment=json.loads(amendmentpath.read_text());amendment_sha=sha(amendmentpath)
    if (amendment.get("original_plan_sha256")!=plan_sha or
        amendment.get("before_effect_estimates")is not True or
        amendment.get("control_tie_convention")!="left_limit_inclusive_horizon"):
        raise RuntimeError("Tie convention amendment must precede estimation and bind the frozen plan")
    d,known,report=prepare_frame(cfg)
    d.to_parquet(out/"all_CN_AD_endpoints_PRIVATE.parquet",index=False)
    known.to_parquet(out/"five_year_status_PRIVATE.parquet",index=False)
    restricted=out/"restricted_local";restricted.mkdir(mode=0o700,exist_ok=True)
    w=known[known.known_horizon&known.complete_adjustment]
    design=make_design(w,include_ad=True)
    design["outcome_raw_PRS_SD"]=w.raw_PRS_SD
    design["AD_ipcw"]=w.AD_ipcw
    design.to_csv(restricted/"WLS_design_PRIVATE.csv",index=False)
    primary=cox_association(d,cfg["execution"]["survival_Rscript"],out/"restricted_local")
    secondary=five_year_contrast(known)
    results=pd.DataFrame([primary,secondary])
    results["q_followup2"]=bh_family(results.p.fillna(1).to_numpy())
    results.to_csv(out/"followup_sensitivity.csv",index=False)
    report.update(status="PASS",plan_sha256=plan_sha,amendment_sha256=amendment_sha,
                  prediction_models_refitted=False,controlled_outputs=True,
                  no_external_validation=True,death_competing_risk_unavailable=True)
    if sha(planpath)!=plan_sha or sha(amendmentpath)!=amendment_sha:raise RuntimeError("Frozen plan changed during estimation")
    (out/"execution.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status="PASS",n_entries=len(d),comparisons=len(results),
                          prediction_models_refitted=False,outputs_private=True)))
if __name__=="__main__":main()
