"""Descriptive publication summaries; no fitting or inferential testing."""
import numpy as np
import pandas as pd

def pool_calibration(frame):
    b=frame[frame.model.eq("PRISM")&frame.condition.eq("plusPRS")&frame.subset.eq("first_eligible_landmark")].copy()
    b["fold_numeric"]=pd.to_numeric(b.fold,errors="coerce")
    b=b[b.fold_numeric.isin([1,2,3])]
    keys=["task","horizon","fold_numeric","risk_lo","risk_hi"]
    if b.duplicated(keys).any():raise ValueError("Duplicate calibration bin keys")
    rows=[]
    for (task,h,lo,hi),q in b.groupby(["task","horizon","risk_lo","risk_hi"],sort=True):
        if set(q.fold_numeric)!={1,2,3}:raise ValueError("All fitted folds must contribute each bin slot")
        n=int(q.n_known.sum());w=float(q.weight_mass.sum())
        if (q.weight_mass<0).any() or not np.isfinite(w):raise ValueError("Invalid IPCW mass")
        if q.loc[q.weight_mass.gt(0),["mean_prediction","observed_prevalence"]].isna().any().any():
            raise ValueError("Positive-weight bins need predictions and outcomes")
        pred=float((q.mean_prediction.fillna(0)*q.weight_mass).sum()/w)if w>0 else np.nan
        obs=float((q.observed_prevalence.fillna(0)*q.weight_mass).sum()/w)if w>0 else np.nan
        rows.append(dict(task=task,horizon=int(h),risk_lo=float(lo),risk_hi=float(hi),
                         n_known=n,weight_mass=w,prediction=pred,observed=obs))
    return pd.DataFrame(rows)

def marginal_components(frame):
    d=frame.copy()
    if d.duplicated("display_alias").any():raise ValueError("One contribution row per participant is required")
    if not d.clinical_group.isin(["CN","MCI","AD"]).all():raise ValueError("Unknown clinical-course group")
    if d[["fold","cognition_main","PRS"]].isna().any().any():raise ValueError("Complete fitted components are required")
    rows=[]
    for (fold,group),q in d.groupby(["fold","clinical_group"],sort=True):
        for label,col in [("C","cognition_main"),("G","PRS")]:
            x=q[col].to_numpy(float)
            if not np.isfinite(x).all():raise ValueError("Finite signed contributions are required")
            lo,median,hi=np.quantile(x,[.25,.5,.75])
            rows.append(dict(component=label,clinical_group=group,fold=int(fold),n=len(x),
                             median=float(median),q25=float(lo),q75=float(hi)))
    return pd.DataFrame(rows)
