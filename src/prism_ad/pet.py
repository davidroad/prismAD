"""Exploratory PRS–amyloid PET trends, using existing one-scan-per-person data.

Only aggregate results leave this script. It does not refit prognostic models or
write participant-level records. Restricted upstream parquet files stay local.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import statsmodels
import statsmodels.formula.api as smf

from .runtime import workspace,input_path
HERE=workspace();PROJECT=HERE;OUT=HERE/'results'
OLDER=input_path('older_cn_pet')
ALL_AGE=input_path('all_age_cn_pet')
LOCKED={}
PRS_MEAN = None
PRS_SD = None
PCS = [f"PC{i}" for i in range(1, 6)]
OLDER_FORMULA = "centiloids ~ PRS_sd + scan_age + female + " + " + ".join(PCS) + " + C(Study) + C(tracer)"
OLDER_LAG_FORMULA = "centiloids ~ PRS_sd + scan_age + CN_lag + female + " + " + ".join(PCS) + " + C(Study) + C(tracer)"
ALL_AGE_FORMULA = "centiloids ~ PRS_sd + pet_age + female + " + " + ".join(PCS) + " + C(Study) + C(tracer)"
ALL_AGE_STUDY_ONLY = "centiloids ~ PRS_sd + pet_age + female + " + " + ".join(PCS) + " + C(Study)"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(name, d, formula, cohort, note=""):
    assert d.subject_id.is_unique, name
    assert d.PRS.notna().all() and d.centiloids.notna().all(), name
    ordinary = smf.ols(formula, data=d).fit()
    robust = ordinary.get_robustcov_results(cov_type="HC3", use_t=True)
    j = list(ordinary.params.index).index("PRS_sd")
    ci = robust.conf_int()[j]
    max_h = float(ordinary.get_influence().hat_matrix_diag.max())
    status = "supported" if ordinary.df_resid >= 20 and max_h < 0.99 else "insufficient_support"
    used = d.loc[ordinary.model.data.row_labels]
    rho, rho_p = spearmanr(used.PRS, used.centiloids)
    return dict(cohort=cohort, scope=name, status=status,
        n_candidate=len(d), n_complete=int(ordinary.nobs),
        n_studies=int(used.Study.nunique()),
        prs_sd_full_roster=PRS_SD, beta_centiloids_per_prs_sd=float(robust.params[j]),
        ci_low=float(ci[0]), ci_high=float(ci[1]), nominal_p=float(robust.pvalues[j]),
        spearman_rho=float(rho), spearman_nominal_p=float(rho_p),
        residual_df=float(ordinary.df_resid), max_leverage=max_h,
        formula=formula, note=note)


def main():
    global PRS_MEAN,PRS_SD
    roster=pd.read_csv(input_path("genotype_roster"),usecols=["PRS"])
    PRS_MEAN,PRS_SD=float(roster.PRS.mean()),float(roster.PRS.std(ddof=1))
    if (OUT / "manifest.json").exists():
        old = json.loads((OUT / "manifest.json").read_text())
        assert old["scope"].startswith("Exploratory, post hoc associations"), "Unknown prior manifest"
        for name, expected in old["outputs_sha256"].items():
            assert sha(OUT / name) == expected, "Unregistered result change: " + name
        # Explicitly revise only this newly generated analysis after the QA review
        # identified follow-up lag as a post-scan covariate. No upstream data change.
    for relative, expected in LOCKED.items():
        assert sha(PROJECT / relative) == expected, relative
    older = pd.read_parquet(OLDER)
    all_age = pd.read_parquet(ALL_AGE)
    assert len(older) == 802 and older.subject_id.is_unique
    assert len(all_age) == 3115 and all_age.subject_id.is_unique
    assert older.high.isin([0.,1.]).all() and older.high.nunique()==2
    assert all_age.group.value_counts().to_dict()["CN_to_AD"] == 19
    all_age["PRS_sd"] = (all_age.PRS - PRS_MEAN) / PRS_SD
    all_age["female"] = all_age.sex.map({1: 0., 2: 1.})
    all_age_cn = all_age[all_age.final_status.eq("CN")].copy()
    high = all_age[all_age.group.eq("resilience_CN")].copy()
    other = all_age[all_age.group.eq("other_CN_no_AD")].copy()
    assert len(all_age_cn)==3030 and len(high)+len(other)==len(all_age_cn)
    assert np.allclose(older.PRS_sd, (older.PRS - PRS_MEAN) / PRS_SD)

    specs = [
        ("older_CN_all", older, OLDER_FORMULA, "age_at_last_classified_75plus", "Main older CN-stage PET population"),
        ("older_CN_high_fixed_interval", older[older.high.eq(1)], OLDER_FORMULA, "age_at_last_classified_75plus", "Fixed high-PRS interval, not a new top quintile"),
        ("older_CN_other", older[older.high.eq(0)], OLDER_FORMULA, "age_at_last_classified_75plus", "Other PRS values"),
        ("older_CN_strict_final_record", older[~older.later_unclassified_visits], OLDER_FORMULA, "age_at_last_classified_75plus", "Final recorded visit explicitly CN"),
        ("older_CN_high_strict_final_record", older[older.high.eq(1) & ~older.later_unclassified_visits], OLDER_FORMULA, "age_at_last_classified_75plus", "Fixed high-PRS interval; final recorded visit explicitly CN"),
        ("older_CN_all_lag_adjusted", older, OLDER_LAG_FORMULA, "age_at_last_classified_75plus", "Sensitivity only: additionally conditions on post-PET follow-up lag"),
        ("older_CN_high_lag_adjusted", older[older.high.eq(1)], OLDER_LAG_FORMULA, "age_at_last_classified_75plus", "Sensitivity only: additionally conditions on post-PET follow-up lag"),
        ("all_age_CN_all", all_age_cn, ALL_AGE_FORMULA, "all_age_CN_stage_PET", "Broader CN group; not the older resilience group"),
        ("all_age_CN_high_fixed_interval", high, ALL_AGE_FORMULA, "all_age_CN_stage_PET", "Broader high-PRS CN group; no minimum post-PET follow-up"),
        ("all_age_CN_other", other, ALL_AGE_FORMULA, "all_age_CN_stage_PET", "Broader other-PRS CN group"),
        ("all_age_CN_high_3year_followup", high[high.classified_followup_span.ge(3)], ALL_AGE_STUDY_ONLY, "all_age_CN_stage_PET", "Three years of classifiable follow-up; tracer dropped to avoid singleton high leverage"),
    ]
    for study in ["A4", "ADNI"]:
        block = high[high.Study.eq(study)].copy()
        tracer_counts = block.tracer.value_counts()
        block = block[block.tracer.isin(tracer_counts[tracer_counts.ge(5)].index)].copy()
        if len(block) >= 40:
            form = ALL_AGE_FORMULA.replace(" + C(Study)", "")
            specs.append(("all_age_CN_high_" + study, block, form, "all_age_CN_stage_PET", "Exploratory within-study sensitivity; tracer cells with fewer than five participants excluded"))

    results = [fit(*spec) for spec in specs]
    assert all(r["status"] == "supported" for r in results), [(r["scope"], r["status"], r["max_leverage"]) for r in results]
    future = all_age[all_age.group.eq("CN_to_AD")]
    rho, p = spearmanr(future.PRS, future.centiloids)
    descriptive = [dict(group="CN_to_AD_after_CN_stage_scan", n=len(future),
        spearman_rho=float(rho), spearman_nominal_p=float(p),
        note="Too few future-AD participants for adjusted group-specific inference"),
        dict(group="older_high_PRS_CN_to_AD_after_CN_stage_scan", n=6,
        spearman_rho=None, spearman_nominal_p=None,
        note="Six older high-PRS future-AD participants; descriptive count only")]

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT / "aggregate_slopes.csv", index=False)
    pd.DataFrame(descriptive).to_csv(OUT / "future_AD_support.csv", index=False)
    manifest = dict(
        scope="Exploratory, post hoc associations in existing one-CN-stage-scan-per-person datasets; no prognostic refits",
        inputs_sha256={key:sha(path) for key,path in [('older_cn_pet',OLDER),('all_age_cn_pet',ALL_AGE)]},
        code_sha256=sha(Path(__file__)),
        outputs_sha256={p.name: sha(p) for p in [OUT / "aggregate_slopes.csv", OUT / "future_AD_support.csv"]},
        environment=dict(python=platform.python_version(), pandas=pd.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__),
        important_boundary="Selected PET subsets and short follow-up; strict final-record and longer-follow-up sensitivities constrain high-PRS interpretation",
        qa_revision="Primary age-75 models exclude post-PET follow-up lag; lag-adjusted models are retained only as sensitivities",
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for r in results:
        print(r["scope"], r["n_complete"], round(r["beta_centiloids_per_prs_sd"], 2),
              tuple(round(r[k], 2) for k in ["ci_low", "ci_high"]))


if __name__ == "__main__":
    main()
