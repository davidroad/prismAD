"""Export fixed aggregate plotting inputs from controlled analysis outputs."""
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import workspace,input_path,sha
def main():
    out=workspace()/'plotting'
    if out.exists():raise RuntimeError('Choose a new aggregate output directory')
    out.mkdir(parents=True)
    files={
      'interpretation_directory':{'training_supported_shape_curves.csv':'training_supported_shape_curves.csv','shape_stability.csv':'shape_stability.csv','adjusted_component_contrasts.csv':'adjusted_component_contrasts.csv'},
      'education_directory':{'results/adjusted_education_tests.csv':'education_tests.csv','results/education_descriptives.csv':'education_descriptives.csv'},
      'paired_directory':{'paired_AP_tests.csv':'paired_AP_tests.csv','current_model_paired_comparisons.csv':'current_model_paired_comparisons.csv'},
      'genetic_directory':{'results/paired_contrasts.csv':'genetic_contrasts.csv','results/metrics.csv':'genetic_metrics.csv'},
      'pet_directory':{'results/aggregate_slopes.csv':'prs_amyloid_slopes.csv'}}
    for key,names in files.items():
        for src,dest in names.items():shutil.copy2(input_path(key)/src,out/dest)
    education=pd.read_parquet(input_path('education_directory')/'LOCAL_RESTRICTED/primary_final75.parquet')
    frequency=education.dropna(subset=['education']).groupby(['status','education'],sort=True).size().rename('n').reset_index()
    assert int(frequency.n.sum())==int(education.education.notna().sum())
    frequency.to_csv(out/'education_frequency.csv',index=False)
    from . import metrics
    metrics.HERE=input_path('prediction_directory')
    data,_=metrics.load();rows=[]
    for task in ('CN_MCI','MCI_AD'):
        for h in (3,5):
            cells=[data[(task,f,h)] for f in (1,2,3)]
            m=pd.concat([c[0] for c in cells],ignore_index=True);keep=m.first_eligible.to_numpy(bool)
            for name in metrics.MODELS:
                pred=np.concatenate([c[1][name] for c in cells])[keep]
                q=m.loc[keep];rows.append(dict(task=task,horizon=h,model=name,**metrics.scores(q.y,pred,q.ipcw,len(q))))
    pd.DataFrame(rows).to_csv(out/'first_visit_metrics_recomputed.csv',index=False)
    d=pd.read_parquet(input_path('interpretation_directory')/'LOCAL_RESTRICTED/fixed_groups_OOF_contributions.parquet')
    d=d[d.model.eq('PRISM_current') & d.horizon.eq(3)].copy()
    assert len(d)>0 and d.subject_id.is_unique
    rows=[]
    for group,g in d.groupby('group'):
        for threshold in np.linspace(0,1,201):
            count=int(g.predicted_risk.le(threshold).sum())
            rows.append(dict(group=group,threshold=float(threshold),n=len(g),count_le=count,cdf=count/len(g)))
    pd.DataFrame(rows).to_csv(out/'clinical_origin_risk_cdf.csv',index=False)
    from .age_summary import summarize
    ages,_,_=summarize();ages.to_csv(out/'age_summary.csv',index=False)
    forbidden={'subject_id','sample_id','SUBJID','IID','FID'}
    for f in out.glob('*.csv'):
        assert not forbidden.intersection(pd.read_csv(f,nrows=0).columns),f.name
    receipt=dict(scope='Aggregate export only; no participant rows exported',output_sha256={f.name:sha(f) for f in sorted(out.glob('*.csv'))})
    (workspace()/'export.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(aggregate_tables=len(receipt['output_sha256']))))
if __name__=='__main__':main()
