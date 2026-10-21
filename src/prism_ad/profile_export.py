"""Export current figure summaries from controlled saved contributions."""
import json
import numpy as np,pandas as pd
from .runtime import input_path,workspace,sha
def main():
    path=input_path('current_profile_components');original=sha(path)
    d=pd.read_csv(path)
    assert len(d)==893 and d.display_alias.is_unique
    d['C_decile']=pd.qcut(d.cognition_main,10,labels=False)+1
    d['PRS_quintile']=pd.qcut(d.raw_PRS,5,labels=False)+1
    rows=[];quintiles=[];correlations=[]
    for group in ['All','CN','MCI','AD']:
        z=d if group=='All'else d[d.clinical_group==group]
        correlations.append(dict(group=group,n=len(z),
            spearman_G_C=z[['PRS','cognition_main']].corr(method='spearman').iloc[0,1],
            pearson_G_C=z[['PRS','cognition_main']].corr().iloc[0,1]))
        for decile,v in z.groupby('C_decile'):
            rows.append(dict(group=group,C_decile=int(decile),n=len(v),
                mean_C=v.cognition_main.mean(),mean_G=v.PRS.mean(),
                min_C=v.cognition_main.min(),max_C=v.cognition_main.max(),
                cognition_factor=np.exp(v.cognition_main.mean()),PRS_factor=np.exp(v.PRS.mean())))
        for (fold,bin_),v in z.groupby(['fold','PRS_quintile']):
            quintiles.append(dict(fold=int(fold),group=group,bin=int(bin_),n=len(v),
                mean_signed_C=v.cognition_main.mean(),mean_signed_G=v.PRS.mean()))
    out=workspace();out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(out/'cognitive_decile_profiles.csv',index=False)
    pd.DataFrame(quintiles).to_csv(out/'overall_components_fold_prs_quintile_summary.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'course_component_correlations.csv',index=False)
    assert sha(path)==original
    (out/'execution.json').write_text(json.dumps(dict(input_sha256=original,
        participant_rows_exported=False,models_refitted=False,tests_performed=False),indent=2)+'\n')
if __name__=='__main__':main()
