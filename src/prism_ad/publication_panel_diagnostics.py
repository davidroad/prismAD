"""Run newly requested post hoc panel diagnostics in private output storage."""
from pathlib import Path
import json,os
import pandas as pd
from .runtime import input_path,workspace,sha
from .panel_diagnostics import run_calibration_diagnostics
from .marginal_tests import points_from_profiles,marginal_tests,bh_ten_family

def _private(path):
    path=Path(path).resolve()
    for parent in [path]+list(path.parents):
        if parent.name in {"Code","PRISM-AD_Public","PRISM-AD_Local"} or (parent/".git").exists():
            raise RuntimeError("All participant values and empirical outputs must stay outside code packages and Git")
    return path

def main():
    os.umask(0o077)
    out=_private(workspace())
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Choose a new empty private diagnostics directory")
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    plan=input_path("panel_analysis_plan")
    contract=json.loads(plan.read_text())
    if (contract["calibration"]["test_slots"]!=4 or contract["marginal"]["test_slots"]!=6
            or contract["calibration"]["logit_clip"]!=1e-8
            or contract["marginal"]["minimum_group_per_fold"]!=5):
        raise RuntimeError("The supplied frozen diagnostic plan differs from the implemented settings")
    profiles=input_path("current_profile_components")
    hashes={str(plan):sha(plan),str(profiles):sha(profiles)}
    source=input_path("source_data_directory")/"plotting/calibration_pooled_first_visit.csv"
    run_calibration_diagnostics(input_path("prediction_directory"),out/"calibration",
                                pooled_bins_path=source if source.exists() else None)
    calibration=pd.read_csv(out/"calibration/calibration_slope_tests_raw.csv")
    points=points_from_profiles(pd.read_csv(profiles))
    marginal,details=marginal_tests(points)
    cal,mar,all_tests=bh_ten_family(calibration,marginal)
    points.to_csv(out/"marginal_component_points.csv",index=False)
    cal.to_csv(out/"calibration_slope_tests.csv",index=False)
    mar.to_csv(out/"marginal_component_tests.csv",index=False)
    all_tests.to_csv(out/"panel_tests_BH10.csv",index=False)
    (out/"marginal_strata.json").write_text(json.dumps(details,indent=2)+"\n")
    assert all(sha(Path(path))==value for path,value in hashes.items())
    (out/"execution.json").write_text(json.dumps({
        "status":"PASS","input_sha256":hashes,"test_family_size":10,
        "profile_participants":int(len(points)//2),"point_rows":len(points),
        "participant_aliases_or_ids_exported":False,"outputs_private":True,
        "predictions_recalibrated":False,"prognostic_models_refitted":False,
        "new_posthoc_diagnostic_fits":4,
        "scope":"Conditional on frozen forecasts, fitted component functions and IPCW weights; no external validation."
    },indent=2)+"\n")
    print(json.dumps({"status":"PASS","tests":len(all_tests),"private_output":str(out)}))
if __name__=="__main__":main()
