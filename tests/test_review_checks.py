"""Synthetic checks for diagnostic timing and model-scale interpretation."""
import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from prism_ad.eligibility_audit import audit,prior_flags
from prism_ad.model_scale_checks import analyze,private_output
from prism_ad.cognitive_strata import fit
from prism_ad.features import endpoint

class ReviewChecks(unittest.TestCase):
    def test_post_prediction_AD_reversion_remains_valid_prediction(self):
        visits=pd.DataFrame(dict(subject_id=["synthetic"]*3,visit=[0.,1.,2.],
                                 diagnosis=[1,3,1],age=[70.,72.,73.]))
        meta=pd.DataFrame([dict(subject_id="synthetic",visit=0.,task="CN_MCI",fold=1,
                               first_eligible=True,event=2,end_time=2.,uncertain_event=False)])
        course=pd.DataFrame([dict(subject_id="synthetic",visit=0.,prior_MCI_AD=False,
            final_status="excluded_prior_AD_reversion",first_AD_recorded=True,
            first_AD_elapsed=2.,course_status="final_MCI_unknown_or_other_state")])
        result=audit(meta,visits,course)
        self.assertEqual(result["status"],"PASS")
        self.assertEqual(result["AD_followed_by_reversion"],1)
        self.assertEqual(result["prior_disqualifying_rows"],0)
    def test_AD_before_prediction_is_disqualifying(self):
        v=pd.DataFrame(dict(subject_id=["synthetic"]*3,visit=[0.,1.,2.],
                            diagnosis=[3,1,1],age=[70.,72.,73.]))
        flags=prior_flags(v);self.assertTrue(flags.iloc[1].prior_MCI_AD)
        rec,why=endpoint(v.age.to_numpy(),v.diagnosis.to_numpy(),1,1)
        self.assertIsNone(rec);self.assertEqual(why,"previous_disqualifying_state")
    def data(self):
        rng=np.random.default_rng(719);rows=[]
        for fold,scale in [(1,.2),(2,.4),(3,.7)]:
            n=180;c=rng.normal(size=n);ad=(np.arange(n)%3==0)
            raw=rng.normal(size=n)+.4*ad+.1*c
            rows.append(pd.DataFrame(dict(fold=fold,clinical_group=np.where(ad,"AD","CN"),
                 cognition_main=c,raw_PRS=raw,PRS=fold+scale*raw,Study="synthetic")))
        return pd.concat(rows,ignore_index=True)
    def test_affine_G_does_not_create_a_new_genetic_signal(self):
        d=self.data();out,identity=analyze(d,0.,1.)
        d["C_decile"]=pd.qcut(d.cognition_main,10,labels=False)+1
        for fold,scale in [(1,.2),(2,.4),(3,.7)]:
            z=d[d.fold.eq(fold)].copy();z["PRS"]=z.raw_PRS
            raw=fit(z,"raw",True);g=out[out.test.eq("within_fold_"+str(fold))].iloc[0]
            self.assertAlmostEqual(g.beta,scale*raw["beta"],places=10)
            self.assertAlmostEqual(g.p,raw["p"],places=10)
        self.assertTrue(all(x["max_abs_error"]<1e-10 for x in identity))
    def test_raw_PRS_uses_SD_units_not_odds_factors(self):
        a,_=analyze(self.data(),0.,1.);b,_=analyze(self.data(),0.,2.)
        x=a[a.test.eq("pooled_raw_PRS_SD")].iloc[0];y=b[b.test.eq("pooled_raw_PRS_SD")].iloc[0]
        self.assertAlmostEqual(x.beta/2,y.beta,places=10)
        self.assertAlmostEqual(x.p,y.p,places=10)
        self.assertTrue(np.isnan(x.factor_ratio));self.assertEqual(x.units,"full_roster_SD")
        self.assertEqual(len(a),4)
    def test_protected_output_cannot_enter_code_package(self):
        with tempfile.TemporaryDirectory()as tmp:
            target=Path(tmp)/"PRISM-AD_Public"/"outputs"
            with patch.dict(os.environ,{"PRISM_AD_WORKSPACE":str(target)}):
                with self.assertRaises(RuntimeError):private_output()
if __name__=="__main__":unittest.main()
