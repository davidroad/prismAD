import unittest
import numpy as np
import pandas as pd
from prism_ad.publication_summaries import pool_calibration,marginal_components
class PublicationSummaries(unittest.TestCase):
    def test_pool_uses_weight_mass_not_fold_mean(self):
        d=pd.DataFrame([dict(model="PRISM",condition="plusPRS",subset="first_eligible_landmark",
          fold=f,task="CN_MCI",horizon=3,risk_lo=0.,risk_hi=.1,n_known=10,
          weight_mass=w,mean_prediction=p,observed_prevalence=p)for f,w,p in [(1,1.,.1),(2,8.,.9),(3,1.,.1)]])
        q=pool_calibration(d).iloc[0]
        self.assertAlmostEqual(q.prediction,.74);self.assertEqual(q.n_known,30)
    def test_missing_fold_slots_rejected(self):
        d=pd.DataFrame([dict(model="PRISM",condition="plusPRS",subset="first_eligible_landmark",
          fold=1,task="CN_MCI",horizon=3,risk_lo=0.,risk_hi=.1,n_known=2,
          weight_mass=2.,mean_prediction=.1,observed_prevalence=0.)])
        with self.assertRaises(ValueError):pool_calibration(d)
    def test_marginals_keep_fold_scales_separate(self):
        d=pd.DataFrame([dict(display_alias=str(i),fold=f,clinical_group="CN",cognition_main=x,PRS=x/10)
                        for i,(f,x)in enumerate([(1,0.),(1,2.),(2,100.),(2,102.)])])
        q=marginal_components(d)
        self.assertEqual(len(q),4)
        self.assertEqual(list(q[q.component.eq("C")]["median"]),[1.,101.])
        self.assertEqual(list(q[q.component.eq("C")].q75-q[q.component.eq("C")].q25),[1.,1.])
    def test_duplicate_participant_rows_rejected(self):
        d=pd.DataFrame([dict(display_alias="one",fold=1,clinical_group="CN",cognition_main=0.,PRS=0.)]*2)
        with self.assertRaises(ValueError):marginal_components(d)
if __name__=="__main__":unittest.main()
