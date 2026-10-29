import unittest
import numpy as np,pandas as pd
from prism_ad.followup_sensitivity import ad_endpoints,bh_family,make_design,five_year_contrast
from prism_ad.features import Censoring
from prism_ad.transitions import normalize_visits

class FollowupSensitivityTests(unittest.TestCase):
 def frame(self,diagnoses,ages):
  raw=pd.DataFrame(dict(SUBJID=["synthetic"]*len(ages),Study=["NACC"]*len(ages),
       PHC_Visit=np.arange(len(ages)),PHC_Age_Cognition=ages,PHC_Diagnosis=diagnoses,
       PHC_MEM=np.ones(len(ages)),PHC_EXF=np.ones(len(ages)),
       PHC_LAN=np.ones(len(ages)),PHC_VSP=np.ones(len(ages))))
  first=pd.DataFrame(dict(subject_id=["synthetic"],visit=[0.]))
  return ad_endpoints(first,normalize_visits(raw)).iloc[0]
 def test_MCI_is_intermediate_for_AD(self):
  r=self.frame([1,2,3],[70.,72.,76.])
  self.assertEqual(r.event,1);self.assertEqual(r.end_time,6.)
 def test_later_reversion_does_not_erase_first_AD(self):
  r=self.frame([1,3,1,2],[70.,73.,74.,75.])
  self.assertEqual(r.event,1);self.assertEqual(r.end_time,3.)
 def test_undated_first_AD_not_replaced_by_later_AD(self):
  r=self.frame([1,2,3,3],[70.,72.,np.nan,76.])
  self.assertEqual(r.event,0);self.assertEqual(r.end_time,2.);self.assertTrue(r.uncertain_event)
 def test_topcoded_first_AD_censors_at_preceding_nonAD(self):
  r=self.frame([1,2,3],[70.,72.,"90+"])
  self.assertEqual(r.event,0);self.assertEqual(r.end_time,2.);self.assertTrue(r.uncertain_event)
 def test_unknown_diagnosis_does_not_extend_observation(self):
  r=self.frame([1,2,np.nan],[70.,72.,76.])
  self.assertEqual(r.end_time,2.)
 def test_horizon_cases_controls_unknown_and_new_weights(self):
  d=pd.DataFrame(dict(end_time=[2.,8.,4.,7.],event=[1,1,0,0],
       age=[70.]*4,study=["NACC"]*4))
  z=Censoring().fit(d).label(d,5.)
  np.testing.assert_array_equal(z["known"],[True,True,False,True])
  np.testing.assert_array_equal(z["y"],[1,0,0,0])
  self.assertEqual(z["weight"][2],0.)
 def synthetic_weighted_data(self):
  rng=np.random.default_rng(24);n=120
  ad=(np.arange(n)%6==0).astype(int)
  return pd.DataFrame(dict(fold=np.tile([1,2,3],40),cognition_main=rng.normal(size=n),
       age=rng.uniform(60,83,n),sex=rng.integers(1,3,n),study=np.where(np.arange(n)%2,"A","B"),
       raw_PRS_SD=.7*ad+rng.normal(size=n),AD_within_horizon=ad,
       known_horizon=np.ones(n,dtype=bool),complete_adjustment=np.ones(n,dtype=bool),
       AD_ipcw=rng.uniform(1,2,n)))
 def test_weighted_contrast_retains_score_units(self):
  d=self.synthetic_weighted_data();a=five_year_contrast(d)
  d.raw_PRS_SD=3*d.raw_PRS_SD+2;b=five_year_contrast(d)
  self.assertAlmostEqual(b["beta"],3*a["beta"],places=10)
  self.assertAlmostEqual(b["se"],3*a["se"],places=10)
  self.assertAlmostEqual(a["p"],b["p"],places=10)
 def test_fold_specific_C_reference_shift_does_not_change_contrast(self):
  d=self.synthetic_weighted_data();a=five_year_contrast(d)
  d.cognition_main=d.cognition_main+10*d.fold
  b=five_year_contrast(d)
  self.assertAlmostEqual(a["beta"],b["beta"],places=10)
  self.assertAlmostEqual(a["se"],b["se"],places=10)
 def test_separate_two_test_BH_preserves_unavailable_slot(self):
  np.testing.assert_allclose(bh_family([.03,1.]),[.06,1.])

if __name__=="__main__":unittest.main()
