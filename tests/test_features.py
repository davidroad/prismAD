"""Synthetic checks; deliberately contains no patient records."""
import unittest
import numpy as np
import pandas as pd
from prism_ad.features import (CURRENT, HISTORY, STUDIES, Censoring, ReverseKM, Transform,
                        endpoint, history_features, shared_bootstrap)


class PilotTests(unittest.TestCase):
    def test_cn_target_first(self):
        r, _ = endpoint(np.array([70.,71.,72.]), np.array([1.,2.,3.]), 0, 1)
        self.assertEqual((r['event'],r['end_time']), (1,1.))

    def test_cn_direct_ad_competing(self):
        r, _ = endpoint(np.array([70.,72.]),np.array([1.,3.]),0,1)
        self.assertEqual(r['event'],2)

    def test_censor_undated_first_event_at_previous_negative(self):
        r, _ = endpoint(np.array([70.,71.,np.nan,74.]),np.array([1.,1.,2.,3.]),0,1)
        self.assertEqual((r['event'],r['end_time'],r['uncertain_event']), (0,1.,True))

    def test_unknown_diagnosis_not_negative_followup(self):
        r, reason = endpoint(np.array([70.,72.]),np.array([1.,np.nan]),0,1)
        self.assertIsNone(r)
        self.assertEqual(reason,'no_positive_classifiable_followup')

    def test_mci_cn_reversion_still_at_risk(self):
        r, _ = endpoint(np.array([70.,71.,72.,74.]),np.array([2.,1.,2.,3.]),2,2)
        self.assertEqual((r['event'],r['end_time']), (1,2.))

    def test_prior_mci_cn_ineligible(self):
        r, _ = endpoint(np.array([70.,71.,72.]),np.array([2.,1.,2.]),1,1)
        self.assertIsNone(r)

    def test_prior_ad_mci_ineligible(self):
        r, _ = endpoint(np.array([70.,71.,72.]),np.array([3.,2.,3.]),1,2)
        self.assertIsNone(r)

    def test_age_boundary(self):
        self.assertIsNone(endpoint(np.array([85.,86.]),np.array([1.,2.]),0,1)[0])
        self.assertIsNotNone(endpoint(np.array([84.9,86.]),np.array([1.,2.]),0,1)[0])

    def test_no_future_or_current_information_in_history(self):
        ages=np.arange(70.,76.); visits=np.arange(1.,7.)
        values=np.arange(24.).reshape(6,4); se=np.ones_like(values); precision=np.ones_like(values)
        a=history_features(ages,visits,values,se,precision,3)
        values[3:]=999.; se[3:]=999.; precision[3:]=0
        b=history_features(ages,visits,values,se,precision,3)
        for x,y in zip(a,b): np.testing.assert_equal(x,y)

    def test_equal_age_not_past(self):
        h,ev,m=history_features(np.array([70.,70.,71.]),np.arange(3.),np.ones((3,4)),np.ones((3,4)),np.ones((3,4)),1)
        self.assertFalse(m.any()); self.assertEqual(h[0],0)

    def test_last_four_visits(self):
        values=np.tile(np.arange(8.)[:,None],(1,4))
        h,ev,m=history_features(np.arange(70.,78.),np.arange(8.),values,values,values,7)
        np.testing.assert_equal(ev[::4,0],[3,4,5,6])
        self.assertEqual(h[0],28); self.assertEqual(m.sum(),16)

    def test_raw_negative_one_preserved(self):
        v=np.full((2,4),-1.)
        h,ev,m=history_features(np.array([70.,71.]),np.array([1.,2.]),v,v,v,1)
        self.assertEqual(m.sum(),4); self.assertEqual(ev[m,0].min(),-1.)

    def test_slope_uses_past_only(self):
        v=np.tile(np.array([2.,1.,100.])[:,None],(1,4))
        h,_,_=history_features(np.array([70.,71.,72.]),np.arange(3.),v,v,v,2)
        self.assertAlmostEqual(h[4],-5.)

    def test_reverse_km_left_limit_and_ties(self):
        km=ReverseKM().fit([1,2,2,3],[0,1,0,1])
        np.testing.assert_allclose(km.before([1,2,3]),[1,.75,.5])

    def test_unknown_is_not_control_and_competing_is_control(self):
        m=pd.DataFrame(dict(study=['ADNI']*4,age=[70.]*4,end_time=[1.,2.,2.,4.],event=[0,1,2,0]))
        lab=Censoring().fit(m).label(m,3.)
        np.testing.assert_equal(lab['known'],[False,True,True,True])
        np.testing.assert_equal(lab['y'],[0,1,0,0])
        np.testing.assert_allclose(lab['weight'],[0,4/3,4/3,4/3])

    def test_shared_seed_deterministic(self):
        lab=dict(known=np.ones(20,bool),weight=np.arange(1.,21.))
        np.testing.assert_equal(shared_bootstrap(lab,42),shared_bootstrap(lab,42))

    def test_transform_invertible_and_current_history_matched(self):
        rng=np.random.default_rng(42); n=20
        c=rng.normal(size=(n,len(CURRENT))); h=rng.normal(size=(n,len(HISTORY)))
        p=rng.normal(size=n); ev=rng.normal(size=(n,16,7)).astype(np.float32); mask=np.ones((n,16),bool)
        c[0,4]=np.nan
        pre=Transform().fit(c,h,p,ev,mask)
        out=pre.apply(c,h,p,ev,mask,np.array(['ADNI']*n),True)
        inverse=out['static'][:,4:8]*pre.scale[4:8]+pre.median[4:8]
        np.testing.assert_allclose(inverse[1:],c[1:,4:8],rtol=1e-5,atol=1e-6)
        cur=pre.apply(c,h,p,ev,mask,np.array(['ADNI']*n),False)
        self.assertFalse(cur['event_mask'].any())
        self.assertTrue(np.all(cur['event_numeric']==0))
        self.assertEqual(out['names'][-4:],['study='+s for s in STUDIES])


if __name__=='__main__': unittest.main(verbosity=2)
