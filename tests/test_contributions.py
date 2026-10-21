import unittest
import numpy as np
import pandas as pd
from prism_ad.contributions import account, centered, correct_family, exact_join, main_group, contrasts


class InterpretationTests(unittest.TestCase):
    def test_missing_history_is_missingness_not_trajectory(self):
        self.assertEqual(main_group('strict_history_events__missing'),'missingness_main')
        self.assertEqual(main_group('strict_history_events'),'past_summary_main')

    def test_disjoint_closure(self):
        names=['PHC_Education','PHC_MEM_z','PHC_MEM_z__missing','study=ADNI']
        main=np.array([[.2,.3,.1,-.1]])
        ledger=np.array([[.1,.5,.05,0,.15,-.1]])
        d=account(names,main,ledger)
        self.assertAlmostEqual(float(sum(d.values())[0]),.7)

    def test_center_removes_offset(self):
        np.testing.assert_allclose(centered([1,2,4]),centered([7,8,10]))

    def test_exact_identity_and_age(self):
        g=pd.DataFrame(dict(subject_id=['a','b'],visit=[1,1],Study=['S','S'],cognition_age=[70.,71.]))
        p=pd.DataFrame(dict(subject_id=['b','a'],visit=[1,1],study=['S','S'],age=[72.,70.]))
        self.assertEqual(exact_join(g,p).subject_id.tolist(),['a'])

    def test_duplicate_prediction_rejected(self):
        g=pd.DataFrame(dict(subject_id=['a'],visit=[1],Study=['S'],cognition_age=[70.]))
        p=pd.DataFrame(dict(subject_id=['a','a'],visit=[1,1],study=['S','S'],age=[70.,70.]))
        with self.assertRaises(AssertionError):exact_join(g,p)

    def test_family_keeps_missing_tests(self):
        q=correct_family([dict(p=.001)]+[dict(p=np.nan) for _ in range(127)])
        self.assertAlmostEqual(q.q_all_128.iloc[0],.128)

    def test_support_failure_not_silently_fitted(self):
        d=pd.DataFrame(dict(group=['resilience_CN']*4,Study=['S']*4,index_age=[70.]*4,
            sex=[1]*4,fold=[1]*4,y=[0.,1.,2.,3.],**{f'PC{i}':[0.]*4 for i in range(1,6)}))
        self.assertTrue(all(x['status']=='insufficient_support' for x in contrasts(d,'y')))


if __name__=='__main__':unittest.main()
