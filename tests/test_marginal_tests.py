"""Meaningful statistical invariance/edge tests; all inputs synthetic."""
import unittest,itertools
import numpy as np,pandas as pd
from prism_ad.marginal_tests import _stratum,stratified_rank_test

class StratifiedRankTests(unittest.TestCase):
    def test_tie_variance_matches_enumerated_null(self):
        values=np.array([0.,0.,1.,2.,3.,3.]);u=[]
        for idx in itertools.combinations(range(6),3):
            keep=np.zeros(6,dtype=bool);keep[list(idx)]=True
            b=values[keep];a=values[~keep]
            u.append(sum(float(y>x)+.5*float(y==x) for x in a for y in b))
        cell=_stratum(values[:3],values[3:])
        self.assertAlmostEqual(np.var(u,ddof=0),cell['variance_u'],places=12)
        self.assertAlmostEqual(np.mean(u),cell['mean_u'],places=12)
    def data(self):
        out=[]
        for f in [1,2,3]:
            for g,shift in [('CN',0.),('AD',.8)]:
                for value in np.linspace(-1,1,8)+shift:
                    out.append(dict(component='G',fold=f,clinical_group=g,value=value+2*f))
        return pd.DataFrame(out)
    def test_positive_foldwise_transform_preserves_test_and_effect(self):
        a=self.data();b=a.copy();b['value']=[(10**f)*x-100*f for x,f in zip(b.value,b.fold)]
        x=stratified_rank_test(a,'G','CN','AD');y=stratified_rank_test(b,'G','CN','AD')
        self.assertAlmostEqual(x['z'],y['z'],places=12)
        self.assertAlmostEqual(x['rank_biserial_b_minus_a'],y['rank_biserial_b_minus_a'],places=12)
    def test_group_swap_reverses_direction_not_p(self):
        d=self.data();a=stratified_rank_test(d,'G','CN','AD');b=stratified_rank_test(d,'G','AD','CN')
        self.assertAlmostEqual(a['z'],-b['z'],places=12)
        self.assertAlmostEqual(a['p'],b['p'],places=12)
        self.assertAlmostEqual(a['rank_biserial_b_minus_a'],-b['rank_biserial_b_minus_a'],places=12)
    def test_all_ties_are_not_estimable(self):
        d=self.data();d['value']=0.;r=stratified_rank_test(d,'G','CN','AD')
        self.assertFalse(r['estimable']);self.assertEqual(r['p'],1.)
if __name__=='__main__':unittest.main()
