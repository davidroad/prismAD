"""Independent numerical checks for the added HC3 and correction wrappers."""
import unittest
import numpy as np,pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from prism_ad.cognitive_strata import design,fit,analyze
class CurrentStatistics(unittest.TestCase):
    @staticmethod
    def data():
        rng=np.random.default_rng(810)
        n=600;score=np.linspace(-2,2,n)
        group=np.where(np.arange(n)%3==0,'AD','CN')
        group[:60]='CN'
        frame=pd.DataFrame(dict(cognition_main=score,clinical_group=group,
            fold=np.arange(n)%3+1,Study=np.where(np.arange(n)%2,'A','B'),
            index_age=rng.normal(75,4,n),sex=np.arange(n)%2))
        # Fold is deliberately not perfectly aligned with group.
        frame['fold']=rng.integers(1,4,n)
        for i in range(1,6):frame['PC'+str(i)]=rng.normal(size=n)
        frame['PRS']=.2*(frame.clinical_group=='AD')+.08*score+rng.normal(0,.3,n)
        frame['C_decile']=pd.qcut(frame.cognition_main,10,labels=False)+1
        return frame
    def test_hc3_t_effect_and_interval_against_statsmodels(self):
        d=self.data();X=design(d,True)
        expected=sm.OLS(d.PRS,X).fit(cov_type='HC3',use_t=True)
        actual=fit(d,'global',True)
        self.assertEqual(actual['status'],'ok')
        np.testing.assert_allclose([actual['beta'],actual['se'],actual['p']],
            [expected.params['AD'],expected.bse['AD'],expected.pvalues['AD']],rtol=2e-10,atol=1e-12)
        np.testing.assert_allclose([actual['ci_low'],actual['ci_high']],
            expected.conf_int().loc['AD'].to_numpy(),rtol=2e-10,atol=1e-12)
    def test_unavailable_decile_keeps_its_BH_slot(self):
        out=analyze(self.data())
        self.assertEqual(len(out),12)
        empty=out[out.test=='C_decile_1'].iloc[0]
        self.assertEqual(empty.status,'group_n_below_5')
        self.assertEqual(empty.marker,'')
        np.testing.assert_allclose(out.q_family12,
            multipletests(out.p.fillna(1),method='fdr_bh')[1],rtol=0,atol=1e-14)
    def test_factor_ratio_is_exponentiated_G_contrast(self):
        r=fit(self.data(),'global',True)
        np.testing.assert_allclose([r['factor_ratio'],r['ratio_ci_low'],r['ratio_ci_high']],
            np.exp([r['beta'],r['ci_low'],r['ci_high']]),rtol=0,atol=1e-14)
if __name__=='__main__':unittest.main()
