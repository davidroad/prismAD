"""Remove PRS before a fresh clinical-only fit."""
import numpy as np
import torch
from .model import StructuralPrismAD as OriginalPrism
from .features import CURRENT

class NoPRSPrism(OriginalPrism):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prs_raw_coefficient.requires_grad_(False)
        self.use_prs = False

    def effective_prs_coefficient(self):
        return self.clinical_anchor.intercept.new_zeros(())

    def forward(self, static, prs, event_numeric, event_domain, event_mask, severity):
        return super().forward(static, torch.zeros_like(prs), event_numeric,
                               event_domain, event_mask, severity)

def no_prs(bundle):
    assert bundle['tabular'].shape[1] == len(bundle['names']) + 1
    assert bundle['names'] == list(CURRENT) + [n+'__missing' for n in CURRENT] + [
        'study=ADNI', 'study=NACC', 'study=ROS', 'study=MAP - Rush']
    result = dict(bundle)
    result['prs'] = np.zeros_like(bundle['prs'])
    result['tabular'] = bundle['tabular'][:, :-1].copy()
    assert np.array_equal(result['tabular'], bundle['static'])
    assert not result['event_mask'].any() and not result['event_numeric'].any()
    assert not result['prs'].any()
    return result
