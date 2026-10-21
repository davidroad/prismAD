"""A deterministic synthetic example with no participant records."""
import json
import numpy as np
from .features import CURRENT,HISTORY,Transform,endpoint
def run(with_model=False):
    rng=np.random.default_rng(42);n=64
    c=rng.normal(size=(n,len(CURRENT)));h=rng.normal(size=(n,len(HISTORY)))
    prs=rng.normal(size=n);events=rng.normal(size=(n,16,7)).astype('float32')
    mask=rng.random((n,16))>.3;mask[0]=False
    pre=Transform().fit(c[:48],h[:48],prs[:48],events[:48],mask[:48])
    bundle=pre.apply(c,h,prs,events,mask,np.array(['ADNI']*n),True)
    rec,_=endpoint(np.array([70.,71.,72.]),np.array([1.,2.,3.]),0,1)
    assert rec['event']==1 and rec['end_time']==1.
    result=dict(synthetic_records=n,endpoint_check=True,finite_features=bool(np.isfinite(bundle['static']).all()),
                training_only_transform=True,model_check='not_requested')
    if with_model:
        import torch
        from .model import StructuralPrismAD
        from .genetic_control import NoPRSPrism,no_prs
        torch.set_num_threads(1);torch.manual_seed(42);torch.backends.mha.set_fastpath_enabled(False)
        def t(b):return {k:torch.as_tensor(b[k]) for k in ['static','prs','event_numeric','event_domain','event_mask','severity']}
        model=StructuralPrismAD(bundle['names'],bundle['static'][:48],'relative_history',pre)
        args=t(bundle);model.eval()
        with torch.no_grad():
            out=model(**args)
            err=(out['risk_logit']-out['risk_group_contribution'].sum(1)).abs().max().item()
            assert err<2e-6 and out['event_signed_contribution'][0].eq(0).all()
            relative=model.relative_inputs(args['static'],args['event_numeric'],args['event_domain'],args['event_mask']).numpy()
            expected=(c[:,4:8][:,np.tile(np.arange(4),4)]-events[:,:,0])/pre.event_value_scale
            np.testing.assert_allclose(relative[:,:,7][mask],expected[mask],atol=2e-5,rtol=1e-5)
        model.train()
        loss=model(**args)['risk_logit'].square().mean()+.0001*model.regularization_penalty()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        current=no_prs(pre.apply(c,h,prs,events,mask,np.array(['ADNI']*n),False))
        control=NoPRSPrism(current['names'],current['static'][:48],'relative_history',pre);control.eval()
        a=t(current)
        with torch.no_grad():
            original=control(**a)['risk_logit'];a['prs']=torch.randn_like(a['prs'])*100
            assert torch.equal(original,control(**a)['risk_logit'])
        result.update(model_check='passed',native_ledger_max_abs_error=err,
                      no_history_zero=True,relative_cognition_raw_units=True,
                      finite_backward_gradients=True,clinical_only_PRS_invariance=True)
    print(json.dumps(result,indent=2))
    return result
