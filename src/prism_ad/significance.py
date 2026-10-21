"""Paired participant-bootstrap AP inference and full-family BH checks."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import average_precision_score
from .runtime import sha

def read(path):
    return pd.read_csv(path)
B=9999

def ap_batch(y,p,weights):
    order=np.argsort(-p,kind='stable')
    # Evaluate recall changes at score-block ends, matching sklearn for ties.
    ends=np.r_[np.flatnonzero(np.diff(p[order])!=0),len(p)-1]
    ww=weights[:,order]
    tp=np.cumsum(ww*y[order],axis=1)[:,ends]
    total=np.cumsum(ww,axis=1)[:,ends]
    precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
    dt=np.diff(np.c_[np.zeros(len(weights)),tp],axis=1)
    return (precision*dt).sum(1)/tp[:,-1]

def bootstrap(m,preds,pairs):
    keys=pd.MultiIndex.from_frame(m[['fold','subject_id']])
    ix,unique=pd.factorize(keys,sort=True)
    folds=np.array([u[0] for u in unique])
    groups=[np.flatnonzero(folds==f) for f in (1,2,3)]
    rng=np.random.default_rng(42)
    y=m.y.to_numpy(int);base=m.ipcw.to_numpy(float)
    models=sorted(set(n for pair in pairs for n in pair))
    point={n:average_precision_score(y[base>0],preds[n][base>0],sample_weight=base[base>0]) for n in models}
    for n in models:
        np.testing.assert_allclose(ap_batch(y,preds[n],base[None,:])[0],point[n],atol=1e-12)
    draws={n:[] for n in models}
    for start in range(0,B,64):
        k=min(64,B-start);counts=np.zeros((k,len(unique)),int)
        for j in range(k):
            for g in groups:
                chosen=rng.choice(g,len(g),replace=True)
                counts[j]+=np.bincount(chosen,minlength=len(unique))
        w=counts[:,ix]*base
        assert ((w*y).sum(1)>0).all() and ((w*(1-y)).sum(1)>0).all()
        for n in models:draws[n].append(ap_batch(y,preds[n],w))
    draws={n:np.concatenate(x) for n,x in draws.items()}
    out=[]
    for a,b in pairs:
        d=point[a]-point[b];dd=draws[a]-draws[b]
        exceed=int((np.abs(dd-d)>=abs(d)).sum());p=(exceed+1)/(B+1)
        out.append(dict(model=a,reference=b,difference=d,p=p,
            p_mc_se=float(np.sqrt(p*(1-p)/(B+1))),n_subjects=len(unique),
            n_rows=len(m),bootstrap_replicates=B,ci_low_9999=np.quantile(dd,.025),ci_high_9999=np.quantile(dd,.975)))
    return out

def verify_t_family(path,qcol,size):
    d=read(path);assert len(d)==size
    good=d.status.eq('ok');z=d[good]
    se=(z.ci_high-z.ci_low)/(2*t.ppf(.975,z.residual_df))
    p=2*t.sf(np.abs(z.beta/se),z.residual_df)
    np.testing.assert_allclose(p,z.p,rtol=2e-9,atol=1e-14)
    expected=multipletests(d.p.fillna(1),method='fdr_bh')[1]
    np.testing.assert_allclose(expected[good],d.loc[good,qcol],rtol=1e-10,atol=1e-14)
    if qcol=='q_all_128':np.testing.assert_allclose(expected,d[qcol])
    return dict(planned=size,estimable=int(good.sum()),fdr_significant=int((expected[good]<.05).sum()),
        verification='Unrounded t/CI/P algebra and full-family BH; not an independent raw-data refit',source_sha256=sha(path))
