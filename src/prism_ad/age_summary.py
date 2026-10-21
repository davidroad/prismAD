"""Aggregate age distributions from controlled validation records."""
import numpy as np
import pandas as pd
from .runtime import input_path,sha
WORK=input_path("prediction_directory")
TASKS=[("CN_MCI","CN to MCI"),("MCI_AD","MCI to AD")]
EPS=1e-8
def label(task, group):
    target = 'MCI' if task == 'CN_MCI' else 'AD'
    return {'target': 'Recorded '+target,
            'event_free': 'No MCI or AD' if task == 'CN_MCI' else 'No recorded AD',
            'competing_ad': 'AD before MCI',
            'unknown': 'Follow-up incomplete', 'all': 'All participants'}[group]

def summarize():
    sources = {}; rows = []; totals = []
    metrics_path = input_path('prediction_directory')/'results/metrics.csv'
    metrics = pd.read_csv(metrics_path)
    sources[str(metrics_path.relative_to(WORK))] = sha(metrics_path)
    for task, _ in TASKS:
        previous = None
        for horizon in (3, 5):
            frames = []
            for fold in (1, 2, 3):
                p = input_path('prediction_directory')/'LOCAL_RESTRICTED'/task/('fold%d_h%d' % (fold, horizon))/'validation.csv'
                sources[str(p.relative_to(WORK))] = sha(p)
                frame = pd.read_csv(p, dtype={'subject_id': str})
                assert frame.fold.eq(fold).all() and frame.task.eq(task).all()
                assert frame.first_eligible.dtype == bool and frame.known.dtype == bool
                frames.append(frame.loc[frame.first_eligible].copy())
            m = pd.concat(frames, ignore_index=True)
            assert not m.subject_id.duplicated().any()
            assert m.age.notna().all() and np.isfinite(m.age).all() and m.age.lt(85).all()
            if previous is not None:
                pd.testing.assert_frame_equal(previous, m[['subject_id','fold','visit','age']])
            previous = m[['subject_id','fold','visit','age']].copy()
            event_by = m.event.ne(0) & m.end_time.le(horizon+EPS)
            known = event_by | m.end_time.ge(horizon-EPS)
            target = m.event.eq(1) & m.end_time.le(horizon+EPS)
            competing = m.event.eq(2) & m.end_time.le(horizon+EPS)
            assert np.array_equal(known, m.known)
            assert np.array_equal(target.astype(int), m.y)
            assert m.loc[~known, 'ipcw'].eq(0).all()
            groups = {'target': target, 'event_free': known & ~event_by,
                      'competing_ad': competing, 'unknown': ~known}
            assert np.all(sum(g.astype(int) for g in groups.values()) == 1)
            assert not competing.any() if task == 'MCI_AD' else True
            check = metrics[(metrics.task == task) & (metrics.horizon == horizon) &
                            (metrics.fold == 'pooled_123') &
                            (metrics.subset == 'first_eligible_landmark') &
                            (metrics.model == 'PRISM_current')]
            assert len(check) == 1
            expected = check.iloc[0]
            assert (len(m), int(known.sum()), int(target.sum())) == (
                int(expected.n_risk), int(expected.n_known), int(expected.n_target))
            totals.append(dict(task=task, horizon=horizon, n=len(m), n_known=int(known.sum()),
                               n_target=int(target.sum()), n_unknown=int((~known).sum())))
            for group, mask in [('all', np.ones(len(m), dtype=bool))]+list(groups.items()):
                a = m.loc[mask, 'age']
                if len(a) == 0: continue
                rows.append(dict(task=task, horizon=horizon, group=group,
                    group_label=label(task, group), n=len(a), age_min=float(a.min()),
                    age_q1=float(a.quantile(.25)), age_median=float(a.median()),
                    age_q3=float(a.quantile(.75)), age_max=float(a.max()),
                    n_age_ge75=int(a.ge(75).sum()), percent_age_ge75=float(a.ge(75).mean()*100)))
    return pd.DataFrame(rows), totals, sources
