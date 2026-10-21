"""Pure landmark, history, training-only transform and IPCW functions.

The endpoint is a recorded diagnosis, not biological onset. No file access here.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pandas as pd

STUDIES = ('ADNI', 'NACC', 'ROS', 'MAP - Rush')
DOMAINS = ('MEM', 'EXF', 'LAN', 'VSP')
LONG = ('memory', 'executive', 'language', 'visuospatial')
CURRENT = ('visit_age', 'PHC_Sex', 'PHC_Education', 'global_z',
           'PHC_MEM_z', 'PHC_EXF_z', 'PHC_LAN_z', 'PHC_VSP_z', 'n_domains')
HISTORY = ('strict_history_events', 'strict_history_span_years',
           'strict_history_domains', 'strict_history_time_since_last_years',
           'strict_history_common_pace_per_5y') + tuple(
    f'strict_history_{d}_{f}' for d in LONG
    for f in ('last', 'time_since_last', 'pace_per_5y', 'events'))
EPS = 1e-8


def endpoint(age, diagnosis, j, state):
    """Arrays are in visit order. An undated first event terminates observation."""
    if not np.isfinite(age[j]):
        return None, 'unknown_index_age'
    if age[j] >= 85:
        return None, 'index_age_ge85'
    if np.any(np.diff(age[np.isfinite(age)]) < -EPS):
        return None, 'decreasing_age'
    if np.any(np.isin(diagnosis[:j], [2, 3] if state == 1 else [3])):
        return None, 'previous_disqualifying_state'
    stop_codes = [2, 3] if state == 1 else [3]
    first = next((i for i in range(j + 1, len(age)) if diagnosis[i] in stop_codes), None)
    prior_stop = len(age) if first is None else first
    negatives = [i for i in range(j, prior_stop)
                 if diagnosis[i] in ([1] if state == 1 else [1, 2]) and np.isfinite(age[i])]
    last = negatives[-1]
    rec = dict(end_time=float(age[last] - age[j]), event=0, uncertain_event=False,
               interval_lower=np.nan, interval_width=np.nan)
    if first is not None:
        if np.isfinite(age[first]):
            dt = float(age[first] - age[j])
            if dt <= EPS:
                return None, 'nonpositive_event_time'
            rec.update(end_time=dt, event=1 if diagnosis[first] == (2 if state == 1 else 3) else 2,
                       interval_lower=float(age[last] - age[j]),
                       interval_width=float(age[first] - age[last]))
        else:
            rec['uncertain_event'] = True
    if rec['end_time'] <= EPS:
        return None, 'no_positive_classifiable_followup'
    return rec, 'eligible'


def history_features(age, visits, values, se, precise, j):
    """Strictly earlier visit AND age. Never use the current value in a slope."""
    valid = (np.arange(len(age)) < j) & (age < age[j] - EPS) & np.isfinite(values).any(axis=1)
    ix = np.flatnonzero(valid)
    h = np.full(21, np.nan, dtype=float)
    h[0] = np.isfinite(values[ix]).sum()
    h[1] = age[ix[-1]] - age[ix[0]] if len(ix) else 0.
    h[2] = np.isfinite(values[ix]).any(axis=0).sum() if len(ix) else 0.
    h[3] = age[j] - age[ix[-1]] if len(ix) else np.nan
    slopes = []
    for d in range(4):
        di = ix[np.isfinite(values[ix, d])]
        pos = 5 + d * 4
        h[pos + 3] = len(di)
        if len(di):
            h[pos] = values[di[-1], d]
            h[pos + 1] = age[j] - age[di[-1]]
        if len(di) >= 2 and np.ptp(age[di]) > EPS:
            x = age[di] - age[di].mean()
            slope = 5 * np.dot(x, values[di, d] - values[di, d].mean()) / np.dot(x, x)
            h[pos + 2] = slope
            slopes.append(slope)
    h[4] = np.mean(slopes) if slopes else np.nan
    # Slot order: oldest to newest of the last4 eligible prior visits, 4 domains.
    # Padded empty slots at the beginning preserve recency alignment.
    ev = np.zeros((16, 7), dtype=np.float32)
    mask = np.zeros(16, dtype=bool)
    chosen = ix[-4:]
    for slot, i in enumerate(chosen, start=4-len(chosen)):
        for d in range(4):
            s = slot * 4 + d
            if not np.isfinite(values[i, d]):
                continue
            mask[s] = True
            ev[s] = (values[i, d], se[i, d] if np.isfinite(se[i, d]) else 0.,
                     float(not np.isfinite(se[i, d])),
                     precise[i, d] if np.isfinite(precise[i, d]) else 0.,
                     float(not np.isfinite(precise[i, d])),
                     np.log1p(age[j] - age[i]), np.log1p(visits[i]))
    return h, ev, mask


class Transform:
    """Fit on training subjects only; no clipping of invertible PHC units."""
    def fit(self, current, history, prs, events, mask):
        x = np.concatenate((current, history), axis=1).astype(float)
        self.raw_names = list(CURRENT + HISTORY)
        self.median = np.zeros(x.shape[1]); self.scale = np.ones(x.shape[1])
        for j in range(x.shape[1]):
            v = x[np.isfinite(x[:, j]), j]
            if len(v):
                self.median[j] = np.median(v)
                self.scale[j] = max(float(np.subtract(*np.percentile(v, [75, 25]))), 1e-6)
                if self.scale[j] < 1e-4:
                    self.scale[j] = max(float(np.std(v)), 1.)
        self.prs_mean = float(np.mean(prs)); self.prs_scale = max(float(np.std(prs)), 1e-8)
        v = events[:, :, 0][mask]
        self.event_value_mean = float(v.mean()) if len(v) else 0.
        self.event_value_scale = max(float(v.std()), 1e-6) if len(v) else 1.
        self.event_other_mean = []; self.event_other_scale = []
        for col in (5, 6):
            z = events[:, :, col][mask]
            self.event_other_mean.append(float(z.mean()) if len(z) else 0.)
            self.event_other_scale.append(max(float(z.std()), 1e-6) if len(z) else 1.)
        self.numeric = [SimpleNamespace(name=n, median=float(m), scale=float(s), log1p=False)
                        for n, m, s in zip(self.raw_names, self.median, self.scale)]
        return self

    def apply(self, current, history, prs, events, mask, study, use_history):
        raw = np.concatenate((current, history), axis=1)
        z = (np.where(np.isfinite(raw), raw, self.median) - self.median) / self.scale
        n = len(self.raw_names) if use_history else len(CURRENT)
        static = np.concatenate((z[:, :n], (~np.isfinite(raw[:, :n])).astype(float),
                                 np.column_stack([study == s for s in STUDIES]).astype(float)), axis=1).astype(np.float32)
        names = self.raw_names[:n] + [a+'__missing' for a in self.raw_names[:n]] + ['study='+s for s in STUDIES]
        ev = events.copy(); m = mask.copy()
        ev[:, :, 0] = (ev[:, :, 0] - self.event_value_mean) / self.event_value_scale
        ev[:, :, 1] /= self.event_value_scale
        for k, c in enumerate((5, 6)):
            ev[:, :, c] = (ev[:, :, c] - self.event_other_mean[k]) / self.event_other_scale[k]
        if not use_history:
            m[:] = False
        ev[~m] = 0.
        pr = ((prs - self.prs_mean) / self.prs_scale).astype(np.float32).reshape(-1, 1)
        severity = z[:, self.raw_names.index('global_z')].astype(np.float32)
        flat = np.concatenate((static, pr), axis=1)
        if use_history:
            flat = np.concatenate((flat, ev.reshape(len(ev), -1), m.astype(np.float32)), axis=1)
        assert np.isfinite(static).all() and np.isfinite(flat).all()
        return dict(static=static, prs=pr, event_numeric=ev, event_mask=m,
                    event_domain=np.tile(np.arange(4), (len(ev), 4)), severity=severity,
                    tabular=flat, names=names)

    def record(self):
        return dict(raw_names=self.raw_names, median=self.median.tolist(), scale=self.scale.tolist(),
                    prs_mean=self.prs_mean, prs_scale=self.prs_scale,
                    event_value_mean=self.event_value_mean, event_value_scale=self.event_value_scale,
                    event_other_mean=self.event_other_mean, event_other_scale=self.event_other_scale,
                    current_phc_units='raw PHC scores; legacy _z names are aliases; no clipping')


class ReverseKM:
    def fit(self, times, event):
        t = np.asarray(times, float); c = np.asarray(event) == 0
        self.times, inv, counts = np.unique(t, return_inverse=True, return_counts=True)
        risk = np.cumsum(counts[::-1])[::-1]
        d = np.bincount(inv, weights=c.astype(float), minlength=len(counts))
        self.after = np.cumprod(1 - d / risk)
        return self

    def before(self, t):
        ix = np.searchsorted(self.times, np.asarray(t), side='left') - 1
        return np.where(ix >= 0, self.after[np.maximum(ix, 0)], 1.)


def strata(meta):
    band = np.where(meta.age.to_numpy() < 70, '<70', np.where(meta.age.to_numpy() < 80, '70-79', '80-84'))
    return np.asarray([f'{s}|{b}' for s, b in zip(meta.study, band)])


class Censoring:
    def fit(self, meta):
        self.curves = {}; self.fallback = {}; lab = strata(meta)
        self.global_curve = ReverseKM().fit(meta.end_time, meta.event)
        for s in STUDIES:
            ix = meta.study.eq(s).to_numpy()
            self.curves[s] = ReverseKM().fit(meta.end_time[ix], meta.event[ix]) if ix.any() else self.global_curve
            for band in ('<70', '70-79', '80-84'):
                key = f'{s}|{band}'; subset = lab == key
                if subset.sum() >= 50:
                    self.curves[key] = ReverseKM().fit(meta.end_time[subset], meta.event[subset])
                else:
                    self.curves[key] = self.curves[s]
                self.fallback[key] = bool(subset.sum() < 50)
        return self

    def label(self, meta, horizon):
        t = meta.end_time.to_numpy(float); e = meta.event.to_numpy(int)
        event_by = (e != 0) & (t <= horizon + EPS)
        known = event_by | (t >= horizon - EPS)
        y = ((e == 1) & (t <= horizon + EPS)).astype(int)
        evaluate_time = np.where(event_by, t, horizon)
        g = np.ones(len(meta)); lab = strata(meta)
        for key, curve in self.curves.items():
            ix = lab == key
            if ix.any():
                g[ix] = curve.before(evaluate_time[ix])
        raw_weight = 1 / np.maximum(g, 1e-12)
        w = np.where(known, np.minimum(raw_weight, 10.), 0.)
        return dict(y=y, known=known, weight=w, g=g, clipped=known & (raw_weight > 10.))


def shared_bootstrap(label, seed):
    ix = np.flatnonzero(label['known'])
    w = label['weight'][ix]
    return np.random.default_rng(seed).choice(ix, size=min(len(ix), 20000), replace=True, p=w/w.sum())
