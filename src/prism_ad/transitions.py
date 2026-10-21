"""Pure, deterministic observed-stage endpoint construction; no file access."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

STATES = {1: 'CN', 2: 'MCI', 3: 'AD'}
EPS = 1e-8


def numeric(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def exact_age(value):
    # Local DD: all ages >=90 are top-coded. Never replace90+ with exact90.
    x = numeric(value)
    return x if math.isfinite(x) and 0 < x < 90 else np.nan


def normalize_visits(raw):
    d = raw.copy()
    d['subject_id'] = d['SUBJID'].astype(str)
    d['visit'] = pd.to_numeric(d['PHC_Visit'], errors='coerce')
    d['diagnosis'] = pd.to_numeric(d['PHC_Diagnosis'], errors='coerce')
    d['age'] = d['PHC_Age_Cognition'].map(exact_age)
    d['topcoded'] = d['PHC_Age_Cognition'].astype(str).str.strip().eq('90+')
    d['n_domains'] = sum(pd.to_numeric(d[c], errors='coerce').notna().astype(int)
                         for c in ('PHC_MEM', 'PHC_EXF', 'PHC_LAN', 'PHC_VSP'))
    # PHC harmonized scores are continuous: do not relabel a valid -1 as missing.
    return d.sort_values(['subject_id', 'visit'], kind='mergesort')


def endpoint_from_entry(group, entry_visit, state, task, subject_id, fold):
    """Use visit order first; never jump over an undated first event."""
    g = group.sort_values('visit', kind='mergesort')
    entry = g[g.visit.eq(entry_visit)]
    out = dict(subject_id=str(subject_id), fold=int(fold), task=task,
               entry_state=STATES[state], entry_visit=float(entry_visit),
               eligible=False, timing_status='unassessed', event_type='unassessed',
               entry_age=np.nan, end_time=np.nan, event_time=np.nan,
               interval_lower=np.nan, interval_width=np.nan,
               last_classified_followup=np.nan, n_later_classified=0,
               entry_n_domains=0, entry_study='unknown', next_state='none',
               next_gap=np.nan, event_next_state='none', event_next_gap=np.nan,
               any_later_cn_before_ad=False, ad_after_entry=False,
               mci_before_ad=False, first_mci_visit=np.nan,
               unknown_diagnosis_after_entry=0, undated_after_entry=0,
               later_topcoded=0, event_interval_unknown_diagnoses=0,
               event_interval_zero_width=False, prior_disease=False,
               any_age_decrease=False, zero_time_next_visit=False)
    if len(entry) != 1 or g.visit.isna().any() or g.visit.duplicated().any():
        out['timing_status'] = 'duplicate_or_missing_visit_key'
        return out
    e = entry.iloc[0]
    out.update(entry_age=float(e.age), entry_n_domains=int(e.n_domains),
               entry_study=str(e.Study))
    if numeric(e.diagnosis) != state:
        out['timing_status'] = 'index_diagnosis_mismatch'
        return out
    prior = g[g.visit.lt(entry_visit)]
    out['prior_disease'] = bool(prior.diagnosis.isin([2, 3] if state == 1 else [3]).any())
    later = g[g.visit.gt(entry_visit)]
    classified = later[later.diagnosis.isin(STATES)].copy()
    out.update(n_later_classified=len(classified),
               unknown_diagnosis_after_entry=int((~later.diagnosis.isin(STATES)).sum()),
               undated_after_entry=int(later.age.isna().sum()),
               later_topcoded=int(later.topcoded.sum()))
    if len(classified):
        out['next_state'] = STATES[int(classified.iloc[0].diagnosis)]
        out['next_gap'] = float(classified.iloc[0].age - e.age)
        out['zero_time_next_visit'] = bool(np.isfinite(out['next_gap']) and abs(out['next_gap']) <= EPS)
    ads = classified[classified.diagnosis.eq(3)]
    out['ad_after_entry'] = bool(len(ads))
    pre_ad = classified if ads.empty else classified[classified.visit.lt(ads.iloc[0].visit)]
    mcis = pre_ad[pre_ad.diagnosis.eq(2)]
    out['mci_before_ad'] = bool(len(mcis))
    if len(mcis):
        out['first_mci_visit'] = float(mcis.iloc[0].visit)
    if state == 2:
        out['any_later_cn_before_ad'] = bool(pre_ad.diagnosis.eq(1).any())
    targets = classified[classified.diagnosis.isin([2, 3] if state == 1 else [3])]
    event = targets.iloc[0] if len(targets) else None
    out['event_type'] = ('MCI' if int(event.diagnosis) == 2 else 'AD') if event is not None else 'censored'
    if event is not None and int(event.diagnosis) == 2:
        after = classified[classified.visit.gt(event.visit)]
        if len(after):
            out['event_next_state'] = STATES[int(after.iloc[0].diagnosis)]
            out['event_next_gap'] = float(after.iloc[0].age - event.age)
        after_pre_ad = after if ads.empty else after[after.visit.lt(ads.iloc[0].visit)]
        out['any_later_cn_before_ad'] = bool(after_pre_ad.diagnosis.eq(1).any())
    if out['prior_disease']:
        out['timing_status'] = 'prior_mci_or_ad' if state == 1 else 'prior_ad'
        return out
    if not np.isfinite(e.age):
        out['timing_status'] = 'index_age_unknown_or_topcoded'
        return out
    dated_all = g[g.visit.ge(entry_visit) & g.age.notna()]
    out['any_age_decrease'] = bool((np.diff(dated_all.age.to_numpy(float)) < -EPS).any())
    if out['any_age_decrease']:
        out['timing_status'] = 'decreasing_age_sequence'
        return out
    dated_classified = classified[classified.age.notna()]
    if len(dated_classified):
        out['last_classified_followup'] = float(dated_classified.age.max() - e.age)
    if event is not None:
        if not np.isfinite(event.age):
            out['timing_status'] = 'first_event_age_unknown_or_topcoded'
            return out
        dt = float(event.age - e.age)
        if dt <= EPS:
            out['timing_status'] = 'first_event_nonpositive_elapsed_time'
            return out
        preceding = g[g.visit.ge(entry_visit) & g.visit.lt(event.visit)
                      & g.diagnosis.isin([1] if state == 1 else [1, 2]) & g.age.notna()]
        last_negative = preceding.iloc[-1]
        lower = float(last_negative.age - e.age)
        out.update(event_time=dt, end_time=dt, interval_lower=lower,
                   interval_width=dt-lower, event_interval_zero_width=bool(abs(dt-lower) <= EPS),
                   event_interval_unknown_diagnoses=int((g.visit.gt(last_negative.visit)
                       & g.visit.lt(event.visit) & ~g.diagnosis.isin(STATES)).sum()),
                   eligible=True, timing_status='event_with_recorded_time')
    else:
        dt = out['last_classified_followup']
        if not np.isfinite(dt) or dt <= EPS:
            out['timing_status'] = 'no_positive_dated_diagnosis_followup'
        else:
            out.update(end_time=dt, eligible=True, timing_status='right_censored')
    return out


def build_endpoints(visits, baseline):
    groups = {str(s): g for s, g in visits.groupby('subject_id', sort=False)}
    rows = []
    for b in baseline.itertuples(index=False):
        state, index = numeric(b.baseline_diagnosis_code), numeric(b.index_visit)
        if state not in (1, 2):
            continue
        g = groups.get(str(b.subject_id))
        if g is None:
            g = visits.iloc[:0]
        task = 'CN_to_MCI' if state == 1 else 'baseline_MCI_to_AD'
        rec = endpoint_from_entry(g, index, int(state), task, b.subject_id, b.fold)
        rec['legacy_index_age'] = numeric(b.index_age)
        rec['legacy_index_age_difference'] = rec['entry_age'] - rec['legacy_index_age']
        rec['has_prs'] = np.isfinite(numeric(b.PRS))
        rows.append(rec)
        # A newly recorded MCI is the entry for the second task, not information
        # available at the preceding CN baseline. Keep incident/prevalent separate.
        if state == 1 and not rec['prior_disease'] and rec['timing_status'] not in (
                'duplicate_or_missing_visit_key', 'index_diagnosis_mismatch') \
                and rec['event_type'] == 'MCI' and np.isfinite(rec['first_mci_visit']):
            inc = endpoint_from_entry(g, rec['first_mci_visit'], 2,
                                      'incident_MCI_to_AD', b.subject_id, b.fold)
            inc['legacy_index_age'] = np.nan
            inc['legacy_index_age_difference'] = np.nan
            inc['has_prs'] = rec['has_prs']
            rows.append(inc)
    result = pd.DataFrame(rows)
    if result.duplicated(['task', 'subject_id']).any():
        raise AssertionError('More than one entry per subject/task')
    if not result.fold.isin(range(1, 10)).all():
        raise AssertionError('Reserved or unknown fold in endpoint rows')
    return result


def quantiles(values):
    x = pd.to_numeric(pd.Series(values), errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return dict(n=len(x), mean=float(x.mean()) if len(x) else np.nan,
                q25=float(x.quantile(.25)) if len(x) else np.nan,
                median=float(x.median()) if len(x) else np.nan,
                q75=float(x.quantile(.75)) if len(x) else np.nan,
                q90=float(x.quantile(.90)) if len(x) else np.nan)


def aggregate(endpoints):
    summary, horizons, intervals, reversions, quality, resolution, features = [], [], [], [], [], [], []
    for task, all_rows in endpoints.groupby('task', sort=True):
        for scope in ['development_all'] + [f'fold_{k:02d}' for k in range(1, 10)]:
            q = all_rows if scope == 'development_all' else all_rows[all_rows.fold.eq(int(scope[-2:]))]
            valid = q[q.eligible]
            target = 'MCI' if task == 'CN_to_MCI' else 'AD'
            events = valid[valid.event_type.eq(target)]
            ctx = dict(task=task, scope=scope)
            summary.append(dict(**ctx, n_entries=len(q), n_time_eligible=len(valid),
                n_target_events=len(events), n_direct_ad_competing=int(valid.event_type.eq('AD').sum()) if target == 'MCI' else 0,
                n_right_censored=int(valid.event_type.eq('censored').sum()),
                n_excluded_timing_or_entry=len(q)-len(valid), n_ever_target_by_visit=int(q.event_type.eq(target).sum()),
                median_recorded_followup=float(valid.last_classified_followup.median()),
                median_observed_time_to_target=float(events.event_time.median()),
                n_baseline_prs=int(valid.has_prs.sum()), n_entry_2plus_domains=int(valid.entry_n_domains.ge(2).sum())))
            for reason, count in q.timing_status.value_counts().items():
                quality.append(dict(**ctx, timing_status=reason, n=int(count)))
            for h in (1., 2., 3., 5.):
                ev = valid.event_type.eq(target) & valid.event_time.le(h+EPS)
                comp = valid.event_type.eq('AD') & valid.event_time.le(h+EPS) if target == 'MCI' else pd.Series(False,index=valid.index)
                free = valid.end_time.ge(h-EPS) & ~ev & ~comp
                unknown = ~ev & ~comp & ~free
                horizons.append(dict(**ctx, horizon_years=h, n_time_eligible=len(valid),
                    target_by_horizon=int(ev.sum()), competing_by_horizon=int(comp.sum()),
                    event_free_observed_through_horizon=int(free.sum()), censored_before_horizon=int(unknown.sum()),
                    event_interval_straddles_horizon=int((valid.event_time.gt(h+EPS) & valid.interval_lower.lt(h-EPS)).sum())))
            if scope != 'development_all':
                continue
            for field in ('last_classified_followup','end_time','event_time','interval_width','next_gap'):
                s = events[field] if field in ('event_time','interval_width') else valid[field]
                intervals.append(dict(task=task, quantity=field, **quantiles(s)))
            w = events.interval_width
            for upper in (.5, 1., 1.5, 2., 3., 5.):
                resolution.append(dict(task=task, quantity='target_event_bracket_width', threshold_years=upper,
                    n=len(w), n_positive_width_le_threshold=int((w.gt(EPS) & w.le(upper+EPS)).sum()),
                    n_zero_width=int(w.abs().le(EPS).sum())))
            for field in ('unknown_diagnosis_after_entry','undated_after_entry','later_topcoded',
                          'event_interval_unknown_diagnoses'):
                quality.append(dict(task=task, scope='development_all', timing_status='flag_'+field,
                                    n=int(q[field].gt(0).sum())))
            if target == 'MCI':
                for state, count in events.event_next_state.value_counts().items():
                    reversions.append(dict(task=task, measure='first_MCI_next_classified_state', category=state,
                                           n=int(count), denominator=len(events)))
                reversions.append(dict(task=task, measure='any_later_CN_before_first_AD', category='yes',
                                       n=int(events.any_later_cn_before_ad.sum()), denominator=len(events)))
            else:
                reversions.append(dict(task=task, measure='any_later_CN_before_first_AD', category='yes',
                                       n=int(valid.any_later_cn_before_ad.sum()), denominator=len(valid)))
            for source, s in valid.groupby('entry_study', sort=True):
                features.append(dict(task=task, study=source, n_time_eligible=len(s),
                    target_events=int(s.event_type.eq(target).sum()), n_prs=int(s.has_prs.sum()),
                    n_2plus_cognitive_domains=int(s.entry_n_domains.ge(2).sum())))
    return {name:pd.DataFrame(rows) for name,rows in (
        ('cohort_summary',summary),('horizon_observability',horizons),('timing_quantiles',intervals),
        ('reversion_confirmation',reversions),('quality_counts',quality),('time_resolution',resolution),
        ('study_feature_support',features))}
