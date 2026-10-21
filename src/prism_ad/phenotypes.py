"""Clinical index, fixed genetic strata and participant-level HC3 comparisons."""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
PCS=[f"PC{i}" for i in range(1,6)]

def exact_age(series):
    x = pd.to_numeric(series, errors='coerce')
    return x.where((x > 0) & (x < 90))

def construct_index(raw, roster):
    """One last classified visit per scored subject; MCI never becomes CN."""
    c = raw[raw.SUBJID.isin(roster.subject_id)].copy()
    c = c.rename(columns={'SUBJID': 'subject_id', 'PHC_Visit': 'visit',
                          'PHC_Diagnosis': 'diagnosis'})
    c['visit'] = pd.to_numeric(c.visit, errors='coerce')
    c['diagnosis'] = pd.to_numeric(c.diagnosis, errors='coerce')
    if c.visit.isna().any() or c.duplicated(['subject_id', 'visit']).any():
        raise ValueError('Missing/duplicate cognition visit keys; explicit resolution required')
    if (c.groupby('subject_id').Study.nunique() > 1).any():
        raise ValueError('Multiple cognition studies per subject')
    c['age'] = exact_age(c.PHC_Age_Cognition)
    c['age_topcoded'] = c.PHC_Age_Cognition.astype(str).str.strip().eq('90+')
    c['sex'] = pd.to_numeric(c.PHC_Sex, errors='coerce').where(
        pd.to_numeric(c.PHC_Sex, errors='coerce').isin([1, 2]))
    c = c.sort_values(['subject_id', 'visit'], kind='stable')
    v = c[c.diagnosis.isin([1, 2, 3])].copy()
    any_ad = v.diagnosis.eq(3).groupby(v.subject_id).any()
    any_mci = v.diagnosis.eq(2).groupby(v.subject_id).any()
    all_cn = v.diagnosis.eq(1).groupby(v.subject_id).all()
    span = v.groupby('subject_id').age.agg(lambda x: x.max() - x.min())
    n_visits = v.groupby('subject_id').size()
    last = v.groupby('subject_id', sort=False).tail(1).copy()
    last['prior_ad_reversion'] = last.subject_id.map(any_ad) & last.diagnosis.ne(3)
    last['any_mci'] = last.subject_id.map(any_mci)
    last['classified_visits'] = last.subject_id.map(n_visits)
    last['observed_classified_span'] = last.subject_id.map(span)
    last['persistent_cn_3y'] = (last.subject_id.map(all_cn)
                                 & last.observed_classified_span.ge(3)
                                 & last.classified_visits.ge(2))
    last['status'] = last.diagnosis.map({1: 'CN', 2: 'MCI', 3: 'AD'})
    last.loc[last.prior_ad_reversion, 'status'] = 'excluded_prior_AD_reversion'
    keep = ['subject_id', 'Study', 'visit', 'age', 'age_topcoded', 'sex',
            'status', 'diagnosis', 'PHC_Education', 'prior_ad_reversion', 'any_mci',
            'classified_visits', 'observed_classified_span', 'persistent_cn_3y']
    keep += [f'PHC_{s}' for s in ['MEM', 'EXF', 'LAN', 'VSP']]
    d = roster.merge(last[keep], on='subject_id', how='left', validate='one_to_one')
    d['status'] = d.status.fillna('no_classified_PHC_v18_visit')
    d['education'] = pd.to_numeric(d.PHC_Education, errors='coerce')
    d['education'] = d.education.where(d.education.ge(0))
    d['cognition_lag'] = 0.0
    return d.sort_values('subject_id', kind='stable').reset_index(drop=True)

def grpa_bounds(prs, diagnosis, fraction):
    prs = np.asarray(prs, dtype=float)
    diagnosis = np.asarray(diagnosis)
    keep = np.isfinite(prs) & np.isin(diagnosis, [1, 3])
    p, y = prs[keep], diagnosis[keep]
    if not (np.any(y == 1) and np.any(y == 3)):
        raise ValueError('CN and AD both required')
    s = np.sort(p)
    bl = float(np.quantile(p[y == 3], .025, method='linear'))
    tu = float(np.quantile(p[y == 1], .975, method='linear'))
    offset = round(len(s) * fraction)
    ib = int(np.argmin(np.abs(s - bl)))
    it = int(np.argmin(np.abs(s - tu)))
    if ib + offset >= len(s) or it - offset < 0:
        raise ValueError('Original R offset outside pooled array; no clamping permitted')
    return dict(fraction=fraction, n_reference=len(s), offset=offset,
                B_lower=bl, B_upper=float(s[ib + offset]),
                T_lower=float(s[it - offset]), T_upper=tu,
                lower_nearest_index_zero_based=ib, upper_nearest_index_zero_based=it)

def bh_with_failures(table, p_col='p', out_col='q'):
    """Keep the full prespecified family, including failed estimands (p=1)."""
    values = pd.to_numeric(table[p_col], errors='coerce')
    table[out_col] = multipletests(values.fillna(1).clip(0, 1), method='fdr_bh')[1]
    table.loc[values.isna(), out_col] = np.nan
    return table

def adjusted_effect(data, feature, lag, extra_cat=(), interaction=False):
    required = [feature, 'age', 'sex', 'Study', 'PRS'] + PCS + [lag, 'g'] + list(extra_cat)
    if interaction:
        required += ['high']
    d = data.dropna(subset=list(dict.fromkeys(required))).copy()
    result = dict(n_candidate=len(data), n_complete_before_study_support=len(d),
                  status='insufficient_support', n0=0, n1=0, n=0, p=np.nan,
                  beta=np.nan, ci_low=np.nan, ci_high=np.nan, beta_sd=np.nan)
    if d.empty:
        return result
    d['_cell'] = (d.g.astype(int) + 2*d.high.astype(int)) if interaction else d.g.astype(int)
    expected = list(range(4 if interaction else 2))
    counts = pd.crosstab(d.Study, d._cell).reindex(columns=expected, fill_value=0)
    common = counts.index[(counts >= 5).all(axis=1)]
    d = d[d.Study.isin(common)].copy()
    n0, n1 = int(d.g.eq(0).sum()), int(d.g.eq(1).sum())
    result.update(n0=n0, n1=n1, n=len(d), n_studies=len(common),
                  studies='|'.join(sorted(map(str, common))))
    if min(n0, n1) < 20 or (interaction and d._cell.value_counts().reindex(expected, fill_value=0).min() < 20):
        return result
    x = pd.DataFrame({'Intercept': np.ones(len(d)), 'g': d.g.to_numpy(float)}, index=d.index)
    target = 'g'
    if interaction:
        x['high'] = d.high.astype(float)
        x['g_x_high'] = d.g * d.high
        target = 'g_x_high'
    a = d.age - 75
    numeric = {'age_centered': a, 'age_squared': a*a, 'PRS': d.PRS,
               **{c: d[c] for c in PCS}, 'measurement_lag': d[lag]}
    for name, values in numeric.items():
        sd = values.std()
        if sd > 1e-15:
            x[name] = (values - values.mean()) / sd
    for c in ['sex', 'Study'] + list(extra_cat):
        levels = pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float)
        x = pd.concat([x, levels], axis=1)
    # Keep a full-rank basis in fixed order, with all focal effects retained.
    kept, basis, dropped = [], [], []
    for name in x:
        v = x[name].to_numpy(float).copy()
        original_norm = np.linalg.norm(v)
        for q in basis:
            v -= q * np.dot(q, v)
        # Reorthogonalize for near-collinear study/scanner dummy sets.
        for q in basis:
            v -= q * np.dot(q, v)
        norm = np.linalg.norm(v)
        if norm > 1e-9 * max(original_norm, 1):
            kept.append(name)
            basis.append(v/norm)
        else:
            dropped.append(name)
    if target not in kept or len(d) - len(kept) < 20:
        result['status'] = 'non_estimable_design'
        return result
    x = x[kept]
    ordinary = sm.OLS(d[feature].astype(float), x).fit()
    leverage = ordinary.get_influence().hat_matrix_diag
    result.update(n_parameters=len(kept), max_leverage=float(max(leverage)),
                  dropped_collinear='|'.join(dropped))
    if max(leverage) >= .99:
        result['status'] = 'excessive_leverage'
        return result
    fit = ordinary.get_robustcov_results(cov_type='HC3', use_t=True)
    k = kept.index(target)
    ci = fit.conf_int()[k]
    beta, sd = float(fit.params[k]), float(d[feature].std())
    result.update(status='ok', beta=beta, ci_low=float(ci[0]), ci_high=float(ci[1]),
                  p=float(fit.pvalues[k]), beta_sd=beta/sd if sd > 0 else np.nan,
                  ci_low_sd=float(ci[0])/sd if sd > 0 else np.nan,
                  ci_high_sd=float(ci[1])/sd if sd > 0 else np.nan,
                  feature_sd_complete=sd, residual_df=float(fit.df_resid))
    return result
