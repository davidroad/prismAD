"""Fold-stratified rank comparisons on private marginal component values."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import ndtr

PAIRS = (("CN", "MCI"), ("CN", "AD"), ("MCI", "AD"))
COMPONENTS = ("C", "G")

def _stratum(a, b):
    """U counts B>A, with one half per tie; variance is conditional on ranks."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n_a, n_b = len(a), len(b); n = n_a + n_b
    if n_a == 0 or n_b == 0:
        raise ValueError("Both groups are required")
    all_values = np.concatenate([a, b])
    if not np.isfinite(all_values).all():
        raise ValueError("Finite component values required")
    ranks = rankdata(all_values, method="average")
    u = float(ranks[n_a:].sum() - n_b * (n_b + 1) / 2.0)
    _, ties = np.unique(all_values, return_counts=True)
    correction = float(np.sum(ties.astype(float)**3 - ties))
    variance = n_a * n_b / 12.0 * (n + 1 - correction / (n * (n - 1)))
    return {"n_a": n_a, "n_b": n_b, "u_b": u,
            "mean_u": n_a * n_b / 2.0, "variance_u": max(0.0, variance),
            "stratum_weight": 1.0 / (n + 1.0), "tied_observations": int(ties[ties > 1].sum())}

def stratified_rank_test(frame, component, group_a, group_b, minimum=5):
    """Van Elteren normal approximation with inverse-(N+1) stratum weights.

    All ranks are computed within fitted fold. Positive effects indicate that
    the second group tends to have higher values. Effect sizes describe
    marginal distributions, not covariate-adjusted or causal differences.
    """
    d = frame.loc[frame.component.eq(component) & frame.clinical_group.isin([group_a, group_b])].copy()
    if set(d.fold.astype(int)) != {1, 2, 3}:
        raise ValueError("All three fitted folds must be present")
    cells = []; score = variance = numer = denom = 0.0; supported = True
    for fold in (1, 2, 3):
        z = d.loc[d.fold.astype(int).eq(fold)]
        a = z.loc[z.clinical_group.eq(group_a), "value"].to_numpy(float)
        b = z.loc[z.clinical_group.eq(group_b), "value"].to_numpy(float)
        cell = _stratum(a, b); cell["fold"] = fold; cells.append(cell)
        supported &= min(len(a), len(b)) >= minimum
        weight = cell["stratum_weight"]
        score += weight * (cell["u_b"] - cell["mean_u"])
        variance += weight**2 * cell["variance_u"]
        numer += weight * cell["u_b"]; denom += weight * len(a) * len(b)
    estimable = bool(supported and variance > 0 and denom > 0)
    statistic = float(score / np.sqrt(variance)) if estimable else np.nan
    p = float(2.0 * ndtr(-abs(statistic))) if estimable else 1.0
    ps = float(numer / denom) if estimable else np.nan
    return {"test_id": "marginal_" + component + "_" + group_a + "_" + group_b,
            "component": component, "group_a": group_a, "group_b": group_b,
            "n_a": sum(c["n_a"] for c in cells), "n_b": sum(c["n_b"] for c in cells),
            "z": statistic, "p": p, "probability_superiority_b_over_a": ps,
            "rank_biserial_b_minus_a": float(2 * ps - 1) if estimable else np.nan,
            "estimable": estimable, "strata": cells}

def marginal_tests(points):
    """Return all six prespecified pairwise comparisons without selection."""
    records = []; details = []
    for component in COMPONENTS:
        for group_a, group_b in PAIRS:
            rec = stratified_rank_test(points, component, group_a, group_b)
            details.append(rec)
            records.append({k: v for k, v in rec.items() if k != "strata"})
    return pd.DataFrame(records), details

def bh_ten_family(calibration, marginal):
    """One complete four-plus-six post hoc family; unavailable slots retain p=1."""
    c = calibration.copy(); m = marginal.copy()
    if len(c) != 4 or len(m) != 6:
        raise ValueError("The planned family requires all four and six test slots")
    if "test_id" not in c:
        c["test_id"] = ["calibration_" + str(r.task) + "_" + str(int(r.horizon)) for r in c.itertuples()]
    all_rows = pd.concat([c, m], ignore_index=True, sort=False)
    if all_rows.test_id.duplicated().any():
        raise ValueError("Test identifiers must be unique")
    p = pd.to_numeric(all_rows.p, errors="raise").fillna(1.0).to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p values must be probabilities")
    order = np.argsort(p, kind="mergesort")
    sorted_q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty(len(p)); q[order] = np.minimum(1.0, sorted_q)
    all_rows["q"] = q
    all_rows["label"] = ["***" if x < .001 else "**" if x < .01 else "*" if x < .05
                         else "(*)" if y < .05 else "n.s." for x, y in zip(q, p)]
    return all_rows.iloc[:4].copy(), all_rows.iloc[4:].copy(), all_rows

def points_from_profiles(profiles):
    """Export unlinked component-value rows to authorized private storage only."""
    d = profiles.copy()
    required = {"display_alias", "fold", "clinical_group", "cognition_main", "PRS"}
    if not required.issubset(d.columns) or not d.display_alias.is_unique:
        raise ValueError("Unique frozen participant profiles are required")
    if not d.clinical_group.isin(["CN", "MCI", "AD"]).all():
        raise ValueError("Unexpected clinical course")
    if not d.fold.isin([1, 2, 3]).all():
        raise ValueError("Unexpected validation fold")
    pieces = []
    for component, column in [("C", "cognition_main"), ("G", "PRS")]:
        values = pd.to_numeric(d[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Finite signed component values required")
        out = d[["clinical_group", "fold"]].copy()
        out["component"] = component; out["value"] = values
        pieces.append(out[["component", "clinical_group", "fold", "value"]])
    # No alias or shared individual-row identifier is exported. Sort each
    # component independently; SVG point order cannot encode the C/G pairing.
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["component", "clinical_group", "fold", "value"], kind="mergesort").reset_index(drop=True)
