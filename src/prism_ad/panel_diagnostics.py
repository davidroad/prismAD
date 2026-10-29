"""Post hoc calibration diagnostics for saved, fixed held-out predictions.

No prognostic-model fitting or recalibration is performed. All participant arrays
and empirical summaries must be written to a caller-supplied private directory.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

TASKS = ("CN_MCI", "MCI_AD")
HORIZONS = (3, 5)
FOLDS = (1, 2, 3)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_directory(path):
    path = Path(path).expanduser().resolve()
    code_root = Path(__file__).resolve().parents[2]
    if path == code_root or code_root in path.parents:
        raise ValueError("Empirical diagnostic outputs must remain outside Code")
    for parent in (path,) + tuple(path.parents):
        if parent.name in ("PRISM-AD_Public", "PRISM-AD_Local") or (parent / ".git").exists():
            raise ValueError("Empirical diagnostic outputs must not enter release code or Git")
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _write_private_csv(frame, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite a frozen diagnostic output")
    frame.to_csv(path, index=False, float_format="%.17g")
    path.chmod(0o600)


def calibration_slope(y, probability, ipcw, clip=1e-8, max_iter=100, tolerance=1e-10):
    """Fit a free-intercept weighted logistic calibration diagnostic.

    The returned slope CI and H0:slope=1 P value use a participant empirical-score
    sandwich covariance with HC1 n/(n-2). Predictions and IPCW are treated as fixed.
    This diagnostic does not alter the supplied probabilities.
    """
    y, p, w = [np.asarray(v, dtype=float) for v in (y, probability, ipcw)]
    if any(v.ndim != 1 for v in (y, p, w)) or not (len(y) == len(p) == len(w)):
        raise ValueError("Diagnostic arrays must be aligned one-dimensional arrays")
    if not all(np.isfinite(v).all() for v in (y, p, w)):
        raise ValueError("Diagnostic arrays must be finite")
    if not np.isin(y, [0., 1.]).all() or ((p < 0) | (p > 1)).any() or (w <= 0).any():
        raise ValueError("Require binary outcomes, probabilities in [0,1] and positive weights")
    if not 0 < clip < .5:
        raise ValueError("Invalid logit clipping bound")
    n = len(y)
    base = dict(n=int(n), n_events=int(y.sum()), n_nonevents=int(n-y.sum()),
                sum_weights=float(w.sum()),
                effective_sample_size=float(w.sum()**2 / np.dot(w, w)),
                clip=float(clip), clip_count=int(np.count_nonzero(np.clip(p, clip, 1-clip) != p)),
                test_null="slope=1", covariance="empirical participant-score sandwich HC1 n/(n-2)",
                fixed_predictions_and_weights=True)
    if n <= 2 or len(np.unique(y)) < 2:
        return dict(base, status="nonestimable_outcome_support", converged=False, p=1.)
    bounded = np.clip(p, clip, 1-clip)
    z = np.log(bounded)-np.log1p(-bounded)
    X = np.column_stack([np.ones(n), z])
    if np.linalg.matrix_rank(X) != 2:
        return dict(base, status="nonestimable_design", converged=False, p=1.)
    # A common normalization conditions the Hessian without changing either the
    # M-estimator or its empirical-score sandwich covariance.
    wn = w / w.mean()
    theta = np.array([0., 1.])
    converged, line_search_failed = False, False
    iterations = 0

    def objective(beta):
        eta = X @ beta
        return float(np.sum(wn * (np.logaddexp(0, eta)-y*eta)))

    for iteration in range(1, max_iter+1):
        iterations = iteration
        mu = expit(X @ theta)
        score = X.T @ (wn*(y-mu))
        bread = X.T @ ((wn*mu*(1-mu))[:, None]*X)
        try:
            step = np.linalg.solve(bread, score)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(step).all():
            break
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break
        old = objective(theta)
        rate = 1.
        while rate >= 2**-30 and objective(theta+rate*step) > old+1e-12*max(1., abs(old)):
            rate *= .5
        if rate < 2**-30:
            line_search_failed = True
            break
        theta = theta+rate*step
        if np.max(np.abs(rate*step)) < tolerance:
            converged = True
            break
    mu = expit(X @ theta)
    score = X.T @ (wn*(y-mu))
    bread = X.T @ ((wn*mu*(1-mu))[:, None]*X)
    diagnostics = dict(iterations=int(iterations), converged=bool(converged),
                       line_search_failed=bool(line_search_failed),
                       score_max_abs=float(np.max(np.abs(score))),
                       hessian_condition=float(np.linalg.cond(bread)),
                       negative_log_likelihood=objective(theta),
                       intercept=float(theta[0]), slope=float(theta[1]))
    if not converged:
        return dict(base, **diagnostics, status="nonestimable_nonconvergence", p=1.)
    try:
        inverse = np.linalg.inv(bread)
        case_score = (wn*(y-mu))[:, None]*X
        covariance = (n/(n-2.)) * inverse @ (case_score.T @ case_score) @ inverse
        variance = np.diag(covariance)
        if (variance <= 0).any() or not np.isfinite(covariance).all():
            raise np.linalg.LinAlgError("Invalid covariance")
        se = np.sqrt(variance)
        leverage = wn*mu*(1-mu)*np.einsum("ij,jk,ik->i", X, inverse, X)
        statistic = float((theta[1]-1.)/se[1])
        q = float(norm.ppf(.975))
        return dict(base, **diagnostics, status="ok", intercept_se=float(se[0]),
                    intercept_ci_low=float(theta[0]-q*se[0]), intercept_ci_high=float(theta[0]+q*se[0]),
                    slope_se=float(se[1]), ci_low=float(theta[1]-q*se[1]), ci_high=float(theta[1]+q*se[1]),
                    z=statistic, p=float(2*norm.sf(abs(statistic))),
                    max_leverage=float(leverage.max()), min_leverage=float(leverage.min()),
                    leverage_sum=float(leverage.sum()), covariance_00=float(covariance[0,0]),
                    covariance_01=float(covariance[0,1]), covariance_11=float(covariance[1,1]))
    except np.linalg.LinAlgError:
        return dict(base, **diagnostics, status="nonestimable_covariance", p=1.)


def _boolean(series, name):
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    if pd.api.types.is_numeric_dtype(series):
        if not series.isin([0, 1]).all():
            raise ValueError("Invalid " + name + " indicator")
        return series.to_numpy(bool)
    lowered = series.astype(str).str.lower()
    if not lowered.isin(["true", "false", "0", "1"]).all():
        raise ValueError("Invalid " + name + " indicator")
    return lowered.isin(["true", "1"]).to_numpy(bool)


def collect_calibration_arrays(prediction_directory, task, horizon):
    """Read one saved task/horizon, validate alignment, and omit all identifiers."""
    root = Path(prediction_directory).expanduser().resolve() / "LOCAL_RESTRICTED"
    selected, ids, sources = [], set(), []
    for fold in FOLDS:
        directory = root / task / ("fold%d_h%d" % (fold, horizon))
        validation = directory / "validation.csv"
        probability = directory / "PRISM_current" / "probability.npy"
        for path in (validation, probability):
            sources.append(dict(path=str(path), sha256=_sha256(path)))
        frame = pd.read_csv(validation)
        p = np.load(probability, allow_pickle=False)
        if p.ndim == 2 and p.shape[1] == 1:
            p = p[:, 0]
        if p.ndim != 1 or len(p) != len(frame):
            raise ValueError("Saved probabilities do not align with validation rows")
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
            raise ValueError("Invalid saved probabilities")
        if not frame["task"].eq(task).all() or not frame["fold"].eq(fold).all():
            raise ValueError("Validation task/fold metadata mismatch")
        first = _boolean(frame["first_eligible"], "first_eligible")
        known = _boolean(frame["known"], "known")
        weight = pd.to_numeric(frame["ipcw"], errors="raise").to_numpy(float)
        keep = first & known & (weight>0)
        q = frame.loc[keep]
        if q["subject_id"].isna().any() or not q["subject_id"].is_unique:
            raise ValueError("Selected participant identities must be unique")
        current_ids = set(q["subject_id"].astype(str))
        if ids.intersection(current_ids):
            raise ValueError("A participant occurs in multiple validation folds")
        ids.update(current_ids)
        selected.append(pd.DataFrame(dict(y=q["y"].to_numpy(float), prediction=p[keep],
                                           weight=weight[keep], fold=np.full(int(keep.sum()), fold, dtype=int))))
    arrays = pd.concat(selected, ignore_index=True)
    if len(arrays) != len(ids):
        raise ValueError("Participant identity accounting failed")
    return arrays, sources


def _pooled_bins(arrays, task, horizon):
    rows = []
    p, w, y = [arrays[k].to_numpy(float) for k in ("prediction", "weight", "y")]
    for k in range(10):
        lo, hi = k/10., (k+1)/10.
        ix = (p >= lo) & ((p < hi) if k < 9 else (p <= 1.))
        mass = float(w[ix].sum())
        rows.append(dict(task=task, horizon=int(horizon), risk_lo=lo, risk_hi=hi,
                         n_known=int(ix.sum()), weight_mass=mass,
                         prediction=float(np.average(p[ix], weights=w[ix])) if ix.any() else np.nan,
                         observed=float(np.average(y[ix], weights=w[ix])) if ix.any() else np.nan))
    return pd.DataFrame(rows)


def run_calibration_diagnostics(prediction_directory, output, pooled_bins_path=None,
                                expected_counts=None):
    """Write four raw-P diagnostics and no-ID arrays to a private directory.

    Multiplicity correction belongs to the caller's prespecified family. No
    q values, stars, changed probabilities or calibrated forecasts are produced.
    """
    output = _private_directory(output)
    restricted = _private_directory(output / "restricted_local")
    results, reconstructed, provenance = [], [], []
    for task in TASKS:
        for horizon in HORIZONS:
            arrays, sources = collect_calibration_arrays(prediction_directory, task, horizon)
            key = "%s_%d" % (task, horizon)
            if expected_counts is not None and len(arrays) != expected_counts[key]:
                raise ValueError("Prespecified known-outcome count mismatch: " + key)
            result = calibration_slope(arrays.y.to_numpy(), arrays.prediction.to_numpy(),
                                       arrays.weight.to_numpy())
            result.update(test_id="calibration_"+key, task=task, horizon=int(horizon))
            results.append(result)
            _write_private_csv(arrays, restricted / ("calibration_"+key+"_PRIVATE.csv"))
            reconstructed.append(_pooled_bins(arrays, task, horizon))
            provenance.extend(sources)
    direct = pd.concat(reconstructed, ignore_index=True)
    bin_check = dict(checked=pooled_bins_path is not None)
    if pooled_bins_path is not None:
        source = Path(pooled_bins_path).expanduser().resolve()
        frozen = pd.read_csv(source)
        keys = ["task", "horizon", "risk_lo", "risk_hi"]
        if frozen.duplicated(keys).any() or len(frozen) != len(direct):
            raise ValueError("Frozen calibration bins are not a unique complete set")
        a = frozen.sort_values(keys).reset_index(drop=True)
        b = direct.sort_values(keys).reset_index(drop=True)
        if not a[keys[:2]].equals(b[keys[:2]]):
            raise ValueError("Calibration bin task/horizon mismatch")
        for col in keys[2:]:
            if not np.allclose(a[col], b[col], rtol=0, atol=1e-14):
                raise ValueError("Calibration bin boundary mismatch")
        if not np.array_equal(a.n_known.to_numpy(), b.n_known.to_numpy()):
            raise ValueError("Participant-pooled calibration support differs from frozen bins")
        errors = {}
        for col in ("weight_mass", "prediction", "observed"):
            av, bv = a[col].to_numpy(float), b[col].to_numpy(float)
            if not np.allclose(av, bv, rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError("Participant-pooled calibration coordinates differ: " + col)
            finite = np.isfinite(av) & np.isfinite(bv)
            errors[col] = float(np.max(np.abs(av[finite]-bv[finite]))) if finite.any() else 0.
        bin_check.update(matched=True, bins=int(len(a)), displayed_bins=int((a.n_known>=5).sum()),
                         max_absolute_errors=errors, source=str(source), sha256=_sha256(source))
    for record in provenance:
        if _sha256(record["path"]) != record["sha256"]:
            raise ValueError("A saved prediction source changed while running diagnostics")
    _write_private_csv(pd.DataFrame(results), output / "calibration_slope_tests_raw.csv")
    _write_private_csv(direct, output / "calibration_bins_direct_PRIVATE.csv")
    qa = dict(calibration_model="logit E(Y) = free intercept + slope * logit(fixed prediction)",
              p_value_null="slope=1", covariance="empirical participant-score sandwich HC1 n/(n-2)",
              confidence_intervals="nominal normal Wald 95%",
              predictions_and_ipcw_fixed=True, no_recalibration=True, no_q_adjustment=True,
              participant_identifiers_exported=False, calibration_bin_check=bin_check,
              sources=provenance)
    qa_path = output / "CALIBRATION_QA.json"
    if qa_path.exists():
        raise FileExistsError("Refusing to overwrite a frozen diagnostic output")
    qa_path.write_text(json.dumps(qa, indent=2)+"\n")
    qa_path.chmod(0o600)
    return pd.DataFrame(results)
