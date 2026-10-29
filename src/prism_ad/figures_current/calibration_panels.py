"""Descriptive calibration panels from privately pooled first-visit bins.

No fitting or recalibration. Unsupported bins interrupt the curves.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .style import apply_figure_style

_REQUIRED = {"task", "horizon", "risk_lo", "risk_hi", "n_known", "weight_mass",
             "prediction", "observed"}
_TEAL = "#007F79"
_GREY = "#9AA4AD"
_HORIZONS = ((3, "o", "-"), (5, "s", "--"))


def _curve(frame):
    """Return gapped coordinates and support without modifying input."""
    z = frame.sort_values(["risk_lo", "risk_hi"]).copy()
    if z.duplicated(["risk_lo", "risk_hi"]).any():
        raise ValueError("Calibration-bin keys must be unique within horizon")
    for field in ["risk_lo", "risk_hi", "n_known", "weight_mass"]:
        z[field] = pd.to_numeric(z[field], errors="raise")
        if not np.isfinite(z[field].to_numpy(float)).all():
            raise ValueError("Calibration bounds, counts and weights must be finite")
    if ((z.risk_lo < 0) | (z.risk_hi > 1) | (z.risk_hi <= z.risk_lo)).any():
        raise ValueError("Calibration bins must be ordered probability intervals")
    if (z.n_known < 0).any() or not np.equal(z.n_known, np.floor(z.n_known)).all():
        raise ValueError("Known-outcome counts must be nonnegative integers")
    if (z.weight_mass < 0).any():
        raise ValueError("Calibration weight mass must be nonnegative")
    x, y, visible_n = [], [], []
    previous_hi = None
    for row in z.itertuples(index=False):
        if previous_hi is not None:
            if row.risk_lo < previous_hi - 1e-10:
                raise ValueError("Calibration bins must not overlap")
            if not np.isclose(row.risk_lo, previous_hi, atol=1e-10, rtol=0):
                x.append(np.nan); y.append(np.nan)
        previous_hi = float(row.risk_hi)
        if row.n_known < 5:
            x.append(np.nan); y.append(np.nan)
            continue
        values = np.asarray([row.prediction, row.observed], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Supported calibration-bin means must be probabilities")
        if row.weight_mass <= 0:
            raise ValueError("Supported calibration bins require positive IPCW mass")
        x.append(float(row.prediction)); y.append(float(row.observed))
        visible_n.append(int(row.n_known))
    return np.asarray(x), np.asarray(y), {
        "bins_total": len(z), "bins_plotted": len(visible_n),
        "bins_suppressed": int(z.n_known.lt(5).sum()),
        "known_outcomes_total": int(z.n_known.sum()),
        "known_outcomes_in_plotted_bins": sum(visible_n),
        "plotted_bin_counts": visible_n,
    }


def draw(ax, rows, task, legend=False):
    """Draw one transition; the compositor supplies its title and letter.

    Input rows are fixed risk bins pooled over fitted folds by IPCW mass.
    Prediction and observed are weighted means, rather than crude fractions.
    The caller keeps these empirical rows private.
    """
    missing = _REQUIRED.difference(rows.columns)
    if missing:
        raise ValueError("Missing pooled calibration columns: " + ", ".join(sorted(missing)))
    if task not in {"CN_MCI", "MCI_AD"}:
        raise ValueError("Unrecognized clinical transition")
    apply_figure_style(font="Nimbus Sans", sizes=(8, 7, 6.5), grid=False)
    plt.rcParams["savefig.bbox"] = None
    diagonal, = ax.plot([0, 1], [0, 1], color=_GREY, lw=0.7, ls=":",
                        label="Identity", zorder=0)
    handles, support = [], {}
    for horizon, marker, linestyle in _HORIZONS:
        z = rows[rows.task.eq(task) & rows.horizon.eq(horizon)]
        if z.empty:
            raise ValueError("Both three- and five-year calibration bins are required")
        x, y, counts = _curve(z)
        curve, = ax.plot(x, y, color=_TEAL, lw=0.95, ls=linestyle, marker=marker,
                         ms=3.2, mew=0.7, mec=_TEAL,
                         mfc=_TEAL if horizon == 3 else "white",
                         label=str(horizon) + " years", zorder=3)
        handles.append(curve); support[str(horizon)] = counts
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1],
           yticks=[0, 0.5, 1], xlabel="Mean predicted risk",
           ylabel="Observed event fraction")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", labelsize=6.5, length=2.5, width=0.6)
    if legend:
        ax.legend(handles=handles + [diagonal], loc="upper left",
                  fontsize=7, frameon=False, handlelength=1.7,
                  handletextpad=0.5, borderaxespad=0.5, labelspacing=0.35)
    return {
        "task": task, "summary": "IPCW mass-weighted first-visit calibration bins",
        "minimum_known_outcomes_per_visible_bin": 5,
        "curve_connections": "Only adjacent supported bins are connected",
        "support": support, "confidence_intervals": False,
        "recalibration_fitted": False, "hypothesis_tests_added": False,
    }
