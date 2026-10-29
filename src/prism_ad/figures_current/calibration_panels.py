"""Calibration points and fixed-prediction slope diagnostics from private summaries.

The renderer fits no model, applies no recalibration, and omits unsupported bins.
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


def draw(ax, rows, task, legend=False, diagnostics=None):
    """Draw binned calibration points; the compositor supplies title/letter.

    The coordinates retain the original IPCW-weighted bin means. Unsupported
    bins are omitted, and no connecting line implies interpolation across bins.
    """
    from matplotlib.lines import Line2D
    missing = _REQUIRED.difference(rows.columns)
    if missing:
        raise ValueError("Missing pooled calibration columns: " + ", ".join(sorted(missing)))
    if task not in {"CN_MCI", "MCI_AD"}:
        raise ValueError("Unrecognized clinical transition")
    apply_figure_style(font="Nimbus Sans", sizes=(8, 7, 6.5), grid=False)
    plt.rcParams["savefig.bbox"] = None
    reference_color = "#ABB6BF"
    colors = {3: _TEAL, 5: "#597C9B"}
    diagonal, = ax.plot([0, 1], [0, 1], color=reference_color, lw=0.7,
                        ls="-", label="Ideal", zorder=0)
    handles, support = [], {}
    for horizon, marker, _ in _HORIZONS:
        z = rows[rows.task.eq(task) & rows.horizon.eq(horizon)]
        if z.empty:
            raise ValueError("Both three- and five-year calibration bins are required")
        x, y, counts = _curve(z)
        valid = np.isfinite(x) & np.isfinite(y)
        color = colors[horizon]
        area = 17.5 if horizon == 3 else 22.0
        face = color if horizon == 3 else "white"
        edge = "white" if horizon == 3 else color
        ax.scatter(x[valid], y[valid], s=area, marker=marker,
                   facecolors=face, edgecolors=edge,
                   linewidths=0.35 if horizon == 3 else 0.85,
                   zorder=4 if horizon == 3 else 3)
        handles.append(Line2D(
            [], [], marker=marker, linestyle="None", color=color,
            markersize=np.sqrt(area), markerfacecolor=face,
            markeredgecolor=edge, markeredgewidth=0.35 if horizon == 3 else 0.85,
            label=str(horizon) + " years"))
        support[str(horizon)] = counts
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1],
           yticks=[0, 0.5, 1], xlabel="Mean predicted risk",
           ylabel="Observed event fraction")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", labelsize=6.5, length=2.5, width=0.6)
    if legend:
        ax.legend(handles=handles + [diagonal], loc="upper left",
                  fontsize=7, frameon=False, handlelength=1.3,
                  handletextpad=0.5, borderaxespad=0.5, labelspacing=0.4)
    diagnostic_annotations = []
    if diagnostics is not None:
        required_diagnostic = {"task", "horizon", "slope", "ci_low", "ci_high", "p", "q", "label"}
        missing_diagnostic = required_diagnostic.difference(diagnostics.columns)
        if missing_diagnostic:
            raise ValueError("Missing calibration diagnostic columns: " + ", ".join(sorted(missing_diagnostic)))
        selected = diagnostics.loc[diagnostics.task.eq(task)].copy()
        if len(selected) != 2 or set(selected.horizon) != {3, 5}:
            raise ValueError("One calibration diagnostic per horizon is required")
        for j, horizon in enumerate((3, 5)):
            row = selected.loc[selected.horizon.eq(horizon)].iloc[0]
            numbers = np.asarray([row.slope, row.ci_low, row.ci_high, row.p, row.q], dtype=float)
            if not np.isfinite(numbers).all() or not row.ci_low <= row.slope <= row.ci_high:
                raise ValueError("Finite, ordered calibration estimates and CIs are required")
            if not 0 <= row.p <= 1 or not 0 <= row.q <= 1:
                raise ValueError("Diagnostic probabilities must be in [0,1]")
            label = "***" if row.q < .001 else "**" if row.q < .01 else "*" if row.q < .05 else "(*)" if row.p < .05 else "n.s."
            if str(row.label) != label:
                raise ValueError("Calibration marker does not match the supplied P and BH q")
            text = (str(horizon) + "y slope " + format(row.slope, ".2f")
                    + " (" + format(row.ci_low, ".2f") + "–"
                    + format(row.ci_high, ".2f") + "), " + label)
            # Two fixed physical offsets leave every calibration point visible.
            # The compositor reserves the corresponding 18-mm bottom strip.
            artist = ax.annotate(text, xy=(0, 0), xycoords=ax.transAxes,
                                 xytext=(0, -(14 + j*4)*72/25.4),
                                 textcoords="offset points", ha="left", va="center",
                                 fontsize=6.5, color=colors[horizon], clip_on=False,
                                 annotation_clip=False)
            artist.set_gid("calibration_slope_" + task + "_" + str(horizon))
            diagnostic_annotations.append(dict(horizon=int(horizon), slope=float(row.slope),
                                               ci_low=float(row.ci_low), ci_high=float(row.ci_high),
                                               p=float(row.p), q=float(row.q), label=label,
                                               interpretation="departure of slope from 1; not comparative accuracy"))
    return {
        "task": task, "summary": "IPCW mass-weighted first-visit calibration bins",
        "display": "binned calibration scatter with no connecting lines",
        "minimum_known_outcomes_per_visible_bin": 5,
        "curve_connections": "None; no interpolation or smoothing",
        "unsupported_bin_gaps_retained": True,
        "horizon_colors": {str(k): v for k, v in colors.items()},
        "marker_sizes_encode_support": False,
        "support": support, "bin_confidence_intervals": False,
        "diagnostic_annotations": diagnostic_annotations,
        "diagnostic_confidence_intervals": "nominal normal Wald 95%; HC1 empirical-score sandwich" if diagnostics is not None else None,
        "recalibration_fitted": False, "hypothesis_tests_added": diagnostics is not None,
        "diagnostic_null": "calibration slope = 1" if diagnostics is not None else None,
    }
