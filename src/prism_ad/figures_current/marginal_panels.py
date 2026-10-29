"""Fold-specific marginal distributions of fitted cognitive/genetic terms.

The renderer consumes aggregate medians and quartiles, never participant rows.
Ranges represent participant interquartile spread, not inferential intervals.
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .style import apply_figure_style, set_frame


COLORS = {"C": "#D8892B", "G": "#7656A4"}
GROUPS = (("CN", "Last CN"), ("MCI", "Last MCI"), ("AD", "Later AD"))
FOLD_STYLES = {1: "-", 2: "--", 3: ":"}
FOLD_OFFSETS = {1: -0.20, 2: 0.0, 3: 0.20}
REQUIRED = {"component", "clinical_group", "fold", "n", "median", "q25", "q75"}


def _validated(rows, component):
    if component not in COLORS:
        raise ValueError("Component must be C or G")
    frame = pd.DataFrame(rows).copy()
    if not REQUIRED.issubset(frame.columns):
        raise ValueError("Marginal summaries lack required columns")
    frame = frame.loc[frame.component.eq(component)].copy()
    numeric = ["fold", "n", "median", "q25", "q75"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Marginal summaries must be finite")
    if not (frame.fold.eq(np.floor(frame.fold)) & frame.n.eq(np.floor(frame.n))
            & frame.n.gt(0)).all():
        raise ValueError("Fold and positive support must be integer-valued")
    frame["fold"] = frame.fold.astype(int)
    expected = {(group, fold) for group, _ in GROUPS for fold in FOLD_STYLES}
    found = list(zip(frame.clinical_group, frame.fold))
    if len(found) != len(expected) or set(found) != expected:
        raise ValueError("Require one summary per clinical group and fitted fold")
    if not (frame.q25.le(frame["median"]) & frame["median"].le(frame.q75)).all():
        raise ValueError("Require q25 <= median <= q75")
    return frame


def draw(ax, rows, component, legend=False):
    """Draw one panel of fold-specific medians and interquartile ranges.

    ax is a publication-sized axes. The caller supplies the title/panel letter.
    rows has component, clinical_group, fold, n, median, q25 and q75 columns.
    Each C/G panel requires CN, MCI and AD summaries for folds 1, 2 and 3.
    No exponentiation, recentering or pooling of the native terms is performed.
    A shared figure fold key is preferred when already supplied by the caller.
    """
    apply_figure_style(font="Nimbus Sans", sizes=(8, 7, 6.5), grid=False)
    data = _validated(rows, component)
    set_frame(ax, "open")
    color = COLORS[component]
    marker = "o" if component == "C" else "s"
    counts = {}
    for group_i, (group, _) in enumerate(GROUPS):
        counts[group] = {}
        for fold in FOLD_STYLES:
            row = data.loc[data.clinical_group.eq(group) & data.fold.eq(fold)].iloc[0]
            y = group_i + FOLD_OFFSETS[fold]
            ax.plot([row.q25, row.q75], [y, y], color=color,
                    linestyle=FOLD_STYLES[fold], linewidth=1.15,
                    solid_capstyle="butt", dash_capstyle="round", zorder=2)
            # End caps keep the quartile range legible when a dash is short.
            for x in (row.q25, row.q75):
                ax.plot([x, x], [y - 0.045, y + 0.045], color=color,
                        linewidth=0.65, solid_capstyle="butt", zorder=2)
            ax.plot(row["median"], y, marker=marker, linestyle="None",
                    markersize=3.6, markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.35, zorder=3)
            counts[group][str(fold)] = int(row.n)

    low, high = float(data.q25.min()), float(data.q75.max())
    span = high - low
    pad = 0.07 * span if span > 0 else max(abs(low) * 0.07, 0.05)
    ax.set_xlim(low - pad, high + pad)
    zero_reference = low <= 0 <= high
    if zero_reference:
        ax.axvline(0, color="#A6AFB5", linewidth=0.6, linestyle=":", zorder=0)
    ax.set_ylim(len(GROUPS) - 0.46, -0.46)
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([label for _, label in GROUPS])
    ax.tick_params(axis="y", length=0, pad=3, labelsize=6.5)
    ax.tick_params(axis="x", labelsize=6.5, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel(("Cognitive" if component == "C" else "Genetic")
                  + " contribution, " + component + "\n(log-odds)",
                  fontsize=8, labelpad=4)
    ax.set_ylabel("")
    if legend:
        handles = [Line2D([], [], color="#65717E", linewidth=1.15,
                          linestyle=style, label="Fold " + str(fold))
                   for fold, style in FOLD_STYLES.items()]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.03),
                  ncol=3, fontsize=7, frameon=False, handlelength=2,
                  handletextpad=0.4, columnspacing=1.0, borderaxespad=0)
    return {"component": component, "summary_rows": len(data),
            "participant_count": int(data.n.sum()), "fold_group_support": counts,
            "interval": "participant interquartile range, not confidence interval",
            "native_scale_retained": True, "pooled_across_folds": False,
            "zero_reference": zero_reference, "new_inference": False}
