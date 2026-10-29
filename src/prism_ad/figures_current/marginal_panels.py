"""Fold-specific marginal distributions of fitted cognitive/genetic terms.

The renderer consumes aggregate medians and quartiles, never participant rows.
Boxes represent participant interquartile spread, not inferential intervals.
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator

from .style import apply_figure_style, set_frame


COLORS = {"C": "#D8892B", "G": "#7656A4"}
GROUPS = (("CN", "Last CN"), ("MCI", "Last MCI"), ("AD", "Later AD"))
FOLD_STYLES = {1: "-", 2: "--", 3: ":"}
FOLD_OFFSETS = {1: -0.22, 2: 0.0, 3: 0.22}
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
    """Draw native signed terms as fold-specific quartile boxes.

    Boxes extend only from q25 to q75, with the median drawn inside. Whiskers,
    extrema, outliers and participant observations are not supplied or inferred.
    The caller supplies the panel title and letter.
    """
    apply_figure_style(font="Nimbus Sans", sizes=(8, 7, 6.5), grid=False)
    data = _validated(rows, component)
    set_frame(ax, "open")
    color = COLORS[component]
    width = 0.17
    counts = {}
    for group_i, (group, _) in enumerate(GROUPS):
        counts[group] = {}
        for fold in FOLD_STYLES:
            row = data.loc[data.clinical_group.eq(group) & data.fold.eq(fold)].iloc[0]
            center = group_i + FOLD_OFFSETS[fold]
            left = center - width / 2
            box = Rectangle((left, row.q25), width, row.q75 - row.q25,
                            facecolor=to_rgba(color, 0.17), edgecolor=color,
                            linewidth=0.9, linestyle=FOLD_STYLES[fold], zorder=2)
            box.set_joinstyle("miter")
            ax.add_patch(box)
            ax.plot([left, left + width], [row["median"], row["median"]],
                    color=color, linewidth=1.35, solid_capstyle="butt", zorder=3)
            counts[group][str(fold)] = int(row.n)

    low, high = float(data.q25.min()), float(data.q75.max())
    span = high - low
    pad = 0.09 * span if span > 0 else max(abs(low) * 0.09, 0.05)
    ax.set_ylim(low - pad, high + pad)
    zero_reference = low <= 0 <= high
    if zero_reference:
        ax.axhline(0, color="#A6AFB5", linewidth=0.6, linestyle=":", zorder=0)
    ax.set_xlim(-0.50, len(GROUPS) - 0.50)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([label for _, label in GROUPS])
    ax.tick_params(axis="x", length=0, pad=3, labelsize=6.5, labelbottom=True)
    ax.tick_params(axis="y", labelsize=6.5, pad=2, labelleft=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_ylabel(("Cognitive" if component == "C" else "Genetic")
                  + " contribution, " + component + "\n(log-odds)",
                  fontsize=8, labelpad=4)
    ax.set_xlabel("Later clinical course", fontsize=8, labelpad=4)
    if legend:
        handles = [Line2D([], [], color="#65717E", linewidth=0.9,
                          linestyle=style, label="Fold " + str(fold))
                   for fold, style in FOLD_STYLES.items()]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.03),
                  ncol=3, fontsize=7, frameon=False, handlelength=2,
                  handletextpad=0.4, columnspacing=1.0, borderaxespad=0)
    return {"component": component, "summary_rows": len(data),
            "participant_count": int(data.n.sum()), "fold_group_support": counts,
            "display": "q25-to-q75 boxes with median lines",
            "interval": "participant interquartile range, not confidence interval",
            "no_whiskers_or_outliers": True,
            "fold_order_within_course": [1, 2, 3],
            "native_scale_retained": True, "pooled_across_folds": False,
            "zero_reference": zero_reference, "new_inference": False}
