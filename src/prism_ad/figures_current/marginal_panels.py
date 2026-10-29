"""Fold-specific participant swarms of fitted cognitive/genetic terms.

Private source files supply observations and revision-specified post hoc annotations.
Horizontal packing preserves each original contribution. Quartile boxes show
participant spread, not confidence intervals.
"""
import os
from pathlib import Path
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
FOLD_OFFSETS = {1: -0.26, 2: 0.0, 3: 0.26}
REQUIRED = {"component", "clinical_group", "fold", "n", "median", "q25", "q75"}
PAIRS = (("CN", "MCI"), ("MCI", "AD"), ("CN", "AD"))


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
    expected = {(g, f) for g, _ in GROUPS for f in FOLD_STYLES}
    if len(frame) != 9 or set(zip(frame.clinical_group, frame.fold)) != expected:
        raise ValueError("Require one summary per clinical group and fitted fold")
    if not (frame.q25.le(frame["median"]) & frame["median"].le(frame.q75)).all():
        raise ValueError("Require q25 <= median <= q75")
    return frame


def _observations(data, component):
    root = Path(os.environ["PRISM_AD_SOURCE_DATA"]) / "plotting"
    frame = pd.read_csv(root / "marginal_component_points.csv")
    if set(frame.columns) != {"component", "clinical_group", "fold", "value"}:
        raise ValueError("Individual plotting file must contain only component, "
                         "clinical_group, fold and value")
    frame = frame.loc[frame.component.eq(component)].copy()
    frame[["fold", "value"]] = frame[["fold", "value"]].apply(
        pd.to_numeric, errors="raise")
    if not np.isfinite(frame[["fold", "value"]].to_numpy(float)).all():
        raise ValueError("Individual contributions must be finite")
    if not frame.fold.eq(np.floor(frame.fold)).all():
        raise ValueError("Fold values must be integers")
    frame["fold"] = frame.fold.astype(int)
    if set(zip(frame.clinical_group, frame.fold)) != {
            (g, f) for g, _ in GROUPS for f in FOLD_STYLES}:
        raise ValueError("Unexpected individual clinical group or fitted fold")
    maximum_error = 0.0
    for row in data.itertuples():
        values = frame.loc[frame.clinical_group.eq(row.clinical_group)
                           & frame.fold.eq(row.fold), "value"].to_numpy(float)
        if len(values) != row.n:
            raise ValueError("Individual support differs from frozen summaries")
        actual = np.quantile(values, [0.25, 0.5, 0.75])
        target = np.array([row.q25, row.median, row.q75])
        if not np.allclose(actual, target, atol=1e-12, rtol=1e-12):
            raise ValueError("Individual quartiles differ from frozen summaries")
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - target))))
    return frame, root, maximum_error


def _pack(y_display, separation):
    """Deterministic circle packing with fixed supplied vertical positions."""
    y = np.asarray(y_display, dtype=float)
    order = np.argsort(y, kind="mergesort")
    placed = np.zeros(len(y), dtype=float)
    sorted_y = y[order]
    left = 0
    for i in range(len(order)):
        while left < i and sorted_y[i] - sorted_y[left] >= separation:
            left += 1
        active_x = placed[left:i]
        dy = sorted_y[i] - sorted_y[left:i]
        if len(active_x):
            dx = np.sqrt(np.maximum(0.0, separation ** 2 - dy ** 2))
            candidates = np.concatenate(([0.0], active_x - dx, active_x + dx))
            preferred = 1.0 if i % 2 else -1.0
            candidates = candidates[np.lexsort(
                (candidates * -preferred, np.abs(candidates)))]
            for candidate in candidates:
                if np.all((candidate - active_x) ** 2 + dy ** 2
                          >= separation ** 2 * (1.0 - 1e-10)):
                    placed[i] = candidate
                    break
            else:
                raise RuntimeError("No valid deterministic beeswarm position")
    result = np.empty_like(placed)
    result[order] = placed
    return result


def _significance(root, component):
    path = root / "marginal_component_tests.csv"
    if not path.is_file():
        return None
    tests = pd.read_csv(path)
    if not {"component", "group_a", "group_b", "p", "q", "label"}.issubset(tests):
        raise ValueError("Supplied marginal tests lack required columns")
    tests = tests.loc[tests.component.eq(component)].copy()
    if len(tests) != 3 or set(zip(tests.group_a, tests.group_b)) != set(PAIRS):
        raise ValueError("Require all three revision-specified clinical-course tests")
    for row in tests.itertuples():
        if not (np.isfinite(row.p) and np.isfinite(row.q)
                and 0 <= row.p <= 1 and 0 <= row.q <= 1):
            raise ValueError("Supplied p and q must be probabilities")
        label = ("***" if row.q < 0.001 else "**" if row.q < 0.01 else
                 "*" if row.q < 0.05 else "(*)" if row.p < 0.05 else "n.s.")
        if row.label != label:
            raise ValueError("Significance label disagrees with supplied p/q")
    return tests


def _brackets(ax, tests, high, span):
    if tests is None:
        return []
    positions = {"CN": 0, "MCI": 1, "AD": 2}
    rendered = []
    for pair in PAIRS:
        row = tests.loc[tests.group_a.eq(pair[0])
                        & tests.group_b.eq(pair[1])].iloc[0]
        x1, x2 = positions[pair[0]], positions[pair[1]]
        level = int(pair == ("CN", "AD"))
        if pair == ("CN", "MCI"):
            x2 -= 0.045
        elif pair == ("MCI", "AD"):
            x1 += 0.045
        y, h = high + span * (0.10 + level * 0.11), span * 0.022
        ax.plot([x1, x1, x2, x2], [y-h, y, y, y-h],
                color="#47545D", linewidth=0.55, clip_on=False, zorder=5)
        ax.text((x1+x2)/2, y+span*0.012, row.label, ha="center",
                va="bottom", fontsize=7, color="#24313B", clip_on=False, zorder=6)
        rendered.append({"group_a": pair[0], "group_b": pair[1],
                         "p": float(row.p), "q": float(row.q), "label": row.label})
    return rendered


def draw(ax, rows, component, legend=False):
    """Draw every observation; read private values and supplied test labels.

    rows supplies frozen n/median/q25/q75. Files are read from the plotting
    subdirectory of PRISM_AD_SOURCE_DATA. Tests may be absent during preview
    and are reported as pending. No inference is performed by this renderer.
    The caller supplies panel titles and letters.
    """
    apply_figure_style(font="Nimbus Sans", sizes=(8, 7, 6.5), grid=False)
    data = _validated(rows, component)
    points, root, quartile_error = _observations(data, component)
    tests = _significance(root, component)
    set_frame(ax, "open")
    color = COLORS[component]
    low, high = float(points.value.min()), float(points.value.max())
    span = high-low if high > low else max(abs(low), 1.0)
    ax.set_ylim(low-0.055*span, high+0.32*span)
    ax.set_xlim(-0.50, 2.50)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([label for _, label in GROUPS])
    ax.tick_params(axis="x", length=0, pad=3, labelsize=6.5, labelbottom=True)
    ax.tick_params(axis="y", labelsize=6.5, pad=2, labelleft=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylabel(("Cognitive" if component == "C" else "Genetic")
                  + " contribution, " + component + "\n(log-odds)",
                  fontsize=8, labelpad=4)
    ax.set_xlabel("Later clinical course", fontsize=8, labelpad=4)
    zero_reference = low <= 0 <= high
    if zero_reference:
        ax.axhline(0, color="#B9C0C5", linewidth=0.5, linestyle=":", zorder=0)

    # Physical-coordinate packing preserves clearance at every export DPI.
    dpi = ax.figure.dpi
    x_pixels = abs(ax.transData.transform((1, 0))[0]
                   - ax.transData.transform((0, 0))[0])
    lane_halfwidth = 0.115
    clusters = []
    for group_i, (group, _) in enumerate(GROUPS):
        for fold in FOLD_STYLES:
            values = np.sort(points.loc[points.clinical_group.eq(group)
                                       & points.fold.eq(fold), "value"].to_numpy(float),
                             kind="mergesort")
            y_pixels = ax.transData.transform(np.column_stack(
                (np.zeros(len(values)), values)))[:, 1]
            clusters.append((group_i, group, fold, values, y_pixels))
    packings = None
    for diameter in (1.15, 1.10, 1.05, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
        separation = diameter*dpi/72.0*1.08
        trial = [_pack(c[4], separation)/x_pixels for c in clusters]
        if max(float(np.max(np.abs(x))) for x in trial) <= lane_halfwidth:
            packings = trial
            break
    if packings is None:
        raise RuntimeError("Beeswarm cannot fit supported sizes; enlarge axes")

    counts, packing_qa, artist_error = {}, [], 0.0
    width = 0.15
    for cluster, packed in zip(clusters, packings):
        group_i, group, fold, values, y_pixels = cluster
        counts.setdefault(group, {})[str(fold)] = len(values)
        center = group_i+FOLD_OFFSETS[fold]
        dots = ax.scatter(center+packed, values, s=diameter**2, color=color,
                          alpha=0.82, linewidths=0, edgecolors="none", zorder=2)
        dots.set_gid("marginal-swarm-"+component+"-"+group+"-"+str(fold))
        artist_error = max(artist_error, float(np.max(np.abs(
            np.asarray(dots.get_offsets(), dtype=float)[:, 1]-values))))
        row = data.loc[data.clinical_group.eq(group) & data.fold.eq(fold)].iloc[0]
        left = center-width/2
        box = Rectangle((left, row.q25), width, row.q75-row.q25,
                        facecolor=to_rgba(color, 0.04),
                        edgecolor=to_rgba(color, 0.82), linewidth=0.65,
                        linestyle=FOLD_STYLES[fold], zorder=3)
        box.set_joinstyle("miter")
        ax.add_patch(box)
        ax.plot([left, left+width], [row["median"], row["median"]],
                color="#3D4449", linewidth=0.85, linestyle=FOLD_STYLES[fold],
                solid_capstyle="butt", zorder=4)
        xy = np.column_stack((packed*x_pixels, y_pixels))
        minimum = float("inf")
        for i in range(len(values)):
            close = np.where(np.abs(y_pixels[i+1:]-y_pixels[i])
                             < separation)[0]+i+1
            if len(close):
                minimum = min(minimum, float(np.min(np.sqrt(
                    np.sum((xy[close]-xy[i])**2, axis=1)))))
        packing_qa.append({"clinical_group": group, "fold": fold, "n": len(values),
                           "maximum_horizontal_offset": float(np.max(np.abs(packed))),
                           "minimum_near_neighbor_clearance_pt": (
                               None if not np.isfinite(minimum) else minimum*72/dpi)})
    annotations = _brackets(ax, tests, high, span)
    if legend:
        handles = [Line2D([], [], color="#65717E", linewidth=0.9,
                          linestyle=style, label="Fold "+str(fold))
                   for fold, style in FOLD_STYLES.items()]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.03),
                  ncol=3, fontsize=7, frameon=False, handlelength=2,
                  handletextpad=0.4, columnspacing=1.0, borderaxespad=0)
    return {"component": component, "summary_rows": len(data),
            "participant_count": int(data.n.sum()), "rendered_points": len(points),
            "fold_group_support": counts,
            "display": "all observations in deterministic horizontal swarms with IQR boxes",
            "interval": "participant interquartile range, not confidence interval",
            "no_whiskers": True, "fold_order_within_course": [1, 2, 3],
            "native_scale_retained": True, "pooled_across_folds": False,
            "zero_reference": zero_reference, "marker_diameter_pt": diameter,
            "marker_fallback_used": diameter != 1.15,
            "vertical_artist_max_error": artist_error,
            "quartile_max_error": quartile_error,
            "horizontal_lane_halfwidth": lane_halfwidth, "packing": packing_qa,
            "significance_pending": tests is None, "supplied_tests": annotations,
            "inference_computed_by_renderer": False}
