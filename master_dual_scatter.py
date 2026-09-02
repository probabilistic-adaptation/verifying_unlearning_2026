"""
Batch-generates "dual" scatterplots for one seed's results folder.

Each entry in SCATTERPLOT_PAIRS produces a single figure with two panels laid
out side by side:

  * left  -- the full scatterplot (every model), exactly like the ones
             master_results.py's run_scatterplots() produces.
  * right -- the same scatterplot zoomed to the axis ranges given by the
             entry's `zoom_x` / `zoom_y` keys, i.e. showing just the models
             that fall into that window.

The data gathering / flattening / colour+marker conventions are all reused
from master_results.py so the two scripts stay in sync.
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from master_results import (
    SEED,
    RESULTS_FOLDER,
    PLOTS_FOLDER,
    EPOCHS, CLASSES, PERCENTS, RUNS, METHODS, BASE, RETRAIN,
    METHOD_MARKERS,
    gather_results,
    _safe_filename,
    _log_transform,
    _method_color,
    _method_marker,
)


# ============================================================
# CONFIG
# ============================================================

# (x_metric, y_metric) pairs to scatterplot. Same schema as
# master_results.SCATTERPLOT_PAIRS, plus two required keys:
#
#   zoom_x = (xmin, xmax)   x-limits for the right-hand (zoomed) panel
#   zoom_y = (ymin, ymax)   y-limits for the right-hand (zoomed) panel
#
# Everything else (xlabel, ylabel, color_by, marker_by, line,
# log_scale, grouped handling) behaves just like in master_results.py.
#
# The zoom windows below are first guesses -- tweak them per plot so the
# right-hand panel frames whatever cluster you care about (typically the
# retrain/base region near the top-right for accuracies).
SCATTERPLOT_PAIRS = [

    # ----------------
    # -- Task Quality
    # ----------------

    dict(x_metric="forget_acc",
         y_metric="retain_acc",
         color_by="method", line=0,
         xlabel="Forget Accuracy", ylabel="Retain Accuracy",
         zoom_x=(98.5, 100), zoom_y=(98.5, 100)),

    dict(x_metric="forget_acc",
         y_metric="test_acc",
         color_by="method", line=0,
         xlabel="Forget accuracy", ylabel="Test accuracy",
         zoom_x=(97, 100), zoom_y=(91, 94)),

    dict(x_metric="retain_acc",
         y_metric="test_acc",
         color_by="method", line=0,
         xlabel="Retain accuracy", ylabel="Test accuracy",
         zoom_x=(97, 100), zoom_y=(91, 94)),

    dict(x_metric="forget_acc",
         y_metric="forget_m_entropy",
         color_by="method", line=0,
         xlabel="Forget accuracy", ylabel="Forget m-entropy`",
         zoom_x=(98.5, 100), zoom_y=(0, 0.1)),

    dict(x_metric="forget_entropy",
         y_metric="forget_m_entropy",
         color_by="method", line=0,
         xlabel="Forget entropy", ylabel="Forget m-entropy`",
         zoom_x=(0, 0.06), zoom_y=(0, 0.06)),


    # ----------------
    # -- MIA (threshold)
    # ----------------

    dict(x_metric="threshold_MIA.efficacy",
         y_metric="threshold_MIA.attack_accuracy",
         color_by="method", line=0,
         xlabel="Efficacy", ylabel="Accuracy",
         zoom_x=(0.02, 0.12), zoom_y=(0.525, 0.55)),

    dict(x_metric="forget_acc",
         y_metric="threshold_MIA.efficacy",
         color_by="method", line=0,
         xlabel="Forget accuracy", ylabel="MIA efficacy",
         zoom_x=(0.0, 0.2), zoom_y=(-0.1, 0.3)),

    # ----------------
    # -- Holistic
    # ----------------

    dict(x_metric="ToW",
         y_metric="ToW_MIA",
         color_by="method", line=0,
         xlabel="ToW", ylabel="ToW_MIA",
         zoom_x=(0.8, 1.0), zoom_y=(0.8, 1.0)),

    # ----------------
    # -- output diffs and divergences
    # ----------------

    dict(x_metric="forget_m_entropy",
         y_metric="outputs.retrain_vs_unlearned.forget.absolute_distance",
         color_by="method",
         xlabel="Forget m-entropy", ylabel="Absolute distance",
         zoom_x=(0.0, 2.0), zoom_y=(0.0, 0.2)),

    # ----------------------------------------
    # ---------------------- Relearn Time ---
    # ----------------------------------------

    dict(x_metric="forget_m_entropy",
         y_metric="relearn_time.avg_epochs",
         color_by="method", line=0,
         xlabel="Forget m_entropy", ylabel="Relearn time",
         zoom_x=(0, .1), zoom_y=(0.0, 5.0)),

    dict(x_metric="weight_differences.original_vs_unlearned.l2_distance",
         y_metric="relearn_time.avg_epochs",
         color_by="method",
         xlabel = "Weight L2-distance (original-vs-unlearned)", ylabel = "Relearn time",
         zoom_x=(0, 3), zoom_y=(0.0, 3)),


    dict(x_metric="forget_m_entropy",
         y_metric="weight_differences.original_vs_unlearned.l2_distance",
         color_by="method", line=0,
         xlabel="Forget m_entropy", ylabel="L2 weight distance, original vs. unlearned",
         zoom_x=(0, .1), zoom_y=(0.0, 5.0)),






]


# ============================================================
# plotting
# ============================================================

def _build_plot_df(final_df, x_metric, y_metric, color_by, marker_by, grouped):
    """
    Mirror of the grouping/aggregation block in
    master_results.results_scatterplot(): returns (plot_df, x_err_col, y_err_col).
    """
    if not grouped:
        return final_df, None, None

    group_cols = list(dict.fromkeys(["method", "epoch", color_by, marker_by]))
    agg = final_df.groupby(group_cols, dropna=False)[[x_metric, y_metric]].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    plot_df = agg.reset_index()
    x_err_col, y_err_col = f"{x_metric}_std", f"{y_metric}_std"
    plot_df = plot_df.rename(columns={f"{x_metric}_mean": x_metric, f"{y_metric}_mean": y_metric})
    return plot_df, x_err_col, y_err_col


def _palettes(plot_df, color_by, marker_by, base_color, retain_color, base_marker, retain_marker):
    """Mirror of the colour/marker-dict block in master_results.results_scatterplot()."""
    colors_list = plt.cm.tab20.colors

    palette_dict, color_idx = {}, 0
    for val in sorted(plot_df[color_by].unique(), key=str):
        if val == -99 or val == "base":
            palette_dict[val] = base_color
        elif val == -98 or val == "retrain":
            palette_dict[val] = retain_color
        elif color_by == "method":
            palette_dict[val] = _method_color(val)
        else:
            palette_dict[val] = colors_list[color_idx % len(colors_list)]
            color_idx += 1

    marker_dict, marker_idx = {}, 0
    for val in sorted(plot_df[marker_by].unique(), key=str):
        if val == -99 or val == "base":
            marker_dict[val] = base_marker
        elif val == -98 or val == "retrain":
            marker_dict[val] = retain_marker
        elif marker_by == "method":
            marker_dict[val] = _method_marker(val)
        else:
            marker_dict[val] = METHOD_MARKERS[marker_idx % len(METHOD_MARKERS)]
            marker_idx += 1

    return palette_dict, marker_dict


def _draw_on_ax(ax, plot_df, x_metric, y_metric, color_by, marker_by,
                palette_dict, marker_dict, x_err_col, y_err_col, grouped):
    if grouped:
        for _, row in plot_df.iterrows():
            color = palette_dict.get(row[color_by], "gray")
            xerr = row[x_err_col] if pd.notna(row[x_err_col]) else 0
            yerr = row[y_err_col] if pd.notna(row[y_err_col]) else 0
            ax.errorbar(
                row[x_metric], row[y_metric], xerr=xerr, yerr=yerr,
                fmt="none", ecolor=color, elinewidth=1.2, capsize=3, alpha=0.6, zorder=1,
            )

    sns.scatterplot(
        data=plot_df, x=x_metric, y=y_metric,
        hue=color_by, style=marker_by,
        palette=palette_dict, markers=marker_dict,
        s=80, alpha=0.7, edgecolor=None, ax=ax, zorder=2, legend=False,
    )
    ax.grid(True, alpha=0.3)


def _reference_line(ax, line):
    """Mirror of the `line` block in master_results.results_scatterplot()."""
    if line not in (1, -1):
        return
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
    if line == 1:
        y_lo, y_hi = lo, hi
    else:
        y_lo, y_hi = 1 - lo, 1 - hi
    ax.plot([lo, hi], [y_lo, y_hi], linestyle="--", color="grey", linewidth=1, zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def dual_results_scatterplot(final_df, x_metric, y_metric, zoom_x, zoom_y,
                             color_by="method", marker_by="method",
                             base_color="black", retain_color="purple",
                             base_marker="o", retain_marker="X",
                             xlabel=None, ylabel=None,
                             grouped=True, line=0, log_scale=False,
                             save_path=None):
    """
    Two-panel scatterplot: left = full, right = zoomed to (zoom_x, zoom_y).

    zoom_x / zoom_y are (min, max) tuples in the same units as the plotted
    metrics (after the log transform, if log_scale=True). The right panel just
    reuses the full plot's data with those axis limits applied, so it shows
    only the models that fall inside the window; a dashed red rectangle on the
    left panel marks that window.
    """
    if log_scale:
        final_df = _log_transform(final_df, (x_metric, y_metric))

    plot_df, x_err_col, y_err_col = _build_plot_df(
        final_df, x_metric, y_metric, color_by, marker_by, grouped
    )
    palette_dict, marker_dict = _palettes(
        plot_df, color_by, marker_by, base_color, retain_color, base_marker, retain_marker
    )

    plt.clf()
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.35)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6))

    for ax in (ax_full, ax_zoom):
        _draw_on_ax(ax, plot_df, x_metric, y_metric, color_by, marker_by,
                    palette_dict, marker_dict, x_err_col, y_err_col, grouped)

    _reference_line(ax_full, line)

    (zx0, zx1) = sorted(zoom_x)
    (zy0, zy1) = sorted(zoom_y)
    ax_zoom.set_xlim(zx0, zx1)
    ax_zoom.set_ylim(zy0, zy1)
    _reference_line(ax_zoom, line)

    # mark the zoom window on the full panel
    ax_full.add_patch(Rectangle(
        (zx0, zy0), zx1 - zx0, zy1 - zy0,
        fill=False, edgecolor="red", linestyle="--", linewidth=1.4, zorder=5,
    ))

    # small red arrow just right of the box, pointing towards the zoomed panel.
    # all three offsets are in x-data units of the full panel:
    yc = (zy0 + zy1) / 2
    span = (zx1 - zx0) or 1.0
    arrow_gap = 0.15 * span   # distance from the box's right edge to the tail
    arrow_len = 0.80 * span   # length of the arrow -- raise this to make it longer
    arrow_nudge = 1.0         # shift the whole arrow this much further right
    tail_x = zx1 + arrow_gap + arrow_nudge
    ax_full.annotate(
        "", xy=(tail_x + arrow_len, yc), xytext=(tail_x, yc),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=1.8),
        annotation_clip=False, zorder=6,
    )

    default_xlabel = ("ln " if log_scale else "") + x_metric.capitalize().replace("_", " ")
    default_ylabel = ("ln " if log_scale else "") + y_metric.capitalize().replace("_", " ")
    # y-axis label only on the left panel; a single x-axis label centred
    # between the two panels
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax_full.set_ylabel(ylabel if ylabel is not None else default_ylabel)
    fig.supxlabel(xlabel if xlabel is not None else default_xlabel)
    # ax_full.set_title("full")
    # ax_zoom.set_title(f"zoom  x{tuple(round(v, 3) for v in (zx0, zx1))}  y{tuple(round(v, 3) for v in (zy0, zy1))}",
    #                   fontsize=11)

    fig.tight_layout()
    if save_path is None:
        plt.show()
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# driving logic
# ============================================================

def run_dual_scatterplots(results, plots_folder):
    out_folder = os.path.join(plots_folder, "dual_scatterplots")
    for pair in SCATTERPLOT_PAIRS:
        pair = dict(pair)
        x_metric = pair.pop("x_metric")
        y_metric = pair.pop("y_metric")
        zoom_x = pair.pop("zoom_x")
        zoom_y = pair.pop("zoom_y")
        for grouped in (True, False):
            subfolder = "grouped" if grouped else "ungrouped"
            save_path = os.path.join(
                out_folder, subfolder,
                f"{_safe_filename(x_metric)}__vs__{_safe_filename(y_metric)}.svg",
            )
            try:
                dual_results_scatterplot(
                    results,
                    x_metric=x_metric,
                    y_metric=y_metric,
                    zoom_x=zoom_x,
                    zoom_y=zoom_y,
                    grouped=grouped,
                    save_path=save_path,
                    **pair,
                )
            except Exception as e:
                print(f"[master_dual_scatter] skipped '{x_metric}' vs '{y_metric}' "
                      f"(grouped={grouped}): {e}")


def main():
    print(f"[master_dual_scatter] gathering results from {RESULTS_FOLDER} (seed {SEED})")
    results = gather_results(
        RESULTS_FOLDER,
        epochs=EPOCHS, classes=CLASSES, percents=PERCENTS, runs=RUNS, methods=METHODS,
        base=BASE, retrain=RETRAIN,
    )
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    run_dual_scatterplots(results, PLOTS_FOLDER)
    print(f"[master_dual_scatter] done. plots saved under "
          f"{os.path.join(PLOTS_FOLDER, 'dual_scatterplots')}")


if __name__ == "__main__":
    main()
