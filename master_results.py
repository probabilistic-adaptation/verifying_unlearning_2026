"""
Batch-generates every barplot/scatterplot in visualize_results.ipynb for one
seed's results folder and saves them to disk (rather than showing them inline).
"""

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================

SEED = 4

RESULTS_FOLDER = f"results/seed_{SEED}"
PLOTS_FOLDER = f"{RESULTS_FOLDER}/plots"

# used only to discover which metrics exist in a "typical" results.json, so we
# know what to barplot by default
REFERENCE_RESULT_FILE = f"{RESULTS_FOLDER}/unlearn/run_1/FT/epoch_30/class_5.json"

# gather_results filters -- None means "no filter"
EPOCHS = None
CLASSES = None
PERCENTS = None
RUNS = None
METHODS = None
BASE = True
RETRAIN = True

# identity/grouping columns that show up in a flattened results.json but
# aren't metrics in their own right, so they're never auto-barplotted
NON_METRIC_COLUMNS = {
    "type", "epoch", "run", "forget_set_type", "unlearning_item", "method",
    "class", "percent",
}

# (x_metric, y_metric) pairs to scatterplot, each with its own plotting
# kwargs (anything results_scatterplot() accepts) -- add/remove pairs here to
# control what gets generated
SCATTERPLOT_PAIRS = [

    # ----------------
    # -- Task Quality
    # ----------------

    dict(x_metric="forget_acc", 
         y_metric="retain_acc",
         title="Forget accuracy vs. Retain accuracy", color_by="method", line=0,
         xlabel = "Forget Accuracy", ylabel = "Retain Accuracy"),

    dict(x_metric="forget_acc", 
         y_metric="test_acc",
         title="Forget accuracy vs. Test accuracy", color_by="method", line=0,
         xlabel = "Forget accuracy", ylabel = "Test accuracy"),

    dict(x_metric="forget_acc",
         y_metric="forgotten_class_fraction",
         title="Forget accuracy vs. Forgotten class share", color_by="method", line=0,
         xlabel = "Forget accuracy", ylabel = "Forgotten class share"),



    # ----------------
    # -- MIA (threshold)
    # ----------------

    dict(x_metric="threshold_MIA.efficacy", 
         y_metric="threshold_MIA.attack_accuracy",
         title="Threshold MIA: attack efficacy vs. accuracy", color_by="method", line=0,
         xlabel = "Efficacy", ylabel = "Accuracy"),

    dict(x_metric="threshold_MIA.efficacy", 
         y_metric="threshold_MIA.auc",
         title="Threshold MIA: attack efficacy vs. AUC", color_by="method", line=0,
         xlabel = "Efficacy", ylabel = "AUC"),

    dict(x_metric="threshold_MIA.efficacy", 
         y_metric="threshold_MIA.f1",
         title="Threshold MIA: attack efficacy vs. F1-score", color_by="method", line=0,
         xlabel = "Efficacy", ylabel = "F1-score"),

    dict(x_metric="threshold_MIA.efficacy", 
         y_metric="threshold_MIA.avg_member_probability_forget",
         title="Threshold MIA: attack efficacy vs. average member probability (forget set)", color_by="method", line=0,
         xlabel = "Efficacy", ylabel = "Average member probability (forget set)"),

    dict(x_metric="threshold_MIA.attack_accuracy", 
         y_metric="threshold_MIA.auc",
         title="Threshold MIA: attack accuaracy vs. AUC", color_by="method", line=0,
         xlabel = "Accuracy", ylabel = "AUC"),

    dict(x_metric="threshold_MIA.attack_accuracy", 
         y_metric="threshold_MIA.f1",
         title="Threshold MIA: attack accuaracy vs. F1-score", color_by="method", line=0,
         xlabel = "Accuracy", ylabel = "F1-score"),

    dict(x_metric="threshold_MIA.attack_accuracy", 
         y_metric="threshold_MIA.avg_member_probability_forget",
         title="Threshold MIA: attack accuracy vs. average member probability (forget set)", color_by="method", line=0,
         xlabel = "Accuracy", ylabel = "Average member probability (forget set)"),

    dict(x_metric="threshold_MIA.auc", 
         y_metric="threshold_MIA.f1",
         title="Threshold MIA: attack AUC vs. F1-score", color_by="method", line=0,
         xlabel = "AUC", ylabel = "F1-score"),

    dict(x_metric="threshold_MIA.auc", 
         y_metric="threshold_MIA.avg_member_probability_forget",
         title="Threshold MIA: attack AUC vs. average member probability (forget set)", color_by="method", line=0,
         xlabel = "AUC", ylabel = "Average member probability (forget set)"),

    dict(x_metric="threshold_MIA.f1", 
         y_metric="threshold_MIA.avg_member_probability_forget",
         title="Threshold MIA: attack F1-score vs. average member probability (forget set)", color_by="method", line=0,
         xlabel = "F1-score", ylabel = "Average member probability (forget set)"),



    # ----------------
    # -- MIA (logistic)
    # ----------------

    dict(x_metric="logistic_regression_MIA.efficacy",
         y_metric="logistic_regression_MIA.attack_accuracy",
         title="Logistic Regression MIA: attack efficacy vs. accuracy", color_by="method", line=0,
         xlabel = "Efficacy", ylabel = "Accuracy"),

    dict(x_metric="logistic_regression_MIA.efficacy",
         y_metric="logistic_regression_MIA.auc",
         title="Logistic Regression MIA: attack efficacy vs. AUC", color_by="method", line=0,
         xlabel = "Efficacy", ylabel = "AUC"),

    dict(x_metric="logistic_regression_MIA.efficacy",
         y_metric="logistic_regression_MIA.f1",
         title="Logistic Regression MIA: attack efficacy vs. F1-score", color_by="method", line=0,
         xlabel = "Efficacy", ylabel = "F1-score"),

    dict(x_metric="logistic_regression_MIA.efficacy",
         y_metric="logistic_regression_MIA.avg_member_probability_forget",
         title="Logistic Regression MIA: attack efficacy vs. average member probability (forget set)", color_by="method", line=0,
         xlabel = "Efficacy", ylabel = "Average member probability (forget set)"),

    dict(x_metric="logistic_regression_MIA.attack_accuracy",
         y_metric="logistic_regression_MIA.auc",
         title="Logistic Regression MIA: attack accuaracy vs. AUC", color_by="method", line=0,
         xlabel = "Accuracy", ylabel = "AUC"),

    dict(x_metric="logistic_regression_MIA.attack_accuracy",
         y_metric="logistic_regression_MIA.f1",
         title="Logistic Regression MIA: attack accuaracy vs. F1-score", color_by="method", line=0,
         xlabel = "Accuracy", ylabel = "F1-score"),

    dict(x_metric="logistic_regression_MIA.attack_accuracy",
         y_metric="logistic_regression_MIA.avg_member_probability_forget",
         title="Logistic Regression MIA: attack accuracy vs. average member probability (forget set)", color_by="method", line=0,
         xlabel = "Accuracy", ylabel = "Average member probability (forget set)"),

    dict(x_metric="logistic_regression_MIA.auc",
         y_metric="logistic_regression_MIA.f1",
         title="Logistic Regression MIA: attack AUC vs. F1-score", color_by="method", line=0,
         xlabel = "AUC", ylabel = "F1-score"),

    dict(x_metric="logistic_regression_MIA.auc",
         y_metric="logistic_regression_MIA.avg_member_probability_forget",
         title="Logistic Regression MIA: attack AUC vs. average member probability (forget set)", color_by="method", line=0,
         xlabel = "AUC", ylabel = "Average member probability (forget set)"),

    dict(x_metric="logistic_regression_MIA.f1",
         y_metric="logistic_regression_MIA.avg_member_probability_forget",
         title="Logistic Regression MIA: attack F1-score vs. average member probability (forget set)", color_by="method", line=0,
         xlabel = "F1-score", ylabel = "Average member probability (forget set)"),





    # ----------------
    # -- RT efficiency 
    # ----------------

    dict(x_metric="total_unlearning_time_up_to_now", 
         y_metric="forget_acc",
         title="Run-time efficiency vs. Forget accuracy", color_by="method", line=0,
         xlabel = "Run-time efficiency", ylabel = "Forget accuracy"),


    # ----------------
    # -- Holistic
    # ----------------
    
    dict(x_metric="ToW",
         y_metric="ToW_MIA",
         title="ToW vs. ToW_MIA", color_by="method", line=0,
         xlabel = "ToW", ylabel = "ToW_MIA"),


    # ----------------------------------------
    # ------------ Performance Differences ---
    # ----------------------------------------


    # ----------------
    # -- (m-)entropy
    # ----------------

    dict(x_metric="forget_entropy", 
         y_metric="forget_m_entropy",
         title="Forget entropy vs. Forget m-entropy", color_by="method", 
         xlabel = "Forget entropy", ylabel = "Forget m-entropy"),

    dict(x_metric="retain_entropy", 
         y_metric="retain_m_entropy",
         title="Retain Entropy vs. Retain M-Entropy", color_by="method",
         xlabel = "Retain entropy", ylabel = "Retain m-entropy"),

    dict(x_metric="forget_entropy", 
         y_metric="retain_entropy",
         title="Forget entropy vs. Retain entropy", color_by="method",
         xlabel = "Forget entropy", ylabel = "Retain entropy"),


    # ----------------
    # -- output diffs and divergences
    # ----------------



    dict(x_metric="outputs.retrain_vs_unlearned.forget.absolute_distance",
         y_metric="outputs.retrain_vs_unlearned.forget.l2_distance",
         title="Absolute distance vs. L2-distance on forget set outputs, retrain vs. unlearned", color_by="method",
         xlabel = "Absolute distance", ylabel = "L2 distance"),

    dict(x_metric="outputs.retrain_vs_unlearned.forget.absolute_distance",
         y_metric="outputs.retrain_vs_unlearned.forget.JS_divergence",
         title="Absolute distance vs. JS divergence on forget set outputs, retrain vs. unlearned", color_by="method",
         xlabel = "Absolute distance", ylabel = "JS divergence"),

    dict(x_metric="outputs.retrain_vs_unlearned.forget.l2_distance",
         y_metric="outputs.retrain_vs_unlearned.forget.JS_divergence",
         title="L2-distance vs. JS divergence on forget set outputs, retrain vs. unlearned", color_by="method",
         xlabel = "L2 distance", ylabel = "JS divergence"),






    dict(x_metric="outputs.retrain_vs_unlearned.forget.JS_divergence",
         y_metric="outputs.bad_teacher_vs_unlearned.forget.ZRF_score",
         title="JS divergence (unlearned vs. retrained) vs. ZRF-score", color_by="method",
         xlabel = "JS divergence", ylabel = "ZRF-score"),





    dict(x_metric="ToW",
         y_metric="outputs.retrain_vs_unlearned.forget_test_avg.KL_divergence",
         title="ToW vs. KL-divergence in retain and forget outputs (averaged)", color_by="method",
         xlabel = "ToW", ylabel = "KL-divergence"),


    dict(x_metric="forgotten_class_fraction",
         y_metric="outputs.retrain_vs_unlearned.forget.prediction_distribution_diff",
         title="Forgotten class share vs. Prediction distribution difference (unlearned vs. retrained)", color_by="method",
         xlabel = "Forgotten class share", ylabel = "Prediction distribution difference"),

    dict(x_metric="outputs.retrain_vs_unlearned.forget.normalized_confusion_distance",
         y_metric="outputs.retrain_vs_unlearned.forget.prediction_distribution_diff",
         title="Normalized confusion distance vs. Prediction distribution difference (unlearned vs. retrained)", color_by="method",
         xlabel = "Normalized confusion distance", ylabel = "Prediction distribution difference"),


    dict(x_metric="forget_acc",
         y_metric="outputs.retrain_vs_unlearned.forget.prediction_distribution_diff",
         title="Forget accuracy vs. Prediction distribution difference (unlearned vs. retrained)", color_by="method",
         xlabel = "Forget accuracy", ylabel = "Prediction distribution difference"),

    dict(x_metric="forget_acc",
         y_metric="outputs.retrain_vs_unlearned.forget.normalized_confusion_distance",
         title="Forget accuracy vs. Normalized confusion distance (unlearned vs. retrained)", color_by="method",
         xlabel = "Forget accuracy", ylabel = "Normalized confusion distance"),






    dict(x_metric="outputs.unlearned.forget_vs_test.wasserstein_distance",
         y_metric="outputs.unlearned.forget_vs_test.ks_statistics",
         title="Wasserstein distance vs. KS statistic on cross-entropy losses, forget vs. test", color_by="method", line=0,
         xlabel = "Wasserstein distance", ylabel = "KS statistic"),





    # ----------------------------------------
    # ---------------------- Relearn Time ---
    # ----------------------------------------


    dict(x_metric="forget_acc",
         y_metric="relearn_time.avg_epochs",
         title="Forget accuracy vs. Relearn time", color_by="method", line=0,
         xlabel = "Forget accuracy", ylabel = "Relearn time"),


    # ----------------
    # -- Weight Differences
    # ----------------

    dict(x_metric="weight_differences.retrain_vs_unlearned.l2_distance",
         y_metric="weight_differences.original_vs_unlearned.l2_distance",
         title="Weight L2-distance: retrain-vs-unlearned vs. original-vs-unlearned", color_by="method",
         xlabel = "retrain-vs-unlearned", ylabel = "original-vs-unlearned"),

    dict(x_metric="forget_acc",
         y_metric="weight_differences.retrain_vs_unlearned.l2_distance",
         title="Forget accuracy vs. weight L2-distance (retrain-vs-unlearned)", color_by="method",
         xlabel = "Forget accuracy", ylabel = "weight L2-distance (retrain-vs-unlearned)"),

    dict(x_metric="forget_acc",
         y_metric="weight_differences.original_vs_unlearned.l2_distance",
         title="Forget accuracy vs. weight L2-distance (original -vs-unlearned)", color_by="method",
         xlabel = "Forget accuracy", ylabel = "weight L2-distance (original-vs-unlearned)"),
]


# ============================================================
# gathering / flattening (from visualize_results.ipynb)
# ============================================================

def flatten_result(d):
    """Flatten nested result dict into a flat dict."""
    flat = {}

    for key in ['type', 'epoch', 'run', 'forget_set_type', 'unlearning_item',
                'method', 'epoch_duration', 'total_unlearning_time_up_to_now', 'ToW',
                'ToW_MIA', 'forgotten_class_fraction']:
        if key in d:
            flat[key] = d[key]

    for metric in ['acc', 'loss', 'entropy', 'm_entropy']:
        if metric in d:
            for split, val in d[metric].items():
                flat[f'{split}_{metric}'] = val

    for block in ['threshold_MIA', 'logistic_regression_MIA', 'outputs',
                  'weight_differences', 'residual_information', 'relearn_time']:
        if block in d:
            block_flat = pd.json_normalize(d[block], sep='.')
            for col in block_flat.columns:
                flat[f'{block}.{col}'] = block_flat[col].iloc[0]

    return flat


def gather_results(results_folder,
                   epochs=None, classes=None, percents=None, runs=None, methods=None,
                   base=True, retrain=True):

    all_files = []
    for root, dirs, files in os.walk(results_folder):
        for fname in files:
            if fname.endswith('.json') and 'config' not in fname:
                all_files.append(os.path.join(root, fname))

    def norm(f):
        return f.replace('\\', '/')

    if not base:
        all_files = [f for f in all_files if '/base/' not in norm(f)]
    if not retrain:
        all_files = [f for f in all_files if '/retrain/' not in norm(f)]

    if classes is not None:
        pattern = rf"class_({'|'.join(map(str, classes))})"
        all_files = [f for f in all_files if re.search(pattern, f)]
    if percents is not None:
        pattern = rf"percent_({'|'.join(map(str, percents))})"
        all_files = [f for f in all_files if re.search(pattern, f)]

    all_rows = []
    for f in all_files:
        with open(f) as fh:
            d = json.load(fh)

        flat = flatten_result(d)

        m = re.search(r'class_(\d+)', f)
        flat['class'] = int(m.group(1)) if m else -99
        m = re.search(r'percent_(\d+)', f)
        flat['percent'] = int(m.group(1)) if m else -99

        result_type = flat.get('type', 'unknown')
        if result_type == 'base':
            flat.setdefault('epoch', -99)
            flat.setdefault('run', -99)
            flat.setdefault('method', 'base')
        elif result_type == 'retrain':
            flat.setdefault('epoch', -98)
            flat.setdefault('method', 'retrain')
            m = re.search(r'retrain_run_(\d+)', norm(f))
            flat.setdefault('run', int(m.group(1)) if m else -99)

        all_rows.append(flat)

    if not all_rows:
        raise ValueError("No results - check settings")

    final_df = pd.DataFrame(all_rows)

    if epochs is not None:
        final_df = final_df[final_df['epoch'].isin(epochs)]
    if runs is not None:
        final_df = final_df[final_df['run'].isin(runs)]
    if methods is not None:
        keep = set(methods) | {'base', 'retrain'}
        final_df = final_df[final_df['method'].isin(keep)]

    if final_df.empty:
        raise ValueError("No results after filtering - check settings")

    return final_df


# ============================================================
# plotting (from visualize_results.ipynb, saving instead of showing)
# ============================================================

METHOD_MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '*', 'p', 'h', '<', '>']

# hard-coded per-method colors so a method's color is stable across plots,
# regardless of which subset of methods happens to be present in a given
# results folder (previously colors were assigned by order-of-appearance, so
# e.g. FT could come out blue in one plot and orange in another). Order here
# just walks the same tab20 palette already in use, in the canonical method
# order from master_hyperparams.py / experiment_config.json, so it reproduces
# the colors already used for FT/GA.
UNLEARNING_METHODS = [
    "FT", "GA", "NegGrad_plus", "RL", "boundary_shrink",
    "bad_teacher", "scrub", "UNSIR",
]
METHOD_COLORS = {
    method: plt.cm.tab20.colors[i % len(plt.cm.tab20.colors)]
    for i, method in enumerate(UNLEARNING_METHODS)
}

# same idea as METHOD_COLORS, but for scatterplot marker shapes -- walks the
# same METHOD_MARKERS list already in use, in the same canonical order, so a
# method's marker shape (not just its color) is stable across plots too
METHOD_MARKER_SHAPES = {
    method: METHOD_MARKERS[i % len(METHOD_MARKERS)]
    for i, method in enumerate(UNLEARNING_METHODS)
}


def _method_marker(method_name):
    """
    Stable marker shape for an unlearning method: from the hard-coded mapping
    above if known, otherwise a deterministic (hash-based, not order-based)
    fallback from the same marker list.
    """
    if method_name in METHOD_MARKER_SHAPES:
        return METHOD_MARKER_SHAPES[method_name]
    idx = int(hashlib.md5(str(method_name).encode()).hexdigest(), 16) % len(METHOD_MARKERS)
    return METHOD_MARKERS[idx]


def _method_color(method_name):
    """
    Stable color for an unlearning method: from the hard-coded palette above
    if known, otherwise a deterministic (hash-based, not order-based) fallback
    from the same colormap, so an unrecognized method still gets a color
    that's consistent across runs rather than shifting with plot order.
    """
    if method_name in METHOD_COLORS:
        return METHOD_COLORS[method_name]
    colors_list = plt.cm.tab20.colors
    idx = int(hashlib.md5(str(method_name).encode()).hexdigest(), 16) % len(colors_list)
    return colors_list[idx]


def _log_transform(final_df, cols):
    final_df = final_df.copy()
    for col in cols:
        n_non_positive = int((final_df[col] <= 0).sum())
        if n_non_positive:
            print(f"log_scale: {n_non_positive} non-positive value(s) in '{col}' set to NaN before taking log.")
        final_df[col] = np.log(final_df[col].where(final_df[col] > 0))
    return final_df


def _save_or_show(fig, save_path):
    if save_path is None:
        plt.show()
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def results_scatterplot(final_df, x_metric, y_metric,
                        color_by="method", marker_by="method",
                        base_color="black", retain_color="purple",
                        base_marker="o", retain_marker="X",
                        title=None, xlabel=None, ylabel=None,
                        grouped=True, line=0, log_scale=False,
                        save_path=None):
    """
    line: 0 for no reference line, 1 to draw y = x, -1 to draw y = 1 - x.
    log_scale: if True, natural-log-transform x_metric and y_metric before
    plotting (non-positive values are dropped, not just relabeled on a log axis).
    save_path: if given, the figure is saved there instead of shown.
    """

    if log_scale:
        final_df = _log_transform(final_df, (x_metric, y_metric))

    plot_df = final_df
    x_err_col = y_err_col = None

    if grouped:
        group_cols = list(dict.fromkeys(["method", "epoch", color_by, marker_by]))
        agg = final_df.groupby(group_cols, dropna=False)[[x_metric, y_metric]].agg(["mean", "std"])
        agg.columns = ["_".join(c) for c in agg.columns]
        plot_df = agg.reset_index()
        x_err_col, y_err_col = f"{x_metric}_std", f"{y_metric}_std"
        plot_df = plot_df.rename(columns={f"{x_metric}_mean": x_metric, f"{y_metric}_mean": y_metric})

    unique_vals = sorted(plot_df[color_by].unique(), key=str)
    colors_list = plt.cm.tab20.colors
    palette_dict = {}
    color_idx = 0
    for val in unique_vals:
        if val == -99 or val == "base":
            palette_dict[val] = base_color
        elif val == -98 or val == "retrain":
            palette_dict[val] = retain_color
        elif color_by == "method":
            palette_dict[val] = _method_color(val)
        else:
            palette_dict[val] = colors_list[color_idx % len(colors_list)]
            color_idx += 1

    unique_markers = sorted(plot_df[marker_by].unique(), key=str)
    marker_dict = {}
    marker_idx = 0
    for val in unique_markers:
        if val == -99 or val == "base":
            marker_dict[val] = base_marker
        elif val == -98 or val == "retrain":
            marker_dict[val] = retain_marker
        elif marker_by == "method":
            marker_dict[val] = _method_marker(val)
        else:
            marker_dict[val] = METHOD_MARKERS[marker_idx % len(METHOD_MARKERS)]
            marker_idx += 1

    plt.clf()
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.35)
    fig = plt.gcf()
    ax = plt.gca()

    if grouped:
        for _, row in plot_df.iterrows():
            color = palette_dict.get(row[color_by], "gray")
            xerr = row[x_err_col] if pd.notna(row[x_err_col]) else 0
            yerr = row[y_err_col] if pd.notna(row[y_err_col]) else 0
            ax.errorbar(
                row[x_metric], row[y_metric],
                xerr=xerr, yerr=yerr,
                fmt="none", ecolor=color, elinewidth=1.2, capsize=3, alpha=0.6, zorder=1
            )

    sns.scatterplot(
        data=plot_df,
        x=x_metric,
        y=y_metric,
        hue=color_by,
        style=marker_by,
        palette=palette_dict,
        markers=marker_dict,
        s=80,
        alpha=0.7,
        edgecolor=None,
        ax=ax,
        zorder=2,
        legend=False
    )
    ax.grid(True, alpha=0.3)
    if line in (1, -1):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
        if line == 1:
            y_lo, y_hi = lo, hi
        else:
            y_lo, y_hi = 1 - lo, 1 - hi
        ax.plot([lo, hi], [y_lo, y_hi], linestyle="--", color="grey", linewidth=1, zorder=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    default_xlabel = ("ln " if log_scale else "") + x_metric.capitalize().replace("_", " ")
    default_ylabel = ("ln " if log_scale else "") + y_metric.capitalize().replace("_", " ")
    plt.xlabel(xlabel if xlabel is not None else default_xlabel)
    plt.ylabel(ylabel if ylabel is not None else default_ylabel)
    _save_or_show(fig, save_path)


def results_barplot(final_df, metric,
                    title=None, ylabel=None,
                    base_color="black", retain_color="purple",
                    log_scale=False,
                    save_path=None):
    """
    Bar chart of a single metric, one bar per (method, epoch) group, with the
    mean and std across runs computed within each group and printed above the bar.

    log_scale: if True, natural-log-transform `metric` before plotting
    (non-positive values are dropped, not just relabeled on a log axis).
    save_path: if given, the figure is saved there instead of shown.
    """
    if log_scale:
        final_df = _log_transform(final_df, (metric,))

    agg = final_df.groupby(["method", "epoch"], dropna=False)[metric].agg(["mean", "std"]).reset_index()

    def sort_key(row):
        if row["method"] == "base":
            return (-2, "", row["epoch"])
        if row["method"] == "retrain":
            return (-1, "", row["epoch"])
        return (0, row["method"], row["epoch"])
    agg = agg.loc[agg.apply(sort_key, axis=1).sort_values().index].reset_index(drop=True)

    def label_for(row):
        if row["method"] in ("base", "retrain"):
            return row["method"]
        return f"{row['method']}\n(epoch {int(row['epoch'])})"
    agg["label"] = agg.apply(label_for, axis=1)

    def color_for(row):
        if row["method"] == "base":
            return base_color
        if row["method"] == "retrain":
            return retain_color
        return _method_color(row["method"])
    bar_colors = [color_for(row) for _, row in agg.iterrows()]

    plt.clf()
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.35)
    fig, ax = plt.subplots(figsize=(max(8, len(agg) * 1.1), 6))

    x = list(range(len(agg)))
    means = agg["mean"].tolist()
    stds = agg["std"].fillna(0).tolist()

    bars = ax.bar(x, means, color=bar_colors, alpha=0.85, yerr=stds, capsize=4,
                  error_kw=dict(elinewidth=1.2, ecolor="#333"))

    for bar, mean, std in zip(bars, means, stds):
        label_text = f"{mean:.3f}\n(±{std:.3f})" if std else f"{mean:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std,
            label_text,
            ha="center", va="bottom", fontsize=11
        )

    ax.set_xticks(x)
    ax.set_xticklabels(agg["label"], rotation=30 if len(agg) > 6 else 0, ha="right" if len(agg) > 6 else "center")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ============================================================
# driving logic
# ============================================================

def _safe_filename(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def discover_barplot_metrics(reference_file, exclude=NON_METRIC_COLUMNS):
    """
    Flatten `reference_file` (a typical results.json) and return every scalar
    metric name in it -- i.e. every barplot we'd want by default.
    """
    with open(reference_file) as fh:
        d = json.load(fh)
    flat = flatten_result(d)
    metrics = []
    for key, val in flat.items():
        if key in exclude:
            continue
        if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
            metrics.append(key)
    return sorted(metrics)


def run_barplots(results, plots_folder):
    metrics = discover_barplot_metrics(REFERENCE_RESULT_FILE)
    print(f"[master_results] discovered {len(metrics)} metrics to barplot from {REFERENCE_RESULT_FILE}")
    out_folder = os.path.join(plots_folder, "barplots")
    for metric in metrics:
        save_path = os.path.join(out_folder, f"{_safe_filename(metric)}.png")
        try:
            results_barplot(
                results,
                metric,
                log_scale=metric.startswith("residual_information."),
                save_path=save_path,
            )
        except Exception as e:
            print(f"[master_results] skipped barplot for '{metric}': {e}")


def run_scatterplots(results, plots_folder):
    out_folder = os.path.join(plots_folder, "scatterplots")
    for pair in SCATTERPLOT_PAIRS:
        pair = dict(pair)
        x_metric = pair.pop("x_metric")
        y_metric = pair.pop("y_metric")
        for grouped in (True, False):
            subfolder = "grouped" if grouped else "ungrouped"
            save_path = os.path.join(
                out_folder, subfolder,
                f"{_safe_filename(x_metric)}__vs__{_safe_filename(y_metric)}.png"
            )
            try:
                results_scatterplot(
                    results,
                    x_metric=x_metric,
                    y_metric=y_metric,
                    grouped=grouped,
                    save_path=save_path,
                    **pair,
                )
            except Exception as e:
                print(f"[master_results] skipped scatterplot for '{x_metric}' vs '{y_metric}' (grouped={grouped}): {e}")


def _method_legend_entries(base_color, retain_color, base_marker, retain_marker):
    """(label, color, marker) triples for base, retrain, then every UNLEARNING_METHODS entry."""
    entries = [("base", base_color, base_marker), ("retrain", retain_color, retain_marker)]
    entries += [(method, _method_color(method), _method_marker(method)) for method in UNLEARNING_METHODS]
    return entries


def plot_method_legend_vertical(plots_folder,
                                base_color="black", retain_color="purple",
                                base_marker="o", retain_marker="X"):
    """
    Standalone reference plot (no data) showing the hard-coded color + marker
    for base, retrain, and every unlearning method in UNLEARNING_METHODS,
    stacked in one column -- a shared key for reading any of the
    scatterplots/barplots above.
    """
    entries = _method_legend_entries(base_color, retain_color, base_marker, retain_marker)

    handles = [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor=color,
              markeredgecolor=color, markersize=11, linestyle="none", label=label)
        for label, color, marker in entries
    ]

    plt.clf()
    fig, ax = plt.subplots(figsize=(3, 0.4 * len(entries) + 0.5))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", frameon=False, title="Method", fontsize=11, title_fontsize=12)
    save_path = os.path.join(plots_folder, "method_legend_vertical.png")
    _save_or_show(fig, save_path)


def plot_method_legend_horizontal(plots_folder,
                                  base_color="black", retain_color="purple",
                                  base_marker="o", retain_marker="X",
                                  width_in=None):
    """
    Same reference legend as plot_method_legend_vertical, but laid out to
    read left-to-right, wrapping onto a new row only if it runs out of
    `width_in` inches of horizontal space, rather than stacking in one
    column. Entry widths are measured from actual rendered text extents (not
    guessed from character count), so wrapping lands in the right place
    regardless of font/label length.

    width_in: if None (default), the figure is sized to fit every entry on a
    single row; pass an explicit width to force wrapping within that budget.
    """
    entries = _method_legend_entries(base_color, retain_color, base_marker, retain_marker)

    fontsize = 11
    marker_w_in = 0.16      # approx. on-canvas footprint of a marker, for spacing purposes
    marker_text_gap = 0.08  # gap between a marker and its label
    entry_gap = 0.35        # gap between one entry's label and the next entry's marker
    row_height = 0.36
    pad = 0.15

    # throwaway figure/axes purely to get a renderer to measure text extents with
    fig = plt.figure(figsize=(width_in or 6.0, 2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    renderer = fig.canvas.get_renderer()

    measured = []
    for label, color, marker in entries:
        t = ax.text(0, 0, label, fontsize=fontsize)
        label_w_in = t.get_window_extent(renderer=renderer).width / fig.dpi
        t.remove()
        measured.append((label, color, marker, label_w_in))

    if width_in is None:
        # fit everything on one row: total content width plus padding on both sides
        entry_widths = [marker_w_in + marker_text_gap + label_w_in for (_, _, _, label_w_in) in measured]
        width_in = 2 * pad + sum(entry_widths) + entry_gap * (len(measured) - 1)

    # greedily wrap entries into rows that each fit within width_in
    rows, current_row, current_width = [], [], pad
    for entry in measured:
        _, _, _, label_w_in = entry
        entry_w_in = marker_w_in + marker_text_gap + label_w_in
        added_w_in = entry_w_in + (entry_gap if current_row else 0)
        if current_row and current_width + added_w_in > width_in - pad:
            rows.append(current_row)
            current_row, current_width = [], pad
            added_w_in = entry_w_in
        current_row.append(entry)
        current_width += added_w_in
    if current_row:
        rows.append(current_row)

    fig_height = 2 * pad + len(rows) * row_height
    fig.set_size_inches(width_in, fig_height)
    ax.clear()
    ax.axis("off")
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, fig_height)
    ax.invert_yaxis()

    for row_idx, row in enumerate(rows):
        y = pad + row_idx * row_height + row_height / 2
        x = pad
        for label, color, marker, label_w_in in row:
            ax.scatter([x + marker_w_in / 2], [y], marker=marker, color=color, s=90, clip_on=False)
            x += marker_w_in + marker_text_gap
            ax.text(x, y, label, fontsize=fontsize, va="center", ha="left")
            x += label_w_in + entry_gap

    save_path = os.path.join(plots_folder, "method_legend_horizontal.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"[master_results] gathering results from {RESULTS_FOLDER}")
    results = gather_results(
        RESULTS_FOLDER,
        epochs=EPOCHS, classes=CLASSES, percents=PERCENTS, runs=RUNS, methods=METHODS,
        base=BASE, retrain=RETRAIN,
    )
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    run_barplots(results, PLOTS_FOLDER)
    run_scatterplots(results, PLOTS_FOLDER)
    plot_method_legend_vertical(PLOTS_FOLDER)
    plot_method_legend_horizontal(PLOTS_FOLDER)
    print(f"[master_results] done. plots saved under {PLOTS_FOLDER}")


if __name__ == "__main__":
    main()
