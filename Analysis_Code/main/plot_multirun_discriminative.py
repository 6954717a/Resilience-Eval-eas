"""
plot_multirun_discriminative.py
===============================
Core discriminative analysis: show that resilience metrics reveal properties
hidden by outcome-centric metrics (SR, task_percent_complete).

Generates 2 figures:
  - Figure MR-1: Discriminative Evidence (3-panel)
  - Figure MR-2: Cross-Model Comparison (3-panel)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from expq1_multirun_loader import load_all_rebound, load_all_stability, IMG_DIR, TABLE_DIR
from plot_style import setup_style, add_panel_label, save_figure, PALETTE, PAL_3TYPE, PAL_FAMILY

MODEL_ORDER = ["Qwen3-8B", "Qwen3.5-9B", "Qwen2.5-3B-r1", "Qwen2.5-3B-r2"]
MODEL_COLORS = {
    "Qwen3-8B": "#2E86AB",
    "Qwen3.5-9B": "#A23B72",
    "Qwen2.5-3B-r1": "#2CA58D",
    "Qwen2.5-3B-r2": "#F18F01",
}


def fig_mr1_discriminative():
    """Figure MR-1: Discriminative Evidence (3-panel)."""
    setup_style()
    df = load_all_rebound()
    success = df[df["is_success"] == 1].copy()

    # Increase figsize slightly to avoid "flatness"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    # (a) SR=100% but different C_rec: clean vs perturbed
    ax = axes[0]
    clean_s = success[success["is_perturbed"] == 0]["c_rec"]
    pert_s = success[success["is_perturbed"] == 1]["c_rec"]

    data_a = pd.DataFrame({
        "Recovery Cost $C_{rec}$": list(clean_s) + list(pert_s),
        "Condition": ["Clean"] * len(clean_s) + ["Perturbed"] * len(pert_s),
    })
    colors_a = [PALETTE["success"], PALETTE["danger"]]
    vp = sns.violinplot(data=data_a, x="Condition", y="Recovery Cost $C_{rec}$",
                        hue="Condition", palette=colors_a, inner="quartile", 
                        alpha=0.35, ax=ax, cut=0, legend=False)
    sns.stripplot(data=data_a, x="Condition", y="Recovery Cost $C_{rec}$",
                  hue="Condition", palette=colors_a, alpha=0.6, size=5, jitter=0.2, ax=ax, legend=False)
    
    ax.set_yscale("symlog", linthresh=10)

    # Means and data counts (supplementary info)
    for i, (label, vals) in enumerate([("Clean", clean_s), ("Perturbed", pert_s)]):
        m = vals.mean()
        n = len(vals)
        ax.scatter(i, m, marker="D", s=80, color="white", edgecolors=PALETTE["dark_text"],
                   linewidths=1.5, zorder=5)
        ax.text(i + 0.18, m, f"$\\mu$={m:.1f}\n$n$={n}", fontsize=11, va="center",
                color=PALETTE["dark_text"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"))

    ax.text(0.5, -0.22, "(a) Same SR=100%,\nDifferent Recovery Cost", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])

    # (b) C_rec distribution within SR=100% by perturbation type
    ax = axes[1]
    pt_order = ["clean", "surface_rewrite", "object_state_toggle", "filler_injection"]
    pt_labels = ["Clean", "Surface\nRewrite", "Object State\nToggle", "Filler\nInjection"]
    avail = [p for p in pt_order if p in success["perturbation_type"].unique()]
    avail_labels = [pt_labels[pt_order.index(p)] for p in avail]

    pal_b = [PALETTE["success"], PALETTE["accent"], PALETTE["danger"], PALETTE["secondary"]]
    avail_pal = [pal_b[pt_order.index(p)] for p in avail]

    bp = sns.boxplot(data=success, x="perturbation_type", y="c_rec",
                     order=avail, hue="perturbation_type", palette=avail_pal,
                     legend=False, width=0.55, linewidth=1.5, fliersize=0,
                     boxprops=dict(alpha=0.6), ax=ax)
    sns.stripplot(data=success, x="perturbation_type", y="c_rec",
                  order=avail, hue="perturbation_type", palette=avail_pal,
                  alpha=0.6, size=4.5, jitter=0.18, ax=ax, legend=False)

    ax.set_yscale("symlog", linthresh=10)

    # Adding supplementary info (n count) for each boxplot
    for i, p_type in enumerate(avail):
        vals = success[success["perturbation_type"] == p_type]["c_rec"]
        m = vals.mean()
        n = len(vals)
        ax.scatter(i, m, marker="D", s=70, color="white", edgecolors=PALETTE["dark_text"],
                   linewidths=1.5, zorder=5)
        ax.text(i + 0.25, m, f"$\\mu$={m:.1f}\n$n$={n}", fontsize=10, va="center", ha="left",
                color=PALETTE["dark_text"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"), zorder=6)

    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail_labels)
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel("$C_{rec}$ (SR=100% episodes)")
    ax.text(0.5, -0.22, "(b) Recovery Cost by\nPerturbation Type", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])

    # (c) Scatter: task completion vs C_rec (all episodes) with SR=100% highlighted
    ax = axes[2]
    non_s = df[df["is_success"] == 0]
    
    # KDE for incomplete (2D density) to address sparsity
    sns.kdeplot(data=non_s, x="task_percent_complete", y="c_rec", 
                ax=ax, color=PALETTE["neutral"], fill=True, alpha=0.2, levels=4, thresh=0.05, zorder=1)
                
    # Add trendline for incomplete
    sns.regplot(data=non_s, x="task_percent_complete", y="c_rec", 
                ax=ax, scatter=False, color=PALETTE["neutral"], 
                line_kws={"linestyle": "--", "alpha": 0.7, "linewidth": 1.5, "zorder": 2})

    # Jitter for success points X to avoid complete overlap since they all equal 1.0
    success_x_jittered = success["task_percent_complete"] + np.random.uniform(-0.015, 0.015, size=len(success))

    ax.scatter(non_s["task_percent_complete"], non_s["c_rec"],
               s=40, alpha=0.65, color=PALETTE["neutral"], label="Incomplete", zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.scatter(success_x_jittered, success["c_rec"],
               s=50, alpha=0.75, color=PALETTE["danger"], label="SR=100%", zorder=4,
               edgecolors="white", linewidths=0.4)

    ax.axvline(1.0, color=PALETTE["neutral"], linestyle=":", alpha=0.5, zorder=0)
    # Mark the SR=100% spread
    y_lo, y_hi = success["c_rec"].quantile(0.05), success["c_rec"].quantile(0.95)
    ax.annotate("", xy=(1.04, y_lo), xytext=(1.04, y_hi),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["danger"], lw=2))
    ax.text(1.07, (y_lo + y_hi) / 2, f"CV={success['c_rec'].std()/success['c_rec'].mean()*100:.0f}%\n$n$={len(success)}",
            fontsize=9.5, color=PALETTE["danger"], va="center", fontweight="bold")
    ax.set_xlabel("Task Completion (%)")
    ax.set_ylabel("Recovery Cost $C_{rec}$")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.text(0.5, -0.22, "(c) Completion vs. Recovery Cost\n(All Episodes)", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])

    fig.subplots_adjust(bottom=0.22)
    save_figure(fig, IMG_DIR / "fig_mr1_discriminative_evidence.png")


def fig_mr2_cross_model():
    """Figure MR-2: Cross-Model Comparison (3-panel)."""
    setup_style()
    df_reb = load_all_rebound()
    df_stab = load_all_stability()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    models_avail = [m for m in MODEL_ORDER if m in df_reb["model"].unique()]

    # (a) C_rec by model (SR=100% only)
    ax = axes[0]
    success = df_reb[df_reb["is_success"] == 1]
    pal_a = [MODEL_COLORS.get(m, PALETTE["neutral"]) for m in models_avail]

    bp = sns.boxplot(data=success, x="model", y="c_rec",
                     order=models_avail, hue="model", palette=pal_a,
                     legend=False, width=0.55, linewidth=1.5, fliersize=0,
                     boxprops=dict(alpha=0.6), ax=ax)
    sns.stripplot(data=success, x="model", y="c_rec",
                  order=models_avail, hue="model", palette=pal_a,
                  alpha=0.6, size=4.5, jitter=0.18, ax=ax, legend=False)
    
    ax.set_yscale("symlog", linthresh=10)

    # Annotate means and N
    for i, m in enumerate(models_avail):
        sub = success[success["model"] == m]["c_rec"]
        m_val = sub.mean()
        n_val = len(sub)
        ax.scatter(i, m_val, marker="D", s=70, color="white", edgecolors=PALETTE["dark_text"],
                   linewidths=1.5, zorder=5)
        ax.text(i + 0.25, m_val, f"$\\mu$={m_val:.1f}\n$n$={n_val}", va="center", ha="left",
                fontsize=10, color=PALETTE["dark_text"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"), zorder=6)

    ax.set_xlabel("")
    ax.set_ylabel("$C_{rec}$ (SR=100%)")
    ax.tick_params(axis="x", rotation=15)
    ax.text(0.5, -0.22, "(a) Recovery Cost by Model\n(SR=100% episodes)", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])

    # (b) Clean vs Perturbed C_rec per model (grouped bar)
    ax = axes[1]
    bar_data = []
    for m in models_avail:
        sub = df_reb[df_reb["model"] == m]
        cl = sub[sub["is_perturbed"] == 0]["c_rec"]
        pt = sub[sub["is_perturbed"] == 1]["c_rec"]
        bar_data.append({"model": m,
                         "Clean": cl.mean() if len(cl) > 0 else np.nan,
                         "Perturbed": pt.mean() if len(pt) > 0 else np.nan,
                         "n_clean": len(cl), "n_pert": len(pt)})

    df_bar = pd.DataFrame(bar_data)
    x = np.arange(len(df_bar))
    w = 0.32
    for j, (cond, color) in enumerate([("Clean", PALETTE["success"]),
                                        ("Perturbed", PALETTE["danger"])]):
        vals = df_bar[cond].values
        offset = (j - 0.5) * w
        bars = ax.bar(x + offset, vals, w * 0.9, color=color, alpha=1.0,
                      edgecolor="white", linewidth=0.8, label=cond)
        for k, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(x[k] + offset, v + 1, f"{v:.1f}", ha="center",
                        va="bottom", fontsize=11, fontweight="bold",
                        color=PALETTE["dark_text"])

    ax.set_xticks(x)
    ax.set_xticklabels(models_avail, rotation=15)
    ax.set_ylabel("Mean $C_{rec}$")
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.text(0.5, -0.22, "(b) Clean vs. Perturbed\n$C_{rec}$ by Model", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])

    # (c) Stability beta by model
    ax = axes[2]
    stab_models = [m for m in MODEL_ORDER if m in df_stab["model"].unique()]
    pal_c = [MODEL_COLORS.get(m, PALETTE["neutral"]) for m in stab_models]

    if len(stab_models) > 0:
        bp = sns.boxplot(data=df_stab, x="model", y="beta_neigh",
                         order=stab_models, hue="model", palette=pal_c,
                         legend=False, width=0.55, linewidth=1.5, fliersize=0,
                         boxprops=dict(alpha=0.6), ax=ax)
        sns.stripplot(data=df_stab, x="model", y="beta_neigh",
                      order=stab_models, hue="model", palette=pal_c,
                      alpha=0.6, size=4.5, jitter=0.18, ax=ax, legend=False)
        for i, m in enumerate(stab_models):
            sub = df_stab[df_stab["model"] == m]["beta_neigh"]
            m_val = sub.mean()
            n_val = len(sub)
            ax.scatter(i, m_val, marker="D", s=70, color="white", edgecolors=PALETTE["dark_text"],
                       linewidths=1.5, zorder=5)
            ax.text(i + 0.25, m_val, f"$\\mu$={m_val:.2f}\n$n$={n_val}", va="center", ha="left",
                    fontsize=10, color=PALETTE["dark_text"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"), zorder=6)

    ax.set_xlabel("")
    ax.set_ylabel("$\\hat{\\beta}_{neighborhood}$")
    ax.tick_params(axis="x", rotation=15)
    ax.text(0.5, -0.22, "(c) Stability ($\\beta$) by Model", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])

    fig.subplots_adjust(bottom=0.25)
    save_figure(fig, IMG_DIR / "fig_mr2_cross_model_comparison.png")


def export_discriminative_table():
    """Export key discriminative statistics."""
    df = load_all_rebound()
    success = df[df["is_success"] == 1]
    clean_s = success[success["is_perturbed"] == 0]["c_rec"]
    pert_s = success[success["is_perturbed"] == 1]["c_rec"]
    u, p_mw = stats.mannwhitneyu(clean_s, pert_s, alternative="two-sided")
    cd = (u / (len(clean_s) * len(pert_s))) * 2 - 1

    rows = []
    rows.append({"analysis": "SR100_clean_vs_perturbed",
                 "n_clean": len(clean_s), "n_perturbed": len(pert_s),
                 "mean_clean": clean_s.mean(), "mean_perturbed": pert_s.mean(),
                 "ratio": pert_s.mean() / (clean_s.mean() + 1e-9),
                 "mann_whitney_U": u, "p_value": p_mw, "cliffs_delta": cd,
                 "cv_overall_pct": success["c_rec"].std() / success["c_rec"].mean() * 100})

    # Per-model stats
    for model in sorted(success["model"].unique()):
        sub = success[success["model"] == model]
        rows.append({"analysis": f"SR100_model_{model}",
                     "n_clean": len(sub[sub["is_perturbed"]==0]),
                     "n_perturbed": len(sub[sub["is_perturbed"]==1]),
                     "mean_clean": sub[sub["is_perturbed"]==0]["c_rec"].mean(),
                     "mean_perturbed": sub[sub["is_perturbed"]==1]["c_rec"].mean(),
                     "ratio": np.nan, "mann_whitney_U": np.nan,
                     "p_value": np.nan, "cliffs_delta": np.nan,
                     "cv_overall_pct": sub["c_rec"].std() / (sub["c_rec"].mean() + 1e-9) * 100})

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "expq1_discriminative_stats.csv", index=False)
    print(f"  [export] {TABLE_DIR / 'expq1_discriminative_stats.csv'}")
    return out


if __name__ == "__main__":
    print("=== Figure MR-1: Discriminative Evidence ===")
    fig_mr1_discriminative()
    print("=== Figure MR-2: Cross-Model Comparison ===")
    fig_mr2_cross_model()
    print("=== Discriminative Stats Table ===")
    tbl = export_discriminative_table()
    print(tbl.to_string())
