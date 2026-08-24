"""
plot_fig_q1_4_ge.py
===================
Figure Q1-4: Graceful Extensibility Validity Under Stress Sweep (3-panel)

(a) λ vs ΔM (Contract Margin change relative to clean baseline)
(b) λ vs Boundary-Hit Rate (Direct evidence of approaching boundary)
(c) Per-task-family margin degradation heatmap (Heterogeneity)

Reference: Requirements.md §Figure Q1-G / Q1-4
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from expq1_data_loader import load_ge, aggregate_ge_stress_sweep, aggregate_ge_by_family, IMG_DIR
from plot_style import setup_style, add_panel_label, save_figure, PALETTE, PAL_FAMILY

def fig_q1_4_ge_stress_response():
    """Generate Figure Q1-4 for GE validity under stress sweep."""
    setup_style()

    df = load_ge()
    
    # Filter for supplementary dataset with valid margins
    df_margin = df[(df["ge_contract_margin_min"] != -1.0) & (df["ge_contract_margin_min"] != 0.0)].copy()
    
    if len(df_margin) > 0:
        agg_sweep = aggregate_ge_stress_sweep(df_margin)
        agg_fam = aggregate_ge_by_family(df_margin)
        clean_ref = df_margin[df_margin["phase_label"] == "clean_reference"].copy()
    else:
        # Fallback if somehow no margin data is loaded
        agg_sweep = aggregate_ge_stress_sweep(df)
        agg_fam = aggregate_ge_by_family(df)
        clean_ref = df[df["phase_label"] == "clean_reference"].copy()

    clean_mean_margin = clean_ref["ge_contract_margin_min"].mean() if "ge_contract_margin_min" in clean_ref else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # ── Panel (a): λ vs ΔM (Delta Margin) ────────
    ax = axes[0]

    lambdas = agg_sweep["stress_lambda"].values
    margins = agg_sweep["margin_mean"].values
    margins_se = agg_sweep["margin_se"].values
    
    # Calculate Delta M
    delta_margins = margins - clean_mean_margin

    # Plot with gradient colors by stress level
    ax.fill_between(lambdas, delta_margins - margins_se, delta_margins + margins_se,
                    alpha=0.2, color=PALETTE["primary"])
    ax.plot(lambdas, delta_margins, "o-", color=PALETTE["primary"],
            linewidth=2, markersize=8, markeredgecolor="white",
            markeredgewidth=1.5, label="Stress Sweep", zorder=3)

    # Clean reference is 0
    ax.axhline(0, color=PALETTE["success"], linestyle="--",
               linewidth=1.5, alpha=0.7, label="Clean Ref. (ΔM=0)")

    # Spearman trend
    if len(lambdas) >= 3:
        rho, p = stats.spearmanr(lambdas, delta_margins)
        ax.text(0.05, 0.05, f"Spearman ρ = {rho:.2f}\np = {p:.3f}",
                transform=ax.transAxes, fontsize=9.5, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_xlabel("Stress Severity λ")
    ax.set_ylabel("Δ Margin $\\Delta M$ (Mean ± SE)")
    ax.text(0.5, -0.22, "(a) Degradation from\nClean Baseline", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)
    
    # ── Panel (b): λ vs Recovery Cost (Supporting Evidence) ────────────────────────────────────
    ax = axes[1]
    
    crec_mean = agg_sweep["c_rec_mean"].values
    crec_se = agg_sweep["c_rec_se"].values
    
    # Filter out lambda = 0 for Panel (b)
    mask_b = lambdas > 0
    lam_b = lambdas[mask_b]
    crec_mean_b = crec_mean[mask_b]
    crec_se_b = crec_se[mask_b]
    
    ax.fill_between(lam_b, crec_mean_b - crec_se_b, crec_mean_b + crec_se_b,
                    alpha=0.2, color=PALETTE["secondary"])
    ax.plot(lam_b, crec_mean_b, "s-", color=PALETTE["secondary"],
            linewidth=2, markersize=8, markeredgecolor="white",
            markeredgewidth=1.5, label="Recovery Cost $C_{rec}$", zorder=3)
           
    if len(lam_b) >= 3:
        rho, p = stats.spearmanr(lam_b, crec_mean_b)
        ax.text(0.05, 0.95, f"Spearman ρ = {rho:.2f}\np = {p:.3f}",
                transform=ax.transAxes, fontsize=9.5, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_xlabel("Stress Severity λ")
    ax.set_ylabel("Recovery Cost $C_{rec}$ (Mean ± SE)")
    ax.text(0.5, -0.22, "(b) Recovery Burden as\nSupporting Evidence", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.set_xlim(0.15, 1.05)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # ── Panel (c): Per-task-family degradation Heatmap ───────────────────────────
    ax = axes[2]

    families = sorted(agg_fam["task_family"].unique())
    lambdas_fam = [0.0] + sorted(agg_fam["stress_lambda"].unique())
    
    # Calculate clean reference margins per family
    clean_fam_margins = {}
    if len(clean_ref) > 0:
        clean_fam = clean_ref.groupby("task_family").agg(margin_mean=("ge_contract_margin_min", "mean"))
        clean_fam_margins = clean_fam["margin_mean"].to_dict()

    heatmap_data = np.zeros((len(families), len(lambdas_fam)))
    for i, fam in enumerate(families):
        for j, lam in enumerate(lambdas_fam):
            if lam == 0.0:
                heatmap_data[i, j] = clean_fam_margins.get(fam, np.nan)
            else:
                val = agg_fam[(agg_fam["task_family"] == fam) & (agg_fam["stress_lambda"] == lam)]["margin_mean"]
                heatmap_data[i, j] = val.iloc[0] if len(val) > 0 else np.nan

    fam_labels = [f.replace("_", " ").title() for f in families]
    lam_labels = [f"{l:.1f}" for l in lambdas_fam]
    
    sns.heatmap(heatmap_data, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f",
                xticklabels=lam_labels, yticklabels=fam_labels,
                cbar_kws={'label': 'Contract Margin $M$'},
                linewidths=1, linecolor='white')

    ax.set_xlabel("Stress Severity λ")
    ax.set_ylabel("Task Family")
    ax.text(0.5, -0.22, "(c) Family-Specific\nBrittleness Heatmap", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])
    
    fig.subplots_adjust(bottom=0.25)
    fig.tight_layout()
    save_figure(fig, IMG_DIR / "fig_q1_4_ge_stress_response.png")

if __name__ == "__main__":
    fig_q1_4_ge_stress_response()
    print("Figure Q1-4 GE Validity generated successfully.")
