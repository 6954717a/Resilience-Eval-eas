"""
plot_fig_q1_3_stability.py
==========================
Figure Q1-3: Stability Validity Under Semantic-Preserving Perturbations (2-panel)

(a) β_neighborhood under original, surface rewrites, and filler injections.
(b) Decomposition into value variation β_vv and output variation β_out.

Reference: Requirements.md §Figure Q1-S / Q1-3
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from expq1_data_loader import load_stability, IMG_DIR, TABLE_DIR
from plot_style import (setup_style, add_panel_label, save_figure,
                        PALETTE, PAL_3TYPE)


def plot_fig_q1_3():
    setup_style()

    df = load_stability()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    ptype_order = ["clean", "surface_rewrite", "filler_injection"]
    ptype_labels = ["Original\n(Intrinsic Var.)", "Surface\nRewrite", "Filler\nInjection"]

    # ── Panel (a): β_neighborhood boxplot ────────────────────────────────
    ax = axes[0]

    bp = sns.boxplot(data=df, x="perturbation_type", y="beta_neigh",
                     order=ptype_order, palette=PAL_3TYPE,
                     width=0.5, linewidth=1.3, fliersize=5,
                     boxprops=dict(alpha=0.75), ax=ax)
    sns.stripplot(data=df, x="perturbation_type", y="beta_neigh",
                  order=ptype_order, color=PALETTE["dark_text"],
                  alpha=0.45, size=6, jitter=0.12, ax=ax)

    # Mean markers
    means = df.groupby("perturbation_type")["beta_neigh"].mean()
    for i, pt in enumerate(ptype_order):
        if pt in means.index:
            ax.scatter(i, means[pt], marker="D", color="white",
                       edgecolors=PALETTE["dark_text"], s=50, zorder=5, linewidths=1.2)
            ax.text(i - 0.47, means[pt], f"μ={means[pt]:.3f}",
                    fontsize=11, va="center", ha="left", color=PALETTE["dark_text"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85), zorder=6)

    ax.set_xticklabels(ptype_labels)
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel("$\\hat{\\beta}$")
    # ax.set_title("Stability Under\nSemantic Perturbation", fontweight="semibold", pad=8)

    # Remove add_panel_label here, use ax.text to put it below
    ax.text(0.5, -0.22, "(a) Stability Under\nSemantic Perturbation", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])

    # ── Panel (b): β decomposition (val vs out) ───────────────────────────
    ax = axes[1]

    decomp_data = []
    for pt in ptype_order:
        sub = df[df["perturbation_type"] == pt]
        if len(sub) > 0:
            decomp_data.append({
                "perturbation": pt,
                "Value Variation\n$\\hat{\\beta}_{val}$": sub["beta_vv"].mean(),
                "Output Variation\n$\\hat{\\beta}_{out}$": sub["beta_out"].mean(),
            })

    if decomp_data:
        df_dec = pd.DataFrame(decomp_data)
        x = np.arange(len(df_dec))
        width = 0.32

        components = [
            ("Value Variation\n$\\hat{\\beta}_{val}$", PALETTE["primary"]),
            ("Output Variation\n$\\hat{\\beta}_{out}$", PALETTE["secondary"]),
        ]

        for j, (comp, color) in enumerate(components):
            vals = df_dec[comp].values
            offset = (j - 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.9, color=color,
                          alpha=1.0, edgecolor="white", linewidth=0.8,
                          label=comp.split("\n")[0])
            # Value labels
            for k, v in enumerate(vals):
                ax.text(x[k] + offset, v + 0.005, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=12, fontweight="bold",
                        color=PALETTE["dark_text"])

        ax.set_xticks(x)
        ax.set_xticklabels(ptype_labels)
        ax.set_ylabel("Mean $\\hat{\\beta}$ Component")
        ax.legend(fontsize=11, loc="upper left", framealpha=0.9)

        # Annotate dominance
        for row in decomp_data:
            vv = row["Value Variation\n$\\hat{\\beta}_{val}$"]
            out = row["Output Variation\n$\\hat{\\beta}_{out}$"]
            total = vv + out
            if total > 0:
                out_pct = out / total * 100
                # We'll add a text note if output dominates
                pass

    # ax.set_title("$\\hat{\\beta}$ Component\nDecomposition", fontweight="semibold", pad=8)
    
    ax.text(0.5, -0.183, "(b) $\\hat{\\beta}$ Component\nDecomposition", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])

    fig.subplots_adjust(bottom=0.2)
    save_figure(fig, IMG_DIR / "fig_q1_3_stability_semantic_rewrite.png")


if __name__ == "__main__":
    plot_fig_q1_3()
