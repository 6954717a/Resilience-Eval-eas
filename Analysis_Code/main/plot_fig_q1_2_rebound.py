"""
plot_fig_q1_2_rebound.py
========================
Figure Q1-2: Rebound Cost Decomposition (3-panel)

(a) Episodes with similar task completion exhibit different recovery costs.
(b) Paired clean/perturbed executions show systematic increase in C_rec.
(c) Decomposition of C_rec identifies cognitive, physical, and state-debt components.

Reference: Requirements.md §Figure Q1-R / Q1-2
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from expq1_data_loader import (load_rebound, aggregate_rebound_paired,
                                IMG_DIR, TABLE_DIR)
from plot_style import (setup_style, add_panel_label, save_figure,
                        PALETTE, PAL_PERTURBATION, PAL_DECOMP)


def plot_fig_q1_2():
    setup_style()

    df = load_rebound()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # ── Panel (a): C_rec by completion bucket ────────────────────────────
    ax = axes[0]

    # Create finer completion bins
    bins = [0, 0.5, 0.99, 1.01]
    labels_b = ["Low\n(0–50%)", "Mid\n(50–99%)", "Full\n(100%)"]
    df["tc_bucket"] = pd.cut(df["task_percent_complete"], bins=bins,
                              labels=labels_b, include_lowest=True)

    pal_bucket = [PALETTE["danger"], PALETTE["accent"], PALETTE["success"]]
    bp = sns.boxplot(data=df, x="tc_bucket", y="c_rec_total",
                     order=labels_b, palette=pal_bucket,
                     width=0.5, linewidth=1.3, fliersize=4,
                     boxprops=dict(alpha=0.75), ax=ax)
    sns.stripplot(data=df, x="tc_bucket", y="c_rec_total",
                  order=labels_b, color=PALETTE["dark_text"],
                  alpha=0.45, size=5, jitter=0.12, ax=ax)

    # Add mean markers
    means = df.groupby("tc_bucket", observed=True)["c_rec_total"].mean()
    for i, label in enumerate(labels_b):
        if label in means.index:
            ax.scatter(i, means[label], marker="D", color="white",
                       edgecolors=PALETTE["dark_text"], s=45, zorder=5, linewidths=1.2)

    ax.set_xlabel("Task Completion Bucket")
    ax.set_ylabel("Recovery Cost $C_{rec}$")
    ax.set_title("Same Completion,\nDifferent Recovery Cost", fontweight="semibold", pad=8)
    add_panel_label(ax, "(a)")

    # ── Panel (b): Paired slope chart – clean vs perturbed ───────────────
    ax = axes[1]
    paired = aggregate_rebound_paired(df)

    if not paired.empty:
        ptypes = paired["perturbation_type"].unique()
        colors_pt = {
            "surface_rewrite": PALETTE["accent"],
            "object_state_toggle": PALETTE["danger"],
        }

        for pt in ptypes:
            sub = paired[paired["perturbation_type"] == pt]
            color = colors_pt.get(pt, PALETTE["neutral"])

            for _, row in sub.iterrows():
                ax.plot([0, 1], [row["c_rec_total_clean"], row["c_rec_total_pert"]],
                        color=color, alpha=0.35, linewidth=1.2, zorder=2)
                ax.scatter([0], [row["c_rec_total_clean"]], color=PALETTE["primary"],
                           s=40, alpha=0.6, edgecolors="white", linewidths=0.5, zorder=3)
                ax.scatter([1], [row["c_rec_total_pert"]], color=color,
                           s=40, alpha=0.6, edgecolors="white", linewidths=0.5, zorder=3)

            # Group means
            mean_clean = sub["c_rec_total_clean"].mean()
            mean_pert = sub["c_rec_total_pert"].mean()
            ax.plot([0, 1], [mean_clean, mean_pert], color=color,
                    linewidth=2.5, zorder=4, label=pt.replace("_", " ").title())
            ax.scatter([0], [mean_clean], color=PALETTE["primary"],
                       s=80, edgecolors="white", linewidths=1.5, zorder=5, marker="D")
            ax.scatter([1], [mean_pert], color=color,
                       s=80, edgecolors="white", linewidths=1.5, zorder=5, marker="D")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Clean", "Perturbed"])
        ax.set_ylabel("Recovery Cost $C_{rec}$")
        ax.set_xlim(-0.3, 1.3)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

        # Wilcoxon test
        all_clean = paired["c_rec_total_clean"].values
        all_pert = paired["c_rec_total_pert"].values
        try:
            stat, p = stats.wilcoxon(all_clean, all_pert, alternative="less")
            ax.text(0.95, 0.05, f"Wilcoxon p = {p:.4f}",
                    transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        except Exception:
            pass

    ax.set_title("Systematic Increase\nUnder Perturbation", fontweight="semibold", pad=8)
    add_panel_label(ax, "(b)")

    # ── Panel (c): C_rec decomposition stacked bar ───────────────────────
    ax = axes[2]

    ptype_order = ["clean", "surface_rewrite", "object_state_toggle"]
    ptype_labels = ["Clean", "Surface\nRewrite", "Object State\nToggle"]

    decomp_data = []
    for pt in ptype_order:
        sub = df[df["perturbation_type"] == pt]
        if len(sub) > 0:
            decomp_data.append({
                "perturbation": pt,
                "Cognitive\n$C_{rec}^{cog}$": sub["c_rec_cog"].mean(),
                "Physical\n$C_{rec}^{phy}$": sub["c_rec_phy"].mean(),
                "State Debt\n$C_{rec}^{debt}$": sub["c_rec_debt"].mean(),
            })

    if decomp_data:
        df_decomp = pd.DataFrame(decomp_data)
        x = np.arange(len(df_decomp))
        width = 0.55

        components = ["Cognitive\n$C_{rec}^{cog}$", "Physical\n$C_{rec}^{phy}$", "State Debt\n$C_{rec}^{debt}$"]
        bottoms = np.zeros(len(df_decomp))

        for j, comp in enumerate(components):
            vals = df_decomp[comp].values
            bars = ax.bar(x, vals, width, bottom=bottoms, color=PAL_DECOMP[j],
                          alpha=0.85, edgecolor="white", linewidth=0.8,
                          label=comp.split("\n")[0])
            # Value labels inside bars (if space allows)
            for k, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 2:
                    ax.text(x[k], b + v / 2, f"{v:.1f}", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold")
            bottoms += vals

        # Total labels on top
        for k, total in enumerate(bottoms):
            ax.text(x[k], total + 0.5, f"Σ={total:.1f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=PALETTE["dark_text"])

        avail_labels = [ptype_labels[i] for i, pt in enumerate(ptype_order)
                        if any(d["perturbation"] == pt for d in decomp_data)]
        ax.set_xticks(x)
        ax.set_xticklabels(avail_labels)
        ax.set_ylabel("Mean Recovery Cost")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    ax.set_title("Recovery Cost\nDecomposition", fontweight="semibold", pad=8)
    add_panel_label(ax, "(c)")

    fig.suptitle("Figure Q1-2: Rebound Validity — Same Completion, Different Recovery Cost",
                 fontsize=13, fontweight="bold", y=1.03, color=PALETTE["dark_text"])
    fig.tight_layout()
    save_figure(fig, IMG_DIR / "fig_q1_2_rebound_cost_decomposition.png")


if __name__ == "__main__":
    plot_fig_q1_2()
