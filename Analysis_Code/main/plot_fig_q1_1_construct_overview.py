"""
plot_fig_q1_1_construct_overview.py
===================================
Figure Q1-1: Construct Validity Overview (3-panel)

(a) Recovery dynamics separate normal and cliff-prone executions.
(b) Conditional distributions within the same SR bucket reveal hidden recovery variance.
(c) Safety-only scores do not fully explain family-level resilience.

Reference: Requirements.md §Figure Q1-1
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from expq1_data_loader import load_rebound, load_ge, IMG_DIR
from plot_style import (setup_style, add_panel_label, save_figure,
                        PALETTE, PAL_PERTURBATION, PAL_FAMILY)


def plot_fig_q1_1():
    setup_style()

    df_reb = load_rebound()
    df_ge = load_ge()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel (a): Recovery dynamics – normal vs cliff-prone ─────────────
    ax = axes[0]
    # Use rebound_num_recovery_windows as proxy for recovery complexity
    # Split by high/low recovery cost (median split)
    median_crec = df_reb["c_rec_total"].median()
    df_reb["recovery_class"] = np.where(
        df_reb["c_rec_total"] > median_crec, "High Recovery\n(Cliff-prone)", "Low Recovery\n(Normal)"
    )

    # Plot overlapping histograms of recovery cost
    colors_rc = [PALETTE["success"], PALETTE["danger"]]
    for i, cls in enumerate(["Low Recovery\n(Normal)", "High Recovery\n(Cliff-prone)"]):
        subset = df_reb[df_reb["recovery_class"] == cls]
        ax.hist(subset["c_rec_total"], bins=8, alpha=0.65, color=colors_rc[i],
                label=cls, edgecolor="white", linewidth=0.8)

    ax.axvline(median_crec, color=PALETTE["neutral"], linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"Median = {median_crec:.1f}")
    ax.set_xlabel("Recovery Cost $C_{rec}$")
    ax.set_ylabel("Episode Count")
    ax.set_title("Recovery Dynamics", fontweight="semibold", pad=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    add_panel_label(ax, "(a)")

    # ── Panel (b): Within-bucket recovery variance ───────────────────────
    ax = axes[1]
    # Bin task_percent_complete into 3 buckets for cleaner visualization
    bins = [0, 0.5, 0.99, 1.01]
    labels = ["Low\n(0–50%)", "Mid\n(50–99%)", "Full\n(100%)"]
    df_reb["tc_bucket"] = pd.cut(df_reb["task_percent_complete"], bins=bins, labels=labels,
                                  include_lowest=True)

    bucket_order = labels
    bp = sns.boxplot(data=df_reb, x="tc_bucket", y="c_rec_total", order=bucket_order,
                     palette=[PALETTE["primary"], PALETTE["accent"], PALETTE["success"]],
                     width=0.55, linewidth=1.2, fliersize=5, ax=ax,
                     boxprops=dict(alpha=0.8))
    sns.stripplot(data=df_reb, x="tc_bucket", y="c_rec_total", order=bucket_order,
                  color=PALETTE["dark_text"], alpha=0.4, size=4, jitter=0.15, ax=ax)

    ax.set_xlabel("Task Completion Bucket")
    ax.set_ylabel("Recovery Cost $C_{rec}$")
    ax.set_title("Within-Bucket Recovery Variance", fontweight="semibold", pad=10)

    # Annotate variance
    for i, label in enumerate(bucket_order):
        subset = df_reb[df_reb["tc_bucket"] == label]["c_rec_total"]
        if len(subset) > 1:
            cv = subset.std() / (subset.mean() + 1e-9) * 100
            ax.text(i, ax.get_ylim()[1] * 0.93, f"CV={cv:.0f}%",
                    ha="center", fontsize=8, color=PALETTE["neutral"])

    add_panel_label(ax, "(b)")

    # ── Panel (c): Safety score vs Family resilience ─────────────────────
    ax = axes[2]
    # Compute per-anchor resilience summary from GE data (stress_sweep)
    sweep = df_ge[df_ge["phase_label"] == "stress_sweep"].copy()
    if not sweep.empty:
        fam_stats = sweep.groupby("anchor_id").agg(
            mean_completion=("task_percent_complete", "mean"),
            mean_crec=("rebound_c_rec", "mean"),
            task_family=("task_family", "first"),
        ).reset_index()

        # Construct a simple "resilience score" = completion * (1 - normalized_crec)
        max_crec = fam_stats["mean_crec"].max() + 1e-9
        fam_stats["resilience_S"] = fam_stats["mean_completion"] * (1 - fam_stats["mean_crec"] / max_crec)

        # Safety proxy: completion alone
        fam_stats["safety_proxy"] = fam_stats["mean_completion"]

        # Scatter with color by task family
        families = sorted(fam_stats["task_family"].unique())
        for i, fam in enumerate(families):
            sub = fam_stats[fam_stats["task_family"] == fam]
            ax.scatter(sub["safety_proxy"], sub["resilience_S"],
                       s=80, alpha=0.8, color=PAL_FAMILY[i % len(PAL_FAMILY)],
                       label=fam.replace("_", " ").title(),
                       edgecolors="white", linewidth=0.5, zorder=3)

        # Diagonal reference line (perfect correlation)
        lims = [0, 1.05]
        ax.plot(lims, lims, "--", color=PALETTE["neutral"], alpha=0.5, linewidth=1)

        # Fit and show correlation
        if len(fam_stats) >= 3:
            r, p = stats.spearmanr(fam_stats["safety_proxy"], fam_stats["resilience_S"])
            ax.text(0.05, 0.95, f"Spearman ρ = {r:.2f}\np = {p:.3f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Safety-Only Proxy (Completion)")
        ax.set_ylabel("Resilience Score $S_\\mathcal{F}$")
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    ax.set_title("Safety vs. Resilience", fontweight="semibold", pad=10)
    add_panel_label(ax, "(c)")

    fig.suptitle("Figure Q1-1: Construct Validity Overview",
                 fontsize=14, fontweight="bold", y=1.02, color=PALETTE["dark_text"])
    fig.tight_layout()
    save_figure(fig, IMG_DIR / "fig_q1_1_construct_validity_overview.png")


if __name__ == "__main__":
    plot_fig_q1_1()
