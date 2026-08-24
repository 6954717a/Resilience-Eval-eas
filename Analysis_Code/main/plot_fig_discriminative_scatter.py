"""
plot_fig_discriminative_scatter.py
==================================
Compact discriminative scatter plot for Exp-Q1.

The figure demonstrates that episodes with the same outcome (SR=100%) can still
have sharply different recovery costs. It uses a broken x-axis: incomplete
episodes remain visible in a compressed low-completion pane, while the successful
cluster around Task Completion=1.0 is expanded for a denser paper subfigure.

Output: fig_discriminative_scatter.png / .pdf
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from expq1_multirun_loader import IMG_DIR, load_all_rebound
from plot_style import PALETTE, save_figure, setup_style


ANALYSIS_EXPQ1_DIR = IMG_DIR / "ExpQ1"
PAPER_EXPQ1_DIR = IMG_DIR.parent.parent.parent / "paper_writing" / "Images" / "ExpQ1"

PERT_COLORS = {
    "clean": PALETTE["primary"],
    "surface_rewrite": PALETTE["danger"],
    "object_state_toggle": PALETTE["accent"],
    "filler_injection": PALETTE["secondary"],
}
PERT_LABELS = {
    "clean": "Clean",
    "surface_rewrite": "Rewrite",
    "object_state_toggle": "Obj.",
    "filler_injection": "Filler",
}


def _add_axis_break_marks(ax_left, ax_right) -> None:
    d = 0.014
    kwargs = dict(color="#777777", clip_on=False, linewidth=1.35)
    ax_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_left.transAxes, **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_left.transAxes, **kwargs)
    ax_right.plot((-d, +d), (-d, +d), transform=ax_right.transAxes, **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), transform=ax_right.transAxes, **kwargs)


def plot_fig_discriminative_scatter() -> None:
    setup_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "SimHei",
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "Arial",
                "DejaVu Sans",
            ],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    ANALYSIS_EXPQ1_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_EXPQ1_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_rebound()
    success_all = df[df["is_success"] == 1].copy()
    non_success_all = df[df["is_success"] == 0].copy()

    lo_cut = success_all["c_rec"].quantile(0.02)
    hi_cut = success_all["c_rec"].quantile(0.98)
    success = success_all[
        (success_all["c_rec"] >= lo_cut) & (success_all["c_rec"] <= hi_cut)
    ].copy()

    hi_cut_ns = non_success_all["c_rec"].quantile(0.95)
    non_success = non_success_all[non_success_all["c_rec"] <= hi_cut_ns].copy()

    crec_min = success["c_rec"].min()
    crec_max = success["c_rec"].max()
    crec_mean = success["c_rec"].mean()
    crec_med = success["c_rec"].median()
    crec_q05 = success["c_rec"].quantile(0.05)
    crec_q25 = success["c_rec"].quantile(0.25)
    crec_q75 = success["c_rec"].quantile(0.75)
    crec_q95 = success["c_rec"].quantile(0.95)

    # This figure is used as a narrow wrapfigure (0.45\textwidth).  Keep the
    # natural canvas compact and make all foreground text deliberately large so
    # the PDF remains readable after LaTeX scaling.
    fig = plt.figure(figsize=(5.85, 4.10))
    gs = gridspec.GridSpec(
        1,
        3,
        width_ratios=[0.76, 3.30, 0.56],
        wspace=0.035,
        left=0.13,
        right=0.985,
        bottom=0.18,
        top=0.86,
    )
    ax_low = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharey=ax_low)
    ax_hist = fig.add_subplot(gs[0, 2], sharey=ax_low)

    ax_low.scatter(
        non_success["task_percent_complete"],
        non_success["c_rec"],
        s=42,
        alpha=0.16,
        color=PALETTE["neutral"],
        edgecolors="white",
        linewidths=0.45,
        marker="o",
        label="Incomp.",
        zorder=3,
    )

    np.random.seed(42)
    for pt in sorted(success["perturbation_type"].unique()):
        sub = success[success["perturbation_type"] == pt]
        jitter_x = 1.0 + np.random.uniform(-0.009, 0.009, size=len(sub))
        ax.scatter(
            jitter_x,
            sub["c_rec"],
            s=112,
            alpha=0.90,
            color=PERT_COLORS.get(pt, PALETTE["neutral"]),
            edgecolors="white",
            linewidths=0.90,
            label=f"{PERT_LABELS.get(pt, pt)}",
            zorder=4,
        )

    ax.axvline(
        1.0,
        color=PALETTE["neutral"],
        linestyle=":",
        alpha=0.50,
        linewidth=1.4,
        zorder=0,
    )
    ax.axhspan(crec_q25, crec_q75, color=PALETTE["danger"], alpha=0.075, zorder=1)
    ax.axhline(
        crec_med,
        color=PALETTE["danger"],
        linestyle="-",
        linewidth=1.60,
        alpha=0.70,
        zorder=2,
    )

    bracket_x = 1.041
    ax.annotate(
        "",
        xy=(bracket_x, crec_min + 0.5),
        xytext=(bracket_x, crec_max - 0.5),
        arrowprops=dict(arrowstyle="<->", color=PALETTE["danger"], lw=3.6),
        zorder=5,
    )
    ax.plot(
        bracket_x,
        crec_mean,
        "D",
        color="white",
        markersize=9.5,
        markeredgecolor=PALETTE["danger"],
        markeredgewidth=2.5,
        zorder=6,
    )
    # ax.text(
    #     1.087,
    #     crec_med,
    #     f"median={crec_med:.1f}",
    #     fontsize=10.7,
    #     color=PALETTE["danger"],
    #     fontweight="bold",
    #     ha="right",
    #     va="bottom",
    #     zorder=7,
    # )

    insight_text = (
        "Same SR = 100%\n"
        f"$C_{{rec}}$: [{crec_min:.0f},{crec_max:.0f}]"
        f"  ($\\mu$={crec_mean:.1f})"
    )
    ax.text(
        0.035,
        0.965,
        insight_text,
        transform=ax.transAxes,
        fontsize=15.6,
        va="top",
        ha="left",
        fontweight="bold",
        color=PALETTE["dark_text"],
        bbox=dict(
            boxstyle="round,pad=0.36",
            facecolor="#FFF3E0",
            alpha=0.96,
            edgecolor=PALETTE["accent"],
            linewidth=1.45,
        ),
        zorder=7,
    )
    ax_low.set_ylabel("Recovery Cost $C_{\\mathrm{rec}}$", fontsize=21, fontweight="bold")
    fig.supxlabel("Task Completion", fontsize=21, fontweight="bold", y=0.050)
    ax_low.tick_params(axis="both", labelsize=14, width=1.2)
    ax.tick_params(axis="both", labelsize=16, width=1.2)
    ax_low.set_xlim(-0.035, 0.835)
    ax.set_xlim(0.968, 1.052)
    ax_low.set_ylim(-5, 105)
    ax_low.set_xticks([0.0, 0.8])
    ax.set_xticks([1.00])
    ax_low.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)
    ax_low.spines["right"].set_visible(False)
    _add_axis_break_marks(ax_low, ax)

    handles_low, labels_low = ax_low.get_legend_handles_labels()
    handles_main, labels_main = ax.get_legend_handles_labels()
    fig.legend(
        handles_low + handles_main,
        labels_low + labels_main,
        prop={"size": 13.8, "weight": "bold"},
        loc="upper center",
        ncol=4,
        framealpha=0.96,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.56, 0.986),
        handletextpad=0.26,
        borderpad=0.22,
        columnspacing=0.52,
        labelspacing=0.18,
    )

    hist_order = ["clean", "object_state_toggle", "surface_rewrite"]
    hist_data = [success[success["perturbation_type"] == pt]["c_rec"] for pt in hist_order]
    hist_colors = [PERT_COLORS[pt] for pt in hist_order]
    ax_hist.hist(
        hist_data,
        bins=18,
        orientation="horizontal",
        stacked=True,
        color=hist_colors,
        alpha=0.24,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_hist.axhspan(crec_q25, crec_q75, color=PALETTE["danger"], alpha=0.075, zorder=0)
    ax_hist.axhline(crec_mean, color=PALETTE["danger"], linestyle="--", linewidth=2.2, alpha=0.88)
    ax_hist.hlines(
        [crec_q05, crec_q95],
        xmin=0,
        xmax=ax_hist.get_xlim()[1],
        color=PALETTE["danger"],
        linestyle=":",
        linewidth=1.0,
        alpha=0.55,
    )
    # ax_hist.text(
    #     0.96,
    #     crec_q75 + 1.2,
    #     "IQR",
    #     fontsize=10.2,
    #     color=PALETTE["danger"],
    #     fontweight="bold",
    #     ha="right",
    #     va="bottom",
    #     transform=ax_hist.get_yaxis_transform(),
    # )
    ax_hist.tick_params(axis="y", labelleft=False, left=False)
    ax_hist.tick_params(axis="x", labelsize=10, colors="#777777")
    ax_hist.set_xlabel("Count", fontsize=12, color="#777777", fontweight="bold")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.grid(False)

    out_png = ANALYSIS_EXPQ1_DIR / "fig_discriminative_scatter.png"
    save_figure(fig, out_png)

    for suffix in [".png", ".pdf"]:
        src = out_png.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, PAPER_EXPQ1_DIR / src.name)
            print(f"  [copy] {PAPER_EXPQ1_DIR / src.name}")


if __name__ == "__main__":
    plot_fig_discriminative_scatter()
