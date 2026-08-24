"""
plot_fig_q1_metric_validity_map.py
==================================
High-density Exp-Q1 validity dashboard.

The figure consolidates three construct-validity claims:
1. Rebound separates same-outcome executions by hidden recovery burden.
2. Stability exposes semantic-perturbation decision inconsistency.
3. Graceful Extensibility maps stress severity and task family to response risk.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from expq1_data_loader import IMG_DIR, load_ge, load_stability
from expq1_multirun_loader import load_all_rebound
from plot_style import PALETTE, PAL_3TYPE, save_figure, setup_style
from plot_task_family_resilience_heatmap import (
    LAMBDA_ORDER,
    TASK_FAMILY_ORDER,
    aggregate_cells,
    compute_risk,
    lambda_star,
    load_rollouts,
    pivot_matrix,
)


ANALYSIS_EXPQ1_DIR = IMG_DIR / "ExpQ1"
PAPER_EXPQ1_DIR = IMG_DIR.parent.parent.parent / "paper_writing" / "Images" / "ExpQ1"

PANEL_TITLE_SIZE = 18.0
AXIS_LABEL_SIZE = 14.2
TICK_SIZE = 12.0
ANNOT_SIZE = 11.5


def _stderr(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) <= 2 or data["x"].nunique() <= 1 or data["y"].nunique() <= 1:
        return math.nan
    return float(data["x"].rank().corr(data["y"].rank()))


def _clean_ge_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ge_contract_margin_min" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out["stress_lambda"] = pd.to_numeric(out["stress_lambda"], errors="coerce")
    out["ge_contract_margin_min"] = pd.to_numeric(
        out["ge_contract_margin_min"], errors="coerce"
    )
    out = out[out["ge_contract_margin_min"].notna()].copy()

    # Older partial GE exports used -1 or exact 0 as missing formal-margin sentinels.
    out = out[
        (out["ge_contract_margin_min"] != -1.0)
        & (out["ge_contract_margin_min"] != 0.0)
    ].copy()
    out = out[out["phase_label"].isin(["clean_reference", "stress_sweep"])].copy()
    out = out[out["task_family"].notna()].copy()
    out = out[out["task_family"].astype(str).str.lower() != "unknown"].copy()

    if "proxy_recovery_cost" not in out.columns:
        if "rebound_c_rec" in out.columns:
            out["proxy_recovery_cost"] = pd.to_numeric(
                out["rebound_c_rec"], errors="coerce"
            )
        else:
            out["proxy_recovery_cost"] = np.nan
    out["proxy_recovery_cost"] = pd.to_numeric(
        out["proxy_recovery_cost"], errors="coerce"
    )

    for col in ["ge_boundary_hit", "ge_hard_boundary_hit", "ge_cliff_flag"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _set_panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(
        title,
        loc="left",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="heavy",
        color=PALETTE["dark_text"],
        pad=12,
    )


def _lighten_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.1, length=4.5)
    for spine in ax.spines.values():
        spine.set_color("#8E8E8E")
        spine.set_linewidth(1.0)


def _annotate_mean(
    ax: plt.Axes,
    idx: int,
    mean_value: float,
    n_value: int,
    fmt: str,
    x_offset: float = 0.17,
) -> None:
    ax.scatter(
        idx,
        mean_value,
        marker="D",
        s=43,
        color="white",
        edgecolors=PALETTE["dark_text"],
        linewidths=1.15,
        zorder=6,
    )
    ax.text(
        idx + x_offset,
        mean_value,
        f"$\\mu$={fmt.format(mean_value)}",
        ha="left",
        va="center",
        fontsize=ANNOT_SIZE,
        fontweight="bold",
        color=PALETTE["dark_text"],
        linespacing=0.95,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="none",
            alpha=0.86,
        ),
        zorder=7,
    )


def _plot_rebound_panel(ax: plt.Axes) -> None:
    rebound = load_all_rebound()
    if rebound.empty:
        ax.text(0.5, 0.5, "No rebound rows", ha="center", va="center")
        ax.set_axis_off()
        return

    rebound = rebound[rebound["is_success"] == 1].copy()
    rebound["c_rec"] = pd.to_numeric(rebound["c_rec"], errors="coerce")
    rebound = rebound[rebound["c_rec"].notna()].copy()

    order = ["clean", "surface_rewrite", "object_state_toggle"]
    labels = ["Clean", "Surface\nRewrite", "Object State\nToggle"]
    palette = {
        "clean": "#74BDB4",
        "surface_rewrite": "#E77474",
        "object_state_toggle": "#F2A32B",
    }
    rebound = rebound[rebound["perturbation_type"].isin(order)].copy()

    sns.boxplot(
        data=rebound,
        x="perturbation_type",
        y="c_rec",
        order=order,
        hue="perturbation_type",
        hue_order=order,
        palette=palette,
        width=0.54,
        linewidth=1.25,
        fliersize=0,
        saturation=0.82,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=rebound,
        x="perturbation_type",
        y="c_rec",
        order=order,
        hue="perturbation_type",
        hue_order=order,
        palette=palette,
        dodge=False,
        jitter=0.19,
        size=3.9,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.25,
        legend=False,
        ax=ax,
    )

    stats = rebound.groupby("perturbation_type")["c_rec"].agg(["mean", "count"])
    for idx, ptype in enumerate(order):
        if ptype in stats.index:
            _annotate_mean(
                ax,
                idx,
                float(stats.loc[ptype, "mean"]),
                int(stats.loc[ptype, "count"]),
                "{:.1f}",
                x_offset=0.16,
            )

    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=TICK_SIZE, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(
        "Recovery cost $C_{\\mathrm{rec}}$",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="semibold",
    )
    _set_panel_title(ax, "(a) Rebound")
    ax.text(
        0.02,
        0.96,
        "SR=100%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.8,
        fontweight="heavy",
        color=PALETTE["dark_text"],
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="#FFF7E6",
            edgecolor=PALETTE["accent"],
            linewidth=1.0,
            alpha=0.95,
        ),
    )
    _lighten_axis(ax)


def _plot_stability_panel(ax: plt.Axes) -> None:
    stability = load_stability()
    if stability.empty:
        ax.text(0.5, 0.5, "No stability rows", ha="center", va="center")
        ax.set_axis_off()
        return

    stability = stability.copy()
    stability["beta_neigh"] = pd.to_numeric(stability["beta_neigh"], errors="coerce")
    stability = stability[stability["beta_neigh"].notna()].copy()

    order = ["clean", "surface_rewrite", "filler_injection"]
    labels = ["Clean", "Surface\nRewrite", "Filler\nInjection"]
    stability = stability[stability["perturbation_type"].isin(order)].copy()

    sns.boxplot(
        data=stability,
        x="perturbation_type",
        y="beta_neigh",
        order=order,
        hue="perturbation_type",
        hue_order=order,
        palette=PAL_3TYPE,
        width=0.54,
        linewidth=1.25,
        fliersize=0,
        saturation=0.84,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=stability,
        x="perturbation_type",
        y="beta_neigh",
        order=order,
        color=PALETTE["dark_text"],
        alpha=0.58,
        size=4.1,
        jitter=0.16,
        ax=ax,
    )

    stats = stability.groupby("perturbation_type")["beta_neigh"].agg(["mean", "count"])
    for idx, ptype in enumerate(order):
        if ptype in stats.index:
            _annotate_mean(
                ax,
                idx,
                float(stats.loc[ptype, "mean"]),
                int(stats.loc[ptype, "count"]),
                "{:.2f}",
                x_offset=0.16,
            )

    ymax = max(0.55, float(stability["beta_neigh"].max()) * 1.17)
    ax.set_ylim(-0.02, ymax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=TICK_SIZE, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(
        "Instability $\\hat{\\beta}$",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="semibold",
    )
    _set_panel_title(ax, "(b) Stability")
    _lighten_axis(ax)


def _plot_distributional_validity(ax_rebound: plt.Axes, ax_stability: plt.Axes) -> None:
    _plot_rebound_panel(ax_rebound)
    _plot_stability_panel(ax_stability)


def _prepare_ge_stress_aggregate(ge: pd.DataFrame) -> pd.DataFrame:
    if ge.empty:
        return pd.DataFrame()

    work = ge.copy()
    work["stress_lambda"] = pd.to_numeric(work["stress_lambda"], errors="coerce")
    work["proxy_recovery_cost"] = pd.to_numeric(
        work["proxy_recovery_cost"], errors="coerce"
    )
    work["margin_loss_source"] = pd.to_numeric(
        work["ge_contract_margin_min"], errors="coerce"
    )

    clean = work[work["phase_label"].eq("clean_reference")]
    if clean.empty:
        clean = work[work["stress_lambda"].eq(0.0)]
    if clean.empty:
        baseline_margin = float(work["margin_loss_source"].mean())
    else:
        baseline_margin = float(clean["margin_loss_source"].mean())

    work["margin_degradation"] = baseline_margin - work["margin_loss_source"]
    agg = (
        work.groupby("stress_lambda")
        .agg(
            n=("margin_loss_source", "size"),
            delta_m=("margin_degradation", "mean"),
            delta_m_se=("margin_degradation", _stderr),
            c_rec_mean=("proxy_recovery_cost", "mean"),
            c_rec_se=("proxy_recovery_cost", _stderr),
            boundary_rate=("ge_boundary_hit", "mean"),
            hard_boundary_rate=("ge_hard_boundary_hit", "mean"),
        )
        .reset_index()
        .sort_values("stress_lambda")
    )
    return agg


def _prepare_ge_risk_aggregate(risk_data: pd.DataFrame) -> pd.DataFrame:
    stress = risk_data[
        risk_data["perturbation_group"].isin(["Clean Reference", "Stress Severity"])
        & risk_data["stress_lambda"].isin(LAMBDA_ORDER)
    ].copy()
    if stress.empty:
        return pd.DataFrame()

    stress["stress_lambda"] = pd.to_numeric(stress["stress_lambda"], errors="coerce")
    stress["resilience_risk"] = pd.to_numeric(
        stress["resilience_risk"], errors="coerce"
    )
    if "rebound_c_rec" in stress.columns:
        stress["c_rec_for_response"] = pd.to_numeric(
            stress["rebound_c_rec"], errors="coerce"
        )
    else:
        stress["c_rec_for_response"] = np.nan

    return (
        stress.groupby("stress_lambda")
        .agg(
            n=("resilience_risk", "size"),
            risk_mean=("resilience_risk", "mean"),
            risk_se=("resilience_risk", _stderr),
            c_rec_mean=("c_rec_for_response", "mean"),
            c_rec_se=("c_rec_for_response", _stderr),
            completion_mean=("task_percent_complete", "mean"),
        )
        .reset_index()
        .sort_values("stress_lambda")
    )


def _plot_ge_trajectory(
    ax: plt.Axes, ge: pd.DataFrame, risk_data: pd.DataFrame
) -> None:
    risk_agg = _prepare_ge_risk_aggregate(risk_data)
    formal_agg = _prepare_ge_stress_aggregate(ge)
    if risk_agg.empty:
        ax.text(0.5, 0.5, "No GE stress-response rows", ha="center", va="center")
        ax.set_axis_off()
        return

    x = risk_agg["stress_lambda"].astype(float).to_numpy()
    y = risk_agg["risk_mean"].astype(float).to_numpy()
    se = risk_agg["risk_se"].astype(float).fillna(0.0).to_numpy()

    ax.fill_between(
        x,
        y - se,
        y + se,
        color=PALETTE["danger"],
        alpha=0.15,
        linewidth=0,
    )
    ax.plot(
        x,
        y,
        "o-",
        color=PALETTE["danger"],
        linewidth=2.2,
        markersize=6.4,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="RISK",
    )
    ax.axvline(0.0, color="#9A9A9A", linestyle="--", linewidth=0.9, alpha=0.85)

    if not formal_agg.empty and "delta_m" in formal_agg.columns:
        formal = formal_agg[formal_agg["stress_lambda"].isin(risk_agg["stress_lambda"])].copy()
        if len(formal) >= 3:
            scaled_margin = pd.to_numeric(formal["delta_m"], errors="coerce")
            if scaled_margin.notna().any() and scaled_margin.max() != scaled_margin.min():
                scaled_margin = (
                    (scaled_margin - scaled_margin.min())
                    / (scaled_margin.max() - scaled_margin.min())
                )
                scaled_margin = 0.08 + 0.18 * scaled_margin
                ax.plot(
                    formal["stress_lambda"],
                    scaled_margin,
                    "o--",
                    color=PALETTE["primary"],
                    linewidth=1.2,
                    markersize=4.2,
                    alpha=0.75,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label="$M_f$ TRACE",
                )

    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(0, max(0.62, float(np.nanmax(y + se)) + 0.08))
    ax.set_xticks(LAMBDA_ORDER)
    ax.set_xlabel(
        "Stress severity $\\lambda$",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="semibold",
    )
    ax.set_ylabel(
        "Stress-response risk",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="semibold",
    )
    _set_panel_title(ax, "(c) GE: Stress-Extensibility")
    _lighten_axis(ax)

    rec = risk_agg["c_rec_mean"].astype(float)
    rec_se = risk_agg["c_rec_se"].astype(float).fillna(0.0)
    rho = _spearman(risk_agg["stress_lambda"], rec)

    ax_rec = inset_axes(
        ax,
        width="39%",
        height="34%",
        loc="upper left",
        bbox_to_anchor=(0.08, 0.03, 0.92, 0.92),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ax_rec.bar(
        x,
        rec,
        yerr=rec_se,
        width=0.095,
        color=PALETTE["accent"],
        alpha=0.65,
        edgecolor="white",
        linewidth=0.6,
        error_kw=dict(ecolor=PALETTE["dark_text"], lw=0.8, capsize=2),
    )
    ax_rec.plot(
        x,
        rec,
        color=PALETTE["danger"],
        linewidth=1.3,
        marker="o",
        markersize=3.3,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    ax_rec.set_title("$C_{\\mathrm{rec}}$ RESPONSE", fontsize=9.8, fontweight="heavy", pad=2)
    ax_rec.set_xlim(-0.03, 1.05)
    ax_rec.tick_params(axis="both", labelsize=8.5, pad=1)
    ax_rec.grid(True, alpha=0.18, linewidth=0.45)
    for spine in ax_rec.spines.values():
        spine.set_color("#B0B0B0")
        spine.set_linewidth(0.55)

    ax.legend(loc="upper right", fontsize=10.7, frameon=True, framealpha=0.9)


def _draw_manual_risk_colorbar(cax: plt.Axes, cmap: LinearSegmentedColormap) -> None:
    """Draw a vector-only colorbar to avoid PDF gradient-image displacement."""
    cax.clear()
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.grid(False)

    n_segments = 64
    for idx in range(n_segments):
        y0 = idx / n_segments
        height = 1.0 / n_segments + 0.001
        cax.add_patch(
            Rectangle(
                (0.18, y0),
                0.38,
                height,
                transform=cax.transAxes,
                facecolor=cmap((idx + 0.5) / n_segments),
                edgecolor="none",
                linewidth=0,
                clip_on=False,
            )
        )

    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.spines["right"].set_visible(True)
    cax.spines["right"].set_color("#444444")
    cax.spines["right"].set_linewidth(1.0)
    cax.spines["left"].set_visible(True)
    cax.spines["left"].set_color("#444444")
    cax.spines["left"].set_linewidth(1.0)

    cax.set_xticks([])
    cax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    cax.set_yticklabels(["Low", "0.25", "0.50", "0.75", "High"])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.tick_params(axis="y", labelsize=10.2, width=1.1, length=4.5, pad=4)
    cax.set_ylabel("Risk", fontsize=12.0, fontweight="heavy", labelpad=12)


def _plot_ge_mapping(ax: plt.Axes, cax: plt.Axes, risk_data: pd.DataFrame) -> None:
    stress_df = risk_data[
        risk_data["perturbation_group"].isin(["Clean Reference", "Stress Severity"])
        & risk_data["stress_lambda"].isin(LAMBDA_ORDER)
    ].copy()
    stress_cells = aggregate_cells(
        stress_df, ["instruction_task_family", "stress_lambda"]
    )
    values, counts = pivot_matrix(
        stress_cells,
        TASK_FAMILY_ORDER,
        LAMBDA_ORDER,
        "instruction_task_family",
        "stress_lambda",
    )
    lambda_stars = lambda_star(stress_cells)

    cmap = LinearSegmentedColormap.from_list(
        "resilience_risk",
        ["#08783F", "#7FBF3F", "#F4D44D", "#F08A24", "#C5161D"],
    )
    cmap.set_bad("#EEEEEE")

    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    row_tick_labels = [
        "Navigation",
        "Object\nTransport",
        "Multi-room\nRearrangement",
        "Collaborative\nHousehold",
    ]
    ax.set_xticks(np.arange(len(LAMBDA_ORDER)))
    ax.set_xticklabels(
        [f"{value:.1f}" for value in LAMBDA_ORDER],
        fontsize=TICK_SIZE,
        fontweight="semibold",
    )
    ax.set_yticks(np.arange(len(TASK_FAMILY_ORDER)))
    ax.set_yticklabels(row_tick_labels, fontsize=11.0, fontweight="bold")
    ax.set_xlabel(
        "Stress severity $\\lambda$",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="semibold",
    )
    ax.set_ylabel("")
    _set_panel_title(ax, "(d) GE: Task-Stress Extensibility")

    ax.set_xticks(np.arange(-0.5, len(LAMBDA_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TASK_FAMILY_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isfinite(values[i, j]):
                continue
            color = "white" if values[i, j] > 0.62 else PALETTE["dark_text"]
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10.0,
                fontweight="bold",
                color=color,
            )

    for i, family in enumerate(TASK_FAMILY_ORDER):
        star = lambda_stars.get(family, math.nan)
        label = "$\\lambda^*$=--" if not np.isfinite(star) else f"$\\lambda^*$={star:.1f}"
        ax.text(
            len(LAMBDA_ORDER) + 0.02,
            i,
            label,
            ha="left",
            va="center",
            fontsize=10.4,
            fontweight="heavy",
            color=PALETTE["dark_text"],
            clip_on=False,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.8)

    _draw_manual_risk_colorbar(cax, cmap)


def plot_fig_q1_metric_validity_map() -> None:
    setup_style()
    ANALYSIS_EXPQ1_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_EXPQ1_DIR.mkdir(parents=True, exist_ok=True)

    ge = _clean_ge_rows(load_ge())
    risk_data = compute_risk(load_rollouts(include_discovered=True))

    fig = plt.figure(figsize=(13.6, 8.25))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.03, 1.12],
        height_ratios=[1.0, 1.06],
        hspace=0.33,
        wspace=0.29,
    )

    ax_rebound = fig.add_subplot(grid[0, 0])
    ax_stability = fig.add_subplot(grid[0, 1])
    ax_ge_traj = fig.add_subplot(grid[1, 0])
    ge_map_grid = grid[1, 1].subgridspec(
        1,
        2,
        width_ratios=[1.0, 0.055],
        wspace=0.72,
    )
    ax_ge_map = fig.add_subplot(ge_map_grid[0, 0])
    cax_ge_map = fig.add_subplot(ge_map_grid[0, 1])

    _plot_distributional_validity(ax_rebound, ax_stability)
    _plot_ge_trajectory(ax_ge_traj, ge, risk_data)
    _plot_ge_mapping(ax_ge_map, cax_ge_map, risk_data)

    out_png = ANALYSIS_EXPQ1_DIR / "fig_q1_metric_validity_map.png"
    save_figure(fig, out_png)

    for suffix in [".png", ".pdf"]:
        src = out_png.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, PAPER_EXPQ1_DIR / src.name)
            print(f"  [copy] {PAPER_EXPQ1_DIR / src.name}")


if __name__ == "__main__":
    plot_fig_q1_metric_validity_map()
