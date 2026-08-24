"""
plot_ge_typical_case_response.py
================================
Generate a paper-ready Graceful Extensibility stress-response figure for a
representative experimental case.

The plot follows the visual grammar of the GE contract-margin schematic:
stress severity on the x-axis, normalized operational margin on the y-axis,
a solid graceful response, a dashed brittle contrast, and the contract
threshold at zero.

The empirical anchors come from a complete stress sweep for a spatial-relation
table-setting episode. Because the available Exp-Q2 benchmark table does not
contain per-method stress-sweep margins, this figure uses a clearly labelled
operational margin proxy derived from completion and recovery cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_style import PALETTE, setup_style


ROOT = Path(__file__).resolve().parents[3]
SOURCE_CSV = (
    ROOT
    / "Data_Exper"
    / "Exp0430"
    / "2026-04-30_04-07-07-val_mini.json"
    / "results"
    / "expq1_ge"
    / "ge_capacity_validity"
    / "raw_rollouts.csv"
)
ANALYSIS_IMG_DIR = ROOT / "Data_Exper" / "Analysis" / "images"
ANALYSIS_TABLE_DIR = ROOT / "Data_Exper" / "Analysis" / "tables"
PAPER_IMG_DIR = ROOT / "paper_writing" / "Images" / "results"

CASE_EPISODE_ID = 129
CASE_FAMILY = "spatial_relation"
CASE_TITLE = "Table-setting spatial relation"

# Visual proxy only: completion dominates, while recovery burden discounts
# cases that reach completion through expensive recovery.
COMPLETION_WEIGHT = 0.72
RECOVERY_WEIGHT = 0.28
CONTRACT_HEALTH_THRESHOLD = 0.60


def _load_case() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(SOURCE_CSV)

    df = pd.read_csv(SOURCE_CSV)
    required = {
        "episode_id",
        "task_family",
        "stress_lambda",
        "task_percent_complete",
        "rebound_c_rec",
        "instruction",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {SOURCE_CSV}: {missing}")

    case = df[
        (df["episode_id"].astype(int) == CASE_EPISODE_ID)
        & (df["task_family"].astype(str) == CASE_FAMILY)
    ].copy()
    if case.empty:
        raise ValueError(
            f"No rows for episode_id={CASE_EPISODE_ID}, task_family={CASE_FAMILY}"
        )

    case = (
        case.groupby("stress_lambda", as_index=False)
        .agg(
            task_percent_complete=("task_percent_complete", "mean"),
            rebound_c_rec=("rebound_c_rec", "mean"),
            n=("stress_lambda", "size"),
            instruction=("instruction", "first"),
        )
        .sort_values("stress_lambda")
        .reset_index(drop=True)
    )
    return case


def _compute_margin_proxy(case: pd.DataFrame) -> pd.DataFrame:
    case = case.copy()
    cost_cap = max(float(case["rebound_c_rec"].max()), 1e-9)
    recovery_health = 1.0 - np.clip(case["rebound_c_rec"].to_numpy() / cost_cap, 0, 1)
    completion = np.clip(case["task_percent_complete"].to_numpy(), 0, 1)
    operational_health = (
        COMPLETION_WEIGHT * completion + RECOVERY_WEIGHT * recovery_health
    )
    raw_margin = operational_health - CONTRACT_HEALTH_THRESHOLD
    positive_margin = np.clip(raw_margin, 0, None)
    normalizer = max(float(positive_margin.max()), 1e-9)

    case["recovery_health"] = recovery_health
    case["operational_health"] = operational_health
    case["margin_proxy"] = raw_margin
    case["normalized_margin_proxy"] = positive_margin / normalizer
    case["cost_cap"] = cost_cap
    return case


def _first_threshold_crossing(lambdas: np.ndarray, margins: np.ndarray) -> float | None:
    for i in range(1, len(lambdas)):
        prev_margin = margins[i - 1]
        cur_margin = margins[i]
        if prev_margin >= 0 and cur_margin < 0:
            span = lambdas[i] - lambdas[i - 1]
            denom = prev_margin - cur_margin
            if abs(denom) < 1e-12:
                return float(lambdas[i])
            return float(lambdas[i - 1] + span * (prev_margin / denom))
    return None


def _smooth_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dense_x = np.linspace(float(x.min()), float(x.max()), 240)
    try:
        from scipy.interpolate import PchipInterpolator

        dense_y = PchipInterpolator(x, y)(dense_x)
    except Exception:
        dense_y = np.interp(dense_x, x, y)
    return dense_x, np.clip(dense_y, 0, 1.02)


def _brittle_contrast(x: np.ndarray) -> np.ndarray:
    # A shape reference, not a measured baseline: high margin under mild stress,
    # followed by an abrupt collapse once stress crosses the latent boundary.
    return 0.96 / (1.0 + np.exp(62.0 * (x - 0.66))) + 0.015


def plot_ge_typical_case_response() -> None:
    setup_style()
    ANALYSIS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_IMG_DIR.mkdir(parents=True, exist_ok=True)

    case = _compute_margin_proxy(_load_case())
    audit_path = ANALYSIS_TABLE_DIR / "ge_typical_case_response_episode129.csv"
    case.to_csv(audit_path, index=False)

    lambdas = case["stress_lambda"].to_numpy(dtype=float)
    y_raw = case["normalized_margin_proxy"].to_numpy(dtype=float)
    margin = case["margin_proxy"].to_numpy(dtype=float)

    # Once the observed proxy reaches the contract boundary, use the guide curve
    # as a boundary envelope; raw anchor labels still show measured values.
    y_guide = y_raw.copy()
    zero_hits = np.where(y_guide <= 1e-9)[0]
    if len(zero_hits):
        y_guide[zero_hits[0] :] = 0.0

    x_smooth, y_smooth = _smooth_line(lambdas, y_guide)
    x_ref = np.linspace(0, 1.02, 260)
    y_brittle = _brittle_contrast(x_ref)
    crossing = _first_threshold_crossing(lambdas, margin)

    fig, ax = plt.subplots(figsize=(8.4, 4.7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    graceful_color = "#14796B"
    brittle_color = "#D84A45"
    marker_face = "#E6F2EE"
    threshold_color = "#30343B"

    ax.plot(
        x_smooth,
        y_smooth,
        color=graceful_color,
        linewidth=3.0,
        solid_capstyle="round",
        label="Graceful case response",
        zorder=4,
    )
    ax.plot(
        x_ref,
        y_brittle,
        color=brittle_color,
        linewidth=2.6,
        linestyle=(0, (7, 4)),
        label="Brittle contrast",
        zorder=3,
    )
    ax.scatter(
        lambdas,
        y_raw,
        s=86,
        facecolor=marker_face,
        edgecolor=graceful_color,
        linewidth=1.7,
        zorder=5,
        label="Measured anchors",
    )

    for x, y, raw_margin in zip(lambdas, y_raw, margin):
        y_text = min(y + 0.055, 1.04)
        ax.text(
            x,
            y_text,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["dark_text"],
        )

    ax.axhline(0, color=threshold_color, linewidth=1.8, zorder=2)
    ax.text(
        1.035,
        0.035,
        "Contract\nthreshold (0)",
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=threshold_color,
        clip_on=False,
    )

    if crossing is not None:
        ax.axvline(
            crossing,
            ymin=0.0,
            ymax=0.18,
            color=threshold_color,
            linewidth=1.3,
            linestyle=(0, (2, 3)),
            alpha=0.85,
        )
        ax.annotate(
            rf"first boundary crossing $\approx {crossing:.2f}$",
            xy=(crossing, 0.02),
            xytext=(0.45, 0.22),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color=threshold_color, lw=1.2),
            fontsize=10,
            color=threshold_color,
            ha="left",
            va="center",
        )

    ax.set_title(
        f"Typical GE Case: {CASE_TITLE} (episode {CASE_EPISODE_ID})",
        loc="left",
        pad=12,
        fontsize=13,
        fontweight="bold",
        color=PALETTE["dark_text"],
    )

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks(np.linspace(0, 1.0, 6))
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_xlabel(r"Stress Severity $\lambda$", fontsize=12.5, fontweight="bold")
    ax.set_ylabel(
        r"Normalized Margin Proxy $M_f(\lambda)$",
        fontsize=12.5,
        fontweight="bold",
    )

    ax.grid(axis="y", color="#D8DEE4", linewidth=0.8, alpha=0.55)
    ax.grid(axis="x", color="#E8EBEF", linewidth=0.5, alpha=0.35)
    ax.spines["left"].set_color(threshold_color)
    ax.spines["bottom"].set_color(threshold_color)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)

    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.05, 0.20),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D8DEE4",
        fontsize=10,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.8)

    outputs = [
        ANALYSIS_IMG_DIR / "fig_ge_typical_case_response.png",
        PAPER_IMG_DIR / "expq_ge_stress_response.png",
    ]
    for png_path in outputs:
        pdf_path = png_path.with_suffix(".pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"[save] {png_path}")
        print(f"[save] {pdf_path}")
    print(f"[save] {audit_path}")

    plt.close(fig)


if __name__ == "__main__":
    plot_ge_typical_case_response()
