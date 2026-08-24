"""
plot_expq2_method_metric_heatmap.py
===================================
Generate the Exp-Q2 method-by-metric resilience heatmap.

The script reads the raw benchmark rows from Data_Class_2/BenchResults.tex,
applies direction-aligned min-max normalization, sorts methods by their
resilience-profile health, and exports publication figures plus a CSV audit
table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from plot_style import setup_style, PALETTE


ROOT = Path(__file__).resolve().parents[3]
BENCH_TEX = ROOT / "Data_Class_2" / "BenchResults.tex"
OUT_DIR = ROOT / "paper_writing" / "Images" / "results"
TABLE_DIR = ROOT / "Data_Exper" / "Analysis" / "tables"

METHOD_ORDER = [
    "ReAct",
    "Reflexion",
    "CycleVLA",
    "CLARE",
    "SayCan",
    "AgentEvolver",
    "InnerMono",
    "CoPAL",
    "SMART-LLM",
    "Baseline",
]

METRICS = [
    "C_rec",
    "beta",
    "lambda_star",
    "SR",
    "Comp.",
    "Steps",
]

# True means larger raw values are healthier; False means smaller values are
# healthier. The normalized heatmap always uses larger = healthier.
DIRECTION = {
    "C_rec": False,
    "beta": False,
    "lambda_star": True,
    "SR": True,
    "Comp.": True,
    "Steps": False,
}

GROUPS = [
    ("Rebound", ["C_rec"]),
    ("Stability", ["beta"]),
    ("Extensibility", ["lambda_star"]),
    ("Outcome Ref.", ["SR", "Comp."]),
    ("Simulation", ["Steps"]),
]

SORT_METRICS = [
    "C_rec",
    "beta",
    "lambda_star",
]


def _parse_number(cell: str) -> float:
    cell = cell.split("\\pm")[0].split("$\\pm$")[0]
    clean = (
        cell.replace("\\%", "")
        .replace("\\", "")
        .replace("$", "")
        .replace(",", "")
        .strip()
    )
    return float(clean)


def load_benchmark_table(path: Path = BENCH_TEX) -> pd.DataFrame:
    """Parse the benchmark rows from BenchResults.tex."""
    if not path.exists():
        raise FileNotFoundError(path)

    pattern = re.compile(r"\\textbf\{(?P<method>[^}]+)\}\s*&(?P<cells>.*?)\\\\")
    rows = []
    text = path.read_text(encoding="utf-8")
    for match in pattern.finditer(text):
        method = match.group("method").strip()
        if method not in METHOD_ORDER:
            continue
        cells = [c.strip() for c in match.group("cells").split("&")]
        if len(cells) != len(METRICS):
            raise ValueError(f"Unexpected metric count for {method}: {len(cells)}")
        row = {"Method": method}
        row.update({metric: _parse_number(cell) for metric, cell in zip(METRICS, cells)})
        rows.append(row)

    if len(rows) != len(METHOD_ORDER):
        found = sorted(row["Method"] for row in rows)
        raise ValueError(f"Expected {len(METHOD_ORDER)} methods, found {len(rows)}: {found}")

    df = pd.DataFrame(rows)
    df["Method"] = pd.Categorical(df["Method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values("Method").reset_index(drop=True)


def normalize_direction_aligned(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized health table where larger is healthier."""
    norm = df[["Method"]].copy()
    for metric in METRICS:
        col = df[metric].astype(float)
        lo, hi = col.min(), col.max()
        if hi == lo:
            norm[metric] = 0.5
        elif DIRECTION[metric]:
            norm[metric] = (col - lo) / (hi - lo)
        else:
            norm[metric] = (hi - col) / (hi - lo)
    norm["ProfileScore"] = norm[SORT_METRICS].mean(axis=1)
    return norm


def format_annotation(metric: str, value: float) -> str:
    if metric in {"SR", "Comp."}:
        return f"{value:.1f}%"
    if metric == "beta":
        return f"{value:.3f}"
    if metric == "lambda_star":
        return f"{value:.2f}"
    if metric == "Steps":
        return f"{value:.0f}"
    return f"{value:.1f}"


def plot_heatmap(df_raw: pd.DataFrame, df_norm: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    setup_style()

    ordered_methods = (
        df_norm.sort_values("ProfileScore", ascending=False)["Method"].astype(str).tolist()
    )
    raw = df_raw.set_index("Method").loc[ordered_methods]
    norm = df_norm.set_index("Method").loc[ordered_methods]

    # Academic sequential palette
    cmap = sns.color_palette("Blues", as_cmap=True)

    fig, ax = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True)
    values = norm[METRICS].to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRICS, rotation=35, ha="right", rotation_mode="anchor", fontsize=11, fontweight="bold")

    display_methods = [
        "Baseline (Ours)" if method == "Baseline" else method for method in ordered_methods
    ]
    ax.set_yticks(np.arange(len(ordered_methods)))
    ax.set_yticklabels(display_methods, fontsize=11)

    # Cell grid.
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered_methods), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5, alpha=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Group boundaries and labels.
    cursor = 0
    for group_idx, (label, group_metrics) in enumerate(GROUPS):
        start = cursor
        end = cursor + len(group_metrics) - 1
        if group_idx > 0:
            ax.axvline(start - 0.5, color=PALETTE["dark_text"], linewidth=1.8)
        ax.text(
            (start + end) / 2,
            -0.85,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=PALETTE["dark_text"],
            clip_on=False,
        )
        cursor = end + 1

    # Raw-value annotations; color changes only for legibility.
    for i, method in enumerate(ordered_methods):
        for j, metric in enumerate(METRICS):
            health = values[i, j]
            text_color = "black" if health < 0.65 else "white"
            fs = 8.5 if metric in ["SR", "Comp.", "Steps"] else 11
            ax.text(
                j,
                i,
                format_annotation(metric, float(raw.loc[method, metric])),
                ha="center",
                va="center",
                fontsize=fs,
                color=text_color,
                fontweight="bold",
            )

    # Highlight our current method wherever it appears after sorting.
    ours_idx = ordered_methods.index("Baseline")
    ax.add_patch(
        plt.Rectangle(
            (-0.5, ours_idx - 0.5),
            len(METRICS),
            1,
            fill=False,
            edgecolor=PALETTE["accent"],
            linewidth=3.0,
        )
    )
    ax.get_yticklabels()[ours_idx].set_color(PALETTE["accent"])
    ax.get_yticklabels()[ours_idx].set_fontweight("bold")

    ax.set_ylim(len(ordered_methods) - 0.5, -1.0)
    
    # Bottom sub-caption text similar to our standard style
    ax.set_xlabel("")
    ax.set_ylabel("Method Ranked by Health")
    ax.text(0.5, -0.15, "(a) Multidimensional Resilience Benchmark Profile", transform=ax.transAxes,
            fontsize=16, fontweight="semibold", ha="center", va="top", color=PALETTE["dark_text"])

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Normalized Health (0=Worst, 1=Best)", fontsize=11, fontweight="bold")

    pdf_path = OUT_DIR / "expq2_method_metric_heatmap.pdf"
    png_path = OUT_DIR / "expq2_method_metric_heatmap.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    audit = raw.reset_index()
    audit.insert(1, "ProfileScore", norm.loc[ordered_methods, "ProfileScore"].to_numpy())
    for metric in METRICS:
        audit[f"{metric}_health"] = norm.loc[ordered_methods, metric].to_numpy()
    audit.to_csv(TABLE_DIR / "expq2_method_metric_heatmap_scores.csv", index=False)

    print(f"[save] {pdf_path}")
    print(f"[save] {png_path}")
    print(f"[save] {TABLE_DIR / 'expq2_method_metric_heatmap_scores.csv'}")


def main() -> None:
    df_raw = load_benchmark_table()
    df_norm = normalize_direction_aligned(df_raw)
    plot_heatmap(df_raw, df_norm)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
