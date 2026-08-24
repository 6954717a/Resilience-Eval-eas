"""
plot_task_family_resilience_heatmap.py
======================================
Build a two-panel task-family resilience heatmap from Exp-Q1/GE rollouts.

Panel (a): instruction-derived task family x stress severity lambda.
Panel (b): controlled perturbation type x stress severity lambda.

The script intentionally uses deterministic instruction rules instead of an
LLM classifier so that the figure is reproducible and auditable.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import PALETTE, setup_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "Data_Exper"
OUT_IMG_DIR = DATA_DIR / "Analysis" / "images" / "ExpQ1"
OUT_TABLE_DIR = DATA_DIR / "Analysis" / "tables"

TASK_FAMILY_ORDER = [
    "Navigation",
    "Object Transport",
    "Multi-room Rearrangement",
    "Collaborative Household",
]

PERTURBATION_ORDER = [
    "Clean Reference",
    "Semantic Rewrite",
    "Object-state Change",
    "Stress Severity",
]

LAMBDA_ORDER = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

ROOM_TERMS = [
    "bathroom",
    "bedroom",
    "dining room",
    "entryway",
    "hallway",
    "kitchen",
    "laundry room",
    "living room",
    "office",
]

HOUSEHOLD_PATTERNS = [
    r"\bset the table\b",
    r"\btidy\b",
    r"\brelaxing atmosphere\b",
    r"\bat each setting\b",
    r"\bhousehold\b",
    r"\bclean\b",
    r"\bcook\b",
    r"\bserve\b",
    r"\bprepare\b",
]

TEMPORAL_PATTERNS = [
    r"\bfirst\b",
    r"\bthen\b",
    r"\bnext\b",
    r"\bafter\b",
    r"\bbefore\b",
    r"\brelocate\b",
]

ACTION_PATTERN = re.compile(r"\b(move|put|place|take|relocate|bring|carry|transfer)\b")


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    label: str


DEFAULT_SOURCES = [
    SourceSpec(
        DATA_DIR
        / "Exp0430"
        / "2026-04-30_04-07-07-val_mini.json"
        / "results"
        / "expq1_ge"
        / "ge_capacity_validity"
        / "raw_rollouts.csv",
        "Exp0430_GE_formal",
    ),
    SourceSpec(
        DATA_DIR
        / "Exp0428"
        / "2026-04-27_09-53-39-val_mini.json"
        / "results"
        / "expq1"
        / "ge_validity"
        / "raw_rollouts.csv",
        "Exp0428_GE_supplement",
    ),
    SourceSpec(
        DATA_DIR
        / "Exp0428"
        / "2026-04-27_15-52-44-val_mini.json"
        / "results"
        / "expq1"
        / "rebound_validity"
        / "raw_rollouts.csv",
        "Exp0428_rebound_perturbations",
    ),
    SourceSpec(
        DATA_DIR
        / "Exp0428"
        / "2026-04-27_09-53-39-val_mini.json"
        / "results"
        / "expq1"
        / "stability_validity"
        / "raw_rollouts.csv",
        "Exp0428_stability_perturbations",
    ),
    SourceSpec(
        DATA_DIR
        / "Exp0425"
        / "exper1"
        / "2026-04-23_06-45-59-val_mini.json"
        / "results"
        / "expq1"
        / "rebound_validity"
        / "raw_rollouts.csv",
        "Exp0425_rebound_perturbations",
    ),
    SourceSpec(
        DATA_DIR
        / "Exp0425"
        / "exper1"
        / "2026-04-23_06-45-59-val_mini.json"
        / "results"
        / "expq1"
        / "stability_validity"
        / "raw_rollouts.csv",
        "Exp0425_stability_perturbations",
    ),
]


def _as_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _contains_any(text: str, patterns: Iterable[str]) -> list[str]:
    return [pat for pat in patterns if re.search(pat, text)]


def _room_mentions(text: str) -> list[str]:
    return [room for room in ROOM_TERMS if room in text]


def _sentence_count(text: str) -> int:
    parts = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    return max(1, len(parts))


def _approx_object_complexity(text: str) -> int:
    # Approximate object multiplicity from comma-separated and "and" phrases.
    comma_count = text.count(",")
    and_count = len(re.findall(r"\band\b", text))
    plural_markers = len(re.findall(r"\b(items|toys|plates|glasses|forks|objects)\b", text))
    return comma_count + and_count + plural_markers


def classify_instruction(instruction: str) -> tuple[str, str, str]:
    """Classify an instruction into one dominant task family plus audit rules."""
    text = str(instruction or "").strip().lower()
    rooms = _room_mentions(text)
    household_hits = _contains_any(text, HOUSEHOLD_PATTERNS)
    temporal_hits = _contains_any(text, TEMPORAL_PATTERNS)
    action_hits = ACTION_PATTERN.findall(text)
    sentence_count = _sentence_count(text)
    object_complexity = _approx_object_complexity(text)
    help_me = "help me" in text

    rules = []
    if help_me:
        rules.append("help_me")
    if household_hits:
        rules.extend(f"household:{hit}" for hit in household_hits)
    if temporal_hits:
        rules.extend(f"temporal:{hit}" for hit in temporal_hits)
    if len(rooms) >= 2:
        rules.append(f"rooms:{'|'.join(rooms)}")
    if len(action_hits) > 0:
        rules.append(f"actions:{len(action_hits)}")
    if object_complexity > 0:
        rules.append(f"object_complexity:{object_complexity}")

    # Dominant-label priority is chosen for visual interpretability. Generic
    # "help me move" alone is recorded, but not enough to override a route or
    # transport task into the collaborative-household row.
    if household_hits:
        label = "Collaborative Household"
    elif temporal_hits or (len(rooms) >= 2 and (sentence_count >= 2 or len(action_hits) >= 2)):
        label = "Multi-room Rearrangement"
    elif len(rooms) >= 2 and len(action_hits) <= 2 and object_complexity <= 1:
        label = "Navigation"
    elif len(action_hits) > 0:
        label = "Object Transport"
    else:
        label = "Navigation"

    return label, "|".join(rooms), ";".join(rules)


def map_perturbation(row: pd.Series) -> str | None:
    perturbation = str(row.get("perturbation_type", "")).strip().lower()
    stress_lambda = float(row.get("stress_lambda", 0.0) or 0.0)

    if perturbation == "clean" or stress_lambda == 0.0 and perturbation in {"", "nan"}:
        return "Clean Reference"
    if perturbation in {"surface_rewrite", "filler_injection"}:
        return "Semantic Rewrite"
    if perturbation in {"object_state_toggle", "state_toggle", "object_state_change"}:
        return "Object-state Change"
    if perturbation in {"irrelevant_perturbation", "stress_perturbation"} or stress_lambda > 0:
        return "Stress Severity"
    return None


def discover_sources(include_discovered: bool) -> list[SourceSpec]:
    sources = [src for src in DEFAULT_SOURCES if src.path.exists()]
    if not include_discovered:
        return sources

    known = {src.path.resolve() for src in sources}
    # Keep the main figure on evaluation/validity runs. Exp-Q3 optimization
    # folders are intentionally excluded because they answer a different claim.
    for root in [DATA_DIR / "Exp0428", DATA_DIR / "Exp0425", DATA_DIR / "Exp_GE"]:
        if not root.exists():
            continue
        for path in root.rglob("raw_rollouts.csv"):
            if "conditions" in path.parts:
                continue
            resolved = path.resolve()
            if resolved in known:
                continue
            sources.append(SourceSpec(path, f"discovered:{path.parent.name}"))
            known.add(resolved)
    return sources


def load_rollouts(include_discovered: bool = True) -> pd.DataFrame:
    rows = []
    required = {"instruction", "perturbation_type", "stress_lambda", "task_percent_complete"}
    for source in discover_sources(include_discovered):
        try:
            df = pd.read_csv(source.path)
        except Exception as exc:
            print(f"[warn] failed to read {source.path}: {exc}")
            continue
        if not required.issubset(df.columns):
            print(f"[skip] missing required columns: {source.path}")
            continue
        df = df.copy()
        df["source_label"] = source.label
        df["source_path"] = str(source.path.relative_to(ROOT))
        rows.append(df)

    if not rows:
        raise FileNotFoundError("No usable raw_rollouts.csv files were found.")

    data = pd.concat(rows, ignore_index=True, sort=False)
    data["stress_lambda"] = _as_numeric(data["stress_lambda"], 0.0)
    data["task_percent_complete"] = _as_numeric(data["task_percent_complete"], 0.0).clip(0, 1)
    if "task_state_success" not in data.columns:
        data["task_state_success"] = np.nan
    data["task_state_success"] = _as_numeric(data["task_state_success"], 0.0).clip(0, 1)

    for col in [
        "ge_contract_margin_min",
        "ge_boundary_hit",
        "ge_hard_boundary_hit",
        "ge_cliff_flag",
        "ge_cell_valid",
    ]:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")

    classified = data["instruction"].apply(classify_instruction)
    data["instruction_task_family"] = classified.apply(lambda item: item[0])
    data["instruction_room_mentions"] = classified.apply(lambda item: item[1])
    data["instruction_classification_rules"] = classified.apply(lambda item: item[2])
    data["perturbation_group"] = data.apply(map_perturbation, axis=1)
    data = data[data["perturbation_group"].notna()].copy()
    return data


def compute_risk(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    margin = pd.to_numeric(df["ge_contract_margin_min"], errors="coerce")
    cell_valid = pd.to_numeric(df["ge_cell_valid"], errors="coerce")
    margin_valid = margin.notna() & (margin > -0.999) & ((margin.abs() > 1e-8) | (cell_valid == 1))

    # A smooth, sign-aware margin risk: negative margins approach high risk,
    # positive margins approach low risk. Placeholder margins use outcome risk.
    margin_risk = 1.0 / (1.0 + np.exp(4.0 * margin.clip(-2, 2)))
    completion_risk = 1.0 - df["task_percent_complete"].clip(0, 1)
    success_risk = 1.0 - df["task_state_success"].clip(0, 1)
    outcome_risk = (0.75 * completion_risk + 0.25 * success_risk).clip(0, 1)
    event_risk = (
        0.45 * pd.to_numeric(df["ge_boundary_hit"], errors="coerce").fillna(0).clip(0, 1)
        + 0.35 * pd.to_numeric(df["ge_hard_boundary_hit"], errors="coerce").fillna(0).clip(0, 1)
        + 0.20 * pd.to_numeric(df["ge_cliff_flag"], errors="coerce").fillna(0).clip(0, 1)
    ).clip(0, 1)

    df["margin_valid_for_risk"] = margin_valid.astype(int)
    df["risk_margin_component"] = np.where(margin_valid, margin_risk, np.nan)
    df["risk_outcome_component"] = outcome_risk
    df["risk_event_component"] = event_risk
    df["resilience_risk"] = np.where(
        margin_valid,
        0.55 * margin_risk + 0.30 * outcome_risk + 0.15 * event_risk,
        0.70 * outcome_risk + 0.30 * event_risk,
    )
    df["resilience_risk"] = pd.to_numeric(df["resilience_risk"], errors="coerce").clip(0, 1)
    return df


def aggregate_cells(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            mean_risk=("resilience_risk", "mean"),
            mean_margin=("ge_contract_margin_min", "mean"),
            mean_completion=("task_percent_complete", "mean"),
            success_rate=("task_state_success", "mean"),
            boundary_rate=("ge_boundary_hit", "mean"),
            hard_boundary_rate=("ge_hard_boundary_hit", "mean"),
            cliff_rate=("ge_cliff_flag", "mean"),
            mean_lambda=("stress_lambda", "mean"),
            valid_margin_rate=("margin_valid_for_risk", "mean"),
            n_rows=("resilience_risk", "size"),
            n_episodes=("episode_id", lambda x: x.nunique() if "episode_id" in df.columns else len(x)),
            sources=("source_label", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    grouped["low_support"] = grouped["n_rows"] < 3
    return grouped


def lambda_star(stress_cells: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for family in TASK_FAMILY_ORDER:
        cells = stress_cells[stress_cells["instruction_task_family"] == family].copy()
        if cells.empty:
            result[family] = math.nan
            continue
        healthy = cells[
            (cells["mean_completion"] >= 0.5)
            & (cells["boundary_rate"].fillna(0) < 0.5)
            & (cells["cliff_rate"].fillna(0) < 0.5)
            & (cells["mean_risk"] <= 0.65)
        ]
        result[family] = float(healthy["stress_lambda"].max()) if not healthy.empty else math.nan
    return result


def pivot_matrix(
    cells: pd.DataFrame,
    rows: list[str],
    cols: list[float] | list[str],
    row_col: str,
    col_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(rows), len(cols)), np.nan)
    counts = np.zeros((len(rows), len(cols)), dtype=int)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            match = cells[(cells[row_col] == row) & (cells[col_col] == col)]
            if not match.empty:
                values[i, j] = float(match["mean_risk"].iloc[0])
                counts[i, j] = int(match["n_rows"].iloc[0])
    return values, counts


def draw_heatmap_panel(
    ax: plt.Axes,
    values: np.ndarray,
    counts: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    xlabel: str,
    cmap: LinearSegmentedColormap,
) -> None:
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="semibold")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10.5, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=11.5, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color=PALETTE["dark_text"])

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.8)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                color = "white" if values[i, j] > 0.62 else PALETTE["dark_text"]
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="semibold",
                    color=color,
                )
            else:
                # Missing cells are intentionally left blank and grey. This is
                # important for perturbation x lambda because only physical
                # stress has a full lambda sweep in the current experiments.
                pass
    return im


def write_instruction_audit(df: pd.DataFrame) -> None:
    group_cols = [
        "episode_id",
        "instruction",
        "instruction_task_family",
        "instruction_room_mentions",
        "instruction_classification_rules",
    ]
    optional_cols = [col for col in ["task_family"] if col in df.columns]
    audit = (
        df.groupby(group_cols, dropna=False)
        .agg(
            source_labels=("source_label", lambda x: ";".join(sorted(set(map(str, x))))),
            source_paths=("source_path", lambda x: ";".join(sorted(set(map(str, x))))),
            source_task_families=(
                optional_cols[0] if optional_cols else "instruction_task_family",
                lambda x: ";".join(sorted(set(map(str, x.dropna())))),
            ),
            n_rollout_rows=("instruction", "size"),
        )
        .reset_index()
        .sort_values(["instruction_task_family", "episode_id"])
    )
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_TABLE_DIR / "task_instruction_classification.csv", index=False)


def write_cell_audit(stress_cells: pd.DataFrame, perturb_cells: pd.DataFrame) -> None:
    stress_out = stress_cells.copy()
    stress_out["panel"] = "task_family_by_lambda"
    perturb_out = perturb_cells.copy()
    perturb_out["panel"] = "perturbation_by_lambda"
    combined = pd.concat([stress_out, perturb_out], ignore_index=True, sort=False)
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_TABLE_DIR / "task_family_heatmap_cells.csv", index=False)


def save_single_heatmap(
    values: np.ndarray,
    counts: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    xlabel: str,
    output_stem: str,
    cmap: LinearSegmentedColormap,
    lambda_stars: dict[str, float] | None = None,
) -> None:
    fig = plt.figure(figsize=(8.6, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.045])
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])

    im = draw_heatmap_panel(ax, values, counts, row_labels, col_labels, title, xlabel, cmap)
    if lambda_stars:
        for i, family in enumerate(row_labels):
            star = lambda_stars.get(family, math.nan)
            label = "$\\lambda^*$ = --" if not np.isfinite(star) else f"$\\lambda^*$ = {star:.1f}"
            ax.text(
                len(col_labels) - 0.18,
                i,
                label,
                ha="left",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=PALETTE["dark_text"],
                clip_on=False,
            )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Risk", fontsize=10.5, fontweight="bold", labelpad=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["Low", "0.25", "0.50", "0.75", "High"])
    cbar.ax.tick_params(labelsize=9.5)

    png_path = OUT_IMG_DIR / f"{output_stem}.png"
    pdf_path = OUT_IMG_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[save] {png_path}")
    print(f"[save] {pdf_path}")


def plot_task_family_heatmap(df: pd.DataFrame) -> None:
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    stress_df = df[
        df["perturbation_group"].isin(["Clean Reference", "Stress Severity"])
        & df["stress_lambda"].isin(LAMBDA_ORDER)
    ].copy()
    perturb_df = df[
        df["perturbation_group"].isin(PERTURBATION_ORDER)
        & df["stress_lambda"].isin(LAMBDA_ORDER)
    ].copy()

    stress_cells = aggregate_cells(stress_df, ["instruction_task_family", "stress_lambda"])
    perturb_cells = aggregate_cells(perturb_df, ["perturbation_group", "stress_lambda"])
    write_cell_audit(stress_cells, perturb_cells)
    write_instruction_audit(df)

    stress_values, stress_counts = pivot_matrix(
        stress_cells,
        TASK_FAMILY_ORDER,
        LAMBDA_ORDER,
        "instruction_task_family",
        "stress_lambda",
    )
    perturb_values, perturb_counts = pivot_matrix(
        perturb_cells,
        PERTURBATION_ORDER,
        LAMBDA_ORDER,
        "perturbation_group",
        "stress_lambda",
    )
    lambda_stars = lambda_star(stress_cells)

    cmap = LinearSegmentedColormap.from_list(
        "resilience_risk",
        ["#08783F", "#7FBF3F", "#F4D44D", "#F08A24", "#C5161D"],
    )
    cmap.set_bad("#EEEEEE")

    stress_labels = [f"{value:.1f}" for value in LAMBDA_ORDER]
    perturb_labels = ["Clean Reference", "Semantic Rewrite", "Object-state Change", "Physical / Stress"]

    save_single_heatmap(
        stress_values,
        stress_counts,
        TASK_FAMILY_ORDER,
        stress_labels,
        "Task family × stress severity",
        "Stress severity $\\lambda$",
        "fig_task_family_stress_heatmap",
        cmap,
        lambda_stars=lambda_stars,
    )
    save_single_heatmap(
        perturb_values,
        perturb_counts,
        perturb_labels,
        stress_labels,
        "Perturbation type × stress severity",
        "Stress severity $\\lambda$",
        "fig_perturbation_stress_heatmap",
        cmap,
    )

    fig = plt.figure(figsize=(14.0, 5.15), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.0, 0.045])
    ax_stress = fig.add_subplot(grid[0, 0])
    ax_perturb = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])

    im = draw_heatmap_panel(
        ax_stress,
        stress_values,
        stress_counts,
        TASK_FAMILY_ORDER,
        stress_labels,
        "(a) GE stress-response map",
        "Stress severity $\\lambda$",
        cmap,
    )
    for i, family in enumerate(TASK_FAMILY_ORDER):
        star = lambda_stars.get(family, math.nan)
        label = "$\\lambda^*$ = --" if not np.isfinite(star) else f"$\\lambda^*$ = {star:.1f}"
        ax_stress.text(
            len(LAMBDA_ORDER) - 0.18,
            i,
            label,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=PALETTE["dark_text"],
            clip_on=False,
        )

    draw_heatmap_panel(
        ax_perturb,
        perturb_values,
        perturb_counts,
        perturb_labels,
        stress_labels,
        "(b) Perturbation-stress map",
        "Stress severity $\\lambda$",
        cmap,
    )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Risk", fontsize=10.5, fontweight="bold", labelpad=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["Low", "0.25", "0.50", "0.75", "High"])
    cbar.ax.tick_params(labelsize=9.5)

    fig.suptitle(
        "Task-family heatmap for embodied resilience evaluation",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["dark_text"],
    )
    fig.text(
        0.5,
        -0.025,
        "Cell color aggregates contract-margin risk, boundary/cliff events, and completion loss. "
        "Cell text reports mean risk.",
        ha="center",
        va="top",
        fontsize=10,
        color=PALETTE["neutral"],
    )

    png_path = OUT_IMG_DIR / "fig_task_family_resilience_heatmap.png"
    pdf_path = OUT_IMG_DIR / "fig_task_family_resilience_heatmap.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[save] {png_path}")
    print(f"[save] {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-discovered-sources",
        action="store_true",
        help="Use only the curated default sources and skip additional Exp0428/Exp0425/Exp0430/Exp_GE scans.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_rollouts(include_discovered=not args.no_discovered_sources)
    data = compute_risk(data)
    plot_task_family_heatmap(data)
    print(f"[save] {OUT_TABLE_DIR / 'task_family_heatmap_cells.csv'}")
    print(f"[save] {OUT_TABLE_DIR / 'task_instruction_classification.csv'}")
    print(
        "[summary] rows={rows} families={families} perturbations={perturbations}".format(
            rows=len(data),
            families=sorted(data["instruction_task_family"].unique()),
            perturbations=sorted(data["perturbation_group"].unique()),
        )
    )


if __name__ == "__main__":
    main()
