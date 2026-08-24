#!/usr/bin/env python3
"""Analyze Exp-Q1 resilience outputs for metric-validity evidence.

The script is intentionally standalone: it reads the two experiment folders
under Data_Exper by default, creates Data_Exper/Analysis, and writes tables,
figures, a manifest, and a short Chinese report.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


DEFAULT_RUN_NAMES = {
    "Qwen3-8B-Instruct": "2026-04-20_12-33-13-val_mini.json",
    "Qwen3.5-9B-Instruct": "2026-04-20_15-09-51-val_mini.json",
}

REBOUND_BLOCK = "rebound_validity"
STABILITY_BLOCK = "stability_validity"
GE_BLOCK = "ge_validity"

REQUIRED_GE_STRESS_LAMBDAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
REBOUND_PAIR_COLUMNS = (
    "episode_id",
    "perturbation_seed",
    "perturbation_type",
    "perturbation_intensity",
)

REBOUND_METRICS = [
    "c_rec_index_observed",
    "c_rec_observed",
    "rebound_c_rec_cog",
    "rebound_c_rec_phy",
    "rebound_c_rec_state_debt",
    "rebound_num_recovery_windows",
    "rebound_raw_window_count",
    "rebound_formal_window_count",
    "rebound_skipped_window_count",
    "rebound_c_rec_valid",
    "rebound_w_rem_star",
    "rebound_g_rec_total",
    "rebound_tracker_window_count",
    "task_percent_complete",
    "task_state_success",
]

STABILITY_METRICS = [
    "beta_hat",
    "beta_trajectory",
    "beta_out",
    "beta_action",
    "beta_progress",
    "beta_success",
]

NUMERIC_COLUMNS = {
    "episode_id",
    "anchor_id",
    "perturbation_seed",
    "perturbation_intensity",
    "stress_lambda",
    "rebound_baseline_loaded",
    "rebound_c_rec",
    "rebound_c_rec_index_100",
    "rebound_c_rec_cog",
    "rebound_c_rec_phy",
    "rebound_c_rec_state_debt",
    "rebound_num_recovery_windows",
    "rebound_raw_window_count",
    "rebound_formal_window_count",
    "rebound_skipped_window_count",
    "rebound_c_rec_valid",
    "rebound_w_rem_star",
    "rebound_g_rec_total",
    "rebound_tracker_window_count",
    "task_percent_complete",
    "task_state_success",
    "stability_beta_neighborhood",
    "stability_beta_trajectory",
    "stability_beta_out",
    "stability_beta_action",
    "stability_beta_progress",
    "stability_beta_success",
    "stability_beta_neighborhood_valid",
    "ge_margin_soft",
    "ge_audc_f",
    "ge_cell_valid",
    "ge_valid_lambda_coverage",
    "ge_boundary_lambda_star",
    "ge_boundary_hit",
    "ge_hard_boundary_hit",
    "beta_hat",
    "beta_trajectory",
    "beta_out",
    "n_clean",
    "n_perturbed",
}


@dataclass(frozen=True)
class RunSpec:
    model: str
    root: Path


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[4]


def default_data_root() -> Path:
    return repo_root_from_file() / "Data_Exper"


def parse_run_specs(values: Optional[Sequence[str]]) -> List[RunSpec]:
    data_root = default_data_root()
    if not values:
        return [
            RunSpec(model=model, root=data_root / folder)
            for model, folder in DEFAULT_RUN_NAMES.items()
        ]

    specs: List[RunSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --run value `{value}`. Expected format Label=PATH."
            )
        label, raw_path = value.split("=", 1)
        specs.append(RunSpec(model=label.strip(), root=Path(raw_path).resolve()))
    return specs


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    tables_dir = output_dir / "tables"
    images_dir = output_dir / "images"
    tables_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "tables": tables_dir, "images": images_dir}


def to_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_block_raw(spec: RunSpec, block: str) -> pd.DataFrame:
    path = spec.root / "results" / "expq1" / block / "raw_rollouts.csv"
    df = read_csv_if_exists(path)
    if df is None:
        return pd.DataFrame()
    df = to_numeric_columns(df)
    df["model"] = spec.model
    df["run_id"] = spec.root.name
    df["subexperiment_name"] = block
    return df


def load_all_raw(specs: Sequence[RunSpec], block: str) -> pd.DataFrame:
    frames = [load_block_raw(spec, block) for spec in specs]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_stability_beta(spec: RunSpec) -> pd.DataFrame:
    path = (
        spec.root
        / "results"
        / "expq1"
        / STABILITY_BLOCK
        / "conditions"
        / "default"
        / "analysis"
        / "statistics"
        / "stability_beta_neighborhood.csv"
    )
    df = read_csv_if_exists(path)
    if df is None:
        return pd.DataFrame()
    df = to_numeric_columns(df)
    df["model"] = spec.model
    df["run_id"] = spec.root.name
    return df


def load_all_stability_beta(specs: Sequence[RunSpec]) -> pd.DataFrame:
    frames = [load_stability_beta(spec) for spec in specs]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_ge_spec(path: Path) -> Dict[str, Any]:
    parts = list(path.parts)
    result: Dict[str, Any] = {
        "phase_label": "unknown",
        "spec_tag": "unknown",
        "perturbation_type": "unknown",
        "perturbation_seed": np.nan,
        "stress_lambda": np.nan,
    }
    try:
        runner_idx = parts.index("runner")
        result["phase_label"] = parts[runner_idx + 1]
    except Exception:
        pass
    spec_part = next((part for part in parts if part.startswith("spec_")), None)
    if spec_part:
        result["spec_tag"] = spec_part
        match = re.match(r"^spec_(?P<kind>.+)_seed(?P<seed>\d+)_lambda(?P<lam>[-+0-9.]+)$", spec_part)
        if match:
            result["perturbation_type"] = match.group("kind")
            result["perturbation_seed"] = int(match.group("seed"))
            result["stress_lambda"] = float(match.group("lam"))
    return result


def _missing_required_stress_lambdas(
    observed: Sequence[float],
    required: Sequence[float] = REQUIRED_GE_STRESS_LAMBDAS,
) -> List[float]:
    """Return planned GE stress levels absent from the observed sweep."""

    finite_observed = [float(value) for value in observed if np.isfinite(value)]
    return [
        float(target)
        for target in required
        if not any(
            math.isclose(float(target), value, rel_tol=0.0, abs_tol=1.0e-9)
            for value in finite_observed
        )
    ]


def load_ge_partial(spec: RunSpec) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ge_root = spec.root / "results" / "expq1" / GE_BLOCK
    csv_paths = sorted(ge_root.rglob("episode_metrics.csv")) if ge_root.exists() else []
    frames: List[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = read_csv_if_exists(csv_path)
        if df is None:
            continue
        df = to_numeric_columns(df)
        parsed = parse_ge_spec(csv_path)
        for key, value in parsed.items():
            if key not in df.columns or df[key].isna().all():
                df[key] = value
        df["model"] = spec.model
        df["run_id"] = spec.root.name
        df["source_csv"] = str(csv_path)
        frames.append(df)

    rows = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    margin_cells = sorted(ge_root.rglob("ge_boundary_cells.csv")) if ge_root.exists() else []
    margin_audc = sorted(ge_root.rglob("ge_boundary_curve_statistics.csv")) if ge_root.exists() else []
    block_summary = ge_root / "summary.csv"
    block_raw = ge_root / "raw_rollouts.csv"
    phases = sorted(rows["phase_label"].dropna().astype(str).unique().tolist()) if not rows.empty else []
    stress_lambdas = sorted(
        pd.to_numeric(rows.get("stress_lambda", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .unique()
        .tolist()
    ) if not rows.empty else []
    missing_stress_lambdas = _missing_required_stress_lambdas(stress_lambdas)
    stress_grid_complete = not missing_stress_lambdas
    stress_lambda_coverage = float(
        (len(REQUIRED_GE_STRESS_LAMBDAS) - len(missing_stress_lambdas))
        / len(REQUIRED_GE_STRESS_LAMBDAS)
    )
    has_stress_sweep = "stress_sweep" in phases
    formal_estimable = bool(
        block_summary.exists()
        and block_raw.exists()
        and margin_cells
        and margin_audc
        and has_stress_sweep
        and stress_grid_complete
    )
    reasons: List[str] = []
    if not block_summary.exists():
        reasons.append("missing_ge_block_summary")
    if not block_raw.exists():
        reasons.append("missing_ge_block_raw_rollouts")
    if not margin_cells:
        reasons.append("missing_margin_cells")
    if not margin_audc:
        reasons.append("missing_margin_audc")
    if not has_stress_sweep:
        reasons.append("missing_stress_sweep_rows")
    if not stress_grid_complete:
        reasons.append("requires_multi_stress_sweep")
    diagnostics = {
        "model": spec.model,
        "run_id": spec.root.name,
        "ge_root": str(ge_root),
        "episode_metrics_files": len(csv_paths),
        "partial_rows": int(len(rows)),
        "phases": phases,
        "stress_lambdas": stress_lambdas,
        "required_stress_lambdas": list(REQUIRED_GE_STRESS_LAMBDAS),
        "missing_stress_lambdas": missing_stress_lambdas,
        "stress_grid_complete": stress_grid_complete,
        "stress_lambda_coverage": stress_lambda_coverage,
        "margin_cells_files": [str(path) for path in margin_cells],
        "margin_audc_files": [str(path) for path in margin_audc],
        "formal_ge_estimable": formal_estimable,
        "incomplete_reason": ";".join(reasons) if reasons else "",
    }
    return rows, diagnostics


def load_all_ge_partial(specs: Sequence[RunSpec]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    frames: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, Any]] = []
    for spec in specs:
        rows, diag = load_ge_partial(spec)
        diagnostics.append(diag)
        if not rows.empty:
            frames.append(rows)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined, diagnostics


def finite_values(values: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_ci(values: Iterable[Any], rng: np.random.Generator, iters: int) -> Tuple[float, float]:
    arr = finite_values(values)
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    samples = rng.choice(arr, size=(iters, len(arr)), replace=True)
    means = np.mean(samples, axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cliffs_delta(a_values: Iterable[Any], b_values: Iterable[Any]) -> float:
    a = finite_values(a_values)
    b = finite_values(b_values)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = 0
    lt = 0
    for av in a:
        gt += int(np.sum(av > b))
        lt += int(np.sum(av < b))
    return float((gt - lt) / (len(a) * len(b)))


def summarize_metric_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    rng: np.random.Generator,
    bootstrap_iters: int,
    extra_counts: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    groupby_cols = list(group_cols)
    grouped = df.groupby(groupby_cols, dropna=False) if groupby_cols else [((), df)]
    for key, bucket in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, Any] = {col: value for col, value in zip(groupby_cols, key)}
        row["n_rows"] = int(len(bucket))
        if "episode_id" in bucket.columns:
            row["n_episode_ids"] = int(bucket["episode_id"].nunique(dropna=True))
        if extra_counts:
            for out_col, source_col in extra_counts.items():
                if source_col in bucket.columns:
                    row[out_col] = int(pd.to_numeric(bucket[source_col], errors="coerce").fillna(0).sum())
        for metric in metric_cols:
            if metric not in bucket.columns:
                continue
            vals = finite_values(bucket[metric])
            row[f"{metric}_n"] = int(len(vals))
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{metric}_median"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
            row[f"{metric}_min"] = float(np.min(vals)) if len(vals) else np.nan
            row[f"{metric}_max"] = float(np.max(vals)) if len(vals) else np.nan
            low, high = bootstrap_ci(vals, rng, bootstrap_iters)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize_pair_key_value(value: Any, column: str = "") -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if column in {"episode_id", "perturbation_type"}:
        text = str(value).strip()
        return text or None
    if column == "perturbation_seed":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return int(numeric) if math.isfinite(numeric) and numeric.is_integer() else None
    if column in {"perturbation_intensity", "stress_lambda"}:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _pair_key_set(frame: pd.DataFrame, pair_cols: Sequence[str]) -> set[Tuple[Any, ...]]:
    if frame.empty or not pair_cols:
        return set()
    return {
        tuple(
            _normalize_pair_key_value(value, column)
            for column, value in zip(pair_cols, values)
        )
        for values in frame[list(pair_cols)].itertuples(index=False, name=None)
        if all(
            _normalize_pair_key_value(value, column) is not None
            for column, value in zip(pair_cols, values)
        )
    }


def _serialize_pair_keys(
    keys: Iterable[Tuple[Any, ...]], pair_cols: Sequence[str]
) -> str:
    ordered = sorted(
        keys,
        key=lambda key: tuple("" if value is None else str(value) for value in key),
    )
    records = [dict(zip(pair_cols, key)) for key in ordered]
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def _serialize_missing_episode_ids(
    keys: Iterable[Tuple[Any, ...]], pair_cols: Sequence[str]
) -> str:
    if "episode_id" not in pair_cols:
        return "[]"
    episode_idx = list(pair_cols).index("episode_id")
    episode_ids = {
        key[episode_idx]
        for key in keys
        if key[episode_idx] is not None
    }
    ordered = sorted(episode_ids, key=lambda value: str(value))
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def _materialize_pairing_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Use perturbation intensity as the canonical fourth pairing coordinate."""

    if df.empty:
        return df
    has_intensity = "perturbation_intensity" in df.columns
    has_stress = "stress_lambda" in df.columns
    if not has_intensity and not has_stress:
        return df
    materialized = df.copy()
    intensity = (
        pd.to_numeric(materialized["perturbation_intensity"], errors="coerce")
        if has_intensity
        else pd.Series(np.nan, index=materialized.index, dtype=float)
    )
    if has_stress:
        intensity = intensity.combine_first(
            pd.to_numeric(materialized["stress_lambda"], errors="coerce")
        )
    materialized["perturbation_intensity"] = intensity
    return materialized


def paired_or_unpaired_test(
    df: pd.DataFrame,
    *,
    group_col: str,
    group_a: str,
    group_b: str,
    metric: str,
    pair_cols: Sequence[str],
    context: str,
    require_paired: bool = False,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "context": context,
        "metric": metric,
        "group_col": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "test_name": "",
        "n_a": 0,
        "n_b": 0,
        "n_pairs": 0,
        "mean_a": np.nan,
        "mean_b": np.nan,
        "delta_b_minus_a": np.nan,
        "p_value": np.nan,
        "effect_size_cliffs_delta": np.nan,
        "requested_pair_columns": json.dumps(list(pair_cols), separators=(",", ":")),
        "pair_key_columns": "[]",
        "missing_pair_columns": "[]",
        "n_unique_pair_keys_a": 0,
        "n_unique_pair_keys_b": 0,
        "n_union_pair_keys": 0,
        "pair_key_coverage": np.nan,
        "paired_observation_coverage": np.nan,
        "n_missing_pair_keys_in_a": 0,
        "n_missing_pair_keys_in_b": 0,
        "missing_pair_keys_in_a": "[]",
        "missing_pair_keys_in_b": "[]",
        "missing_episode_ids_in_a": "[]",
        "missing_episode_ids_in_b": "[]",
        "n_invalid_pair_rows_a": 0,
        "n_invalid_pair_rows_b": 0,
        "n_duplicate_pair_keys_a": 0,
        "n_duplicate_pair_keys_b": 0,
        "duplicate_pair_keys_a": "[]",
        "duplicate_pair_keys_b": "[]",
        "require_paired": bool(require_paired),
        "note": "",
    }
    if df.empty or metric not in df.columns or group_col not in df.columns:
        base["note"] = "missing_data"
        return base

    sub = df[df[group_col].isin([group_a, group_b])].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[metric])
    a = sub[sub[group_col] == group_a]
    b = sub[sub[group_col] == group_b]
    base["n_a"] = int(len(a))
    base["n_b"] = int(len(b))
    base["mean_a"] = float(a[metric].mean()) if len(a) else np.nan
    base["mean_b"] = float(b[metric].mean()) if len(b) else np.nan
    if np.isfinite(base["mean_a"]) and np.isfinite(base["mean_b"]):
        base["delta_b_minus_a"] = float(base["mean_b"] - base["mean_a"])
    base["effect_size_cliffs_delta"] = cliffs_delta(b[metric], a[metric])

    requested_pair_cols = list(dict.fromkeys(pair_cols))
    # A compared group cannot also be part of its own matching identity. This
    # retains the four-coordinate key for Judge/model comparisons, while a
    # perturbation-vs-perturbation test naturally matches on the other three.
    usable_pair_cols = [
        col for col in requested_pair_cols if col in sub.columns and col != group_col
    ]
    missing_pair_cols = [
        col for col in requested_pair_cols if col not in sub.columns and col != group_col
    ]
    base["pair_key_columns"] = json.dumps(usable_pair_cols, separators=(",", ":"))
    base["missing_pair_columns"] = json.dumps(missing_pair_cols, separators=(",", ":"))
    if usable_pair_cols:
        keyed = sub.copy()
        keyed["_pair_key"] = [
            tuple(
                _normalize_pair_key_value(value, column)
                for column, value in zip(usable_pair_cols, values)
            )
            for values in keyed[usable_pair_cols].itertuples(index=False, name=None)
        ]
        valid_key_mask = keyed["_pair_key"].map(
            lambda key: all(value is not None for value in key)
        )
        invalid_keyed = keyed[~valid_key_mask]
        keyed = keyed[valid_key_mask].copy()
        base["n_invalid_pair_rows_a"] = int(
            (invalid_keyed[group_col] == group_a).sum()
        )
        base["n_invalid_pair_rows_b"] = int(
            (invalid_keyed[group_col] == group_b).sum()
        )

        cell_counts = Counter(
            (str(group), key)
            for group, key in keyed[[group_col, "_pair_key"]].itertuples(
                index=False, name=None
            )
        )
        duplicate_keys_a = {
            key
            for (group, key), count in cell_counts.items()
            if group == str(group_a) and count > 1
        }
        duplicate_keys_b = {
            key
            for (group, key), count in cell_counts.items()
            if group == str(group_b) and count > 1
        }
        base["n_duplicate_pair_keys_a"] = len(duplicate_keys_a)
        base["n_duplicate_pair_keys_b"] = len(duplicate_keys_b)
        base["duplicate_pair_keys_a"] = _serialize_pair_keys(
            duplicate_keys_a, usable_pair_cols
        )
        base["duplicate_pair_keys_b"] = _serialize_pair_keys(
            duplicate_keys_b, usable_pair_cols
        )

        keys_a = {
            key
            for group, key in keyed[[group_col, "_pair_key"]].itertuples(
                index=False, name=None
            )
            if str(group) == str(group_a)
        }
        keys_b = {
            key
            for group, key in keyed[[group_col, "_pair_key"]].itertuples(
                index=False, name=None
            )
            if str(group) == str(group_b)
        }
        pairable_keys_a = keys_a - duplicate_keys_a
        pairable_keys_b = keys_b - duplicate_keys_b
        matched_keys = pairable_keys_a & pairable_keys_b
        union_keys = keys_a | keys_b
        missing_in_a = keys_b - keys_a
        missing_in_b = keys_a - keys_b
        base["n_unique_pair_keys_a"] = len(keys_a)
        base["n_unique_pair_keys_b"] = len(keys_b)
        base["n_union_pair_keys"] = len(union_keys)
        base["n_pairs"] = len(matched_keys)
        base["pair_key_coverage"] = (
            float(len(matched_keys) / len(union_keys)) if union_keys else np.nan
        )
        observation_denominator = len(a) + len(b)
        base["paired_observation_coverage"] = (
            float(2 * len(matched_keys) / observation_denominator)
            if observation_denominator
            else np.nan
        )
        base["n_missing_pair_keys_in_a"] = len(missing_in_a)
        base["n_missing_pair_keys_in_b"] = len(missing_in_b)
        base["missing_pair_keys_in_a"] = _serialize_pair_keys(
            missing_in_a, usable_pair_cols
        )
        base["missing_pair_keys_in_b"] = _serialize_pair_keys(
            missing_in_b, usable_pair_cols
        )
        base["missing_episode_ids_in_a"] = _serialize_missing_episode_ids(
            missing_in_a, usable_pair_cols
        )
        base["missing_episode_ids_in_b"] = _serialize_missing_episode_ids(
            missing_in_b, usable_pair_cols
        )
        pairing_integrity_reasons: List[str] = []
        if missing_pair_cols:
            pairing_integrity_reasons.append("pair_columns_missing")
        if base["n_invalid_pair_rows_a"] or base["n_invalid_pair_rows_b"]:
            pairing_integrity_reasons.append("pair_key_values_missing")
        if duplicate_keys_a or duplicate_keys_b:
            pairing_integrity_reasons.append("duplicate_pair_keys")
        if pairing_integrity_reasons:
            base["test_name"] = "invalid_pairing_cells"
            base["note"] = ";".join(pairing_integrity_reasons)
            return base

        pairable = keyed[
            keyed.apply(
                lambda row: row["_pair_key"] in matched_keys
                and cell_counts[(str(row[group_col]), row["_pair_key"])] == 1,
                axis=1,
            )
        ].copy()
        for index, column in enumerate(usable_pair_cols):
            pairable[column] = pairable["_pair_key"].map(
                lambda key, position=index: key[position]
            )
        pivot = pairable.pivot(
            index=usable_pair_cols,
            columns=group_col,
            values=metric,
        ).dropna(subset=[group_a, group_b], how="any")
        if len(pivot) >= 2:
            diffs = pivot[group_b] - pivot[group_a]
            base["test_name"] = "paired_wilcoxon"
            try:
                if np.allclose(diffs.to_numpy(dtype=float), 0.0):
                    base["p_value"] = 1.0
                    base["note"] = "all_paired_differences_zero"
                else:
                    base["p_value"] = float(stats.wilcoxon(pivot[group_b], pivot[group_a]).pvalue)
            except Exception as exc:
                base["note"] = f"wilcoxon_failed:{exc}"
            return base

        if require_paired:
            base["test_name"] = "insufficient_paired_samples"
            base["note"] = "need_at_least_two_unique_pairs"
            return base

    elif require_paired:
        base["test_name"] = "invalid_pairing_cells"
        base["note"] = "pair_columns_missing"
        return base

    if len(a) >= 2 and len(b) >= 2:
        base["test_name"] = "mann_whitney_u"
        try:
            base["p_value"] = float(stats.mannwhitneyu(b[metric], a[metric], alternative="two-sided").pvalue)
        except Exception as exc:
            base["note"] = f"mann_whitney_failed:{exc}"
    else:
        base["test_name"] = "insufficient_samples"
        base["note"] = "need_at_least_two_values_per_group"
    return base


def prepare_rebound(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    df = raw.copy()
    for col in [
        "rebound_baseline_loaded",
        "rebound_c_rec",
        "rebound_c_rec_index_100",
        "rebound_c_rec_cog",
        "rebound_c_rec_phy",
        "rebound_c_rec_state_debt",
        "rebound_num_recovery_windows",
        "rebound_raw_window_count",
        "rebound_formal_window_count",
        "rebound_skipped_window_count",
        "rebound_c_rec_valid",
        "rebound_w_rem_star",
        "rebound_g_rec_total",
        "rebound_tracker_window_count",
        "task_percent_complete",
        "task_state_success",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "rebound_c_rec_valid" in df.columns:
        df["c_rec_eligible"] = df["rebound_c_rec_valid"].fillna(0) >= 1.0
    else:
        df["c_rec_eligible"] = df.get("rebound_baseline_loaded", 0).fillna(0) >= 1.0
    df["c_rec_warmup"] = ~df["c_rec_eligible"]
    df["c_rec_observed"] = np.where(df["c_rec_eligible"], df.get("rebound_c_rec", np.nan), np.nan)
    raw_cost = (
        pd.to_numeric(df["rebound_c_rec"], errors="coerce")
        if "rebound_c_rec" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    artifact_index = (
        pd.to_numeric(df["rebound_c_rec_index_100"], errors="coerce")
        if "rebound_c_rec_index_100" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    derived_index = 100.0 - 100.0 / (1.0 + raw_cost)
    derived_index = derived_index.where(
        np.isfinite(raw_cost) & raw_cost.ge(0.0)
    ).clip(upper=np.nextafter(100.0, 0.0))

    artifact_present = np.isfinite(artifact_index)
    artifact_valid = artifact_present & artifact_index.ge(0.0) & artifact_index.lt(100.0)
    comparable = artifact_present & np.isfinite(derived_index)
    artifact_matches = pd.Series(pd.NA, index=df.index, dtype="boolean")
    artifact_matches.loc[comparable] = np.isclose(
        artifact_index.loc[comparable],
        derived_index.loc[comparable],
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    artifact_abs_error = (artifact_index - derived_index).abs().where(comparable)
    artifact_status = pd.Series("missing", index=df.index, dtype="object")
    artifact_status.loc[artifact_present & ~artifact_valid] = "invalid_range"
    artifact_status.loc[artifact_valid & ~np.isfinite(derived_index)] = "raw_unavailable"
    artifact_status.loc[comparable & artifact_matches.fillna(False)] = "matches_raw"
    artifact_status.loc[comparable & ~artifact_matches.fillna(False)] = "mismatch"

    # The bounded index is a display/description projection of the raw paper
    # cost. Artifact-provided values are retained only as audit evidence.
    df["rebound_c_rec_index_100_artifact"] = artifact_index
    df["rebound_c_rec_index_100_artifact_present"] = artifact_present
    df["rebound_c_rec_index_100_artifact_valid"] = artifact_valid
    df["rebound_c_rec_index_100_artifact_matches_raw"] = artifact_matches
    df["rebound_c_rec_index_100_artifact_abs_error"] = artifact_abs_error
    df["rebound_c_rec_index_100_artifact_status"] = artifact_status
    df["rebound_c_rec_index_100"] = derived_index
    df["c_rec_index_observed"] = np.where(
        df["c_rec_eligible"],
        derived_index,
        np.nan,
    )
    return df


def build_rebound_outputs(
    rebound_raw: pd.DataFrame,
    tables_dir: Path,
    rng: np.random.Generator,
    bootstrap_iters: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rebound = prepare_rebound(rebound_raw)
    formal = rebound[rebound["c_rec_eligible"]].copy() if not rebound.empty else pd.DataFrame()
    warmup = rebound[rebound["c_rec_warmup"]].copy() if not rebound.empty else pd.DataFrame()

    formal_summary = summarize_metric_group(
        formal,
        ["model", "perturbation_type"],
        REBOUND_METRICS,
        rng,
        bootstrap_iters,
    )
    warmup_summary = summarize_metric_group(
        warmup,
        ["model", "perturbation_type"],
        [
            "rebound_c_rec",
            "rebound_c_rec_valid",
            "rebound_raw_window_count",
            "rebound_tracker_window_count",
            "rebound_formal_window_count",
            "rebound_skipped_window_count",
            "rebound_num_recovery_windows",
            "task_percent_complete",
            "task_state_success",
        ],
        rng,
        bootstrap_iters,
    )
    if not warmup_summary.empty:
        warmup_summary["diagnostic_note"] = (
            "c_rec_valid=0; c_rec is unmeasurable and excluded from formal recovery cost"
        )

    outcome_context = summarize_metric_group(
        rebound,
        ["model", "perturbation_type"],
        [
            "task_percent_complete",
            "task_state_success",
            "rebound_tracker_window_count",
            "rebound_num_recovery_windows",
        ],
        rng,
        bootstrap_iters,
        extra_counts={"c_rec_eligible_count": "c_rec_eligible", "c_rec_warmup_count": "c_rec_warmup"},
    )

    index_diagnostic_columns = [
        "model",
        "episode_id",
        "perturbation_type",
        "perturbation_seed",
        "perturbation_intensity",
        "rebound_c_rec",
        "rebound_c_rec_index_100",
        "rebound_c_rec_index_100_artifact",
        "rebound_c_rec_index_100_artifact_present",
        "rebound_c_rec_index_100_artifact_valid",
        "rebound_c_rec_index_100_artifact_matches_raw",
        "rebound_c_rec_index_100_artifact_abs_error",
        "rebound_c_rec_index_100_artifact_status",
    ]
    index_diagnostics = rebound[
        [column for column in index_diagnostic_columns if column in rebound.columns]
    ]

    formal_summary.to_csv(tables_dir / "expq1_rebound_formal_cost.csv", index=False, encoding="utf-8-sig")
    warmup_summary.to_csv(tables_dir / "expq1_rebound_warmup_diagnostics.csv", index=False, encoding="utf-8-sig")
    outcome_context.to_csv(tables_dir / "expq1_task_outcome_context.csv", index=False, encoding="utf-8-sig")
    formal.to_csv(tables_dir / "expq1_rebound_formal_rows.csv", index=False, encoding="utf-8-sig")
    index_diagnostics.to_csv(
        tables_dir / "expq1_rebound_index_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return rebound, formal, formal_summary, warmup_summary


def build_stability_outputs(
    beta: pd.DataFrame,
    tables_dir: Path,
    rng: np.random.Generator,
    bootstrap_iters: int,
) -> pd.DataFrame:
    if beta.empty:
        summary = pd.DataFrame()
    else:
        summary = summarize_metric_group(
            beta,
            ["model"],
            STABILITY_METRICS + ["n_clean", "n_perturbed"],
            rng,
            bootstrap_iters,
        )
    summary.to_csv(tables_dir / "expq1_stability_beta.csv", index=False, encoding="utf-8-sig")
    beta.to_csv(tables_dir / "expq1_stability_beta_anchor_rows.csv", index=False, encoding="utf-8-sig")
    return summary


def build_ge_outputs(
    ge_rows: pd.DataFrame,
    diagnostics: Sequence[Mapping[str, Any]],
    tables_dir: Path,
    rng: np.random.Generator,
    bootstrap_iters: int,
) -> pd.DataFrame:
    if ge_rows.empty:
        summary = pd.DataFrame(diagnostics)
    else:
        summary = summarize_metric_group(
            ge_rows,
            ["model", "phase_label", "perturbation_type", "perturbation_seed", "stress_lambda"],
            ["task_percent_complete", "task_state_success", "ge_margin_soft", "ge_audc_f"],
            rng,
            bootstrap_iters,
        )
        diag_df = pd.DataFrame(diagnostics)
        keep = [
            "model",
            "formal_ge_estimable",
            "incomplete_reason",
            "episode_metrics_files",
            "partial_rows",
            "required_stress_lambdas",
            "missing_stress_lambdas",
            "stress_grid_complete",
            "stress_lambda_coverage",
        ]
        summary = summary.merge(diag_df[keep], on="model", how="left")
    summary.to_csv(tables_dir / "expq1_ge_partial_reconstruction.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(tables_dir / "expq1_ge_diagnostics.csv", index=False, encoding="utf-8-sig")
    return summary


def build_stat_tests(
    formal_rebound: pd.DataFrame,
    beta: pd.DataFrame,
    model_order: Sequence[str],
    tables_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    formal_rebound = _materialize_pairing_intensity(formal_rebound)
    if len(model_order) >= 2 and not formal_rebound.empty:
        model_a, model_b = model_order[0], model_order[1]
        for perturbation in sorted(formal_rebound["perturbation_type"].dropna().astype(str).unique()):
            sub = formal_rebound[formal_rebound["perturbation_type"].astype(str) == perturbation]
            for metric in [
                "c_rec_observed",
                "rebound_c_rec_cog",
                "rebound_c_rec_phy",
                "rebound_num_recovery_windows",
                "rebound_tracker_window_count",
            ]:
                rows.append(
                    paired_or_unpaired_test(
                        sub,
                        group_col="model",
                        group_a=model_a,
                        group_b=model_b,
                        metric=metric,
                        pair_cols=REBOUND_PAIR_COLUMNS,
                        context=f"rebound_model_comparison/{perturbation}",
                        require_paired=True,
                    )
                )
        for model in model_order:
            sub = formal_rebound[formal_rebound["model"] == model]
            perturbations = sorted(sub["perturbation_type"].dropna().astype(str).unique())
            if {"surface_rewrite", "object_state_toggle"}.issubset(set(perturbations)):
                for metric in ["c_rec_observed", "rebound_c_rec_cog", "rebound_c_rec_phy"]:
                    rows.append(
                        paired_or_unpaired_test(
                            sub,
                            group_col="perturbation_type",
                            group_a="surface_rewrite",
                            group_b="object_state_toggle",
                            metric=metric,
                            pair_cols=REBOUND_PAIR_COLUMNS,
                            context=f"rebound_perturbation_comparison/{model}",
                            require_paired=True,
                        )
                    )

    if len(model_order) >= 2 and not beta.empty:
        model_a, model_b = model_order[0], model_order[1]
        for metric in ["beta_hat", "beta_trajectory", "beta_out"]:
            rows.append(
                paired_or_unpaired_test(
                    beta,
                    group_col="model",
                    group_a=model_a,
                    group_b=model_b,
                    metric=metric,
                    pair_cols=["anchor_id"],
                    context="stability_model_comparison",
                )
            )

    tests = pd.DataFrame(rows)
    tests.to_csv(tables_dir / "expq1_stat_tests.csv", index=False, encoding="utf-8-sig")
    return tests


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )


def save_empty_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_rebound_cost(formal: pd.DataFrame, formal_summary: pd.DataFrame, images_dir: Path) -> None:
    path = images_dir / "rebound_cost_validity.png"
    if formal.empty:
        save_empty_plot(path, "Rebound Cost Validity", "No baseline-loaded C_rec rows were available.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    sns.boxplot(
        data=formal,
        x="perturbation_type",
        y="c_rec_index_observed",
        hue="model",
        ax=axes[0],
        palette=["#2A9D8F", "#E76F51"],
        showfliers=False,
    )
    sns.stripplot(
        data=formal,
        x="perturbation_type",
        y="c_rec_index_observed",
        hue="model",
        ax=axes[0],
        palette=["#1B6F66", "#A94732"],
        dodge=True,
        size=3,
        alpha=0.45,
        legend=False,
    )
    axes[0].set_title("Bounded recovery-cost index after baseline is loaded")
    axes[0].set_xlabel("Perturbation")
    axes[0].set_ylabel("C_rec,index100 (lower is better)")
    axes[0].set_ylim(0.0, 100.0)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:2], labels[:2], title="Model", loc="upper left")

    component_cols = ["rebound_c_rec_cog", "rebound_c_rec_phy", "rebound_c_rec_state_debt"]
    component = formal.groupby("model")[component_cols].mean().reset_index()
    bottom = np.zeros(len(component))
    x = np.arange(len(component))
    colors = ["#264653", "#F4A261", "#8AB17D"]
    labels = ["Cognitive", "Physical", "State debt"]
    for col, color, label in zip(component_cols, colors, labels):
        values = component[col].to_numpy(dtype=float)
        axes[1].bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[1].set_xticks(x, component["model"], rotation=20, ha="right")
    axes[1].set_title("Raw component magnitudes")
    axes[1].set_ylabel("Component value")
    axes[1].legend(loc="upper right")
    fig.suptitle("Rebound metric validity: success can hide recovery cost", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_stability(beta: pd.DataFrame, images_dir: Path) -> None:
    path = images_dir / "stability_beta_validity.png"
    if beta.empty:
        save_empty_plot(path, "Stability Beta Validity", "No anchor-level beta rows were available.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.0, 1.15]})
    sns.boxplot(
        data=beta,
        x="model",
        y="beta_hat",
        hue="model",
        ax=axes[0],
        palette=["#457B9D", "#E9C46A"],
        showfliers=False,
        legend=False,
    )
    sns.stripplot(data=beta, x="model", y="beta_hat", ax=axes[0], color="#1D3557", size=4, alpha=0.55)
    axes[0].set_title("Anchor-level beta_hat")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("beta_hat (action-output divergence)")
    axes[0].tick_params(axis="x", rotation=20)

    comp = beta.groupby("model")[["beta_trajectory", "beta_out"]].mean().reset_index()
    x = np.arange(len(comp))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        comp["beta_trajectory"],
        width,
        label="beta_trajectory",
        color="#2A9D8F",
    )
    axes[1].bar(x + width / 2, comp["beta_out"], width, label="beta_out", color="#E76F51")
    axes[1].set_xticks(x, comp["model"], rotation=20, ha="right")
    axes[1].set_title("Beta decomposition")
    axes[1].set_ylabel("Mean component value")
    axes[1].legend()
    fig.suptitle("Stability metric validity: beta separates action consistency from task success", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ge_status(ge_summary: pd.DataFrame, diagnostics: Sequence[Mapping[str, Any]], images_dir: Path) -> None:
    path = images_dir / "ge_partial_status.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    if ge_summary.empty or "phase_label" not in ge_summary.columns:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "No GE episode_metrics rows found", ha="center", va="center")
    else:
        plot_df = ge_summary.copy()
        sns.barplot(
            data=plot_df,
            x="phase_label",
            y="n_rows",
            hue="model",
            ax=axes[0],
            palette=["#2A9D8F", "#E76F51"],
        )
        axes[0].set_title("Recovered GE partial rows")
        axes[0].set_xlabel("Phase")
        axes[0].set_ylabel("Episode rows")
        axes[0].tick_params(axis="x", rotation=20)
    axes[1].axis("off")
    lines = ["Formal GE estimability check"]
    for diag in diagnostics:
        status = "complete" if diag.get("formal_ge_estimable") else "incomplete"
        lambdas = diag.get("stress_lambdas", [])
        phases = ", ".join(diag.get("phases", [])) or "none"
        lines.append(
            f"{diag.get('model')}: {status}\n"
            f"  phases={phases}\n"
            f"  lambdas={lambdas}\n"
            f"  reason={diag.get('incomplete_reason') or 'none'}"
        )
    axes[1].text(0.0, 1.0, "\n\n".join(lines), ha="left", va="top", family="monospace", fontsize=8)
    fig.suptitle("Graceful Extensibility: partial reconstruction, not formal AUDC evidence", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_logic_panel(
    formal: pd.DataFrame,
    beta: pd.DataFrame,
    ge_summary: pd.DataFrame,
    diagnostics: Sequence[Mapping[str, Any]],
    images_dir: Path,
) -> None:
    path = images_dir / "expq1_resilience_metrics_logic.png"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    if formal.empty:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "No formal C_rec", ha="center", va="center")
    else:
        overall = formal.groupby("model")["c_rec_observed"].mean().reset_index()
        sns.barplot(
            data=overall,
            x="model",
            y="c_rec_observed",
            hue="model",
            ax=axes[0],
            palette=["#2A9D8F", "#E76F51"],
            legend=False,
        )
        axes[0].set_title("Rebound: recovery cost")
        axes[0].set_ylabel("Mean formal C_rec")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=20)

    if beta.empty:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No beta rows", ha="center", va="center")
    else:
        beta_overall = beta.groupby("model")["beta_hat"].mean().reset_index()
        sns.barplot(
            data=beta_overall,
            x="model",
            y="beta_hat",
            hue="model",
            ax=axes[1],
            palette=["#457B9D", "#E9C46A"],
            legend=False,
        )
        axes[1].set_title("Stability: beta")
        axes[1].set_ylabel("Mean beta_hat")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=20)

    axes[2].axis("off")
    complete = sum(1 for diag in diagnostics if diag.get("formal_ge_estimable"))
    total = len(diagnostics)
    partial_rows = int(ge_summary["n_rows"].sum()) if not ge_summary.empty and "n_rows" in ge_summary.columns else 0
    text = (
        "Graceful Extensibility\n"
        f"formal complete runs: {complete}/{total}\n"
        f"partial rows recovered: {partial_rows}\n\n"
        "Current evidence: diagnostic only\n"
        "Missing: stress sweep + margin/AUDC aggregation"
    )
    axes[2].text(0.5, 0.5, text, ha="center", va="center", fontsize=10)
    axes[2].set_title("GE: stress boundary/capacity")

    fig.suptitle("Exp-Q1 metric-validity evidence across three resilience dimensions", y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 2) -> str:
    try:
        val = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored Markdown table.

    Avoids pandas.to_markdown so the analysis script does not depend on the
    optional `tabulate` package in lightweight experiment environments.
    """
    if df.empty:
        return ""
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(fmt(value, 4))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def first_value(df: pd.DataFrame, model: str, perturbation: str, col: str) -> Any:
    if df.empty or col not in df.columns:
        return np.nan
    sub = df[(df["model"] == model) & (df["perturbation_type"].astype(str) == perturbation)]
    if sub.empty:
        return np.nan
    return sub.iloc[0][col]


def write_report(
    output_dir: Path,
    specs: Sequence[RunSpec],
    formal_summary: pd.DataFrame,
    warmup_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    ge_summary: pd.DataFrame,
    diagnostics: Sequence[Mapping[str, Any]],
    tests: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Exp-Q1 Resilience Metrics Analysis")
    lines.append("")
    lines.append("## Inputs and Outputs")
    lines.append("")
    for spec in specs:
        lines.append(f"- `{spec.model}`: `{spec.root}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append("- Figure directory: `Analysis/images/`")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    if formal_summary.empty:
        lines.append("- Rebound: No formal C_rec rows with `baseline_loaded=1` are available. Only raw-window diagnostics are supported, so no recovery-cost conclusion can be drawn.")
    else:
        lines.append("- Rebound: Formal C_rec rows are available. Rows with `baseline_loaded=0` are excluded as warm-up so the initial `c_rec=0` without a StageBaseline is not treated as an observed recovery cost.")
        for model in formal_summary["model"].dropna().astype(str).unique():
            parts = []
            for perturbation in ["surface_rewrite", "object_state_toggle", "filler_injection"]:
                mean = first_value(formal_summary, model, perturbation, "c_rec_observed_mean")
                n = first_value(formal_summary, model, perturbation, "c_rec_observed_n")
                if pd.notna(mean):
                    parts.append(f"{perturbation}: mean C_rec={fmt(mean)} (n={int(n) if pd.notna(n) else 'NA'})")
            if parts:
                lines.append(f"  - {model}: " + "; ".join(parts))
    if not warmup_summary.empty:
        warmup_n = int(warmup_summary["n_rows"].sum())
        lines.append(f"- Rebound warm-up: {warmup_n} rows with `baseline_loaded=0` are marked unmeasurable. They remain available for raw tracker-window diagnostics but are excluded from the formal mean C_rec.")

    if stability_summary.empty:
        lines.append("- Stability: No anchor-level `statistics/stability_beta_neighborhood.csv` is available, so formal beta-stability statistics cannot be computed.")
    else:
        lines.append("- Stability: Beta statistics use anchor-level `statistics/stability_beta_neighborhood.csv` to avoid inflating the sample size with duplicate raw-rollout rows.")
        for _, row in stability_summary.iterrows():
            lines.append(
                f"  - {row['model']}: beta_hat mean={fmt(row.get('beta_hat_mean'))}, "
                f"beta_out mean={fmt(row.get('beta_out_mean'))}, anchors={int(row.get('n_rows', 0))}"
            )

    incomplete = [diag for diag in diagnostics if not diag.get("formal_ge_estimable")]
    if incomplete:
        lines.append("- Graceful Extensibility: The current GE block supports only partial reconstruction and cannot support a formal stress-boundary capacity conclusion.")
        for diag in incomplete:
            lines.append(
                f"  - {diag.get('model')}: rows={diag.get('partial_rows')}, "
                f"phases={diag.get('phases')}, reason={diag.get('incomplete_reason')}"
            )
    else:
        lines.append("- Graceful Extensibility: The GE stress sweep and margin/AUDC aggregation are complete and can support formal GE statistics.")
    lines.append("")

    lines.append("## Metric Interpretation")
    lines.append("")
    lines.append("- Rebound interpretation: `C_rec` summarizes recovery cost after task perturbation, including cognitive recovery cost. It complements success by quantifying extra cost behind successful or near-successful outcomes.")
    lines.append("- Stability interpretation: `beta_hat` summarizes action-output differences across semantically perturbed inputs, providing a view of planner sensitivity distinct from task success.")
    lines.append("- GE interpretation: Judge-calibrated boundaries and stress curves summarize remaining stress capacity. Incomplete stress sweeps identify execution gaps and do not support formal GE claims.")
    lines.append("")

    lines.append("## Statistical Tests")
    lines.append("")
    if tests.empty:
        lines.append("- Insufficient paired samples are available for statistical testing.")
    else:
        display = tests[["context", "metric", "group_a", "group_b", "test_name", "n_pairs", "n_a", "n_b", "p_value", "effect_size_cliffs_delta"]].head(12)
        lines.append(dataframe_to_markdown(display))
        lines.append("")
        lines.append("Full test results are in `Analysis/tables/expq1_stat_tests.csv`.")
    lines.append("")

    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `tables/expq1_rebound_formal_cost.csv`")
    lines.append("- `tables/expq1_rebound_warmup_diagnostics.csv`")
    lines.append("- `tables/expq1_rebound_index_diagnostics.csv`")
    lines.append("- `tables/expq1_stability_beta.csv`")
    lines.append("- `tables/expq1_ge_partial_reconstruction.csv`")
    lines.append("- `tables/expq1_stat_tests.csv`")
    lines.append("- `images/rebound_cost_validity.png`")
    lines.append("- `images/stability_beta_validity.png`")
    lines.append("- `images/ge_partial_status.png`")
    lines.append("- `images/expq1_resilience_metrics_logic.png`")
    lines.append("")

    (output_dir / "expq1_resilience_metrics_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_manifest(
    output_dir: Path,
    specs: Sequence[RunSpec],
    diagnostics: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, str],
) -> None:
    manifest = {
        "analysis_name": "expq1_resilience_metrics",
        "runs": [{"model": spec.model, "root": str(spec.root), "exists": spec.root.exists()} for spec in specs],
        "ge_diagnostics": list(diagnostics),
        "outputs": dict(outputs),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        help="Run mapping in Label=PATH format. Defaults to the two Data_Exper Exp-Q1 runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_data_root() / "Analysis",
        help="Analysis output directory.",
    )
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260421)
    args = parser.parse_args()

    specs = parse_run_specs(args.run)
    dirs = ensure_output_dirs(args.output_dir)
    rng = np.random.default_rng(args.seed)
    setup_plot_style()

    rebound_raw = load_all_raw(specs, REBOUND_BLOCK)
    beta = load_all_stability_beta(specs)
    ge_rows, ge_diagnostics = load_all_ge_partial(specs)

    rebound_all, rebound_formal, rebound_summary, warmup_summary = build_rebound_outputs(
        rebound_raw, dirs["tables"], rng, args.bootstrap_iters
    )
    stability_summary = build_stability_outputs(beta, dirs["tables"], rng, args.bootstrap_iters)
    ge_summary = build_ge_outputs(ge_rows, ge_diagnostics, dirs["tables"], rng, args.bootstrap_iters)
    tests = build_stat_tests(rebound_formal, beta, [spec.model for spec in specs], dirs["tables"])

    plot_rebound_cost(rebound_formal, rebound_summary, dirs["images"])
    plot_stability(beta, dirs["images"])
    plot_ge_status(ge_summary, ge_diagnostics, dirs["images"])
    plot_logic_panel(rebound_formal, beta, ge_summary, ge_diagnostics, dirs["images"])

    write_report(
        dirs["root"],
        specs,
        rebound_summary,
        warmup_summary,
        stability_summary,
        ge_summary,
        ge_diagnostics,
        tests,
    )
    write_manifest(
        dirs["root"],
        specs,
        ge_diagnostics,
        {
            "report": str(dirs["root"] / "expq1_resilience_metrics_report.md"),
            "tables": str(dirs["tables"]),
            "images": str(dirs["images"]),
        },
    )

    print(f"Wrote Exp-Q1 analysis to {dirs['root']}")


if __name__ == "__main__":
    main()
