#!/usr/bin/env python3
# isort: skip_file
"""
Shared orchestration helpers for the Exp-Q1 / Q2 / Q3 experiment runners.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gc
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from habitat_llm.examples.planner_demo_resilience import (
    DEFAULT_RESILIENCE_CFG,
    run_resilience_experiment,
)
from habitat_llm.examples.resilience_helpers import (
    extract_resilience_cfg,
    normalize_rollout_rows,
    run_margin_aggregation,
    run_stability_aggregation,
    write_csv_rows,
    write_json,
)

DEFAULT_SUMMARY_METRICS: List[str] = [
    "task_percent_complete",
    "task_state_success",
    "proposition_satisfied_fraction",
    "rebound_c_rec",
    "rebound_c_rec_cog",
    "rebound_c_rec_phy",
    "rebound_c_rec_state_debt",
    "rebound_c_rec_plan",
    "rebound_c_rec_sim",
    "rebound_num_recovery_windows",
    "rebound_raw_window_count",
    "rebound_formal_window_count",
    "rebound_skipped_window_count",
    "rebound_c_rec_valid",
    "rebound_w_rem_star",
    "rebound_g_rec_total",
    "stability_beta",
    "stability_beta_neighborhood",
    "stability_beta_vv",
    "stability_beta_out",
    "stability_beta_oscillation",
    "stability_beta_action",
    "stability_beta_progress",
    "stability_beta_success",
    "stability_beta_perturbation",
    "stability_beta_perturbation_vv",
    "stability_beta_perturbation_out",
    "stability_beta_perturbation_oscillation",
    "stability_beta_perturbation_action",
    "stability_beta_perturbation_progress",
    "stability_beta_perturbation_success",
    "stability_beta_perturbation_valid",
    "stability_beta_neighborhood_valid",
    "stability_p_cbf",
    "stability_td_error_mean_abs",
    "stability_td_error_p95_abs",
    "stability_gae_mean_abs",
    "stability_gae_p95_abs",
    "stability_return_target_variance",
    "stability_oscillation_loss",
    "stability_oscillation_loss_valid",
    "stability_value_delta_variance",
    "ge_contract_margin_min",
    "ge_margin_soft",
    "ge_margin_hard",
    "ge_cpsc_lambda_star",
    "ge_hard_lambda_star",
    "ge_audc_f",
    "ge_audc_norm_f",
    "ge_cell_valid",
    "ge_valid_lambda_coverage",
    "ge_degradation_rate",
    "ge_max_degradation_rate",
    "ge_margin_drop_f",
    "ge_monotonicity_violations",
    "ge_cliff_flag",
    "ge_boundary_hit",
    "ge_hard_boundary_hit",
]


def _materialize(node: Any) -> Any:
    if isinstance(node, DictConfig):
        return OmegaConf.to_container(node, resolve=True)
    return node


def _as_mapping(node: Any) -> Dict[str, Any]:
    materialized = _materialize(node)
    return dict(materialized) if isinstance(materialized, Mapping) else {}


def _as_list(node: Any) -> List[Any]:
    materialized = _materialize(node)
    if materialized is None:
        return []
    if isinstance(materialized, list):
        return list(materialized)
    return [materialized]


def clone_config(config: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def merge_config_patch(config: DictConfig, patch: Optional[Mapping[str, Any]]) -> DictConfig:
    merged = clone_config(config)
    if patch:
        merged = OmegaConf.merge(merged, OmegaConf.create(dict(patch)))
    return merged


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _maybe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_revision() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _ge_analysis_outputs(analysis_dir: Path) -> Dict[str, str]:
    candidates = {
        "margin_episode_contracts_csv": analysis_dir / "resilience_margin_episode_contracts.csv",
        "margin_episode_diagnostics_csv": analysis_dir / "resilience_margin_episode_diagnostics.csv",
        "margin_invalid_rows_csv": analysis_dir / "resilience_margin_invalid_rows.csv",
        "margin_cells_csv": analysis_dir / "resilience_margin_cells.csv",
        "margin_cell_diagnostics_csv": analysis_dir / "resilience_margin_cell_diagnostics.csv",
        "margin_capacity_csv": analysis_dir / "resilience_margin_capacity.csv",
        "margin_capacity_diagnostics_csv": analysis_dir / "resilience_margin_capacity_diagnostics.csv",
        "margin_audc_csv": analysis_dir / "resilience_margin_audc.csv",
    }
    return {key: str(path) for key, path in candidates.items() if path.exists()}


def summarize_rollouts(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    if not group_keys:
        grouped[tuple()] = list(rows)
    else:
        for row in rows:
            key = tuple(row.get(group_key, "") for group_key in group_keys)
            grouped[key].append(row)

    summaries: List[Dict[str, Any]] = []
    for key, bucket in grouped.items():
        summary: Dict[str, Any] = {}
        for group_key, value in zip(group_keys, key):
            summary[group_key] = value
        summary["n_rollouts"] = len(bucket)
        summary["n_episode_ids"] = len(
            {str(row.get("episode_id", row.get("anchor_id", ""))) for row in bucket}
        )
        success_values = [
            _maybe_float(row.get("task_state_success")) for row in bucket
        ]
        success_values = [value for value in success_values if value is not None]
        if success_values:
            summary["task_state_success_rate"] = _mean(success_values)

        for metric_key in metric_keys:
            values = [
                _maybe_float(row.get(metric_key))
                for row in bucket
            ]
            values = [value for value in values if value is not None]
            if not values:
                continue
            summary[f"{metric_key}_mean"] = _mean(values)
            summary[f"{metric_key}_median"] = _median(values)
            summary[f"{metric_key}_min"] = min(values)
            summary[f"{metric_key}_max"] = max(values)
        summaries.append(summary)
    return summaries


def _build_condition_list(block_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    conditions = _as_list(block_cfg.get("conditions"))
    if conditions:
        return [dict(condition) for condition in conditions]
    return [
        {
            "name": "default",
            "label": "default",
            "patch": {},
            "phases": _as_list(block_cfg.get("phases")),
        }
    ]


def _compose_runner_patch(
    suite_cfg: Mapping[str, Any],
    block_cfg: Mapping[str, Any],
    condition_cfg: Mapping[str, Any],
    phase_cfg: Mapping[str, Any],
    results_dir: Path,
) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    for candidate in (
        _as_mapping(suite_cfg.get("shared", {})).get("base_patch"),
        block_cfg.get("base_patch"),
        condition_cfg.get("patch"),
        phase_cfg.get("patch"),
    ):
        if candidate:
            patch = OmegaConf.to_container(
                OmegaConf.merge(OmegaConf.create(patch), OmegaConf.create(candidate)),
                resolve=True,
            )
    episode_ids = block_cfg.get("episode_ids")
    if episode_ids:
        patch["episode_ids"] = list(episode_ids)
    patch.setdefault("paths", {})
    patch["paths"]["results_dir"] = str(results_dir)
    return patch


def _run_condition(
    config: DictConfig,
    suite_key: str,
    suite_cfg: Mapping[str, Any],
    block_name: str,
    block_cfg: Mapping[str, Any],
    condition_cfg: Mapping[str, Any],
    condition_dir: Path,
) -> Dict[str, Any]:
    condition_name = str(condition_cfg.get("name") or "default")
    condition_label = str(condition_cfg.get("label") or condition_name)
    phases = _as_list(condition_cfg.get("phases")) or _as_list(block_cfg.get("phases"))
    if not phases:
        raise ValueError(f"{suite_key}.{block_name}.{condition_name} has no phases configured.")

    baseline_path: Optional[Path] = None
    phase_runs: List[Dict[str, Any]] = []
    combined_rows: List[Dict[str, Any]] = []

    for phase in phases:
        phase_cfg = dict(phase)
        phase_label = str(phase_cfg.get("label") or "phase")
        phase_dir = condition_dir / "runner" / phase_label
        merged_cfg = merge_config_patch(
            config,
            _compose_runner_patch(
                suite_cfg,
                block_cfg,
                condition_cfg,
                phase_cfg,
                phase_dir,
            ),
        )
        resilience_cfg = extract_resilience_cfg(merged_cfg, DEFAULT_RESILIENCE_CFG)
        result = run_resilience_experiment(
            merged_cfg,
            resilience_cfg=resilience_cfg,
            baseline_override_path=baseline_path,
            postprocess_modes=[],
        )
        if result.get("baseline_path") and Path(result["baseline_path"]).exists():
            baseline_path = Path(result["baseline_path"])

        phase_rows = [dict(row) for row in result.get("raw_rows", [])]
        # Extract lightweight metadata then release the large result dict
        phase_run_meta = {
            "phase_label": phase_label,
            "results_dir": str(phase_dir),
            "raw_output": str(result.get("raw_output")),
            "baseline_path": str(result.get("baseline_path")),
            "executed_specs": [dict(spec.__dict__) for spec in result.get("executed_specs", [])],
            "row_count": len(phase_rows),
        }
        del result

        for row in phase_rows:
            row["suite_name"] = suite_key
            row["subexperiment_name"] = block_name
            row["condition_name"] = condition_name
            row["condition_label"] = condition_label
            row["phase_label"] = phase_label
        combined_rows.extend(phase_rows)
        del phase_rows
        phase_runs.append(phase_run_meta)
        gc.collect()

    condition_post_cfg = merge_config_patch(
        config,
        _compose_runner_patch(
            suite_cfg,
            block_cfg,
            condition_cfg,
            {"patch": {}},
            condition_dir,
        ),
    )
    resilience_cfg = extract_resilience_cfg(condition_post_cfg, DEFAULT_RESILIENCE_CFG)
    dataset_path = OmegaConf.select(config, "habitat.dataset.data_path")
    combined_rows = normalize_rollout_rows(
        combined_rows,
        clean_tag=str(
            (resilience_cfg.get("stage_baseline", {}) or {}).get("clean_tag", "clean")
        ),
        dataset_path=str(dataset_path) if dataset_path else None,
    )
    postprocess = [str(mode) for mode in _as_list(block_cfg.get("postprocess"))]
    analysis_dir = condition_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if "stability" in postprocess:
        run_stability_aggregation(combined_rows, analysis_dir, resilience_cfg)
    if "margin" in postprocess:
        run_margin_aggregation(
            combined_rows,
            analysis_dir,
            resilience_cfg,
            dataset_path=str(dataset_path) if dataset_path else None,
        )
    condition_raw_path = write_csv_rows(
        analysis_dir / "raw_rollouts.csv",
        combined_rows,
    )
    ge_outputs = _ge_analysis_outputs(analysis_dir)

    # Release in-memory rows — data is now persisted to CSV on disk
    row_count = len(combined_rows)
    del combined_rows
    gc.collect()
    logging.info(
        "[expq] Condition '%s' complete: %d rows written to %s",
        condition_name, row_count, condition_raw_path,
    )

    return {
        "condition_name": condition_name,
        "condition_label": condition_label,
        "row_count": row_count,
        "phase_runs": phase_runs,
        "baseline_path": str(baseline_path) if baseline_path else None,
        "analysis_dir": str(analysis_dir),
        "condition_raw_rollouts_csv": str(condition_raw_path),
        "postprocess": postprocess,
        "ge_outputs": ge_outputs,
    }


def _write_ge_capacity_summary(
    subexperiment_dir: Path,
    condition_runs: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    capacity_rows: List[Dict[str, Any]] = []
    cell_rows: List[Dict[str, Any]] = []
    for condition_run in condition_runs:
        analysis_dir = Path(str(condition_run.get("analysis_dir", "")))
        condition_name = str(condition_run.get("condition_name", ""))
        condition_label = str(condition_run.get("condition_label", ""))
        for row in _read_csv_rows(analysis_dir / "resilience_margin_capacity.csv"):
            row["condition_name"] = condition_name
            row["condition_label"] = condition_label
            capacity_rows.append(row)
        for row in _read_csv_rows(analysis_dir / "resilience_margin_cells.csv"):
            row["condition_name"] = condition_name
            row["condition_label"] = condition_label
            cell_rows.append(row)

    outputs: Dict[str, str] = {}
    if capacity_rows:
        path = write_csv_rows(subexperiment_dir / "capacity_summary.csv", capacity_rows)
        outputs["capacity_summary_csv"] = str(path)
    if cell_rows:
        path = write_csv_rows(subexperiment_dir / "margin_cells_summary.csv", cell_rows)
        outputs["margin_cells_summary_csv"] = str(path)
    return outputs


def _finalize_subexperiment(
    config: DictConfig,
    suite_key: str,
    suite_cfg: Mapping[str, Any],
    block_name: str,
    block_cfg: Mapping[str, Any],
    subexperiment_dir: Path,
    condition_csv_paths: Sequence[Path],
    condition_runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Finalize a subexperiment by reading condition rows from disk CSV files."""
    dataset_path = OmegaConf.select(config, "habitat.dataset.data_path")
    # Read rows from persisted CSV files instead of holding them in memory
    all_rows: List[Dict[str, Any]] = []
    for csv_path in condition_csv_paths:
        all_rows.extend(_read_csv_rows(csv_path))
    raw_rollouts_path = write_csv_rows(
        subexperiment_dir / "raw_rollouts.csv",
        all_rows,
    )
    summary_rows = summarize_rollouts(
        all_rows,
        group_keys=[str(key) for key in _as_list(block_cfg.get("summary_group_keys"))],
        metric_keys=[
            str(key)
            for key in _as_list(block_cfg.get("summary_metrics"))
            or DEFAULT_SUMMARY_METRICS
        ],
    )
    row_count = len(all_rows)
    del all_rows  # Release rows after summarization
    gc.collect()

    summary_path = write_csv_rows(subexperiment_dir / "summary.csv", summary_rows)
    ge_summary_outputs = _write_ge_capacity_summary(
        subexperiment_dir,
        condition_runs,
    )

    metadata = {
        "suite_name": suite_key,
        "subexperiment_name": block_name,
        "argument": block_cfg.get("argument"),
        "headline_metric": block_cfg.get("headline_metric"),
        "support_metrics": _as_list(block_cfg.get("support_metrics")),
        "baseline_model": suite_cfg.get("baseline_model"),
        "optimized_model": suite_cfg.get("optimized_model"),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "episode_ids": _as_list(block_cfg.get("episode_ids")),
        "postprocess": _as_list(block_cfg.get("postprocess")),
        "summary_group_keys": _as_list(block_cfg.get("summary_group_keys")),
        "summary_metrics": _as_list(block_cfg.get("summary_metrics"))
        or list(DEFAULT_SUMMARY_METRICS),
        "row_count": row_count,
        "condition_runs": list(condition_runs),
        "config_name": HydraConfig.get().job.config_name if HydraConfig.initialized() else None,
        "git_revision": _git_revision(),
        "suite_config": dict(suite_cfg),
        "block_config": dict(block_cfg),
        "outputs": {
            "raw_rollouts_csv": str(raw_rollouts_path),
            "summary_csv": str(summary_path),
            **ge_summary_outputs,
        },
    }
    metadata_path = subexperiment_dir / "experiment_metadata.json"
    write_json(metadata_path, metadata)
    return {
        "subexperiment_name": block_name,
        "directory": str(subexperiment_dir),
        "raw_rollouts_csv": str(raw_rollouts_path),
        "summary_csv": str(summary_path),
        **ge_summary_outputs,
        "metadata_json": str(metadata_path),
        "row_count": row_count,
    }


def run_named_suite(config: DictConfig, suite_key: str) -> Dict[str, Any]:
    suite_cfg = _as_mapping(OmegaConf.select(config, suite_key))
    if not suite_cfg:
        raise ValueError(f"Missing suite config `{suite_key}`.")

    output_root = Path(str(config.paths.results_dir)) / str(
        suite_cfg.get("output_subdir", suite_key)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "suite_name": suite_key,
        "output_root": str(output_root),
        "subexperiments": [],
        "git_revision": _git_revision(),
    }
    for block_name, block_cfg_raw in _as_mapping(suite_cfg.get("subexperiments")).items():
        block_cfg = dict(block_cfg_raw)
        if not bool(block_cfg.get("enabled", True)):
            continue

        subexperiment_dir = output_root / str(block_name)
        subexperiment_dir.mkdir(parents=True, exist_ok=True)
        # Collect CSV paths on disk instead of accumulating rows in memory
        condition_csv_paths: List[Path] = []
        condition_runs: List[Dict[str, Any]] = []

        for condition_cfg in _build_condition_list(block_cfg):
            condition_dir = subexperiment_dir / "conditions" / str(
                condition_cfg.get("name", "default")
            )
            condition_result = _run_condition(
                config,
                suite_key,
                suite_cfg,
                str(block_name),
                block_cfg,
                condition_cfg,
                condition_dir,
            )
            # Row data lives on disk; only keep the CSV path and metadata
            condition_csv_paths.append(
                Path(condition_result["condition_raw_rollouts_csv"])
            )
            condition_runs.append(condition_result)
            gc.collect()

        manifest["subexperiments"].append(
            _finalize_subexperiment(
                config,
                suite_key,
                suite_cfg,
                str(block_name),
                block_cfg,
                subexperiment_dir,
                condition_csv_paths,
                condition_runs,
            )
        )
        del condition_csv_paths, condition_runs
        gc.collect()

    write_json(output_root / "suite_manifest.json", manifest)
    return manifest
