#!/usr/bin/env python3
# isort: skip_file
"""
Shared orchestration helpers for the Exp-Q1 / Q2 / Q3 experiment runners.
"""
from __future__ import annotations

from collections import defaultdict
import gc
import logging
import math
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
    run_boundary_aggregation,
    run_stability_aggregation,
    write_csv_rows,
    write_json,
)
from habitat_llm.evaluation.reporting import (
    POSTPROCESS_METRIC_KEYS,
    RUNTIME_METRIC_KEYS,
    read_csv_rows,
)

DEFAULT_SUMMARY_METRICS: List[str] = [
    "task_percent_complete",
    "task_state_success",
    "proposition_satisfied_fraction",
    "sim_step_count",
    "replanning_count_0",
    "replanning_count_1",
    *RUNTIME_METRIC_KEYS,
    *POSTPROCESS_METRIC_KEYS,
]

EMPTY_ROLLOUT_FIELDS = tuple(
    dict.fromkeys(
        (
            "episode_id",
            "anchor_id",
            "perturbation_type",
            "perturbation_seed",
            "perturbation_intensity",
            "perturbation_stress_family",
            "perturbation_level_index",
            "perturbation_variant",
            "perturbation_complete_unit_count",
            "perturbation_stress_realized",
            "perturbation_text_dose",
            "perturbation_alignment_valid",
            "perturbation_bank_path",
            "canonical_instruction",
            "policy_instruction",
            "run_seed",
            "suite_name",
            "subexperiment_name",
            "condition_name",
            "condition_label",
            "phase_label",
            "judge_model",
            *RUNTIME_METRIC_KEYS,
            *POSTPROCESS_METRIC_KEYS,
        )
    )
)

DEFAULT_JUDGE_PAIR_KEYS = (
    "episode_id",
    "perturbation_seed",
    "perturbation_type",
    "perturbation_intensity",
)


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
    return list(read_csv_rows(path))


def _ge_analysis_outputs(analysis_dir: Path) -> Dict[str, str]:
    candidates = {
        "boundary_episode_summary_csv": analysis_dir / "diagnostics" / "ge_boundary_episode_summary.csv",
        "boundary_episode_diagnostics_csv": analysis_dir / "diagnostics" / "ge_boundary_episode_diagnostics.csv",
        "boundary_invalid_rows_csv": analysis_dir / "diagnostics" / "ge_boundary_invalid_rows.csv",
        "boundary_cells_csv": analysis_dir / "statistics" / "ge_boundary_cells.csv",
        "boundary_cell_diagnostics_csv": analysis_dir / "diagnostics" / "ge_boundary_cell_diagnostics.csv",
        "boundary_capacity_csv": analysis_dir / "resilience_boundary_capacity.csv",
        "boundary_capacity_diagnostics_csv": analysis_dir / "diagnostics" / "ge_boundary_capacity_diagnostics.csv",
        "boundary_curve_statistics_csv": analysis_dir / "statistics" / "ge_boundary_curve_statistics.csv",
        "boundary_reference_statistics_csv": analysis_dir / "statistics" / "ge_boundary_reference_statistics.csv",
        "boundary_baseline_json": analysis_dir / "statistics" / "ge_boundary_baseline.json",
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


def build_configured_summary_comparison(
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Build aligned candidate-minus-reference deltas from summary rows."""
    group_key = str(comparison_cfg.get("group_key") or "").strip()
    reference = str(comparison_cfg.get("reference") or "").strip()
    candidate = str(comparison_cfg.get("candidate") or "").strip()
    match_keys = [str(key) for key in _as_list(comparison_cfg.get("match_keys"))]
    metrics = [str(key) for key in _as_list(comparison_cfg.get("metrics"))]
    statistic = str(comparison_cfg.get("statistic") or "mean").strip()
    if not group_key or not reference or not candidate or not metrics:
        return []

    def _match_key(row: Mapping[str, Any]) -> tuple:
        return tuple(str(row.get(key, "")) for key in match_keys)

    reference_rows = {
        _match_key(row): row
        for row in summary_rows
        if str(row.get(group_key, "")) == reference
    }
    candidate_rows = {
        _match_key(row): row
        for row in summary_rows
        if str(row.get(group_key, "")) == candidate
    }
    comparisons: List[Dict[str, Any]] = []
    for key in sorted(set(reference_rows) & set(candidate_rows)):
        reference_row = reference_rows[key]
        candidate_row = candidate_rows[key]
        record: Dict[str, Any] = {
            group_key + "_reference": reference,
            group_key + "_candidate": candidate,
        }
        for match_key, value in zip(match_keys, key):
            record[match_key] = value
        metric_count = 0
        for metric in metrics:
            summary_key = f"{metric}_{statistic}"
            reference_value = _maybe_float(reference_row.get(summary_key))
            candidate_value = _maybe_float(candidate_row.get(summary_key))
            if reference_value is None or candidate_value is None:
                continue
            delta = candidate_value - reference_value
            record[f"{metric}_reference"] = reference_value
            record[f"{metric}_candidate"] = candidate_value
            record[f"{metric}_delta_candidate_minus_reference"] = delta
            record[f"{metric}_relative_delta"] = (
                delta / abs(reference_value) if abs(reference_value) > 1.0e-12 else ""
            )
            metric_count += 1
        if metric_count:
            comparisons.append(record)
    return comparisons


def build_paired_rollout_comparison(
    rollout_rows: Sequence[Mapping[str, Any]],
    comparison_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Compare Judge conditions on aligned episode-level rollout cells.

    Unlike summary-level subtraction, this preserves the experimental pairing
    unit and emits reference-only/candidate-only cells so missing workers are
    visible in the comparison artifact rather than silently changing a family
    mean.
    """

    group_key = str(comparison_cfg.get("group_key") or "").strip()
    reference = str(comparison_cfg.get("reference") or "").strip()
    candidate = str(comparison_cfg.get("candidate") or "").strip()
    metrics = [str(key) for key in _as_list(comparison_cfg.get("metrics"))]
    pair_keys = [
        str(key)
        for key in (
            _as_list(comparison_cfg.get("pair_keys"))
            or DEFAULT_JUDGE_PAIR_KEYS
        )
    ]
    if not group_key or not reference or not candidate or not metrics:
        return []
    if tuple(pair_keys) != DEFAULT_JUDGE_PAIR_KEYS:
        raise ValueError(
            "Judge rollout comparison requires the exact pairing keys "
            f"{list(DEFAULT_JUDGE_PAIR_KEYS)}; received {pair_keys}"
        )

    def _pair_key(
        row: Mapping[str, Any],
    ) -> tuple[Optional[tuple], List[str]]:
        values: List[str] = []
        missing: List[str] = []
        for key in pair_keys:
            raw = row.get(key)
            text = "" if raw is None else str(raw).strip()
            if not text:
                missing.append(key)
                values.append("")
                continue
            if key == "perturbation_seed":
                try:
                    numeric = float(raw)
                    if not math.isfinite(numeric) or not numeric.is_integer():
                        raise ValueError
                    text = str(int(numeric))
                except (TypeError, ValueError):
                    missing.append(key)
                    text = ""
            elif key == "perturbation_intensity":
                try:
                    numeric = float(raw)
                    if not math.isfinite(numeric):
                        raise ValueError
                    text = format(numeric, ".17g")
                except (TypeError, ValueError):
                    missing.append(key)
                    text = ""
            values.append(text)
        return (tuple(values) if not missing else None), missing

    def _index(
        condition: str,
    ) -> tuple[
        Dict[tuple, List[Mapping[str, Any]]],
        List[tuple[Mapping[str, Any], List[str]]],
    ]:
        indexed: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
        invalid: List[tuple[Mapping[str, Any], List[str]]] = []
        for row in rollout_rows:
            if str(row.get(group_key, "")) == condition:
                key, missing = _pair_key(row)
                if key is None:
                    invalid.append((row, missing))
                else:
                    indexed[key].append(row)
        return indexed, invalid

    reference_rows, reference_invalid_rows = _index(reference)
    candidate_rows, candidate_invalid_rows = _index(candidate)
    reference_keys = set(reference_rows)
    candidate_keys = set(candidate_rows)
    union_keys = reference_keys | candidate_keys
    paired_keys = {
        key
        for key in reference_keys & candidate_keys
        if len(reference_rows[key]) == 1 and len(candidate_rows[key]) == 1
    }
    reference_duplicate_keys = {
        key for key, bucket in reference_rows.items() if len(bucket) != 1
    }
    candidate_duplicate_keys = {
        key for key, bucket in candidate_rows.items() if len(bucket) != 1
    }

    reference_count = len(reference_keys)
    candidate_count = len(candidate_keys)
    union_count = len(union_keys)
    paired_count = len(paired_keys)
    coverage_fields = {
        "reference_pair_count": reference_count,
        "candidate_pair_count": candidate_count,
        "paired_pair_count": paired_count,
        "reference_only_pair_count": len(reference_keys - candidate_keys),
        "candidate_only_pair_count": len(candidate_keys - reference_keys),
        "pair_union_count": union_count,
        "reference_invalid_pair_key_row_count": len(reference_invalid_rows),
        "candidate_invalid_pair_key_row_count": len(candidate_invalid_rows),
        "reference_duplicate_pair_key_count": len(reference_duplicate_keys),
        "candidate_duplicate_pair_key_count": len(candidate_duplicate_keys),
        "pair_coverage_reference": (
            paired_count / reference_count if reference_count else 0.0
        ),
        "pair_coverage_candidate": (
            paired_count / candidate_count if candidate_count else 0.0
        ),
        "pair_coverage_union": (
            paired_count / union_count if union_count else 0.0
        ),
    }

    comparisons: List[Dict[str, Any]] = []
    for key in sorted(union_keys):
        reference_bucket = reference_rows.get(key, [])
        candidate_bucket = candidate_rows.get(key, [])
        duplicate_reference = len(reference_bucket) > 1
        duplicate_candidate = len(candidate_bucket) > 1
        if duplicate_reference or duplicate_candidate:
            pairing_status = "duplicate_pair_key"
        elif reference_bucket and candidate_bucket:
            pairing_status = "paired"
        elif reference_bucket:
            pairing_status = "reference_only"
        else:
            pairing_status = "candidate_only"

        record: Dict[str, Any] = {
            group_key + "_reference": reference,
            group_key + "_candidate": candidate,
            "pairing_status": pairing_status,
            "reference_rows_in_pair": len(reference_bucket),
            "candidate_rows_in_pair": len(candidate_bucket),
            "pair_key_valid": True,
            "pair_key_unique": not (
                duplicate_reference or duplicate_candidate
            ),
            "missing_pair_keys": "",
            **coverage_fields,
        }
        for pair_key, value in zip(pair_keys, key):
            record[pair_key] = value

        reference_family = (
            str(reference_bucket[0].get("task_family", ""))
            if reference_bucket
            else ""
        )
        candidate_family = (
            str(candidate_bucket[0].get("task_family", ""))
            if candidate_bucket
            else ""
        )
        record["task_family_reference"] = reference_family
        record["task_family_candidate"] = candidate_family
        record["task_family_match"] = bool(
            reference_family
            and candidate_family
            and reference_family == candidate_family
        )

        paired_metric_count = 0
        for metric in metrics:
            reference_value = (
                _maybe_float(reference_bucket[0].get(metric))
                if len(reference_bucket) == 1
                else None
            )
            candidate_value = (
                _maybe_float(candidate_bucket[0].get(metric))
                if len(candidate_bucket) == 1
                else None
            )
            record[f"{metric}_reference"] = (
                "" if reference_value is None else reference_value
            )
            record[f"{metric}_candidate"] = (
                "" if candidate_value is None else candidate_value
            )
            if reference_value is None or candidate_value is None:
                record[f"{metric}_delta_candidate_minus_reference"] = ""
                record[f"{metric}_relative_delta"] = ""
                continue
            delta = candidate_value - reference_value
            record[f"{metric}_delta_candidate_minus_reference"] = delta
            record[f"{metric}_relative_delta"] = (
                delta / abs(reference_value)
                if abs(reference_value) > 1.0e-12
                else ""
            )
            paired_metric_count += 1
        record["paired_metric_count"] = paired_metric_count
        comparisons.append(record)

    for side, invalid_rows in (
        ("reference", reference_invalid_rows),
        ("candidate", candidate_invalid_rows),
    ):
        for row, missing in invalid_rows:
            record = {
                group_key + "_reference": reference,
                group_key + "_candidate": candidate,
                "pairing_status": f"{side}_invalid_pair_key",
                "reference_rows_in_pair": int(side == "reference"),
                "candidate_rows_in_pair": int(side == "candidate"),
                "pair_key_valid": False,
                "pair_key_unique": False,
                "missing_pair_keys": ",".join(missing),
                **coverage_fields,
            }
            for pair_key in pair_keys:
                raw = row.get(pair_key)
                record[pair_key] = "" if raw is None else str(raw).strip()
            family = str(row.get("task_family", ""))
            record["task_family_reference"] = (
                family if side == "reference" else ""
            )
            record["task_family_candidate"] = (
                family if side == "candidate" else ""
            )
            record["task_family_match"] = False
            for metric in metrics:
                value = _maybe_float(row.get(metric))
                record[f"{metric}_reference"] = (
                    value if side == "reference" and value is not None else ""
                )
                record[f"{metric}_candidate"] = (
                    value if side == "candidate" and value is not None else ""
                )
                record[f"{metric}_delta_candidate_minus_reference"] = ""
                record[f"{metric}_relative_delta"] = ""
            record["paired_metric_count"] = 0
            comparisons.append(record)
    return comparisons


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
        stage_baseline_cfg = resilience_cfg.get("stage_baseline", {}) or {}
        judge_model = str(
            stage_baseline_cfg.get("judge_model")
            or OmegaConf.select(merged_cfg, "evaluation.critic.llm_model")
            or OmegaConf.select(merged_cfg, "evaluation.adca.llm_model")
            or "unconfigured-judge"
        )
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
            row["judge_model"] = judge_model
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
    if "stability" in postprocess and combined_rows:
        run_stability_aggregation(
            combined_rows,
            analysis_dir,
            resilience_cfg,
            baseline_path=(
                baseline_path
                if baseline_path is not None and baseline_path.exists()
                else None
            ),
            dataset_path=str(dataset_path) if dataset_path else None,
        )
    if "margin" in postprocess and combined_rows:
        run_boundary_aggregation(
            combined_rows,
            analysis_dir,
            resilience_cfg,
            dataset_path=str(dataset_path) if dataset_path else None,
        )
    condition_raw_path = write_csv_rows(
        analysis_dir / "raw_rollouts.csv",
        combined_rows,
        empty_fieldnames=EMPTY_ROLLOUT_FIELDS,
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
        for row in _read_csv_rows(analysis_dir / "resilience_boundary_capacity.csv"):
            row["condition_name"] = condition_name
            row["condition_label"] = condition_label
            capacity_rows.append(row)
        for row in _read_csv_rows(analysis_dir / "statistics" / "ge_boundary_cells.csv"):
            row["condition_name"] = condition_name
            row["condition_label"] = condition_label
            cell_rows.append(row)

    outputs: Dict[str, str] = {}
    if capacity_rows:
        path = write_csv_rows(subexperiment_dir / "boundary_capacity_summary.csv", capacity_rows)
        outputs["boundary_capacity_summary_csv"] = str(path)
    if cell_rows:
        path = write_csv_rows(subexperiment_dir / "boundary_cells_summary.csv", cell_rows)
        outputs["boundary_cells_summary_csv"] = str(path)
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
        empty_fieldnames=EMPTY_ROLLOUT_FIELDS,
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

    summary_group_keys = [
        str(key) for key in _as_list(block_cfg.get("summary_group_keys"))
    ]
    summary_path = write_csv_rows(
        subexperiment_dir / "summary.csv",
        summary_rows,
        empty_fieldnames=(*summary_group_keys, "n_rollouts", "n_episode_ids"),
    )
    comparison_cfg = _as_mapping(block_cfg.get("comparison"))
    if str(comparison_cfg.get("group_key") or "") == "judge_model":
        comparison_rows = build_paired_rollout_comparison(
            all_rows,
            comparison_cfg,
        )
    else:
        comparison_rows = build_configured_summary_comparison(
            summary_rows,
            comparison_cfg,
        )
    del all_rows  # Release rows after paired comparison and summarization.
    gc.collect()
    comparison_path: Optional[Path] = None
    if comparison_rows:
        comparison_path = write_csv_rows(
            subexperiment_dir
            / str(comparison_cfg.get("output_filename") or "group_comparison.csv"),
            comparison_rows,
        )
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
            **(
                {"configured_comparison_csv": str(comparison_path)}
                if comparison_path is not None
                else {}
            ),
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
        **(
            {"configured_comparison_csv": str(comparison_path)}
            if comparison_path is not None
            else {}
        ),
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
