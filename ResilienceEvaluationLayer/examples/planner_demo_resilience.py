#!/usr/bin/env python3
# isort: skip_file
r"""
planner_demo_resilience.py - E1 / E2 runner for the Resilience evaluation suite.

This script is a thin wrapper around :func:`planner_demo_mp_new.run_planner`
that adds two orthogonal axes on top of an existing Hydra config:

* **Perturbation spec axis** (``resilience.perturbations``): rolls out each
  episode once per spec in the grid, registering a
  :class:`~habitat_llm.evaluation.perturbation.PerturbationInjector` on the
  evaluation runner before each run.
* **Seed axis** (``resilience.seeds``): per-spec repetition count required
  for the neighborhood stability aggregation.

The module also exposes :func:`run_resilience_experiment`, which is reused by
the higher-level ``planner_demo_expq*.py`` experiment orchestrators.
"""
from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from habitat_llm.evaluation.perturbation import PerturbationSpec
from habitat_llm.examples import planner_demo_mp_new as base_runner
from habitat_llm.examples.resilience_helpers import (
    build_spec_grid,
    extract_resilience_cfg,
    generate_stage_baseline,
    merge_spec_csvs,
    partition_specs,
    resolve_baseline_output_path,
    resolve_stage_baseline_startup_candidate,
    rollout_one_spec,
    run_margin_aggregation,
    run_stability_aggregation,
    teardown_between_specs,
    update_episode_aggregate_from_clean_spec,
)

logger = logging.getLogger(__name__)


DEFAULT_RESILIENCE_CFG: Dict[str, Any] = {
    "mode": "stability",  # one of {"rebound", "stability", "margin"}
    "perturbations": ["clean"],
    "seeds": [0],
    "intensities": [1.0],
    "output_subdir": "resilience",
    "baseline_json": "stage_baseline.json",
    "alpha": 0.5,
    "beta_weight_vv": 0.5,
    "beta_weight_out": 0.5,
    "stage_baseline": {
        "enabled": True,
        "baseline_only": False,
        "clean_tag": "clean",
        "write_persistent_copy": True,
        "persistent_store_dir": "evaluation/stage_baselines",
        "latest_alias": "stage_baseline_latest.json",
        "episode_aggregate": {
            "enabled": True,
            "subdir": "episode_aggregates",
            "merge_existing": True,
            "fallback_when_hash_missing": True,
            "allow_partial_compose": True,
        },
    },
}


def run_resilience_experiment(
    config: DictConfig,
    *,
    resilience_cfg: Optional[Dict[str, Any]] = None,
    baseline_override_path: Optional[Path] = None,
    postprocess_modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute one resilience batch and optionally post-process merged raw rows."""
    resilience_cfg = resilience_cfg or extract_resilience_cfg(
        config, DEFAULT_RESILIENCE_CFG
    )
    mode = str(resilience_cfg.get("mode", "stability"))
    specs = build_spec_grid(resilience_cfg)
    if not specs:
        raise ValueError(
            "No perturbation specs configured; set resilience.perturbations/seeds."
        )

    base_results_dir = Path(str(config.paths.results_dir))
    resilience_dir = base_results_dir / str(
        resilience_cfg.get("output_subdir", "resilience")
    )
    resilience_dir.mkdir(parents=True, exist_ok=True)
    run_local_baseline_path = resolve_baseline_output_path(
        base_results_dir,
        resilience_cfg,
    )
    baseline_path = (
        Path(baseline_override_path)
        if baseline_override_path is not None
        else run_local_baseline_path
    )

    experiment_seed = int(resilience_cfg.get("experiment_seed", 47668090))
    stage_baseline_cfg = resilience_cfg.get("stage_baseline", {}) or {}
    stage_baseline_enabled = bool(stage_baseline_cfg.get("enabled", True))
    baseline_only = bool(stage_baseline_cfg.get("baseline_only", False))
    clean_tag = str(stage_baseline_cfg.get("clean_tag", "clean"))
    clean_specs, other_specs = partition_specs(specs, clean_tag)
    aggregate_fallback_path: Optional[Path] = None
    baseline_resolution = resolve_stage_baseline_startup_candidate(
        config=config,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs,
        run_local_path=run_local_baseline_path,
        baseline_override_path=Path(baseline_override_path)
        if baseline_override_path is not None
        else None,
    )
    selected_initial_path = baseline_resolution.get("selected_path")
    if selected_initial_path and bool(baseline_resolution.get("selected_path_exists")):
        baseline_path = Path(str(selected_initial_path))
    if baseline_resolution.get("source") == "episode_aggregate_composed":
        aggregate_fallback_path = baseline_path
        logger.info(
            "Using episode-aggregate stage baseline because hash baseline is missing: "
            "run_local=%s exists=%s persistent=%s exists=%s composed=%s",
            baseline_resolution.get("run_local_hash_path"),
            baseline_resolution.get("run_local_hash_exists"),
            baseline_resolution.get("persistent_hash_path"),
            baseline_resolution.get("persistent_hash_exists"),
            aggregate_fallback_path,
        )
    else:
        logger.info(
            "Stage baseline startup resolution: source=%s selected=%s "
            "run_local_exists=%s persistent_exists=%s",
            baseline_resolution.get("source"),
            baseline_resolution.get("selected_path"),
            baseline_resolution.get("run_local_hash_exists"),
            baseline_resolution.get("persistent_hash_exists"),
        )

    if baseline_only and not clean_specs:
        raise ValueError(
            "resilience.stage_baseline.baseline_only=true requires at least "
            "one clean perturbation spec."
        )

    executed_specs: List[PerturbationSpec] = []
    per_spec_csvs: List[Path] = []
    episode_aggregate_updates: List[Dict[str, Any]] = []
    refresh_resolutions: List[Dict[str, Any]] = []
    if stage_baseline_enabled and clean_specs:
        for clean_spec in clean_specs:
            clean_executed, clean_csvs = _run_spec_batch(
                config,
                [clean_spec],
                resilience_dir,
                baseline_path,
                experiment_seed,
            )
            executed_specs.extend(clean_executed)
            per_spec_csvs.extend(clean_csvs)
            if clean_executed:
                aggregate_update = update_episode_aggregate_from_clean_spec(
                    resilience_output_dir=resilience_dir,
                    resilience_cfg=resilience_cfg,
                    clean_spec=clean_spec,
                    config=config,
                    clean_specs_for_signature=clean_specs,
                )
                if aggregate_update:
                    episode_aggregate_updates.append(aggregate_update)
                if baseline_override_path is None and not run_local_baseline_path.exists():
                    refreshed_resolution = resolve_stage_baseline_startup_candidate(
                        config=config,
                        resilience_cfg=resilience_cfg,
                        clean_specs=clean_specs,
                        run_local_path=run_local_baseline_path,
                    )
                    refresh_resolutions.append(refreshed_resolution)
                    selected_path = refreshed_resolution.get("selected_path")
                    if selected_path and bool(
                        refreshed_resolution.get("selected_path_exists")
                    ):
                        baseline_path = Path(str(selected_path))
                        if (
                            refreshed_resolution.get("source")
                            == "episode_aggregate_composed"
                        ):
                            aggregate_fallback_path = baseline_path
                            logger.info(
                                "Refreshed episode-aggregate stage baseline after %s: %s",
                                clean_spec.tag(),
                                aggregate_fallback_path,
                            )
    else:
        executed_specs, per_spec_csvs = _run_spec_batch(
            config,
            clean_specs,
            resilience_dir,
            baseline_path,
            experiment_seed,
        )

    generated_baseline_path: Optional[Path] = None
    if stage_baseline_enabled and clean_specs:
        generated_baseline_path = generate_stage_baseline(
            base_results_dir=base_results_dir,
            resilience_output_dir=resilience_dir,
            resilience_cfg=resilience_cfg,
            clean_specs=clean_specs,
            config=config,
            update_episode_aggregate=not bool(episode_aggregate_updates),
        )
    elif baseline_path.exists():
        generated_baseline_path = baseline_path
    if generated_baseline_path is not None:
        baseline_path = generated_baseline_path

    if baseline_only:
        logger.info(
            "Baseline-only mode enabled; skipping non-clean perturbation specs."
        )
    else:
        if other_specs and stage_baseline_enabled and not baseline_path.exists():
            logger.warning(
                "Running perturbed specs without a generated stage baseline "
                "at %s; formal C_rec fields will remain zero.",
                baseline_path,
            )
        extra_executed, extra_csvs = _run_spec_batch(
            config, other_specs, resilience_dir, baseline_path, experiment_seed
        )
        executed_specs.extend(extra_executed)
        per_spec_csvs.extend(extra_csvs)

    raw_output = resilience_dir / (
        "resilience_raw.csv" if mode == "stability" else f"resilience_{mode}.csv"
    )
    raw_rows = merge_spec_csvs(executed_specs, per_spec_csvs, raw_output)
    logger.info("Wrote %d merged rows -> %s", len(raw_rows), raw_output)

    requested_postprocess = (
        list(postprocess_modes)
        if postprocess_modes is not None
        else [mode] if mode in {"stability", "margin"} else []
    )
    dataset_path = OmegaConf.select(config, "habitat.dataset.data_path")

    stability_aggregates: Dict[str, Any] = {}
    margin_aggregates: Dict[str, Any] = {}
    if "stability" in requested_postprocess:
        stability_aggregates = run_stability_aggregation(
            raw_rows,
            resilience_dir,
            resilience_cfg,
            raw_output_path=raw_output,
        )
        logger.info("Aggregated β̂ for %d anchors.", len(stability_aggregates))
    if "margin" in requested_postprocess:
        margin_aggregates = run_margin_aggregation(
            raw_rows,
            resilience_dir,
            resilience_cfg,
            dataset_path=str(dataset_path) if dataset_path else None,
            raw_output_path=raw_output,
        )
        logger.info(
            "Aggregated contract margin for %d (family, lambda, e) cells.",
            len(margin_aggregates),
        )

    with open(
        resilience_dir / "resilience_config_resolved.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "resilience": resilience_cfg,
                "specs": [asdict(s) for s in specs],
                "executed_specs": [asdict(s) for s in executed_specs],
                "stage_baseline_path": str(generated_baseline_path or baseline_path),
                "stage_baseline_exists": bool(
                    (generated_baseline_path or baseline_path).exists()
                ),
                "postprocess_modes": requested_postprocess,
                "episode_aggregate_fallback_path": (
                    str(aggregate_fallback_path)
                    if aggregate_fallback_path is not None
                    else None
                ),
                "stage_baseline_startup_resolution": baseline_resolution,
                "episode_aggregate_incremental_updates": episode_aggregate_updates,
                "episode_aggregate_refresh_resolutions": refresh_resolutions,
            },
            handle,
            indent=2,
        )

    return {
        "mode": mode,
        "resilience_cfg": resilience_cfg,
        "specs": specs,
        "executed_specs": executed_specs,
        "resilience_dir": resilience_dir,
        "raw_output": raw_output,
        "raw_rows": raw_rows,
        "baseline_path": baseline_path,
        "generated_baseline_path": generated_baseline_path,
        "episode_aggregate_fallback_path": aggregate_fallback_path,
        "stage_baseline_startup_resolution": baseline_resolution,
        "episode_aggregate_incremental_updates": episode_aggregate_updates,
        "episode_aggregate_refresh_resolutions": refresh_resolutions,
        "stage_baseline_exists": bool(
            (generated_baseline_path or baseline_path).exists()
        ),
        "stability_aggregates": stability_aggregates,
        "margin_aggregates": margin_aggregates,
        "postprocess_modes": requested_postprocess,
    }


@hydra.main(
    config_path="../conf",
    config_name="baselines/qwen3_centralized_zero_shot_react_summary_vllm_state",
    version_base=None,
)
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_resilience_experiment(config)


def _run_spec_batch(
    config: DictConfig,
    specs: List[PerturbationSpec],
    resilience_dir: Path,
    baseline_path: Path,
    experiment_seed: int,
) -> tuple[List[PerturbationSpec], List[Path]]:
    """Execute a list of specs sequentially, returning executed specs and CSV paths."""
    executed: List[PerturbationSpec] = []
    csvs: List[Path] = []
    for spec in specs:
        try:
            csv_path = rollout_one_spec(
                config,
                spec,
                resilience_dir,
                base_runner=base_runner,
                baseline_path=baseline_path,
                seed=experiment_seed,
            )
            executed.append(spec)
            csvs.append(csv_path)
        except Exception as exc:
            logger.error("Spec %s failed: %s", spec.tag(), exc, exc_info=True)
        finally:
            teardown_between_specs()
    return executed, csvs


if __name__ == "__main__":
    main()
