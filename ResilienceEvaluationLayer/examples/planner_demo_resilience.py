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
import hashlib
import logging
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

from habitat_llm.evaluation.perturbation import (
    PerturbationSpec,
    derive_run_seed,
    spec_artifact_key,
)
from habitat_llm.evaluation.critic_export_contract import (
    load_full_stability_export,
)
from habitat_llm.evaluation.reporting import (
    POSTPROCESS_METRIC_KEYS,
    RUNTIME_METRIC_KEYS,
    primary_display_items,
    read_csv_rows,
)
from habitat_llm.evaluation.core.critic_lifecycle import (
    checkpoint_component_id,
    ensure_bootstrap_critic_components,
    preflight_reward_shaper,
    train_value_checkpoint_from_exports,
)
from habitat_llm.evaluation.stage_baseline import (
    STAGE_BASELINE_ALIGNMENT_SCOPE,
    baseline_supported_episode_ids,
)
from habitat_llm.examples import planner_demo_mp_new as base_runner
from habitat_llm.examples.resilience_helpers import (
    build_spec_grid,
    build_stage_baseline_signature,
    collect_critic_exports,
    _index_critic_exports,
    extract_resilience_cfg,
    generate_stage_baseline,
    merge_spec_csvs,
    partition_specs,
    plan_stage_baseline_calibration,
    resolve_baseline_output_path,
    resolve_stage_baseline_seed_lifecycle,
    resolve_persistent_baseline_dir,
    resolve_stage_baseline_startup_candidate,
    rollout_one_spec,
    spec_output_dir,
    run_boundary_aggregation,
    run_stability_aggregation,
    teardown_between_specs,
    update_episode_aggregate_from_clean_spec,
    write_csv_rows,
)
from habitat_llm.examples.resilience_scheduler import (
    BatchExecutionError,
    BatchRun,
    ExecutionSettings,
    chunk_episode_ids,
    run_bounded_jobs,
)

logger = logging.getLogger(__name__)


DEFAULT_RESILIENCE_CFG: Dict[str, Any] = {
    "mode": "stability",  # one of {"rebound", "stability", "margin", "joint"}
    "perturbations": ["clean"],
    "seeds": [0, 1, 2],
    "intensities": [1.0],
    "output_subdir": "resilience",
    "baseline_json": "stage_baseline.json",
    "alpha": 0.5,
    "execution": {
        "max_parallel_workers": 1,
        "episodes_per_worker": 1,
        "min_available_memory_gb": 48.0,
    },
    "stage_baseline": {
        "enabled": True,
        "baseline_only": False,
        "clean_tag": "clean",
        "evidence_store_enabled": True,
        "reuse_policy": "reuse_and_extend",
        "extension_samples_per_episode": 0,
        "write_persistent_copy": True,
        "persistent_store_dir": "evaluation/stage_baselines",
        "runtime_subdir": "runtime",
        "statistics_subdir": "statistics",
        "archive_subdir": "archives",
        "latest_alias": "stage_baseline_latest.json",
        "support": {
            "min_independent_trajectories": 1,
            "min_transitions": 1,
            "require_successful_clean": False,
            "allow_family_covariance_shrinkage": True,
        },
        "seed_lifecycle": {
            "enabled": True,
            "initial_calibration_seeds": [0, 1, 2],
            "evaluation_seeds": [0, 1, 2],
            "allow_paired_seeds": True,
            "extension_seed_start": 3,
            "critic_training_seeds": [100, 101, 102],
        },
        "episode_aggregate": {
            "enabled": True,
            "subdir": "by_episode",
            "merge_existing": True,
            "fallback_when_hash_missing": True,
            "allow_partial_compose": False,
        },
    },
}


def _critic_config_mapping(config: DictConfig) -> Dict[str, Any]:
    node = OmegaConf.select(config, "evaluation.critic")
    if node is None:
        return {}
    resolved = OmegaConf.to_container(node, resolve=True)
    return dict(resolved) if isinstance(resolved, Mapping) else {}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_critic_fields(config: DictConfig, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        OmegaConf.update(
            config,
            f"evaluation.critic.{key}",
            value,
            merge=False,
            force_add=True,
        )


def _critic_phase_config(
    config: DictConfig,
    phase: str,
    lifecycle: Mapping[str, Any],
) -> DictConfig:
    phase_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    _set_critic_fields(
        phase_config,
        {
            "phase": str(phase),
            "state_encoder_checkpoint": lifecycle.get(
                "state_encoder_checkpoint", ""
            ),
            "state_encoder_id": lifecycle.get("state_encoder_id", ""),
            "value_checkpoint": lifecycle.get("value_checkpoint", ""),
            "value_checkpoint_id": lifecycle.get("value_checkpoint_id", ""),
            "freeze_state_encoder": True,
            "update_value_network": False,
            "save_checkpoints": False,
            "require_component_checkpoints": bool(
                lifecycle.get("state_encoder_checkpoint")
                and lifecycle.get("value_checkpoint")
            ),
            "reward_shaper_valid": bool(
                lifecycle.get("reward_shaper_valid", False)
            ),
            "critic_lifecycle_valid": bool(lifecycle.get("valid", False)),
        },
    )
    return phase_config


def _prepare_critic_lifecycle(
    config: DictConfig,
    *,
    resilience_cfg: Mapping[str, Any],
    resilience_dir: Path,
    experiment_seed: int,
    execution_settings: ExecutionSettings,
) -> Dict[str, Any]:
    """Preflight the Judge and materialize shared/frozen critic components."""

    critic_cfg = _critic_config_mapping(config)
    if not critic_cfg:
        lifecycle = {
            "valid": False,
            "reason": "critic_config_missing",
            "reward_shaper_valid": False,
        }
        _set_critic_fields(
            config,
            {
                "critic_lifecycle_valid": False,
                "reward_shaper_valid": False,
            },
        )
        return lifecycle

    judge_model = str(
        critic_cfg.get("llm_model")
        or OmegaConf.select(config, "evaluation.planner.plan_config.llm.model")
        or "unconfigured-judge"
    )
    preflight = preflight_reward_shaper(critic_cfg)
    stage_cfg = dict(resilience_cfg.get("stage_baseline", {}) or {})
    model_root = resolve_persistent_baseline_dir(stage_cfg) / "_critic_components"
    bootstrap = ensure_bootstrap_critic_components(
        critic_cfg,
        model_root=model_root,
        judge_model=judge_model,
    )
    state_path = str(
        critic_cfg.get("state_encoder_checkpoint")
        or bootstrap["state_encoder_checkpoint"]
    )
    state_id = checkpoint_component_id(state_path, "state_encoder")

    explicit_value_path = str(critic_cfg.get("value_checkpoint") or "")
    explicit_value_id = checkpoint_component_id(
        explicit_value_path, "value_network"
    )
    training_manifest: Dict[str, Any] = {}
    value_path = explicit_value_path
    value_id = explicit_value_id
    value_trained = bool(explicit_value_id)

    lifecycle_cfg = dict(critic_cfg.get("lifecycle", {}) or {})
    seed_cfg = dict(stage_cfg.get("seed_lifecycle", {}) or {})
    training_seeds = [
        int(value)
        for value in lifecycle_cfg.get(
            "training_seeds",
            seed_cfg.get("critic_training_seeds", [100, 101, 102]),
        )
    ]
    training_identity = {
        "judge_model": judge_model,
        "judge_base_url": str(critic_cfg.get("llm_base_url") or ""),
        "state_encoder_id": state_id,
        "dataset_path": str(OmegaConf.select(config, "habitat.dataset.data_path") or ""),
        "training_seeds": training_seeds,
        "value_hidden_dims": critic_cfg.get("value_hidden_dims", [256, 128, 64]),
        "gamma": critic_cfg.get("gamma", 0.99),
        "gae_lambda": critic_cfg.get("gae_lambda", 0.95),
        "shaping_weight": critic_cfg.get("shaping_weight", 0.0),
        "reward_shaper_json_mode": critic_cfg.get("reward_shaper_json_mode"),
        "call_on_planning_steps": critic_cfg.get("call_on_planning_steps"),
        "llm_call_frequency": critic_cfg.get("llm_call_frequency"),
        "planning_step_window": critic_cfg.get("planning_step_window", {}),
        "evaluate_on_rebound": critic_cfg.get("evaluate_on_rebound"),
    }
    training_key = hashlib.sha256(
        json.dumps(training_identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
    trained_path = (
        model_root
        / "value_network"
        / re.sub(r"[^A-Za-z0-9_.-]+", "-", judge_model)
        / training_key
        / "value_network.pt"
    )

    if not value_path and trained_path.exists():
        value_path = str(trained_path)
        value_id = checkpoint_component_id(value_path, "value_network")
        value_trained = bool(value_id)
        training_manifest = {
            "reused": True,
            "value_checkpoint": value_path,
            "value_checkpoint_id": value_id,
        }
    elif not value_path and bool(preflight.get("valid", False)):
        training_cfg = dict(resilience_cfg)
        training_cfg["perturbations"] = [str(stage_cfg.get("clean_tag", "clean"))]
        training_cfg["seeds"] = training_seeds
        training_specs = build_spec_grid(training_cfg)
        training_config = OmegaConf.create(
            OmegaConf.to_container(config, resolve=False)
        )
        _set_critic_fields(
            training_config,
            {
                "phase": "training_collect",
                "state_encoder_checkpoint": state_path,
                "state_encoder_id": state_id,
                "value_checkpoint": bootstrap["bootstrap_value_checkpoint"],
                "value_checkpoint_id": bootstrap[
                    "bootstrap_value_checkpoint_id"
                ],
                "freeze_state_encoder": True,
                "update_value_network": False,
                "save_checkpoints": False,
                "require_component_checkpoints": True,
                "reward_shaper_valid": True,
                "critic_lifecycle_valid": False,
                "analysis_export_include_reward_fields": True,
            },
        )
        training_dir = resilience_dir / "critic_training"
        training_dir.mkdir(parents=True, exist_ok=True)
        successful_training_specs, _ = _run_spec_batch(
            training_config,
            training_specs,
            training_dir,
            None,
            experiment_seed,
            execution_settings,
        )
        training_exports = collect_critic_exports(
            successful_training_specs, training_dir
        )
        training_manifest = train_value_checkpoint_from_exports(
            critic_cfg,
            training_exports,
            bootstrap_checkpoint=Path(bootstrap["bootstrap_value_checkpoint"]),
            output_path=trained_path,
            judge_model=judge_model,
            state_encoder_id=state_id,
        )
        if bool(training_manifest.get("valid", False)):
            value_path = str(training_manifest["value_checkpoint"])
            value_id = str(training_manifest["value_checkpoint_id"])
            value_trained = True

    lifecycle_valid = bool(
        preflight.get("valid", False)
        and state_id
        and value_id
        and value_trained
    )
    if not value_path:
        # If training is unavailable, workers use the deterministic bootstrap
        # checkpoint and keep that fact as provenance.
        value_path = bootstrap["bootstrap_value_checkpoint"]
        value_id = bootstrap["bootstrap_value_checkpoint_id"]
    lifecycle = {
        "valid": lifecycle_valid,
        "judge_model": judge_model,
        "state_encoder_checkpoint": state_path,
        "state_encoder_id": state_id,
        "value_checkpoint": value_path,
        "value_checkpoint_id": value_id,
        "value_checkpoint_trained": value_trained,
        "reward_shaper_valid": bool(preflight.get("valid", False)),
        "reward_shaper_preflight": preflight,
        "value_training": training_manifest,
        "training_identity": training_identity,
        "training_episode_ids": _resolve_batch_episode_ids(config),
        "checkpoint_file_hashes_before": {
            "state_encoder": _file_sha256(state_path),
            "value_network": _file_sha256(value_path),
        },
    }
    _set_critic_fields(
        config,
        {
            "state_encoder_checkpoint": state_path,
            "state_encoder_id": state_id,
            "value_checkpoint": value_path,
            "value_checkpoint_id": value_id,
            "freeze_state_encoder": True,
            "update_value_network": False,
            "require_component_checkpoints": True,
            "reward_shaper_valid": bool(preflight.get("valid", False)),
            "critic_lifecycle_valid": lifecycle_valid,
        },
    )
    (resilience_dir / "critic_lifecycle.json").write_text(
        json.dumps(lifecycle, indent=2, default=str), encoding="utf-8"
    )
    return lifecycle


def _verify_critic_checkpoints(lifecycle: Dict[str, Any]) -> None:
    if not lifecycle.get("state_encoder_checkpoint") or not lifecycle.get(
        "value_checkpoint"
    ):
        lifecycle["checkpoint_hashes_unchanged"] = False
        lifecycle["valid"] = False
        lifecycle.setdefault("reason", "critic_component_checkpoint_missing")
        return
    before = dict(lifecycle.get("checkpoint_file_hashes_before") or {})
    after = {
        "state_encoder": _file_sha256(
            str(lifecycle.get("state_encoder_checkpoint") or "")
        ),
        "value_network": _file_sha256(
            str(lifecycle.get("value_checkpoint") or "")
        ),
    }
    lifecycle["checkpoint_file_hashes_after"] = after
    lifecycle["checkpoint_hashes_unchanged"] = before == after
    if before != after:
        lifecycle["valid"] = False
        lifecycle["reason"] = "critic_checkpoint_mutated_during_reference_or_evaluation"


def _stage_baseline_lineage(
    baseline_path: Path,
    critic_lifecycle: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    baseline_available = False
    if baseline_path.exists():
        try:
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                metadata = dict(payload.get("metadata") or {})
                baseline_available = bool(payload.get("stages"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            metadata = {}
    signature = dict(metadata.get("signature") or {})
    return {
        "stage_baseline_path": str(baseline_path.resolve()),
        "stage_baseline_id": str(metadata.get("baseline_id") or ""),
        "baseline_contract_id": str(
            metadata.get("baseline_contract_id")
            or signature.get("baseline_contract_id")
            or ""
        ),
        "stage_baseline_parent_id": str(
            metadata.get("stage_baseline_parent_id") or ""
        ),
        "stage_baseline_alignment_scope": str(
            metadata.get("alignment_scope") or STAGE_BASELINE_ALIGNMENT_SCOPE
        ),
        "stage_baseline_available": baseline_available,
        "_stage_baseline_episode_ids": baseline_supported_episode_ids(metadata),
        "state_encoder_id": str(
            metadata.get("state_encoder_id")
            or critic_lifecycle.get("state_encoder_id")
            or ""
        ),
        "value_checkpoint_id": str(
            metadata.get("value_checkpoint_id")
            or critic_lifecycle.get("value_checkpoint_id")
            or ""
        ),
        "judge_model": str(
            metadata.get("judge_model")
            or signature.get("judge_model")
            or critic_lifecycle.get("judge_model")
            or "unconfigured-judge"
        ),
        "reward_shaper_valid": bool(
            critic_lifecycle.get("reward_shaper_valid", False)
        ),
        "critic_lifecycle_valid": bool(critic_lifecycle.get("valid", False)),
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
    stage_baseline_cfg = resilience_cfg.get("stage_baseline", {}) or {}
    clean_tag = str(stage_baseline_cfg.get("clean_tag", "clean"))
    seed_lifecycle = resolve_stage_baseline_seed_lifecycle(resilience_cfg)
    base_results_dir = Path(str(config.paths.results_dir))
    resilience_dir = base_results_dir / str(
        resilience_cfg.get("output_subdir", "resilience")
    )
    resilience_dir.mkdir(parents=True, exist_ok=True)
    experiment_seed = int(resilience_cfg.get("experiment_seed", 47668090))
    execution_settings = ExecutionSettings.from_mapping(
        resilience_cfg.get("execution", {})
    )
    critic_lifecycle = _prepare_critic_lifecycle(
        config,
        resilience_cfg=resilience_cfg,
        resilience_dir=resilience_dir,
        experiment_seed=experiment_seed,
        execution_settings=execution_settings,
    )
    calibration_plan: Dict[str, Any] = {}
    calibration_episode_ids_by_seed: Dict[int, List[str]] = {}
    if seed_lifecycle["enabled"]:
        calibration_cfg = dict(resilience_cfg)
        calibration_cfg["perturbations"] = [clean_tag]
        calibration_cfg["seeds"] = list(seed_lifecycle["calibration_seeds"])
        evaluation_cfg = dict(resilience_cfg)
        evaluation_cfg["seeds"] = list(seed_lifecycle["evaluation_seeds"])
        if mode == "margin":
            evaluation_cfg["intensities"] = list(
                (resilience_cfg.get("boundary_margin", {}) or {}).get(
                    "stress_lambdas", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                )
            )
        requested_calibration_specs = build_spec_grid(calibration_cfg)
        signature = build_stage_baseline_signature(
            config=config,
            resilience_cfg=resilience_cfg,
            clean_specs=requested_calibration_specs,
        )
        if baseline_override_path is None:
            calibration_plan = plan_stage_baseline_calibration(
                stage_baseline_cfg=stage_baseline_cfg,
                signature=signature,
                episode_ids=_resolve_batch_episode_ids(config),
            )
            logger.info(
                "StageBaseline calibration plan: observed=%s scheduled=%s "
                "latest_available=%s",
                calibration_plan.get("trajectory_count_by_episode", {}),
                calibration_plan.get("scheduled", []),
                calibration_plan.get("latest_available", False),
            )
            calibration_episode_ids_by_seed = {
                int(item["seed"]): [str(value) for value in item["episode_ids"]]
                for item in calibration_plan.get("scheduled", [])
            }
            scheduled_cfg = dict(calibration_cfg)
            scheduled_cfg["seeds"] = sorted(calibration_episode_ids_by_seed)
            clean_specs = build_spec_grid(scheduled_cfg)
        else:
            clean_specs = []
        other_specs = build_spec_grid(evaluation_cfg)
        specs = [*clean_specs, *other_specs]
        include_calibration_rows = False
    else:
        specs = build_spec_grid(resilience_cfg)
        clean_specs, other_specs = partition_specs(specs, clean_tag)
        include_calibration_rows = True
    if not specs:
        raise ValueError(
            "No perturbation specs configured; set resilience.perturbations/seeds."
        )

    run_local_baseline_path = resolve_baseline_output_path(
        base_results_dir,
        resilience_cfg,
    )
    baseline_path = (
        Path(baseline_override_path)
        if baseline_override_path is not None
        else run_local_baseline_path
    )

    stage_baseline_enabled = bool(stage_baseline_cfg.get("enabled", True))
    baseline_only = bool(stage_baseline_cfg.get("baseline_only", False))
    aggregate_fallback_path: Optional[Path] = None
    baseline_resolution = resolve_stage_baseline_startup_candidate(
        config=config,
        resilience_cfg=resilience_cfg,
        clean_specs=(
            requested_calibration_specs
            if seed_lifecycle["enabled"]
            else clean_specs
        ),
        run_local_path=run_local_baseline_path,
        baseline_override_path=Path(baseline_override_path)
        if baseline_override_path is not None
        else None,
    )
    selected_initial_path = baseline_resolution.get("selected_path")
    fitting_current_calibration = bool(
        seed_lifecycle["enabled"]
        and clean_specs
        and baseline_override_path is None
    )
    if selected_initial_path and bool(
        baseline_resolution.get("selected_path_exists")
    ):
        baseline_path = Path(str(selected_initial_path))
    if fitting_current_calibration:
        # Reference workers only collect the clean trajectories used to fit the
        # baseline.  C_rec is undefined until that fit has completed, so these
        # workers must not be pointed at a fabricated, absent JSON file.
        logger.info(
            "Fitting a fresh immutable StageBaseline from the current "
            "calibration phase; reference workers collect evidence without "
            "loading a StageBaseline."
        )
    elif baseline_resolution.get("source") == "episode_aggregate_composed":
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

    if (
        baseline_only
        and not clean_specs
        and baseline_override_path is None
        and not baseline_path.exists()
        and not bool(calibration_plan.get("materialize_existing_evidence"))
    ):
        raise ValueError(
            "resilience.stage_baseline.baseline_only=true requires at least "
            "one clean perturbation spec."
        )

    executed_specs: List[PerturbationSpec] = []
    executed_clean_specs: List[PerturbationSpec] = []
    per_spec_csvs: List[Path] = []
    episode_aggregate_updates: List[Dict[str, Any]] = []
    refresh_resolutions: List[Dict[str, Any]] = []
    if stage_baseline_enabled and clean_specs:
        for clean_spec in clean_specs:
            reference_config = _critic_phase_config(
                config, "reference", critic_lifecycle
            )
            scheduled_episode_ids = calibration_episode_ids_by_seed.get(
                int(clean_spec.seed)
            )
            if scheduled_episode_ids:
                OmegaConf.update(
                    reference_config,
                    "episode_ids",
                    list(scheduled_episode_ids),
                    merge=False,
                    force_add=True,
                )
            clean_executed, clean_csvs = _run_spec_batch(
                reference_config,
                [clean_spec],
                resilience_dir,
                None,
                experiment_seed,
                execution_settings,
            )
            if include_calibration_rows:
                executed_specs.extend(clean_executed)
            executed_clean_specs.extend(clean_executed)
            if include_calibration_rows:
                per_spec_csvs.extend(clean_csvs)
            if (
                clean_executed
                and not seed_lifecycle["enabled"]
                and not bool(stage_baseline_cfg.get("evidence_store_enabled", True))
            ):
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
        if not seed_lifecycle["enabled"]:
            executed_clean_specs, per_spec_csvs = _run_spec_batch(
                _critic_phase_config(config, "reference", critic_lifecycle),
                clean_specs,
                resilience_dir,
                None,
                experiment_seed,
                execution_settings,
            )
            executed_specs.extend(executed_clean_specs)

    generated_baseline_path: Optional[Path] = None
    if stage_baseline_enabled and (
        clean_specs or bool(calibration_plan.get("materialize_existing_evidence"))
    ):
        generated_baseline_path = generate_stage_baseline(
            base_results_dir=base_results_dir,
            resilience_output_dir=resilience_dir,
            resilience_cfg=resilience_cfg,
            # Keep requested calibration specs in the signature even when an
            # entire clean seed fails; successful exports are discovered from
            # the canonical per-spec artifacts and completeness is explicit.
            clean_specs=clean_specs,
            config=config,
            # The cumulative fit aggregates all calibration seeds in one
            # estimator pass. Incremental per-seed summaries normalize their
            # features independently and therefore are diagnostic-only.
            update_episode_aggregate=(
                seed_lifecycle["enabled"]
                or not bool(episode_aggregate_updates)
            ),
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
                "at %s; C_rec and Stability fields will be marked missing.",
                baseline_path,
            )
        extra_executed, extra_csvs = _run_spec_batch(
            _critic_phase_config(config, "evaluation", critic_lifecycle),
            other_specs,
            resilience_dir,
            baseline_path,
            experiment_seed,
            execution_settings,
        )
        executed_specs.extend(extra_executed)
        per_spec_csvs.extend(extra_csvs)

    _verify_critic_checkpoints(critic_lifecycle)
    (resilience_dir / "critic_lifecycle.json").write_text(
        json.dumps(critic_lifecycle, indent=2, default=str), encoding="utf-8"
    )

    raw_output = resilience_dir / (
        "resilience_raw.csv" if mode == "stability" else f"resilience_{mode}.csv"
    )
    baseline_signature = baseline_resolution.get("signature") or {}
    lineage = _stage_baseline_lineage(baseline_path, critic_lifecycle)
    judge_model = str(lineage.get("judge_model") or "unconfigured-judge")
    judge_base_url = str(baseline_signature.get("judge_base_url") or "")
    if executed_specs:
        raw_rows = merge_spec_csvs(
            executed_specs,
            per_spec_csvs,
            raw_output,
            experiment_seed=experiment_seed,
            row_metadata={
                "judge_base_url": judge_base_url,
                **lineage,
            },
        )
    else:
        raw_rows = []
        write_csv_rows(
            raw_output,
            raw_rows,
            empty_fieldnames=tuple(
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
                        "judge_model",
                        "judge_base_url",
                        "stage_baseline_path",
                        "stage_baseline_id",
                        "baseline_contract_id",
                        "stage_baseline_parent_id",
                        "stage_baseline_alignment_scope",
                        "stage_baseline_available",
                        "stage_baseline_episode_id",
                        "stage_baseline_episode_match",
                        "state_encoder_id",
                        "value_checkpoint_id",
                        "reward_shaper_valid",
                        "critic_lifecycle_valid",
                        *RUNTIME_METRIC_KEYS,
                        *POSTPROCESS_METRIC_KEYS,
                    )
                )
            ),
        )
        logger.warning(
            "No episode rollout completed; wrote an empty raw-results artifact. "
            "Failure evidence remains in %s.",
            resilience_dir / "execution_manifest.json",
        )
    logger.info("Wrote %d merged rows -> %s", len(raw_rows), raw_output)

    if postprocess_modes is not None:
        requested_postprocess = list(postprocess_modes)
    elif mode == "joint":
        # Rebound is already collected inside every episode rollout.
        # Stability and GE require the complete clean/stress neighborhood and
        # therefore run, in that order, on the same merged rollout rows.
        requested_postprocess = ["stability", "margin"]
    else:
        requested_postprocess = [mode] if mode in {"stability", "margin"} else []
    if (
        "margin" in requested_postprocess
        and "stability" not in requested_postprocess
    ):
        requested_postprocess.insert(0, "stability")
    dataset_path = OmegaConf.select(config, "habitat.dataset.data_path")

    stability_aggregates: Dict[str, Any] = {}
    margin_aggregates: Dict[str, Any] = {}
    if "stability" in requested_postprocess and raw_rows:
        stability_aggregates = run_stability_aggregation(
            raw_rows,
            resilience_dir,
            resilience_cfg,
            baseline_path=baseline_path,
            dataset_path=str(dataset_path) if dataset_path else None,
            raw_output_path=raw_output,
        )
        logger.info("Aggregated β̂ for %d anchors.", len(stability_aggregates))
    if "margin" in requested_postprocess and raw_rows:
        margin_aggregates = run_boundary_aggregation(
            raw_rows,
            resilience_dir,
            resilience_cfg,
            dataset_path=str(dataset_path) if dataset_path else None,
            raw_output_path=raw_output,
        )
        logger.info(
            "Aggregated stress-boundary margin for %d (family, lambda, e) cells.",
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
                "stage_baseline_seed_lifecycle": seed_lifecycle,
                "stage_baseline_calibration_plan": calibration_plan,
                "critic_lifecycle": critic_lifecycle,
                "stage_baseline_lineage": lineage,
                "calibration_specs": [asdict(s) for s in clean_specs],
                "evaluation_specs": [asdict(s) for s in other_specs],
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
    baseline_path: Optional[Path],
    experiment_seed: int,
    settings: ExecutionSettings,
) -> tuple[List[PerturbationSpec], List[Path]]:
    """Execute episode-isolated workers with bounded host concurrency."""

    if not specs:
        return [], []

    # A seed is the scheduling barrier.  Do not allow the first jobs from the
    # next seed to overlap the tail of the current seed, even when worker slots
    # become available at different times.
    seed_order = list(dict.fromkeys(int(spec.seed) for spec in specs))
    if len(seed_order) > 1:
        completed_specs: List[PerturbationSpec] = []
        completed_csvs: List[Path] = []
        for seed in seed_order:
            seed_specs = [spec for spec in specs if int(spec.seed) == seed]
            logger.info(
                "Starting seed-major resilience batch seed=%d specs=%d",
                seed,
                len(seed_specs),
            )
            seed_completed, seed_csvs = _run_spec_batch(
                config,
                seed_specs,
                resilience_dir,
                baseline_path,
                experiment_seed,
                settings,
            )
            completed_specs.extend(seed_completed)
            completed_csvs.extend(seed_csvs)
        return completed_specs, completed_csvs

    if settings.episodes_per_worker != 1:
        logger.warning(
            "Overriding resilience.execution.episodes_per_worker=%d to 1 so "
            "one episode failure cannot discard another episode's result.",
            settings.episodes_per_worker,
        )
        settings = ExecutionSettings(
            max_parallel_workers=settings.max_parallel_workers,
            episodes_per_worker=1,
            min_available_memory_gb=settings.min_available_memory_gb,
        )

    episode_ids = _resolve_batch_episode_ids(config)
    episode_chunks = chunk_episode_ids(episode_ids, settings.episodes_per_worker)
    if not episode_chunks:
        raise ValueError("No episodes selected for resilience rollout")

    # Each invocation writes into a fresh attempt subtree. Worker CSV writers
    # may append, so reusing a prior worker directory would silently mix reruns.
    attempt_id = uuid.uuid4().hex
    jobs = []
    job_metadata: Dict[str, Dict[str, Any]] = {}
    for spec_index, spec in enumerate(specs):
        artifact_key = spec_artifact_key(spec)
        spec_dir = spec_output_dir(resilience_dir, spec)
        run_seed = derive_run_seed(experiment_seed, spec.seed)
        for chunk_index, episode_chunk in enumerate(episode_chunks):
            worker_dir = (
                spec_dir
                / "workers"
                / f"attempt_{attempt_id}"
                / _episode_worker_name(episode_chunk, chunk_index)
            )
            job_id = (
                f"{artifact_key}::{attempt_id}::{','.join(episode_chunk)}"
            )
            payload = {
                "spec_index": spec_index,
                "chunk_index": chunk_index,
                "spec": spec,
                "episode_ids": episode_chunk,
                "worker_dir": worker_dir,
            }
            jobs.append((job_id, payload))
            job_metadata[job_id] = {
                "spec": asdict(spec),
                "spec_artifact_key": artifact_key,
                "run_seed": run_seed,
                "perturbation_requested": float(spec.intensity),
                "episode_ids": list(episode_chunk),
                "worker_dir": str(worker_dir),
                "resource_usage_path": str(worker_dir / "resource_usage.json"),
                "worker_log_path": str(worker_dir / "worker.log"),
            }

    def run_job(payload: Mapping[str, Any]) -> Dict[str, Any]:
        csv_path = rollout_one_spec(
            config,
            payload["spec"],
            resilience_dir,
            base_runner=base_runner,
            baseline_path=baseline_path,
            seed=experiment_seed,
            episode_ids=payload["episode_ids"],
            worker_dir=payload["worker_dir"],
        )
        return {
            "spec_index": int(payload["spec_index"]),
            "chunk_index": int(payload["chunk_index"]),
            "csv_path": str(csv_path),
        }

    try:
        report = run_bounded_jobs(
            jobs,
            run_job,
            settings,
            continue_after_worker_failure=True,
        )
    except BatchExecutionError as exc:
        _append_execution_manifest(
            resilience_dir,
            exc.report,
            settings,
            job_metadata,
            status="failed",
        )
        raise RuntimeError(
            "Resilience worker batch failed; StageBaseline/GE post-processing was stopped. "
            f"See {resilience_dir / 'execution_manifest.json'}"
        ) from exc

    per_spec_worker_csvs: Dict[int, List[tuple[int, Path]]] = {
        index: [] for index in range(len(specs))
    }
    failed_outcomes = [
        outcome for outcome in report.outcomes if outcome.status == "failed"
    ]
    for outcome in failed_outcomes:
        metadata = job_metadata.get(outcome.job_id, {})
        logger.error(
            "Resilience worker failed job=%s episodes=%s error=%s",
            outcome.job_id,
            metadata.get("episode_ids", []),
            outcome.error or "unknown worker error",
        )
    for outcome in report.outcomes:
        if outcome.status != "completed":
            continue
        result = outcome.result or {}
        per_spec_worker_csvs[int(result["spec_index"])].append(
            (int(result["chunk_index"]), Path(str(result["csv_path"])))
        )

    successful_specs: List[PerturbationSpec] = []
    canonical_csvs: List[Path] = []
    for spec_index, spec in enumerate(specs):
        ordered_worker_csvs = [
            path
            for _chunk_index, path in sorted(per_spec_worker_csvs[spec_index])
        ]
        if not ordered_worker_csvs:
            logger.warning(
                "Spec %s produced no successful episode rows; excluding it from "
                "metric aggregation.",
                spec.tag(),
            )
            continue
        canonical_path = spec_output_dir(resilience_dir, spec) / "episode_metrics.csv"
        _merge_worker_episode_csvs(ordered_worker_csvs, canonical_path)
        successful_specs.append(spec)
        canonical_csvs.append(canonical_path)

    _append_execution_manifest(
        resilience_dir,
        report,
        settings,
        job_metadata,
        status="completed_with_failures" if failed_outcomes else "completed",
    )
    if failed_outcomes:
        logger.warning(
            "Completed resilience batch with %d failed episode worker(s); "
            "successful episodes remain available for aggregation. See %s.",
            len(failed_outcomes),
            resilience_dir / "execution_manifest.json",
        )
    teardown_between_specs()
    return successful_specs, canonical_csvs


def _resolve_batch_episode_ids(config: DictConfig) -> List[str]:
    configured_ids = config.get("episode_ids", None)
    if configured_ids is None:
        configured_ids = config.get("evolve_episode_ids", None)
    if configured_ids is not None:
        return [str(episode_id) for episode_id in configured_ids]

    dataset = base_runner.CollaborationDatasetV0(config.habitat.dataset)
    episode_indices = config.get("episode_indices", None)
    if episode_indices is not None:
        indices = [int(index) for index in episode_indices]
        if any(index < 0 or index >= len(dataset.episodes) for index in indices):
            raise ValueError("Episode index out of range")
        return [str(dataset.episodes[index].episode_id) for index in indices]
    return [str(episode.episode_id) for episode in dataset.episodes]


def _episode_worker_name(episode_ids: Sequence[str], chunk_index: int) -> str:
    slug = "_".join(
        re.sub(r"[^A-Za-z0-9_.-]+", "-", str(episode_id)).strip("-") or "episode"
        for episode_id in episode_ids
    )
    if len(episode_ids) == 1:
        return f"episode_{slug}"
    return f"chunk_{chunk_index:04d}_episodes_{slug}"


def _merge_worker_episode_csvs(worker_csvs: Sequence[Path], output_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for csv_path in worker_csvs:
        if not csv_path.exists():
            raise RuntimeError(f"Worker episode CSV is missing: {csv_path}")
        critic_exports = _index_critic_exports(csv_path.parent)
        for raw_row in read_csv_rows(csv_path):
            row = dict(raw_row)
            # Rebuild the link only from this attempt's unique worker tree.
            row.pop("critic_export_path", None)
            export_path = critic_exports.get(str(row.get("episode_id", "")))
            if export_path:
                row["critic_export_path"] = export_path
                export = load_full_stability_export(Path(export_path))
                if export.valid:
                    for key in (
                        "judge_model",
                        "state_encoder_id",
                        "value_checkpoint_id",
                        "reward_shaper_valid",
                        "critic_lifecycle_valid",
                    ):
                        value = export.metadata.get(key)
                        if value not in (None, ""):
                            row[key] = value
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No episode rows were produced for {output_path.parent.name}")
    write_csv_rows(output_path, rows)
    for row in rows:
        episode_id = row.get("episode_id", "unknown")
        print(f"Metrics For {output_path.parent.name} Episode {episode_id}:")
        for key, value in primary_display_items(row):
            print(f"  {key}: {value}")


def _append_execution_manifest(
    resilience_dir: Path,
    report: BatchRun,
    settings: ExecutionSettings,
    job_metadata: Mapping[str, Mapping[str, Any]],
    *,
    status: str,
) -> None:
    manifest_path = resilience_dir / "execution_manifest.json"
    manifest: Dict[str, Any] = {"batches": []}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
                manifest.setdefault("batches", [])
        except (OSError, json.JSONDecodeError):
            pass

    jobs = []
    for outcome in report.outcomes:
        metadata = dict(job_metadata.get(outcome.job_id, {}))
        resource_path = Path(str(metadata.get("resource_usage_path", "")))
        resource_usage = None
        if resource_path.is_file():
            try:
                resource_usage = json.loads(resource_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        jobs.append(
            {
                "job_id": outcome.job_id,
                "status": outcome.status,
                "error": outcome.error,
                "started_at": outcome.started_at,
                "finished_at": outcome.finished_at,
                **metadata,
                "resource_usage": resource_usage,
            }
        )

    manifest["execution"] = {
        "max_parallel_workers": settings.max_parallel_workers,
        "episodes_per_worker": settings.episodes_per_worker,
        "min_available_memory_gb": settings.min_available_memory_gb,
        "worker_failure_policy": "continue",
    }
    status_counts = {
        outcome_status: sum(
            1 for outcome in report.outcomes if outcome.status == outcome_status
        )
        for outcome_status in ("completed", "failed", "not_started")
    }
    failed_episode_ids = [
        str(episode_id)
        for outcome in report.outcomes
        if outcome.status == "failed"
        for episode_id in job_metadata.get(outcome.job_id, {}).get(
            "episode_ids", []
        )
    ]
    manifest["batches"].append(
        {
            "status": status,
            "max_active_workers": report.max_active_workers,
            "status_counts": status_counts,
            "failed_episode_ids": failed_episode_ids,
            "jobs": jobs,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
