#!/usr/bin/env python3
# isort: skip_file
"""
Utility functions for :mod:`planner_demo_resilience`.

Extracted to keep the main entry-point script minimal and focused on the
high-level experiment orchestration logic.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from habitat_llm.config_redaction import redacted_config_yaml
from habitat_llm.evaluation.metrics.extensibility import (
    BoundaryCollector,
    BoundaryMargin,
    BoundaryThresholds,
)
from habitat_llm.evaluation.metrics.statistical_baseline import (
    write_statistical_baselines,
)
from habitat_llm.evaluation.stage_baseline import (
    STAGE_BASELINE_ALIGNMENT_SCOPE,
    StageBaselineEstimator,
    StageBaselineEvidenceStore,
    baseline_supported_episode_ids,
    judge_models_match,
    normalize_judge_model,
)
from habitat_llm.evaluation.core.critic_lifecycle import (
    checkpoint_component_id,
)
from habitat_llm.evaluation.critic_export_contract import load_full_stability_export
from habitat_llm.evaluation.llm_evaluator.reward_shaper import (
    REWARD_SHAPER_SCHEMA_VERSION,
    REWARD_SHAPER_SCORE_WEIGHTS,
)
from habitat_llm.evaluation.stage_baseline.stage_baseline import (
    DEFAULT_FORMAL_MIN_TRAJECTORIES,
    DEFAULT_FORMAL_MIN_TRANSITIONS,
    RECOVERY_FEATURE_VERSION,
    REBOUND_FORMULA_VERSION,
    STAGE_BASELINE_ESTIMATOR_VERSION,
    STAGE_BASELINE_SCHEMA_VERSION,
    STAGE_RESOLVER_VERSION,
)
from habitat_llm.evaluation.metrics.stability import (
    BetaAggregate,
    StabilityCollector,
)
from habitat_llm.evaluation.metrics.stability.trajectory_loss import (
    compute_stage_trajectory_loss,
)
from habitat_llm.evaluation.metrics.stability.row_augmentation import (
    augment_rows_with_beta_aggregates,
)
from habitat_llm.evaluation.perturbation import (
    PerturbationInjector,
    PerturbationSpec,
    STRESS_FAMILY,
    build_perturbation_spec_grid,
    derive_run_seed,
    spec_artifact_key,
)
from habitat_llm.evaluation.reporting.tabular import (
    read_csv_rows,
    write_csv_rows_exact,
)

logger = logging.getLogger(__name__)
_EPISODE_FAMILY_CACHE: Dict[str, Dict[str, str]] = {}
_ROLLOUT_CONFIG_LOCK = threading.Lock()


# ──────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────
def deep_merge_dicts(
    base: Dict[str, Any], updates: Mapping[str, Any]
) -> Dict[str, Any]:
    """Recursively merge *updates* into *base* without dropping defaults."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def extract_resilience_cfg(
    config: DictConfig,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Read ``resilience.*`` from the merged Hydra config with fallbacks."""
    merged = dict(defaults)
    raw = (
        OmegaConf.select(config, "resilience")
        if hasattr(config, "resilience")
        else None
    )
    if raw is not None:
        resolved = OmegaConf.to_container(raw, resolve=True)
        if isinstance(resolved, Mapping):
            merged = deep_merge_dicts(
                merged, {str(k): v for k, v in resolved.items()}
            )
    if not isinstance(merged.get("perturbations"), (list, tuple)):
        merged["perturbations"] = [str(merged["perturbations"])]
    if not isinstance(merged.get("seeds"), (list, tuple)):
        merged["seeds"] = [int(merged["seeds"])]
    if not isinstance(merged.get("intensities"), (list, tuple)):
        merged["intensities"] = [float(merged["intensities"])]
    return merged


def build_spec_grid(cfg: Mapping[str, Any]) -> List[PerturbationSpec]:
    return build_perturbation_spec_grid(cfg)


def _stage_baseline_support_config(
    stage_baseline_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return statistical support settings; ``formal`` is a legacy alias."""

    configured = stage_baseline_cfg.get("support")
    if configured is None:
        configured = stage_baseline_cfg.get("formal", {})
    return dict(configured or {})


def resolve_stage_baseline_seed_lifecycle(
    resilience_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve the calibration/evaluation seed lifecycle and pairing policy."""

    stage_cfg = resilience_cfg.get("stage_baseline", {}) or {}
    lifecycle = stage_cfg.get("seed_lifecycle", {}) or {}
    enabled = bool(lifecycle.get("enabled", False))
    calibration = [
        int(value)
        for value in lifecycle.get(
            "initial_calibration_seeds",
            lifecycle.get("calibration_seeds", [0, 1, 2]),
        )
    ]
    evaluation = [
        int(value)
        for value in lifecycle.get("evaluation_seeds", [0, 1, 2])
    ]
    if enabled:
        if not calibration or not evaluation:
            raise ValueError(
                "StageBaseline seed lifecycle requires non-empty calibration "
                "and evaluation seed sets"
            )
    else:
        legacy = [int(value) for value in resilience_cfg.get("seeds", [0])]
        calibration = list(legacy)
        evaluation = list(legacy)
    return {
        "enabled": enabled,
        "calibration_seeds": list(dict.fromkeys(calibration)),
        "evaluation_seeds": list(dict.fromkeys(evaluation)),
        "disjoint": not bool(set(calibration) & set(evaluation)) if enabled else False,
        "allow_paired_seeds": True,
    }


def spec_output_dir(
    resilience_output_dir: Path,
    spec: PerturbationSpec,
) -> Path:
    return resilience_output_dir / spec_artifact_key(spec)


def partition_specs(
    specs: Sequence[PerturbationSpec],
    clean_tag: str,
) -> Tuple[List[PerturbationSpec], List[PerturbationSpec]]:
    clean_specs: List[PerturbationSpec] = []
    other_specs: List[PerturbationSpec] = []
    for spec in specs:
        if spec.is_clean or str(spec.kind) == str(clean_tag):
            clean_specs.append(spec)
        else:
            other_specs.append(spec)
    return clean_specs, other_specs


def resolve_baseline_output_path(
    base_results_dir: Path,
    resilience_cfg: Mapping[str, Any],
) -> Path:
    return (
        base_results_dir
        / str(resilience_cfg.get("output_subdir", "resilience"))
        / str(resilience_cfg.get("baseline_json", "stage_baseline.json"))
    )


def habitat_llm_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_persistent_baseline_dir(
    stage_baseline_cfg: Mapping[str, Any],
) -> Path:
    raw_dir = str(
        stage_baseline_cfg.get(
            "persistent_store_dir",
            "evaluation/stage_baselines",
        )
    )
    path = Path(raw_dir)
    if path.is_absolute():
        return path
    return habitat_llm_root() / path


def resolve_judge_baseline_root(
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> Path:
    """Return the persistent root dedicated to one reward Judge model."""
    judge_slug = _slugify(
        signature.get("judge_model_slug") or signature.get("judge_model"),
        default="unconfigured-judge",
    )
    return resolve_persistent_baseline_dir(stage_baseline_cfg) / judge_slug


def resolve_runtime_baseline_root(
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> Path:
    subdir = str(stage_baseline_cfg.get("runtime_subdir", "runtime")).strip()
    return resolve_judge_baseline_root(stage_baseline_cfg, signature) / (
        subdir or "runtime"
    )


def resolve_statistics_baseline_root(
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> Path:
    subdir = str(stage_baseline_cfg.get("statistics_subdir", "statistics")).strip()
    return resolve_judge_baseline_root(stage_baseline_cfg, signature) / (
        subdir or "statistics"
    )


def resolve_stage_baseline_evidence_store(
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> StageBaselineEvidenceStore:
    contract_id = "judge_episode_baseline"
    contract_metadata = {
        "judge_model": signature.get("judge_model"),
        "alignment_scope": STAGE_BASELINE_ALIGNMENT_SCOPE,
    }
    return StageBaselineEvidenceStore(
        resolve_judge_baseline_root(stage_baseline_cfg, signature),
        contract_id=contract_id,
        contract_metadata=contract_metadata,
    )


def plan_stage_baseline_calibration(
    *,
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
    episode_ids: Sequence[Any],
) -> Dict[str, Any]:
    """Schedule the missing cells of the configured fixed clean-seed grid."""

    store = resolve_stage_baseline_evidence_store(stage_baseline_cfg, signature)
    store.ensure_contract()
    episodes = [str(value) for value in episode_ids]
    observed = store.evidence_seed_map(episodes)
    seed_cfg = dict(stage_baseline_cfg.get("seed_lifecycle", {}) or {})
    initial = [
        int(value)
        for value in seed_cfg.get(
            "initial_calibration_seeds",
            seed_cfg.get("calibration_seeds", [0, 1, 2]),
        )
    ]
    evaluation = sorted(
        {int(value) for value in seed_cfg.get("evaluation_seeds", [0, 1, 2])}
    )
    latest = store.latest_snapshot()
    latest_baseline_id = ""
    if latest is not None:
        try:
            with open(latest, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            metadata = payload.get("metadata", {})
            latest_baseline_id = str(metadata.get("baseline_id") or "")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            pass

    scheduled: Dict[int, List[str]] = defaultdict(list)
    for episode_id in episodes:
        present = set(observed.get(episode_id, []))
        for seed in initial:
            if seed not in present:
                scheduled[int(seed)].append(episode_id)

    return {
        "contract_id": store.contract_id,
        "contract_root": str(store.contract_root),
        "observed_seed_map": observed,
        "initial_calibration_seeds": initial,
        "evaluation_seeds": evaluation,
        "scheduled": [
            {"seed": seed, "episode_ids": values}
            for seed, values in sorted(scheduled.items())
        ],
        "latest_snapshot": str(latest) if latest else None,
        "latest_baseline_id": latest_baseline_id,
        "latest_available": latest is not None,
        "trajectory_count_by_episode": {
            episode_id: len(set(observed.get(episode_id, [])))
            for episode_id in episodes
        },
        "materialize_existing_evidence": bool(
            not scheduled and latest is None and any(observed.values())
        ),
        "reuse_policy": str(
            stage_baseline_cfg.get("reuse_policy", "reuse_and_extend")
        ),
    }


def _slugify(value: Any, *, default: str = "na", max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    if not text:
        return default
    return text[:max_len]


def _normalize_sequence(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _select_nested(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cursor: Any = mapping
    for key in path:
        if not isinstance(cursor, Mapping):
            return default
        cursor = cursor.get(key)
        if cursor is None:
            return default
    return cursor


def _identity_digest(payload: Any, *, length: int = 12) -> str:
    """Return a deterministic digest for resolved, JSON-compatible config data."""

    return hashlib.sha1(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:length]


def _planner_identity_labels(payload: Any) -> List[str]:
    """Collect human-readable planner implementation/model labels."""

    labels: set[str] = set()

    def _visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in {
                    "_target_",
                    "engine",
                    "llm_model",
                    "model_name",
                    "served_model_name",
                } and item not in (None, ""):
                    labels.add(str(item))
                else:
                    _visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _visit(item)

    _visit(payload)
    return sorted(labels)


def _configured_baseline_models(cfg_dict: Mapping[str, Any]) -> List[str]:
    labels: List[str] = []
    for key, value in sorted(cfg_dict.items(), key=lambda item: str(item[0])):
        if not isinstance(value, Mapping):
            continue
        baseline_model = value.get("baseline_model")
        if baseline_model not in (None, ""):
            labels.append(f"{key}:{baseline_model}")
    return labels


def _active_hydra_config_name() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return str(HydraConfig.get().job.config_name or "").strip()
    except (AttributeError, ImportError, RuntimeError, ValueError):
        pass
    return ""


def _resolve_evaluated_policy_identity(
    cfg_dict: Mapping[str, Any],
    resilience_cfg: Mapping[str, Any],
    stage_baseline_cfg: Mapping[str, Any],
) -> Dict[str, str]:
    """Fingerprint the evaluated policy and baseline-relevant resolved config.

    Storage/output paths and perturbation sampling axes are deliberately absent:
    they do not define the clean policy being calibrated. Planner prompts,
    generation settings, agent/world-model configuration, critic configuration,
    and the recovery estimator settings remain part of the identity.
    """

    evaluation_cfg = cfg_dict.get("evaluation")
    evaluation_cfg = evaluation_cfg if isinstance(evaluation_cfg, Mapping) else {}
    evaluation_agents = evaluation_cfg.get("agents")
    evaluation_agents = (
        evaluation_agents if isinstance(evaluation_agents, Mapping) else {}
    )
    planner_payload: Dict[str, Any] = {}
    if isinstance(evaluation_cfg.get("planner"), Mapping):
        planner_payload["centralized"] = evaluation_cfg["planner"]
    for agent_name, agent_cfg in sorted(
        evaluation_agents.items(), key=lambda item: str(item[0])
    ):
        if isinstance(agent_cfg, Mapping) and isinstance(
            agent_cfg.get("planner"), Mapping
        ):
            planner_payload[str(agent_name)] = agent_cfg["planner"]

    policy_payload = {
        "evaluation": {
            key: evaluation_cfg[key]
            for key in (
                "type",
                "planner",
                "agents",
                "replan_thresh",
                "max_episode_sim_steps",
                "context_evolve",
            )
            if key in evaluation_cfg
        },
        "agents": cfg_dict.get("agents", {}),
        "world_model": cfg_dict.get("world_model", {}),
        "agent_asymmetry": cfg_dict.get("agent_asymmetry"),
    }
    critic_identity = dict(evaluation_cfg.get("critic", {}) or {})
    for key in (
        "phase",
        "state_encoder_checkpoint",
        "state_encoder_id",
        "value_checkpoint",
        "value_checkpoint_id",
        "freeze_state_encoder",
        "update_value_network",
        "require_component_checkpoints",
        "reward_shaper_valid",
        "critic_lifecycle_valid",
        "save_checkpoints",
        "checkpoint_dir",
        "analysis_save_dir",
        "analysis_export_dir",
        "reward_shaper_log_dir",
        "debug_dir",
    ):
        critic_identity.pop(key, None)
    baseline_config_payload = {
        "policy": policy_payload,
        "critic": critic_identity,
        "metrics": evaluation_cfg.get("metrics", {}),
        "c_rec": resilience_cfg.get("c_rec", {}),
    }
    planner_digest = _identity_digest(planner_payload)
    policy_digest = _identity_digest(policy_payload)
    config_digest = _identity_digest(baseline_config_payload)

    explicit_planner = str(stage_baseline_cfg.get("planner_identity") or "").strip()
    planner_labels = _planner_identity_labels(planner_payload)
    planner_identity = explicit_planner or (
        "|".join(planner_labels) if planner_labels else f"planner-{planner_digest}"
    )
    explicit_policy = str(stage_baseline_cfg.get("policy_identity") or "").strip()
    policy_identity = explicit_policy or f"policy-{policy_digest}"
    explicit_config = str(stage_baseline_cfg.get("config_identity") or "").strip()
    hydra_config = _active_hydra_config_name()
    configured_models = _configured_baseline_models(cfg_dict)
    if explicit_config:
        config_identity = explicit_config
        config_identity_source = "resilience.stage_baseline.config_identity"
    elif hydra_config:
        config_identity = hydra_config
        config_identity_source = "hydra.job.config_name"
    elif configured_models:
        config_identity = "|".join(configured_models)
        config_identity_source = "suite.baseline_model"
    else:
        config_identity = f"resolved-{config_digest}"
        config_identity_source = "resolved_baseline_config"

    return {
        "planner_identity": planner_identity,
        "planner_config_signature": planner_digest,
        "evaluated_policy_identity": policy_identity,
        "evaluated_policy_signature": policy_digest,
        "config_identity": config_identity,
        "config_identity_source": config_identity_source,
        "baseline_config_signature": config_digest,
    }


def _resolve_judge_identity(
    cfg_dict: Mapping[str, Any],
    stage_baseline_cfg: Mapping[str, Any],
) -> Dict[str, str]:
    """Resolve the model that defines the reward scale calibrated by a baseline.

    ``evaluation.critic.llm_model`` is authoritative because it produces the
    online shaped reward. An explicit StageBaseline value is useful for configs
    that construct the critic dynamically, while the ADCA model is retained as
    analysis provenance.
    """
    explicit_model = str(stage_baseline_cfg.get("judge_model") or "").strip()
    critic_model = str(
        _select_nested(cfg_dict, ("evaluation", "critic", "llm_model"), "") or ""
    ).strip()
    adca_model = str(
        _select_nested(cfg_dict, ("evaluation", "adca", "llm_model"), "") or ""
    ).strip()
    if critic_model:
        if explicit_model and explicit_model.casefold() != critic_model.casefold():
            raise ValueError(
                "StageBaseline judge_model must match "
                "evaluation.critic.llm_model: "
                f"{explicit_model!r} != {critic_model!r}"
            )
        judge_model = critic_model
        source = "evaluation.critic.llm_model"
    elif explicit_model:
        judge_model = explicit_model
        source = "resilience.stage_baseline.judge_model_without_critic"
    elif adca_model:
        judge_model = adca_model
        source = "evaluation.adca.llm_model"
    else:
        judge_model = "unconfigured-judge"
        source = "fallback"

    explicit_base_url = str(stage_baseline_cfg.get("judge_base_url") or "").strip()
    critic_base_url = str(
        _select_nested(cfg_dict, ("evaluation", "critic", "llm_base_url"), "")
        or ""
    ).strip()
    adca_base_url = str(
        _select_nested(cfg_dict, ("evaluation", "adca", "llm_base_url"), "")
        or ""
    ).strip()
    if (
        critic_model
        and explicit_base_url
        and critic_base_url
        and explicit_base_url.rstrip("/").casefold()
        != critic_base_url.rstrip("/").casefold()
    ):
        raise ValueError(
            "StageBaseline judge_base_url must match the reward critic base URL: "
            f"{explicit_base_url!r} != {critic_base_url!r}"
        )
    judge_base_url = (
        critic_base_url
        if critic_model
        else explicit_base_url or adca_base_url
    )
    return {
        "judge_model": judge_model,
        "judge_model_slug": _slugify(judge_model, default="unconfigured-judge"),
        "judge_model_source": source,
        "judge_base_url": judge_base_url,
        "critic_judge_model": critic_model,
        "adca_judge_model": adca_model,
    }


def build_stage_baseline_signature(
    *,
    config: Optional[Mapping[str, Any]],
    resilience_cfg: Mapping[str, Any],
    clean_specs: Sequence[PerturbationSpec],
) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        resolved = OmegaConf.to_container(config, resolve=True)
        cfg_dict = dict(resolved) if isinstance(resolved, Mapping) else {}
    else:
        cfg_dict = dict(config or {})
    dataset_path = str(
        _select_nested(cfg_dict, ("habitat", "dataset", "data_path"), "")
    )
    dataset_content_hash = ""
    dataset_file = Path(dataset_path)
    if dataset_path and not dataset_file.is_absolute() and not dataset_file.is_file():
        dataset_file = habitat_llm_root().parent / dataset_file
    if dataset_path and dataset_file.is_file():
        digest = hashlib.sha256()
        with open(dataset_file, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        dataset_content_hash = digest.hexdigest()
    elif dataset_path:
        dataset_content_hash = "unresolved:" + dataset_path.replace("\\", "/")
    episode_ids = (
        _normalize_sequence(cfg_dict.get("episode_ids"))
        or _normalize_sequence(cfg_dict.get("evolve_episode_ids"))
        or _normalize_sequence(resilience_cfg.get("anchors"))
    )
    stage_baseline_cfg = resilience_cfg.get("stage_baseline", {}) or {}
    seed_lifecycle = resolve_stage_baseline_seed_lifecycle(resilience_cfg)
    formal_cfg = _stage_baseline_support_config(stage_baseline_cfg)
    judge_identity = _resolve_judge_identity(cfg_dict, stage_baseline_cfg)
    policy_identity = _resolve_evaluated_policy_identity(
        cfg_dict,
        resilience_cfg,
        stage_baseline_cfg,
    )
    critic_cfg = _select_nested(cfg_dict, ("evaluation", "critic"), {}) or {}
    state_encoder_checkpoint = str(
        critic_cfg.get("state_encoder_checkpoint") or ""
    )
    value_checkpoint = str(critic_cfg.get("value_checkpoint") or "")
    state_encoder_id = str(critic_cfg.get("state_encoder_id") or "") or (
        checkpoint_component_id(state_encoder_checkpoint, "state_encoder")
        if state_encoder_checkpoint
        else ""
    )
    value_checkpoint_id = str(critic_cfg.get("value_checkpoint_id") or "") or (
        checkpoint_component_id(value_checkpoint, "value_network")
        if value_checkpoint
        else ""
    )
    reward_shaper_payload = {
        key: critic_cfg.get(key)
        for key in (
            "use_llm_shaping",
            "llm_model",
            "llm_base_url",
            "shaping_weight",
            "reward_shaper_json_mode",
            "call_on_planning_steps",
            "llm_call_frequency",
        )
    }
    reward_shaper_payload.update(
        {
            "schema_version": REWARD_SHAPER_SCHEMA_VERSION,
            "score_weights": REWARD_SHAPER_SCORE_WEIGHTS,
            "planning_step_window": critic_cfg.get("planning_step_window", {}),
            "evaluate_on_rebound": critic_cfg.get("evaluate_on_rebound"),
        }
    )
    payload = {
        "dataset_path": dataset_path,
        "dataset_content_hash": dataset_content_hash,
        "episode_ids": episode_ids,
        "clean_specs": [spec.tag() for spec in clean_specs],
        "seeds": list(seed_lifecycle["calibration_seeds"]),
        "calibration_seeds": list(seed_lifecycle["calibration_seeds"]),
        "evaluation_seeds": list(seed_lifecycle["evaluation_seeds"]),
        "seed_lifecycle_enabled": bool(seed_lifecycle["enabled"]),
        "allow_paired_seeds": bool(seed_lifecycle["allow_paired_seeds"]),
        "intensities": [float(val) for val in resilience_cfg.get("intensities", [])],
        "baseline_json": str(
            resilience_cfg.get("baseline_json", "stage_baseline.json")
        ),
        "output_subdir": str(resilience_cfg.get("output_subdir", "resilience")),
        **judge_identity,
        **policy_identity,
        "stage_baseline_schema_version": STAGE_BASELINE_SCHEMA_VERSION,
        "recovery_feature_version": RECOVERY_FEATURE_VERSION,
        "rebound_formula_version": REBOUND_FORMULA_VERSION,
        "stage_resolver_version": STAGE_RESOLVER_VERSION,
        "stage_baseline_estimator_version": STAGE_BASELINE_ESTIMATOR_VERSION,
        "formal_min_trajectories": int(
            formal_cfg.get(
                "min_independent_trajectories",
                DEFAULT_FORMAL_MIN_TRAJECTORIES,
            )
        ),
        "formal_min_transitions": int(
            formal_cfg.get("min_transitions", DEFAULT_FORMAL_MIN_TRANSITIONS)
        ),
        "require_successful_clean": bool(
            formal_cfg.get("require_successful_clean", False)
        ),
        "allow_family_covariance_shrinkage": bool(
            formal_cfg.get("allow_family_covariance_shrinkage", False)
        ),
        "state_encoder_id": state_encoder_id,
        "value_checkpoint_id": value_checkpoint_id,
        "alignment_scope": STAGE_BASELINE_ALIGNMENT_SCOPE,
        "reward_shaper_signature": _identity_digest(reward_shaper_payload),
        "reward_shaper_valid": bool(
            critic_cfg.get(
                "reward_shaper_valid",
                not bool(critic_cfg.get("use_llm_shaping", False)),
            )
        ),
        "critic_lifecycle_valid": bool(
            critic_cfg.get("critic_lifecycle_valid", False)
        ),
    }
    # Per-episode aggregates may be composed across episode shards and clean
    # sampling runs, but must never cross dataset, Judge, policy/planner, or
    # estimator identities. Keep that merge identity separate from the run
    # signature, which also records episode and sampling axes.
    episode_signature_payload = {
        key: payload.get(key)
        for key in (
            "dataset_content_hash",
            "judge_model",
            "judge_base_url",
            "planner_identity",
            "planner_config_signature",
            "evaluated_policy_identity",
            "evaluated_policy_signature",
            "config_identity",
            "config_identity_source",
            "baseline_config_signature",
            "stage_baseline_schema_version",
            "recovery_feature_version",
            "rebound_formula_version",
            "stage_resolver_version",
            "stage_baseline_estimator_version",
            "formal_min_trajectories",
            "formal_min_transitions",
            "require_successful_clean",
            "allow_family_covariance_shrinkage",
            "alignment_scope",
            "state_encoder_id",
            "reward_shaper_signature",
        )
    }
    payload["baseline_contract_id"] = _identity_digest(
        episode_signature_payload, length=24
    )
    payload["episode_baseline_signature"] = payload["baseline_contract_id"]
    digest = _identity_digest(payload)
    dataset_slug = _slugify(Path(dataset_path).stem or "dataset", default="dataset")
    if episode_ids:
        head = "_".join(_slugify(ep, default="ep") for ep in episode_ids[:6])
        suffix = "-etc" if len(episode_ids) > 6 else ""
        episode_slug = f"eps-{head}{suffix}"
    else:
        episode_slug = "eps-all"
    seed_values = [str(seed) for seed in payload.get("calibration_seeds", [])]
    seed_slug = _slugify("_".join(seed_values) or "seed-default", default="seed-default")
    payload["signature"] = digest
    payload["archive_filename"] = (
        f"stage_baseline__{dataset_slug}__{episode_slug}__s-{seed_slug}__{digest}.json"
    )
    return payload


def _episode_aggregate_cfg(
    stage_baseline_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    raw = stage_baseline_cfg.get("episode_aggregate", {}) if stage_baseline_cfg else {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def resolve_episode_aggregate_root(
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> Path:
    cfg = _episode_aggregate_cfg(stage_baseline_cfg)
    subdir = str(cfg.get("subdir", "by_episode")).strip() or "by_episode"
    return resolve_statistics_baseline_root(stage_baseline_cfg, signature) / subdir


def _signature_dataset_slug(signature: Mapping[str, Any]) -> str:
    dataset_path = str(signature.get("dataset_path") or "dataset")
    dataset_name = _slugify(Path(dataset_path).stem or "dataset", default="dataset")
    normalized_path = dataset_path.replace("\\", "/")
    return f"{dataset_name}__{_identity_digest(normalized_path, length=10)}"


def _episode_baseline_identity_slug(signature: Mapping[str, Any]) -> str:
    identity = str(
        signature.get("episode_baseline_signature")
        or signature.get("signature")
        or ""
    ).strip()
    if not identity:
        raise ValueError("StageBaseline signature has no episode merge identity")
    return f"identity-{_slugify(identity, default='missing-identity')}"


def _episode_set_slug(episode_ids: Sequence[Any]) -> str:
    ids = [str(ep) for ep in episode_ids]
    if not ids:
        return "eps-all"
    head = "_".join(_slugify(ep, default="ep") for ep in ids[:6])
    digest = hashlib.sha1(
        json.dumps(ids, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:10]
    suffix = "-etc" if len(ids) > 6 else ""
    return f"eps-{head}{suffix}__{digest}"


def _episode_baseline_path(
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
    episode_id: Any,
) -> Path:
    root = resolve_episode_aggregate_root(stage_baseline_cfg, signature)
    dataset_slug = _signature_dataset_slug(signature)
    identity_slug = _episode_baseline_identity_slug(signature)
    ep_slug = _slugify(episode_id, default="unknown")
    return (
        root
        / dataset_slug
        / identity_slug
        / f"episode_{ep_slug}"
        / "stage_baseline.json"
    )


def _runtime_episode_baseline_path(
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
    episode_id: Any,
    runtime_record_id: Optional[str],
) -> Path:
    dataset_slug = _signature_dataset_slug(signature)
    identity_slug = _episode_baseline_identity_slug(signature)
    ep_slug = _slugify(episode_id, default="unknown")
    record_slug = _slugify(
        runtime_record_id or signature.get("signature"),
        default="clean",
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"stage_baseline__{record_slug}__{timestamp}.json"
    return (
        resolve_runtime_baseline_root(stage_baseline_cfg, signature)
        / dataset_slug
        / identity_slug
        / f"episode_{ep_slug}"
        / filename
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_baseline_signature_status(
    path: Path,
    expected_signature: Mapping[str, Any],
    *,
    episode_scope: bool = False,
) -> Tuple[bool, str]:
    """Check that a readable baseline belongs to the configured Judge."""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"baseline_unreadable:{type(exc).__name__}"
    if not isinstance(payload, Mapping):
        return False, "baseline_payload_not_mapping"
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return False, "baseline_metadata_missing"
    recorded_signature = metadata.get("signature", {})
    if not isinstance(recorded_signature, Mapping):
        recorded_signature = {}
    expected_judge = str(expected_signature.get("judge_model") or "").strip()
    recorded_judge = str(
        metadata.get("judge_model")
        or recorded_signature.get("judge_model")
        or ""
    ).strip()
    if expected_judge and not judge_models_match(expected_judge, recorded_judge):
        return False, "judge_model_mismatch"
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        return False, "stage_baseline_statistics_missing"
    return True, ""


def _load_validated_stage_baselines(
    path: Path,
    expected_signature: Mapping[str, Any],
    *,
    episode_scope: bool = False,
) -> Dict[Tuple[str, int, str], Any]:
    valid, reason = _stage_baseline_signature_status(
        path,
        expected_signature,
        episode_scope=episode_scope,
    )
    if not valid:
        raise ValueError(
            f"Refusing to load StageBaseline with incompatible identity at "
            f"{path}: {reason}"
        )
    return StageBaselineEstimator.load(path)


def _baseline_total_samples(baselines: Mapping[Tuple[str, int, str], Any]) -> int:
    total = 0
    for baseline in baselines.values():
        try:
            total += int(getattr(baseline, "n_samples", 0) or 0)
        except (TypeError, ValueError):
            continue
    return int(total)


def _anchor_key_to_record(
    key: Tuple[str, int, str],
    baseline: Optional[Any] = None,
) -> Dict[str, Any]:
    family, stage, modality = key
    record: Dict[str, Any] = {
        "family": str(family),
        "stage": int(stage),
        "modality": str(modality),
    }
    if baseline is not None:
        try:
            record["n_samples"] = int(getattr(baseline, "n_samples", 0) or 0)
        except (TypeError, ValueError):
            record["n_samples"] = 0
        for attr in ("W_rem_star_swe", "W_rem_seg_bar", "tau_star"):
            try:
                record[attr] = float(getattr(baseline, attr))
            except (TypeError, ValueError, AttributeError):
                continue
        for attr in ("n_transitions", "n_trajectories", "n_seeds"):
            try:
                record[attr] = int(getattr(baseline, attr))
            except (TypeError, ValueError, AttributeError):
                record[attr] = 0
        record["formal_valid"] = bool(getattr(baseline, "formal_valid", False))
        record["formal_missing_reason"] = str(
            getattr(baseline, "formal_missing_reason", "") or ""
        )
        record["recovery_covariance_rank"] = dict(
            getattr(baseline, "recovery_covariance_rank", {}) or {}
        )
        record["recovery_covariance_condition"] = dict(
            getattr(baseline, "recovery_covariance_condition", {}) or {}
        )
        record["recovery_covariance_condition_reason"] = dict(
            getattr(baseline, "recovery_covariance_condition_reason", {}) or {}
        )
        record["recovery_covariance_shrinkage"] = dict(
            getattr(baseline, "recovery_covariance_shrinkage", {}) or {}
        )
        record["recovery_covariance_scope"] = dict(
            getattr(baseline, "recovery_covariance_scope", {}) or {}
        )
        record["recovery_covariance_source"] = dict(
            getattr(baseline, "recovery_covariance_source", {}) or {}
        )
        record["formal_support_scope"] = str(
            getattr(baseline, "formal_support_scope", "exact_stage")
            or "exact_stage"
        )
        episode_stage_work = dict(
            getattr(baseline, "W_rem_star_by_episode_stage", {}) or {}
        )
        record["remaining_work_episode_count"] = len(episode_stage_work)
        record["remaining_work_episode_ids"] = sorted(
            str(value) for value in episode_stage_work
        )
        record["duplicate_episode_seed_trajectories"] = list(
            getattr(baseline, "duplicate_episode_seed_trajectories", []) or []
        )
    return record


def _anchor_records(
    baselines: Mapping[Tuple[str, int, str], Any],
    keys: Optional[Sequence[Tuple[str, int, str]]] = None,
) -> List[Dict[str, Any]]:
    selected = list(keys) if keys is not None else list(baselines.keys())
    return [
        _anchor_key_to_record(key, baselines.get(key))
        for key in sorted(selected, key=lambda item: (str(item[0]), int(item[1]), str(item[2])))
    ]


def _baseline_provenance(
    baselines: Mapping[Tuple[str, int, str], Any],
) -> Dict[str, Any]:
    trajectory_ids = sorted(
        {
            str(value)
            for baseline in baselines.values()
            for value in getattr(baseline, "trajectory_ids", [])
        }
    )
    seed_ids = sorted(
        {
            int(value)
            for baseline in baselines.values()
            for value in getattr(baseline, "seed_ids", [])
        }
    )
    source_hashes = sorted(
        {
            str(value)
            for baseline in baselines.values()
            for value in getattr(baseline, "source_hashes", [])
        }
    )
    return {
        "trajectory_ids": trajectory_ids,
        "seed_ids": seed_ids,
        "source_hashes": source_hashes,
        "formal_anchor_count": sum(
            1 for baseline in baselines.values() if bool(getattr(baseline, "formal_valid", False))
        ),
        "diagnostic_anchor_count": sum(
            1 for baseline in baselines.values() if not bool(getattr(baseline, "formal_valid", False))
        ),
    }


def _calibration_completeness(
    *,
    signature: Mapping[str, Any],
    baselines: Mapping[Tuple[str, int, str], Any],
    estimator_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    expected_episode_ids = [str(value) for value in signature.get("episode_ids", [])]
    expected_seeds = [int(value) for value in signature.get("calibration_seeds", signature.get("seeds", []))]
    succeeded = [dict(value) for value in estimator_stats.get("succeeded_trajectories", [])]
    succeeded_pairs = [
        (str(value.get("episode_id")), int(value["seed"]))
        for value in succeeded
        if value.get("episode_id") is not None and value.get("seed") is not None
    ]
    succeeded_counts = Counter(succeeded_pairs)
    succeeded_keys = set(succeeded_counts)
    expected_keys = {
        (episode_id, seed)
        for episode_id in expected_episode_ids
        for seed in expected_seeds
    }
    missing = sorted(expected_keys - succeeded_keys)
    unexpected = sorted(succeeded_keys - expected_keys)
    duplicates = sorted(
        (episode_id, seed, count)
        for (episode_id, seed), count in succeeded_counts.items()
        if count > 1
    )
    missing_seed_metadata = [
        dict(value) for value in succeeded if value.get("seed") is None
    ]
    provenance = _baseline_provenance(baselines)
    reasons: List[str] = []
    if not expected_episode_ids:
        reasons.append("expected_episode_ids_missing")
    if not expected_seeds:
        reasons.append("calibration_seeds_missing")
    if missing:
        reasons.append("calibration_trajectories_incomplete")
    if unexpected:
        reasons.append("unexpected_calibration_trajectories")
    if duplicates:
        reasons.append("duplicate_episode_seed_trajectories")
    if missing_seed_metadata:
        reasons.append("calibration_seed_metadata_missing")
    if provenance["formal_anchor_count"] <= 0:
        reasons.append("no_formal_anchor")
    return {
        "formal_valid": not reasons,
        "formal_missing_reasons": reasons,
        "expected_episode_ids": expected_episode_ids,
        "expected_seeds": expected_seeds,
        "expected_trajectories": [
            {"episode_id": episode_id, "seed": seed}
            for episode_id, seed in sorted(expected_keys)
        ],
        "succeeded_trajectories": succeeded,
        "missing_trajectories": [
            {"episode_id": episode_id, "seed": seed}
            for episode_id, seed in missing
        ],
        "unexpected_trajectories": [
            {"episode_id": episode_id, "seed": seed}
            for episode_id, seed in unexpected
        ],
        "duplicate_trajectories": [
            {"episode_id": episode_id, "seed": seed, "count": count}
            for episode_id, seed, count in duplicates
        ],
        "missing_seed_metadata_trajectories": missing_seed_metadata,
        "excluded_trajectories": list(
            estimator_stats.get("excluded_trajectories", [])
        ),
        "source_hashes": list(estimator_stats.get("source_hashes", [])),
        **provenance,
    }


def _anchor_update_summary(
    existing: Mapping[Tuple[str, int, str], Any],
    incoming: Mapping[Tuple[str, int, str], Any],
    merged: Mapping[Tuple[str, int, str], Any],
) -> Dict[str, Any]:
    existing_keys = set(existing.keys())
    incoming_keys = set(incoming.keys())
    merged_keys = set(merged.keys())
    new_keys = incoming_keys - existing_keys
    updated_keys = incoming_keys & existing_keys
    retained_keys = existing_keys - incoming_keys
    return {
        "existing_anchor_count": len(existing_keys),
        "incoming_anchor_count": len(incoming_keys),
        "merged_anchor_count": len(merged_keys),
        "new_anchor_count": len(new_keys),
        "updated_anchor_count": len(updated_keys),
        "retained_anchor_count": len(retained_keys),
        "incoming_anchors": _anchor_records(incoming),
        "new_anchors": _anchor_records(merged, list(new_keys)),
        "updated_anchors": _anchor_records(merged, list(updated_keys)),
        "retained_anchors": _anchor_records(merged, list(retained_keys)),
        "merged_anchors": _anchor_records(merged),
    }


def resolve_persistent_hash_baseline_path(
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> Path:
    archive_subdir = str(
        stage_baseline_cfg.get("archive_subdir", "archives")
    ).strip()
    return (
        resolve_statistics_baseline_root(stage_baseline_cfg, signature)
        / (archive_subdir or "archives")
        / str(signature["archive_filename"])
    )


def compose_episode_aggregate_baseline(
    *,
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
    episode_ids: Sequence[Any],
) -> Optional[Path]:
    """Compose per-episode persistent baselines into one loadable file."""
    cfg = _episode_aggregate_cfg(stage_baseline_cfg)
    if not bool(cfg.get("enabled", True)):
        return None
    ids = [str(ep) for ep in episode_ids if str(ep)]
    if not ids:
        return None

    allow_partial = bool(cfg.get("allow_partial_compose", False))
    combined: Dict[Tuple[str, int, str], Any] = {}
    used_paths: List[str] = []
    used_episode_records: List[Dict[str, Any]] = []
    missing_ids: List[str] = []
    incomplete_episode_ids: List[str] = []
    unexpected_seed_episode_ids: List[str] = []
    expected_seeds = {
        int(value)
        for value in signature.get("calibration_seeds", signature.get("seeds", []))
    }
    for ep_id in ids:
        ep_path = _episode_baseline_path(stage_baseline_cfg, signature, ep_id)
        if not ep_path.exists():
            missing_ids.append(ep_id)
            continue
        ep_baselines = _load_validated_stage_baselines(
            ep_path,
            signature,
            episode_scope=True,
        )
        if not ep_baselines:
            missing_ids.append(ep_id)
            continue
        combined = StageBaselineEstimator.merge_baseline_maps(combined, ep_baselines)
        episode_provenance = _baseline_provenance(ep_baselines)
        observed_seeds = set(episode_provenance["seed_ids"])
        missing_seeds = sorted(expected_seeds - observed_seeds)
        unexpected_seeds = sorted(observed_seeds - expected_seeds)
        if missing_seeds:
            incomplete_episode_ids.append(ep_id)
        if unexpected_seeds:
            unexpected_seed_episode_ids.append(ep_id)
        used_paths.append(str(ep_path))
        used_episode_records.append(
            {
                "episode_id": str(ep_id),
                "path": str(ep_path),
                "baseline_entry_count": len(ep_baselines),
                "total_samples": _baseline_total_samples(ep_baselines),
                "seed_ids": episode_provenance["seed_ids"],
                "source_hashes": episode_provenance["source_hashes"],
                "missing_seed_ids": missing_seeds,
                "unexpected_seed_ids": unexpected_seeds,
            }
        )

    if not combined:
        return None

    root = resolve_statistics_baseline_root(stage_baseline_cfg, signature)
    dataset_slug = _signature_dataset_slug(signature)
    identity_slug = _episode_baseline_identity_slug(signature)
    output_path = (
        root
        / "composed"
        / dataset_slug
        / identity_slug
        / f"stage_baseline__{_episode_set_slug(ids)}__episode_aggregate.json"
    )
    composed_provenance = _baseline_provenance(combined)
    formal_reasons: List[str] = []
    if missing_ids:
        formal_reasons.append("missing_episode_baselines")
    if incomplete_episode_ids:
        formal_reasons.append("incomplete_calibration_seed_coverage")
    if unexpected_seed_episode_ids:
        formal_reasons.append("unexpected_calibration_seed_coverage")
    if composed_provenance["formal_anchor_count"] <= 0:
        formal_reasons.append("no_formal_anchor")
    metadata = {
        "baseline_kind": "episode_aggregate_composed",
        "signature": dict(signature),
        "judge_model": str(signature.get("judge_model") or ""),
        "judge_model_source": str(signature.get("judge_model_source") or ""),
        "judge_base_url": str(signature.get("judge_base_url") or ""),
        "dataset_path": str(signature.get("dataset_path") or ""),
        "episode_ids": ids,
        "used_episode_baseline_paths": used_paths,
        "used_episode_baselines": used_episode_records,
        "used_episode_baseline_count": len(used_episode_records),
        "missing_episode_ids": missing_ids,
        "incomplete_episode_ids": sorted(incomplete_episode_ids),
        "unexpected_seed_episode_ids": sorted(unexpected_seed_episode_ids),
        "expected_calibration_seeds": sorted(expected_seeds),
        "allow_partial_compose": allow_partial,
        "formal_valid": not formal_reasons,
        "formal_missing_reasons": formal_reasons,
        **composed_provenance,
        "baseline_entry_count": len(combined),
        "total_samples": _baseline_total_samples(combined),
        "composed_anchors": _anchor_records(combined),
        "merge_method": "sample_weighted_summary_merge",
        "created_at": _utc_now_iso(),
    }
    StageBaselineEstimator.write(combined, output_path, metadata=metadata)
    write_json(output_path.with_suffix(".manifest.json"), metadata)
    return output_path


def resolve_stage_baseline_startup_candidate(
    *,
    config: Optional[Mapping[str, Any]],
    resilience_cfg: Mapping[str, Any],
    clean_specs: Sequence[PerturbationSpec],
    run_local_path: Path,
    baseline_override_path: Optional[Path] = None,
) -> Dict[str, Any]:
    stage_baseline_cfg = resilience_cfg.get("stage_baseline", {}) or {}
    signature = build_stage_baseline_signature(
        config=config,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs,
    )
    persistent_hash_path = resolve_persistent_hash_baseline_path(
        stage_baseline_cfg,
        signature,
    )
    result: Dict[str, Any] = {
        "signature": dict(signature),
        "run_local_hash_path": str(run_local_path),
        "run_local_hash_exists": bool(run_local_path.exists()),
        "run_local_hash_valid": False,
        "run_local_hash_invalid_reason": "",
        "persistent_hash_path": str(persistent_hash_path),
        "persistent_hash_exists": bool(persistent_hash_path.exists()),
        "persistent_hash_valid": False,
        "persistent_hash_invalid_reason": "",
        "episode_aggregate_enabled": bool(
            _episode_aggregate_cfg(stage_baseline_cfg).get("enabled", True)
        ),
        "source": "missing",
        "selected_path": None,
        "fallback_reason": "",
    }

    if baseline_override_path is not None:
        override_path = Path(baseline_override_path)
        if not override_path.exists():
            raise ValueError(
                f"Explicit StageBaseline override does not exist: {override_path}"
            )
        valid, reason = _stage_baseline_signature_status(
            override_path,
            signature,
            episode_scope=True,
        )
        if not valid:
            raise ValueError(
                "Refusing explicit StageBaseline override with a different "
                f"Judge or no statistics at {override_path}: {reason}"
            )
        result.update(
            {
                "source": "explicit_override",
                "selected_path": str(override_path),
                "selected_path_exists": True,
                "explicit_override_valid": True,
                "explicit_override_invalid_reason": "",
            }
        )
        return result

    evidence_store: Optional[StageBaselineEvidenceStore] = None
    if bool(stage_baseline_cfg.get("evidence_store_enabled", True)):
        evidence_store = resolve_stage_baseline_evidence_store(
            stage_baseline_cfg, signature
        )
        latest_snapshot = evidence_store.latest_snapshot()
        result["contract_snapshot_path"] = (
            str(latest_snapshot) if latest_snapshot is not None else None
        )
        if latest_snapshot is not None:
            valid, reason = _stage_baseline_signature_status(
                latest_snapshot,
                signature,
            )
            result["contract_snapshot_valid"] = bool(valid)
            result["contract_snapshot_invalid_reason"] = str(reason or "")
            if valid:
                result.update(
                    {
                        "source": "contract_snapshot",
                        "selected_path": str(latest_snapshot),
                        "selected_path_exists": True,
                    }
                )
                return result

    if run_local_path.exists():
        valid, reason = _stage_baseline_signature_status(
            run_local_path,
            signature,
        )
        result["run_local_hash_valid"] = valid
        result["run_local_hash_invalid_reason"] = reason
        if valid:
            result.update(
                {
                    "source": "run_local_hash",
                    "selected_path": str(run_local_path),
                    "selected_path_exists": True,
                }
            )
            return result

    if persistent_hash_path.exists():
        valid, reason = _stage_baseline_signature_status(
            persistent_hash_path,
            signature,
        )
        result["persistent_hash_valid"] = valid
        result["persistent_hash_invalid_reason"] = reason
        if valid:
            result.update(
                {
                    "source": "persistent_hash",
                    "selected_path": str(persistent_hash_path),
                    "selected_path_exists": True,
                }
            )
            return result

    cfg = _episode_aggregate_cfg(stage_baseline_cfg)
    if not bool(cfg.get("enabled", True)) or not bool(
        cfg.get("fallback_when_hash_missing", True)
    ):
        result["fallback_reason"] = "hash_baseline_missing_episode_aggregate_disabled"
        result["selected_path_exists"] = False
        return result

    composed_path = compose_episode_aggregate_baseline(
        stage_baseline_cfg=stage_baseline_cfg,
        signature=signature,
        episode_ids=signature.get("episode_ids") or (),
    )
    result["episode_aggregate_composed_path"] = (
        str(composed_path) if composed_path else None
    )
    result["episode_aggregate_composed_exists"] = bool(
        composed_path is not None and composed_path.exists()
    )
    if composed_path is not None and composed_path.exists():
        composed_valid, composed_reason = _stage_baseline_signature_status(
            composed_path,
            signature,
            episode_scope=True,
        )
        result["episode_aggregate_composed_valid"] = composed_valid
        result["episode_aggregate_composed_invalid_reason"] = composed_reason
        if composed_valid:
            result.update(
                {
                    "source": "episode_aggregate_composed",
                    "selected_path": str(composed_path),
                    "selected_path_exists": True,
                    "fallback_reason": "hash_baseline_missing",
                }
            )
        else:
            result.update(
                {
                    "selected_path_exists": False,
                    "fallback_reason": (
                        "episode_aggregate_unavailable:" + composed_reason
                    ),
                }
            )
    else:
        invalid_candidates = [
            reason
            for reason in (
                result.get("run_local_hash_invalid_reason"),
                result.get("persistent_hash_invalid_reason"),
            )
            if reason
        ]
        result.update(
            {
                "selected_path_exists": False,
                "fallback_reason": (
                    "hash_baseline_identity_invalid:"
                    + "|".join(str(reason) for reason in invalid_candidates)
                    if invalid_candidates
                    else "hash_baseline_missing_episode_aggregate_unavailable"
                ),
            }
        )
    return result


def resolve_episode_aggregate_fallback_path(
    *,
    config: Optional[Mapping[str, Any]],
    resilience_cfg: Mapping[str, Any],
    clean_specs: Sequence[PerturbationSpec],
) -> Optional[Path]:
    stage_baseline_cfg = resilience_cfg.get("stage_baseline", {}) or {}
    cfg = _episode_aggregate_cfg(stage_baseline_cfg)
    if not bool(cfg.get("enabled", True)):
        return None
    if not bool(cfg.get("fallback_when_hash_missing", True)):
        return None
    signature = build_stage_baseline_signature(
        config=config,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs,
    )
    return compose_episode_aggregate_baseline(
        stage_baseline_cfg=stage_baseline_cfg,
        signature=signature,
        episode_ids=signature.get("episode_ids") or (),
    )


def update_episode_aggregated_baselines(
    *,
    estimator: StageBaselineEstimator,
    episode_baselines: Mapping[str, Mapping[Tuple[str, int, str], Any]],
    episode_stats: Mapping[str, Any],
    stage_baseline_cfg: Mapping[str, Any],
    signature: Mapping[str, Any],
    runtime_record_id: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _episode_aggregate_cfg(stage_baseline_cfg)
    if not bool(cfg.get("enabled", True)):
        return {}

    merge_existing = bool(cfg.get("merge_existing", True))
    written_paths: Dict[str, str] = {}
    runtime_paths: Dict[str, str] = {}
    per_episode_anchor_updates: Dict[str, Any] = {}
    for ep_id, incoming in episode_baselines.items():
        incoming_provenance = _baseline_provenance(incoming)
        runtime_path = _runtime_episode_baseline_path(
            stage_baseline_cfg,
            signature,
            ep_id,
            runtime_record_id,
        )
        runtime_metadata = {
            "baseline_kind": "episode_runtime",
            "signature": dict(signature),
            "judge_model": str(signature.get("judge_model") or ""),
            "judge_model_source": str(signature.get("judge_model_source") or ""),
            "judge_base_url": str(signature.get("judge_base_url") or ""),
            "dataset_path": str(signature.get("dataset_path") or ""),
            "episode_id": str(ep_id),
            "runtime_record_id": str(
                runtime_record_id or signature.get("signature") or ""
            ),
            "source_export_count": int(
                (episode_stats.get("episode_export_counts") or {}).get(str(ep_id), 0)
            ),
            "source_record_count": int(
                (episode_stats.get("episode_record_counts") or {}).get(str(ep_id), 0)
            ),
            "baseline_entry_count": len(incoming),
            "total_samples": _baseline_total_samples(incoming),
            "formal_valid": incoming_provenance["formal_anchor_count"] > 0,
            **incoming_provenance,
            "recorded_at": _utc_now_iso(),
        }
        estimator.write(incoming, runtime_path, metadata=runtime_metadata)
        runtime_paths[str(ep_id)] = str(runtime_path)

        output_path = _episode_baseline_path(stage_baseline_cfg, signature, ep_id)
        existing = (
            _load_validated_stage_baselines(
                output_path,
                signature,
                episode_scope=True,
            )
            if merge_existing and output_path.exists()
            else {}
        )
        merged = StageBaselineEstimator.merge_baseline_maps(existing, incoming)
        merged_provenance = _baseline_provenance(merged)
        expected_seeds = {
            int(value)
            for value in signature.get(
                "calibration_seeds",
                signature.get("seeds", []),
            )
        }
        missing_seed_ids = sorted(
            expected_seeds - set(merged_provenance["seed_ids"])
        )
        unexpected_seed_ids = sorted(
            set(merged_provenance["seed_ids"]) - expected_seeds
        )
        formal_reasons: List[str] = []
        if missing_seed_ids:
            formal_reasons.append("incomplete_calibration_seed_coverage")
        if unexpected_seed_ids:
            formal_reasons.append("unexpected_calibration_seed_coverage")
        if merged_provenance["formal_anchor_count"] <= 0:
            formal_reasons.append("no_formal_anchor")
        existing_entry_count = len(existing)
        incoming_entry_count = len(incoming)
        anchor_update = _anchor_update_summary(existing, incoming, merged)
        metadata = {
            "baseline_kind": "episode_aggregate",
            "signature": dict(signature),
            "judge_model": str(signature.get("judge_model") or ""),
            "judge_model_source": str(signature.get("judge_model_source") or ""),
            "judge_base_url": str(signature.get("judge_base_url") or ""),
            "dataset_path": str(signature.get("dataset_path") or ""),
            "episode_id": str(ep_id),
            "incoming_runtime_baseline_path": str(runtime_path),
            "source_export_count": int(
                (episode_stats.get("episode_export_counts") or {}).get(str(ep_id), 0)
            ),
            "source_record_count": int(
                (episode_stats.get("episode_record_counts") or {}).get(str(ep_id), 0)
            ),
            "previous_baseline_entry_count": existing_entry_count,
            "previous_total_samples": _baseline_total_samples(existing),
            "incoming_baseline_entry_count": incoming_entry_count,
            "incoming_total_samples": _baseline_total_samples(incoming),
            "baseline_entry_count": len(merged),
            "merged_total_samples": _baseline_total_samples(merged),
            "anchor_update": anchor_update,
            "merge_method": (
                "sample_weighted_summary_merge" if existing else "exact_run_median"
            ),
            "expected_calibration_seeds": sorted(expected_seeds),
            "missing_seed_ids": missing_seed_ids,
            "unexpected_seed_ids": unexpected_seed_ids,
            "formal_valid": not formal_reasons,
            "formal_missing_reasons": formal_reasons,
            **merged_provenance,
            "updated_at": _utc_now_iso(),
        }
        estimator.write(merged, output_path, metadata=metadata)
        write_json(output_path.with_suffix(".manifest.json"), metadata)
        written_paths[str(ep_id)] = str(output_path)
        per_episode_anchor_updates[str(ep_id)] = anchor_update

    composed_path = compose_episode_aggregate_baseline(
        stage_baseline_cfg=stage_baseline_cfg,
        signature=signature,
        episode_ids=signature.get("episode_ids") or (),
    )
    composed_baselines = (
        _load_validated_stage_baselines(
            composed_path,
            signature,
            episode_scope=True,
        )
        if composed_path is not None and composed_path.exists()
        else {}
    )
    return {
        "judge_model": str(signature.get("judge_model") or ""),
        "judge_baseline_root": str(
            resolve_judge_baseline_root(stage_baseline_cfg, signature)
        ),
        "runtime_root": str(
            resolve_runtime_baseline_root(stage_baseline_cfg, signature)
        ),
        "statistics_root": str(
            resolve_statistics_baseline_root(stage_baseline_cfg, signature)
        ),
        "episode_aggregate_root": str(
            resolve_episode_aggregate_root(stage_baseline_cfg, signature)
        ),
        "episode_runtime_paths": runtime_paths,
        "episode_baseline_paths": written_paths,
        "episode_aggregate_composed_path": str(composed_path) if composed_path else None,
        "episode_aggregate_count": len(written_paths),
        "per_episode_anchor_updates": per_episode_anchor_updates,
        "composed_anchor_count": len(composed_baselines),
        "composed_anchors": _anchor_records(composed_baselines),
    }


def build_stage_baseline_manifest(
    *,
    baseline_path: Path,
    export_paths: Sequence[Path],
    baselines: Mapping[Tuple[str, int, str], Any],
    clean_specs: Sequence[PerturbationSpec],
    signature: Optional[Mapping[str, Any]] = None,
    persistent_path: Optional[Path] = None,
    latest_alias_path: Optional[Path] = None,
    preferred_path: Optional[Path] = None,
) -> Dict[str, Any]:
    anchor_groups = {
        (family, stage) for family, stage, _modality in baselines.keys()
    }
    manifest = {
        "baseline_path": str(baseline_path),
        "preferred_baseline_path": str(preferred_path or baseline_path),
        "persistent_baseline_path": str(persistent_path) if persistent_path else None,
        "latest_alias_path": str(latest_alias_path) if latest_alias_path else None,
        "clean_spec_tags": [spec.tag() for spec in clean_specs],
        "export_file_count": len(export_paths),
        "anchor_group_count": len(anchor_groups),
        "baseline_entry_count": len(baselines),
    }
    if signature:
        manifest["signature"] = dict(signature)
    return manifest


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv_rows(
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fieldnames: Optional[Sequence[str]] = None,
) -> Path:
    fieldnames = (
        sorted({str(key) for row in rows for key in row.keys()})
        if rows
        else [str(key) for key in empty_fieldnames or ()]
    )
    return write_csv_rows_exact(
        output_path,
        rows,
        fieldnames=fieldnames,
        overwrite_policy="replace",
        empty_policy="write_header" if fieldnames else "error",
    )


def _write_csv_artifact(
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fieldnames: Sequence[str],
) -> Path:
    """Replace one run-scoped CSV, writing a header when it has no rows."""

    fieldnames = (
        sorted({str(key) for row in rows for key in row.keys()})
        if rows
        else [str(key) for key in empty_fieldnames]
    )
    if not fieldnames:
        raise ValueError(f"CSV artifact schema is empty: {output_path}")
    return write_csv_rows_exact(
        output_path,
        rows,
        fieldnames=fieldnames,
        overwrite_policy="replace",
        empty_policy="write_header",
    )


def inject_baseline_path(cfg: DictConfig, baseline_path: Path) -> None:
    if cfg.get("evaluation") is None:
        cfg.evaluation = OmegaConf.create({})
    if cfg.evaluation.get("rebound_tracker") is None:
        cfg.evaluation.rebound_tracker = OmegaConf.create({})
    cfg.evaluation.rebound_tracker.baseline_path = str(
        Path(baseline_path).expanduser().resolve()
    )


# ──────────────────────────────────────────────────────────────────────
# Injection helpers
# ──────────────────────────────────────────────────────────────────────
def install_injector_on_runner(
    eval_runner: Any,
    spec: PerturbationSpec,
) -> None:
    """Attach a fresh :class:`PerturbationInjector` to *eval_runner*."""
    injector = PerturbationInjector(spec=spec)
    eval_runner.perturbation_injector = injector


# ──────────────────────────────────────────────────────────────────────
# Per-spec rollout
# ──────────────────────────────────────────────────────────────────────
_runner_hook_installed = False


def enable_runtime_injector_hook(base_runner: Any) -> None:
    """Monkey-patch :func:`base_runner.run_planner` exactly once."""
    global _runner_hook_installed
    if _runner_hook_installed:
        return
    original = getattr(base_runner, "run_planner", None)
    if original is None:
        raise RuntimeError(
            "planner_demo_mp_new.run_planner not found; cannot hook resilience injector"
        )

    def _patched_run_planner(config: DictConfig, *args: Any, **kwargs: Any) -> Any:
        return original(config, *args, **kwargs)

    setattr(base_runner, "_resilience_patched_run_planner", _patched_run_planner)
    _runner_hook_installed = True


def rollout_one_spec(
    config: DictConfig,
    spec: PerturbationSpec,
    resilience_output_dir: Path,
    base_runner: Any,
    baseline_path: Optional[Path] = None,
    seed: int = 47668090,
    episode_ids: Optional[Sequence[Any]] = None,
    worker_dir: Optional[Path] = None,
) -> Path:
    """Run one isolated perturbation worker for one episode shard.

    Each spec is executed in a **subprocess** to guarantee full native-memory
    isolation.  When the child process exits, the OS reclaims *all* resources
    (Habitat C++ simulator heap, GPU allocations, mmap regions) that
    ``env_interface.env.close()`` cannot fully release within the same process.

    Returns the path to the per-spec CSV that was produced.
    """
    import copy
    import subprocess
    import sys

    from habitat_llm.utils import fix_config, setup_config

    cfg = copy.deepcopy(config)
    run_seed = derive_run_seed(seed, spec.seed)
    spec_dir = spec_output_dir(resilience_output_dir, spec)
    spec_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(worker_dir) if worker_dir is not None else spec_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    try:
        cfg.paths.epi_result_file_path = str(run_dir / "episode_metrics.csv")
        cfg.paths.run_result_file_path = str(run_dir / "run_result_log.csv")
        cfg.paths.end_result_file_path = str(run_dir / "end_result_log.csv")
        cfg.paths.results_dir = str(run_dir)
        if baseline_path is not None:
            inject_baseline_path(cfg, baseline_path)
    except Exception as exc:
        logger.warning("Unable to redirect paths for spec %s: %s", spec.tag(), exc)

    if cfg.get("evaluation") is None:
        cfg.evaluation = OmegaConf.create({})
    runtime_extras = dict(spec.extras)
    bank_path = runtime_extras.get("bank_path")
    if bank_path not in (None, ""):
        resolved_bank_path = Path(str(bank_path)).expanduser()
        if not resolved_bank_path.is_absolute():
            resolved_bank_path = (Path.cwd() / resolved_bank_path).resolve()
        runtime_extras["bank_path"] = str(resolved_bank_path)
    cfg.evaluation.runtime_perturbation = OmegaConf.create(
        {
            "kind": spec.kind,
            "seed": int(spec.seed),
            "intensity": float(spec.intensity),
            "extras": runtime_extras,
            "run_seed": int(run_seed),
        }
    )
    cfg.evaluation.runtime_perturbation_seed = int(spec.seed)
    cfg.evaluation.runtime_run_seed = int(run_seed)
    # Resilience workers keep metric-relevant planner events in memory. Rich
    # state around replans is emitted by Critic under <worker>/analyze instead
    # of retaining full WorldGraph snapshots and per-sim-step planner payloads.
    cfg.evaluation.planner_history_mode = "replan_window"
    cfg.evaluation.retain_action_history_world_graph = False
    cfg.evaluation.log_detailed_traces = False
    cfg.num_proc = 1
    if episode_ids is not None:
        for key in (
            "episode_indices",
            "evolve_episode_ids",
            "start_index",
            "end_index",
        ):
            if cfg.get(key, None) is not None:
                del cfg[key]
        cfg.episode_ids = [str(episode_id) for episode_id in episode_ids]
    if cfg.evaluation.get("critic") is not None:
        # Eq. (5) is a trajectory average, so sparse top-k audit exports are
        # insufficient for formal Stability. Keep one state row per transition.
        cfg.evaluation.critic.analysis_export_minimal_schema = True
        cfg.evaluation.critic.analysis_export_full_trajectory = True
        cfg.evaluation.critic.analysis_export_record_mode = "state_only"

    # setup_config registers Hydra plugins and seeds process-global RNGs. Keep
    # that short preparation section serialized; only the isolated subprocess
    # execution itself runs concurrently.
    with _ROLLOUT_CONFIG_LOCK:
        fix_config(cfg)
        cfg = setup_config(cfg, run_seed)

        # Serialize the fully-resolved config so the subprocess can load it
        # without relying on Hydra argument reconstruction.
        spec_config_path = run_dir / "_resolved_config.yaml"
        OmegaConf.set_struct(cfg, False)
        with open(spec_config_path, "w", encoding="utf-8") as handle:
            handle.write(
                redacted_config_yaml(cfg, resolve=True, replacement=None)
            )
        OmegaConf.set_struct(cfg, True)

    csv_output_path = Path(str(cfg.paths.epi_result_file_path))

    logger.info(
        "[resilience] Launching subprocess for spec %s -> %s",
        spec.tag(),
        run_dir,
    )

    # Launch a child process that loads the config and runs run_planner.
    # Using the module entry point avoids Hydra re-initialization issues.
    child_cmd = [
        sys.executable,
        "-m",
        "habitat_llm.examples.resilience_helpers",
        "--spec-config",
        str(spec_config_path),
        "--resource-usage",
        str(run_dir / "resource_usage.json"),
    ]
    worker_log_path = run_dir / "worker.log"
    with open(worker_log_path, "w", encoding="utf-8") as worker_log:
        result = subprocess.run(
            child_cmd,
            check=False,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        resource_usage_path = run_dir / "resource_usage.json"
        worker_error = ""
        if resource_usage_path.is_file():
            try:
                resource_usage = json.loads(
                    resource_usage_path.read_text(encoding="utf-8")
                )
                worker_error = str(resource_usage.get("error") or "")
            except (OSError, json.JSONDecodeError):
                worker_error = ""
        worker_log_tail = ""
        try:
            log_lines = worker_log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            worker_log_tail = "\n".join(log_lines[-40:])
        except OSError:
            worker_log_tail = ""
        raise RuntimeError(
            f"Subprocess for spec {spec.tag()} exited with code "
            f"{result.returncode}"
            + (f": {worker_error}" if worker_error else "")
            + (
                f". Worker log tail:\n{worker_log_tail}"
                if worker_log_tail
                else ""
            )
            + f". Check {worker_log_path}"
        )

    if not csv_output_path.exists():
        raise RuntimeError(
            f"Subprocess for spec {spec.tag()} completed without "
            f"episode metrics: {csv_output_path}"
        )

    return csv_output_path


# ──────────────────────────────────────────────────────────────────────
# Post-processing: merge, aggregate, baseline generation
# ──────────────────────────────────────────────────────────────────────
def _index_critic_exports(spec_dir: Path) -> Dict[str, str]:
    """Map episode id to its critic export inside one perturbation run."""

    index: Dict[str, str] = {}
    export_paths = [
        *spec_dir.rglob("critic_export-episode_*.json"),
        *spec_dir.rglob("critic_export-episode_*.jsonl"),
    ]
    for path in sorted(export_paths):
        episode_id = ""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                first_char = handle.read(1)
                handle.seek(0)
                if first_char == "{":
                    payload = json.load(handle)
                    if isinstance(payload, Mapping):
                        episode_id = str(payload.get("episode_id") or "")
        except (OSError, json.JSONDecodeError):
            continue
        if episode_id:
            index[episode_id] = str(path.resolve())
    return index


def merge_spec_csvs(
    specs: Sequence[PerturbationSpec],
    per_spec_csvs: Sequence[Path],
    output_path: Path,
    *,
    experiment_seed: int = 47668090,
    row_metadata: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if len(specs) != len(per_spec_csvs):
        raise ValueError(
            "Each perturbation spec must have exactly one per-spec CSV: "
            f"specs={len(specs)}, csvs={len(per_spec_csvs)}"
        )
    merged_rows: List[Dict[str, Any]] = []
    metadata = dict(row_metadata or {})
    has_episode_alignment = "_stage_baseline_episode_ids" in metadata
    supported_episode_ids = {
        str(value) for value in metadata.pop("_stage_baseline_episode_ids", ())
    }
    for spec, csv_path in zip(specs, per_spec_csvs):
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Per-spec CSV is missing for {spec_artifact_key(spec)}: {csv_path}"
            )
        for row in read_csv_rows(csv_path):
            row = dict(row)
            row["perturbation_type"] = spec.kind
            row["perturbation_seed"] = spec.seed
            row["perturbation_intensity"] = spec.intensity
            row["perturbation_requested"] = spec.intensity
            stress_family = str(
                spec.extras.get("stress_family") or spec.kind or "clean"
            )
            variant = str(spec.extras.get("variant") or "")
            row.setdefault("perturbation_stress_family", stress_family)
            row.setdefault("stress_family", stress_family)
            row.setdefault("perturbation_variant", variant)
            row.setdefault(
                "perturbation_level_index",
                0 if spec.is_clean else int(round(5.0 * spec.intensity)),
            )
            row.setdefault(
                "perturbation_complete_unit_count",
                0 if spec.is_clean else "",
            )
            row.setdefault(
                "perturbation_text_dose",
                0.0 if spec.is_clean else "",
            )
            row.setdefault(
                "perturbation_alignment_valid",
                bool(spec.is_clean),
            )
            row.setdefault(
                "perturbation_bank_path",
                str(spec.extras.get("bank_path") or ""),
            )
            row["run_seed"] = derive_run_seed(experiment_seed, spec.seed)
            if row.get("perturbation_realized") in (None, ""):
                row["perturbation_realized"] = 0.0 if spec.is_clean else ""
            if row.get("perturbation_stress_realized") in (None, ""):
                row["perturbation_stress_realized"] = row.get(
                    "perturbation_realized", 0.0 if spec.is_clean else ""
                )
            if row.get("perturbation_valid") in (None, ""):
                row["perturbation_valid"] = bool(spec.is_clean)
            if row.get("perturbation_reason") in (None, ""):
                row["perturbation_reason"] = (
                    "clean_noop"
                    if spec.is_clean
                    else "missing_runtime_perturbation_audit"
                )
            if row.get("perturbation_components") in (None, ""):
                row["perturbation_components"] = "{}"
            row.setdefault("anchor_id", row.get("episode_id", "unknown"))
            for key, value in metadata.items():
                if key in {"reward_shaper_valid", "critic_lifecycle_valid"}:
                    parent_valid = (
                        str(value).strip().lower()
                        in {"1", "1.0", "true", "yes"}
                    )
                    if not parent_valid:
                        row[key] = False
                        continue
                    if row.get(key) not in (None, ""):
                        continue
                elif key in {
                    "judge_model",
                    "state_encoder_id",
                    "value_checkpoint_id",
                } and row.get(key) not in (None, ""):
                    continue
                row[key] = value
            if has_episode_alignment:
                episode_id = str(row.get("episode_id") or "")
                episode_match = episode_id in supported_episode_ids
                row["stage_baseline_episode_id"] = (
                    episode_id if episode_match else ""
                )
                row["stage_baseline_episode_match"] = episode_match
            merged_rows.append(row)
    if not merged_rows:
        raise RuntimeError(
            "No episode rows were available for the merged resilience CSV; "
            f"refusing to leave a stale artifact at {output_path}"
        )
    write_csv_rows(output_path, merged_rows)
    return merged_rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _load_episode_family_map(dataset_path: Optional[str]) -> Dict[str, str]:
    if not dataset_path:
        return {}
    dataset_key = str(Path(dataset_path))
    if dataset_key in _EPISODE_FAMILY_CACHE:
        return dict(_EPISODE_FAMILY_CACHE[dataset_key])

    try:
        from habitat_llm.evaluation.experiments.common import (
            infer_task_family,
            load_dataset_episodes,
        )

        family_map: Dict[str, str] = {}
        for episode in load_dataset_episodes(dataset_key):
            episode_id = str(episode.get("episode_id", ""))
            if not episode_id:
                continue
            family_map[episode_id] = infer_task_family(episode)
        _EPISODE_FAMILY_CACHE[dataset_key] = dict(family_map)
        return family_map
    except Exception as exc:
        logger.warning(
            "Failed to infer task family from dataset `%s`: %s",
            dataset_path,
            exc,
        )
        return {}


def normalize_rollout_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    clean_tag: str = "clean",
    dataset_path: Optional[str] = None,
    default_evolve_step: int = 0,
    legacy_compatibility: bool = False,
) -> List[Dict[str, Any]]:
    family_map = _load_episode_family_map(dataset_path)
    normalized: List[Dict[str, Any]] = []

    for row in raw_rows:
        # Normalization is a reporting boundary; do not mutate runtime/source
        # evidence supplied by callers while adding compatibility fields.
        materialized = dict(row)
        episode_id = str(
            materialized.get("episode_id")
            or materialized.get("anchor_id")
            or "unknown"
        )
        materialized["episode_id"] = episode_id
        materialized.setdefault("anchor_id", episode_id)

        perturbation_type = str(materialized.get("perturbation_type") or clean_tag)
        materialized["perturbation_type"] = perturbation_type
        is_clean = perturbation_type.strip().lower() in {
            "",
            str(clean_tag).lower(),
            "clean",
            "baseline",
        }

        if not materialized.get("task_family"):
            materialized["task_family"] = family_map.get(episode_id, "other")

        task_percent_complete = _optional_float(
            materialized.get("task_percent_complete")
        )
        task_state_success = _optional_float(
            materialized.get("task_state_success")
        )
        materialized["task_percent_complete"] = (
            0.0
            if legacy_compatibility and task_percent_complete is None
            else "" if task_percent_complete is None else task_percent_complete
        )
        materialized["task_state_success"] = (
            0.0
            if legacy_compatibility and task_state_success is None
            else "" if task_state_success is None else task_state_success
        )
        total_step_count = materialized.get("total_step_count")
        if total_step_count in (None, ""):
            total_step_count = materialized.get("sim_step_count")
        if total_step_count in (None, ""):
            materialized["total_step_count"] = 0 if legacy_compatibility else ""
        else:
            try:
                materialized["total_step_count"] = int(total_step_count)
            except (TypeError, ValueError):
                materialized["total_step_count"] = (
                    0 if legacy_compatibility else ""
                )

        q_value = materialized.get("proposition_satisfied_fraction")
        if q_value in (None, "") and legacy_compatibility:
            # Older episode CSVs did not always preserve the proposition
            # fraction explicitly. Fall back to the already-exported completion
            # ratio only in explicitly requested legacy-reporting mode.
            materialized["proposition_satisfied_fraction"] = _safe_float(
                materialized.get("task_percent_complete"),
                0.0,
            )
            materialized.setdefault(
                "proposition_satisfied_fraction_source",
                "task_percent_complete_fallback",
            )
        elif q_value in (None, ""):
            materialized["proposition_satisfied_fraction"] = ""
            materialized["proposition_satisfied_fraction_source"] = "missing"
        else:
            parsed_q = _optional_float(q_value)
            materialized["proposition_satisfied_fraction"] = (
                "" if parsed_q is None else parsed_q
            )

        missing_outcomes = []
        if task_percent_complete is None:
            missing_outcomes.append("task_percent_complete")
        if task_state_success is None:
            missing_outcomes.append("task_state_success")
        if _optional_float(materialized["proposition_satisfied_fraction"]) is None:
            missing_outcomes.append("proposition_satisfied_fraction")
        materialized["task_outcome_valid"] = not missing_outcomes
        materialized["task_outcome_missing_reason"] = (
            "" if not missing_outcomes else "missing:" + ",".join(missing_outcomes)
        )

        stress_lambda = materialized.get("stress_lambda")
        if stress_lambda in (None, ""):
            stress_lambda = materialized.get("ge_stress_lambda")
        if stress_lambda in (None, ""):
            stress_lambda = 0.0 if is_clean else _safe_float(
                materialized.get("perturbation_intensity"),
                0.0,
            )
        materialized["stress_lambda"] = _safe_float(stress_lambda, 0.0)
        materialized["ge_stress_lambda"] = materialized["stress_lambda"]

        requested_stress = materialized.get("perturbation_requested")
        if requested_stress in (None, ""):
            requested_stress = materialized["stress_lambda"]
        materialized["stress_requested"] = _safe_float(
            requested_stress,
            0.0,
        )
        # Frozen instruction/context stress exports a dedicated realized
        # coordinate and a final runtime alignment decision.  Prefer those
        # fields at the reporting boundary so the generic perturbation audit
        # cannot silently replace the formal six-level contract.
        realized_stress_raw = materialized.get("perturbation_stress_realized")
        if realized_stress_raw in (None, ""):
            realized_stress_raw = materialized.get("perturbation_realized")
        realized_stress = _optional_float(realized_stress_raw)
        audit_valid_raw = materialized.get("perturbation_alignment_valid")
        if audit_valid_raw in (None, ""):
            audit_valid_raw = materialized.get("perturbation_valid")
        audit_flag_present = audit_valid_raw not in (None, "")
        audit_valid = str(audit_valid_raw).strip().lower() in {
            "1",
            "1.0",
            "true",
            "yes",
        }
        if is_clean:
            materialized["stress_realized"] = 0.0
            materialized["stress_valid"] = True
            materialized["stress_invalid_reason"] = ""
        elif not audit_flag_present:
            materialized["stress_realized"] = (
                "" if realized_stress is None else realized_stress
            )
            materialized["stress_valid"] = False
            materialized["stress_invalid_reason"] = (
                "missing_runtime_perturbation_audit"
            )
            materialized["stress_validation_source"] = (
                "fail_closed_missing_perturbation_valid"
            )
        elif not audit_valid:
            materialized["stress_realized"] = (
                "" if realized_stress is None else realized_stress
            )
            materialized["stress_valid"] = False
            materialized["stress_invalid_reason"] = str(
                materialized.get("perturbation_reason")
                or "runtime_perturbation_audit_invalid"
            )
        elif (
            realized_stress is None
            or not math.isfinite(realized_stress)
            or realized_stress <= 0.0
            or realized_stress > 1.0
        ):
            materialized["stress_realized"] = (
                "" if realized_stress is None else realized_stress
            )
            materialized["stress_valid"] = False
            materialized["stress_invalid_reason"] = (
                "out_of_range_realized_perturbation_dose"
                if realized_stress is not None
                and (not math.isfinite(realized_stress) or realized_stress > 1.0)
                else "missing_or_nonpositive_realized_stress"
            )
        else:
            materialized["stress_realized"] = realized_stress
            materialized["stress_valid"] = True
            materialized["stress_invalid_reason"] = ""
        materialized["stress_text_dose"] = materialized.get(
            "perturbation_text_dose",
            materialized.get("stress_text_dose", ""),
        )
        materialized["stress_alignment_valid"] = bool(
            materialized.get("stress_valid", False)
        )

        evolve_step = materialized.get("evolve_step")
        if evolve_step in (None, ""):
            evolve_step = materialized.get("ge_evolve_step_e")
        evolve_step = _safe_int(evolve_step, default_evolve_step)
        materialized["evolve_step"] = evolve_step
        materialized["ge_evolve_step_e"] = evolve_step

        c_rec_value = _optional_float(
            materialized.get("rebound_c_rec", materialized.get("c_rec"))
        )
        if c_rec_value is None:
            materialized.setdefault("rebound_c_rec", "")
            materialized.setdefault("rebound_c_rec_valid", 0)
            materialized.setdefault(
                "rebound_c_rec_missing_reason",
                "missing_rebound_c_rec",
            )
        else:
            materialized["rebound_c_rec"] = c_rec_value
            if "rebound_c_rec_valid" not in materialized and legacy_compatibility:
                baseline_loaded = _optional_float(materialized.get("rebound_baseline_loaded"))
                if baseline_loaded is not None and baseline_loaded < 1.0:
                    materialized["rebound_c_rec_valid"] = 0
                    materialized.setdefault(
                        "rebound_c_rec_missing_reason",
                        "stage_baseline_missing",
                    )
                else:
                    materialized["rebound_c_rec_valid"] = 1
            elif "rebound_c_rec_valid" not in materialized:
                materialized["rebound_c_rec_valid"] = 0
                materialized.setdefault(
                    "rebound_c_rec_missing_reason",
                    "missing_rebound_c_rec_validity",
                )

        stability_proxy = _optional_float(
            materialized.get(
                "stability_value_delta_variance",
                materialized.get("value_delta_variance"),
            )
        )
        if stability_proxy is None:
            materialized.setdefault("stability_value_delta_variance", "")
        else:
            materialized["stability_value_delta_variance"] = stability_proxy

        p_cbf = _optional_float(materialized.get("stability_p_cbf"))
        if p_cbf is None:
            materialized.setdefault("stability_p_cbf", "")
        else:
            materialized["stability_p_cbf"] = p_cbf

        collision_rate = _optional_float(
            materialized.get(
                "safety_collision_rate",
                materialized.get("collision_rate"),
            )
        )
        if collision_rate is None:
            materialized.setdefault("safety_collision_rate", "")
        else:
            materialized["safety_collision_rate"] = collision_rate

        safety_score = _optional_float(materialized.get("safety_score"))
        if safety_score is None:
            materialized.setdefault("safety_score", "")
        else:
            materialized["safety_score"] = safety_score

        unsafe_rate = _optional_float(materialized.get("unsafe_rate"))
        if unsafe_rate is None:
            materialized.setdefault("unsafe_rate", "")
        else:
            materialized["unsafe_rate"] = unsafe_rate

        if materialized.get("safety_valid") in (None, ""):
            materialized.setdefault("safety_valid", "")
            materialized.setdefault("safety_scope", "")
            materialized.setdefault(
                "safety_missing_reason", "missing_safety_validity"
            )

        normalized_cbf_penalty_rate = _optional_float(
            materialized.get("normalized_cbf_penalty_rate")
        )
        if normalized_cbf_penalty_rate is None:
            normalized_total_steps = _optional_float(
                materialized.get("total_step_count")
            )
            if (
                p_cbf is not None
                and normalized_total_steps is not None
                and normalized_total_steps > 0
            ):
                normalized_cbf_penalty_rate = float(
                    p_cbf / normalized_total_steps
                )
        if normalized_cbf_penalty_rate is None:
            materialized.setdefault("normalized_cbf_penalty_rate", "")
        else:
            materialized["normalized_cbf_penalty_rate"] = normalized_cbf_penalty_rate

        normalized.append(materialized)

    return normalized


def run_stability_aggregation(
    raw_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    cfg: Mapping[str, Any],
    *,
    baseline_path: Optional[Path] = None,
    dataset_path: Optional[str] = None,
    raw_output_path: Optional[Path] = None,
) -> Dict[str, BetaAggregate]:
    if not raw_rows:
        raise ValueError("Stability aggregation requires at least one rollout row")
    # Both the direct runner and Exp-Q suite must enter the shared pairing
    # logic with the same reporting aliases. The suite already normalizes its
    # rows; the direct runner reaches this function immediately after merging
    # worker CSVs and otherwise has only ``perturbation_intensity``.
    normalized_inputs = normalize_rollout_rows(
        raw_rows,
        clean_tag=str(
            (cfg.get("stage_baseline", {}) or {}).get("clean_tag", "clean")
        ),
        dataset_path=dataset_path,
    )
    for row, normalized in zip(raw_rows, normalized_inputs):
        row.update(normalized)
    statistics_dir = output_dir / "statistics"
    diagnostics_dir = output_dir / "diagnostics"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    expected_lineage: Dict[str, str] = {}
    supported_episode_ids: set[str] = set()
    baseline_available = baseline_path is None
    baseline_artifact_missing_reason = ""
    if baseline_path is not None:
        try:
            with open(baseline_path, "r", encoding="utf-8") as handle:
                baseline_payload = json.load(handle)
            baseline_metadata = dict(baseline_payload.get("metadata") or {})
            baseline_signature = dict(baseline_metadata.get("signature") or {})
            expected_lineage = {
                "stage_baseline_path": str(Path(baseline_path).resolve()),
                "judge_model": str(
                    baseline_metadata.get("judge_model")
                    or baseline_signature.get("judge_model")
                    or ""
                ),
            }
            supported_episode_ids = set(
                baseline_supported_episode_ids(baseline_metadata)
            )
            baseline_available = bool(baseline_payload.get("stages"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            expected_lineage = {}
            baseline_available = False
            baseline_artifact_missing_reason = "stage_baseline_unreadable"
        baselines = StageBaselineEstimator.load(baseline_path)
        regularizer = float(
            (cfg.get("stability", {}) or {}).get(
                "mahalanobis_regularizer", 1.0e-3
            )
        )
        minimum_coverage = float(
            (cfg.get("stability", {}) or {}).get(
                "minimum_trajectory_coverage", 0.0
            )
        )
        result_cache: Dict[str, Dict[str, Any]] = {}
        for row in raw_rows:
            export_path = str(row.get("critic_export_path") or "")
            if not export_path:
                row.update(
                    {
                        "stability_trajectory_loss": "",
                        "stability_trajectory_loss_valid": 0,
                        "stability_trajectory_loss_steps": 0,
                        "stability_trajectory_loss_candidate_steps": 0,
                        "stability_trajectory_loss_coverage": 0.0,
                        "stability_trajectory_loss_minimum_coverage": (
                            minimum_coverage
                        ),
                        "stability_trajectory_loss_missing_reason": (
                            "critic_export_path_missing"
                        ),
                    }
                )
                continue
            if export_path not in result_cache:
                result_cache[export_path] = compute_stage_trajectory_loss(
                    Path(export_path),
                    baselines,
                    regularizer=regularizer,
                    minimum_coverage=minimum_coverage,
                ).to_row()
            row.update(result_cache[export_path])

    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        if baseline_path is not None and not raw_row.get("stage_baseline_path"):
            raw_row["stage_baseline_path"] = str(Path(baseline_path).resolve())
        lineage_present = any(
            raw_row.get(key) not in (None, "")
            for key in (
                "stage_baseline_path",
                "judge_model",
            )
        )
        if not lineage_present and not expected_lineage:
            continue
        mismatched = [
            key
            for key, expected in expected_lineage.items()
            if key != "judge_model"
            and expected
            and str(raw_row.get(key) or "") != expected
        ]
        judge_mismatch = bool(
            expected_lineage.get("judge_model")
            and normalize_judge_model(raw_row.get("judge_model"))
            != normalize_judge_model(expected_lineage["judge_model"])
        )
        episode_id = str(raw_row.get("episode_id") or "")
        episode_mismatch = bool(
            baseline_path is not None and episode_id not in supported_episode_ids
        )
        export_episode_mismatch = False
        export_path = str(raw_row.get("critic_export_path") or "")
        if export_path:
            export = load_full_stability_export(Path(export_path))
            if export.valid:
                export_episode_ids = {
                    str(item.get("episode_id") or "") for item in export.records
                }
                export_episode_ids.discard("")
                export_episode_mismatch = export_episode_ids != {episode_id}
        if (
            not baseline_available
            or mismatched
            or judge_mismatch
            or episode_mismatch
            or export_episode_mismatch
        ):
            reason = (
                baseline_artifact_missing_reason or "stage_baseline_missing"
                if not baseline_available
                else (
                    "stage_baseline_judge_mismatch"
                    if judge_mismatch
                    else (
                        "stage_baseline_episode_mismatch"
                        if episode_mismatch
                        else (
                            "critic_export_episode_mismatch"
                            if export_episode_mismatch
                            else "mixed_baseline_lineage"
                        )
                    )
                )
            )
            raw_row.update(
                {
                    "stability_trajectory_loss": "",
                    "stability_trajectory_loss_valid": 0,
                    "stability_trajectory_loss_missing_reason": reason,
                    "stability_lineage_valid": 0,
                    "stability_lineage_missing_reason": reason,
                }
            )
        else:
            raw_row["stability_lineage_valid"] = 1
            raw_row["stability_lineage_missing_reason"] = ""

    beta_cfg = cfg.get("beta", {}) or {}
    alpha = float(beta_cfg.get("alpha", cfg.get("alpha", 0.5)))
    include_rebound = bool(
        beta_cfg.get(
            "include_rebound",
            cfg.get("include_rebound_in_stability", False),
        )
    )

    def _aggregate_bundle(rows: Sequence[Mapping[str, Any]]) -> Tuple[Any, ...]:
        active_collector = StabilityCollector(
            config=dict(cfg.get("stability", {}) or {})
        )
        active_aggregates, active_stats_rows = (
            active_collector.aggregate_beta_over_anchors(
                rows,
                alpha=alpha,
                include_rebound=include_rebound,
            )
        )
        active_perturbations, _ = active_collector.aggregate_beta_by_perturbation(
            rows,
            alpha=alpha,
            include_rebound=include_rebound,
        )
        active_stress, _ = active_collector.aggregate_beta_by_stress(
            rows,
            alpha=alpha,
            include_rebound=include_rebound,
        )
        active_stress_seed = active_collector.aggregate_beta_by_stress_seed(
            rows,
            alpha=alpha,
            include_rebound=include_rebound,
        )
        active_families = active_collector.aggregate_beta_by_family(
            active_aggregates
        )
        return (
            active_collector,
            active_aggregates,
            active_stats_rows,
            active_perturbations,
            active_stress,
            active_stress_seed,
            active_families,
        )

    (
        collector,
        aggregates,
        stats_rows,
        perturbation_aggregates,
        stress_aggregates,
        stress_seed_aggregates,
        family_aggregates,
    ) = _aggregate_bundle(raw_rows)

    # First materialize lambda-local beta so BoundaryCollector can evaluate the
    # complete formal GE contract (outcomes, safety, C_rec, lineage and beta).
    # Then re-fit Stability on exactly that same episode x seed support set.
    # This prevents a seed excluded from GE from remaining in the headline
    # Stability estimate.
    augment_rows_with_beta_aggregates(
        raw_rows,
        aggregates,
        perturbation_aggregates,
        family_aggregates,
        stress_aggregates,
        stress_seed_aggregates,
    )

    def _is_formal_stress_row(row: Mapping[str, Any]) -> bool:
        return bool(
            str(
                row.get("perturbation_stress_family")
                or row.get("stress_family")
                or ""
            ).strip().lower()
            == STRESS_FAMILY
            or row.get("perturbation_bank_path") not in (None, "")
        )

    formal_rows = [row for row in raw_rows if _is_formal_stress_row(row)]
    excluded_formal_rows: List[Mapping[str, Any]] = []
    if formal_rows:
        pairing_probe = BoundaryCollector(
            thresholds=BoundaryThresholds.from_mapping(
                dict(cfg.get("boundary_margin", {}) or {})
            )
        )
        pairing_probe.aggregate(raw_rows)

        def _evidence_pair_key(
            evidence: Any,
        ) -> Tuple[str, int, str, str, str, int]:
            return (
                str(evidence.episode_id),
                int(evidence.perturbation_seed),
                str(evidence.stress_family),
                normalize_judge_model(evidence.judge_model),
                str(evidence.family),
                int(evidence.evolve_step),
            )

        eligible_pairs = {
            _evidence_pair_key(evidence)
            for evidence in pairing_probe.episode_evidence
            if evidence.strict_stress_contract
            and evidence.paired_grid_eligible
            and evidence.valid
        }

        def _row_pair_key(
            row: Mapping[str, Any],
        ) -> Tuple[str, int, str, str, str, int]:
            try:
                seed = int(float(row.get("perturbation_seed", 0) or 0))
            except (TypeError, ValueError):
                seed = 0
            try:
                evolve_step = int(
                    float(
                        row.get(
                            "evolve_step",
                            row.get("ge_evolve_step_e", 0),
                        )
                        or 0
                    )
                )
            except (TypeError, ValueError):
                evolve_step = 0
            return (
                str(row.get("episode_id") or row.get("anchor_id") or "unknown"),
                seed,
                str(
                    row.get("perturbation_stress_family")
                    or row.get("stress_family")
                    or ""
                ),
                normalize_judge_model(row.get("judge_model")),
                str(row.get("task_family") or "other"),
                evolve_step,
            )

        shared_rows: List[Mapping[str, Any]] = []
        for row in raw_rows:
            if not _is_formal_stress_row(row):
                shared_rows.append(row)
                continue
            eligible = _row_pair_key(row) in eligible_pairs
            row["stability_shared_pair_eligible"] = int(eligible)
            row["stability_shared_pair_missing_reason"] = (
                "" if eligible else "excluded_from_shared_formal_pair_set"
            )
            if eligible:
                shared_rows.append(row)
            else:
                excluded_formal_rows.append(row)

        (
            collector,
            aggregates,
            stats_rows,
            perturbation_aggregates,
            stress_aggregates,
            stress_seed_aggregates,
            family_aggregates,
        ) = _aggregate_bundle(shared_rows)

        # Remove the provisional aggregate fields before materializing the
        # shared-P result. Per-rollout trajectory evidence remains untouched.
        for row in raw_rows:
            for key in list(row):
                if key.startswith("stability_beta_") or key in {
                    "stability_clean_ref_valid",
                    "stability_clean_ref_strategy",
                    "stability_n_clean_ref",
                }:
                    row.pop(key, None)

        augment_rows_with_beta_aggregates(
            raw_rows,
            aggregates,
            perturbation_aggregates,
            family_aggregates,
            stress_aggregates,
            stress_seed_aggregates,
        )
        for row in excluded_formal_rows:
            for key in list(row):
                if key.startswith("stability_beta_") or key in {
                    "stability_clean_ref_valid",
                    "stability_clean_ref_strategy",
                    "stability_n_clean_ref",
                }:
                    row.pop(key, None)
            row.update(
                {
                    "stability_beta_neighborhood": "",
                    "stability_beta_neighborhood_valid": 0,
                    "stability_beta_neighborhood_missing_reason": (
                        "excluded_from_shared_formal_pair_set"
                    ),
                    "stability_beta_at_lambda": "",
                    "stability_beta_at_lambda_valid": 0,
                    "stability_beta_at_lambda_missing_reason": (
                        "excluded_from_shared_formal_pair_set"
                    ),
                    "stability_beta_at_lambda_scope": "invalid",
                }
            )

    collector.write_beta_neighborhood_csv(
        aggregates, statistics_dir / "stability_beta_neighborhood.csv"
    )
    collector.write_beta_family_csv(
        family_aggregates,
        output_dir / "resilience_stability_summary.csv",
    )
    collector.write_beta_neighborhood_csv(
        perturbation_aggregates,
        diagnostics_dir / "stability_beta_by_perturbation.csv",
    )
    collector.write_beta_neighborhood_csv(
        stress_aggregates,
        diagnostics_dir / "stability_beta_by_stress.csv",
    )
    tests = collector.pairwise_family_tests(stats_rows, metric_key="beta_hat")
    with open(
        statistics_dir / "stability_beta_family_tests.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(tests, handle, indent=2)
    if raw_output_path is not None:
        write_csv_rows(raw_output_path, raw_rows)
    return aggregates


def run_boundary_aggregation(
    raw_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    cfg: Mapping[str, Any],
    *,
    dataset_path: Optional[str] = None,
    raw_output_path: Optional[Path] = None,
) -> Dict[Tuple[Any, ...], Any]:
    if not raw_rows:
        raise ValueError("Boundary aggregation requires at least one rollout row")
    if not all(
        "stability_beta_at_lambda" in row for row in raw_rows
    ):
        raise ValueError(
            "Boundary aggregation requires lambda-local Stability evidence; "
            "run_stability_aggregation on the same rows first."
        )
    statistics_dir = output_dir / "statistics"
    diagnostics_dir = output_dir / "diagnostics"
    normalized_rows = normalize_rollout_rows(
        raw_rows,
        clean_tag=str((cfg.get("stage_baseline", {}) or {}).get("clean_tag", "clean")),
        dataset_path=dataset_path,
    )
    thresholds = BoundaryThresholds.from_mapping(
        dict(cfg.get("boundary_margin", {}) or {})
    )
    collector = BoundaryCollector(thresholds=thresholds)
    margins = collector.aggregate(normalized_rows)

    judge_model = next(
        (
            str(row.get("judge_model"))
            for row in normalized_rows
            if row.get("judge_model")
        ),
        str((cfg.get("stage_baseline", {}) or {}).get("judge_model") or "unconfigured-judge"),
    )
    judge_base_url = next(
        (
            str(row.get("judge_base_url"))
            for row in normalized_rows
            if row.get("judge_base_url")
        ),
        "",
    )
    stage_baseline_cfg = cfg.get("stage_baseline", {}) or {}
    signature = {
        "judge_model": judge_model,
        "judge_model_slug": _slugify(
            judge_model, default="unconfigured-judge"
        ),
    }
    dataset_slug = _slugify(
        Path(str(dataset_path or "dataset")).stem,
        default="dataset",
    )
    persistent_runtime_evidence = (
        resolve_runtime_baseline_root(stage_baseline_cfg, signature)
        / "family_boundaries"
        / dataset_slug
        / (
            "ge_boundary_clean_evidence__"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + ".csv"
        )
    )
    clean_reference_rows = [
        row.to_diagnostic_dict()
        for row in collector.episode_evidence
        if row.valid
        and (
            collector._is_explicit_clean(row)
            or abs(float(row.stress_lambda)) <= 1.0e-12
        )
    ]
    write_persistent_copy = bool(
        stage_baseline_cfg.get("write_persistent_copy", True)
    )
    if clean_reference_rows and write_persistent_copy:
        write_csv_rows(persistent_runtime_evidence, clean_reference_rows)
    baseline_metadata = {
        "scope": "task_family",
        "metric": "ge_boundary",
        "judge_model": judge_model,
        "judge_base_url": judge_base_url,
        "dataset_path": str(dataset_path or ""),
        "runtime_evidence": "diagnostics/ge_boundary_episode_diagnostics.csv",
        "persistent_runtime_evidence": (
            str(persistent_runtime_evidence)
            if clean_reference_rows and write_persistent_copy
            else ""
        ),
    }
    write_statistical_baselines(
        collector.statistical_baselines,
        statistics_dir / "ge_boundary_baseline.json",
        metadata=baseline_metadata,
    )
    reference_rows = collector.statistical_baseline_rows()
    _write_csv_artifact(
        statistics_dir / "ge_boundary_reference_statistics.csv",
        reference_rows,
        empty_fieldnames=(
            "task_family",
            "reference_evolve_step",
            "judge_model",
            "judge_base_url",
            "scope",
            "reference_valid",
            "reference_missing_reason",
            "min_samples",
            "metric",
            "direction",
            "n",
            "mean",
            "std",
            "lower_quantile",
            "median",
            "upper_quantile",
            "quantile_alpha",
        ),
    )
    if collector.statistical_baselines:
        persistent_path = (
            resolve_statistics_baseline_root(stage_baseline_cfg, signature)
            / "family_boundaries"
            / dataset_slug
            / "ge_boundary_baseline_latest.json"
        )
        if write_persistent_copy:
            write_statistical_baselines(
                collector.statistical_baselines,
                persistent_path,
                metadata=baseline_metadata,
            )

    audc_map = collector.finalize_audc(margins) if margins else {}
    episode_rows = [row.to_dict() for row in collector.episode_evidence if row.valid]
    episode_diagnostic_rows = [
        row.to_diagnostic_dict() for row in collector.episode_evidence
    ]
    invalid_rows = list(collector.invalid_rows)
    episode_summary_fields = tuple(
        collector.episode_evidence[0].to_dict().keys()
    )
    episode_diagnostic_fields = tuple(
        collector.episode_evidence[0].to_diagnostic_dict().keys()
    )
    _write_csv_artifact(
        diagnostics_dir / "ge_boundary_episode_summary.csv",
        episode_rows,
        empty_fieldnames=episode_summary_fields,
    )
    _write_csv_artifact(
        diagnostics_dir / "ge_boundary_episode_diagnostics.csv",
        episode_diagnostic_rows,
        empty_fieldnames=episode_diagnostic_fields,
    )
    _write_csv_artifact(
        diagnostics_dir / "ge_boundary_invalid_rows.csv",
        invalid_rows,
        empty_fieldnames=episode_summary_fields,
    )

    cell_rows = [
        collector.margin_to_dict(margin) for margin in margins.values()
    ]
    cell_diagnostic_rows = [
        collector.margin_to_diagnostic_dict(margin)
        for margin in margins.values()
    ]
    margin_template = BoundaryMargin()
    _write_csv_artifact(
        statistics_dir / "ge_boundary_cells.csv",
        cell_rows,
        empty_fieldnames=tuple(collector.margin_to_dict(margin_template).keys()),
    )
    _write_csv_artifact(
        diagnostics_dir / "ge_boundary_cell_diagnostics.csv",
        cell_diagnostic_rows,
        empty_fieldnames=tuple(
            collector.margin_to_diagnostic_dict(margin_template).keys()
        ),
    )
    capacity_fields = (
        "judge_model",
        "task_family",
        "family",
        "stress_family",
        "stress_contract_formal",
        "evolve_step",
        "ge_boundary_lambda_star",
        "ge_hard_lambda_star",
        "ge_audc_f",
        "ge_audc_norm_f",
        "ge_valid_lambda_coverage",
        "ge_valid_cell_count",
        "ge_capacity_valid",
        "ge_capacity_missing_reason",
        "ge_max_degradation_rate",
        "ge_margin_drop_f",
        "ge_monotonicity_violations",
        "ge_monotonicity_violation",
        "ge_cliff_flag",
        "ge_cliff_lambda",
        "ge_coverage_warning",
        "ge_total_episode_seed_pairs",
        "ge_complete_episode_seed_pairs",
        "ge_episode_seed_pair_coverage",
    )
    capacity_diagnostic_fields = (
        *capacity_fields,
        "ge_missing_lambdas",
        "ge_first_failing_lambda",
        "ge_last_passing_lambda",
        "ge_first_hard_failing_lambda",
        "ge_last_hard_passing_lambda",
        "ge_expected_lambda_count",
        "ge_valid_lambda_count",
        "ge_cell_count",
    )
    _write_csv_artifact(
        output_dir / "resilience_boundary_capacity.csv",
        collector.capacity_rows,
        empty_fieldnames=capacity_fields,
    )
    _write_csv_artifact(
        diagnostics_dir / "ge_boundary_capacity_diagnostics.csv",
        collector.capacity_diagnostic_rows,
        empty_fieldnames=capacity_diagnostic_fields,
    )

    audc_rows = []
    for curve, audc in sorted(audc_map.items(), key=lambda item: str(item[0])):
        if len(curve) == 4:
            judge_model, family, stress_family, evolve_step = curve
        else:
            family, evolve_step = curve
            judge_model = "unconfigured-judge"
            stress_family = "legacy"
        capacity = next(
            (
                row
                for row in collector.capacity_rows
                if str(row.get("judge_model")) == str(judge_model)
                and str(row.get("task_family")) == str(family)
                and str(row.get("stress_family")) == str(stress_family)
                and int(row.get("evolve_step", 0) or 0) == int(evolve_step)
            ),
            {},
        )
        audc_rows.append(
            {
                "judge_model": judge_model,
                "task_family": family,
                "stress_family": stress_family,
                "evolve_step": evolve_step,
                "ge_audc_f": float(audc),
                "ge_audc_norm_f": capacity.get("ge_audc_norm_f", 0.0),
                "ge_boundary_lambda_star": capacity.get(
                    "ge_boundary_lambda_star", ""
                ),
                "ge_hard_lambda_star": capacity.get("ge_hard_lambda_star", ""),
                "ge_valid_lambda_coverage": capacity.get(
                    "ge_valid_lambda_coverage", 0.0
                ),
                "ge_margin_drop_f": capacity.get("ge_margin_drop_f", 0.0),
                "ge_monotonicity_violations": capacity.get(
                    "ge_monotonicity_violations", 0
                ),
                "ge_monotonicity_violation": capacity.get(
                    "ge_monotonicity_violation", 0
                ),
                "ge_max_degradation_rate": capacity.get(
                    "ge_max_degradation_rate", 0.0
                ),
                "ge_cliff_flag": capacity.get("ge_cliff_flag", 0),
            }
        )
    _write_csv_artifact(
        statistics_dir / "ge_boundary_curve_statistics.csv",
        audc_rows,
        empty_fieldnames=(
            "judge_model",
            "task_family",
            "stress_family",
            "evolve_step",
            "ge_audc_f",
            "ge_audc_norm_f",
            "ge_boundary_lambda_star",
            "ge_hard_lambda_star",
            "ge_valid_lambda_coverage",
            "ge_margin_drop_f",
            "ge_monotonicity_violations",
            "ge_monotonicity_violation",
            "ge_max_degradation_rate",
            "ge_cliff_flag",
        ),
    )

    margin_lookup: Dict[Tuple[Any, ...], Any] = {}
    for margin in margins.values():
        if bool(getattr(margin, "ge_formal_stress_contract", False)):
            key = (
                str(getattr(margin, "ge_judge_model", "unconfigured-judge")),
                margin.family,
                str(getattr(margin, "ge_stress_family", "unresolved_stress_family")),
                float(margin.stress_lambda),
                int(margin.evolve_step),
            )
        else:
            key = (
                margin.family,
                float(margin.stress_lambda),
                int(margin.evolve_step),
            )
        margin_lookup[key] = margin
    for row in normalized_rows:
        strict_stress_contract = bool(
            str(
                row.get("perturbation_stress_family")
                or row.get("stress_family")
                or ""
            ).strip().lower()
            == STRESS_FAMILY
            or row.get("perturbation_bank_path") not in (None, "")
        )
        if strict_stress_contract:
            key = (
                normalize_judge_model(
                    str(row.get("judge_model") or "unconfigured-judge")
                ),
                str(row.get("task_family", "other")),
                str(
                    row.get("perturbation_stress_family")
                    or row.get("stress_family")
                    or "unresolved_stress_family"
                ),
                float(row.get("stress_lambda", 0.0) or 0.0),
                int(row.get("evolve_step", 0) or 0),
            )
        else:
            key = (
                str(row.get("task_family", "other")),
                float(row.get("stress_lambda", 0.0) or 0.0),
                int(row.get("evolve_step", 0) or 0),
            )
        margin = margin_lookup.get(key)
        if margin is None:
            continue
        row.update(margin.get_summary_for_csv())

    if raw_output_path is not None:
        write_csv_rows(raw_output_path, normalized_rows)
    # Preserve the historical in-memory result contract, but only commit the
    # normalized/enriched rows after aggregation and artifact writes succeed.
    for source_row, normalized_row in zip(raw_rows, normalized_rows):
        if isinstance(source_row, dict):
            source_row.clear()
            source_row.update(normalized_row)
    return margins


def collect_critic_exports(
    specs: Sequence[PerturbationSpec],
    resilience_output_dir: Path,
) -> List[Path]:
    """Collect exports explicitly linked by the latest successful attempt.

    Each canonical per-spec CSV is replaced after a successful rollout and
    contains paths discovered inside that attempt's unique worker directory.
    The spec directory deliberately retains older attempts for auditing, so a
    recursive scan here could silently substitute historical evidence when the
    current attempt produced no critic export.
    """

    export_paths: List[Path] = []
    for spec in specs:
        csv_path = spec_output_dir(resilience_output_dir, spec) / "episode_metrics.csv"
        if not csv_path.is_file():
            continue
        for row in read_csv_rows(csv_path):
            raw_path = str(row.get("critic_export_path") or "").strip()
            if not raw_path:
                continue
            export_path = Path(raw_path)
            if not export_path.is_absolute():
                export_path = csv_path.parent / export_path
            if export_path.is_file() and export_path.suffix.lower() in {
                ".json",
                ".jsonl",
                ".gz",
            }:
                export_paths.append(export_path)
    deduped = {str(path.resolve()): path for path in export_paths}
    return list(deduped.values())


def collect_critic_export_provenance(
    specs: Sequence[PerturbationSpec],
    resilience_output_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Map each current-attempt export to its immutable clean-spec identity."""

    provenance: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        for path in collect_critic_exports([spec], resilience_output_dir):
            record = {
                "seed": int(spec.seed),
                "spec_tag": spec.tag(),
                "spec_artifact_key": spec_artifact_key(spec),
            }
            export = load_full_stability_export(path)
            if export.valid:
                episode_ids = {
                    str(item.get("episode_id") or "") for item in export.records
                }
                episode_ids.discard("")
                if len(episode_ids) == 1:
                    record["episode_id"] = next(iter(episode_ids))
                if export.metadata.get("judge_model") not in (None, ""):
                    record["judge_model"] = str(export.metadata["judge_model"])
            provenance[str(path.resolve())] = record
    return provenance


def update_episode_aggregate_from_clean_spec(
    *,
    resilience_output_dir: Path,
    resilience_cfg: Mapping[str, Any],
    clean_spec: PerturbationSpec,
    config: Optional[Mapping[str, Any]] = None,
    clean_specs_for_signature: Optional[Sequence[PerturbationSpec]] = None,
) -> Dict[str, Any]:
    """Incrementally update persistent per-episode baselines after one clean pass.

    A clean spec normally represents one full traversal of the configured
    ``episode_ids`` for a single seed. Updating here lets later clean seeds and
    perturbed specs consume the best available episode aggregate without waiting
    for the slower all-clean hash baseline.
    """
    stage_baseline_cfg = resilience_cfg.get("stage_baseline", {}) or {}
    if not bool(_episode_aggregate_cfg(stage_baseline_cfg).get("enabled", True)):
        return {}

    export_paths = collect_critic_exports([clean_spec], resilience_output_dir)
    if not export_paths:
        logger.warning(
            "No critic exports found for clean spec %s; skipping episode aggregate update.",
            clean_spec.tag(),
        )
        return {}

    c_rec_cfg = resilience_cfg.get("c_rec", {}) or {}
    formal_cfg = _stage_baseline_support_config(stage_baseline_cfg)
    formal_min_trajectories = int(
        formal_cfg.get(
            "min_independent_trajectories",
            DEFAULT_FORMAL_MIN_TRAJECTORIES,
        )
    )
    formal_min_transitions = int(
        formal_cfg.get("min_transitions", DEFAULT_FORMAL_MIN_TRANSITIONS)
    )
    estimator = StageBaselineEstimator(
        tau_pln=float(c_rec_cfg.get("tau_pln", 0.1)),
        tau_per=float(c_rec_cfg.get("tau_per", 0.1)),
        tau_coord=float(c_rec_cfg.get("tau_coord", 0.1)),
        formal_min_trajectories=formal_min_trajectories,
        formal_min_transitions=formal_min_transitions,
        require_successful_clean=bool(formal_cfg.get("require_successful_clean", False)),
        allow_family_covariance_shrinkage=bool(
            formal_cfg.get("allow_family_covariance_shrinkage", False)
        ),
    )
    signature = build_stage_baseline_signature(
        config=config,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs_for_signature or [clean_spec],
    )
    _baselines, episode_baselines, baseline_stats = (
        estimator.estimate_grouped_from_exports(
            export_paths,
            source_metadata=collect_critic_export_provenance(
                [clean_spec], resilience_output_dir
            ),
        )
    )
    if not episode_baselines:
        logger.warning(
            "No valid full-trajectory critic export for clean spec %s; "
            "skipping episode baseline update. Reasons: %s",
            clean_spec.tag(),
            baseline_stats.get("skipped_export_reasons", {}),
        )
        return {}
    manifest = update_episode_aggregated_baselines(
        estimator=estimator,
        episode_baselines=episode_baselines,
        episode_stats=baseline_stats,
        stage_baseline_cfg=stage_baseline_cfg,
        signature=signature,
        runtime_record_id=clean_spec.tag(),
    )
    if not manifest:
        return {}

    manifest.update(
        {
            "update_kind": "incremental_clean_spec",
            "clean_spec_tag": clean_spec.tag(),
            "source_export_count": len(export_paths),
            "source_export_paths": [str(path) for path in export_paths],
            "updated_episode_count": len(manifest.get("episode_baseline_paths") or {}),
            "updated_anchor_count": int(manifest.get("composed_anchor_count") or 0),
            "updated_at": _utc_now_iso(),
        }
    )
    logger.info(
        "Updated episode aggregate baseline from clean spec %s: %d export files, "
        "%d episode baselines -> %s",
        clean_spec.tag(),
        len(export_paths),
        len(episode_baselines),
        manifest.get("episode_aggregate_composed_path"),
    )
    return manifest


def _generate_cumulative_stage_baseline(
    *,
    base_results_dir: Path,
    resilience_output_dir: Path,
    resilience_cfg: Mapping[str, Any],
    clean_specs: Sequence[PerturbationSpec],
    config: Optional[Mapping[str, Any]],
) -> Optional[Path]:
    """Ingest clean evidence and refit one immutable snapshot from all evidence."""

    stage_baseline_cfg = resilience_cfg.get("stage_baseline", {}) or {}
    signature = build_stage_baseline_signature(
        config=config,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs,
    )
    store = resolve_stage_baseline_evidence_store(stage_baseline_cfg, signature)
    store.ensure_contract()
    prior_path = store.latest_snapshot()
    prior_baseline_id = ""
    if prior_path is not None:
        with open(prior_path, "r", encoding="utf-8") as handle:
            prior_baseline_id = str(
                (json.load(handle).get("metadata") or {}).get("baseline_id") or ""
            )

    current_exports = collect_critic_exports(clean_specs, resilience_output_dir)
    current_metadata = collect_critic_export_provenance(
        clean_specs, resilience_output_dir
    )
    for path in current_exports:
        resolved = str(path.resolve())
        metadata = current_metadata.setdefault(resolved, {})
        metadata.update(
            {
                "reward_shaper_valid": bool(
                    signature.get("reward_shaper_valid", False)
                ),
                "critic_lifecycle_valid": bool(
                    signature.get("critic_lifecycle_valid", False)
                ),
                "state_encoder_id": str(signature.get("state_encoder_id") or ""),
                "value_checkpoint_id": str(
                    signature.get("value_checkpoint_id") or ""
                ),
                "judge_model": str(signature.get("judge_model") or ""),
            }
        )
    ingest = store.ingest_exports(
        current_exports,
        source_metadata=current_metadata,
    )
    if not ingest["accepted_count"] and prior_path is not None:
        manifest = {
            "baseline_kind": "cumulative_evidence_snapshot_reused",
            "baseline_contract_id": store.contract_id,
            "stage_baseline_id": prior_baseline_id,
            "preferred_path": str(prior_path),
            "parent_baseline_id": prior_baseline_id,
            "ingest": ingest,
            "created_at": _utc_now_iso(),
        }
        write_json(
            resolve_baseline_output_path(base_results_dir, resilience_cfg).with_name(
                "stage_baseline_manifest.json"
            ),
            manifest,
        )
        return prior_path

    episode_ids = [str(value) for value in signature.get("episode_ids", [])]
    evidence_paths, evidence_metadata, evidence_records = store.estimator_inputs()
    if not evidence_paths:
        logger.warning(
            "No compatible successful clean evidence exists for StageBaseline "
            "contract %s.",
            store.contract_id,
        )
        return None

    c_rec_cfg = resilience_cfg.get("c_rec", {}) or {}
    formal_cfg = _stage_baseline_support_config(stage_baseline_cfg)
    formal_min_trajectories = int(
        formal_cfg.get(
            "min_independent_trajectories",
            DEFAULT_FORMAL_MIN_TRAJECTORIES,
        )
    )
    formal_min_transitions = int(
        formal_cfg.get("min_transitions", DEFAULT_FORMAL_MIN_TRANSITIONS)
    )
    estimator = StageBaselineEstimator(
        tau_pln=float(c_rec_cfg.get("tau_pln", 0.1)),
        tau_per=float(c_rec_cfg.get("tau_per", 0.1)),
        tau_coord=float(c_rec_cfg.get("tau_coord", 0.1)),
        formal_min_trajectories=formal_min_trajectories,
        formal_min_transitions=formal_min_transitions,
        require_successful_clean=bool(
            formal_cfg.get("require_successful_clean", False)
        ),
        allow_family_covariance_shrinkage=bool(
            formal_cfg.get("allow_family_covariance_shrinkage", False)
        ),
    )
    baselines, _episode_baselines, estimator_stats = (
        estimator.estimate_grouped_from_exports(
            evidence_paths,
            source_metadata=evidence_metadata,
        )
    )
    if not baselines:
        logger.warning(
            "Cumulative StageBaseline refit rejected all evidence for contract %s: %s",
            store.contract_id,
            estimator_stats.get("skipped_export_reasons", {}),
        )
        return prior_path

    lifecycle = resolve_stage_baseline_seed_lifecycle(resilience_cfg)
    initial_seeds = set(int(value) for value in lifecycle["calibration_seeds"])
    observed_pairs = {
        (str(record.episode_id), int(record.seed)) for record in evidence_records
    }
    expected_pairs = {
        (episode_id, seed)
        for episode_id in episode_ids
        for seed in initial_seeds
    }
    missing_pairs = sorted(expected_pairs - observed_pairs)
    episode_support: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"trajectories": 0, "transitions": 0}
    )
    for record in evidence_records:
        support = episode_support[str(record.episode_id)]
        support["trajectories"] += 1
        support["transitions"] += int(record.total_transitions)
    available_episode_ids = sorted(episode_support)
    calibration = {
        "baseline_available": bool(baselines),
        "expected_episode_ids": episode_ids,
        "initial_calibration_seeds": sorted(initial_seeds),
        "observed_seed_map": store.evidence_seed_map(episode_ids),
        "missing_initial_trajectories": [
            {"episode_id": episode_id, "seed": seed}
            for episode_id, seed in missing_pairs
        ],
        "evidence_count": len(evidence_records),
        "evidence_ids": sorted(record.evidence_id for record in evidence_records),
        "episode_support": {
            episode_id: dict(support)
            for episode_id, support in sorted(episode_support.items())
        },
        "available_episode_ids": available_episode_ids,
    }
    configured_output = resolve_baseline_output_path(base_results_dir, resilience_cfg)
    output_path = configured_output
    if output_path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = output_path.with_name(
            f"{output_path.stem}__snapshot-{timestamp}{output_path.suffix}"
        )
    metadata = {
        "baseline_kind": "cumulative_evidence_snapshot",
        "signature": signature,
        "baseline_contract_id": store.contract_id,
        "stage_baseline_parent_id": prior_baseline_id or None,
        "state_encoder_id": str(signature.get("state_encoder_id") or ""),
        "value_checkpoint_id": str(signature.get("value_checkpoint_id") or ""),
        "value_checkpoint_ids": sorted(
            {
                str(record.value_checkpoint_id)
                for record in evidence_records
                if record.value_checkpoint_id
            }
        ),
        "value_network_alignment_role": "diagnostic_only",
        "alignment_scope": STAGE_BASELINE_ALIGNMENT_SCOPE,
        "judge_model": str(signature.get("judge_model") or ""),
        "reward_shaper_valid": bool(signature.get("reward_shaper_valid", False)),
        "critic_lifecycle_valid": bool(
            signature.get("critic_lifecycle_valid", False)
        ),
        "estimator_stats": estimator_stats,
        "calibration": calibration,
        "baseline_available": True,
        "source_hashes": sorted(record.source_hash for record in evidence_records),
        "evidence_ids": calibration["evidence_ids"],
        "baseline_entry_count": len(baselines),
        "total_samples": _baseline_total_samples(baselines),
        "anchors": _anchor_records(baselines),
        "successful_episode_ids": available_episode_ids,
        "available_episode_ids": available_episode_ids,
        "excluded_evidence": list(ingest.get("excluded", [])),
        "conflicting_evidence": list(ingest.get("conflicts", [])),
        "created_at": _utc_now_iso(),
    }
    estimator.write(baselines, output_path, metadata=metadata)
    with open(output_path, "r", encoding="utf-8") as handle:
        written_metadata = dict(json.load(handle).get("metadata") or {})
    baseline_id = str(written_metadata.get("baseline_id") or "")
    metadata["baseline_available"] = True
    snapshot_path = store.publish_snapshot(
        output_path,
        baseline_id=baseline_id,
        metadata={
            **metadata,
            "ingest": ingest,
            "run_local_path": str(output_path),
        },
        update_latest=True,
    )
    manifest = {
        **metadata,
        "stage_baseline_id": baseline_id,
        "preferred_path": str(snapshot_path),
        "run_local_path": str(output_path),
        "ingest": ingest,
        "source_export_paths": [str(path) for path in current_exports],
    }
    write_json(output_path.with_name("stage_baseline_manifest.json"), manifest)
    logger.info(
        "Materialized cumulative StageBaseline %s from %d evidence trajectories "
        "(%d newly accepted) -> %s",
        baseline_id,
        len(evidence_records),
        int(ingest["accepted_count"]),
        snapshot_path,
    )
    return snapshot_path


def generate_stage_baseline(
    *,
    base_results_dir: Path,
    resilience_output_dir: Path,
    resilience_cfg: Mapping[str, Any],
    clean_specs: Sequence[PerturbationSpec],
    config: Optional[Mapping[str, Any]] = None,
    update_episode_aggregate: bool = True,
) -> Optional[Path]:
    stage_baseline_cfg = resilience_cfg.get("stage_baseline", {}) or {}
    if bool(stage_baseline_cfg.get("evidence_store_enabled", True)):
        return _generate_cumulative_stage_baseline(
            base_results_dir=base_results_dir,
            resilience_output_dir=resilience_output_dir,
            resilience_cfg=resilience_cfg,
            clean_specs=clean_specs,
            config=config,
        )
    if not clean_specs:
        logger.warning(
            "Stage baseline generation requested but no clean specs were run."
        )
        return None

    export_paths = collect_critic_exports(clean_specs, resilience_output_dir)
    if not export_paths:
        logger.warning(
            "No critic exports found under clean specs; skipping stage baseline generation."
        )
        return None

    c_rec_cfg = resilience_cfg.get("c_rec", {}) or {}
    formal_cfg = _stage_baseline_support_config(stage_baseline_cfg)
    estimator = StageBaselineEstimator(
        tau_pln=float(c_rec_cfg.get("tau_pln", 0.1)),
        tau_per=float(c_rec_cfg.get("tau_per", 0.1)),
        tau_coord=float(c_rec_cfg.get("tau_coord", 0.1)),
        formal_min_trajectories=int(
            formal_cfg.get("min_independent_trajectories", DEFAULT_FORMAL_MIN_TRAJECTORIES)
        ),
        formal_min_transitions=int(
            formal_cfg.get("min_transitions", DEFAULT_FORMAL_MIN_TRANSITIONS)
        ),
        require_successful_clean=bool(formal_cfg.get("require_successful_clean", False)),
        allow_family_covariance_shrinkage=bool(
            formal_cfg.get("allow_family_covariance_shrinkage", False)
        ),
    )
    signature = build_stage_baseline_signature(
        config=config,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs,
    )
    baselines, episode_baselines, baseline_stats = estimator.estimate_grouped_from_exports(
        export_paths,
        source_metadata=collect_critic_export_provenance(
            clean_specs, resilience_output_dir
        ),
    )
    if not baselines:
        logger.warning(
            "No StageBaseline was generated because every clean critic export "
            "failed the full-trajectory contract or had no usable trajectory. "
            "Reasons: %s",
            baseline_stats.get("skipped_export_reasons", {}),
        )
        return None
    calibration = _calibration_completeness(
        signature=signature,
        baselines=baselines,
        estimator_stats=baseline_stats,
    )
    common_metadata = {
        "signature": signature,
        "estimator_stats": baseline_stats,
        "calibration": calibration,
        "formal_valid": bool(calibration["formal_valid"]),
        "formal_missing_reasons": list(calibration["formal_missing_reasons"]),
        "source_hashes": list(calibration["source_hashes"]),
    }
    configured_output_path = resolve_baseline_output_path(
        base_results_dir, resilience_cfg
    )
    output_path = configured_output_path
    if output_path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = output_path.with_name(
            f"{output_path.stem}__calibration-{timestamp}{output_path.suffix}"
        )
        logger.warning(
            "Preserving immutable run-local StageBaseline %s; writing the "
            "new calibration to %s",
            configured_output_path,
            output_path,
        )
    estimator.write(
        baselines,
        output_path,
        metadata={
            "baseline_kind": "hash_run_local",
            **common_metadata,
        },
    )

    with open(output_path, "r", encoding="utf-8") as handle:
        current_baseline_id = str(
            (json.load(handle).get("metadata") or {}).get("baseline_id") or ""
        )

    persistent_path: Optional[Path] = None
    latest_alias_path: Optional[Path] = None
    preferred_path = output_path
    persistence_status: Dict[str, Any] = {
        "canonical_path": None,
        "written_path": None,
        "reused": False,
        "conflict": False,
        "formal_calibration": bool(calibration["formal_valid"]),
    }
    if bool(stage_baseline_cfg.get("write_persistent_copy", True)):
        canonical_persistent_path = resolve_persistent_hash_baseline_path(
            stage_baseline_cfg,
            signature,
        )
        persistence_status["canonical_path"] = str(canonical_persistent_path)
        if not canonical_persistent_path.exists():
            persistent_path = canonical_persistent_path
        else:
            with open(canonical_persistent_path, "r", encoding="utf-8") as handle:
                existing_baseline_id = str(
                    (json.load(handle).get("metadata") or {}).get("baseline_id")
                    or ""
                )
            if existing_baseline_id == current_baseline_id:
                persistent_path = canonical_persistent_path
                persistence_status["reused"] = True
            else:
                persistence_status["conflict"] = True
                persistent_path = canonical_persistent_path.with_name(
                    f"{canonical_persistent_path.stem}__baseline-"
                    f"{current_baseline_id[:12]}{canonical_persistent_path.suffix}"
                )
                logger.warning(
                    "Preserving immutable persistent StageBaseline %s; "
                    "writing the distinct calibration to %s",
                    canonical_persistent_path,
                    persistent_path,
                )
        if not persistence_status["reused"]:
            if persistent_path.exists():
                with open(persistent_path, "r", encoding="utf-8") as handle:
                    stored_baseline_id = str(
                        (json.load(handle).get("metadata") or {}).get("baseline_id")
                        or ""
                    )
                if stored_baseline_id != current_baseline_id:
                    timestamp = datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%S%fZ"
                    )
                    persistent_path = persistent_path.with_name(
                        f"{persistent_path.stem}__{timestamp}{persistent_path.suffix}"
                    )
            if not persistent_path.exists():
                estimator.write(
                    baselines,
                    persistent_path,
                    metadata={
                        "baseline_kind": "hash_persistent",
                        **common_metadata,
                    },
                )
        persistence_status["written_path"] = str(persistent_path)
        latest_alias = str(
            stage_baseline_cfg.get("latest_alias", "stage_baseline_latest.json")
        ).strip()
        if latest_alias:
            latest_alias_path = (
                resolve_statistics_baseline_root(stage_baseline_cfg, signature)
                / latest_alias
            )
            estimator.write(
                baselines,
                latest_alias_path,
                metadata={
                    "baseline_kind": "latest_alias",
                    **common_metadata,
                },
            )

    episode_aggregate_manifest: Dict[str, Any] = {}
    if update_episode_aggregate:
        episode_aggregate_manifest = update_episode_aggregated_baselines(
            estimator=estimator,
            episode_baselines=episode_baselines,
            episode_stats=baseline_stats,
            stage_baseline_cfg=stage_baseline_cfg,
            signature=signature,
            runtime_record_id=f"clean_batch_{signature.get('signature', '')}",
        )
    else:
        episode_aggregate_manifest = {
            "update_skipped": True,
            "skip_reason": "incremental_clean_spec_updates_already_applied",
        }

    manifest = build_stage_baseline_manifest(
        baseline_path=output_path,
        export_paths=export_paths,
        baselines=baselines,
        clean_specs=clean_specs,
        signature=signature,
        persistent_path=persistent_path,
        latest_alias_path=latest_alias_path,
        preferred_path=preferred_path,
    )
    manifest["calibration"] = calibration
    manifest["formal_valid"] = bool(calibration["formal_valid"])
    manifest["formal_missing_reasons"] = list(
        calibration["formal_missing_reasons"]
    )
    manifest["persistence"] = persistence_status
    if episode_aggregate_manifest:
        manifest["episode_aggregate"] = episode_aggregate_manifest
    manifest_path = output_path.with_name("stage_baseline_manifest.json")
    write_json(manifest_path, manifest)
    if persistent_path is not None:
        write_json(persistent_path.with_suffix(".manifest.json"), manifest)

    logger.info(
        "Generated stage baseline from %d export files (%d anchor groups, %d entries) -> %s",
        len(export_paths),
        manifest["anchor_group_count"],
        len(baselines),
        preferred_path,
    )
    return preferred_path


def teardown_between_specs() -> None:
    """Best-effort GPU/memory teardown between spec rollouts."""
    try:
        import gc
        gc.collect()
        # import torch  # type: ignore
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     try:
        #         torch.cuda.ipc_collect()
        #     except Exception:
        #         pass
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# Subprocess entry point for rollout_one_spec isolation
# ──────────────────────────────────────────────────────────────────────
def _subprocess_rollout_spec_entry() -> None:
    """Child-process entry point invoked by :func:`rollout_one_spec`.

    Usage::

        python -m habitat_llm.examples.resilience_helpers \
            --spec-config /path/to/_resolved_config.yaml

    The function loads the serialized config, creates the dataset, and runs
    ``planner_demo_mp_new.run_planner`` exactly once.  When this process
    exits, the OS reclaims *all* native memory.
    """
    import argparse
    import os
    import sys

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(
        description="Subprocess worker for resilience spec rollout"
    )
    parser.add_argument(
        "--spec-config",
        required=True,
        help="Path to the resolved YAML config for this spec.",
    )
    parser.add_argument(
        "--resource-usage",
        required=False,
        help="Path for worker resource-usage JSON.",
    )
    args = parser.parse_args()

    config_path = Path(args.spec_config)
    if not config_path.exists():
        print(f"[resilience-worker] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    import platform
    import time
    import traceback

    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    exit_code = 0
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    cfg: Optional[DictConfig] = None
    episode_ids: List[str] = []
    try:
        cfg = OmegaConf.load(config_path)
        OmegaConf.set_struct(cfg, False)
        cfg.num_proc = 1
        # A resilience worker represents an exact experimental cell.  It must
        # surface the first episode failure instead of continuing to an empty
        # aggregate that hides the original exception.
        cfg.fail_on_episode_error = True

        from habitat_llm.agent.env.dataset import CollaborationDatasetV0
        from habitat_llm.examples import planner_demo_mp_new as base_runner

        dataset = CollaborationDatasetV0(cfg.habitat.dataset)
        if (
            cfg.get("episode_indices", None) is not None
            or cfg.get("episode_ids", None) is not None
            or cfg.get("evolve_episode_ids", None) is not None
        ):
            dataset = base_runner._select_episode_subset(dataset, cfg)
        episode_ids = [str(episode.episode_id) for episode in dataset.episodes]

        logger.info(
            "[resilience-worker] Starting run_planner with %d episodes -> %s",
            len(dataset.episodes),
            cfg.paths.results_dir,
        )
        base_runner.run_planner(cfg, dataset)
        metrics_path = Path(str(cfg.paths.epi_result_file_path))
        if not metrics_path.is_file():
            raise RuntimeError(f"episode metrics were not written: {metrics_path}")
        written_rows = list(read_csv_rows(metrics_path))
        expected_counts = Counter(episode_ids)
        written_counts = Counter(
            str(row.get("episode_id", "")) for row in written_rows
        )
        missing_episode_ids = sorted(
            episode_id
            for episode_id, expected_count in expected_counts.items()
            if written_counts[episode_id] < expected_count
        )
        duplicate_episode_ids = sorted(
            episode_id
            for episode_id, count in written_counts.items()
            if episode_id in expected_counts
            and count > expected_counts[episode_id]
        )
        unexpected_episode_ids = sorted(
            episode_id
            for episode_id in written_counts
            if episode_id not in expected_counts
        )
        if missing_episode_ids or duplicate_episode_ids or unexpected_episode_ids:
            raise RuntimeError(
                "episode metrics do not match the requested rollout design; "
                f"missing={missing_episode_ids}, "
                f"duplicate={duplicate_episode_ids}, "
                f"unexpected={unexpected_episode_ids}"
            )
        logger.info("[resilience-worker] run_planner completed.")
    except Exception as exc:
        exit_code = 1
        error = f"{type(exc).__name__}: {exc}"
        error_traceback = traceback.format_exc()
        logger.exception("[resilience-worker] rollout failed")
    finally:
        peak_rss_gb: Optional[float] = None
        try:
            import resource

            peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if platform.system() == "Darwin":
                peak_rss_gb = peak_rss / (1024.0**3)
            else:
                peak_rss_gb = peak_rss / (1024.0**2)
        except (ImportError, OSError, ValueError):
            pass

        runtime_perturbation: Mapping[str, Any] = {}
        if cfg is not None:
            raw_spec = OmegaConf.select(cfg, "evaluation.runtime_perturbation")
            if raw_spec is not None:
                converted = OmegaConf.to_container(raw_spec, resolve=True)
                if isinstance(converted, Mapping):
                    runtime_perturbation = converted
        usage_path = (
            Path(args.resource_usage)
            if args.resource_usage
            else config_path.parent / "resource_usage.json"
        )
        write_json(
            usage_path,
            {
                "pid": os.getpid(),
                "episode_ids": episode_ids,
                "spec": dict(runtime_perturbation),
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.monotonic() - started_monotonic,
                "exit_code": exit_code,
                "peak_rss_gb": peak_rss_gb,
                "error": error,
                "traceback": error_traceback,
            },
        )

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    _subprocess_rollout_spec_entry()
