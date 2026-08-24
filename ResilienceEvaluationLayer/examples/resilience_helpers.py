#!/usr/bin/env python3
# isort: skip_file
"""
Utility functions for :mod:`planner_demo_resilience`.

Extracted to keep the main entry-point script minimal and focused on the
high-level experiment orchestration logic.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from habitat_llm.evaluation.metrics.extensibility import MarginCollector, MarginThresholds
from habitat_llm.evaluation.metrics.rebound import StageBaselineEstimator
from habitat_llm.evaluation.metrics.stability import BetaAggregate, StabilityCollector
from habitat_llm.evaluation.perturbation import PerturbationInjector, PerturbationSpec

logger = logging.getLogger(__name__)
_EPISODE_FAMILY_CACHE: Dict[str, Dict[str, str]] = {}


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
    specs: List[PerturbationSpec] = []
    for kind in cfg["perturbations"]:
        for seed in cfg["seeds"]:
            for intensity in cfg["intensities"]:
                specs.append(
                    PerturbationSpec(
                        kind=str(kind),
                        seed=int(seed),
                        intensity=float(intensity),
                    )
                )
    return specs


def spec_output_dir(
    resilience_output_dir: Path,
    spec: PerturbationSpec,
) -> Path:
    return (
        resilience_output_dir
        / f"spec_{spec.kind}_seed{spec.seed}_lambda{spec.intensity:.2f}"
    )


def partition_specs(
    specs: Sequence[PerturbationSpec],
    clean_tag: str,
) -> Tuple[List[PerturbationSpec], List[PerturbationSpec]]:
    clean_specs: List[PerturbationSpec] = []
    other_specs: List[PerturbationSpec] = []
    for spec in specs:
        if str(spec.kind) == str(clean_tag):
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
    episode_ids = (
        _normalize_sequence(cfg_dict.get("episode_ids"))
        or _normalize_sequence(cfg_dict.get("evolve_episode_ids"))
        or _normalize_sequence(resilience_cfg.get("anchors"))
    )
    payload = {
        "dataset_path": dataset_path,
        "episode_ids": episode_ids,
        "clean_specs": [spec.tag() for spec in clean_specs],
        "seeds": [int(seed) for seed in resilience_cfg.get("seeds", [])],
        "intensities": [float(val) for val in resilience_cfg.get("intensities", [])],
        "baseline_json": str(
            resilience_cfg.get("baseline_json", "stage_baseline.json")
        ),
        "output_subdir": str(resilience_cfg.get("output_subdir", "resilience")),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:12]
    dataset_slug = _slugify(Path(dataset_path).stem or "dataset", default="dataset")
    if episode_ids:
        head = "_".join(_slugify(ep, default="ep") for ep in episode_ids[:6])
        suffix = "-etc" if len(episode_ids) > 6 else ""
        episode_slug = f"eps-{head}{suffix}"
    else:
        episode_slug = "eps-all"
    seed_values = [str(seed) for seed in resilience_cfg.get("seeds", [])]
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
) -> Path:
    cfg = _episode_aggregate_cfg(stage_baseline_cfg)
    subdir = str(cfg.get("subdir", "episode_aggregates")).strip() or "episode_aggregates"
    return resolve_persistent_baseline_dir(stage_baseline_cfg) / subdir


def _signature_dataset_slug(signature: Mapping[str, Any]) -> str:
    dataset_path = str(signature.get("dataset_path") or "dataset")
    return _slugify(Path(dataset_path).stem or "dataset", default="dataset")


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
    root = resolve_episode_aggregate_root(stage_baseline_cfg)
    dataset_slug = _signature_dataset_slug(signature)
    ep_slug = _slugify(episode_id, default="unknown")
    return root / dataset_slug / f"episode_{ep_slug}" / "stage_baseline.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    return resolve_persistent_baseline_dir(stage_baseline_cfg) / str(
        signature["archive_filename"]
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

    allow_partial = bool(cfg.get("allow_partial_compose", True))
    combined: Dict[Tuple[str, int, str], Any] = {}
    used_paths: List[str] = []
    used_episode_records: List[Dict[str, Any]] = []
    missing_ids: List[str] = []
    for ep_id in ids:
        ep_path = _episode_baseline_path(stage_baseline_cfg, signature, ep_id)
        if not ep_path.exists():
            missing_ids.append(ep_id)
            continue
        ep_baselines = StageBaselineEstimator.load(ep_path)
        if not ep_baselines:
            missing_ids.append(ep_id)
            continue
        combined = StageBaselineEstimator.merge_baseline_maps(combined, ep_baselines)
        used_paths.append(str(ep_path))
        used_episode_records.append(
            {
                "episode_id": str(ep_id),
                "path": str(ep_path),
                "baseline_entry_count": len(ep_baselines),
                "total_samples": _baseline_total_samples(ep_baselines),
            }
        )

    if not combined or (missing_ids and not allow_partial):
        return None

    root = resolve_episode_aggregate_root(stage_baseline_cfg)
    dataset_slug = _signature_dataset_slug(signature)
    output_path = (
        root
        / dataset_slug
        / "composed"
        / f"stage_baseline__{_episode_set_slug(ids)}__episode_aggregate.json"
    )
    metadata = {
        "baseline_kind": "episode_aggregate_composed",
        "dataset_path": str(signature.get("dataset_path") or ""),
        "episode_ids": ids,
        "used_episode_baseline_paths": used_paths,
        "used_episode_baselines": used_episode_records,
        "used_episode_baseline_count": len(used_episode_records),
        "missing_episode_ids": missing_ids,
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
        "persistent_hash_path": str(persistent_hash_path),
        "persistent_hash_exists": bool(persistent_hash_path.exists()),
        "episode_aggregate_enabled": bool(
            _episode_aggregate_cfg(stage_baseline_cfg).get("enabled", True)
        ),
        "source": "missing",
        "selected_path": None,
        "fallback_reason": "",
    }

    if baseline_override_path is not None:
        result.update(
            {
                "source": "explicit_override",
                "selected_path": str(baseline_override_path),
                "selected_path_exists": bool(Path(baseline_override_path).exists()),
            }
        )
        return result

    if run_local_path.exists():
        result.update(
            {
                "source": "run_local_hash",
                "selected_path": str(run_local_path),
                "selected_path_exists": True,
            }
        )
        return result

    if persistent_hash_path.exists():
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
                "fallback_reason": "hash_baseline_missing_episode_aggregate_unavailable",
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
) -> Dict[str, Any]:
    cfg = _episode_aggregate_cfg(stage_baseline_cfg)
    if not bool(cfg.get("enabled", True)):
        return {}

    merge_existing = bool(cfg.get("merge_existing", True))
    written_paths: Dict[str, str] = {}
    per_episode_anchor_updates: Dict[str, Any] = {}
    for ep_id, incoming in episode_baselines.items():
        output_path = _episode_baseline_path(stage_baseline_cfg, signature, ep_id)
        existing = (
            StageBaselineEstimator.load(output_path)
            if merge_existing and output_path.exists()
            else {}
        )
        merged = StageBaselineEstimator.merge_baseline_maps(existing, incoming)
        existing_entry_count = len(existing)
        incoming_entry_count = len(incoming)
        anchor_update = _anchor_update_summary(existing, incoming, merged)
        metadata = {
            "baseline_kind": "episode_aggregate",
            "dataset_path": str(signature.get("dataset_path") or ""),
            "episode_id": str(ep_id),
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
        StageBaselineEstimator.load(composed_path)
        if composed_path is not None and composed_path.exists()
        else {}
    )
    return {
        "episode_aggregate_root": str(resolve_episode_aggregate_root(stage_baseline_cfg)),
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
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return output_path
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_path


def inject_baseline_path(cfg: DictConfig, baseline_path: Path) -> None:
    if cfg.get("evaluation") is None:
        cfg.evaluation = OmegaConf.create({})
    if cfg.evaluation.get("rebound_tracker") is None:
        cfg.evaluation.rebound_tracker = OmegaConf.create({})
    cfg.evaluation.rebound_tracker.baseline_path = str(baseline_path)


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
) -> Path:
    """Run ``planner_demo_mp_new.run_planner`` once for one perturbation spec.

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
    spec_dir = spec_output_dir(resilience_output_dir, spec)
    spec_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    try:
        cfg.paths.epi_result_file_path = str(spec_dir / "episode_metrics.csv")
        cfg.paths.run_result_file_path = str(spec_dir / "run_result_log.csv")
        cfg.paths.end_result_file_path = str(spec_dir / "end_result_log.csv")
        cfg.paths.results_dir = str(spec_dir)
        if baseline_path is not None:
            inject_baseline_path(cfg, baseline_path)
    except Exception as exc:
        logger.warning("Unable to redirect paths for spec %s: %s", spec.tag(), exc)

    if cfg.get("evaluation") is None:
        cfg.evaluation = OmegaConf.create({})
    cfg.evaluation.runtime_perturbation = OmegaConf.create(
        {
            "kind": spec.kind,
            "seed": int(spec.seed),
            "intensity": float(spec.intensity),
        }
    )
    cfg.evaluation.runtime_perturbation_seed = int(spec.seed)

    fix_config(cfg)
    cfg = setup_config(cfg, seed)

    # Serialize the fully-resolved config so the subprocess can load it
    # without relying on Hydra argument reconstruction.
    spec_config_path = spec_dir / "_resolved_config.yaml"
    OmegaConf.set_struct(cfg, False)
    with open(spec_config_path, "w", encoding="utf-8") as handle:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))
    OmegaConf.set_struct(cfg, True)

    csv_output_path = Path(str(cfg.paths.epi_result_file_path))

    logger.info(
        "[resilience] Launching subprocess for spec %s -> %s",
        spec.tag(),
        spec_dir,
    )

    # Launch a child process that loads the config and runs run_planner.
    # Using the module entry point avoids Hydra re-initialization issues.
    child_cmd = [
        sys.executable,
        "-m",
        "habitat_llm.examples.resilience_helpers",
        "--spec-config",
        str(spec_config_path),
    ]
    result = subprocess.run(child_cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess for spec {spec.tag()} exited with code "
            f"{result.returncode}.  Check logs in {spec_dir}"
        )

    if not csv_output_path.exists():
        logger.warning(
            "[resilience] Subprocess finished but CSV not found: %s",
            csv_output_path,
        )

    return csv_output_path


# ──────────────────────────────────────────────────────────────────────
# Post-processing: merge, aggregate, baseline generation
# ──────────────────────────────────────────────────────────────────────
def merge_spec_csvs(
    specs: Sequence[PerturbationSpec],
    per_spec_csvs: Sequence[Path],
    output_path: Path,
) -> List[Dict[str, Any]]:
    merged_rows: List[Dict[str, Any]] = []
    for spec, csv_path in zip(specs, per_spec_csvs):
        if not csv_path.exists():
            logger.warning("Per-spec CSV missing: %s", csv_path)
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["perturbation_type"] = spec.kind
                row["perturbation_seed"] = spec.seed
                row["perturbation_intensity"] = spec.intensity
                row.setdefault("anchor_id", row.get("episode_id", "unknown"))
                merged_rows.append(row)
    if merged_rows:
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
) -> List[Dict[str, Any]]:
    family_map = _load_episode_family_map(dataset_path)
    normalized: List[Dict[str, Any]] = []

    for row in raw_rows:
        materialized = row if isinstance(row, dict) else dict(row)
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

        materialized["task_percent_complete"] = _safe_float(
            materialized.get("task_percent_complete"),
            0.0,
        )
        materialized["task_state_success"] = _safe_float(
            materialized.get("task_state_success"),
            0.0,
        )
        total_step_count = materialized.get("total_step_count")
        if total_step_count in (None, ""):
            total_step_count = materialized.get("sim_step_count")
        materialized["total_step_count"] = _safe_int(total_step_count, 0)

        q_value = materialized.get("proposition_satisfied_fraction")
        if q_value in (None, ""):
            # Older episode CSVs did not always preserve the proposition
            # fraction explicitly. Fall back to the already-exported completion
            # ratio instead of collapsing the Q_f contract term to zero.
            materialized["proposition_satisfied_fraction"] = _safe_float(
                materialized.get("task_percent_complete"),
                0.0,
            )
            materialized.setdefault(
                "proposition_satisfied_fraction_source",
                "task_percent_complete_fallback",
            )
        else:
            materialized["proposition_satisfied_fraction"] = _safe_float(
                q_value,
                0.0,
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
            if "rebound_c_rec_valid" not in materialized:
                baseline_loaded = _optional_float(materialized.get("rebound_baseline_loaded"))
                if baseline_loaded is not None and baseline_loaded < 1.0:
                    materialized["rebound_c_rec_valid"] = 0
                    materialized.setdefault(
                        "rebound_c_rec_missing_reason",
                        "stage_baseline_missing",
                    )
                else:
                    materialized["rebound_c_rec_valid"] = 1

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

        normalized_cbf_penalty_rate = _optional_float(
            materialized.get("normalized_cbf_penalty_rate")
        )
        if normalized_cbf_penalty_rate is None:
            if p_cbf is not None and materialized["total_step_count"] > 0:
                normalized_cbf_penalty_rate = float(
                    p_cbf / float(materialized["total_step_count"])
                )
        if normalized_cbf_penalty_rate is None:
            materialized.setdefault("normalized_cbf_penalty_rate", "")
        else:
            materialized["normalized_cbf_penalty_rate"] = normalized_cbf_penalty_rate

        normalized.append(materialized)

    return normalized


def augment_rows_with_beta_aggregates(
    raw_rows: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, BetaAggregate],
    perturbation_aggregates: Optional[Mapping[Tuple[str, str], BetaAggregate]] = None,
) -> None:
    for row in raw_rows:
        anchor_id = str(row.get("anchor_id", row.get("episode_id", "unknown")))
        aggregate = aggregates.get(anchor_id)
        if aggregate is None:
            pass
        else:
            row["stability_beta_neighborhood"] = float(aggregate.beta_hat)
            row["stability_beta_vv"] = float(aggregate.beta_vv)
            row["stability_beta_out"] = float(aggregate.beta_out)
            row["stability_beta_oscillation"] = float(aggregate.beta_oscillation)
            row["stability_beta_action"] = float(aggregate.beta_action)
            row["stability_beta_progress"] = float(aggregate.beta_progress)
            row["stability_beta_success"] = float(aggregate.beta_success)
            row["stability_clean_ref_valid"] = int(aggregate.clean_ref_valid)
            row["stability_clean_ref_strategy"] = aggregate.clean_ref_strategy
            row["stability_n_clean_ref"] = int(aggregate.n_clean)
            row["stability_beta_neighborhood_valid"] = int(
                aggregate.beta_neighborhood_valid
            )
            row["stability_beta_neighborhood_missing_reason"] = aggregate.missing_reason
            row["stability_beta_scope"] = aggregate.beta_scope
            row.setdefault("task_family", aggregate.task_family)

        perturbation_type = str(row.get("perturbation_type", "") or "").strip().lower()
        is_clean = perturbation_type in ("", "clean", "baseline")
        perturbation_aggregate = (
            None
            if perturbation_aggregates is None
            else perturbation_aggregates.get((anchor_id, perturbation_type))
        )
        if is_clean:
            row["stability_beta_perturbation"] = 0.0
            row["stability_beta_perturbation_vv"] = 0.0
            row["stability_beta_perturbation_out"] = 0.0
            row["stability_beta_perturbation_oscillation"] = 0.0
            row["stability_beta_perturbation_action"] = 0.0
            row["stability_beta_perturbation_progress"] = 0.0
            row["stability_beta_perturbation_success"] = 0.0
            row["stability_beta_perturbation_valid"] = 1
            row["stability_beta_perturbation_scope"] = "clean_reference"
        elif perturbation_aggregate is not None:
            row["stability_beta_perturbation"] = float(perturbation_aggregate.beta_hat)
            row["stability_beta_perturbation_vv"] = float(perturbation_aggregate.beta_vv)
            row["stability_beta_perturbation_out"] = float(perturbation_aggregate.beta_out)
            row["stability_beta_perturbation_oscillation"] = float(
                perturbation_aggregate.beta_oscillation
            )
            row["stability_beta_perturbation_action"] = float(
                perturbation_aggregate.beta_action
            )
            row["stability_beta_perturbation_progress"] = float(
                perturbation_aggregate.beta_progress
            )
            row["stability_beta_perturbation_success"] = float(
                perturbation_aggregate.beta_success
            )
            row["stability_beta_perturbation_valid"] = int(
                perturbation_aggregate.beta_neighborhood_valid
            )
            row["stability_beta_perturbation_scope"] = perturbation_aggregate.beta_scope


def run_stability_aggregation(
    raw_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    cfg: Mapping[str, Any],
    *,
    raw_output_path: Optional[Path] = None,
) -> Dict[str, BetaAggregate]:
    collector = StabilityCollector(config=dict(cfg.get("stability", {}) or {}))
    beta_cfg = cfg.get("beta", {}) or {}
    aggregates, stats_rows = collector.aggregate_beta_over_anchors(
        raw_rows,
        alpha=float(beta_cfg.get("alpha", cfg.get("alpha", 0.5))),
        weight_vv=float(
            beta_cfg.get("weight_vv", cfg.get("beta_weight_vv", 0.5))
        ),
        weight_out=float(
            beta_cfg.get("weight_out", cfg.get("beta_weight_out", 0.5))
        ),
        include_rebound=bool(
            beta_cfg.get(
                "include_rebound",
                cfg.get("include_rebound_in_stability", False),
            )
        ),
    )
    perturbation_aggregates, perturbation_rows = collector.aggregate_beta_by_perturbation(
        raw_rows,
        alpha=float(beta_cfg.get("alpha", cfg.get("alpha", 0.5))),
        weight_vv=float(
            beta_cfg.get("weight_vv", cfg.get("beta_weight_vv", 0.5))
        ),
        weight_out=float(
            beta_cfg.get("weight_out", cfg.get("beta_weight_out", 0.5))
        ),
        include_rebound=bool(
            beta_cfg.get(
                "include_rebound",
                cfg.get("include_rebound_in_stability", False),
            )
        ),
    )
    if aggregates:
        collector.write_beta_neighborhood_csv(
            aggregates, output_dir / "resilience_beta.csv"
        )
        augment_rows_with_beta_aggregates(
            raw_rows, aggregates, perturbation_aggregates
        )
    if perturbation_aggregates:
        collector.write_beta_neighborhood_csv(
            perturbation_aggregates,
            output_dir / "resilience_beta_by_perturbation.csv",
        )
    tests = collector.pairwise_family_tests(stats_rows, metric_key="beta_hat")
    if tests:
        with open(
            output_dir / "resilience_beta_family_tests.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(tests, handle, indent=2)
    if raw_output_path is not None and raw_rows:
        write_csv_rows(raw_output_path, raw_rows)
    return aggregates


def run_margin_aggregation(
    raw_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    cfg: Mapping[str, Any],
    *,
    dataset_path: Optional[str] = None,
    raw_output_path: Optional[Path] = None,
) -> Dict[Tuple[str, float, int], Any]:
    normalized_rows = normalize_rollout_rows(
        raw_rows,
        clean_tag=str((cfg.get("stage_baseline", {}) or {}).get("clean_tag", "clean")),
        dataset_path=dataset_path,
    )
    thresholds = MarginThresholds.from_mapping(
        dict(cfg.get("contract_margin", {}) or {})
    )
    collector = MarginCollector(thresholds=thresholds)
    margins = collector.aggregate(normalized_rows)
    if not margins:
        if collector.episode_evidence:
            valid_rows = [row.to_dict() for row in collector.episode_evidence if row.valid]
            diagnostic_rows = [
                row.to_diagnostic_dict() for row in collector.episode_evidence
            ]
            invalid_rows = list(collector.invalid_rows)
            if valid_rows:
                write_csv_rows(
                    output_dir / "resilience_margin_episode_contracts.csv",
                    valid_rows,
                )
            if diagnostic_rows:
                write_csv_rows(
                    output_dir / "resilience_margin_episode_diagnostics.csv",
                    diagnostic_rows,
                )
            if invalid_rows:
                write_csv_rows(
                    output_dir / "resilience_margin_invalid_rows.csv",
                    invalid_rows,
                )
        if raw_output_path is not None and normalized_rows:
            write_csv_rows(raw_output_path, normalized_rows)
        return {}

    audc_map = collector.finalize_audc(margins)
    episode_rows = [row.to_dict() for row in collector.episode_evidence if row.valid]
    episode_diagnostic_rows = [
        row.to_diagnostic_dict() for row in collector.episode_evidence
    ]
    invalid_rows = list(collector.invalid_rows)
    if episode_rows:
        write_csv_rows(
            output_dir / "resilience_margin_episode_contracts.csv",
            episode_rows,
        )
    if episode_diagnostic_rows:
        write_csv_rows(
            output_dir / "resilience_margin_episode_diagnostics.csv",
            episode_diagnostic_rows,
        )
    if invalid_rows:
        write_csv_rows(
            output_dir / "resilience_margin_invalid_rows.csv",
            invalid_rows,
        )
    MarginCollector.write_csv(margins, output_dir / "resilience_margin_cells.csv")
    MarginCollector.write_diagnostics_csv(
        margins,
        output_dir / "resilience_margin_cell_diagnostics.csv",
    )
    if collector.capacity_rows:
        write_csv_rows(
            output_dir / "resilience_margin_capacity.csv",
            collector.capacity_rows,
        )
    if collector.capacity_diagnostic_rows:
        write_csv_rows(
            output_dir / "resilience_margin_capacity_diagnostics.csv",
            collector.capacity_diagnostic_rows,
        )

    audc_rows = [
        {
            "task_family": family,
            "evolve_step": evolve_step,
            "ge_audc_f": float(audc),
            "ge_audc_norm_f": next(
                (
                    row.get("ge_audc_norm_f", 0.0)
                    for row in collector.capacity_rows
                    if str(row.get("task_family")) == str(family)
                    and int(row.get("evolve_step", 0) or 0) == int(evolve_step)
                ),
                0.0,
            ),
            "ge_boundary_lambda_star": next(
                (
                    row.get("ge_boundary_lambda_star", "")
                    for row in collector.capacity_rows
                    if str(row.get("task_family")) == str(family)
                    and int(row.get("evolve_step", 0) or 0) == int(evolve_step)
                ),
                "",
            ),
            "ge_cpsc_lambda_star": next(
                (
                    row.get("ge_cpsc_lambda_star", "")
                    for row in collector.capacity_rows
                    if str(row.get("task_family")) == str(family)
                    and int(row.get("evolve_step", 0) or 0) == int(evolve_step)
                ),
                "",
            ),
            "ge_hard_lambda_star": next(
                (
                    row.get("ge_hard_lambda_star", "")
                    for row in collector.capacity_rows
                    if str(row.get("task_family")) == str(family)
                    and int(row.get("evolve_step", 0) or 0) == int(evolve_step)
                ),
                "",
            ),
            "ge_valid_lambda_coverage": next(
                (
                    row.get("ge_valid_lambda_coverage", 0.0)
                    for row in collector.capacity_rows
                    if str(row.get("task_family")) == str(family)
                    and int(row.get("evolve_step", 0) or 0) == int(evolve_step)
                ),
                0.0,
            ),
            "ge_margin_drop_f": next(
                (
                    row.get("ge_margin_drop_f", 0.0)
                    for row in collector.capacity_rows
                    if str(row.get("task_family")) == str(family)
                    and int(row.get("evolve_step", 0) or 0) == int(evolve_step)
                ),
                0.0,
            ),
            "ge_monotonicity_violations": next(
                (
                    row.get("ge_monotonicity_violations", 0)
                    for row in collector.capacity_rows
                    if str(row.get("task_family")) == str(family)
                    and int(row.get("evolve_step", 0) or 0) == int(evolve_step)
                ),
                0,
            ),
            "ge_max_degradation_rate": next(
                (
                    row.get("ge_max_degradation_rate", 0.0)
                    for row in collector.capacity_rows
                    if str(row.get("task_family")) == str(family)
                    and int(row.get("evolve_step", 0) or 0) == int(evolve_step)
                ),
                0.0,
            ),
            "ge_cliff_flag": next(
                (
                    row.get("ge_cliff_flag", 0)
                    for row in collector.capacity_rows
                    if str(row.get("task_family")) == str(family)
                    and int(row.get("evolve_step", 0) or 0) == int(evolve_step)
                ),
                0,
            ),
        }
        for (family, evolve_step), audc in sorted(audc_map.items())
    ]
    if audc_rows:
        write_csv_rows(output_dir / "resilience_margin_audc.csv", audc_rows)

    margin_lookup = {
        (margin.family, float(margin.stress_lambda), int(margin.evolve_step)): margin
        for margin in margins.values()
    }
    for row in normalized_rows:
        key = (
            str(row.get("task_family", "other")),
            float(row.get("stress_lambda", 0.0) or 0.0),
            int(row.get("evolve_step", 0) or 0),
        )
        margin = margin_lookup.get(key)
        if margin is None:
            continue
        row.update(margin.get_summary_for_csv())

    if raw_output_path is not None and normalized_rows:
        write_csv_rows(raw_output_path, normalized_rows)
    return margins


def collect_critic_exports(
    specs: Sequence[PerturbationSpec],
    resilience_output_dir: Path,
) -> List[Path]:
    export_paths: List[Path] = []
    for spec in specs:
        s_dir = spec_output_dir(resilience_output_dir, spec)
        if not s_dir.exists():
            continue
        export_paths.extend(sorted(s_dir.rglob("critic_export-episode_*.json")))
        export_paths.extend(sorted(s_dir.rglob("critic_export-episode_*.jsonl")))
    deduped = {str(path.resolve()): path for path in export_paths}
    return list(deduped.values())


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
    estimator = StageBaselineEstimator(
        tau_pln=float(c_rec_cfg.get("tau_pln", 0.1)),
        tau_per=float(c_rec_cfg.get("tau_per", 0.1)),
        tau_coord=float(c_rec_cfg.get("tau_coord", 0.1)),
    )
    signature = build_stage_baseline_signature(
        config=config,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs_for_signature or [clean_spec],
    )
    _baselines, episode_baselines, baseline_stats = (
        estimator.estimate_grouped_from_exports(export_paths)
    )
    manifest = update_episode_aggregated_baselines(
        estimator=estimator,
        episode_baselines=episode_baselines,
        episode_stats=baseline_stats,
        stage_baseline_cfg=stage_baseline_cfg,
        signature=signature,
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


def generate_stage_baseline(
    *,
    base_results_dir: Path,
    resilience_output_dir: Path,
    resilience_cfg: Mapping[str, Any],
    clean_specs: Sequence[PerturbationSpec],
    config: Optional[Mapping[str, Any]] = None,
    update_episode_aggregate: bool = True,
) -> Optional[Path]:
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
    estimator = StageBaselineEstimator(
        tau_pln=float(c_rec_cfg.get("tau_pln", 0.1)),
        tau_per=float(c_rec_cfg.get("tau_per", 0.1)),
        tau_coord=float(c_rec_cfg.get("tau_coord", 0.1)),
    )
    stage_baseline_cfg = resilience_cfg.get("stage_baseline", {}) or {}
    signature = build_stage_baseline_signature(
        config=config,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs,
    )
    baselines, episode_baselines, baseline_stats = estimator.estimate_grouped_from_exports(
        export_paths
    )
    output_path = resolve_baseline_output_path(base_results_dir, resilience_cfg)
    estimator.write(
        baselines,
        output_path,
        metadata={
            "baseline_kind": "hash_run_local",
            "signature": signature,
            "estimator_stats": baseline_stats,
        },
    )

    persistent_path: Optional[Path] = None
    latest_alias_path: Optional[Path] = None
    preferred_path = output_path
    if bool(stage_baseline_cfg.get("write_persistent_copy", True)):
        persistent_dir = resolve_persistent_baseline_dir(stage_baseline_cfg)
        persistent_path = persistent_dir / str(signature["archive_filename"])
        estimator.write(
            baselines,
            persistent_path,
            metadata={
                "baseline_kind": "hash_persistent",
                "signature": signature,
                "estimator_stats": baseline_stats,
            },
        )
        latest_alias = str(
            stage_baseline_cfg.get("latest_alias", "stage_baseline_latest.json")
        ).strip()
        if latest_alias:
            latest_alias_path = persistent_dir / latest_alias
            estimator.write(
                baselines,
                latest_alias_path,
                metadata={
                    "baseline_kind": "latest_alias",
                    "signature": signature,
                    "estimator_stats": baseline_stats,
                },
            )
        preferred_path = persistent_path

    episode_aggregate_manifest: Dict[str, Any] = {}
    if update_episode_aggregate:
        episode_aggregate_manifest = update_episode_aggregated_baselines(
            estimator=estimator,
            episode_baselines=episode_baselines,
            episode_stats=baseline_stats,
            stage_baseline_cfg=stage_baseline_cfg,
            signature=signature,
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
    args = parser.parse_args()

    config_path = Path(args.spec_config)
    if not config_path.exists():
        print(f"[resilience-worker] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)

    from habitat_llm.agent.env.dataset import CollaborationDatasetV0
    from habitat_llm.examples import planner_demo_mp_new as base_runner

    dataset = CollaborationDatasetV0(cfg.habitat.dataset)
    if (
        cfg.get("episode_indices", None) is not None
        or cfg.get("episode_ids", None) is not None
        or cfg.get("evolve_episode_ids", None) is not None
    ):
        dataset = base_runner._select_episode_subset(dataset, cfg)

    logger.info(
        "[resilience-worker] Starting run_planner with %d episodes -> %s",
        len(dataset.episodes),
        cfg.paths.results_dir,
    )
    base_runner.run_planner(cfg, dataset)
    logger.info("[resilience-worker] run_planner completed.")


if __name__ == "__main__":
    _subprocess_rollout_spec_entry()
