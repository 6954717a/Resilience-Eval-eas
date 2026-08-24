#!/usr/bin/env python3

"""
Modular analysis helpers for EvaluationRunner.

Keeps episode execution separate from optional metric logging and
post-hoc analysis pipelines such as ADCA, Rebound, SayCan, resilience,
and critic statistics.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from habitat_llm.evaluation.eval_logging import (
    log_adca_analysis,
    log_planner_data,
    log_rebound_analysis,
    log_saycan_analysis,
    save_detailed_traces,
    write_critic_stats,
)
from habitat_llm.evaluation.llm_evaluator.adca_analyzer import ADCAAnalyzer
from habitat_llm.evaluation.monitors import MonitorRegistry, SafetyMonitor

# Import dimension-based metrics modules
from habitat_llm.evaluation.metrics import (
    ReboundCollector,
    log_rebound_metrics,
    StabilityCollector,
    log_stability_metrics,
)
from habitat_llm.evaluation.stage_baseline import (
    StageBaselineEstimator,
    baseline_episode_matches,
    judge_models_match,
)
from habitat_llm.evaluation.reporting import primary_display_items

# Import MetricsAggregator from core
from habitat_llm.evaluation.core import MetricsAggregator, EpisodeMetrics

logger = logging.getLogger(__name__)


class EvaluationAnalysisCoordinator:
    """
    Handles optional analysis and logging that should not clutter the main
    EvaluationRunner execution flow.
    """

    def __init__(self, evaluation_runner_config: Any, env_interface: Any):
        self.evaluation_runner_config = evaluation_runner_config
        self.env_interface = env_interface
        self.adca_analyzer = self._initialize_adca_analyzer()
        safety_config = {}
        if hasattr(self.evaluation_runner_config, "safety_monitor"):
            safety_config = OmegaConf.to_container(
                self.evaluation_runner_config.safety_monitor,
                resolve=True,
            )
        self.monitors = MonitorRegistry(
            {"safety": SafetyMonitor(safety_config)}
        )
        # Compatibility alias for integrations that still access the concrete
        # monitor. New collectors consume immutable summary snapshots instead.
        self.safety_monitor = self.monitors.require("safety")

        # Runtime collectors. Formal beta and GE are computed only after the
        # clean/perturbed neighborhood and stress grid have been assembled.
        metrics_config = self._get_metrics_config()
        self.rebound_collector = ReboundCollector(metrics_config.get("rebound", {}))
        self.stability_collector = StabilityCollector(metrics_config.get("stability", {}))

        self.metrics_aggregator = MetricsAggregator(
            rebound_collector=self.rebound_collector,
            stability_collector=self.stability_collector,
            config=metrics_config,
        )

        # Cached per-stage baselines (loaded lazily on first multidim call).
        self._stage_baselines: Optional[Dict[tuple, Any]] = None
        self._stage_baselines_path: Optional[str] = None
        self._stage_baseline_metadata: Dict[str, Any] = {}
        self._stage_baseline_load_reason: str = ""

    def reset(self) -> None:
        self.monitors.reset()

    def update_step_health_metrics(
        self,
        *,
        planner_info: Dict[str, Any],
        agent_collisions: Dict[int, bool],
        critic: Optional[Any] = None,
    ) -> None:
        safety_step_data = {
            "agent_collisions": agent_collisions,
            "collision_scope": planner_info.get("agent_collision_scope", ""),
        }
        # Distance evidence is optional and currently not produced by every
        # Habitat sensor stack.  Forward it when a runtime integration supplies
        # it; SafetyMonitor records an explicit collision-only scope otherwise.
        if planner_info.get("min_obstacle_dist") not in (None, ""):
            safety_step_data["min_obstacle_dist"] = planner_info[
                "min_obstacle_dist"
            ]
        monitor_updates = self.monitors.update_all(
            {"safety": safety_step_data}
        )
        planner_info.update(monitor_updates.get("safety", {}))

        if critic is not None:
            planner_info["value_function_variance"] = critic.compute_value_variance(
                window=10
            )

    def _get_metrics_config(self) -> Dict[str, Any]:
        """
        Extract metrics configuration with legacy mapping support.

        Returns:
            Dictionary with metrics config for each dimension
        """
        config = OmegaConf.to_container(self.evaluation_runner_config, resolve=True)

        metrics_config = dict(config.get("metrics") or {})
        metrics_config.setdefault("rebound", {"enabled": True, "gamma": 2.0})
        metrics_config.setdefault(
            "stability", {"enabled": True, "use_saycan_scores": False}
        )

        # Map planner.plan_config.rebound -> metrics.rebound
        if "planner" in config and "plan_config" in config["planner"]:
            plan_config = config["planner"]["plan_config"]
            if "rebound" in plan_config:
                rebound_cfg = plan_config["rebound"]
                metrics_config["rebound"]["enabled"] = rebound_cfg.get("enabled", True)
                if "gamma" in rebound_cfg:
                    metrics_config["rebound"]["gamma"] = rebound_cfg["gamma"]

        root = getattr(self.env_interface, "conf", None)
        resilience_cfg = (
            OmegaConf.select(root, "resilience", default=None)
            or OmegaConf.select(root, "evaluation.resilience", default=None)
        )
        if resilience_cfg is not None:
            materialized = OmegaConf.to_container(resilience_cfg, resolve=True)
            if isinstance(materialized, dict):
                metrics_config["rebound"]["c_rec"] = dict(
                    materialized.get("c_rec") or {}
                )
                metrics_config["stability"].update(
                    dict(materialized.get("stability") or {})
                )

        return metrics_config

    def _initialize_adca_analyzer(self) -> Optional[ADCAAnalyzer]:
        adca_enabled = OmegaConf.select(
            self.evaluation_runner_config, "adca.enabled", default=False
        )
        logger.info("Checking ADCA configuration: enabled=%s", adca_enabled)
        if not adca_enabled:
            return None

        llm_client = None
        llm_base_url = OmegaConf.select(
            self.evaluation_runner_config, "adca.llm_base_url", default=None
        )
        llm_api_key = OmegaConf.select(
            self.evaluation_runner_config, "adca.llm_api_key", default=None
        )
        adca_config_obj = getattr(self.evaluation_runner_config, "adca", None)
        adca_config = (
            OmegaConf.to_container(adca_config_obj, resolve=True)
            if adca_config_obj
            else {}
        )

        api_key_status = "set" if llm_api_key else "not set"
        env_key_status = "set" if os.environ.get("OPENAI_API_KEY") else "not set"
        logger.info(
            "ADCA config values: llm_base_url=%s, llm_api_key=%s",
            llm_base_url,
            api_key_status,
        )
        logger.info("Environment OPENAI_API_KEY: %s", env_key_status)

        if llm_base_url or llm_api_key:
            try:
                from openai import OpenAI

                api_key = llm_api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
                if llm_base_url:
                    llm_client = OpenAI(base_url=llm_base_url, api_key=api_key)
                    logger.info(
                        "Created custom ADCA LLM client with base_url=%s",
                        llm_base_url,
                    )
                else:
                    logger.warning(
                        "ADCA config missing llm_base_url. llm_api_key=%s",
                        api_key_status,
                    )
            except Exception as exc:
                logger.error("Failed to create custom ADCA LLM client: %s", exc)
        else:
            logger.warning(
                "ADCA config missing both llm_base_url and llm_api_key. "
                "llm_base_url=%s, llm_api_key=%s",
                llm_base_url,
                api_key_status,
            )

        if llm_client is None and self.env_interface is not None and hasattr(
            self.env_interface, "llm_client"
        ):
            llm_client = self.env_interface.llm_client
            logger.info("Using env_interface LLM client for ADCA")

        if llm_client is None:
            logger.warning("ADCA enabled but no LLM client found")
            return None

        try:
            analyzer = ADCAAnalyzer(llm_client, adca_config)
            logger.info("ADCA Analyzer initialized successfully")
            return analyzer
        except Exception as exc:
            logger.error("Failed to initialize ADCA Analyzer: %s", exc)
            return None

    def log_planner_data(
        self,
        *,
        output_dir: str,
        episode_filename: str,
        agents: Dict[int, Any],
        planner_infos: List[Dict[str, Any]],
        current_instruction: str,
        canonical_instruction: Optional[str] = None,
        policy_instruction: Optional[str] = None,
    ) -> None:
        log_planner_data(
            output_dir=output_dir,
            episode_filename=episode_filename,
            agents=agents,
            planner_infos=planner_infos,
            current_instruction=current_instruction,
            canonical_instruction=canonical_instruction,
            policy_instruction=policy_instruction,
            env_interface=self.env_interface,
            log_detailed_traces=self.evaluation_runner_config.log_detailed_traces,
        )

    def save_detailed_traces(
        self,
        *,
        output_dir: str,
        episode_filename: str,
        current_instruction: str,
    ) -> None:
        save_detailed_traces(
            output_dir=output_dir,
            episode_filename=episode_filename,
            current_instruction=current_instruction,
            env_interface=self.env_interface,
        )

    def construct_adca_trajectory(self) -> List[Dict[str, str]]:
        all_actions = []
        for _, history in self.env_interface.agent_action_history.items():
            all_actions.extend(history)
        all_actions.sort(key=lambda elem: elem.timestamp)

        trajectory = []
        for elem in all_actions:
            action_content = elem.to_string()
            trajectory.append(
                {
                    "action": f"Agent {elem.agent_uid}: {action_content}",
                    "observation": elem.response if elem.response else "",
                }
            )
        return trajectory

    def log_episode_analyses(
        self,
        *,
        output_dir: str,
        episode_filename: str,
        current_instruction: str,
        canonical_instruction: Optional[str] = None,
        policy_instruction: Optional[str] = None,
        planner: Any,
        planner_infos: List[Dict[str, Any]],
        info: Dict[str, Any],
        critic: Optional[Any] = None,
    ) -> None:
        """
        Log episode analyses using dimension-based metrics system.

        Collects runtime Rebound and Stability evidence. Formal β and GE are
        calculated later by the resilience experiment post-processor.
        """
        # Collect registered single-episode evidence. Boundary/GE is computed
        # later from the complete clean/perturbed stress grid.
        episode_metrics = self.metrics_aggregator.collect_episode_metrics(
            planner=planner,
            planner_infos=planner_infos,
            critic=critic,
            total_steps=info.get("total_step_count", 0),
            monitor_registry=self.monitors,
            info=info,
        )

        # Augment Rebound metrics with the multidim C_rec pipeline when the
        # OnlineReboundTracker left a summary in ``info``.
        self._augment_rebound_multidim(
            episode_metrics=episode_metrics,
            planner_infos=planner_infos,
            info=info,
        )

        # Persist metrics to disk
        self._persist_metrics(
            episode_metrics=episode_metrics,
            output_dir=output_dir,
            episode_filename=episode_filename,
        )

        # Update info dict with CSV summary for all dimensions
        csv_summary = self.metrics_aggregator.get_csv_summary(episode_metrics)
        info.update(csv_summary)

        # Independent Analysis Modules (preserved)
        self._log_critic_stats(
            output_dir=output_dir,
            episode_filename=episode_filename,
            info=info,
            critic=critic,
        )
        self._log_adca(
            output_dir=output_dir,
            episode_filename=episode_filename,
            current_instruction=policy_instruction or current_instruction,
            canonical_instruction=canonical_instruction,
            policy_instruction=policy_instruction,
            info=info,
            critic=critic,
        )

    def _hydra_root_config(self) -> Any:
        """Full Hydra config (workspace root).
        """
        env = getattr(self, "env_interface", None)
        if env is None:
            return None
        return getattr(env, "conf", None)

    def _resolve_persistent_baseline_fallback(self) -> Optional[str]:
        """Resolve the Judge-scoped baseline outside timestamped run outputs."""
        cfg = self.evaluation_runner_config
        root = self._hydra_root_config()

        def _select(conf: Any, key: str) -> Any:
            if conf is None:
                return None
            return OmegaConf.select(conf, key, default=None)

        raw_dir = (
            _select(cfg, "resilience.stage_baseline.persistent_store_dir")
            or _select(root, "resilience.stage_baseline.persistent_store_dir")
            or "evaluation/stage_baselines"
        )
        latest_alias = (
            _select(cfg, "resilience.stage_baseline.latest_alias")
            or _select(root, "resilience.stage_baseline.latest_alias")
            or "stage_baseline_latest.json"
        )
        if not latest_alias:
            return None
        judge_model = (
            _select(root, "resilience.stage_baseline.judge_model")
            or _select(root, "evaluation.critic.llm_model")
            or _select(root, "evaluation.adca.llm_model")
            or _select(cfg, "critic.llm_model")
            or "unconfigured-judge"
        )
        judge_slug = re.sub(
            r"[^A-Za-z0-9._-]+", "-", str(judge_model)
        ).strip("-._") or "unconfigured-judge"
        statistics_subdir = (
            _select(root, "resilience.stage_baseline.statistics_subdir")
            or _select(cfg, "resilience.stage_baseline.statistics_subdir")
            or "statistics"
        )
        persistent_dir = Path(str(raw_dir))
        if not persistent_dir.is_absolute():
            persistent_dir = Path(__file__).resolve().parents[1] / persistent_dir
        judge_root = persistent_dir / judge_slug

        # Cumulative StageBaseline fitting stores a stable pointer under the
        # Judge-scoped evidence contract.  The pointer contains a path relative
        # to its contract directory; the snapshot itself is independent of the
        # current Hydra outputs/<timestamp> directory.
        contracts_root = judge_root / "contracts"
        preferred_pointer = contracts_root / "judge_episode_baseline" / "latest.json"
        pointer_paths = [preferred_pointer]
        if contracts_root.is_dir():
            pointer_paths.extend(
                sorted(
                    (
                        path
                        for path in contracts_root.glob("*/latest.json")
                        if path != preferred_pointer
                    ),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )
        for pointer_path in pointer_paths:
            if not pointer_path.is_file():
                continue
            try:
                with open(pointer_path, "r", encoding="utf-8") as handle:
                    pointer = json.load(handle)
                contract_root = pointer_path.parent
                snapshot_path = contract_root / str(
                    pointer.get("snapshot_path") or ""
                )
                if snapshot_path.is_file():
                    return str(snapshot_path.resolve())
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
                continue

        legacy_alias = (
            judge_root / str(statistics_subdir) / str(latest_alias)
        )
        return str(legacy_alias.resolve()) if legacy_alias.is_file() else None

    def _critic_phase(self) -> str:
        root = self._hydra_root_config()
        phase = ""
        if root is not None:
            phase = str(
                OmegaConf.select(
                    root,
                    "evaluation.critic.phase",
                    default="",
                )
                or ""
            )
        if not phase:
            phase = str(
                OmegaConf.select(
                    self.evaluation_runner_config,
                    "critic.phase",
                    default="evaluation",
                )
                or "evaluation"
            )
        return phase.strip().lower()

    @staticmethod
    def _configured_path(path: Any) -> Path:
        candidate = Path(os.path.expanduser(str(path)))
        if candidate.is_absolute():
            return candidate
        return Path(__file__).resolve().parents[1] / candidate

    def _resolve_baseline_path(self) -> Optional[str]:
        """Resolve the ``stage_baseline.json`` path from Hydra config.

        Lookup order:

        1. An existing explicit ``evaluation.rebound_tracker.baseline_path``.
        2. The Judge-scoped persistent snapshot selected by ``latest.json``.
        3. An existing run-local baseline under ``paths.results_dir``.

        Missing explicit paths are ignored because they commonly point into a
        previous Hydra ``outputs/<timestamp>`` directory.
        """
        cfg = self.evaluation_runner_config
        root = self._hydra_root_config()

        def _select(conf: Any, key: str) -> Any:
            if conf is None:
                return None
            return OmegaConf.select(conf, key, default=None)

        # 1: explicit path on the evaluation subtree (usual case). A stale
        # path from an older outputs/<timestamp> run does not shadow the
        # persistent Judge-scoped snapshot.
        path = _select(cfg, "rebound_tracker.baseline_path")
        if path:
            candidate = self._configured_path(path)
            if candidate.is_file():
                return str(candidate.resolve())
        # 2: full Hydra root nests rebound settings under ``evaluation.*``.
        path = _select(root, "evaluation.rebound_tracker.baseline_path")
        if path:
            candidate = self._configured_path(path)
            if candidate.is_file():
                return str(candidate.resolve())

        # 3: the persistent store is the cross-run source of truth. It is not
        # nested under Hydra's timestamped results directory.
        persistent_fallback = self._resolve_persistent_baseline_fallback()
        if persistent_fallback:
            return persistent_fallback

        # 4: a run-local baseline is useful only when it already exists.
        baseline_json = (
            _select(cfg, "resilience.baseline_json")
            or _select(root, "resilience.baseline_json")
            or "stage_baseline.json"
        )
        output_subdir = (
            _select(cfg, "resilience.output_subdir")
            or _select(root, "resilience.output_subdir")
            or "resilience"
        )
        results_dir = _select(cfg, "paths.results_dir") or _select(root, "paths.results_dir")
        if results_dir:
            run_local_path = os.path.join(
                str(results_dir),
                str(output_subdir),
                str(baseline_json),
            )
            if os.path.isfile(run_local_path):
                return str(Path(run_local_path).resolve())
        return None

    def _load_stage_baselines(self) -> Dict[tuple, Any]:
        """Load and cache the per-anchor stage baselines mapping."""
        phase = self._critic_phase()
        if phase in {"train", "training_collect", "reference"}:
            self._stage_baselines = {}
            self._stage_baselines_path = None
            self._stage_baseline_metadata = {"critic_phase": phase}
            self._stage_baseline_load_reason = (
                "stage_baseline_calibration_in_progress"
                if phase == "reference"
                else "stage_baseline_not_used_for_critic_training"
            )
            return {}
        path = self._resolve_baseline_path()
        if not path:
            self._stage_baselines = {}
            self._stage_baselines_path = None
            self._stage_baseline_metadata = {}
            self._stage_baseline_load_reason = "stage_baseline_missing"
            return {}
        if self._stage_baselines is not None and self._stage_baselines_path == path:
            return self._stage_baselines
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            metadata = dict(payload.get("metadata") or {})
            signature = dict(metadata.get("signature") or {})
            artifact_judge = str(
                metadata.get("judge_model") or signature.get("judge_model") or ""
            )
            root = self._hydra_root_config()
            configured_judge = str(
                (
                    OmegaConf.select(
                        root,
                        "evaluation.critic.llm_model",
                        default="",
                    )
                    if root is not None
                    else ""
                )
                or ""
            )
            if not configured_judge:
                configured_judge = str(
                    OmegaConf.select(
                        self.evaluation_runner_config,
                        "critic.llm_model",
                        default="",
                    )
                    or ""
                )
            if not judge_models_match(artifact_judge, configured_judge):
                self._stage_baseline_metadata = metadata
                self._stage_baseline_load_reason = "stage_baseline_judge_mismatch"
                logger.warning(
                    "Refusing StageBaseline %s: artifact Judge %r does not match "
                    "configured critic Judge %r.",
                    path,
                    artifact_judge,
                    configured_judge,
                )
                baselines = {}
            else:
                baselines = StageBaselineEstimator.load(path)
                self._stage_baseline_metadata = metadata
                self._stage_baseline_load_reason = ""
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load stage baselines from %s: %s", path, exc)
            baselines = {}
            self._stage_baseline_metadata = {}
            self._stage_baseline_load_reason = "stage_baseline_load_failed"
        self._stage_baselines = baselines
        self._stage_baselines_path = path
        if baselines:
            logger.info("Loaded %d stage baselines from %s", len(baselines), path)
        return baselines

    def _augment_rebound_multidim(
        self,
        *,
        episode_metrics: EpisodeMetrics,
        planner_infos: List[Dict[str, Any]],
        info: Dict[str, Any],
    ) -> None:
        """Run the multidim ``C_rec`` pipeline on the tracker summary, if any.

        Populates ``episode_metrics.rebound`` with raw tracker diagnostics plus
        formal ``C_rec`` metrics. Tracker windows remain visible even when the
        clean baseline is missing; only the formal recovery-window fields
        depend on the baseline.
        """
        # Accept both the canonical underscored key (current behaviour) and
        # the legacy ``rebound_tracker_summary`` key for backwards.
        tracker_summary = None
        if isinstance(info, dict):
            tracker_summary = (
                info.get("_rebound_tracker_summary")
                or info.get("rebound_tracker_summary")
            )
        if not tracker_summary:
            if episode_metrics.rebound is not None:
                episode_metrics.rebound.c_rec_valid = False
                episode_metrics.rebound.c_rec_missing_reason = (
                    "rebound_tracker_summary_missing"
                )
            return
        if episode_metrics.rebound is None:
            return

        rebound_metrics = episode_metrics.rebound
        tracker_windows = list(tracker_summary.get("windows") or [])
        tracker_window_count = int(len(tracker_windows))
        tracker_window_open_final = bool(
            tracker_summary.get("window_open_final")
            or tracker_summary.get("rebound_window_open")
        )
        rebound_metrics.template_distribution = dict(
            tracker_summary.get("template_distribution") or {}
        )
        rebound_metrics.num_template_events = int(
            tracker_summary.get("total_events") or 0
        )
        rebound_metrics.rebound_events = tracker_windows
        rebound_metrics.tracker_window_count = tracker_window_count
        rebound_metrics.tracker_window_open_final = tracker_window_open_final

        baselines = self._load_stage_baselines()
        alignment_reason = self._stage_baseline_load_reason
        episode_id = str(tracker_summary.get("episode_id") or "")
        if not episode_id:
            transitions = tracker_summary.get("transitions") or planner_infos
            if transitions and isinstance(transitions[0], dict):
                episode_id = str(transitions[0].get("episode_id") or "")
        if baselines and episode_id and not baseline_episode_matches(
            self._stage_baseline_metadata, episode_id
        ):
            baselines = {}
            alignment_reason = "stage_baseline_episode_mismatch"
        rebound_metrics.baseline_loaded = bool(baselines)
        if (
            not baselines
            and tracker_window_count > 0
            and alignment_reason
            not in {
                "stage_baseline_calibration_in_progress",
                "stage_baseline_not_used_for_critic_training",
            }
        ):
            logger.warning(
                "Rebound tracker detected %d raw window(s) but no stage baseline "
                "was loaded from %s; formal C_rec will be marked invalid rather than zero.",
                tracker_window_count,
                self._resolve_baseline_path(),
            )
        try:
            episode_metrics.rebound = self.rebound_collector.compute_stage_crec_multidim(
                transitions=(tracker_summary.get("transitions") or planner_infos),
                baselines=baselines,
                tracker_summary=tracker_summary,
                metrics=episode_metrics.rebound,
            )
            if alignment_reason and not episode_metrics.rebound.c_rec_valid:
                episode_metrics.rebound.c_rec_missing_reason = alignment_reason
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("compute_stage_crec_multidim failed: %s", exc)
            rebound_metrics.c_rec_valid = False
            rebound_metrics.c_rec_missing_reason = (
                f"c_rec_computation_error:{type(exc).__name__}"
            )

    def _persist_metrics(
        self,
        episode_metrics: EpisodeMetrics,
        output_dir: str,
        episode_filename: str,
    ) -> None:
        """
        Persist all collected metrics to disk.

        Args:
            episode_metrics: EpisodeMetrics with all collected metrics
            output_dir: Output directory for metric files
            episode_filename: Episode filename for naming
        """
        # Persist Rebound metrics
        if episode_metrics.rebound:
            try:
                log_rebound_metrics(
                    rebound_metrics=episode_metrics.rebound,
                    output_dir=output_dir,
                    episode_filename=episode_filename,
                    env_interface=self.env_interface,
                )
            except Exception as exc:
                logger.warning("Failed to persist Rebound metrics: %s", exc)

        # Persist Stability metrics
        if episode_metrics.stability:
            try:
                log_stability_metrics(
                    stability_metrics=episode_metrics.stability,
                    output_dir=output_dir,
                    episode_filename=episode_filename,
                    env_interface=self.env_interface,
                )
            except Exception as exc:
                logger.warning("Failed to persist Stability metrics: %s", exc)

        # Keep the human-facing episode result compact while retaining a
        # machine-readable pointer to the dimension-specific diagnostics.
        try:
            summary = self.metrics_aggregator.get_csv_summary(episode_metrics)
            episode_id = str(
                self.env_interface.env.env.env._env.current_episode.episode_id
            )
            summary_dir = Path(output_dir) / "analyses" / "resilience_summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / (
                f"resilience_primary-episode_{episode_id}_{episode_filename}.json"
            )
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "episode_id": episode_id,
                        "episode_filename": episode_filename,
                        "primary_metrics": dict(primary_display_items(summary)),
                        "diagnostic_directories": {
                            "rebound": "../rebound",
                            "stability": "../stability",
                        },
                    },
                    handle,
                    indent=2,
                )
        except Exception as exc:
            logger.warning("Failed to persist concise resilience summary: %s", exc)

    def _log_critic_stats(
        self,
        *,
        output_dir: str,
        episode_filename: str,
        info: Dict[str, Any],
        critic: Optional[Any],
    ) -> None:
        if critic is None:
            return
        try:
            critic_stats = critic.get_statistics()
            info["critic_stats"] = critic_stats
            if self.evaluation_runner_config.get("log_critic_stats", True):
                write_critic_stats(
                    critic_stats=critic_stats,
                    output_dir=output_dir,
                    episode_filename=episode_filename,
                    env_interface=self.env_interface,
                    critic=critic,
                )
        except Exception as exc:
            logger.warning("Failed to log critic statistics: %s", exc)

    def _log_adca(
        self,
        *,
        output_dir: str,
        episode_filename: str,
        current_instruction: str,
        canonical_instruction: Optional[str],
        policy_instruction: Optional[str],
        info: Dict[str, Any],
        critic: Optional[Any],
    ) -> None:
        if self.adca_analyzer is None:
            return
        try:
            trajectory = self.construct_adca_trajectory()
            outcome_score = 1.0 if info.get("task_state_success", 0.0) > 0.5 else 0.0
            adca_result = self.adca_analyzer.analyze(
                trajectory,
                outcome_score,
                current_instruction,
            )
            if adca_result:
                log_adca_analysis(
                    adca_result=adca_result,
                    output_dir=output_dir,
                    episode_filename=episode_filename,
                    env_interface=self.env_interface,
                    current_instruction=current_instruction,
                    canonical_instruction=canonical_instruction,
                    policy_instruction=policy_instruction,
                    critic=critic,
                )
        except Exception as exc:
            logger.error("ADCA Analysis failed: %s", exc)
