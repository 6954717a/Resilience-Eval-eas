#!/usr/bin/env python3

"""
Modular analysis helpers for EvaluationRunner.

Keeps episode execution separate from optional metric logging and
post-hoc analysis pipelines such as ADCA, Rebound, SayCan, resilience,
and critic statistics.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
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
from habitat_llm.evaluation.monitors.degradation_monitor import DegradationMonitor
from habitat_llm.evaluation.llm_evaluator.adca_analyzer import ADCAAnalyzer
from habitat_llm.evaluation.monitors.safety_monitor import SafetyMonitor

# Import dimension-based metrics modules
from habitat_llm.evaluation.metrics import (
    ReboundCollector,
    log_rebound_metrics,
    StabilityCollector,
    log_stability_metrics,
    DegradationCollector,
    log_degradation_metrics,
)
from habitat_llm.evaluation.metrics.rebound import StageBaselineEstimator

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
        self.safety_monitor = SafetyMonitor(safety_config)

        degradation_config = {}
        if hasattr(self.evaluation_runner_config, "degradation_monitor"):
            degradation_config = OmegaConf.to_container(
                self.evaluation_runner_config.degradation_monitor,
                resolve=True,
            )
        self.degradation_monitor = DegradationMonitor(degradation_config)

        # Initialize dimension-based collectors
        metrics_config = self._get_metrics_config()
        self.rebound_collector = ReboundCollector(metrics_config.get("rebound", {}))
        self.stability_collector = StabilityCollector(metrics_config.get("stability", {}))
        self.degradation_collector = DegradationCollector(metrics_config.get("degradation", {}))

        # Initialize MetricsAggregator for unified collection
        self.metrics_aggregator = MetricsAggregator(
            rebound_collector=self.rebound_collector,
            stability_collector=self.stability_collector,
            degradation_collector=self.degradation_collector,
            config=metrics_config,
        )

        # Cached per-stage baselines (loaded lazily on first multidim call).
        self._stage_baselines: Optional[Dict[tuple, Any]] = None
        self._stage_baselines_path: Optional[str] = None

    def reset(self) -> None:
        self.safety_monitor.reset()
        self.degradation_monitor.reset()

    def update_step_health_metrics(
        self,
        *,
        planner_info: Dict[str, Any],
        agent_collisions: Dict[int, bool],
        current_progress: float,
        critic: Optional[Any] = None,
    ) -> None:
        safety_step_data = {
            "agent_collisions": agent_collisions,
        }
        safety_metrics = self.safety_monitor.check_safety(safety_step_data)
        planner_info.update(safety_metrics)

        deg_step_data = {
            "task_percent_complete": current_progress,
        }
        deg_metrics = self.degradation_monitor.update(deg_step_data)
        planner_info.update(deg_metrics)

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

        # Check for new metrics config format
        if "metrics" in config:
            return config["metrics"]

        # Legacy mapping: map old config keys to new structure
        metrics_config = {
            "rebound": {"enabled": True, "gamma": 2.0},
            "stability": {"enabled": True, "use_saycan_scores": False},
            "degradation": {"enabled": True},
        }

        # Map planner.plan_config.rebound -> metrics.rebound
        if "planner" in config and "plan_config" in config["planner"]:
            plan_config = config["planner"]["plan_config"]
            if "rebound" in plan_config:
                rebound_cfg = plan_config["rebound"]
                metrics_config["rebound"]["enabled"] = rebound_cfg.get("enabled", True)
                if "gamma" in rebound_cfg:
                    metrics_config["rebound"]["gamma"] = rebound_cfg["gamma"]

        # Map evaluation.context_evolve -> metrics.degradation (if exists)
        if "context_evolve" in config:
            metrics_config["degradation"]["enabled"] = config["context_evolve"].get("enabled", False)

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
    ) -> None:
        log_planner_data(
            output_dir=output_dir,
            episode_filename=episode_filename,
            agents=agents,
            planner_infos=planner_infos,
            current_instruction=current_instruction,
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
        planner: Any,
        planner_infos: List[Dict[str, Any]],
        info: Dict[str, Any],
        critic: Optional[Any] = None,
    ) -> None:
        """
        Log episode analyses using dimension-based metrics system.

        Organizes metrics into three core dimensions:
        1. Rebound: Recovery from failures (B_epi, MTTR, MTBF, RR, T_rec)
        2. Stability: Output consistency (β, σ²_V, N_replan, P_cbf)
        3. Degradation: Graceful degradation (AUC_loss, P_cliff, T_rec, L_bd, NRR)

        Uses MetricsAggregator for unified collection to avoid duplicate extraction.
        """
        # Collect all three-dimensional metrics using MetricsAggregator
        episode_metrics = self.metrics_aggregator.collect_episode_metrics(
            planner=planner,
            planner_infos=planner_infos,
            critic=critic,
            safety_monitor=self.safety_monitor,
            degradation_monitor=self.degradation_monitor,
            total_steps=info.get("total_step_count", 0),
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
            current_instruction=current_instruction,
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
        """Resolve the repo-stable persistent baseline alias, if configured."""
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
        persistent_dir = Path(str(raw_dir))
        if not persistent_dir.is_absolute():
            persistent_dir = Path(__file__).resolve().parents[1] / persistent_dir
        return str(persistent_dir / str(latest_alias))

    def _resolve_baseline_path(self) -> Optional[str]:
        """Resolve the ``stage_baseline.json`` path from Hydra config.

        Lookup order:

        1. ``evaluation.rebound_tracker.baseline_path`` (evaluation subtree).
        2. ``evaluation.rebound_tracker.baseline_path`` on the **root** config
           (if someone nests it there).
        3. ``${paths.results_dir}/${resilience.output_subdir}/${resilience.baseline_json}``
           using **root** ``resilience`` + ``paths`` (matches ``resilience_config.yaml``).
        """
        cfg = self.evaluation_runner_config
        root = self._hydra_root_config()

        def _select(conf: Any, key: str) -> Any:
            if conf is None:
                return None
            return OmegaConf.select(conf, key, default=None)

        # 1: explicit path on the evaluation subtree (usual case).
        path = _select(cfg, "rebound_tracker.baseline_path")
        if path:
            return os.path.expanduser(str(path))
        # 2: full Hydra root nests rebound settings under ``evaluation.*``.
        path = _select(root, "evaluation.rebound_tracker.baseline_path")
        if path:
            return os.path.expanduser(str(path))

        # 3: resilience bundle (defaults match conf/evaluation/resilience_config.yaml).
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
            if os.path.exists(run_local_path):
                return run_local_path
            persistent_fallback = self._resolve_persistent_baseline_fallback()
            if persistent_fallback and os.path.exists(persistent_fallback):
                return persistent_fallback
            return run_local_path
        persistent_fallback = self._resolve_persistent_baseline_fallback()
        if persistent_fallback and os.path.exists(persistent_fallback):
            return persistent_fallback
        # Last resort: cwd-relative filename (may still resolve if user copied the file).
        return str(baseline_json)

    def _load_stage_baselines(self) -> Dict[tuple, Any]:
        """Load and cache the per-anchor stage baselines mapping."""
        path = self._resolve_baseline_path()
        if not path:
            return {}
        if self._stage_baselines is not None and self._stage_baselines_path == path:
            return self._stage_baselines
        try:
            baselines = StageBaselineEstimator.load(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load stage baselines from %s: %s", path, exc)
            baselines = {}
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
        rebound_metrics.baseline_loaded = bool(baselines)
        if not baselines and tracker_window_count > 0:
            logger.warning(
                "Rebound tracker detected %d raw window(s) but no stage baseline "
                "was loaded from %s; formal C_rec will be marked invalid rather than zero.",
                tracker_window_count,
                self._resolve_baseline_path(),
            )
        try:
            episode_metrics.rebound = self.rebound_collector.compute_stage_crec_multidim(
                transitions=planner_infos,
                baselines=baselines,
                tracker_summary=tracker_summary,
                metrics=episode_metrics.rebound,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("compute_stage_crec_multidim failed: %s", exc)

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

        # Persist Degradation metrics
        if episode_metrics.degradation:
            try:
                log_degradation_metrics(
                    degradation_metrics=episode_metrics.degradation,
                    output_dir=output_dir,
                    episode_filename=episode_filename,
                    env_interface=self.env_interface,
                )
            except Exception as exc:
                logger.warning("Failed to persist Degradation metrics: %s", exc)

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
                    critic=critic,
                )
        except Exception as exc:
            logger.error("ADCA Analysis failed: %s", exc)
