#!/usr/bin/env python3

"""Episode-level collection for the paper's runtime resilience evidence.

Rebound and the single-run Stability diagnostics are collected here.  Formal
neighborhood Stability and Graceful Extensibility are intentionally left to
post-processing because they require multiple clean/perturbed runs and a stress
grid, respectively.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from habitat_llm.evaluation.metrics.collection import (
    CollectorRegistry,
    EpisodeCollectionContext,
)
from habitat_llm.evaluation.monitors.base import MonitorRegistry
from habitat_llm.evaluation.reporting import empty_runtime_metric_summary


@dataclass
class EpisodeMetrics:
    """Runtime results for one episode."""

    rebound: Optional[Any] = None
    stability: Optional[Any] = None
    extra_dimensions: Dict[str, Any] = field(default_factory=dict)

    def set_dimension(self, dimension: str, value: Any) -> None:
        if dimension in {"rebound", "stability"}:
            setattr(self, dimension, value)
        else:
            self.extra_dimensions[dimension] = value

    def values_by_dimension(self) -> Dict[str, Any]:
        values = dict(self.extra_dimensions)
        if self.rebound is not None:
            values["rebound"] = self.rebound
        if self.stability is not None:
            values["stability"] = self.stability
        return values


class MetricsAggregator:
    """Collect registered runtime dimensions without manufacturing values.

    Boundary/GE remains a dataset postprocessor because it requires a stress
    grid.  It is intentionally not forced into this episode-collector API.
    """

    def __init__(
        self,
        rebound_collector: Any = None,
        stability_collector: Any = None,
        config: Optional[Dict[str, Any]] = None,
        *,
        collectors: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.rebound_collector = rebound_collector
        self.stability_collector = stability_collector
        self.config = config or {}
        self.collectors = CollectorRegistry()
        if rebound_collector is not None:
            self.collectors.register("rebound", rebound_collector)
        if stability_collector is not None:
            self.collectors.register("stability", stability_collector)
        for dimension, collector in (collectors or {}).items():
            self.collectors.register(
                dimension,
                collector,
                replace=self.collectors.get(dimension) is not None,
            )
        self.rebound_collector = self.collectors.get("rebound")
        self.stability_collector = self.collectors.get("stability")

    def collect_episode_metrics(
        self,
        planner: Any,
        planner_infos: List[Dict[str, Any]],
        critic: Optional[Any],
        safety_monitor: Any = None,
        total_steps: int = 0,
        *,
        monitor_registry: Optional[MonitorRegistry] = None,
        monitor_summaries: Optional[Mapping[str, Mapping[str, Any]]] = None,
        info: Optional[Mapping[str, Any]] = None,
    ) -> EpisodeMetrics:
        """Collect one rollout's Rebound and Stability evidence.

        Collection failures are evaluation-integrity failures and therefore
        propagate to the caller instead of being silently converted to zeros.
        """

        summaries: Dict[str, Mapping[str, Any]] = {
            str(name): dict(summary or {})
            for name, summary in (monitor_summaries or {}).items()
        }
        if monitor_registry is not None:
            summaries.update(monitor_registry.summaries())
        if safety_monitor is not None and "safety" not in summaries:
            get_summary = getattr(safety_monitor, "get_summary", None)
            if not callable(get_summary):
                get_summary = getattr(safety_monitor, "get_statistics", None)
            if not callable(get_summary):
                raise TypeError(
                    "safety_monitor exposes neither get_summary() nor "
                    "get_statistics()"
                )
            summaries["safety"] = dict(get_summary() or {})

        context = EpisodeCollectionContext(
            planner=planner,
            planner_infos=tuple(planner_infos),
            critic=critic,
            total_steps=int(total_steps),
            monitor_summaries=summaries,
            info=dict(info or {}),
        )
        enabled = {
            dimension: self._is_enabled(dimension)
            for dimension in self.collectors.as_mapping()
        }
        collected = self.collectors.collect(context, enabled=enabled)
        metrics = EpisodeMetrics()
        for dimension, value in collected.items():
            metrics.set_dimension(dimension, value)
        return metrics

    def _is_enabled(self, dimension: str) -> bool:
        return bool(self.config.get(dimension, {}).get("enabled", True))

    def get_csv_summary(self, metrics: EpisodeMetrics) -> Dict[str, Any]:
        """Flatten runtime results using explicit missing values."""

        summary = empty_runtime_metric_summary()
        for value in metrics.values_by_dimension().values():
            get_csv_summary = getattr(value, "get_summary_for_csv", None)
            if callable(get_csv_summary):
                summary.update(get_csv_summary())
            elif isinstance(value, Mapping):
                summary.update(value)
        return summary


__all__ = ["EpisodeMetrics", "MetricsAggregator"]
