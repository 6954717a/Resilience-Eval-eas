#!/usr/bin/env python3

"""Generic episode-collection contracts shared by metric dimensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True)
class EpisodeCollectionContext:
    """Stable input boundary between the runner and metric collectors."""

    planner: Any
    planner_infos: Sequence[Mapping[str, Any]]
    critic: Optional[Any]
    total_steps: int
    monitor_summaries: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        planner_infos = tuple(_freeze_value(item) for item in self.planner_infos)
        summaries = MappingProxyType(
            {
                str(name): _freeze_value(summary or {})
                for name, summary in self.monitor_summaries.items()
            }
        )
        object.__setattr__(self, "planner_infos", planner_infos)
        object.__setattr__(self, "monitor_summaries", summaries)
        object.__setattr__(self, "info", _freeze_value(self.info))

    def monitor_summary(self, name: str) -> Dict[str, Any]:
        return dict(self.monitor_summaries.get(str(name), {}) or {})


@runtime_checkable
class EpisodeMetricCollector(Protocol):
    """Collector invoked once at the end of an episode."""

    dimension: str

    def collect_episode(self, context: EpisodeCollectionContext) -> Any:
        ...


class CollectorRegistry:
    """Dimension-keyed registry for heterogeneous episode collectors."""

    def __init__(
        self,
        collectors: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._collectors: Dict[str, Any] = {}
        for dimension, collector in (collectors or {}).items():
            self.register(dimension, collector)

    def register(
        self,
        dimension: str,
        collector: Any,
        *,
        replace: bool = False,
    ) -> Any:
        key = str(dimension).strip()
        if not key:
            raise ValueError("Collector dimension cannot be empty")
        if key in self._collectors and not replace:
            raise ValueError(f"Collector already registered: {key}")
        if not callable(getattr(collector, "collect_episode", None)):
            raise TypeError(
                f"Collector {key!r} does not implement collect_episode()"
            )
        declared = str(getattr(collector, "dimension", key))
        if declared != key:
            raise ValueError(
                f"Collector dimension mismatch: registered={key}, declared={declared}"
            )
        self._collectors[key] = collector
        return collector

    def get(self, dimension: str, default: Any = None) -> Any:
        return self._collectors.get(str(dimension), default)

    def collect(
        self,
        context: EpisodeCollectionContext,
        enabled: Optional[Mapping[str, bool]] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for dimension, collector in self._collectors.items():
            if enabled is not None and not bool(enabled.get(dimension, True)):
                continue
            results[dimension] = collector.collect_episode(context)
        return results

    def as_mapping(self) -> Mapping[str, Any]:
        return dict(self._collectors)


__all__ = [
    "CollectorRegistry",
    "EpisodeCollectionContext",
    "EpisodeMetricCollector",
]
