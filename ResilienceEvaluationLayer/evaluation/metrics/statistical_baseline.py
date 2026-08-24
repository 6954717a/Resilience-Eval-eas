#!/usr/bin/env python3

"""Reusable statistical references for resilience metrics.

``StageBaseline`` remains the stage-local reference needed by Rebound and
Stability.  This module supplies the common statistical layer for references
whose natural key is different, such as GE's fixed ``task_family``
clean boundary.  The representation is intentionally metric-agnostic so new
resilience dimensions can reuse the same sample-count, quantile, provenance,
and persistence semantics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union


def empirical_quantile(values: Sequence[float], q: float) -> float:
    """Return the linearly interpolated empirical quantile."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    q = min(max(float(q), 0.0), 1.0)
    index = q * (len(ordered) - 1)
    lower_index = int(math.floor(index))
    upper_index = int(math.ceil(index))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = index - lower_index
    return (
        ordered[lower_index] * (1.0 - weight)
        + ordered[upper_index] * weight
    )


@dataclass(frozen=True)
class MetricReference:
    """A clean-sample distribution summary for one named metric."""

    metric: str
    direction: str
    n: int
    mean: float
    std: float
    lower_quantile: float
    median: float
    upper_quantile: float
    quantile_alpha: float

    @classmethod
    def estimate(
        cls,
        metric: str,
        direction: str,
        values: Iterable[float],
        quantile_alpha: float,
    ) -> "MetricReference":
        samples = [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ]
        alpha = min(max(float(quantile_alpha), 0.0), 0.5)
        n = len(samples)
        mean = sum(samples) / n if n else 0.0
        variance = (
            sum((value - mean) ** 2 for value in samples) / (n - 1)
            if n >= 2
            else 0.0
        )
        return cls(
            metric=str(metric),
            direction=str(direction),
            n=n,
            mean=float(mean),
            std=float(math.sqrt(max(variance, 0.0))),
            lower_quantile=empirical_quantile(samples, alpha),
            median=empirical_quantile(samples, 0.5),
            upper_quantile=empirical_quantile(samples, 1.0 - alpha),
            quantile_alpha=alpha,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "n": int(self.n),
            "mean": float(self.mean),
            "std": float(self.std),
            "lower_quantile": float(self.lower_quantile),
            "median": float(self.median),
            "upper_quantile": float(self.upper_quantile),
            "quantile_alpha": float(self.quantile_alpha),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricReference":
        return cls(
            metric=str(payload.get("metric", "")),
            direction=str(payload.get("direction", "higher_is_better")),
            n=int(payload.get("n", 0)),
            mean=float(payload.get("mean", 0.0)),
            std=float(payload.get("std", 0.0)),
            lower_quantile=float(payload.get("lower_quantile", 0.0)),
            median=float(payload.get("median", 0.0)),
            upper_quantile=float(payload.get("upper_quantile", 0.0)),
            quantile_alpha=float(payload.get("quantile_alpha", 0.25)),
        )


@dataclass
class StatisticalBaseline:
    """A Judge-scoped statistical reference at an explicit semantic scope."""

    scope: str
    key: Dict[str, Any]
    references: Dict[str, MetricReference]
    judge_model: str = "unconfigured-judge"
    judge_base_url: str = ""
    min_samples: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        counts = [reference.n for reference in self.references.values()]
        return min(counts) if counts else 0

    @property
    def valid(self) -> bool:
        return bool(self.references) and self.n_samples >= int(self.min_samples)

    @property
    def missing_reason(self) -> str:
        if not self.references:
            return "clean_reference_missing"
        if not self.valid:
            return "insufficient_clean_reference_samples"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "key": dict(self.key),
            "judge_model": self.judge_model,
            "judge_base_url": self.judge_base_url,
            "min_samples": int(self.min_samples),
            "n_samples": int(self.n_samples),
            "valid": int(self.valid),
            "missing_reason": self.missing_reason,
            "references": {
                name: reference.to_dict()
                for name, reference in sorted(self.references.items())
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StatisticalBaseline":
        raw_references = payload.get("references") or {}
        references = {
            str(name): MetricReference.from_dict(reference)
            for name, reference in raw_references.items()
            if isinstance(reference, Mapping)
        }
        return cls(
            scope=str(payload.get("scope", "")),
            key=dict(payload.get("key") or {}),
            references=references,
            judge_model=str(payload.get("judge_model", "unconfigured-judge")),
            judge_base_url=str(payload.get("judge_base_url", "")),
            min_samples=int(payload.get("min_samples", 2)),
            metadata=dict(payload.get("metadata") or {}),
        )


def write_statistical_baselines(
    baselines: Mapping[Tuple[str, int], StatisticalBaseline],
    output_path: Union[str, Path],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist a collection without conflating runtime evidence and estimates."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "baseline_kind": "resilience_statistical_reference",
        "metadata": dict(metadata or {}),
        "baselines": [
            baseline.to_dict()
            for _, baseline in sorted(
                baselines.items(), key=lambda item: (str(item[0][0]), int(item[0][1]))
            )
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def read_statistical_baselines(
    input_path: Union[str, Path],
) -> Dict[Tuple[str, int], StatisticalBaseline]:
    path = Path(input_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result: Dict[Tuple[str, int], StatisticalBaseline] = {}
    for raw in payload.get("baselines", []):
        if not isinstance(raw, Mapping):
            continue
        baseline = StatisticalBaseline.from_dict(raw)
        family = str(baseline.key.get("task_family", "other"))
        reference_step = int(
            baseline.key.get(
                "reference_evolve_step", baseline.key.get("evolve_step", 0)
            )
        )
        result[(family, reference_step)] = baseline
    return result
