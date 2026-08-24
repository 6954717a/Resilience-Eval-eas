from pathlib import Path
import sys
import types

import pytest


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if "habitat_llm" not in sys.modules:
    root_package = types.ModuleType("habitat_llm")
    root_package.__path__ = [str(_PACKAGE_ROOT)]
    sys.modules["habitat_llm"] = root_package
if "habitat_llm.evaluation" not in sys.modules:
    evaluation_package = types.ModuleType("habitat_llm.evaluation")
    evaluation_package.__path__ = [str(_PACKAGE_ROOT / "evaluation")]
    sys.modules["habitat_llm.evaluation"] = evaluation_package


from habitat_llm.evaluation.core.metrics_aggregator import MetricsAggregator
from habitat_llm.evaluation.metrics.collection import EpisodeCollectionContext
from habitat_llm.evaluation.metrics.extensibility import (
    BoundaryCollector,
    BoundaryThresholds,
)
from habitat_llm.evaluation.metrics.stability.stability_collector import (
    StabilityCollector,
)
from habitat_llm.evaluation.monitors import MonitorRegistry, SafetyMonitor


class _Monitor:
    def __init__(self) -> None:
        self.values = []

    def reset(self) -> None:
        self.values.clear()

    def update(self, step_data):
        self.values.append(step_data["value"])
        return {"latest": self.values[-1]}

    def get_summary(self):
        return {
            "count": len(self.values),
            "values": tuple(self.values),
            "nested": {"items": list(self.values)},
        }


class _Result:
    def __init__(self, summary):
        self.summary = summary

    def get_summary_for_csv(self):
        return dict(self.summary)


class _Collector:
    dimension = "custom"

    def __init__(self) -> None:
        self.context = None

    def collect_episode(self, context):
        self.context = context
        return _Result(
            {
                "custom_total_steps": context.total_steps,
                "custom_monitor_count": context.monitor_summary("example").get(
                    "count", -1
                ),
            }
        )


def test_monitor_registry_routes_typed_payloads_and_snapshots():
    monitor = _Monitor()
    registry = MonitorRegistry({"example": monitor})

    assert registry.update_all({"example": {"value": 3}}) == {
        "example": {"latest": 3}
    }
    assert registry.summaries()["example"]["values"] == (3,)

    registry.reset()
    assert registry.summary("example")["count"] == 0
    with pytest.raises(KeyError, match="unregistered"):
        registry.update_all({"unknown": {"value": 1}})


def test_monitor_registry_rejects_incomplete_or_duplicate_monitors():
    registry = MonitorRegistry({"example": _Monitor()})
    with pytest.raises(ValueError, match="already registered"):
        registry.register("example", _Monitor())
    with pytest.raises(TypeError, match="does not implement"):
        registry.register("broken", object())
    mismatched = _Monitor()
    mismatched.monitor_name = "other"
    with pytest.raises(ValueError, match="name mismatch"):
        registry.register("mismatched", mismatched)


def test_generic_collector_receives_summary_not_live_monitor():
    monitor = _Monitor()
    monitor.update({"value": 7})
    registry = MonitorRegistry({"example": monitor})
    collector = _Collector()
    aggregator = MetricsAggregator(
        config={"custom": {"enabled": True}},
        collectors={"custom": collector},
    )

    metrics = aggregator.collect_episode_metrics(
        planner=object(),
        planner_infos=[{"step": 1}],
        critic=None,
        total_steps=9,
        monitor_registry=registry,
    )

    assert metrics.extra_dimensions["custom"].summary["custom_monitor_count"] == 1
    assert not hasattr(collector.context, "monitors")
    with pytest.raises(TypeError):
        collector.context.monitor_summaries["example"]["count"] = 999
    with pytest.raises(AttributeError):
        collector.context.monitor_summary("example")["nested"]["items"].append(999)
    monitor.update({"value": 8})
    assert collector.context.monitor_summary("example")["count"] == 1
    assert aggregator.get_csv_summary(metrics)["custom_total_steps"] == 9


def test_collector_registry_respects_dimension_enable_flag():
    collector = _Collector()
    aggregator = MetricsAggregator(
        config={"custom": {"enabled": False}},
        collectors={"custom": collector},
    )

    metrics = aggregator.collect_episode_metrics(
        planner=object(),
        planner_infos=[],
        critic=None,
        total_steps=1,
    )

    assert metrics.extra_dimensions == {}
    assert collector.context is None


def test_legacy_safety_get_statistics_path_is_preserved():
    class LegacySafety:
        def get_statistics(self):
            return {
                "total_cbf_penalty": 2.0,
                "collision_rate": 0.5,
                "safety_score": 0.0,
                "unsafe_rate": 1.0,
            }

    aggregator = MetricsAggregator(
        stability_collector=StabilityCollector(),
        config={"stability": {"enabled": True}},
    )
    metrics = aggregator.collect_episode_metrics(
        planner=object(),
        planner_infos=[],
        critic=None,
        safety_monitor=LegacySafety(),
        total_steps=1,
    )

    assert metrics.stability.p_cbf == 2.0
    assert metrics.stability.safety_score == 0.0
    assert metrics.stability.unsafe_rate == 1.0


def test_safety_zero_score_is_preserved_and_missing_distance_is_explicit():
    monitor = SafetyMonitor({"enabled": True, "collision_penalty": 1.0})
    monitor.update(
        {
            "agent_collisions": {0: True, 1: True},
            "collision_scope": "agent_pair_collision",
        }
    )
    summary = monitor.get_summary()

    assert summary["safety_score"] == 0.0
    assert summary["safety_valid"] is True
    assert summary["safety_scope"] == "agent_pair_collision"
    assert summary["distance_constraint_available"] is False
    assert summary["distance_constraint_missing_reason"] == "sensor_not_supplied"

    metrics = StabilityCollector().collect_episode(
        EpisodeCollectionContext(
            planner=object(),
            planner_infos=(),
            critic=None,
            total_steps=1,
            monitor_summaries={"safety": summary},
        )
    )
    assert metrics.safety_score == 0.0
    assert metrics.unsafe_rate == 1.0
    assert metrics.distance_constraint_available is False
    assert metrics.distance_constraint_missing_reason == "sensor_not_supplied"
    csv_summary = metrics.get_summary_for_csv()
    assert csv_summary["safety_distance_constraint_available"] is False

    boundary_row = {
        **csv_summary,
        "episode_id": "78",
        "anchor_id": "78",
        "task_family": "placement_only",
        "stress_lambda": 0.5,
        "evolve_step": 0,
        "task_percent_complete": 0.8,
        "proposition_satisfied_fraction": 0.8,
        "rebound_c_rec": 1.0,
        "rebound_c_rec_valid": 1,
        "stability_beta_neighborhood": 0.1,
        "stability_beta_neighborhood_valid": 1,
    }
    boundary = BoundaryCollector(
        BoundaryThresholds(required_safety_channels=("agent_pair_collision",))
    )
    boundary.aggregate([boundary_row])
    evidence = boundary.episode_evidence[0]
    assert evidence.safety_load == 1.0
    assert evidence.metric_sources["unsafe_rate"] == "unsafe_rate"


def test_missing_safety_observations_are_not_reported_as_perfect_safety():
    summary = SafetyMonitor({"enabled": True}).get_summary()

    assert summary["safety_valid"] is False
    assert summary["safety_score"] is None
    assert summary["unsafe_rate"] is None

    row = {
        "episode_id": "missing-safety",
        "task_family": "placement_only",
        "stress_lambda": 0.0,
        "evolve_step": 0,
        "task_percent_complete": 1.0,
        "proposition_satisfied_fraction": 1.0,
        "rebound_c_rec": 0.0,
        "rebound_c_rec_valid": 1,
        "stability_beta_neighborhood": 0.0,
        "stability_beta_neighborhood_valid": 1,
        **StabilityCollector().collect_episode(
            EpisodeCollectionContext(
                planner=object(),
                planner_infos=(),
                critic=None,
                total_steps=0,
                monitor_summaries={"safety": summary},
            )
        ).get_summary_for_csv(),
    }
    collector = BoundaryCollector(
        BoundaryThresholds(required_safety_channels=("agent_pair_collision",))
    )
    collector.aggregate([row])

    assert collector.episode_evidence[0].valid is False
    assert "no_step_observations" in collector.episode_evidence[0].missing_reasons


def test_unscoped_collision_mapping_is_not_treated_as_complete_sensor_data():
    monitor = SafetyMonitor({"enabled": True})
    monitor.update({"agent_collisions": {0: False}})

    summary = monitor.get_summary()
    assert summary["safety_valid"] is False
    assert summary["unsafe_rate"] is None
    assert summary["collision_rate"] is None


def test_stability_reads_replan_count_without_fabricating_step_from_snapshot():
    metrics = StabilityCollector().collect_episode(
        EpisodeCollectionContext(
            planner=object(),
            planner_infos=(
                {
                    "total_step_count": 3,
                    "replanning_count": {0: 2, 1: 1},
                },
            ),
            critic=None,
            total_steps=3,
        )
    )

    assert metrics.n_replan == 2
    assert metrics.replan_steps == []


def test_stability_reads_replan_step_from_counter_transition():
    metrics = StabilityCollector().collect_episode(
        EpisodeCollectionContext(
            planner=object(),
            planner_infos=(
                {
                    "total_step_count": 2,
                    "replanning_count": {0: 1, 1: 1},
                },
                {
                    "total_step_count": 3,
                    "replanning_count": {0: 2, 1: 1},
                },
            ),
            critic=None,
            total_steps=3,
        )
    )

    assert metrics.n_replan == 2
    assert metrics.replan_steps == [3]
