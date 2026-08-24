import json
import math
from pathlib import Path
import sys
import types

import pytest
import numpy as np


# Keep the focused metric tests independent of the Habitat simulator import.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if "habitat_llm" not in sys.modules:
    root_package = types.ModuleType("habitat_llm")
    root_package.__path__ = [str(_PACKAGE_ROOT)]
    sys.modules["habitat_llm"] = root_package
if "habitat_llm.evaluation" not in sys.modules:
    evaluation_package = types.ModuleType("habitat_llm.evaluation")
    evaluation_package.__path__ = [str(_PACKAGE_ROOT / "evaluation")]
    sys.modules["habitat_llm.evaluation"] = evaluation_package
if "habitat_llm.evaluation.metrics" not in sys.modules:
    metrics_package = types.ModuleType("habitat_llm.evaluation.metrics")
    metrics_package.__path__ = [str(_PACKAGE_ROOT / "evaluation" / "metrics")]
    sys.modules["habitat_llm.evaluation.metrics"] = metrics_package


from habitat_llm.evaluation.metrics.rebound import (
    CogLoadTracker,
    ReboundCollector,
    ReboundMetrics,
    StageAnchor,
    StageBaseline,
    StageBaselineDim,
    StageBaselineEstimator,
)
from habitat_llm.evaluation.metrics.rebound.rebound_collector import (
    _select_baseline_with_level,
)
from habitat_llm.evaluation.metrics.rebound.formal_cost import (
    bounded_recovery_cost_index,
    recovery_intensity,
    regularized_mahalanobis,
)
from habitat_llm.evaluation.metrics.rebound.recovery_features import (
    build_recovery_feature_groups,
    normalized_cognitive_excess,
)
from habitat_llm.evaluation.metrics.stability import StabilityCollector, StabilityMetrics
from habitat_llm.evaluation.stage_baseline.stage_baseline import (
    RECOVERY_POOL_MODALITY,
    STABILITY_POOL_MODALITY,
    _oas_from_sufficient_statistics,
    _recovery_sufficient_statistics,
)
from habitat_llm.evaluation.metrics.stability.trajectory_loss import (
    compute_stage_trajectory_loss,
)
from habitat_llm.evaluation.reporting import primary_display_items


def _critic_record(index: int, *, value_scale: float = 1.0) -> dict:
    return {
        "episode_id": "78",
        "transition_index": index,
        "state_kind": "state",
        "task_family": "other",
        "predicted_value": value_scale * float(index),
        "td_error": value_scale * float(index + 1) / 10.0,
        "gae_advantage": value_scale * float(index + 1) / 5.0,
        "task_percent_complete": 0.0,
        "proposition_satisfied_fraction": 0.0,
        "planning_transition": bool(index % 2),
        "analysis_tags": ["replan_anchor"] if index % 2 else [],
        "info_snapshot": {
            "episode_id": "78",
            "task_family": "other",
            "task_percent_complete": 0.0,
            "proposition_satisfied_fraction": 0.0,
            "proposition_total": 1,
            "proposition_satisfied_count": 0,
            "planning_step_count": index,
            "sim_step_count": index,
        },
    }


def _full_export_payload(records: list[dict]) -> dict:
    count = len(records)
    return {
        "episode_id": "78",
        "task_family": "other",
        "export_manifest": {
            "full_stability_trajectory": True,
            "minimal_schema": True,
            "record_mode": "state_only",
        },
        "selection_summary": {
            "total_transitions": count,
            "selected_transitions": count,
        },
        "records": records,
    }


def _clean_multimodal_records(
    episode_id: str,
    *,
    sim_stride: int,
) -> list[dict]:
    records = [_critic_record(index) for index in range(4)]
    actions = (None, "Navigate[target]", "Pick[target]", None)
    for index, (record, action) in enumerate(zip(records, actions)):
        progress = 1.0 if index == len(records) - 1 else 0.0
        record["episode_id"] = str(episode_id)
        record["task_percent_complete"] = progress
        record["proposition_satisfied_fraction"] = progress
        record["info_snapshot"].update(
            {
                "episode_id": str(episode_id),
                "task_percent_complete": progress,
                "proposition_satisfied_fraction": progress,
                "proposition_total": 1,
                "proposition_satisfied_count": 1 if progress == 1.0 else 0,
                "planning_step_count": index,
                "sim_step_count": index * int(sim_stride),
            }
        )
        if action is not None:
            record["recent_action"] = action
            record["info_snapshot"]["recent_action"] = action
    records[-1]["task_state_success"] = 1.0
    records[-1]["info_snapshot"]["task_state_success"] = 1.0
    return records


def test_invalid_formal_runtime_metrics_are_missing_not_zero():
    rebound = ReboundMetrics(
        c_rec=1.0,
        c_rec_index_100=50.0,
        c_rec_valid=False,
    )
    rebound_summary = rebound.get_summary_for_csv()
    assert rebound_summary["rebound_c_rec"] is None
    assert rebound_summary["rebound_c_rec_index_100"] is None

    stability = StabilityMetrics()
    stability.finalize_episode_proxy()
    stability_summary = stability.get_summary_for_csv()
    assert "stability_beta" not in stability_summary
    assert stability_summary["stability_episode_proxy"] is None

    display = dict(primary_display_items(rebound_summary))
    assert str(display["rebound_c_rec"]).startswith("N/A")
    assert set(display) == {"rebound_c_rec", "ge_boundary_lambda_star"}


def test_primary_display_has_one_headline_per_resilience_dimension():
    display = primary_display_items(
        {
            "task_state_success": 1.0,
            "task_percent_complete": 1.0,
            "sim_step_count": 123,
            "replanning_count_1": 4,
            "replanning_count_0": 3,
            "rebound_c_rec": 0.4,
            "rebound_c_rec_valid": 1,
            "rebound_c_rec_cog": 0.2,
            "stability_beta_family": 0.15,
            "stability_beta_family_valid": 1,
            "stability_beta_neighborhood": 0.12,
            "stability_beta_neighborhood_valid": 1,
            "ge_boundary_lambda_star": 0.8,
            "ge_capacity_valid": 1,
            "ge_boundary_margin": 0.1,
            "ge_audc_f": 0.9,
        }
    )
    assert [key for key, _ in display] == [
        "task_state_success",
        "task_percent_complete",
        "sim_step_count",
        "replanning_count_0",
        "replanning_count_1",
        "rebound_c_rec",
        "stability_beta_family",
        "ge_boundary_lambda_star",
    ]


def test_censored_recovery_window_makes_formal_c_rec_unavailable():
    metrics = ReboundCollector({}).compute_stage_crec_multidim(
        transitions=[],
        baselines={
            ("other", 0, "planning"): StageBaseline(
                family="other", stage=0, modality="planning"
            )
        },
        tracker_summary={
            "windows": [
                {
                    "t_d": 2,
                    "t_r": 5,
                    "recovered": False,
                    "closure_reason": "episode_tail_censored",
                    "anchor_t_d": {
                        "family": "other",
                        "stage": 0,
                        "modality": "planning",
                    },
                }
            ]
        },
    )

    assert metrics.c_rec_valid is False
    assert metrics.c_rec_missing_reason == "episode_tail_censored"
    assert metrics.formal_window_count == 0
    assert metrics.skipped_window_count == 1


def test_rebound_eq2_uses_covariance_norm_and_channel_rms():
    magnitude = regularized_mahalanobis(
        [3.0, 4.0],
        [[1.0, 0.0], [0.0, 1.0]],
        sample_count=4,
        regularizer=0.0,
    )
    assert abs(magnitude - 5.0 / math.sqrt(2.0)) < 1.0e-9
    assert abs(
        recovery_intensity([3.0, 4.0, 0.0]) - 5.0 / (3.0**0.5)
    ) < 1.0e-12
    assert regularized_mahalanobis(
        [1.0], [[1.0]], sample_count=1, regularizer=0.0
    ) == pytest.approx(1.0)


def test_rebound_features_are_mean_centered_one_sided_and_bounded():
    assert normalized_cognitive_excess(1.0, 0.0, 0.0) == 1.0
    assert normalized_cognitive_excess(3.0, 2.0, 0.5) == 0.25

    magnitude = regularized_mahalanobis(
        [3.0, 4.0],
        [[1.0, 0.0], [0.0, 1.0]],
        mean=[1.0, 5.0],
        sample_count=4,
        regularizer=0.0,
    )
    # delta=[2,-1] -> one-sided [2,0], then divide d^2 by k=2.
    assert magnitude is not None
    assert abs(magnitude - math.sqrt(2.0)) < 1.0e-9
    assert (
        regularized_mahalanobis(
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            mean=[],
            sample_count=4,
            regularizer=1.0e-3,
        )
        is None
    )
    assert bounded_recovery_cost_index(0.0) == 0.0
    assert bounded_recovery_cost_index(1.0) == 50.0
    assert bounded_recovery_cost_index(9.0) == 90.0
    assert bounded_recovery_cost_index(None) is None


def test_rebound_index_is_monotone_and_strictly_below_100():
    raw = [0.0, 1.0e-9, 0.1, 1.0, 9.0, 1.0e6]
    indices = [bounded_recovery_cost_index(value) for value in raw]

    assert all(value is not None for value in indices)
    numeric = [float(value) for value in indices if value is not None]
    assert all(0.0 <= value < 100.0 for value in numeric)
    assert all(left < right for left, right in zip(numeric, numeric[1:]))
    assert 0.0 <= bounded_recovery_cost_index(1.0e300) < 100.0
    assert bounded_recovery_cost_index(-1.0) is None
    assert bounded_recovery_cost_index(float("inf")) is None
    assert bounded_recovery_cost_index(float("nan")) is None


def test_shared_recovery_feature_builder_matches_all_three_channels():
    groups = build_recovery_feature_groups(
        volumes={"pln": 1.0, "per": 3.0, "coord": 0.0},
        baseline_volumes={"pln": 0.0, "per": 2.0, "coord": 0.0},
        validities={"pln": 0.0, "per": 0.5, "coord": 1.0},
        plan_work=2.0,
        sim_work=3.0,
        plan_baseline=1.0,
        sim_baseline=2.0,
        idle_gap=0.25,
        redo_event=True,
        progress_debt=0.2,
        goal_debt=0.1,
        risk=0.3,
        progress_baseline=0.1,
        goal_baseline=0.2,
        risk_baseline=0.1,
    )

    assert groups["cog"] == pytest.approx([1.0, 0.25, 0.0])
    assert groups["phy"] == pytest.approx([1.0, 0.75, 1.0])
    assert groups["debt"] == pytest.approx([2.0, 0.5, 2.0])


def test_cognitive_validity_uses_first_future_observation_without_step_offset():
    tracker = CogLoadTracker(lookahead_L=1)
    tracker.observe_step(1, {"replanned": {0: True}})
    tracker.observe_step(2, {"replanned": {0: False}})
    tracker.bind_validity([0.0, 1.0], [0.0, 0.0])

    assert tracker.records()[0].get("pln").validity == 1.0


def test_wait_is_edge_deduplicated_idle_not_coordination_load():
    tracker = CogLoadTracker()
    first = tracker.observe_step(
        0,
        {"high_level_actions": {"0": "Wait[]"}},
        modality="coordination",
    )
    repeated = tracker.observe_step(
        1,
        {"high_level_actions": {"0": "Wait[]"}},
        modality="coordination",
    )
    coordinated = tracker.observe_step(
        2,
        {"high_level_actions": {"0": "Coordinate[agent_1]"}},
        modality="coordination",
    )
    repeated_coordination = tracker.observe_step(
        3,
        {"high_level_actions": {"0": "Coordinate[agent_1]"}},
        modality="coordination",
    )

    assert first.get("coord").volume == 0.0
    assert repeated.get("coord").volume == 0.0
    assert coordinated.get("coord").volume == 1.0
    assert repeated_coordination.get("coord").volume == 0.0


def test_wait_contributes_to_physical_gap_not_cognitive_coordination():
    identity = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    baseline = StageBaseline(
        family="other",
        stage=0,
        modality="planning",
        tau_star=1.0,
        delta_p_bar=1.0,
        delta_q_bar=1.0,
        W_rem_seg_bar=4.0,
        W_rem_star_swe=4.0,
        by_dim={
            "plan": StageBaselineDim(tau_bar=1.0, W_rem_seg_bar=4.0),
            "sim": StageBaselineDim(tau_bar=1.0, W_rem_seg_bar=4.0),
            "runtime": StageBaselineDim(tau_bar=0.0, W_rem_seg_bar=0.0),
        },
        v_bar_channels={"pln": 0.0, "per": 0.0, "coord": 0.0},
        recovery_mean={
            "cog": [0.0, 0.0, 0.0],
            "phy": [0.0, 0.0, 0.0],
            "debt": [0.0, 0.0, 0.0],
        },
        recovery_covariance={
            "cog": identity,
            "phy": identity,
            "debt": identity,
        },
        recovery_covariance_n_samples=4,
        trajectory_ids=["clean-78-seed0"],
        trajectory_provenance={
            "clean-78-seed0": {
                "episode_id": "78",
                "seed": 0,
                "source_hash": "hash-78-0",
            }
        },
        remaining_work_by_trajectory={"clean-78-seed0": 4.0},
        W_rem_star_by_episode_stage={"78": {0: 4.0}},
        W_rem_support_by_episode_stage={"78": {0: 3}},
    )
    anchor = {
        "family": "other",
        "stage": 0,
        "modality": "planning",
        "progress": 0.0,
    }
    metrics = ReboundCollector({}).compute_stage_crec_multidim(
        transitions=[
            {
                "planning_step_count": 0,
                "sim_step_count": 0,
                "task_percent_complete": 0.0,
                "proposition_satisfied_fraction": 0.0,
            },
            {
                "planning_step_count": 1,
                "sim_step_count": 0,
                "task_percent_complete": 0.0,
                "proposition_satisfied_fraction": 0.0,
                "high_level_actions": {"0": "Wait[]"},
            },
        ],
        baselines={("other", 0, "planning"): baseline},
        tracker_summary={
            "episode_id": "78",
            "windows": [
                {
                    "t_d": 1,
                    "t_r": 1,
                    "recovered": True,
                    "anchor_t_d": anchor,
                    "anchor_t_r": anchor,
                }
            ],
            "anchors": [anchor, anchor],
            "cognitive_steps": [
                {
                    "step": 1,
                    "channels": {
                        "coord": {"volume": 0.0, "validity": 0.0}
                    },
                }
            ],
        },
    )

    assert metrics.c_rec_valid is True
    assert metrics.c_rec_cog_coord == 0.0
    assert metrics.stage_recovery_records[0].W_rem_star == 4.0
    assert metrics.stage_recovery_records[0].phy_swe_gap == 1.0

    missing_episode = ReboundCollector({}).compute_stage_crec_multidim(
        transitions=[
            {
                "planning_step_count": 0,
                "sim_step_count": 0,
                "task_percent_complete": 0.0,
                "proposition_satisfied_fraction": 0.0,
            },
            {
                "planning_step_count": 1,
                "sim_step_count": 0,
                "task_percent_complete": 0.0,
                "proposition_satisfied_fraction": 0.0,
                "high_level_actions": {"0": "Wait[]"},
            },
        ],
        baselines={("other", 0, "planning"): baseline},
        tracker_summary={
            "episode_id": "99",
            "windows": [
                {
                    "t_d": 1,
                    "t_r": 1,
                    "recovered": True,
                    "anchor_t_d": anchor,
                    "anchor_t_r": anchor,
                }
            ],
            "anchors": [anchor, anchor],
        },
    )
    assert missing_episode.c_rec_valid is False
    assert (
        missing_episode.c_rec_missing_reason
        == "episode_stage_remaining_work_missing"
    )


def test_stage_baseline_contains_paper_stability_moments_and_filters_next_rows():
    state_records = [_critic_record(index) for index in range(5)]
    interleaved = []
    for record in state_records:
        interleaved.append(record)
        interleaved.append(
            {
                **record,
                "state_kind": "next_state",
                "predicted_value": 999.0,
                "td_error": None,
                "gae_advantage": None,
            }
        )

    estimator = StageBaselineEstimator()
    baselines, by_episode, stats = estimator.estimate_grouped_from_exports([])
    assert baselines == {}
    assert by_episode == {}
    assert stats["total_records"] == 0

    accumulators = estimator.estimate_from_episode(interleaved, family_override="other")
    finalized = {key: acc.finalize(key) for key, acc in accumulators.items()}
    assert finalized
    baseline = next(iter(finalized.values()))
    assert baseline.stability_n_samples == len(state_records) - 1
    assert len(baseline.stability_mean) == 6
    assert len(baseline.stability_covariance) == 6
    assert len(baseline.stability_covariance[0]) == 6
    assert set(baseline.recovery_mean) == {"cog", "phy", "debt"}
    assert set(baseline.recovery_covariance) == {"cog", "phy", "debt"}
    assert baseline.recovery_covariance_n_samples == len(state_records) - 1


def test_formal_beta_uses_stage_trajectory_loss_supremum(tmp_path):
    clean_records = [_critic_record(index) for index in range(6)]
    estimator = StageBaselineEstimator()
    accumulators = estimator.estimate_from_episode(
        clean_records, family_override="other"
    )
    baselines = {key: acc.finalize(key) for key, acc in accumulators.items()}
    for baseline in baselines.values():
        baseline.formal_valid = True
        baseline.formal_support_scope = "exact_stage"
        baseline.formal_missing_reason = ""

    clean_path = tmp_path / "clean.json"
    perturbed_path = tmp_path / "perturbed.json"
    clean_path.write_text(
        json.dumps(_full_export_payload(clean_records)),
        encoding="utf-8",
    )
    perturbed_records = [
        _critic_record(index, value_scale=1.8) for index in range(6)
    ]
    perturbed_path.write_text(
        json.dumps(_full_export_payload(perturbed_records)),
        encoding="utf-8",
    )

    clean_loss = compute_stage_trajectory_loss(clean_path, baselines)
    perturbed_loss = compute_stage_trajectory_loss(perturbed_path, baselines)
    assert clean_loss.valid and perturbed_loss.valid

    collector = StabilityCollector()
    aggregates, _ = collector.aggregate_beta_over_anchors(
        [
            {
                "anchor_id": "78",
                "task_family": "other",
                "perturbation_type": "clean",
                "stability_trajectory_loss": clean_loss.loss,
                "stability_trajectory_loss_valid": 1,
                "action_sequence": "Navigate|Pick",
                "task_state_success": 1.0,
                "task_percent_complete": 1.0,
            },
            {
                "anchor_id": "78",
                "task_family": "other",
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 0.9,
                "perturbation_realized": 0.5,
                "perturbation_valid": 1,
                "stability_trajectory_loss": perturbed_loss.loss,
                "stability_trajectory_loss_valid": 1,
                "action_sequence": "Navigate|Wait|Pick",
                "task_state_success": 1.0,
                "task_percent_complete": 1.0,
            },
        ],
    )
    aggregate = aggregates["78"]
    expected = abs(perturbed_loss.loss - clean_loss.loss) / (0.5 + 1.0e-6)
    assert aggregate.beta_neighborhood_valid
    assert abs(aggregate.beta_hat - expected) < 1.0e-9
    assert aggregate.beta_hat == aggregate.beta_trajectory
    family = collector.aggregate_beta_by_family(aggregates)["other"]
    assert family.beta_family_valid
    assert family.beta_family == aggregate.beta_hat


def test_stage_trajectory_loss_uses_available_baseline_statistics(tmp_path):
    records = [_critic_record(index) for index in range(6)]
    accumulators = StageBaselineEstimator().estimate_from_episode(
        records, family_override="other"
    )
    baselines = {key: acc.finalize(key) for key, acc in accumulators.items()}
    for baseline in baselines.values():
        baseline.formal_valid = False
        baseline.formal_missing_reason = "baseline_diagnostic_only"

    export_path = tmp_path / "diagnostic_baseline_export.json"
    export_path.write_text(
        json.dumps(_full_export_payload(records)), encoding="utf-8"
    )
    result = compute_stage_trajectory_loss(export_path, baselines)

    assert result.valid is True
    assert result.loss is not None
    assert result.missing_reason == ""


def test_stage_trajectory_loss_uses_formal_stage_pool_for_diagnostic_modality(
    tmp_path,
):
    records = [_critic_record(index) for index in range(6)]
    accumulators = StageBaselineEstimator().estimate_from_episode(
        records, family_override="other"
    )
    baselines = {key: acc.finalize(key) for key, acc in accumulators.items()}
    exact = baselines[("other", 0, "planning")]
    # Rebound may promote this sparse exact modality through hierarchical
    # recovery components, but its exact Stability moments still must not
    # outrank the dedicated family x stage Stability pool.
    exact.formal_valid = True
    exact.formal_support_scope = "hierarchical_component_pool"
    exact.formal_missing_reason = ""

    stage_pool = StageBaseline.from_dict(exact.to_dict())
    stage_pool.modality = "__stability__"
    stage_pool.formal_valid = True
    stage_pool.formal_support_scope = "family_stage_stability_pool"
    stage_pool.formal_missing_reason = ""
    stage_pool.formal_min_trajectories = 1
    stage_pool.formal_min_transitions = 2
    stage_pool.n_trajectories = 1
    stage_pool.formal_component_support = {
        "stability": {
            "scope": "family_stage_stability_pool",
            "valid": True,
            "source_modality": "__stability__",
            "family": "other",
            "stage": 0,
            "n_trajectories": 1,
            "n_transitions": stage_pool.stability_n_samples,
            "min_trajectories": 1,
            "min_transitions": 2,
        }
    }
    baselines[("other", 0, "__stability__")] = stage_pool

    export_path = tmp_path / "formal_stage_pool_export.json"
    export_path.write_text(
        json.dumps(_full_export_payload(records)), encoding="utf-8"
    )
    result = compute_stage_trajectory_loss(export_path, baselines)

    assert result.valid is True
    assert result.loss is not None
    assert result.used_transitions == 5
    assert result.candidate_transitions == 5
    assert result.coverage == 1.0

    stage_pool.formal_component_support["stability"]["valid"] = False
    still_available = compute_stage_trajectory_loss(export_path, baselines)
    assert still_available.valid is True


def test_stage_trajectory_loss_backtracks_terminal_records_to_last_stage(tmp_path):
    baseline_records = [_critic_record(index) for index in range(6)]
    accumulators = StageBaselineEstimator().estimate_from_episode(
        baseline_records, family_override="other"
    )
    baselines = {key: acc.finalize(key) for key, acc in accumulators.items()}
    for baseline in baselines.values():
        baseline.formal_valid = True
        baseline.formal_support_scope = "exact_stage"
        baseline.formal_missing_reason = ""

    evaluation_records = [_critic_record(index) for index in range(6)]
    for record in evaluation_records[-2:]:
        record["task_percent_complete"] = 1.0
        record["proposition_satisfied_fraction"] = 1.0
        record["info_snapshot"].update(
            {
                "task_percent_complete": 1.0,
                "proposition_satisfied_fraction": 1.0,
                "proposition_satisfied_count": 1,
            }
        )

    export_path = tmp_path / "terminal_backtrack_export.json"
    export_path.write_text(
        json.dumps(_full_export_payload(evaluation_records)), encoding="utf-8"
    )
    result = compute_stage_trajectory_loss(export_path, baselines)

    assert result.valid is True
    assert result.loss is not None
    assert result.used_transitions == 5
    assert result.candidate_transitions == 5
    assert result.coverage == 1.0


def test_stage_trajectory_loss_excludes_leading_unknown_stage_from_coverage(
    tmp_path,
):
    baseline_records = [_critic_record(index) for index in range(5)]
    accumulators = StageBaselineEstimator().estimate_from_episode(
        baseline_records, family_override="other"
    )
    baselines = {key: acc.finalize(key) for key, acc in accumulators.items()}
    for baseline in baselines.values():
        baseline.formal_valid = True
        baseline.formal_support_scope = "exact_stage"
        baseline.formal_missing_reason = ""

    evaluation_records = [_critic_record(index) for index in range(5)]
    for record in evaluation_records[:2]:
        record["info_snapshot"].pop("proposition_total")
        record["info_snapshot"].pop("proposition_satisfied_count")

    export_path = tmp_path / "leading_unknown_export.json"
    export_path.write_text(
        json.dumps(_full_export_payload(evaluation_records)), encoding="utf-8"
    )
    result = compute_stage_trajectory_loss(export_path, baselines)

    assert result.valid is True
    assert result.used_transitions == 3
    assert result.candidate_transitions == 3
    assert result.coverage == 1.0


def test_formal_trajectory_loss_requires_complete_export_and_full_coverage(tmp_path):
    clean_records = [_critic_record(index) for index in range(6)]
    estimator = StageBaselineEstimator()
    accumulators = estimator.estimate_from_episode(
        clean_records, family_override="other"
    )
    baselines = {key: acc.finalize(key) for key, acc in accumulators.items()}
    for baseline in baselines.values():
        baseline.formal_valid = True
        baseline.formal_support_scope = "exact_stage"
        baseline.formal_missing_reason = ""

    sparse_path = tmp_path / "sparse.json"
    sparse_payload = _full_export_payload(clean_records[::2])
    sparse_payload["selection_summary"] = {
        "total_transitions": len(clean_records),
        "selected_transitions": len(clean_records[::2]),
    }
    sparse_path.write_text(json.dumps(sparse_payload), encoding="utf-8")
    sparse_loss = compute_stage_trajectory_loss(sparse_path, baselines)
    assert not sparse_loss.valid
    assert sparse_loss.loss is None
    assert sparse_loss.missing_reason == "critic_export_selection_incomplete"

    incomplete_features = [dict(record) for record in clean_records]
    incomplete_features[3].pop("td_error")
    incomplete_path = tmp_path / "incomplete_features.json"
    incomplete_path.write_text(
        json.dumps(_full_export_payload(incomplete_features)),
        encoding="utf-8",
    )
    strict = compute_stage_trajectory_loss(incomplete_path, baselines)
    assert strict.valid
    assert strict.loss is not None
    assert strict.used_transitions == 4
    assert strict.candidate_transitions == 5
    assert strict.coverage == 0.8
    assert strict.missing_reason == ""

    relaxed = compute_stage_trajectory_loss(
        incomplete_path,
        baselines,
        minimum_coverage=0.8,
    )
    assert relaxed.valid
    assert relaxed.loss is not None


def test_stage_baseline_skips_invalid_export_with_reason(tmp_path):
    records = [_critic_record(index) for index in range(4)]
    payload = _full_export_payload(records)
    payload["records"] = [records[0], records[2], records[3]]
    payload["selection_summary"] = {
        "total_transitions": 3,
        "selected_transitions": 3,
    }
    invalid_path = tmp_path / "index_gap.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    baselines, by_episode, stats = StageBaselineEstimator().estimate_grouped_from_exports(
        [invalid_path]
    )

    assert baselines == {}
    assert by_episode == {}
    assert stats["accepted_export_file_count"] == 0
    assert stats["skipped_export_file_count"] == 1
    assert stats["skipped_export_reasons"] == {
        "critic_export_transition_index_non_contiguous": 1
    }


def test_stage_baseline_v3_deduplicates_identical_clean_sources(tmp_path):
    records = [_critic_record(index) for index in range(6)]
    records[-1].update(
        {
            "task_state_success": 1.0,
            "task_percent_complete": 1.0,
            "proposition_satisfied_fraction": 1.0,
        }
    )
    records[-1]["info_snapshot"].update(
        {
            "task_state_success": 1.0,
            "task_percent_complete": 1.0,
            "proposition_satisfied_fraction": 1.0,
        }
    )
    payload = json.dumps(_full_export_payload(records))
    first = tmp_path / "clean_seed0.json"
    duplicate = tmp_path / "clean_seed1_duplicate.json"
    first.write_text(payload, encoding="utf-8")
    duplicate.write_text(payload, encoding="utf-8")

    estimator = StageBaselineEstimator(require_successful_clean=True)
    baselines, _by_episode, stats = estimator.estimate_grouped_from_exports(
        [first, duplicate],
        source_metadata={
            str(first.resolve()): {"seed": 0},
            str(duplicate.resolve()): {"seed": 1},
        },
    )

    assert baselines
    assert len(stats["source_hashes"]) == 1
    assert len(stats["duplicate_source_hashes"]) == 1
    assert all(baseline.n_trajectories == 1 for baseline in baselines.values())


def test_stage_baseline_w_rem_uses_pooled_tau_and_episode_stage_scope(tmp_path):
    export_paths = []
    source_metadata = {}
    for episode_id, stride in (("78", 1), ("79", 3)):
        records = _clean_multimodal_records(
            episode_id,
            sim_stride=stride,
        )
        payload = _full_export_payload(records)
        payload["episode_id"] = episode_id
        path = tmp_path / f"episode_{episode_id}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        export_paths.append(path)
        source_metadata[str(path.resolve())] = {"seed": 0}

    estimator = StageBaselineEstimator(
        require_successful_clean=True,
        formal_min_trajectories=1,
        formal_min_transitions=1,
    )
    baselines, by_episode, stats = estimator.estimate_grouped_from_exports(
        export_paths,
        source_metadata=source_metadata,
    )

    pooled_tau = {
        (row["family"], row["stage"], row["modality"]): row["tau_star"]
        for row in stats["pooled_tau_by_anchor"]
    }
    assert pooled_tau[("other", 0, "navigation")] == 2.0
    assert pooled_tau[("other", 0, "manipulation")] == 2.0

    navigation = baselines[("other", 0, "navigation")]
    manipulation = baselines[("other", 0, "manipulation")]
    expected = {"78": {0: 1.25}, "79": {0: 3.75}}
    assert navigation.W_rem_star_by_episode_stage == expected
    assert manipulation.W_rem_star_by_episode_stage == expected
    assert navigation.W_rem_star_swe == manipulation.W_rem_star_swe == 2.5
    assert navigation.w_rem_star_for_episode_stage("78") == 1.25
    assert navigation.w_rem_star_for_episode_stage("missing") is None
    assert by_episode["78"][("other", 0, "navigation")].W_rem_star_swe == 1.25
    assert {
        record["episode_id"]
        for record in navigation.trajectory_provenance.values()
    } == {"78", "79"}

    roundtrip = StageBaseline.from_dict(navigation.to_dict())
    assert roundtrip.W_rem_star_by_episode_stage == expected
    assert roundtrip.trajectory_provenance == navigation.trajectory_provenance


def test_stage_baseline_materializes_formal_stage_component_pools(tmp_path):
    export_paths = []
    source_metadata = {}
    for seed in range(3):
        records = [
            _critic_record(index, value_scale=1.0 + seed / 100.0)
            for index in range(12)
        ]
        for index, record in enumerate(records):
            progress = 1.0 if index == len(records) - 1 else 0.0
            record["task_percent_complete"] = progress
            record["proposition_satisfied_fraction"] = progress
            record["info_snapshot"].update(
                {
                    "task_percent_complete": progress,
                    "proposition_satisfied_fraction": progress,
                    "proposition_satisfied_count": int(progress),
                    "proposition_total": 1,
                    "planning_step_count": index,
                    "sim_step_count": index,
                }
            )
            if 1 <= index <= 2:
                action = (
                    "Pick[target]"
                    if seed == 0 and index == 1
                    else "Navigate[target]"
                )
                record["recent_action"] = action
                record["info_snapshot"]["recent_action"] = action
        records[-1]["task_state_success"] = 1.0
        records[-1]["info_snapshot"]["task_state_success"] = 1.0
        path = tmp_path / f"clean_seed{seed}.json"
        path.write_text(
            json.dumps(_full_export_payload(records)),
            encoding="utf-8",
        )
        export_paths.append(path)
        source_metadata[str(path.resolve())] = {"seed": seed}

    estimator = StageBaselineEstimator(require_successful_clean=True)
    baselines, by_episode, stats = estimator.estimate_grouped_from_exports(
        export_paths,
        source_metadata=source_metadata,
    )

    for baseline_map in (baselines, by_episode["78"]):
        stability_pool = baseline_map[
            ("other", 0, STABILITY_POOL_MODALITY)
        ]
        recovery_pool = baseline_map[("other", 0, RECOVERY_POOL_MODALITY)]
        assert stability_pool.formal_valid is True
        assert (
            stability_pool.formal_support_scope
            == "family_stage_stability_pool"
        )
        assert stability_pool.stability_n_samples == 30
        assert (
            stability_pool.formal_component_support["stability"][
                "n_trajectories"
            ]
            == 3
        )
        assert recovery_pool.formal_valid is True
        assert (
            recovery_pool.formal_support_scope
            == "family_stage_recovery_pool"
        )
        assert recovery_pool.recovery_covariance_n_samples == 30

        manipulation = baseline_map[("other", 0, "manipulation")]
        assert manipulation.n_trajectories == 1
        assert manipulation.n_transitions == 1
        assert manipulation.formal_valid is True
        assert (
            manipulation.formal_support_scope
            == "hierarchical_component_pool"
        )
        support = manipulation.formal_component_support
        assert support["remaining_work"]["valid_episode_ids"] == ["78"]
        assert support["recovery"]["scope"] == "family_stage_pool"
        assert support["tau"]["scope"] == "exact_stage_modality"
        assert all(
            scope == "family_stage_pool"
            for scope in manipulation.recovery_covariance_scope.values()
        )
        assert manipulation.v_bar_channels == recovery_pool.v_bar_channels
        assert manipulation.dim("plan").tau_bar == recovery_pool.dim(
            "plan"
        ).tau_bar
        assert manipulation.w_rem_star_for_episode_stage("78", 0) is not None

        terminal_pools = [
            baseline
            for (family, stage, modality), baseline in baseline_map.items()
            if family == "other"
            and stage < 0
            and modality in {STABILITY_POOL_MODALITY, RECOVERY_POOL_MODALITY}
        ]
        assert terminal_pools
        assert all(not baseline.formal_valid for baseline in terminal_pools)

    assert stats["hierarchical_component_promoted_anchor_count"] >= 2


def test_stage_baseline_formal_metadata_uses_stage_component_coverage(tmp_path):
    family = "other"
    stage = 0
    diagnostic_exact = StageBaseline(
        family=family,
        stage=stage,
        modality="manipulation",
        formal_valid=False,
        formal_support_scope="diagnostic",
        formal_missing_reason="insufficient_transitions",
    )
    stability_pool = StageBaseline(
        family=family,
        stage=stage,
        modality=STABILITY_POOL_MODALITY,
        formal_valid=True,
        formal_support_scope="family_stage_stability_pool",
    )
    recovery_pool = StageBaseline(
        family=family,
        stage=stage,
        modality=RECOVERY_POOL_MODALITY,
        formal_valid=True,
        formal_support_scope="family_stage_recovery_pool",
    )
    path = tmp_path / "stage_baseline.json"
    StageBaselineEstimator.write(
        {
            (family, stage, "manipulation"): diagnostic_exact,
            (family, stage, STABILITY_POOL_MODALITY): stability_pool,
            (family, stage, RECOVERY_POOL_MODALITY): recovery_pool,
        },
        path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert metadata["formal_valid"] is True
    assert metadata["formal_required_nonterminal_coverage_valid"] is False
    assert metadata["diagnostic_required_nonterminal_anchor_count"] == 1
    assert metadata["formal_stability_pool_coverage_valid"] is True
    assert metadata["formal_recovery_pool_coverage_valid"] is True


def test_remaining_work_component_support_is_episode_local():
    baseline = StageBaseline(
        family="other",
        stage=0,
        modality="manipulation",
        formal_min_trajectories=3,
        W_rem_star_by_episode_stage={"78": {0: 4.0}, "79": {0: 5.0}},
        W_rem_support_by_episode_stage={"78": {0: 3}, "79": {0: 2}},
    )

    support = StageBaselineEstimator._remaining_work_component_support(baseline)

    assert support["valid"] is True
    assert support["valid_episode_ids"] == ["78"]
    assert support["invalid_episode_ids"] == ["79"]
    assert baseline.w_rem_star_for_episode_stage("78", 0) == 4.0
    assert baseline.w_rem_star_for_episode_stage("79", 0) == 5.0


def test_stage_baseline_merge_marks_duplicate_episode_seed_nonformal():
    def _baseline(trajectory_id: str, source_hash: str, value: float):
        return StageBaseline(
            family="other",
            stage=0,
            modality="planning",
            n_samples=30,
            n_transitions=30,
            n_trajectories=1,
            trajectory_ids=[trajectory_id],
            seed_ids=[0],
            source_hashes=[source_hash],
            trajectory_provenance={
                trajectory_id: {
                    "episode_id": "78",
                    "seed": 0,
                    "source_hash": source_hash,
                }
            },
            remaining_work_by_trajectory={trajectory_id: value},
            W_rem_star_by_episode_stage={"78": {0: value}},
            formal_min_trajectories=1,
            formal_min_transitions=1,
            formal_valid=True,
        )

    merged = StageBaselineEstimator.merge_baseline_maps(
        {("other", 0, "planning"): _baseline("trajectory-a", "hash-a", 2.0)},
        {("other", 0, "planning"): _baseline("trajectory-b", "hash-b", 4.0)},
    )[("other", 0, "planning")]

    assert merged.formal_valid is False
    assert "duplicate_episode_seed_trajectories" in merged.formal_missing_reason
    assert merged.duplicate_episode_seed_trajectories == [
        {
            "episode_id": "78",
            "seed": 0,
            "trajectory_ids": ["trajectory-a", "trajectory-b"],
        }
    ]
    assert merged.W_rem_star_by_episode_stage == {"78": {0: 3.0}}


def test_stage_baseline_merge_requires_trajectory_provenance():
    key = ("other", 0, "planning")
    baseline = StageBaseline(
        family=key[0],
        stage=key[1],
        modality=key[2],
        n_samples=30,
        n_transitions=30,
        n_trajectories=1,
        trajectory_ids=["trajectory-without-provenance"],
        remaining_work_by_trajectory={"trajectory-without-provenance": 2.0},
        formal_min_trajectories=1,
        formal_min_transitions=1,
        formal_valid=True,
    )

    merged = StageBaselineEstimator.merge_baseline_maps({}, {key: baseline})[key]

    assert merged.formal_valid is False
    assert "trajectory_provenance_missing" in merged.formal_missing_reason


def test_stage_baseline_merge_recomputes_oas_from_additive_moments():
    left_values = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [2.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    right_values = np.asarray(
        [[4.0, 1.0, 3.0], [8.0, 2.0, 1.0]],
        dtype=np.float64,
    )

    def _baseline(values, trajectory_id, source_hash):
        stats = _recovery_sufficient_statistics(values)
        mean, covariance, shrinkage, rank, condition = (
            _oas_from_sufficient_statistics(stats)
        )
        return StageBaseline(
            family="other",
            stage=0,
            modality="planning",
            recovery_mean={"cog": mean},
            recovery_covariance={"cog": covariance},
            recovery_covariance_n_samples=len(values),
            recovery_sufficient_statistics={"cog": stats},
            recovery_covariance_rank={"cog": rank},
            recovery_covariance_condition={"cog": condition},
            recovery_covariance_shrinkage={"cog": shrinkage},
            n_samples=len(values),
            n_transitions=len(values),
            n_trajectories=1,
            trajectory_ids=[trajectory_id],
            source_hashes=[source_hash],
            remaining_work_by_trajectory={trajectory_id: 2.0},
        )

    left = _baseline(left_values, "left", "hash-left")
    right = _baseline(right_values, "right", "hash-right")
    merged = StageBaselineEstimator.merge_baseline_maps(
        {("other", 0, "planning"): left},
        {("other", 0, "planning"): right},
    )[("other", 0, "planning")]

    pooled_stats = _recovery_sufficient_statistics(
        np.vstack([left_values, right_values])
    )
    expected_mean, expected_covariance, expected_shrinkage, expected_rank, _ = (
        _oas_from_sufficient_statistics(pooled_stats)
    )
    assert np.allclose(merged.recovery_mean["cog"], expected_mean)
    assert np.allclose(merged.recovery_covariance["cog"], expected_covariance)
    assert merged.recovery_covariance_shrinkage["cog"] == pytest.approx(
        expected_shrinkage
    )
    assert merged.recovery_covariance_rank["cog"] == expected_rank
    assert merged.recovery_sufficient_statistics["cog"]["n_samples"] == 5
    roundtrip = StageBaseline.from_dict(merged.to_dict())
    assert roundtrip.recovery_sufficient_statistics == (
        merged.recovery_sufficient_statistics
    )
    assert roundtrip.remaining_work_by_trajectory == (
        merged.remaining_work_by_trajectory
    )


def test_stage_baseline_merge_rejects_mixed_recovery_normalizers():
    values = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [2.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    stats = _recovery_sufficient_statistics(values)
    mean, covariance, shrinkage, rank, condition = (
        _oas_from_sufficient_statistics(stats)
    )

    def _baseline(trajectory_id, source_hash, delta_p_bar):
        return StageBaseline(
            family="other",
            stage=0,
            modality="planning",
            delta_p_bar=delta_p_bar,
            recovery_mean={"cog": mean},
            recovery_covariance={"cog": covariance},
            recovery_covariance_n_samples=len(values),
            recovery_sufficient_statistics={"cog": stats},
            recovery_covariance_rank={"cog": rank},
            recovery_covariance_condition={"cog": condition},
            recovery_covariance_shrinkage={"cog": shrinkage},
            n_samples=len(values),
            n_transitions=len(values),
            n_trajectories=1,
            trajectory_ids=[trajectory_id],
            source_hashes=[source_hash],
            remaining_work_by_trajectory={trajectory_id: 2.0},
        )

    merged = StageBaselineEstimator.merge_baseline_maps(
        {
            ("other", 0, "planning"): _baseline(
                "left", "hash-left", delta_p_bar=1.0
            )
        },
        {
            ("other", 0, "planning"): _baseline(
                "right", "hash-right", delta_p_bar=2.0
            )
        },
    )[("other", 0, "planning")]

    assert merged.formal_valid is False
    assert "recovery_feature_normalizer_mismatch" in merged.formal_missing_reason


def test_stage_baseline_merge_recomputes_w_rem_from_trajectory_summaries():
    left = StageBaseline(
        family="other",
        stage=0,
        modality="planning",
        W_rem_star_swe=99.0,
        n_samples=1000,
        n_transitions=1000,
        n_trajectories=2,
        trajectory_ids=["a", "b"],
        source_hashes=["hash-left"],
        remaining_work_by_trajectory={"a": 2.0, "b": 100.0},
    )
    right = StageBaseline(
        family="other",
        stage=0,
        modality="planning",
        W_rem_star_swe=3.0,
        n_samples=1,
        n_transitions=1,
        n_trajectories=1,
        trajectory_ids=["c"],
        source_hashes=["hash-right"],
        remaining_work_by_trajectory={"c": 3.0},
    )
    merged = StageBaselineEstimator.merge_baseline_maps(
        {("other", 0, "planning"): left},
        {("other", 0, "planning"): right},
    )[("other", 0, "planning")]

    assert merged.W_rem_star_swe == 3.0
    assert merged.remaining_work_by_trajectory == {
        "a": 2.0,
        "b": 100.0,
        "c": 3.0,
    }


def test_family_covariance_shrinkage_is_explicit_and_preserves_local_moments():
    trajectory_ids = ["clean-78-0", "clean-78-1", "clean-78-2"]
    provenance = {
        trajectory_id: {
            "episode_id": "78",
            "seed": seed,
            "source_hash": f"hash-{seed}",
        }
        for seed, trajectory_id in enumerate(trajectory_ids)
    }

    def _undersampled_stage(stage: int, offset: float) -> StageBaseline:
        values = np.asarray(
            [
                [offset + index / 10.0, float(index % 3), float(index % 2)]
                for index in range(15)
            ],
            dtype=np.float64,
        )
        stats = _recovery_sufficient_statistics(values)
        return StageBaseline(
            family="other",
            stage=stage,
            modality="planning",
            recovery_sufficient_statistics={
                group: dict(stats) for group in ("cog", "phy", "debt")
            },
            n_samples=15,
            n_transitions=15,
            n_trajectories=3,
            trajectory_ids=list(trajectory_ids),
            trajectory_provenance={
                key: dict(value) for key, value in provenance.items()
            },
            remaining_work_by_trajectory={
                trajectory_id: 4.0 - stage for trajectory_id in trajectory_ids
            },
            W_rem_star_by_episode_stage={"78": {stage: 4.0 - stage}},
            formal_valid=False,
            formal_missing_reason="insufficient_transitions",
            formal_support_scope="diagnostic",
        )

    baselines = {
        ("other", 0, "planning"): _undersampled_stage(0, 0.0),
        ("other", 1, "planning"): _undersampled_stage(1, 1.0),
    }
    assert StageBaselineEstimator().allow_family_covariance_shrinkage is False
    assert all(not baseline.formal_valid for baseline in baselines.values())

    pools = StageBaselineEstimator._build_family_covariance_pools(baselines)
    applied = StageBaselineEstimator._apply_family_covariance_shrinkage(
        baselines,
        pools,
    )

    assert applied == 2
    for baseline in baselines.values():
        assert baseline.formal_valid is True
        assert baseline.formal_missing_reason == ""
        assert baseline.formal_support_scope == "family_modality_cross_stage"
        assert baseline.recovery_covariance_n_samples == 30
        for group in ("cog", "phy", "debt"):
            assert baseline.recovery_sufficient_statistics[group]["n_samples"] == 15
            assert (
                baseline.recovery_covariance_scope[group]
                == "family_modality_cross_stage"
            )
            source = baseline.recovery_covariance_source[group]
            assert source["pooled_stages"] == [0, 1]
            assert source["n_samples"] == 30
            assert source["n_trajectories"] == 3


@pytest.mark.parametrize(
    "stage,reason",
    [
        (-1, "terminal_or_unknown_stage;insufficient_transitions"),
        (0, "insufficient_transitions;trajectory_provenance_missing"),
        (0, "insufficient_transitions;remaining_work_trajectory_summary_missing"),
        (0, "insufficient_transitions;episode_stage_remaining_work_missing"),
        (0, "insufficient_transitions;duplicate_episode_seed_trajectories"),
        (0, "insufficient_transitions;recovery_feature_normalizer_mismatch"),
        (0, "insufficient_transitions;recovery_sufficient_statistics_missing"),
    ],
)
def test_family_covariance_shrinkage_does_not_revive_invalid_anchors(
    stage,
    reason,
):
    stats = _recovery_sufficient_statistics(np.ones((30, 3), dtype=np.float64))
    mean, covariance, shrinkage, rank, condition = (
        _oas_from_sufficient_statistics(stats)
    )
    pools = {
        ("other", "planning", group): {
            "valid": True,
            "stats": stats,
            "mean": mean,
            "covariance": covariance,
            "shrinkage": shrinkage,
            "rank": rank,
            "condition": condition,
            "n_trajectories": 3,
            "stages": [0, 1],
        }
        for group in ("cog", "phy", "debt")
    }
    candidate = StageBaseline(
        family="other",
        stage=stage,
        modality="planning",
        trajectory_ids=["a", "b", "c"],
        trajectory_provenance={
            name: {"episode_id": "78", "seed": seed, "source_hash": name}
            for seed, name in enumerate(("a", "b", "c"))
        },
        remaining_work_by_trajectory={"a": 3.0, "b": 3.0, "c": 3.0},
        W_rem_star_by_episode_stage={"78": {stage: 3.0}},
        formal_valid=False,
        formal_missing_reason=reason,
        formal_support_scope="diagnostic",
    )

    applied = StageBaselineEstimator._apply_family_covariance_shrinkage(
        {("other", stage, "planning"): candidate},
        pools,
    )

    assert applied == 0
    assert candidate.formal_valid is False
    assert candidate.formal_missing_reason == reason
    assert candidate.formal_support_scope == "diagnostic"


def test_stage_baseline_wait_uses_runtime_idle_gap_feature():
    records = [_critic_record(index) for index in range(3)]
    for record in records:
        record["info_snapshot"]["sim_step_count"] = 0
    records[1]["high_level_actions"] = {"0": "Wait[]"}

    estimator = StageBaselineEstimator()
    accumulators = estimator.estimate_from_episode(
        records,
        family_override="other",
        trajectory_id="wait-trajectory",
    )
    baseline = next(
        accumulator.finalize(key)
        for key, accumulator in accumulators.items()
    )

    # One no-progress Wait with one planning step contributes idle_gap=1 to
    # the second physical channel; the non-Wait transition contributes zero.
    phy_stats = baseline.recovery_sufficient_statistics["phy"]
    assert phy_stats["n_samples"] == 2
    assert phy_stats["sum"] == pytest.approx([0.0, 1.0, 0.0])


def test_runtime_marks_embedded_family_covariance_as_shrinkage():
    anchor = StageAnchor(family="other", stage=2, modality="planning")
    baseline = StageBaseline(
        family="other",
        stage=2,
        modality="planning",
        formal_valid=True,
        formal_support_scope="family_modality_cross_stage",
    )

    selected, level = _select_baseline_with_level(
        {anchor.key(): baseline},
        anchor,
    )

    assert selected is baseline
    assert level == "exact_statistics_incomplete"


def test_formal_beta_rejects_incomplete_neighborhood():
    collector = StabilityCollector()
    aggregates, _ = collector.aggregate_beta_over_anchors(
        [
            {
                "anchor_id": "78",
                "perturbation_type": "clean",
                "stability_trajectory_loss": 1.0,
                "stability_trajectory_loss_valid": 1,
                "task_state_success": 1.0,
                "task_percent_complete": 1.0,
            },
            {
                "anchor_id": "78",
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 0.5,
                "perturbation_realized": 0.5,
                "perturbation_valid": 1,
                "stability_trajectory_loss": 1.5,
                "stability_trajectory_loss_valid": 1,
                "task_state_success": 0.8,
                "task_percent_complete": 0.8,
            },
            {
                "anchor_id": "78",
                "perturbation_type": "filler_injection",
                "perturbation_intensity": 0.5,
                "perturbation_realized": 0.5,
                "perturbation_valid": 1,
                "stability_trajectory_loss": "",
                "stability_trajectory_loss_valid": 0,
                "task_state_success": 0.5,
                "task_percent_complete": 0.5,
            },
        ]
    )
    aggregate = aggregates["78"]
    assert not aggregate.beta_neighborhood_valid
    assert aggregate.beta_hat is None
    assert aggregate.beta_trajectory is None
    assert aggregate.missing_reason == "perturbed_stage_trajectory_loss_incomplete"
    family = collector.aggregate_beta_by_family(aggregates)["other"]
    assert not family.beta_family_valid
    assert family.beta_family is None


def test_formal_beta_fails_closed_on_invalid_realized_stress():
    collector = StabilityCollector()
    aggregates, _ = collector.aggregate_beta_over_anchors(
        [
            {
                "anchor_id": "78",
                "perturbation_type": "clean",
                "stability_trajectory_loss": 1.0,
                "stability_trajectory_loss_valid": 1,
            },
            {
                "anchor_id": "78",
                "perturbation_type": "irrelevant_perturbation",
                "perturbation_intensity": 0.5,
                "perturbation_realized": 0.4,
                "perturbation_valid": 0,
                "perturbation_reason": "planner_visible_change_not_observed",
                "stability_trajectory_loss": 1.5,
                "stability_trajectory_loss_valid": 1,
            },
        ]
    )

    aggregate = aggregates["78"]
    assert not aggregate.beta_neighborhood_valid
    assert aggregate.beta_hat is None
    assert aggregate.beta_out is None
    assert aggregate.missing_reason == "perturbation_realization_invalid"


def test_stability_rejects_out_of_range_realized_stress():
    run = StabilityCollector._normalize_raw_runs(
        [
            {
                "anchor_id": "78",
                "perturbation_type": "surface_rewrite",
                "perturbation_valid": 1,
                "perturbation_realized": 2.0,
            }
        ]
    )[0]

    assert not run.perturbation_realization_valid
    assert (
        run.perturbation_realization_reason
        == "out_of_range_realized_perturbation_dose"
    )


def test_stability_formal_stress_pairs_each_seed_with_its_own_clean_row():
    collector = StabilityCollector()
    rows = []
    for seed, clean_loss in ((0, 0.0), (1, 100.0)):
        for stress_lambda in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            is_clean = stress_lambda == 0.0
            row = {
                "anchor_id": "78",
                "episode_id": "78",
                "task_family": "placement_only",
                "perturbation_seed": seed,
                "perturbation_type": (
                    "clean"
                    if is_clean
                    else "llm_instruction_context_stress"
                ),
                "stress_lambda": stress_lambda,
                "perturbation_stress_family": (
                    "llm_instruction_context_stress"
                ),
                "perturbation_variant": f"variant_{seed}",
                "perturbation_bank_path": "/banks/instruction_context_v1",
                "perturbation_stress_realized": stress_lambda,
                "perturbation_text_dose": stress_lambda,
                "perturbation_alignment_valid": 1,
                "stability_trajectory_loss": (
                    clean_loss + 5.0 * stress_lambda
                ),
                "stability_trajectory_loss_valid": 1,
                "task_state_success": 1.0,
                "task_percent_complete": 1.0,
                "action_sequence": "Navigate|Pick|Place",
            }
            if not is_clean:
                row["perturbation_valid"] = 1
            rows.append(row)

    aggregates, _ = collector.aggregate_beta_by_stress(rows)

    aggregate = aggregates[
        ("78", "llm_instruction_context_stress", 0.2)
    ]
    assert aggregate.beta_neighborhood_valid
    assert aggregate.clean_ref_strategy == "episode_seed_exact"
    assert aggregate.n_paired_seeds == 2
    assert aggregate.beta_hat == pytest.approx(
        1.0 / (0.2 + 1.0e-6), abs=1.0e-12
    )

    headline, _ = collector.aggregate_beta_over_anchors(rows)
    formal_anchor = headline["78"]
    assert formal_anchor.beta_neighborhood_valid
    assert formal_anchor.clean_ref_strategy == "episode_seed_exact"
    assert formal_anchor.n_paired_seeds == 2
    assert formal_anchor.beta_hat == pytest.approx(
        5.0 / (1.0 + 1.0e-6), abs=1.0e-12
    )
    family = collector.aggregate_beta_by_family(headline)["placement_only"]
    assert family.beta_family_valid
    assert family.beta_family == formal_anchor.beta_hat
