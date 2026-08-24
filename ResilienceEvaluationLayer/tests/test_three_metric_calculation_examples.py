"""Ten auditable calculation examples for Rebound, Stability, and GE.

These tests are numeric oracles, not simulator smoke tests.  Every expected
value is derived independently in the test so a regression in the metric
implementation cannot pass merely because an output field exists.
"""

import math
from pathlib import Path
import sys
import types
from typing import Optional

import pytest


# Keep calculation-only validation independent of the Habitat simulator.
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


from habitat_llm.evaluation.metrics.extensibility import (
    BoundaryCollector,
    BoundaryMargin,
    BoundaryThresholds,
)
from habitat_llm.evaluation.metrics.rebound import (
    ReboundCollector,
    StageBaseline,
    StageBaselineDim,
)
from habitat_llm.evaluation.metrics.stability import StabilityCollector


def _identity3() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _simple_stage_baseline() -> StageBaseline:
    """Return a baseline that makes the Rebound arithmetic transparent."""

    return StageBaseline(
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
        recovery_covariance={
            "cog": _identity3(),
            "phy": _identity3(),
            "debt": _identity3(),
        },
        recovery_mean={
            "cog": [0.0, 0.0, 0.0],
            "phy": [0.0, 0.0, 0.0],
            "debt": [0.0, 0.0, 0.0],
        },
        recovery_covariance_n_samples=4,
    )


def _single_window_inputs(*, recovered: bool = True) -> tuple[list[dict], dict]:
    transitions = [
        {
            "planning_step_count": 0,
            "sim_step_count": 0,
            "task_percent_complete": 0.0,
            "proposition_satisfied_fraction": 0.0,
        },
        {
            "planning_step_count": 1,
            "sim_step_count": 2,
            "task_percent_complete": 0.0,
            "proposition_satisfied_fraction": 0.0,
        },
    ]
    anchor = {
        "family": "other",
        "stage": 0,
        "modality": "planning",
        "progress": 0.0,
    }
    tracker_summary = {
        "windows": [
            {
                "t_d": 1,
                "t_r": 1,
                "t_end": 1,
                "recovered": recovered,
                "closure_reason": (
                    "recovery_confirmed" if recovered else "episode_tail_censored"
                ),
                "anchor_t_d": anchor,
                "anchor_t_r": anchor,
            }
        ],
        "anchors": [anchor, anchor],
        "progress_deltas": [0.0, 0.0],
        "q_deltas": [0.0, 0.0],
        "cognitive_steps": [],
    }
    return transitions, tracker_summary


def _stability_row(
    *,
    loss: Optional[float],
    perturbation_type: str,
    realized: Optional[float] = None,
    valid_loss: bool = True,
) -> dict:
    row = {
        "anchor_id": "78",
        "task_family": "placement_only",
        "perturbation_type": perturbation_type,
        "stability_trajectory_loss": loss,
        "stability_trajectory_loss_valid": int(valid_loss),
        "task_state_success": 1.0,
        "task_percent_complete": 1.0,
        "action_sequence": "Navigate|Pick|Place",
    }
    if perturbation_type != "clean":
        row.update(
            {
                "perturbation_valid": 1,
                "perturbation_realized": realized,
            }
        )
    return row


# ---------------------------------------------------------------------------
# Rebound: four cases
# ---------------------------------------------------------------------------


def test_case_01_rebound_no_disruption_is_valid_zero():
    """No t_d means no recovery burden: C_rec=0 is valid, not N/A."""

    metrics = ReboundCollector({}).compute_stage_crec_multidim(
        transitions=[],
        baselines={},
        tracker_summary={"windows": []},
    )

    assert metrics.c_rec_valid is True
    assert metrics.c_rec == 0.0
    assert metrics.c_rec_index_100 == 0.0
    assert metrics.c_rec_observed_lower_bound is None
    assert metrics.c_rec_missing_reason == ""


def test_case_02_rebound_one_window_matches_eq2_and_eq3():
    """One physical excess step has a closed-form C_rec oracle."""

    transitions, tracker_summary = _single_window_inputs()
    baseline = _simple_stage_baseline()
    metrics = ReboundCollector(
        {"c_rec": {"mahalanobis_regularizer": 0.0, "epsilon": 1.0e-6}}
    ).compute_stage_crec_multidim(
        transitions=transitions,
        baselines={("other", 0, "planning"): baseline},
        tracker_summary=tracker_summary,
    )

    # x_phy=[0,1,0], mu=0, Sigma=I, so the dimension-normalized
    # g_phy=sqrt(1/(3*(1+1e-12))).
    # g_rec=sqrt((0^2+g_phy^2+0^2)/3), time weight=2/1,
    # and W_rem*=4.
    g_phy = math.sqrt(1.0 / (3.0 * (1.0 + 1.0e-12)))
    expected = (2.0 * g_phy / math.sqrt(3.0)) / (4.0 + 1.0e-6)

    assert metrics.c_rec_valid is True
    assert metrics.formal_window_count == 1
    assert metrics.c_rec == pytest.approx(expected, abs=1.0e-12)
    assert metrics.c_rec_index_100 == pytest.approx(
        100.0 * expected / (1.0 + expected), abs=1.0e-12
    )


def test_case_03_rebound_unrecovered_window_is_na():
    """A detected disruption without evidence-backed t_r is censored."""

    transitions, tracker_summary = _single_window_inputs(recovered=False)
    metrics = ReboundCollector({}).compute_stage_crec_multidim(
        transitions=transitions,
        baselines={("other", 0, "planning"): _simple_stage_baseline()},
        tracker_summary=tracker_summary,
    )

    assert metrics.c_rec_valid is False
    assert metrics.c_rec_missing_reason == "episode_tail_censored"
    assert metrics.formal_window_count == 0
    assert metrics.skipped_window_count == 1
    assert metrics.c_rec_index_100 is None
    assert metrics.c_rec_observed_lower_bound is not None
    g_phy = math.sqrt(1.0 / (3.0 * (1.0 + 1.0e-3)))
    expected_observed = (2.0 * g_phy / math.sqrt(3.0)) / (4.0 + 1.0e-6)
    assert metrics.c_rec_observed_lower_bound == pytest.approx(
        expected_observed, abs=1.0e-12
    )
    summary = metrics.get_summary_for_csv()
    assert summary["rebound_c_rec"] is None
    assert summary["rebound_c_rec_index_100"] is None
    assert summary["rebound_c_rec_observed_lower_bound"] > 0.0


def test_case_04_rebound_disruption_without_baseline_is_na():
    """A closed window cannot be normalized without its StageBaseline."""

    transitions, tracker_summary = _single_window_inputs()
    metrics = ReboundCollector({}).compute_stage_crec_multidim(
        transitions=transitions,
        baselines={},
        tracker_summary=tracker_summary,
    )

    assert metrics.c_rec_valid is False
    assert metrics.c_rec_missing_reason == "stage_baseline_missing"
    assert metrics.skipped_window_count == 1
    assert metrics.c_rec_index_100 is None
    assert metrics.c_rec_observed_lower_bound is None


# ---------------------------------------------------------------------------
# Stability: three cases
# ---------------------------------------------------------------------------


def test_case_05_stability_single_perturbation_matches_eq6():
    """beta=|3-2|/(0.5+epsilon) for one clean/perturbed pair."""

    collector = StabilityCollector()
    aggregates, _ = collector.aggregate_beta_over_anchors(
        [
            _stability_row(loss=2.0, perturbation_type="clean"),
            _stability_row(
                loss=3.0,
                perturbation_type="surface_rewrite",
                realized=0.5,
            ),
        ]
    )
    aggregate = aggregates["78"]
    expected = 1.0 / (0.5 + 1.0e-6)

    assert aggregate.beta_neighborhood_valid is True
    assert aggregate.beta_hat == pytest.approx(expected, abs=1.0e-12)


def test_case_06_stability_uses_clean_mean_and_perturbation_supremum():
    """The formal beta uses mean(clean) and max over perturbed sensitivities."""

    collector = StabilityCollector()
    aggregates, _ = collector.aggregate_beta_over_anchors(
        [
            _stability_row(loss=1.0, perturbation_type="clean"),
            _stability_row(loss=3.0, perturbation_type="clean"),
            _stability_row(
                loss=2.5,
                perturbation_type="surface_rewrite",
                realized=0.25,
            ),
            _stability_row(
                loss=5.0,
                perturbation_type="filler_injection",
                realized=1.0,
            ),
        ]
    )
    aggregate = aggregates["78"]
    expected = max(
        abs(2.5 - 2.0) / (0.25 + 1.0e-6),
        abs(5.0 - 2.0) / (1.0 + 1.0e-6),
    )

    assert aggregate.beta_neighborhood_valid is True
    assert aggregate.beta_hat == pytest.approx(expected, abs=1.0e-12)


def test_case_07_stability_incomplete_neighborhood_is_na():
    """One missing planned perturbation trace invalidates the anchor beta."""

    collector = StabilityCollector()
    aggregates, _ = collector.aggregate_beta_over_anchors(
        [
            _stability_row(loss=1.0, perturbation_type="clean"),
            _stability_row(
                loss=1.5,
                perturbation_type="surface_rewrite",
                realized=0.5,
            ),
            _stability_row(
                loss=None,
                perturbation_type="filler_injection",
                realized=0.5,
                valid_loss=False,
            ),
        ]
    )
    aggregate = aggregates["78"]

    assert aggregate.beta_neighborhood_valid is False
    assert aggregate.beta_hat is None
    assert aggregate.missing_reason == "perturbed_stage_trajectory_loss_incomplete"


# ---------------------------------------------------------------------------
# GE: three cases
# ---------------------------------------------------------------------------


def test_case_08_ge_bottom_k_soft_margin_matches_manual_ratios():
    """The two weakest ratios 1.1 and 1.2 give M=(1.1+1.2)/2-1=0.15."""

    thresholds = BoundaryThresholds(bottom_k=2, ratio_clip_max=2.0)
    margin = BoundaryMargin.compute(
        family="placement_only",
        stress_lambda=0.4,
        evolve_step=0,
        pc_f=0.84,
        q_f=0.77,
        u_f=0.025,
        c_rec_p90_f=0.8,
        v_stab_p90_f=0.1,
        thresholds=thresholds,
    )

    assert margin.pc_term == pytest.approx(1.2)
    assert margin.q_term == pytest.approx(1.1)
    assert margin.margin == pytest.approx(0.15)
    assert margin.hard_margin == pytest.approx(0.1)
    assert margin.hard_gate_pass is True


def test_case_09_ge_soft_boundary_and_hard_gate_are_distinct():
    """Soft M=0 can pass while the weakest hard ratio 0.9 fails the gate."""

    thresholds = BoundaryThresholds(
        bottom_k=2,
        ratio_clip_max=2.0,
        hard_gate_delta=0.05,
    )
    margin = BoundaryMargin.compute(
        family="placement_only",
        stress_lambda=0.6,
        evolve_step=0,
        pc_f=0.63,
        q_f=0.77,
        u_f=0.01,
        c_rec_p90_f=0.5,
        v_stab_p90_f=0.1,
        thresholds=thresholds,
    )

    assert margin.margin == pytest.approx(0.0)
    assert margin.hard_margin == pytest.approx(-0.1)
    assert margin.boundary_hit is False
    assert margin.hard_gate_pass is False
    assert margin.hard_boundary_hit is True


def test_case_10_ge_discrete_capacity_and_audc_match_manual_curve():
    """Margins [0.4,0.1,-0.2] yield lambda*=0.5 and AUDC=1.1."""

    thresholds = BoundaryThresholds(stress_lambdas=(0.0, 0.5, 1.0))
    collector = BoundaryCollector(thresholds=thresholds)
    margins = {
        ("placement_only", 0.0, 0): BoundaryMargin(
            family="placement_only",
            stress_lambda=0.0,
            evolve_step=0,
            margin=0.4,
            hard_gate_pass=True,
            cell_valid=True,
        ),
        ("placement_only", 0.5, 0): BoundaryMargin(
            family="placement_only",
            stress_lambda=0.5,
            evolve_step=0,
            margin=0.1,
            hard_gate_pass=True,
            cell_valid=True,
        ),
        ("placement_only", 1.0, 0): BoundaryMargin(
            family="placement_only",
            stress_lambda=1.0,
            evolve_step=0,
            margin=-0.2,
            hard_gate_pass=False,
            cell_valid=True,
        ),
    }

    audc = collector.finalize_audc(margins)[("placement_only", 0)]
    cell = margins[("placement_only", 0.5, 0)]

    # Trapezoids over y=M+1: 0.5*(1.4+1.1)*0.5 +
    #                           0.5*(1.1+0.8)*0.5 = 1.1.
    assert audc == pytest.approx(1.1)
    assert cell.boundary_lambda_star == 0.5
    assert cell.hard_lambda_star == 0.5
    assert cell.valid_lambda_coverage == 1.0
    assert cell.capacity_valid is True
