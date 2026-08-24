r"""Canonical resilience result schema and console presentation.

The evaluation has two different result lifecycles:

* runtime rows contain one rollout's Rebound result and Stability diagnostics;
* post-processing rows contain neighborhood Stability and family/stress GE.

Keeping those schemas separate prevents a single episode from looking as if it
had already estimated :math:`\beta_f` or :math:`\lambda_f^*`.  Formal values use
``None``/an empty CSV cell when their validity condition is false.  A numeric
zero is therefore reserved for an observed zero.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Tuple


REBOUND_RESULT_KEYS: Tuple[str, ...] = (
    "rebound_c_rec",
    "rebound_c_rec_index_100",
    "rebound_c_rec_cog",
    "rebound_c_rec_phy",
    "rebound_c_rec_state_debt",
)

REBOUND_DIAGNOSTIC_KEYS: Tuple[str, ...] = (
    "rebound_num_recovery_windows",
    "rebound_raw_window_count",
    "rebound_formal_window_count",
    "rebound_skipped_window_count",
    "rebound_c_rec_valid",
    "rebound_c_rec_missing_reason",
    "rebound_c_rec_observed_lower_bound",
    "rebound_g_rec_observed_total",
    "rebound_observed_censored_window_count",
    "rebound_c_rec_formula_version",
    "rebound_baseline_match_level",
    "rebound_w_rem_star",
    "rebound_g_rec_total",
    "rebound_baseline_loaded",
    "rebound_tracker_window_count",
    "rebound_tracker_window_open_final",
)

STABILITY_RUNTIME_KEYS: Tuple[str, ...] = (
    # A single-run diagnostic. It is deliberately not named beta.
    "stability_episode_proxy",
    "stability_episode_proxy_source",
    "stability_value_variance",
    "stability_value_delta_variance",
    "stability_td_error_mean_abs",
    "stability_td_error_p95_abs",
    "stability_gae_mean_abs",
    "stability_gae_p95_abs",
    "stability_return_target_variance",
    "stability_oscillation_loss",
    "stability_oscillation_loss_valid",
    "stability_oscillation_source",
    "stability_critic_trace_len",
    "stability_n_replan",
    "stability_p_cbf",
    "safety_collision_rate",
    "safety_score",
    "unsafe_rate",
    "safety_valid",
    "safety_scope",
    "safety_missing_reason",
    "safety_distance_constraint_available",
    "safety_distance_observation_count",
    "safety_distance_missing_count",
    "safety_distance_constraint_missing_reason",
)

STABILITY_POSTPROCESS_KEYS: Tuple[str, ...] = (
    # Eq. (6)'s paper-facing task-family supremum.
    "stability_beta_family",
    "stability_beta_family_valid",
    "stability_beta_family_n_anchors",
    "stability_beta_family_n_valid_anchors",
    "stability_beta_family_missing_reason",
    # Episode/anchor neighborhood result used to build the family supremum.
    "stability_beta_neighborhood",
    "stability_beta_trajectory",
    "stability_beta_out",
    "stability_beta_oscillation",
    "stability_beta_action",
    "stability_beta_progress",
    "stability_beta_success",
    "stability_clean_ref_valid",
    "stability_clean_ref_strategy",
    "stability_n_clean_ref",
    "stability_beta_neighborhood_valid",
    "stability_beta_neighborhood_missing_reason",
    "stability_beta_scope",
    # GE consumes the lambda-local load, never the all-lambda supremum above.
    "stability_beta_at_lambda",
    "stability_beta_at_lambda_valid",
    "stability_beta_at_lambda_missing_reason",
    "stability_beta_at_lambda_scope",
    "stability_beta_at_lambda_stress_family",
    "stability_beta_at_lambda_n_paired_seeds",
    "stability_shared_pair_eligible",
    "stability_shared_pair_missing_reason",
    # Per-perturbation slices are diagnostics, not a second headline beta.
    "stability_beta_perturbation",
    "stability_beta_perturbation_trajectory",
    "stability_beta_perturbation_out",
    "stability_beta_perturbation_oscillation",
    "stability_beta_perturbation_action",
    "stability_beta_perturbation_progress",
    "stability_beta_perturbation_success",
    "stability_beta_perturbation_valid",
    "stability_beta_perturbation_scope",
)

GE_POSTPROCESS_KEYS: Tuple[str, ...] = (
    "ge_boundary_margin",
    "ge_margin_soft",
    "ge_margin_hard",
    "ge_ratio_pc",
    "ge_ratio_q",
    "ge_ratio_rec",
    "ge_ratio_stab",
    "ge_ratio_safe",
    "ge_hard_gate_pass",
    "ge_boundary_hit",
    "ge_hard_boundary_hit",
    "ge_cell_valid",
    "ge_cell_missing_reason",
    "ge_boundary_lambda_star",
    "ge_hard_lambda_star",
    "ge_audc_f",
    "ge_audc_norm_f",
    "ge_valid_lambda_coverage",
    "ge_capacity_valid",
    "ge_capacity_missing_reason",
    "ge_degradation_rate",
    "ge_max_degradation_rate",
    "ge_margin_drop_f",
    "ge_monotonicity_violations",
    "ge_monotonicity_violation",
    "ge_cliff_flag",
    "ge_cliff_lambda",
)

RUNTIME_METRIC_KEYS: Tuple[str, ...] = (
    *REBOUND_RESULT_KEYS,
    *REBOUND_DIAGNOSTIC_KEYS,
    *STABILITY_RUNTIME_KEYS,
)

POSTPROCESS_METRIC_KEYS: Tuple[str, ...] = (
    *STABILITY_POSTPROCESS_KEYS,
    *GE_POSTPROCESS_KEYS,
)


def empty_runtime_metric_summary() -> Dict[str, Any]:
    """Return explicit *not estimated* defaults for one runtime rollout."""

    summary: Dict[str, Any] = {key: None for key in RUNTIME_METRIC_KEYS}
    summary.update(
        {
            "rebound_c_rec_valid": 0.0,
            "rebound_c_rec_missing_reason": "not_estimated",
            "rebound_baseline_match_level": "none",
            "stability_oscillation_loss_valid": 0.0,
            "stability_episode_proxy_source": "not_estimated",
        }
    )
    return summary


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "1.0", "true", "yes"}
    return bool(value)


def _present(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _append_present(
    items: List[Tuple[str, Any]], metrics: Mapping[str, Any], keys: Iterable[str]
) -> None:
    for key in keys:
        value = metrics.get(key)
        if _present(value):
            items.append((key, value))


def primary_display_items(metrics: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    """Return task outcome plus one headline per resilience dimension."""

    items: List[Tuple[str, Any]] = []
    _append_present(
        items,
        metrics,
        (
            "task_state_success",
            "task_percent_complete",
            "sim_step_count",
        ),
    )
    replanning_keys = [
        str(key)
        for key in metrics
        if str(key).startswith("replanning_count_")
    ]

    def replanning_sort_key(key: str) -> Tuple[int, Any]:
        agent_id = key.removeprefix("replanning_count_")
        try:
            return (0, int(agent_id))
        except ValueError:
            return (1, agent_id)

    _append_present(items, metrics, sorted(replanning_keys, key=replanning_sort_key))

    if any(key in metrics for key in REBOUND_RESULT_KEYS + REBOUND_DIAGNOSTIC_KEYS):
        if _truthy(metrics.get("rebound_c_rec_valid")):
            _append_present(items, metrics, ("rebound_c_rec",))
        else:
            reason = metrics.get("rebound_c_rec_missing_reason") or "not_estimated"
            items.append(("rebound_c_rec", f"N/A ({reason})"))

    if "stability_beta_family_valid" in metrics:
        if _truthy(metrics.get("stability_beta_family_valid")):
            _append_present(items, metrics, ("stability_beta_family",))
        else:
            reason = (
                metrics.get("stability_beta_family_missing_reason")
                or "incomplete_anchor_neighborhoods"
            )
            items.append(("stability_beta_family", f"N/A ({reason})"))
    elif any(key in metrics for key in STABILITY_RUNTIME_KEYS):
        if _present(metrics.get("stability_episode_proxy")):
            items.append(("stability_episode_proxy", metrics["stability_episode_proxy"]))
        else:
            items.append(("stability_episode_proxy", "N/A (not_estimated)"))

    if any(key in metrics for key in GE_POSTPROCESS_KEYS):
        if _truthy(metrics.get("ge_capacity_valid")):
            _append_present(items, metrics, ("ge_boundary_lambda_star",))
        else:
            reason = (
                metrics.get("ge_capacity_missing_reason")
                or "insufficient_valid_stress_levels"
            )
            items.append(("ge_boundary_lambda_star", f"N/A ({reason})"))
    elif any(key in metrics for key in RUNTIME_METRIC_KEYS):
        items.append(
            ("ge_boundary_lambda_star", "N/A (requires_multi_stress_sweep)")
        )

    return items
