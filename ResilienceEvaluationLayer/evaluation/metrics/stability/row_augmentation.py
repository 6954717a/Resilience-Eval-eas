"""Attach aggregate Stability evidence to rollout rows.

This module keeps CSV materialization separate from both metric estimation and
experiment orchestration.  In particular, the global neighborhood beta used by
the standalone Stability metric is not reused as the GE load at every stress
level; GE receives the lambda-local beta explicitly.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

from habitat_llm.evaluation.perturbation.bank import STRESS_FAMILY

from .stability_collector import BetaAggregate, BetaFamilyAggregate


def _finite_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _formal_value(aggregate: BetaAggregate, value: Any) -> Any:
    return (
        float(value)
        if aggregate.beta_neighborhood_valid and value is not None
        else ""
    )


def augment_rows_with_beta_aggregates(
    raw_rows: Sequence[MutableMapping[str, Any]],
    aggregates: Mapping[str, BetaAggregate],
    perturbation_aggregates: Optional[
        Mapping[Tuple[str, str], BetaAggregate]
    ] = None,
    family_aggregates: Optional[Mapping[str, BetaFamilyAggregate]] = None,
    stress_aggregates: Optional[
        Mapping[
            Tuple[Any, ...],
            BetaAggregate,
        ]
    ] = None,
    stress_seed_aggregates: Optional[
        Mapping[Tuple[str, str, float, int], BetaAggregate]
    ] = None,
) -> None:
    """Materialize standalone, diagnostic, family and lambda-local betas."""

    for row in raw_rows:
        anchor_id = str(row.get("anchor_id", row.get("episode_id", "unknown")))
        aggregate = aggregates.get(anchor_id)
        if aggregate is not None:
            row["stability_beta_neighborhood"] = _formal_value(
                aggregate, aggregate.beta_hat
            )
            row["stability_beta_trajectory"] = _formal_value(
                aggregate, aggregate.beta_trajectory
            )
            row["stability_beta_vv"] = _formal_value(
                aggregate, aggregate.beta_vv
            )
            for field, value in (
                ("stability_beta_out", aggregate.beta_out),
                ("stability_beta_oscillation", aggregate.beta_oscillation),
                ("stability_beta_action", aggregate.beta_action),
                ("stability_beta_progress", aggregate.beta_progress),
                ("stability_beta_success", aggregate.beta_success),
            ):
                row[field] = "" if value is None else float(value)
            row["stability_clean_ref_valid"] = int(aggregate.clean_ref_valid)
            row["stability_clean_ref_strategy"] = aggregate.clean_ref_strategy
            row["stability_n_clean_ref"] = int(aggregate.n_clean)
            row["stability_beta_neighborhood_valid"] = int(
                aggregate.beta_neighborhood_valid
            )
            row["stability_beta_neighborhood_missing_reason"] = (
                aggregate.missing_reason
            )
            row["stability_beta_scope"] = aggregate.beta_scope
            row.setdefault("task_family", aggregate.task_family)

        family = str(
            aggregate.task_family
            if aggregate is not None
            else row.get("task_family", "other")
        )
        family_aggregate = (
            family_aggregates.get(family)
            if family_aggregates is not None
            else None
        )
        if family_aggregate is not None:
            row["stability_beta_family"] = (
                float(family_aggregate.beta_family)
                if family_aggregate.beta_family_valid
                and family_aggregate.beta_family is not None
                else ""
            )
            row["stability_beta_family_valid"] = int(
                family_aggregate.beta_family_valid
            )
            row["stability_beta_family_n_anchors"] = int(
                family_aggregate.n_anchors
            )
            row["stability_beta_family_n_valid_anchors"] = int(
                family_aggregate.n_valid_anchors
            )
            row["stability_beta_family_missing_reason"] = (
                family_aggregate.missing_reason
            )

        perturbation_type = str(
            row.get("perturbation_type", "") or ""
        ).strip().lower()
        is_clean = perturbation_type in ("", "clean", "baseline")
        perturbation_aggregate = (
            None
            if perturbation_aggregates is None
            else perturbation_aggregates.get((anchor_id, perturbation_type))
        )
        if is_clean:
            clean_valid = bool(aggregate and aggregate.clean_ref_valid)
            clean_value: Any = 0.0 if clean_valid else ""
            for field in (
                "stability_beta_perturbation",
                "stability_beta_perturbation_trajectory",
                "stability_beta_perturbation_vv",
                "stability_beta_perturbation_out",
                "stability_beta_perturbation_oscillation",
                "stability_beta_perturbation_action",
                "stability_beta_perturbation_progress",
                "stability_beta_perturbation_success",
            ):
                row[field] = clean_value
            row["stability_beta_perturbation_valid"] = int(clean_valid)
            row["stability_beta_perturbation_scope"] = (
                "clean_reference" if clean_valid else "invalid_clean_reference"
            )
        elif perturbation_aggregate is not None:
            row["stability_beta_perturbation"] = _formal_value(
                perturbation_aggregate, perturbation_aggregate.beta_hat
            )
            row["stability_beta_perturbation_trajectory"] = _formal_value(
                perturbation_aggregate,
                perturbation_aggregate.beta_trajectory,
            )
            row["stability_beta_perturbation_vv"] = _formal_value(
                perturbation_aggregate, perturbation_aggregate.beta_vv
            )
            for field, value in (
                ("stability_beta_perturbation_out", perturbation_aggregate.beta_out),
                (
                    "stability_beta_perturbation_oscillation",
                    perturbation_aggregate.beta_oscillation,
                ),
                (
                    "stability_beta_perturbation_action",
                    perturbation_aggregate.beta_action,
                ),
                (
                    "stability_beta_perturbation_progress",
                    perturbation_aggregate.beta_progress,
                ),
                (
                    "stability_beta_perturbation_success",
                    perturbation_aggregate.beta_success,
                ),
            ):
                row[field] = "" if value is None else float(value)
            row["stability_beta_perturbation_valid"] = int(
                perturbation_aggregate.beta_neighborhood_valid
            )
            row["stability_beta_perturbation_scope"] = (
                perturbation_aggregate.beta_scope
            )

        requested_lambda = 0.0 if is_clean else _finite_float(
            row.get(
                "stress_lambda",
                row.get(
                    "ge_stress_lambda",
                    row.get("perturbation_intensity"),
                ),
            )
        )
        strict_stress_contract = bool(
            str(
                row.get("perturbation_stress_family")
                or row.get("stress_family")
                or ""
            ).strip().lower()
            == STRESS_FAMILY
            or row.get("perturbation_bank_path") not in (None, "")
        )
        stress_family = str(
            row.get("perturbation_stress_family")
            or row.get("stress_family")
            or "unresolved_stress_family"
        )
        try:
            perturbation_seed = int(float(row.get("perturbation_seed", 0) or 0))
        except (TypeError, ValueError):
            perturbation_seed = 0
        stress_aggregate = (
            None
            if stress_aggregates is None or requested_lambda is None
            else (
                (
                    stress_seed_aggregates.get(
                        (
                            anchor_id,
                            stress_family,
                            round(float(requested_lambda), 12),
                            perturbation_seed,
                        )
                    )
                    if stress_seed_aggregates is not None
                    else stress_aggregates.get(
                        (
                            anchor_id,
                            stress_family,
                            round(float(requested_lambda), 12),
                        )
                    )
                )
                if strict_stress_contract
                else stress_aggregates.get(
                    (anchor_id, round(float(requested_lambda), 12))
                )
            )
        )
        if stress_aggregate is None:
            row["stability_beta_at_lambda"] = ""
            row["stability_beta_at_lambda_valid"] = 0
            row["stability_beta_at_lambda_missing_reason"] = (
                "missing_stress_lambda"
                if requested_lambda is None
                else "missing_beta_at_requested_lambda"
            )
            row["stability_beta_at_lambda_scope"] = "invalid"
        else:
            row["stability_beta_at_lambda"] = _formal_value(
                stress_aggregate, stress_aggregate.beta_hat
            )
            row["stability_beta_at_lambda_valid"] = int(
                stress_aggregate.beta_neighborhood_valid
            )
            row["stability_beta_at_lambda_missing_reason"] = (
                stress_aggregate.missing_reason
            )
            row["stability_beta_at_lambda_scope"] = stress_aggregate.beta_scope
            row["stability_beta_at_lambda_stress_family"] = (
                stress_aggregate.stress_family
            )
            row["stability_beta_at_lambda_n_paired_seeds"] = int(
                stress_aggregate.n_paired_seeds
            )


__all__ = ["augment_rows_with_beta_aggregates"]
