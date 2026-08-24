"""Canonical proposition-outcome extraction for evaluation artifacts.

The Habitat measure ``auto_eval_proposition_tracker`` is the source of truth
for proposition completion.  Keeping the conversion in one module prevents
the runner, critic exports, and CSV reporting from silently using different
definitions of completion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Tuple


PROPOSITION_OUTCOME_SOURCE = "auto_eval_proposition_tracker"
MISSING_PROPOSITION_OUTCOME_SOURCE = "missing:auto_eval_proposition_tracker"


@dataclass(frozen=True)
class PropositionOutcome:
    """Materialized completion state for one environment observation."""

    satisfied_count: int
    total: int
    fraction: Optional[float]
    satisfied_at: Tuple[Any, ...] = ()

    @property
    def valid(self) -> bool:
        return self.fraction is not None

    def info_fields(self, *, include_tracker: bool = False) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "proposition_satisfied_count": self.satisfied_count,
            "proposition_total": self.total,
            "proposition_satisfied_fraction": self.fraction,
            "proposition_satisfied_fraction_source": (
                PROPOSITION_OUTCOME_SOURCE
                if self.valid
                else MISSING_PROPOSITION_OUTCOME_SOURCE
            ),
        }
        if self.satisfied_at:
            satisfied_at = list(self.satisfied_at)
            fields["proposition_satisfied_at"] = satisfied_at
            if include_tracker:
                fields["auto_eval_proposition_tracker"] = {
                    "proposition_satisfied_at": satisfied_at,
                }
        return fields


def proposition_outcome_from_info(
    info: Optional[Mapping[str, Any]],
    task_context: Optional[Mapping[str, Any]] = None,
) -> PropositionOutcome:
    """Extract completion exclusively from ``auto_eval_proposition_tracker``.

    ``proposition_count`` is retained only as a dimensional fallback for state
    encoding.  Without the tracker there is no observed completion fraction,
    so the formal outcome remains invalid instead of being fabricated as zero.
    """

    tracker = {}
    if isinstance(info, Mapping):
        candidate = info.get("auto_eval_proposition_tracker", {}) or {}
        if isinstance(candidate, Mapping):
            tracker = candidate
    tracker_has_frontier = "proposition_satisfied_at" in tracker
    satisfied_at = tracker.get("proposition_satisfied_at")
    if tracker_has_frontier and isinstance(satisfied_at, (list, tuple)):
        values = tuple(satisfied_at)

        def _is_satisfied(value: Any) -> bool:
            if value is None:
                return False
            try:
                return int(value) >= 0
            except (TypeError, ValueError):
                return False

        total = len(values)
        satisfied = sum(1 for value in values if _is_satisfied(value))
        fraction = float(satisfied / total) if total > 0 else None
        return PropositionOutcome(satisfied, total, fraction, values)

    if (
        not tracker_has_frontier
        and isinstance(info, Mapping)
        and info.get("proposition_satisfied_fraction_source")
        == PROPOSITION_OUTCOME_SOURCE
    ):
        # A reporting/serialization boundary may intentionally trim the bulky
        # tracker after the runner has materialized its canonical scalar. The
        # source marker distinguishes that value from an unverified fallback.
        try:
            existing_fraction = float(info.get("proposition_satisfied_fraction"))
        except (TypeError, ValueError):
            existing_fraction = math.nan
        if math.isfinite(existing_fraction) and 0.0 <= existing_fraction <= 1.0:
            try:
                existing_count = int(info.get("proposition_satisfied_count", 0))
                existing_total = int(info.get("proposition_total", 0))
            except (TypeError, ValueError):
                existing_count = 0
                existing_total = 0
            return PropositionOutcome(
                existing_count,
                existing_total,
                existing_fraction,
            )

    # A missing tracker frontier has no observed outcome, but the declared
    # proposition count is still useful as an input dimension for the state
    # encoder.  Keep ``fraction=None`` so reporting never turns absence into a
    # fabricated zero-completion observation.
    proposition_count = 0
    if isinstance(task_context, Mapping):
        proposition_count = int(task_context.get("proposition_count") or 0)
    elif isinstance(info, Mapping):
        proposition_count = int(info.get("proposition_count") or 0)
    return PropositionOutcome(0, proposition_count, None)


def materialize_proposition_outcome(
    info: Dict[str, Any],
    *,
    include_tracker: bool = False,
) -> PropositionOutcome:
    """Write the canonical scalar outcome fields into an ``info`` mapping."""

    outcome = proposition_outcome_from_info(info)
    info.update(outcome.info_fields(include_tracker=include_tracker))
    return outcome


__all__ = [
    "PropositionOutcome",
    "PROPOSITION_OUTCOME_SOURCE",
    "materialize_proposition_outcome",
    "proposition_outcome_from_info",
]
