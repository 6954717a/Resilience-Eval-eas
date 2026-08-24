"""Auditable perturbation dose realization.

The requested dose is an experiment control.  The realized dose is derived
from the mutation that was actually observed; it is never inferred from the
planner's downstream reward, success, or recovery response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import hashlib
from typing import Any, Dict, Mapping, Optional


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def text_sha256(text: str) -> str:
    """Return a stable digest without persisting prompt text in an audit row."""

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def text_change_components(original: str, perturbed: str) -> Dict[str, Any]:
    """Compute deterministic, outcome-independent prompt change signals."""

    original = str(original)
    perturbed = str(perturbed)
    original_tokens = original.split()
    perturbed_tokens = perturbed.split()
    edit_distance = 1.0 - SequenceMatcher(
        None,
        original,
        perturbed,
        autojunk=False,
    ).ratio()
    return {
        "changed": int(original != perturbed),
        "normalized_edit_distance": _clamp_unit(edit_distance),
        "original_char_count": len(original),
        "perturbed_char_count": len(perturbed),
        "original_token_count": len(original_tokens),
        "perturbed_token_count": len(perturbed_tokens),
        "added_token_count": max(0, len(perturbed_tokens) - len(original_tokens)),
        "token_length_ratio": (
            float(len(perturbed_tokens) / max(len(original_tokens), 1))
        ),
    }


@dataclass(frozen=True)
class PerturbationAudit:
    """One requested-dose -> realized-dose observation."""

    kind: str
    requested: float
    realized: float
    valid: bool
    reason: str = ""
    components: Mapping[str, Any] = field(default_factory=dict)
    applied: bool = False
    seed: int = 0
    episode_id: str = "unknown"
    original_hash: str = ""
    perturbed_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested", _clamp_unit(self.requested))
        object.__setattr__(self, "realized", _clamp_unit(self.realized))
        object.__setattr__(self, "components", dict(self.components))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "requested": float(self.requested),
            "realized": float(self.realized),
            "valid": bool(self.valid),
            "reason": self.reason,
            "components": dict(self.components),
            "applied": bool(self.applied),
            "seed": int(self.seed),
            "episode_id": self.episode_id,
            "original_hash": self.original_hash,
            "perturbed_hash": self.perturbed_hash,
        }


def prompt_audit(
    *,
    kind: str,
    requested: float,
    original: str,
    perturbed: str,
    metadata: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
    episode_id: str = "unknown",
) -> PerturbationAudit:
    """Build an audit whose dose comes from observed prompt operations."""

    components = text_change_components(original, perturbed)
    components.update(dict(metadata or {}))
    operation_count = int(components.get("operation_count", 0) or 0)
    max_operations = int(components.get("max_operations", 0) or 0)
    changed = bool(components["changed"])

    if requested <= 0.0:
        valid = not changed
        reason = "zero_intensity" if valid else "zero_intensity_changed_prompt"
        realized = 0.0 if valid else float(components["normalized_edit_distance"])
    elif not changed:
        valid = False
        reason = "no_prompt_change_realized"
        realized = 0.0
    else:
        valid = True
        reason = ""
        realized = (
            float(operation_count / max_operations)
            if max_operations > 0
            else float(components["normalized_edit_distance"])
        )

    return PerturbationAudit(
        kind=kind,
        requested=requested,
        realized=realized,
        valid=valid,
        reason=reason,
        components=components,
        applied=changed,
        seed=seed,
        episode_id=episode_id,
        original_hash=text_sha256(original),
        perturbed_hash=text_sha256(perturbed),
    )
