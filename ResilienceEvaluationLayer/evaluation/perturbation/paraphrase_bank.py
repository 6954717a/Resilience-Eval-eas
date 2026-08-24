#!/usr/bin/env python3

r"""
Deterministic paraphrase bank for prompt-level perturbations.

Keeping the paraphrases offline removes the non-determinism that would come
from asking an LLM to rewrite the instruction at runtime, and it makes the
E2 (:math:`\hat\beta`) experiment fully reproducible across seeds.

Two modes are supported:

* ``surface_rewrite``    — word-level synonym / ordering rewrites that keep
                           meaning identical (semantic equivalence); used for
                           the "semantic-equivalent rewriting" half of the
                           prompt perturbation family.
* ``filler_injection``   — leave the task unchanged but inject benign filler
                           (e.g. "please be thorough") that could distract a
                           weak planner without changing the contract.

If a per-anchor ``overrides`` dict is provided, it takes precedence over the
default rules; this lets callers pin in hand-authored paraphrases for
specific episodes (recommended for the paper's headline numbers).
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


# Deterministic synonym replacement rules. Keys are case-insensitive source
# phrases; values are the canonical rewrite. Ordering is significant — longer
# phrases are tried first.
_SURFACE_REWRITES: List[tuple] = [
    ("please", "kindly"),
    ("could you", "would you be able to"),
    ("move the", "relocate the"),
    ("pick up", "grab"),
    ("put", "place"),
    ("on top of", "onto"),
    ("inside the", "into the"),
    ("then", "after that"),
    ("and then", "and afterwards"),
    # ("living room", "lounge"),
    # ("kitchen", "cooking area"),
    # ("bedroom", "sleeping room"),
    # ("bathroom", "restroom"),
    ("clean up", "tidy up"),
    ("turn on", "switch on"),
    ("turn off", "switch off"),
]

_FILLERS: List[str] = [
    "Please be thorough and careful.",
    "Note that time is not critical.",
    "Your goal is to complete the task safely.",
    "Remember to double-check the final placement.",
    "Work at your own pace.",
]


@dataclass
class ParaphraseBank:
    """Deterministic instruction paraphraser."""

    overrides: Dict[str, List[str]] = field(default_factory=dict)
    extra_fillers: List[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ParaphraseBank":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            return cls()
        overrides = payload.get("overrides", {}) or {}
        extras = payload.get("extra_fillers", []) or []
        return cls(
            overrides={
                str(k): [str(p) for p in (v or [])] for k, v in overrides.items()
            },
            extra_fillers=[str(x) for x in extras],
        )

    # ------------------------------------------------------------------
    # Rewriting helpers
    # ------------------------------------------------------------------
    def rewrite(
        self,
        instruction: str,
        mode: str = "surface_rewrite",
        anchor_id: Optional[str] = None,
        seed: int = 0,
        intensity: float = 1.0,
    ) -> str:
        """Return a paraphrased version of ``instruction``.

        The rewrite is deterministic given ``anchor_id`` and ``seed``.  For a
        fixed input and seed, the number of operations is non-decreasing in
        ``intensity``.  Zero intensity is always an exact no-op.
        """
        rewritten, _metadata = self.rewrite_with_metadata(
            instruction,
            mode=mode,
            anchor_id=anchor_id,
            seed=seed,
            intensity=intensity,
        )
        return rewritten

    def rewrite_with_metadata(
        self,
        instruction: str,
        mode: str = "surface_rewrite",
        anchor_id: Optional[str] = None,
        seed: int = 0,
        intensity: float = 1.0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Return the rewrite and the operations used to realize its dose."""

        if not isinstance(instruction, str) or not instruction.strip():
            return instruction, {
                "operation_count": 0,
                "max_operations": 0,
                "rewrite_source": "empty_instruction",
            }
        intensity = _validate_intensity(intensity)
        if intensity <= 0.0:
            return instruction, {
                "operation_count": 0,
                "max_operations": 0,
                "rewrite_source": "zero_intensity",
            }

        # Overrides are calibrated full-dose candidates. Partial doses use the
        # compositional rules below so an override is never silently presented
        # as a fractional perturbation.
        if intensity >= 1.0 and anchor_id and anchor_id in self.overrides:
            options = list(self.overrides[anchor_id])
            if options:
                idx = seed % len(options)
                return options[idx], {
                    "operation_count": 1,
                    "max_operations": 1,
                    "rewrite_source": "anchor_override",
                }

        rng = random.Random(f"{anchor_id or ''}::{seed}::{mode}")
        if mode == "surface_rewrite":
            return self._surface_rewrite(instruction, intensity)
        if mode == "filler_injection":
            return self._filler_injection(instruction, rng, intensity)
        raise ValueError(f"Unknown paraphrase mode: {mode!r}")

    # ------------------------------------------------------------------
    def _surface_rewrite(
        self,
        instruction: str,
        intensity: float,
    ) -> Tuple[str, Dict[str, Any]]:
        # Materialize a prefix of the full rewrite sequence. Prefix selection
        # makes operation count monotone for every fixed instruction.
        rules = sorted(_SURFACE_REWRITES, key=lambda p: len(p[0]), reverse=True)
        steps: List[str] = []
        output = instruction
        for source, target in rules:
            if len(steps) >= 3:
                break
            pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
            new_output, n_sub = pattern.subn(target, output, count=1)
            if n_sub > 0:
                output = new_output
                steps.append(output)
        if not steps:
            # Guarantee at least a trivial, meaning-preserving rewrite.
            output = output.strip()
            if not output.endswith("."):
                output = output + "."
            output = "Instruction: " + output
            steps.append(output)
        operation_count = max(1, int(math.ceil(intensity * len(steps) - 1.0e-12)))
        operation_count = min(operation_count, len(steps))
        return steps[operation_count - 1], {
            "operation_count": operation_count,
            "max_operations": len(steps),
            "rewrite_source": "surface_rules",
        }

    def _filler_injection(
        self,
        instruction: str,
        rng: random.Random,
        intensity: float,
    ) -> Tuple[str, Dict[str, Any]]:
        pool = list(_FILLERS) + list(self.extra_fillers)
        if not pool:
            return instruction, {
                "operation_count": 0,
                "max_operations": 0,
                "rewrite_source": "empty_filler_pool",
            }
        max_fillers = min(2, len(pool))
        n = max(1, int(math.ceil(intensity * max_fillers - 1.0e-12)))
        ranked = list(pool)
        rng.shuffle(ranked)
        picks = ranked[: min(n, max_fillers)]
        suffix = " ".join(picks)
        output = instruction.rstrip()
        if not output.endswith("."):
            output = output + "."
        return f"{output} {suffix}", {
            "operation_count": len(picks),
            "max_operations": max_fillers,
            "rewrite_source": "filler_bank",
            "filler_count": len(picks),
        }


def _validate_intensity(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("intensity must be a real number in [0, 1]")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"intensity must be finite and in [0, 1]; got {value!r}")
    return parsed
