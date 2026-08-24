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
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union


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
    ) -> str:
        """Return a paraphrased version of ``instruction``.

        The rewrite is deterministic given ``anchor_id`` and ``seed``.
        """
        if not isinstance(instruction, str) or not instruction.strip():
            return instruction
        # Per-anchor overrides win when available.
        if anchor_id and anchor_id in self.overrides:
            options = list(self.overrides[anchor_id])
            if options:
                idx = seed % len(options)
                return options[idx]

        rng = random.Random(f"{anchor_id or ''}::{seed}::{mode}")
        if mode == "surface_rewrite":
            return self._surface_rewrite(instruction, rng)
        if mode == "filler_injection":
            return self._filler_injection(instruction, rng)
        raise ValueError(f"Unknown paraphrase mode: {mode!r}")

    # ------------------------------------------------------------------
    def _surface_rewrite(self, instruction: str, rng: random.Random) -> str:
        # Try each rule in order (longest first); replace only the first match.
        rules = sorted(_SURFACE_REWRITES, key=lambda p: len(p[0]), reverse=True)
        applied = 0
        # Aim for 1-3 rewrites; keeps rewrites truthfully paraphrasic.
        max_rewrites = rng.randint(1, 3)
        output = instruction
        for source, target in rules:
            if applied >= max_rewrites:
                break
            pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
            new_output, n_sub = pattern.subn(target, output, count=1)
            if n_sub > 0:
                output = new_output
                applied += 1
        if applied == 0:
            # Guarantee at least a trivial, meaning-preserving rewrite.
            output = output.strip()
            if not output.endswith("."):
                output = output + "."
            output = "Instruction: " + output
        return output

    def _filler_injection(self, instruction: str, rng: random.Random) -> str:
        pool = list(_FILLERS) + list(self.extra_fillers)
        if not pool:
            return instruction
        n = rng.randint(1, 2)
        picks = rng.sample(pool, k=min(n, len(pool)))
        suffix = " ".join(picks)
        output = instruction.rstrip()
        if not output.endswith("."):
            output = output + "."
        return f"{output} {suffix}"
