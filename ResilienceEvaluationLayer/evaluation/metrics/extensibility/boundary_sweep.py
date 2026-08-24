#!/usr/bin/env python3

"""
Boundary sweep generator
========================

Generates the parameterized stress path ``θ(λ) | λ ∈ [0,1]`` used by the
AUDC / Contract-Margin experiment (§3.2 of the resilience design doc).

The sweep is intentionally declarative: it only produces
:class:`~habitat_llm.evaluation.perturbation.PerturbationSpec` instances; the
actual execution belongs to the experiment runner (``planner_demo_resilience``
/ ``planner_demo_evolve``). This keeps the stress path reproducible via
config (``conf/evaluation/resilience_config.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from habitat_llm.evaluation.perturbation import PerturbationSpec


@dataclass
class BoundarySweep:
    r"""Parameterized stress sweep along ``lambda`` for a chosen perturbation kind.

    Attributes
    ----------
    kind:
        Perturbation family used as the stress axis (defaults to
        ``irrelevant_perturbation`` per §3.2 of the design doc).
    lambdas:
        Stress grid :math:`\{\lambda_i\}` in ``[0, 1]``. Must be sorted
        ascending.
    seeds:
        Repeat seeds per lambda value.
    """

    kind: str = "irrelevant_perturbation"
    lambdas: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    seeds: Tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        lambdas = tuple(sorted(float(v) for v in self.lambdas))
        for v in lambdas:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"lambda must be in [0,1]; got {v!r}")
        self.lambdas = lambdas
        self.seeds = tuple(int(s) for s in self.seeds)

    def iter_specs(self) -> List[PerturbationSpec]:
        """Produce one :class:`PerturbationSpec` per ``(λ, seed)`` pair."""
        specs: List[PerturbationSpec] = []
        for lam in self.lambdas:
            if lam <= 0.0:
                # λ = 0 is the clean reference (intensity = 0 → no mutation).
                for seed in self.seeds:
                    specs.append(
                        PerturbationSpec(
                            kind="clean",
                            seed=int(seed),
                            intensity=0.0,
                        )
                    )
            else:
                for seed in self.seeds:
                    specs.append(
                        PerturbationSpec(
                            kind=self.kind,
                            seed=int(seed),
                            intensity=float(lam),
                        )
                    )
        return specs

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def num_runs(self) -> int:
        return len(self.lambdas) * len(self.seeds)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "lambdas": list(self.lambdas),
            "seeds": list(self.seeds),
            "num_runs": self.num_runs,
        }
