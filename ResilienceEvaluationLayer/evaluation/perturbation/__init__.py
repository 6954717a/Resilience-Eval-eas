"""
Perturbation package for Stability (E2) and Graceful Extensibility (E3).

This package contains the runtime :class:`PerturbationInjector` that modifies
either the shared :class:`WorldGraph` / env state (`world_level` perturbations)
or the LLM planner prompt stream (`prompt_level` paraphrase rewrites), as well
as a static paraphrase bank that keeps the prompt-level perturbations
deterministic (avoiding runtime LLM non-determinism).
"""

from .injector import PerturbationInjector, PerturbationSpec
from .paraphrase_bank import ParaphraseBank

__all__ = [
    "PerturbationInjector",
    "PerturbationSpec",
    "ParaphraseBank",
]
