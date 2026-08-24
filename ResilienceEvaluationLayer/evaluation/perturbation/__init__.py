"""
Perturbation package for Stability (E2) and Graceful Extensibility (E3).

This package contains the runtime :class:`PerturbationInjector` that modifies
either the shared :class:`WorldGraph` / env state (`world_level` perturbations)
or the LLM planner prompt stream (`prompt_level` paraphrase rewrites), as well
as a static paraphrase bank that keeps the prompt-level perturbations
deterministic (avoiding runtime LLM non-determinism).
"""

from .audit import PerturbationAudit
from .bank import (
    STRESS_FAMILY,
    STRESS_GRID,
    FrozenPerturbationBank,
    FrozenPerturbationBankError,
    FrozenPerturbationEntry,
)
from .calibrator import CandidateAssessment, LLMCandidateCalibrator
from .injector import (
    PerturbationInjector,
    PerturbationSpec,
    build_perturbation_spec_grid,
    derive_run_seed,
    spec_artifact_key,
)
from .paraphrase_bank import ParaphraseBank

__all__ = [
    "PerturbationInjector",
    "PerturbationSpec",
    "PerturbationAudit",
    "FrozenPerturbationBank",
    "FrozenPerturbationBankError",
    "FrozenPerturbationEntry",
    "STRESS_FAMILY",
    "STRESS_GRID",
    "ParaphraseBank",
    "CandidateAssessment",
    "LLMCandidateCalibrator",
    "build_perturbation_spec_grid",
    "derive_run_seed",
    "spec_artifact_key",
]
