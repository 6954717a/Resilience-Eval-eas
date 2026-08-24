"""Optional LLM-assisted prompt candidate generation and validation.

No client is created here.  Callers may inject a strict-JSON completion
function for an offline calibration job; normal evaluation never calls the
network.  LLM output can reject a semantically unsafe candidate, but realized
stress is always computed from deterministic observed-change signals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .audit import PerturbationAudit, prompt_audit


JsonResult = Union[str, Mapping[str, Any]]
CompletionFn = Callable[[Sequence[Mapping[str, str]]], JsonResult]


@dataclass(frozen=True)
class CandidateAssessment:
    text: str
    audit: PerturbationAudit
    llm_checked: bool = False
    llm_semantic_equivalent: Optional[bool] = None
    violations: Sequence[str] = ()
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "audit": self.audit.to_dict(),
            "llm_checked": self.llm_checked,
            "llm_semantic_equivalent": self.llm_semantic_equivalent,
            "violations": list(self.violations),
            "rationale": self.rationale,
        }


class LLMCandidateCalibrator:
    """Generate/validate candidates through an injected strict-JSON client."""

    def __init__(
        self,
        completion_fn: Optional[CompletionFn] = None,
        *,
        model_identity: str = "",
    ) -> None:
        self.completion_fn = completion_fn
        self.model_identity = str(model_identity)

    @property
    def enabled(self) -> bool:
        return self.completion_fn is not None

    def generate_candidates(
        self,
        instruction: str,
        *,
        kind: str,
        requested: float,
        task_contract: Optional[Mapping[str, Any]] = None,
        count: int = 3,
    ) -> List[str]:
        """Return offline candidates, or ``[]`` when no client was injected."""

        if self.completion_fn is None:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate semantics-preserving perturbation candidates. "
                    "Return strict JSON with exactly one key, candidates, whose "
                    "value is a list of strings. Preserve every entity, quantity, "
                    "relation, negation, and temporal constraint. Do not score stress."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": instruction,
                        "kind": kind,
                        "requested_dose": float(requested),
                        "candidate_count": int(max(1, count)),
                        "task_contract": dict(task_contract or {}),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        payload = _strict_object(self.completion_fn(messages), {"candidates"})
        candidates = payload["candidates"]
        if not isinstance(candidates, list) or not all(
            isinstance(value, str) and value.strip() for value in candidates
        ):
            raise ValueError("candidates must be a list of non-empty strings")
        return [str(value) for value in candidates[: max(1, int(count))]]

    def assess_candidate(
        self,
        instruction: str,
        candidate: str,
        *,
        kind: str,
        requested: float,
        required_terms: Sequence[str] = (),
        task_contract: Optional[Mapping[str, Any]] = None,
        seed: int = 0,
        episode_id: str = "unknown",
    ) -> CandidateAssessment:
        """Apply deterministic invariants, then optional LLM semantic review."""

        base_audit = prompt_audit(
            kind=kind,
            requested=requested,
            original=instruction,
            perturbed=candidate,
            metadata={"dose_source": "deterministic_text_change"},
            seed=seed,
            episode_id=episode_id,
        )
        candidate_folded = candidate.casefold()
        missing_terms = [
            str(term)
            for term in required_terms
            if str(term).casefold() not in candidate_folded
        ]
        local_valid = base_audit.valid and not missing_terms
        local_reason = (
            base_audit.reason
            if not base_audit.valid
            else (
                "missing_required_terms:" + ",".join(missing_terms)
                if missing_terms
                else ""
            )
        )

        llm_checked = False
        semantic_equivalent: Optional[bool] = None
        violations: List[str] = []
        rationale = ""
        if self.completion_fn is not None:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Audit semantic equivalence. Return strict JSON with exactly "
                        "semantic_equivalent (bool), violations (list[str]), and "
                        "rationale (str). The task contract and symbolic invariants "
                        "cannot be overridden by this review."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original": instruction,
                            "candidate": candidate,
                            "kind": kind,
                            "task_contract": dict(task_contract or {}),
                            "required_terms": list(required_terms),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ]
            payload = _strict_object(
                self.completion_fn(messages),
                {"semantic_equivalent", "violations", "rationale"},
            )
            semantic_equivalent = payload["semantic_equivalent"]
            violations_raw = payload["violations"]
            rationale_raw = payload["rationale"]
            if not isinstance(semantic_equivalent, bool):
                raise ValueError("semantic_equivalent must be bool")
            if not isinstance(violations_raw, list) or not all(
                isinstance(value, str) for value in violations_raw
            ):
                raise ValueError("violations must be list[str]")
            if not isinstance(rationale_raw, str):
                raise ValueError("rationale must be str")
            llm_checked = True
            violations = list(violations_raw)
            rationale = rationale_raw

        valid = local_valid and semantic_equivalent is not False
        if not local_valid:
            reason = local_reason
        elif semantic_equivalent is False:
            reason = "llm_semantic_rejection"
        else:
            reason = ""
        components = dict(base_audit.components)
        components.update(
            {
                "required_term_count": len(required_terms),
                "missing_required_terms": missing_terms,
                "llm_semantic_checked": llm_checked,
                "llm_model_identity": self.model_identity,
            }
        )
        # Deliberately preserve base_audit.realized: an LLM never supplies the
        # perturbation norm or outcome-response magnitude.
        audit = replace(
            base_audit,
            valid=valid,
            reason=reason,
            components=components,
        )
        return CandidateAssessment(
            text=candidate,
            audit=audit,
            llm_checked=llm_checked,
            llm_semantic_equivalent=semantic_equivalent,
            violations=tuple(violations),
            rationale=rationale,
        )


def _strict_object(value: JsonResult, expected_keys: set) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("completion did not return valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("completion must return a JSON object")
    keys = set(value.keys())
    if keys != expected_keys:
        raise ValueError(
            f"completion keys must be exactly {sorted(expected_keys)}; got {sorted(keys)}"
        )
    return dict(value)
