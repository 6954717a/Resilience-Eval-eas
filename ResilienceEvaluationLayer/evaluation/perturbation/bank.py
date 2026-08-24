"""Frozen instruction/context stress-bank support.

The bank is generated before an evaluation run.  Runtime code only reads the
materialized sidecar and therefore never needs to call an LLM.  This module is
deliberately independent from the perturbation injector so it can also be used
by the offline dataset-generation command.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


BANK_SCHEMA_VERSION = "frozen_perturbation_bank_v6"
VARIANT_SCHEMA_VERSION = "frozen_perturbation_variant_v6"
STRESS_FAMILY = "llm_instruction_context_stress"
STRESS_GRID: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TEXT_TOKENIZER_VERSION = "resilience_text_v2"
QUALITY_CONTRACT_VERSION = "instruction_context_quality_v3"
CONTEXT_SCHEDULE_VERSION = "partnr_hierarchy_balanced_v4"
PARTNR_HIERARCHY_LEVELS: Tuple[str, ...] = ("room", "furniture", "object")
CONTEXT_SEPARATOR = "\n\nScene context: "
# Text dose is a manipulation check, while lambda remains the exact k/5
# intervention coordinate. A six-point cumulative path quantizes lexical edit
# counts, so 0.05 rejected otherwise well-balanced integer budgets by only one
# token. The direct dose bound remains primary; the ratio is a secondary guard
# against one operation dominating the path.
DEFAULT_TEXT_DOSE_TOLERANCE = 0.075
DEFAULT_UNIT_BUDGET_RATIO_LIMIT = 1.50

_TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*")
_REPEATED_PUNCTUATION_RE = re.compile(r"([^\w\s])\1{2,}")
_META_LANGUAGE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("this_wording", re.compile(r"\bthis wording\b", re.I)),
    (
        "preserves_request",
        re.compile(
            r"\bpreserv(?:e|es|ing)\b.{0,24}"
            r"\b(?:request|task|sequence|meaning)\b",
            re.I,
        ),
    ),
    ("keeps_unchanged", re.compile(r"\bkeep(?:s|ing)?\b.{0,24}\bunchanged\b", re.I)),
    (
        "adds_no_condition",
        re.compile(
            r"\badd(?:s|ing)?\s+no\s+(?:condition|goal|requirement)\b",
            re.I,
        ),
    ),
    ("same_task", re.compile(r"\b(?:same|original)\s+task\b", re.I)),
)
_MANNER_MODIFIERS = {
    "carefully",
    "cautiously",
    "diligently",
    "efficiently",
    "gently",
    "methodically",
    "precisely",
    "properly",
    "responsibly",
    "safely",
    "securely",
    "systematically",
    "thoroughly",
}


class FrozenPerturbationBankError(ValueError):
    """A bank lookup or sidecar-schema error with a machine-readable reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        message = self.reason if not self.detail else f"{self.reason}: {self.detail}"
        super().__init__(message)


def resilience_text_v2_tokenize(text: str) -> Tuple[str, ...]:
    """Return the frozen lexical tokens used by the v3 stress contract.

    NFKC normalization and case folding make capitalization irrelevant. Entity
    handles joined by ``_``/``-`` remain one token. Pure punctuation is
    deliberately excluded so brackets and decoration cannot manufacture text
    dose without adding lexical information.
    """

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return tuple(_TEXT_TOKEN_RE.findall(normalized))


def resilience_text_v1_tokenize(text: str) -> Tuple[str, ...]:
    """Deprecated source-compatible alias for the active frozen tokenizer."""

    return resilience_text_v2_tokenize(text)


def _bracket_errors(text: str) -> List[str]:
    pairs = {")": "(", "]": "[", "}": "{"}
    openings = set(pairs.values())
    stack: List[str] = []
    maximum_depth = 0
    for character in str(text):
        if character in openings:
            stack.append(character)
            maximum_depth = max(maximum_depth, len(stack))
        elif character in pairs:
            if not stack or stack[-1] != pairs[character]:
                return ["unbalanced_brackets"]
            stack.pop()
    errors: List[str] = []
    if stack:
        errors.append("unbalanced_brackets")
    if maximum_depth > 1:
        errors.append("nested_brackets")
    return errors


def normalize_rewrite_text(text: str) -> str:
    """Normalize a rewrite for variant-independence comparisons."""

    return " ".join(resilience_text_v2_tokenize(text))


def _has_repeated_clause(text: str) -> bool:
    clauses = [
        normalize_rewrite_text(value)
        for value in re.split(r"[.!?;]+", str(text))
        if normalize_rewrite_text(value)
    ]
    return len(clauses) != len(set(clauses))


def rewrite_edit_signature(source: str, target: str) -> Tuple[Any, ...]:
    """Return the lexical edit script, excluding unchanged token spans."""

    source_tokens = resilience_text_v2_tokenize(source)
    target_tokens = resilience_text_v2_tokenize(target)
    matcher = SequenceMatcher(None, source_tokens, target_tokens, autojunk=False)
    return tuple(
        (
            tag,
            tuple(source_tokens[source_start:source_end]),
            tuple(target_tokens[target_start:target_end]),
        )
        for tag, source_start, source_end, target_start, target_end
        in matcher.get_opcodes()
        if tag != "equal"
    )


def assess_rewrite_quality(
    canonical_instruction: str,
    rewrite_levels: Sequence[str],
) -> Dict[str, Any]:
    """Apply deterministic natural-language and anti-padding hard gates."""

    if len(rewrite_levels) != 2:
        return {
            "contract_version": QUALITY_CONTRACT_VERSION,
            "valid": False,
            "errors": ["rewrite_levels_invalid"],
            "per_level": [],
        }
    canonical = str(canonical_instruction).strip()
    canonical_tokens = resilience_text_v2_tokenize(canonical)
    canonical_modifier_counts = Counter(
        token for token in canonical_tokens if token in _MANNER_MODIFIERS
    )
    errors: List[str] = []
    per_level: List[Dict[str, Any]] = []
    for index, raw_rewrite in enumerate(rewrite_levels, start=1):
        rewrite = str(raw_rewrite).strip()
        tokens = resilience_text_v2_tokenize(rewrite)
        level_errors: List[str] = []
        if _REPEATED_PUNCTUATION_RE.search(rewrite):
            level_errors.append("repeated_punctuation_padding")
        level_errors.extend(_bracket_errors(rewrite))
        if _has_repeated_clause(rewrite):
            level_errors.append("repeated_clause_padding")
        for label, pattern in _META_LANGUAGE_PATTERNS:
            if pattern.search(rewrite):
                level_errors.append(f"meta_language:{label}")
        modifier_counts = Counter(
            token for token in tokens if token in _MANNER_MODIFIERS
        )
        added_modifier_count = sum(
            max(0, count - canonical_modifier_counts.get(token, 0))
            for token, count in modifier_counts.items()
        )
        if added_modifier_count > 0:
            level_errors.append("added_manner_modifier")
        if added_modifier_count >= 3:
            level_errors.append("manner_modifier_stuffing")
        length_ratio = float(len(tokens)) / float(max(len(canonical_tokens), 1))
        if length_ratio < 0.55:
            level_errors.append("rewrite_too_short")
        if length_ratio > 1.75:
            level_errors.append("rewrite_too_long")
        level_errors = list(dict.fromkeys(level_errors))
        errors.extend(f"rewrite_{index}:{error}" for error in level_errors)
        per_level.append(
            {
                "level": index,
                "lexical_token_count": len(tokens),
                "canonical_length_ratio": length_ratio,
                "added_manner_modifier_count": added_modifier_count,
                "errors": level_errors,
            }
        )
    normalized = [normalize_rewrite_text(value) for value in rewrite_levels]
    canonical_normalized = normalize_rewrite_text(canonical)
    if not normalized[0] or not normalized[1]:
        errors.append("rewrite_empty_after_normalization")
    if normalized[0] == canonical_normalized:
        errors.append("rewrite_1_not_lexically_distinct")
    if normalized[1] in {canonical_normalized, normalized[0]}:
        errors.append("rewrite_2_not_lexically_distinct")
    errors = list(dict.fromkeys(errors))
    return {
        "contract_version": QUALITY_CONTRACT_VERSION,
        "valid": not errors,
        "errors": errors,
        "per_level": per_level,
    }


def _plain_contract_value(value: Any) -> Any:
    """Convert dataset/runtime episode values to comparable JSON-like data."""

    if isinstance(value, Mapping):
        return {
            str(key): _plain_contract_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_contract_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain_contract_value(getattr(value, field.name))
            for field in fields(value)
        }
    getstate = getattr(value, "__getstate__", None)
    if callable(getstate):
        state = getstate()
        if isinstance(state, Mapping):
            return _plain_contract_value(state)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def _episode_value(episode: Any, key: str, default: Any = None) -> Any:
    if isinstance(episode, Mapping):
        return episode.get(key, default)
    return getattr(episode, key, default)


def _iter_contract_strings(value: Any) -> Sequence[str]:
    strings: List[str] = []
    if isinstance(value, str):
        if value.strip():
            strings.append(value.strip())
    elif isinstance(value, Mapping):
        for nested in value.values():
            strings.extend(_iter_contract_strings(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            strings.extend(_iter_contract_strings(nested))
    return strings


def _entity_aliases(value: str) -> Sequence[str]:
    raw = str(value).strip()
    if not raw:
        return ()
    candidates = {raw}
    without_pipe = raw.split("|", 1)[0].strip()
    candidates.add(without_pipe)
    name = Path(without_pipe.replace("\\", "/")).name
    for suffix in (".object_config.json", ".scene_instance.json", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    candidates.add(name)
    candidates.add(re.sub(r"(?::|_)[0-9]+$", "", name))
    return tuple(
        candidate.strip()
        for candidate in candidates
        if len(candidate.strip()) >= 2
        and not candidate.startswith(("data/", "./"))
    )


def _contains_entity(text: str, entity: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(entity)}(?![A-Za-z0-9_])",
        str(text),
        flags=re.I,
    ) is not None


def build_episode_task_contract(episode: Any) -> Dict[str, Any]:
    """Build the path-addressed task witness from JSON or a runtime episode.

    The comparison is structural and intentionally uses neither hashes nor
    generated IDs. It covers the task graph and the scene facts from which the
    three non-task context units are selected.
    """

    if episode is None:
        raise ValueError("episode is required to build the task contract")
    instruction = _episode_value(episode, "instruction", "")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("episode instruction must be a non-empty string")

    propositions = _plain_contract_value(
        _episode_value(episode, "evaluation_propositions", [])
    )
    dependencies = _plain_contract_value(
        _episode_value(episode, "evaluation_proposition_dependencies", [])
    )
    constraints = _plain_contract_value(
        _episode_value(episode, "evaluation_constraints", [])
    )
    object_states = _plain_contract_value(
        _episode_value(episode, "object_states", {})
    )

    goal_values: List[str] = []
    if isinstance(propositions, list):
        for proposition in propositions:
            if isinstance(proposition, Mapping):
                goal_values.extend(
                    _iter_contract_strings(proposition.get("args", {}))
                )
    goal_entities = {
        alias for value in goal_values for alias in _entity_aliases(value)
    }

    scene_values: List[str] = []
    info = _plain_contract_value(_episode_value(episode, "info", {}))
    initial_state: Any = []
    initial_furniture_state: Any = []
    if isinstance(info, Mapping):
        extra_info = info.get("extra_info", {})
        if isinstance(extra_info, Mapping):
            initial_state = extra_info.get("initial_state", [])
            initial_furniture_state = extra_info.get(
                "initial_furniture_state", []
            )
        if not initial_state:
            initial_state = info.get("initial_state", [])
        if not initial_furniture_state:
            initial_furniture_state = info.get("initial_furniture_state", [])
        if isinstance(initial_state, list):
            for element in initial_state:
                if not isinstance(element, Mapping):
                    continue
                for key in (
                    "object_classes",
                    "allowed_regions",
                    "furniture_names",
                    "name",
                ):
                    scene_values.extend(_iter_contract_strings(element.get(key, [])))
        if isinstance(initial_furniture_state, list):
            for element in initial_furniture_state:
                if not isinstance(element, Mapping):
                    continue
                for key in (
                    "furniture_classes",
                    "furniture_names",
                    "allowed_regions",
                    "name",
                ):
                    scene_values.extend(_iter_contract_strings(element.get(key, [])))
    name_to_receptacle = _plain_contract_value(
        _episode_value(episode, "name_to_receptacle", {})
    )
    if isinstance(name_to_receptacle, Mapping):
        scene_values.extend(str(key) for key in name_to_receptacle)
        scene_values.extend(_iter_contract_strings(name_to_receptacle))
    scene_values.extend(
        _iter_contract_strings(
            _plain_contract_value(_episode_value(episode, "rigid_objs", []))
        )
    )
    scene_values.extend(
        _iter_contract_strings(
            _plain_contract_value(_episode_value(episode, "markers", []))
        )
    )
    if isinstance(object_states, Mapping):
        for state_values in object_states.values():
            if isinstance(state_values, Mapping):
                scene_values.extend(str(key) for key in state_values)
    scene_entities = {
        alias for value in scene_values for alias in _entity_aliases(value)
    }
    folded_goals = {value.casefold() for value in goal_entities}
    instance_alias_bases = {
        re.sub(r"(?::|_)[0-9]+$", "", entity).casefold()
        for entity in scene_entities
        if re.search(r"(?::|_)[0-9]+$", entity)
    }
    allowed_context_entities = sorted(
        entity
        for entity in scene_entities
        if entity.casefold() not in folded_goals
        and not _contains_entity(instruction, entity)
        # Do not let one physical object occupy multiple context units through
        # both its class alias (``book``) and instance handle (``book_2``).
        and not (
            entity.casefold() in instance_alias_bases
            and not re.search(r"(?::|_)[0-9]+$", entity)
        )
    )[:100]
    protected_terms = sorted(
        entity for entity in goal_entities if _contains_entity(instruction, entity)
    )
    return {
        "canonical_instruction": instruction,
        "scene_id": str(_episode_value(episode, "scene_id", "")),
        "evaluation_propositions": propositions,
        "evaluation_dependencies": dependencies,
        "evaluation_constraints": constraints,
        "object_states": object_states,
        "episode_initial_state": _plain_contract_value(initial_state),
        "episode_initial_furniture_state": _plain_contract_value(
            initial_furniture_state
        ),
        "name_to_receptacle": name_to_receptacle,
        "task_goal_entities": sorted(goal_entities),
        "protected_instruction_terms": protected_terms,
        "allowed_context_entities": allowed_context_entities,
        "allowed_context_facts": [
            f"The scene also contains {entity}."
            for entity in allowed_context_entities
        ],
    }


def task_contracts_equal(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    """Compare runtime-reconstructable episode semantics without IDs or hashes.

    A generated contract may additionally contain Habitat scene metadata used
    to prove context grounding. Runtime workers do not initialize the metadata
    extractor, so compatibility is based on the structural episode fields that
    both sides can reconstruct from the packed dataset.
    """

    runtime_keys = (
        "canonical_instruction",
        "scene_id",
        "evaluation_propositions",
        "evaluation_dependencies",
        "evaluation_constraints",
        "object_states",
        "episode_initial_state",
        "episode_initial_furniture_state",
        "name_to_receptacle",
    )
    expected_runtime = {
        key: _plain_contract_value(expected.get(key)) for key in runtime_keys
    }
    observed_runtime = {
        key: _plain_contract_value(observed.get(key)) for key in runtime_keys
    }
    return expected_runtime == observed_runtime


def token_levenshtein_distance(
    source: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
) -> int:
    """Return Levenshtein edit distance over resilience-text tokens."""

    source_tokens = (
        resilience_text_v2_tokenize(source)
        if isinstance(source, str)
        else tuple(str(token) for token in source)
    )
    target_tokens = (
        resilience_text_v2_tokenize(target)
        if isinstance(target, str)
        else tuple(str(token) for token in target)
    )
    if len(source_tokens) > len(target_tokens):
        source_tokens, target_tokens = target_tokens, source_tokens
    previous = list(range(len(source_tokens) + 1))
    for row_index, target_token in enumerate(target_tokens, start=1):
        current = [row_index]
        for column_index, source_token in enumerate(source_tokens, start=1):
            insertion = current[column_index - 1] + 1
            deletion = previous[column_index] + 1
            substitution = previous[column_index - 1] + (
                source_token != target_token
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return int(previous[-1])


def materialize_policy_instruction(
    canonical_instruction: str,
    rewrite_levels: Sequence[str],
    context_units: Sequence[str],
    level_index: int,
) -> str:
    """Materialize one member of the fixed six-level cumulative curriculum."""

    if level_index not in range(6):
        raise ValueError(f"level_index must be in [0, 5]; got {level_index!r}")
    if len(rewrite_levels) != 2 or len(context_units) != 3:
        raise ValueError("the curriculum requires two rewrites and three contexts")
    if level_index == 0:
        return str(canonical_instruction)
    rewritten = str(rewrite_levels[0] if level_index == 1 else rewrite_levels[1])
    if level_index <= 2:
        return rewritten
    return rewritten + "".join(
        CONTEXT_SEPARATOR + str(context)
        for context in context_units[: level_index - 2]
    )


def compute_unit_budgets(
    canonical_instruction: str,
    rewrite_levels: Sequence[str],
    context_units: Sequence[str],
) -> Tuple[int, ...]:
    """Compute the two edit and three appended-context operation budgets."""

    if len(rewrite_levels) != 2 or len(context_units) != 3:
        raise ValueError("the curriculum requires two rewrites and three contexts")
    return (
        token_levenshtein_distance(canonical_instruction, rewrite_levels[0]),
        token_levenshtein_distance(rewrite_levels[0], rewrite_levels[1]),
        *(
            len(resilience_text_v2_tokenize(CONTEXT_SEPARATOR + context))
            for context in context_units
        ),
    )


def cumulative_text_doses(unit_budgets: Sequence[int]) -> Tuple[float, ...]:
    """Return doses for levels 0..5 from the five operation budgets."""

    if len(unit_budgets) != 5:
        raise ValueError("unit_budgets must contain exactly five values")
    parsed = tuple(int(value) for value in unit_budgets)
    total = sum(parsed)
    if total <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cumulative = 0
    doses: List[float] = [0.0]
    for budget in parsed:
        cumulative += budget
        doses.append(float(cumulative) / float(total))
    return tuple(doses)


@dataclass(frozen=True)
class FrozenPerturbationEntry:
    """One resolved level from a frozen episode/variant stress path."""

    episode_id: str
    family: str
    variant: str
    lambda_value: float
    canonical_instruction: str
    policy_instruction: str
    level_index: int
    complete_unit_count: int
    stress_realized: float
    text_dose: float
    alignment_valid: bool
    reason: str
    unit_budgets: Tuple[int, ...]
    components: Mapping[str, Any]
    bank_path: str
    task_contract: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "family": self.family,
            "variant": self.variant,
            "lambda_value": self.lambda_value,
            "canonical_instruction": self.canonical_instruction,
            "policy_instruction": self.policy_instruction,
            "level_index": self.level_index,
            "complete_unit_count": self.complete_unit_count,
            "stress_realized": self.stress_realized,
            "text_dose": self.text_dose,
            "alignment_valid": self.alignment_valid,
            "reason": self.reason,
            "unit_budgets": list(self.unit_budgets),
            "components": dict(self.components),
            "bank_path": self.bank_path,
        }


class FrozenPerturbationBank:
    """Path-addressed reader for frozen LLM instruction/context sidecars."""

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        entry_paths: Mapping[Tuple[str, str], str],
    ) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self._entry_paths = dict(entry_paths)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "FrozenPerturbationBank":
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FrozenPerturbationBankError(
                "bank_path_missing", f"directory does not exist: {root}"
            )
        manifest_path = root / "manifest.json"
        manifest = _read_json_object(manifest_path, "bank_manifest_invalid")
        if manifest.get("schema_version") != BANK_SCHEMA_VERSION:
            raise FrozenPerturbationBankError(
                "bank_schema_unsupported",
                f"expected {BANK_SCHEMA_VERSION!r}, got "
                f"{manifest.get('schema_version')!r}",
            )
        family = manifest.get("family")
        if family != STRESS_FAMILY:
            raise FrozenPerturbationBankError(
                "bank_family_mismatch",
                f"expected {STRESS_FAMILY!r}, got {family!r}",
            )
        grid = manifest.get("stress_grid")
        if not _is_stress_grid(grid):
            raise FrozenPerturbationBankError(
                "bank_stress_grid_invalid", f"expected {list(STRESS_GRID)!r}"
            )
        validation = manifest.get("validation")
        if not isinstance(validation, Mapping):
            raise FrozenPerturbationBankError(
                "bank_manifest_validation_invalid", "validation must be an object"
            )
        if validation.get("text_tokenizer") != TEXT_TOKENIZER_VERSION:
            raise FrozenPerturbationBankError(
                "bank_manifest_text_tokenizer_mismatch",
                f"expected {TEXT_TOKENIZER_VERSION!r}",
            )
        if validation.get("quality_contract") != QUALITY_CONTRACT_VERSION:
            raise FrozenPerturbationBankError(
                "bank_manifest_quality_contract_mismatch",
                f"expected {QUALITY_CONTRACT_VERSION!r}",
            )
        if validation.get("independent_reviewer") is not True:
            raise FrozenPerturbationBankError(
                "bank_manifest_independent_reviewer_required"
            )
        if validation.get("unit_budget_ratio_enforced") is not False:
            raise FrozenPerturbationBankError(
                "bank_manifest_unit_budget_ratio_semantics_mismatch"
            )
        generator = manifest.get("generator")
        if (
            not isinstance(generator, Mapping)
            or generator.get("context_schedule") != CONTEXT_SCHEDULE_VERSION
            or not str(generator.get("model") or "").strip()
            or not str(generator.get("review_model") or "").strip()
            or str(generator.get("model")).strip().casefold()
            == str(generator.get("review_model")).strip().casefold()
        ):
            raise FrozenPerturbationBankError(
                "bank_manifest_generator_contract_invalid"
            )
        entry_paths = _validate_manifest_entries(manifest)
        return cls(root=root, manifest=manifest, entry_paths=entry_paths)

    def resolve(
        self,
        episode_id: Union[str, int],
        family: str,
        lambda_value: float,
        variant: Union[str, int],
        current_task_contract: Union[Mapping[str, Any], None] = None,
    ) -> FrozenPerturbationEntry:
        """Resolve an episode/variant/level or raise a reason-coded error."""

        episode_key = _safe_path_component(episode_id, "episode_id")
        family_key = _safe_path_component(family, "family")
        if family_key != STRESS_FAMILY:
            raise FrozenPerturbationBankError(
                "bank_family_mismatch",
                f"expected {STRESS_FAMILY!r}, got {family_key!r}",
            )
        variant_key = _normalize_variant(variant)
        level_index = _stress_level_index(lambda_value)
        manifest_relative_path = self._entry_paths.get(
            (episode_key, variant_key)
        )
        if manifest_relative_path is None:
            raise FrozenPerturbationBankError(
                "bank_entry_not_manifested",
                f"episode={episode_key!r}, variant={variant_key!r}",
            )
        variant_path = (
            self.root
            / "episodes"
            / episode_key
            / family_key
            / f"{variant_key}.json"
        )
        expected_relative_path = variant_path.relative_to(self.root).as_posix()
        if manifest_relative_path != expected_relative_path:
            raise FrozenPerturbationBankError(
                "bank_entry_manifest_path_mismatch",
                f"expected {expected_relative_path!r}, got "
                f"{manifest_relative_path!r}",
            )
        if not variant_path.is_file():
            raise FrozenPerturbationBankError(
                "bank_entry_missing", f"sidecar does not exist: {variant_path}"
            )
        payload = _read_json_object(variant_path, "bank_variant_invalid")
        normalized = _validate_variant_payload(
            payload,
            expected_episode_id=episode_key,
            expected_family=family_key,
            expected_variant=variant_key,
        )
        witness = _variant_witness(payload)
        rewrites = [str(value) for value in payload.get("rewrite_levels", [])]
        canonical = str(payload.get("canonical_instruction") or "")
        normalized_rewrites = tuple(
            normalize_rewrite_text(value) for value in rewrites
        )
        edit_signatures = (
            rewrite_edit_signature(canonical, rewrites[0]),
            rewrite_edit_signature(rewrites[0], rewrites[1]),
        )
        for (
            listed_episode,
            listed_variant,
        ), relative_path in self._entry_paths.items():
            if listed_episode != episode_key or listed_variant == variant_key:
                continue
            sibling_path = self.root / Path(relative_path)
            sibling_payload = _read_json_object(
                sibling_path, "bank_variant_invalid"
            )
            _validate_variant_payload(
                sibling_payload,
                expected_episode_id=episode_key,
                expected_family=family_key,
                expected_variant=listed_variant,
            )
            if _variant_witness(sibling_payload) == witness:
                raise FrozenPerturbationBankError(
                    "bank_variant_not_independent",
                    f"{variant_key!r} duplicates {listed_variant!r}",
                )
            sibling_rewrites = [
                str(value) for value in sibling_payload.get("rewrite_levels", [])
            ]
            sibling_canonical = str(
                sibling_payload.get("canonical_instruction") or ""
            )
            if sibling_canonical != canonical:
                raise FrozenPerturbationBankError(
                    "bank_variant_canonical_mismatch",
                    f"{listed_variant!r} uses a different canonical instruction",
                )
            sibling_normalized = tuple(
                normalize_rewrite_text(value) for value in sibling_rewrites
            )
            sibling_signatures = (
                rewrite_edit_signature(sibling_canonical, sibling_rewrites[0]),
                rewrite_edit_signature(sibling_rewrites[0], sibling_rewrites[1]),
            )
            if any(
                current == sibling
                for current, sibling in zip(
                    normalized_rewrites, sibling_normalized
                )
            ):
                raise FrozenPerturbationBankError(
                    "bank_variant_rewrite_not_independent",
                    f"{variant_key!r} reuses a rewrite from {listed_variant!r}",
                )
            if edit_signatures == sibling_signatures:
                raise FrozenPerturbationBankError(
                    "bank_variant_edit_path_not_independent",
                    f"{variant_key!r} duplicates the edit path of "
                    f"{listed_variant!r}",
                )

        budgets = normalized["unit_budgets"]
        doses = normalized["text_doses"]
        policy_instructions = normalized["policy_instructions"]
        validation = normalized["validation"]
        task_contract = normalized["task_contract"]
        tolerance = float(
            validation.get("text_dose_tolerance", DEFAULT_TEXT_DOSE_TOLERANCE)
        )
        ratio_limit = float(
            validation.get(
                "unit_budget_ratio_limit", DEFAULT_UNIT_BUDGET_RATIO_LIMIT
            )
        )
        positive = all(value > 0 for value in budgets)
        budget_ratio = (
            float(max(budgets)) / float(min(budgets)) if positive else math.inf
        )
        dose_errors = tuple(
            abs(doses[index] - STRESS_GRID[index]) for index in range(6)
        )
        approved = validation.get("approved") is True
        semantic_review_valid = _semantic_review_is_approved(
            validation.get("semantic_review")
        )
        runtime_contract_checked = current_task_contract is not None
        # ``resolve`` is also used by offline bank inspectors that do not own a
        # runtime episode.  They may validate the sidecar itself; the injector
        # always supplies the current contract for formal execution.
        runtime_contract_matches = bool(
            not runtime_contract_checked
            or task_contracts_equal(task_contract, current_task_contract or {})
        )
        formal_thresholds_valid = bool(
            tolerance <= DEFAULT_TEXT_DOSE_TOLERANCE + 1.0e-12
            and ratio_limit <= DEFAULT_UNIT_BUDGET_RATIO_LIMIT + 1.0e-12
        )
        alignment_valid = (
            approved
            and semantic_review_valid
            and runtime_contract_matches
            and formal_thresholds_valid
            and positive
            and max(dose_errors) <= min(
                tolerance, DEFAULT_TEXT_DOSE_TOLERANCE
            )
            + 1.0e-12
        )
        if not approved:
            reason = "bank_not_approved"
        elif not semantic_review_valid:
            reason = "bank_semantic_review_rejected"
        elif not runtime_contract_matches:
            reason = "runtime_task_contract_mismatch"
        elif not formal_thresholds_valid:
            reason = "bank_validation_threshold_relaxed"
        elif not positive:
            reason = "bank_non_positive_unit_budget"
        elif max(dose_errors) > tolerance + 1.0e-12:
            reason = "bank_dose_alignment_failed"
        else:
            reason = ""

        components = {
            "schema_version": VARIANT_SCHEMA_VERSION,
            "text_tokenizer": TEXT_TOKENIZER_VERSION,
            "operation_count": level_index,
            "max_operations": 5,
            "unit_budgets": list(budgets),
            "unit_budget_ratio": budget_ratio,
            "unit_budget_ratio_limit": ratio_limit,
            "unit_budget_ratio_warning": bool(
                budget_ratio > ratio_limit + 1.0e-12
            ),
            "unit_budget_ratio_enforced": False,
            "text_dose_error": dose_errors[level_index],
            "max_text_dose_error": max(dose_errors),
            "text_dose_tolerance": tolerance,
            "semantic_review": validation.get("semantic_review", {}),
            "semantic_review_valid": semantic_review_valid,
            "quality_contract": QUALITY_CONTRACT_VERSION,
            "rewrite_quality": normalized["rewrite_quality"],
            "runtime_task_contract_checked": runtime_contract_checked,
            "runtime_task_contract_matches": runtime_contract_matches,
            "formal_thresholds_valid": formal_thresholds_valid,
            "context_levels": list(normalized["context_levels"]),
            "context_rooms": list(normalized["context_rooms"]),
            "generator": payload.get("generator", {}),
            "variant_path": str(variant_path),
        }
        return FrozenPerturbationEntry(
            episode_id=episode_key,
            family=family_key,
            variant=variant_key,
            lambda_value=STRESS_GRID[level_index],
            canonical_instruction=normalized["canonical_instruction"],
            policy_instruction=policy_instructions[level_index],
            level_index=level_index,
            complete_unit_count=level_index,
            stress_realized=float(level_index) / 5.0,
            text_dose=doses[level_index],
            alignment_valid=alignment_valid,
            reason=reason,
            unit_budgets=budgets,
            components=components,
            bank_path=str(self.root),
            task_contract=task_contract,
        )


def build_variant_payload(
    *,
    episode_id: Union[str, int],
    variant: Union[str, int],
    canonical_instruction: str,
    rewrite_levels: Sequence[str],
    context_units: Sequence[str],
    semantic_review: Mapping[str, Any],
    approved: bool,
    generator: Mapping[str, Any],
    task_contract: Mapping[str, Any],
    family: str = STRESS_FAMILY,
    text_dose_tolerance: float = DEFAULT_TEXT_DOSE_TOLERANCE,
    unit_budget_ratio_limit: float = DEFAULT_UNIT_BUDGET_RATIO_LIMIT,
) -> Dict[str, Any]:
    """Build the canonical on-disk representation used by the offline command."""

    episode_key = _safe_path_component(episode_id, "episode_id")
    variant_key = _normalize_variant(variant)
    rewrites = [str(value).strip() for value in rewrite_levels]
    contexts = [str(value).strip() for value in context_units]
    canonical = str(canonical_instruction)
    rewrite_quality = assess_rewrite_quality(canonical, rewrites)
    budgets = compute_unit_budgets(canonical, rewrites, contexts)
    doses = cumulative_text_doses(budgets)
    levels = []
    for index, stress in enumerate(STRESS_GRID):
        levels.append(
            {
                "lambda": stress,
                "level_index": index,
                "complete_unit_count": index,
                "stress_realized": float(index) / 5.0,
                "text_dose": doses[index],
                "policy_instruction": materialize_policy_instruction(
                    canonical, rewrites, contexts, index
                ),
            }
        )
    return {
        "schema_version": VARIANT_SCHEMA_VERSION,
        "family": str(family),
        "episode_id": episode_key,
        "variant": variant_key,
        "canonical_instruction": canonical,
        "task_contract": dict(task_contract),
        "rewrite_levels": rewrites,
        "context_units": contexts,
        "context_separator": CONTEXT_SEPARATOR,
        "unit_budgets": list(budgets),
        "levels": levels,
        "validation": {
            "approved": bool(approved),
            "text_tokenizer": TEXT_TOKENIZER_VERSION,
            "quality_contract": QUALITY_CONTRACT_VERSION,
            "rewrite_quality": rewrite_quality,
            "text_dose_tolerance": float(text_dose_tolerance),
            "unit_budget_ratio_limit": float(unit_budget_ratio_limit),
            "unit_budget_ratio_enforced": False,
            "semantic_review": dict(semantic_review),
        },
        "generator": dict(generator),
    }


def _validate_variant_payload(
    payload: Mapping[str, Any],
    *,
    expected_episode_id: str,
    expected_family: str,
    expected_variant: str,
) -> Dict[str, Any]:
    if payload.get("schema_version") != VARIANT_SCHEMA_VERSION:
        raise FrozenPerturbationBankError(
            "bank_variant_schema_unsupported",
            f"expected {VARIANT_SCHEMA_VERSION!r}, got "
            f"{payload.get('schema_version')!r}",
        )
    expected_values = {
        "episode_id": expected_episode_id,
        "family": expected_family,
        "variant": expected_variant,
    }
    for key, expected in expected_values.items():
        if str(payload.get(key, "")) != expected:
            raise FrozenPerturbationBankError(
                f"bank_variant_{key}_mismatch",
                f"expected {expected!r}, got {payload.get(key)!r}",
            )
    canonical = payload.get("canonical_instruction")
    rewrites = payload.get("rewrite_levels")
    contexts = payload.get("context_units")
    if not isinstance(canonical, str) or not canonical.strip():
        raise FrozenPerturbationBankError(
            "bank_canonical_instruction_invalid", "expected non-empty string"
        )
    if not _is_non_empty_string_list(rewrites, 2):
        raise FrozenPerturbationBankError(
            "bank_rewrite_levels_invalid", "expected exactly two non-empty strings"
        )
    if not _is_non_empty_string_list(contexts, 3):
        raise FrozenPerturbationBankError(
            "bank_context_units_invalid", "expected exactly three non-empty strings"
        )
    if len({str(value).strip() for value in contexts}) != 3:
        raise FrozenPerturbationBankError(
            "bank_context_units_not_unique", "context units must be distinct"
        )
    if payload.get("context_separator") != CONTEXT_SEPARATOR:
        raise FrozenPerturbationBankError(
            "bank_context_separator_mismatch",
            "sidecar was generated with a different serialization separator",
        )
    rewrites_clean = [str(value).strip() for value in rewrites]
    contexts_clean = [str(value).strip() for value in contexts]
    rewrite_quality = assess_rewrite_quality(canonical, rewrites_clean)
    if rewrite_quality.get("valid") is not True:
        raise FrozenPerturbationBankError(
            "bank_rewrite_quality_invalid",
            repr(rewrite_quality.get("errors", [])),
        )
    computed_budgets = compute_unit_budgets(
        canonical, rewrites_clean, contexts_clean
    )
    stored_budgets = payload.get("unit_budgets")
    if (
        not isinstance(stored_budgets, list)
        or len(stored_budgets) != 5
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in stored_budgets
        )
        or tuple(stored_budgets) != computed_budgets
    ):
        raise FrozenPerturbationBankError(
            "bank_unit_budgets_mismatch",
            f"expected computed budgets {list(computed_budgets)!r}",
        )
    doses = cumulative_text_doses(computed_budgets)
    levels = payload.get("levels")
    if not isinstance(levels, list) or len(levels) != 6:
        raise FrozenPerturbationBankError(
            "bank_levels_invalid", "expected exactly six materialized levels"
        )
    policies: List[str] = []
    for index, level in enumerate(levels):
        if not isinstance(level, Mapping):
            raise FrozenPerturbationBankError(
                "bank_level_invalid", f"level {index} is not an object"
            )
        expected_policy = materialize_policy_instruction(
            canonical, rewrites_clean, contexts_clean, index
        )
        scalar_checks = (
            _float_equal(level.get("lambda"), STRESS_GRID[index]),
            level.get("level_index") == index,
            level.get("complete_unit_count") == index,
            _float_equal(level.get("stress_realized"), float(index) / 5.0),
            _float_equal(level.get("text_dose"), doses[index]),
            level.get("policy_instruction") == expected_policy,
        )
        if not all(scalar_checks):
            raise FrozenPerturbationBankError(
                "bank_level_materialization_mismatch",
                f"level {index} does not match the cumulative curriculum",
            )
        policies.append(expected_policy)
    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        raise FrozenPerturbationBankError(
            "bank_validation_invalid", "validation must be an object"
        )
    if not isinstance(validation.get("approved"), bool):
        raise FrozenPerturbationBankError(
            "bank_approval_invalid", "validation.approved must be bool"
        )
    if validation.get("text_tokenizer") != TEXT_TOKENIZER_VERSION:
        raise FrozenPerturbationBankError(
            "bank_text_tokenizer_mismatch",
            f"expected {TEXT_TOKENIZER_VERSION!r}",
        )
    if validation.get("quality_contract") != QUALITY_CONTRACT_VERSION:
        raise FrozenPerturbationBankError(
            "bank_quality_contract_mismatch",
            f"expected {QUALITY_CONTRACT_VERSION!r}",
        )
    stored_rewrite_quality = validation.get("rewrite_quality")
    if (
        not isinstance(stored_rewrite_quality, Mapping)
        or _plain_contract_value(stored_rewrite_quality) != rewrite_quality
    ):
        raise FrozenPerturbationBankError(
            "bank_rewrite_quality_witness_mismatch",
            "stored quality witness does not match deterministic assessment",
        )
    for key in ("text_dose_tolerance", "unit_budget_ratio_limit"):
        value = validation.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise FrozenPerturbationBankError(
                "bank_validation_threshold_invalid", f"invalid {key}"
            )
    if validation.get("unit_budget_ratio_enforced") is not False:
        raise FrozenPerturbationBankError(
            "bank_unit_budget_ratio_semantics_mismatch",
            "v4 records ratio as a diagnostic; it is not a validity gate",
        )
    semantic_review = validation.get("semantic_review")
    if not isinstance(semantic_review, Mapping):
        raise FrozenPerturbationBankError(
            "bank_semantic_review_invalid", "semantic_review must be an object"
        )
    task_contract = payload.get("task_contract")
    required_contract_keys = {
        "canonical_instruction",
        "scene_id",
        "evaluation_propositions",
        "evaluation_dependencies",
        "evaluation_constraints",
        "object_states",
        "episode_initial_state",
        "episode_initial_furniture_state",
        "name_to_receptacle",
        "task_goal_entities",
        "protected_instruction_terms",
        "goal_specification",
        "scene_hierarchy",
        "context_fact_catalog",
        "allowed_context_entities",
        "allowed_context_facts",
        "scene_hierarchy_formal_valid",
    }
    if (
        not isinstance(task_contract, Mapping)
        or not required_contract_keys.issubset(task_contract)
        or task_contract.get("canonical_instruction") != canonical
    ):
        raise FrozenPerturbationBankError(
            "bank_task_contract_invalid",
            "task_contract is incomplete or does not match canonical_instruction",
        )
    if task_contract.get("scene_hierarchy_formal_valid") is not True:
        raise FrozenPerturbationBankError(
            "bank_scene_hierarchy_not_formal",
            str(task_contract.get("scene_hierarchy_missing_reason") or ""),
        )
    hierarchy = task_contract.get("scene_hierarchy")
    hierarchy_root = hierarchy.get("root") if isinstance(hierarchy, Mapping) else None
    if (
        not isinstance(hierarchy, Mapping)
        or hierarchy.get("source") != "partnr_episode_metadata"
        or str(hierarchy.get("scene_id") or "")
        != str(task_contract.get("scene_id") or "")
        or not isinstance(hierarchy_root, Mapping)
        or hierarchy_root.get("level") != "scene"
        or str(hierarchy_root.get("scene_id") or "")
        != str(task_contract.get("scene_id") or "")
    ):
        raise FrozenPerturbationBankError(
            "bank_scene_hierarchy_invalid",
            "expected MetadataExtractor hierarchy for the same scene",
        )
    catalog = task_contract.get("context_fact_catalog")
    if not isinstance(catalog, list) or len(catalog) < 3:
        raise FrozenPerturbationBankError(
            "bank_context_fact_catalog_invalid",
            "expected at least three typed non-goal facts",
        )
    catalog_texts: List[str] = []
    for fact in catalog:
        if (
            not isinstance(fact, Mapping)
            or fact.get("level") not in {"room", "furniture", "object"}
            or not isinstance(fact.get("text"), str)
            or not str(fact.get("text")).strip()
            or not isinstance(fact.get("entities"), list)
        ):
            raise FrozenPerturbationBankError(
                "bank_context_fact_catalog_invalid",
                "malformed typed hierarchy fact",
            )
        catalog_texts.append(str(fact["text"]).strip())
    if len(set(catalog_texts)) != len(catalog_texts):
        raise FrozenPerturbationBankError(
            "bank_context_fact_catalog_invalid", "duplicate fact text"
        )
    allowed_context_facts = {
        str(value) for value in task_contract.get("allowed_context_facts", [])
    }
    if allowed_context_facts != set(catalog_texts):
        raise FrozenPerturbationBankError(
            "bank_context_fact_catalog_mismatch",
            "allowed_context_facts must equal the typed catalog text",
        )
    if any(context not in allowed_context_facts for context in contexts_clean):
        raise FrozenPerturbationBankError(
            "bank_context_not_in_task_contract",
            "every context unit must be a frozen allowed_context_fact",
        )
    catalog_by_text = {
        str(fact["text"]).strip(): fact
        for fact in catalog
        if isinstance(fact, Mapping)
    }
    context_levels = [
        str(catalog_by_text[context].get("level") or "")
        for context in contexts_clean
    ]
    context_rooms = [
        str(catalog_by_text[context].get("room") or "")
        for context in contexts_clean
    ]
    available_levels = {
        str(fact.get("level") or "")
        for fact in catalog
        if isinstance(fact, Mapping)
    }
    if set(PARTNR_HIERARCHY_LEVELS).issubset(available_levels) and tuple(
        context_levels
    ) != PARTNR_HIERARCHY_LEVELS:
        raise FrozenPerturbationBankError(
            "bank_context_hierarchy_order_invalid",
            "available typed facts must be ordered room, furniture, object",
        )
    generator = payload.get("generator")
    if (
        not isinstance(generator, Mapping)
        or generator.get("context_schedule") != CONTEXT_SCHEDULE_VERSION
        or generator.get("quality_contract") != QUALITY_CONTRACT_VERSION
        or not str(generator.get("model") or "").strip()
        or not str(generator.get("review_model") or "").strip()
        or str(generator.get("model")).strip().casefold()
        == str(generator.get("review_model")).strip().casefold()
        or generator.get("context_levels") != context_levels
        or generator.get("context_rooms") != context_rooms
        or generator.get("hierarchy_coverage") != sorted(set(context_levels))
    ):
        raise FrozenPerturbationBankError(
            "bank_context_schedule_witness_invalid",
            "generator hierarchy witness does not match the selected facts",
        )
    return {
        "canonical_instruction": canonical,
        "unit_budgets": computed_budgets,
        "text_doses": doses,
        "policy_instructions": tuple(policies),
        "validation": dict(validation),
        "rewrite_quality": rewrite_quality,
        "task_contract": dict(task_contract),
        "context_levels": tuple(context_levels),
        "context_rooms": tuple(context_rooms),
    }


def _semantic_review_is_approved(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    required_bools = {
        "semantic_equivalent": True,
        "entity_bindings_preserved": True,
        "terminal_semantics_preserved": True,
        "new_task_directive": False,
        "changed_quantity": False,
        "changed_negation": False,
        "changed_temporal_order": False,
        "context_grounded": True,
        "context_task_irrelevant": True,
        "natural_language_valid": True,
        "padding_detected": False,
    }
    if any(
        not isinstance(value.get(key), bool)
        or value.get(key) is not expected
        for key, expected in required_bools.items()
    ):
        return False
    violations = value.get("violations")
    extra_goals = value.get("extra_goals")
    return bool(
        isinstance(violations, list)
        and not violations
        and isinstance(extra_goals, list)
        and not extra_goals
        and isinstance(value.get("rationale"), str)
    )


def _variant_witness(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(
        str(value).strip()
        for value in [
            *(payload.get("rewrite_levels") or []),
            *(payload.get("context_units") or []),
        ]
    )


def _validate_manifest_entries(
    manifest: Mapping[str, Any],
) -> Dict[Tuple[str, str], str]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise FrozenPerturbationBankError(
            "bank_manifest_entries_invalid", "entries must be a list"
        )
    declared_variants = manifest.get("variants")
    if not isinstance(declared_variants, list):
        raise FrozenPerturbationBankError(
            "bank_manifest_variants_invalid", "variants must be a list"
        )
    normalized_variant_list = [
        _normalize_variant(value) for value in declared_variants
    ]
    variants = set(normalized_variant_list)
    if len(variants) != len(normalized_variant_list):
        raise FrozenPerturbationBankError(
            "bank_manifest_variants_invalid", "variants must be unique"
        )
    declared_episode_ids = manifest.get("episode_ids")
    if not isinstance(declared_episode_ids, list):
        raise FrozenPerturbationBankError(
            "bank_manifest_episode_ids_invalid", "episode_ids must be a list"
        )
    episode_ids = [
        _safe_path_component(value, "episode_id")
        for value in declared_episode_ids
    ]
    if len(set(episode_ids)) != len(episode_ids):
        raise FrozenPerturbationBankError(
            "bank_manifest_episode_ids_invalid", "episode_ids must be unique"
        )
    episode_id_set = set(episode_ids)
    indexed: Dict[Tuple[str, str], str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise FrozenPerturbationBankError(
                "bank_manifest_entry_invalid", f"entry {index} is not an object"
            )
        episode_key = _safe_path_component(
            entry.get("episode_id", ""), "episode_id"
        )
        variant_key = _normalize_variant(entry.get("variant", ""))
        if episode_key not in episode_id_set:
            raise FrozenPerturbationBankError(
                "bank_manifest_entry_episode_undeclared", episode_key
            )
        if variant_key not in variants:
            raise FrozenPerturbationBankError(
                "bank_manifest_entry_variant_undeclared", variant_key
            )
        expected = (
            Path("episodes")
            / episode_key
            / STRESS_FAMILY
            / f"{variant_key}.json"
        ).as_posix()
        actual = str(entry.get("path") or "")
        if actual != expected:
            raise FrozenPerturbationBankError(
                "bank_manifest_entry_path_invalid",
                f"entry {index}: expected {expected!r}, got {actual!r}",
            )
        unit_budgets = entry.get("unit_budgets")
        if (
            entry.get("rewrite_quality_valid") is not True
            or not isinstance(unit_budgets, list)
            or len(unit_budgets) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in unit_budgets
            )
        ):
            raise FrozenPerturbationBankError(
                "bank_manifest_entry_quality_invalid",
                f"entry {index} lacks a valid quality/budget witness",
            )
        key = (episode_key, variant_key)
        if key in indexed:
            raise FrozenPerturbationBankError(
                "bank_manifest_entry_duplicate", repr(key)
            )
        indexed[key] = actual
    generated_count = manifest.get("generated_entry_count")
    if (
        isinstance(generated_count, bool)
        or not isinstance(generated_count, int)
        or generated_count != len(indexed)
    ):
        raise FrozenPerturbationBankError(
            "bank_manifest_generated_count_mismatch",
            f"expected {len(indexed)}, got {generated_count!r}",
        )
    failed_count = manifest.get("failed_entry_count")
    if (
        isinstance(failed_count, bool)
        or not isinstance(failed_count, int)
        or failed_count < 0
    ):
        raise FrozenPerturbationBankError(
            "bank_manifest_failed_count_invalid", repr(failed_count)
        )
    expected_count = manifest.get("expected_entry_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != generated_count + failed_count
        or expected_count != len(episode_ids) * len(variants)
    ):
        raise FrozenPerturbationBankError(
            "bank_manifest_expected_count_mismatch",
            "expected both generated+failed and episode_ids*variants to equal "
            f"{expected_count!r}; generated+failed={generated_count + failed_count}, "
            f"episode_ids*variants={len(episode_ids) * len(variants)}, "
        )
    status = manifest.get("status")
    expected_status = "complete" if failed_count == 0 else "partial"
    if status != expected_status:
        raise FrozenPerturbationBankError(
            "bank_manifest_status_mismatch",
            f"expected {expected_status!r}, got {status!r}",
        )
    return indexed


def _read_json_object(path: Path, reason: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenPerturbationBankError(reason, str(exc)) from exc
    if not isinstance(value, Mapping):
        raise FrozenPerturbationBankError(reason, "expected a JSON object")
    return dict(value)


def _safe_path_component(value: Union[str, int], label: str) -> str:
    parsed = str(value).strip()
    if (
        not parsed
        or parsed in {".", ".."}
        or "/" in parsed
        or "\\" in parsed
        or "\x00" in parsed
    ):
        raise FrozenPerturbationBankError(
            f"bank_{label}_invalid", f"unsafe path component {parsed!r}"
        )
    return parsed


def _normalize_variant(value: Union[str, int]) -> str:
    parsed = _safe_path_component(value, "variant")
    return parsed if parsed.startswith("variant_") else f"variant_{parsed}"


def _stress_level_index(value: float) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenPerturbationBankError(
            "bank_lambda_invalid", f"expected numeric lambda, got {value!r}"
        )
    parsed = float(value)
    if not math.isfinite(parsed):
        raise FrozenPerturbationBankError(
            "bank_lambda_invalid", f"lambda is not finite: {value!r}"
        )
    for index, expected in enumerate(STRESS_GRID):
        if math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1.0e-9):
            return index
    raise FrozenPerturbationBankError(
        "bank_lambda_not_in_grid",
        f"{parsed!r} is not one of {list(STRESS_GRID)!r}",
    )


def _is_stress_grid(value: Any) -> bool:
    return isinstance(value, list) and len(value) == len(STRESS_GRID) and all(
        _float_equal(actual, expected)
        for actual, expected in zip(value, STRESS_GRID)
    )


def _is_non_empty_string_list(value: Any, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _float_equal(actual: Any, expected: float) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and math.isfinite(float(actual))
        and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-9)
    )


__all__ = [
    "BANK_SCHEMA_VERSION",
    "VARIANT_SCHEMA_VERSION",
    "STRESS_FAMILY",
    "STRESS_GRID",
    "TEXT_TOKENIZER_VERSION",
    "QUALITY_CONTRACT_VERSION",
    "CONTEXT_SCHEDULE_VERSION",
    "PARTNR_HIERARCHY_LEVELS",
    "CONTEXT_SEPARATOR",
    "DEFAULT_TEXT_DOSE_TOLERANCE",
    "DEFAULT_UNIT_BUDGET_RATIO_LIMIT",
    "FrozenPerturbationBankError",
    "FrozenPerturbationEntry",
    "FrozenPerturbationBank",
    "resilience_text_v2_tokenize",
    "resilience_text_v1_tokenize",
    "normalize_rewrite_text",
    "rewrite_edit_signature",
    "assess_rewrite_quality",
    "token_levenshtein_distance",
    "materialize_policy_instruction",
    "compute_unit_budgets",
    "cumulative_text_doses",
    "build_episode_task_contract",
    "task_contracts_equal",
    "build_variant_payload",
]
