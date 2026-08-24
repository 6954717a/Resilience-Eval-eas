import json

import pytest

pytest.importorskip(
    "dataset_generation",
    reason="requires the optional PARTNR dataset-generation checkout",
)

from dataset_generation.benchmark_generation.generate_perturbation_bank import (
    GENERATION_SCHEMA_KEYS,
    REVIEW_SCHEMA_KEYS,
    STRESS_FAMILY,
    _BANK_MODULE,
    _compose_semantic_review,
    _dose_safe_rewrite_interval,
    _generate_variant,
    _local_assessment,
    _nearest_feasible_rewrite_budgets,
    _select_balanced_context_schedule,
    build_episode_task_contract,
)
from dataset_generation.benchmark_generation.partnr_episode_semantics import (
    enrich_partnr_task_contract,
)
from dataset_generation.benchmark_generation.stress_prompt_renderer import (
    generation_system_prompt,
    render_generation_user_prompt,
    render_review_user_prompt,
    review_system_prompt,
)


CANONICAL = (
    "Put the toy food and the toy fire truck on the other chest of drawers "
    "in the bedroom."
)
FOOD_HANDLE = "FOOD_ASSET_:0000"
TRUCK_HANDLE = "TRUCK_ASSET_:0000"
TARGET_0 = "TARGET_ASSET_:0000"
TARGET_1 = "TARGET_ASSET_:0001"


def _contract():
    episode = {
        "episode_id": "78",
        "scene_id": "fixture_scene",
        "instruction": CANONICAL,
        "evaluation_propositions": [
            {
                "function_name": "is_on_top",
                "args": {
                    "object_handles": [FOOD_HANDLE],
                    "receptacle_handles": [TARGET_0, TARGET_1],
                    "number": 1,
                    "is_same_receptacle": False,
                },
            },
            {
                "function_name": "is_on_top",
                "args": {
                    "object_handles": [TRUCK_HANDLE],
                    "receptacle_handles": [TARGET_0, TARGET_1],
                    "number": 1,
                    "is_same_receptacle": False,
                },
            },
        ],
        "evaluation_proposition_dependencies": [],
        "evaluation_constraints": [
            {
                "type": "TerminalSatisfactionConstraint",
                "args": {"proposition_indices": [0, 1]},
            }
        ],
    }
    metadata = {
        "instruction": CANONICAL,
        "rooms": ["bedroom_0", "dining_room_0"],
        "room_to_id": {
            "bedroom_0": "bedroom",
            "dining_room_0": "dining room",
        },
        "objects": ["toy_food_0", "toy_fire_truck_0"],
        "object_to_handle": {
            "toy_food_0": FOOD_HANDLE,
            "toy_fire_truck_0": TRUCK_HANDLE,
        },
        "object_to_recep": {
            "toy_food_0": "chest_of_drawers_2",
            "toy_fire_truck_0": "chest_of_drawers_2",
        },
        "object_to_room": {
            "toy_food_0": "bedroom_0",
            "toy_fire_truck_0": "bedroom_0",
        },
        "recep_to_room": {
            "chest_of_drawers_0": "bedroom_0",
            "chest_of_drawers_1": "bedroom_0",
            "chest_of_drawers_2": "bedroom_0",
            "chair_0": "dining_room_0",
            "table_3": "dining_room_0",
        },
        "recep_to_handle": {
            "chest_of_drawers_0": TARGET_0,
            "chest_of_drawers_1": TARGET_1,
            "chest_of_drawers_2": "SOURCE_ASSET_:0000",
            "chair_0": "CHAIR_ASSET_:0000",
            "table_3": "TABLE_ASSET_:0000",
        },
    }
    return enrich_partnr_task_contract(
        build_episode_task_contract(episode),
        episode,
        metadata,
        metadata_source="fixture",
    )


def test_episode78_surface_contract_uses_typed_context_names():
    contract = _contract()

    assert contract["protected_instruction_terms"] == [
        "other chest of drawers",
        "toy fire truck",
        "toy food",
        "bedroom",
    ]
    facts = contract["allowed_context_facts"]
    assert 'Scene fact: the house contains the room "dining_room_0".' in facts
    assert 'Scene fact: furniture "chair_0" is in room "dining_room_0".' in facts
    assert all("bedroom_0" not in value for value in facts)

    summary = contract["surface_goal_summary"]
    assert summary["goal_facts"][0]["surface_entities"]["furniture"] == [
        "chest of drawers"
    ]
    assert "different receptacles are not required" in summary["goal_facts"][0][
        "control_semantics"
    ]["is_same_receptacle"]


def test_generation_prompt_is_sectioned_and_does_not_expose_handles():
    contract = _contract()
    schedule = _select_balanced_context_schedule(
        contract,
        variant="variant_0",
        text_dose_tolerance=0.075,
        unit_budget_ratio_limit=1.5,
    )
    prompt = render_generation_user_prompt(
        canonical_instruction=CANONICAL,
        canonical_entity_phrases=contract["canonical_entity_phrases"],
        surface_goal_summary=contract["surface_goal_summary"],
        task_surface_bindings=contract["task_surface_bindings"],
        rewrite_budget_target=schedule["rewrite_unit_budget_target"],
        rewrite_budget_interval=schedule["rewrite_budget_interval"],
        rewrite_token_count_interval=[14, 27],
        previous_variant_rewrites=[],
        feedback={},
    )

    assert GENERATION_SCHEMA_KEYS == {"rewrite_level_1", "rewrite_level_2"}
    assert "[CANONICAL_TASK]" in prompt
    assert "[GOAL_LEDGER]" in prompt
    assert "\n\n" not in prompt
    assert FOOD_HANDLE not in prompt
    assert TARGET_0 not in prompt
    assert "fixed_context_units" not in prompt
    assert "internal_candidate_counts" not in prompt
    assert "context_fact_catalog" not in prompt
    assert '"family":"rearrangement"' in prompt
    goal_line = next(
        line for line in prompt.splitlines() if line.startswith("[GOAL_LEDGER] ")
    )
    goal_ledger = json.loads(goal_line.split(" ", 1)[1])
    assert goal_ledger["propositions"][0]["predicate"] == "is_on_top"
    assert len(prompt) < 12000


def test_gpt56_prompts_have_no_blank_or_wrapped_paragraph_lines():
    contract = _contract()
    generation = generation_system_prompt("resilience_text_v2")
    review = review_system_prompt()
    review_user = render_review_user_prompt(
        canonical_instruction=CANONICAL,
        canonical_entity_phrases=contract["canonical_entity_phrases"],
        surface_goal_summary=contract["surface_goal_summary"],
        rewrite_levels=[
            "In the bedroom, move the toy food and the toy fire truck onto "
            "the other chest of drawers.",
            "The toy food and the toy fire truck belong on the other chest "
            "of drawers in the bedroom.",
        ],
    )

    for prompt in (generation, review, review_user):
        assert "\n\n" not in prompt
        assert all(line and line == line.strip() for line in prompt.splitlines())
    assert "[PARTNR_PREDICATES]" in generation
    assert "DifferentArgConstraint" in generation
    assert "[PROCEDURE]" in generation
    assert "[APPROVAL_POLARITY]" in review
    assert "rewrite_level_1:" in review


def test_generation_prompt_selects_one_relevant_benchmark_few_shot():
    def rendered(summary):
        return render_generation_user_prompt(
            canonical_instruction="Complete the task.",
            canonical_entity_phrases=[],
            surface_goal_summary=summary,
            task_surface_bindings=[],
            rewrite_budget_target=4,
            rewrite_budget_interval=[2, 6],
            rewrite_token_count_interval=[3, 10],
            previous_variant_rewrites=[],
            feedback={},
        )

    temporal = rendered(
        {
            "goal_facts": [],
            "dependencies": [{"relation_type": "after_satisfied"}],
            "constraints": [],
        }
    )
    tied = rendered(
        {
            "goal_facts": [],
            "dependencies": [],
            "constraints": [{"type": "DifferentArgConstraint", "args": {}}],
        }
    )
    state = rendered(
        {
            "goal_facts": [{"index": 0, "predicate": "is_clean"}],
            "dependencies": [],
            "constraints": [],
        }
    )

    assert '"family":"temporal"' in temporal
    assert '"family":"argument_tie"' in tied
    assert '"family":"object_state"' in state
    assert all(value.count("[RELEVANT_FEW_SHOT]") == 1 for value in (temporal, tied, state))


def test_episode78_natural_rewrites_pass_deterministic_contract():
    contract = _contract()
    schedule = _select_balanced_context_schedule(
        contract,
        variant="variant_0",
        text_dose_tolerance=0.075,
        unit_budget_ratio_limit=1.5,
    )
    candidate = {
        "rewrite_level_1": (
            "In the bedroom, move the toy food and the toy fire truck onto "
            "the other chest of drawers."
        ),
        "rewrite_level_2": (
            "The toy food and the toy fire truck belong on the other chest "
            "of drawers in the bedroom."
        ),
        "context_units": schedule["context_units"],
    }
    result = _local_assessment(
        contract,
        candidate,
        text_dose_tolerance=0.075,
        unit_budget_ratio_limit=1.5,
        expected_context_units=schedule["context_units"],
    )

    assert result["valid"], result["errors"]
    assert result["unit_budgets"] == [8, 9, 10, 10, 10]


def test_joint_dose_guidance_excludes_individually_in_range_bad_pair():
    safe_interval = _dose_safe_rewrite_interval(
        [11, 11, 11],
        target_budget=11,
        text_dose_tolerance=0.075,
    )
    recommended = _nearest_feasible_rewrite_budgets(
        [16, 15],
        [11, 11, 11],
        target_budget=11,
        text_dose_tolerance=0.075,
    )

    assert safe_interval == [8, 14]
    assert recommended == [14, 15]


def test_llm_review_cannot_overturn_deterministic_context_proof():
    review = {
        "semantic_equivalent": True,
        "entity_bindings_preserved": True,
        "terminal_semantics_preserved": True,
        "new_task_directive": False,
        "changed_quantity": False,
        "changed_negation": False,
        "changed_temporal_order": False,
        "natural_language_valid": True,
        "padding_detected": False,
        "extra_goals": [],
        "violations": [],
        "rationale": "Both rewrites preserve the canonical task.",
    }

    composed = _compose_semantic_review(review)

    assert composed["context_grounded"] is True
    assert composed["context_task_irrelevant"] is True
    assert composed["review_scope"] == "rewrite_semantics_only"


def test_episode78_fake_clients_materialize_a_valid_v6_prompt_variant():
    class FakeClient:
        model = "fake-generation"

        def complete(self, *, expected_keys, **_kwargs):
            if expected_keys == GENERATION_SCHEMA_KEYS:
                return {
                    "rewrite_level_1": (
                        "In the bedroom, move the toy food and the toy fire "
                        "truck onto the other chest of drawers."
                    ),
                    "rewrite_level_2": (
                        "The toy food and the toy fire truck belong on the "
                        "other chest of drawers in the bedroom."
                    ),
                }
            assert expected_keys == REVIEW_SCHEMA_KEYS
            return {
                "semantic_equivalent": True,
                "entity_bindings_preserved": True,
                "terminal_semantics_preserved": True,
                "new_task_directive": False,
                "changed_quantity": False,
                "changed_negation": False,
                "changed_temporal_order": False,
                "natural_language_valid": True,
                "padding_detected": False,
                "extra_goals": [],
                "violations": [],
                "rationale": "The rewrites preserve the task.",
            }

    class FakeReviewer(FakeClient):
        model = "fake-reviewer"

    payload, failure = _generate_variant(
        episode_id="78",
        variant="variant_0",
        contract=_contract(),
        generation_client=FakeClient(),
        review_client=FakeReviewer(),
        max_attempts=1,
        text_dose_tolerance=0.075,
        unit_budget_ratio_limit=1.5,
        generation_temperature=0.0,
        forbidden_witnesses=set(),
        previous_variant_rewrites=[],
    )

    assert payload is not None, failure
    assert payload["generator"]["prompt_version"].endswith("_v6")
    assert payload["validation"]["semantic_review"]["context_grounded"] is True
    _BANK_MODULE._validate_variant_payload(
        payload,
        expected_episode_id="78",
        expected_family=STRESS_FAMILY,
        expected_variant="variant_0",
    )
