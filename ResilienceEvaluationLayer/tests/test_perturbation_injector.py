import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_perturbation_modules():
    package_root = Path(__file__).resolve().parents[1]
    perturbation_root = package_root / "evaluation" / "perturbation"
    package_paths = {
        "habitat_llm": package_root,
        "habitat_llm.evaluation": package_root / "evaluation",
        "habitat_llm.evaluation.perturbation": perturbation_root,
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    audit = _load_module(
        "habitat_llm.evaluation.perturbation.audit",
        perturbation_root / "audit.py",
    )
    bank = _load_module(
        "habitat_llm.evaluation.perturbation.paraphrase_bank",
        perturbation_root / "paraphrase_bank.py",
    )
    calibrator = _load_module(
        "habitat_llm.evaluation.perturbation.calibrator",
        perturbation_root / "calibrator.py",
    )
    injector = _load_module(
        "habitat_llm.evaluation.perturbation.injector",
        perturbation_root / "injector.py",
    )
    package = sys.modules["habitat_llm.evaluation.perturbation"]
    public_attrs = {
        "PerturbationAudit": audit.PerturbationAudit,
        "ParaphraseBank": bank.ParaphraseBank,
        "CandidateAssessment": calibrator.CandidateAssessment,
        "LLMCandidateCalibrator": calibrator.LLMCandidateCalibrator,
        "PerturbationSpec": injector.PerturbationSpec,
        "PerturbationInjector": injector.PerturbationInjector,
        "derive_run_seed": injector.derive_run_seed,
        "spec_artifact_key": injector.spec_artifact_key,
        "build_perturbation_spec_grid": injector.build_perturbation_spec_grid,
    }
    for name, value in public_attrs.items():
        setattr(package, name, value)
    package.__all__ = list(public_attrs)
    return audit, bank, calibrator, injector


AUDIT, BANK, CALIBRATOR, INJECTOR = _bootstrap_perturbation_modules()
ParaphraseBank = BANK.ParaphraseBank
PerturbationInjector = INJECTOR.PerturbationInjector
PerturbationSpec = INJECTOR.PerturbationSpec
LLMCandidateCalibrator = CALIBRATOR.LLMCandidateCalibrator


@pytest.mark.parametrize("intensity", [-0.01, 1.01, math.nan, math.inf])
def test_spec_rejects_out_of_range_or_non_finite_intensity(intensity):
    with pytest.raises(ValueError):
        PerturbationSpec(kind="surface_rewrite", intensity=intensity)


def test_spec_rejects_non_numeric_intensity_and_unknown_kind():
    with pytest.raises(TypeError):
        PerturbationSpec(kind="surface_rewrite", intensity="0.5")
    with pytest.raises(TypeError):
        PerturbationSpec(kind="surface_rewrite", intensity=True)
    with pytest.raises(ValueError):
        PerturbationSpec(kind="unregistered_stress", intensity=0.5)


@pytest.mark.parametrize("kind", ["surface_rewrite", "filler_injection"])
def test_prompt_zero_intensity_is_exact_noop_with_valid_audit(kind):
    instruction = "Please pick up the cup and then put it on top of the table."
    injector = PerturbationInjector(PerturbationSpec(kind=kind, intensity=0.0))

    rewritten, audit = injector.rewrite_instruction_with_audit(
        instruction,
        {"episode_id": "78"},
    )

    assert rewritten == instruction
    assert audit.requested == 0.0
    assert audit.realized == 0.0
    assert audit.valid
    assert not audit.applied
    assert audit.reason == "zero_intensity"
    assert injector.latest_audit == audit


@pytest.mark.parametrize("kind", ["surface_rewrite", "filler_injection"])
def test_prompt_operation_count_and_realized_dose_are_monotone(kind):
    instruction = "Please pick up the cup and then put it on top of the table."
    observations = []
    for intensity in (0.0, 0.2, 0.5, 0.8, 1.0):
        injector = PerturbationInjector(
            PerturbationSpec(kind=kind, seed=4, intensity=intensity)
        )
        _rewritten, audit = injector.rewrite_instruction_with_audit(
            instruction,
            {"episode_id": "78"},
        )
        observations.append(
            (audit.components.get("operation_count", 0), audit.realized)
        )

    counts = [value[0] for value in observations]
    realized = [value[1] for value in observations]
    assert counts == sorted(counts)
    assert realized == sorted(realized)
    assert counts[0] == 0
    assert realized[0] == 0.0
    assert counts[-1] >= 2
    assert realized[-1] == 1.0


def test_runner_owned_rewrite_can_clear_legacy_planner_hooks():
    planner_a = types.SimpleNamespace(_prompt_perturbation_hook=lambda value: value)
    planner_b = types.SimpleNamespace(_prompt_perturbation_hook=lambda value: value)

    PerturbationInjector.clear_prompt_hooks({0: planner_a, 1: planner_b})

    assert planner_a._prompt_perturbation_hook is None
    assert planner_b._prompt_perturbation_hook is None


def test_frozen_stress_passes_runtime_episode_contract_to_bank():
    captured = {}

    class _RecordingBank:
        def resolve(self, **kwargs):
            captured.update(kwargs)
            raise INJECTOR.FrozenPerturbationBankError(
                "sentinel_bank_failure", "expected test stop"
            )

    instruction = "Put the cup on the table."
    episode = {
        "episode_id": "78",
        "scene_id": "fixture_scene",
        "instruction": instruction,
        "evaluation_propositions": [
            {
                "function_name": "is_on_top",
                "args": {
                    "object_handles": ["cup.handle"],
                    "receptacle_handles": ["table.handle"],
                    "number": 1,
                },
            }
        ],
        "evaluation_proposition_dependencies": [],
        "evaluation_constraints": [],
    }
    injector = PerturbationInjector(
        PerturbationSpec(
            kind="llm_instruction_context_stress",
            seed=0,
            intensity=0.2,
            extras={"bank_path": "missing", "variant": "variant_0"},
        )
    )
    injector._frozen_bank_error = None
    injector._frozen_bank = _RecordingBank()

    rewritten, audit = injector.rewrite_instruction_with_audit(
        instruction,
        episode,
    )

    assert rewritten == instruction
    assert audit.reason == "sentinel_bank_failure"
    assert captured["episode_id"] == "78"
    assert captured["current_task_contract"]["canonical_instruction"] == instruction


class _WorldGraph:
    def __init__(self):
        self.counterfactual_distractors = []

    def get_world_descr(self):
        suffix = " ".join(self.counterfactual_distractors)
        return f"Furniture: table Objects: cup {suffix}".strip()


def test_irrelevant_world_change_reports_realized_and_visible_components():
    env = types.SimpleNamespace(world_graph={0: _WorldGraph(), 1: _WorldGraph()})
    injector = PerturbationInjector(
        PerturbationSpec(
            kind="irrelevant_perturbation",
            seed=3,
            intensity=0.4,
        ),
        irrelevant_object_count=5,
    )

    audit = injector.apply_world_perturbation_with_audit(
        env,
        {"episode_id": "78"},
    )

    assert audit.valid and audit.applied
    assert audit.realized == pytest.approx(0.4)
    assert audit.components["applied_graph_count"] == 2
    for row in audit.components["per_agent"]:
        assert row["components"]["added_count"] == 2
        assert row["components"]["planner_visible_added_token_count"] == 2
        assert row["components"]["planner_visible_token_ratio"] > 0.0


def test_begin_rollout_reapplies_same_cell_to_rebuilt_world_graph():
    first_graph = _WorldGraph()
    env = types.SimpleNamespace(world_graph={0: first_graph})
    injector = PerturbationInjector(
        PerturbationSpec(
            kind="irrelevant_perturbation",
            seed=3,
            intensity=0.4,
        ),
        irrelevant_object_count=5,
    )

    first = injector.apply_world_perturbation_with_audit(
        env,
        {"episode_id": "78"},
    )
    rebuilt_graph = _WorldGraph()
    env.world_graph = {0: rebuilt_graph}
    injector.begin_rollout()
    second = injector.apply_world_perturbation_with_audit(
        env,
        {"episode_id": "78"},
    )

    assert first.valid and second.valid
    assert second is not first
    assert first_graph.counterfactual_distractors
    assert rebuilt_graph.counterfactual_distractors


def test_irrelevant_world_change_without_planner_visible_measurement_fails_closed():
    env = types.SimpleNamespace(
        world_graph={0: types.SimpleNamespace(counterfactual_distractors=[])}
    )
    injector = PerturbationInjector(
        PerturbationSpec(kind="irrelevant_perturbation", intensity=0.4),
        irrelevant_object_count=5,
    )

    audit = injector.apply_world_perturbation_with_audit(
        env,
        {"episode_id": "78"},
    )

    assert not audit.valid
    assert not audit.applied
    assert audit.realized == 0.0
    assert "planner_visible_measurement_unavailable" in audit.reason


def test_world_adapter_without_observed_change_fails_closed():
    env = types.SimpleNamespace(world_graph={0: types.SimpleNamespace()})
    injector = PerturbationInjector(
        PerturbationSpec(kind="room_shift", intensity=1.0)
    )

    audit = injector.apply_world_perturbation_with_audit(env, {"episode_id": "78"})

    assert not audit.valid
    assert not audit.applied
    assert audit.realized == 0.0
    assert "unsupported_room_shift_adapter" in audit.reason


def _object_state_episode(
    predicate="is_clean",
    handle="cup.handle",
):
    return {
        "episode_id": "78",
        "evaluation_propositions": [
            {
                "function_name": predicate,
                "args": {"object_handles": [handle]},
            }
        ],
    }


class _StateObject:
    def __init__(
        self,
        name="cup_0",
        sim_handle="cup.handle",
        states=None,
    ):
        self.name = name
        self.sim_handle = sim_handle
        self.properties = {"states": dict(states or {"is_clean": False})}

    def set_state(self, update):
        self.properties["states"].update(update)


class _StateWorldGraph:
    def __init__(self):
        self.obj = _StateObject()

    def get_all_objects(self):
        return [self.obj]


class _NoOpStateObject(_StateObject):
    def set_state(self, update):
        del update


class _NoOpStateWorldGraph(_StateWorldGraph):
    def __init__(self):
        self.obj = _NoOpStateObject()


class _MixedStateWorldGraph:
    def __init__(self):
        self.unrelated = _StateObject(
            name="a_unrelated",
            sim_handle="other.handle",
        )
        self.writable = _StateObject()

    def get_all_objects(self):
        return [self.unrelated, self.writable]


def test_object_state_adapter_verifies_before_after_change():
    world_graph = _StateWorldGraph()
    env = types.SimpleNamespace(world_graph={0: world_graph})
    injector = PerturbationInjector(
        PerturbationSpec(kind="object_state_toggle", intensity=0.2)
    )

    audit = injector.apply_world_perturbation_with_audit(
        env,
        _object_state_episode(),
    )

    assert audit.valid and audit.applied
    assert audit.realized == 1.0
    assert world_graph.obj.properties["states"]["is_clean"] is True


def test_object_state_adapter_mutates_episode_target_not_first_state_object():
    world_graph = _MixedStateWorldGraph()
    env = types.SimpleNamespace(world_graph={0: world_graph})
    injector = PerturbationInjector(
        PerturbationSpec(kind="object_state_toggle", intensity=1.0)
    )

    audit = injector.apply_world_perturbation_with_audit(
        env,
        _object_state_episode(),
    )

    assert audit.valid
    assert world_graph.unrelated.properties["states"]["is_clean"] is False
    assert world_graph.writable.properties["states"]["is_clean"] is True


def test_multi_graph_partial_failure_rolls_back_and_is_idempotent():
    good_graph = _StateWorldGraph()
    bad_graph = _NoOpStateWorldGraph()
    env = types.SimpleNamespace(world_graph={0: good_graph, 1: bad_graph})
    injector = PerturbationInjector(
        PerturbationSpec(kind="object_state_toggle", intensity=1.0)
    )

    first = injector.apply_world_perturbation_with_audit(
        env,
        _object_state_episode(),
    )
    second = injector.apply_world_perturbation_with_audit(
        env,
        _object_state_episode(),
    )

    assert not first.valid
    assert not first.applied
    assert first.realized == 0.0
    assert good_graph.obj.properties["states"]["is_clean"] is False
    assert second is first
    assert good_graph.obj.properties["states"]["is_clean"] is False
    good_result = first.components["per_agent"][0]
    assert good_result["components"]["rollback_succeeded"] is True


def test_object_state_supports_furniture_and_shared_agent_graphs():
    furniture = _StateObject(
        name="washer_0",
        sim_handle="washer.handle",
        states={"is_powered_on": False},
    )

    class _FurnitureGraph:
        def get_all_objects(self):
            return []

        def get_all_furnitures(self):
            return [furniture]

    # Distinct WorldGraph wrappers may still share the same Entity node after
    # graph merge; the target must be toggled exactly once in that case too.
    env = types.SimpleNamespace(
        world_graph={0: _FurnitureGraph(), 1: _FurnitureGraph()}
    )
    injector = PerturbationInjector(
        PerturbationSpec(kind="object_state_toggle", intensity=1.0)
    )

    audit = injector.apply_world_perturbation_with_audit(
        env,
        _object_state_episode("is_powered_on", "washer.handle"),
    )

    assert audit.valid and audit.applied
    assert furniture.properties["states"]["is_powered_on"] is True
    assert audit.components["graph_count"] == 1
    assert audit.components["agent_view_count"] == 2


def test_llm_calibrator_is_offline_by_default():
    calibrator = LLMCandidateCalibrator()

    assert calibrator.generate_candidates(
        "Move cup to table.",
        kind="surface_rewrite",
        requested=0.5,
    ) == []
    assessment = calibrator.assess_candidate(
        "Move cup to table.",
        "Relocate cup to table.",
        kind="surface_rewrite",
        requested=0.5,
        required_terms=("cup", "table"),
    )
    assert assessment.audit.valid
    assert not assessment.llm_checked


def test_llm_calibrator_uses_strict_json_and_never_sets_realized_from_llm():
    calls = []

    def completion(messages):
        calls.append(messages)
        if "Generate" in messages[0]["content"]:
            return '{"candidates":["Relocate cup to table."]}'
        return {
            "semantic_equivalent": True,
            "violations": [],
            "rationale": "entities and relation are preserved",
        }

    calibrator = LLMCandidateCalibrator(completion, model_identity="judge-v1")
    candidates = calibrator.generate_candidates(
        "Move cup to table.",
        kind="surface_rewrite",
        requested=0.5,
        count=1,
    )
    assessment = calibrator.assess_candidate(
        "Move cup to table.",
        candidates[0],
        kind="surface_rewrite",
        requested=0.5,
        required_terms=("cup", "table"),
    )

    expected = AUDIT.text_change_components(
        "Move cup to table.", candidates[0]
    )["normalized_edit_distance"]
    assert len(calls) == 2
    assert assessment.audit.valid
    assert assessment.audit.realized == pytest.approx(expected)
    assert assessment.audit.components["llm_model_identity"] == "judge-v1"


def test_llm_semantic_rejection_cannot_be_overridden_by_local_score():
    def completion(_messages):
        return {
            "semantic_equivalent": False,
            "violations": ["destination changed"],
            "rationale": "not equivalent",
        }

    assessment = LLMCandidateCalibrator(completion).assess_candidate(
        "Move cup to table.",
        "Move cup to shelf.",
        kind="surface_rewrite",
        requested=0.5,
        required_terms=("cup",),
    )

    assert not assessment.audit.valid
    assert assessment.audit.reason == "llm_semantic_rejection"
    assert assessment.audit.realized > 0.0


def test_llm_calibrator_rejects_non_strict_json():
    def completion(_messages):
        return {"candidates": ["candidate"], "score": 0.9}

    with pytest.raises(ValueError, match="exactly"):
        LLMCandidateCalibrator(completion).generate_candidates(
            "Move cup.",
            kind="surface_rewrite",
            requested=0.5,
        )


def test_paired_run_seed_and_spec_artifact_identity_helpers():
    assert INJECTOR.derive_run_seed(123, 4) == INJECTOR.derive_run_seed(123, 4)
    assert INJECTOR.derive_run_seed(123, 4) != INJECTOR.derive_run_seed(123, 5)

    first = PerturbationSpec(kind="surface_rewrite", seed=4, intensity=0.101)
    second = PerturbationSpec(kind="surface_rewrite", seed=4, intensity=0.104)
    candidate_a = PerturbationSpec(
        kind="surface_rewrite",
        seed=4,
        intensity=0.5,
        extras={"candidate_id": "a"},
    )
    candidate_b = PerturbationSpec(
        kind="surface_rewrite",
        seed=4,
        intensity=0.5,
        extras={"candidate_id": "b"},
    )
    assert INJECTOR.spec_artifact_key(first) != INJECTOR.spec_artifact_key(second)
    assert INJECTOR.spec_artifact_key(candidate_a) != INJECTOR.spec_artifact_key(
        candidate_b
    )


def test_spec_grid_has_one_zero_dose_clean_reference_per_seed():
    specs = INJECTOR.build_perturbation_spec_grid(
        {
            "perturbations": ["clean", "baseline", "surface_rewrite"],
            "seeds": [3, 3, 4],
            "intensities": [0.2, 0.8],
        }
    )

    clean = [spec for spec in specs if spec.is_clean]
    stressed = [spec for spec in specs if not spec.is_clean]
    assert [(spec.seed, spec.intensity) for spec in clean] == [(3, 0.0), (4, 0.0)]
    assert {(spec.seed, spec.intensity) for spec in stressed} == {
        (3, 0.2),
        (3, 0.8),
        (4, 0.2),
        (4, 0.8),
    }
    assert [
        (spec.seed, spec.kind, spec.intensity) for spec in specs
    ] == [
        (3, "clean", 0.0),
        (3, "surface_rewrite", 0.2),
        (3, "surface_rewrite", 0.8),
        (4, "clean", 0.0),
        (4, "surface_rewrite", 0.2),
        (4, "surface_rewrite", 0.8),
    ]
