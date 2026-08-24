import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest


def _load_proposition_outcome_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "proposition_outcome.py"
    )
    module_name = "proposition_outcome_test_target"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_OUTCOME = _load_proposition_outcome_module()
materialize_proposition_outcome = _OUTCOME.materialize_proposition_outcome
proposition_outcome_from_info = _OUTCOME.proposition_outcome_from_info


def _load_instruct_utils(monkeypatch):
    world_graph_module = types.ModuleType(
        "habitat_llm.world_model.world_graph"
    )
    world_graph_module.WorldGraph = type("WorldGraph", (), {})
    monkeypatch.setitem(
        sys.modules,
        "habitat_llm.world_model.world_graph",
        world_graph_module,
    )
    module_path = (
        Path(__file__).resolve().parents[1]
        / "llm"
        / "instruct"
        / "utils.py"
    )
    module_name = "habitat_llm.llm.instruct.utils_parser_test_target"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_proposition_outcome_materializes_tracker_fraction():
    info = {
        "auto_eval_proposition_tracker": {
            "proposition_satisfied_at": [0, -1, None, 4],
        }
    }

    outcome = proposition_outcome_from_info(info)
    assert outcome.satisfied_count == 2
    assert outcome.total == 4
    assert outcome.fraction == pytest.approx(0.5)

    materialize_proposition_outcome(info)
    assert info["proposition_satisfied_count"] == 2
    assert info["proposition_total"] == 4
    assert info["proposition_satisfied_fraction"] == pytest.approx(0.5)


def test_missing_tracker_does_not_fabricate_zero_outcome():
    outcome = proposition_outcome_from_info(
        {"task_percent_complete": 0.75},
        {"proposition_count": 5},
    )
    assert outcome.satisfied_count == 0
    assert outcome.total == 5
    assert outcome.fraction is None
    assert outcome.valid is False


def test_trimmed_tracker_preserves_only_canonical_materialized_scalar():
    info = {
        "auto_eval_proposition_tracker": {
            "proposition_satisfied_at": [0, -1],
        }
    }
    materialize_proposition_outcome(info)
    del info["auto_eval_proposition_tracker"]

    preserved = materialize_proposition_outcome(info)
    assert preserved.fraction == pytest.approx(0.5)
    assert info["proposition_satisfied_fraction"] == pytest.approx(0.5)
    assert info["proposition_satisfied_fraction_source"] == (
        "auto_eval_proposition_tracker"
    )

    untrusted = {"proposition_satisfied_fraction": 0.75}
    missing = materialize_proposition_outcome(untrusted)
    assert missing.fraction is None
    assert untrusted["proposition_satisfied_fraction"] is None
    assert untrusted["proposition_satisfied_fraction_source"].startswith(
        "missing:"
    )


def test_action_parser_rejects_non_numeric_agent_directive(monkeypatch):
    utils = _load_instruct_utils(monkeypatch)
    agents = [SimpleNamespace(uid=0), SimpleNamespace(uid=1)]

    parsed = utils.actions_parser(
        agents,
        "Agent_Action: Navigate[chair_0]",
    )

    assert set(parsed) == {0, 1}
    for payload in parsed.values():
        assert payload[:2] == (None, None)
        assert "Agent_<numeric_id>_Action" in payload[2]


def test_action_parser_accepts_only_strict_numeric_directive(monkeypatch):
    utils = _load_instruct_utils(monkeypatch)
    agents = [SimpleNamespace(uid=0)]

    assert utils.actions_parser(
        agents,
        "Agent_0_Action: Wait[]",
    ) == {0: ("Wait", None, None)}
    malformed = utils.actions_parser(
        agents,
        "Agent_0_Action : Wait[]",
    )
    assert malformed[0][0] is None
    assert malformed[0][2].startswith("Invalid Action directive")


@pytest.mark.parametrize(
    "response",
    [
        "Agent_0_Action: Wait[]\nAgent_Action: Navigate[chair_0]",
        "Agent_Action: Navigate[chair_0]\nAgent_0_Action: Wait[]",
    ],
)
def test_action_parser_does_not_swallow_malformed_mixed_directive(
    monkeypatch,
    response,
):
    utils = _load_instruct_utils(monkeypatch)
    agents = [SimpleNamespace(uid=0)]

    parsed = utils.actions_parser(agents, response)

    assert parsed[0][:2] == (None, None)
    assert "Agent_<numeric_id>_Action" in parsed[0][2]


def test_action_parser_preserves_restricted_action_error(monkeypatch):
    utils = _load_instruct_utils(monkeypatch)
    agents = [SimpleNamespace(uid=0), SimpleNamespace(uid=1)]

    parsed = utils.actions_parser(
        agents,
        "Agent_0_Action: Fill[cup_0]",
    )

    assert parsed[0][:2] == (None, None)
    assert "cannot fill" in parsed[0][2]
