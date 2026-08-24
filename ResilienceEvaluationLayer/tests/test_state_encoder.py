from types import SimpleNamespace

import pytest
import torch

from habitat_llm.evaluation.evaluation_runner import EvaluationRunner
from habitat_llm.evaluation.networks.state_encoder import StateEncoder


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def encode(
        self,
        text,
        convert_to_tensor=True,
        device="cpu",
        show_progress_bar=False,
    ):
        return torch.zeros(384, dtype=torch.float32, device=device)


@pytest.fixture
def patched_state_encoder(monkeypatch):
    monkeypatch.setattr(
        "habitat_llm.evaluation.networks.state_encoder.SentenceTransformer",
        DummySentenceTransformer,
    )
    return StateEncoder(device="cpu", debug=True)


def test_state_encoder_includes_extended_task_and_action_context(patched_state_encoder):
    encoder = patched_state_encoder
    world_state = {
        "agent_poses": {
            0: {"position": [1.0, 0.0, 0.0], "yaw": 0.25},
        },
        "object_positions": {
            "mug_0": {"position": [2.0, 0.0, 0.0], "parent": "table_0"},
        },
        "furniture_positions": {
            "table_0": {"position": [4.0, 0.0, 0.0]},
            "dining_room": {"position": [6.0, 0.0, 0.0]},
        },
        "agent_holdings": {
            0: {"held_object": "mug_0"},
        },
        "task_context": {
            "instruction": "Place the mug on the table in the dining room.",
            "objects": ["mug_0"],
            "receptacles": ["table_0"],
            "rooms": ["dining room"],
            "entities": ["mug_0", "lamp_1"],
            "proposition_count": 2,
        },
    }
    info = {
        "task_percent_complete": 0.5,
        "planning_step_count": 3,
        "action_history": [
            SimpleNamespace(action=("Navigate", "table_0"), response="Reached the table")
        ],
        "action_responses": {0: "Reached the table"},
        "auto_eval_proposition_tracker": {
            "proposition_satisfied_at": [0, -1],
        },
    }

    text_desc = encoder._generate_state_description(world_state, info)
    assert "Task rooms: dining room." in text_desc
    assert "Task entities: lamp_1." in text_desc
    assert "Completed propositions: 1/2." in text_desc
    assert "Recent action: Navigate[table_0]." in text_desc
    assert "Recent feedback: Reached the table." in text_desc
    assert "Planning step: 3." in text_desc

    features = encoder._extract_numerical_features(world_state, info)
    feature_map = dict(zip(encoder._last_feature_names, features.tolist()))

    assert feature_map["min_task_obj_dist_norm"] == pytest.approx(0.1)
    assert feature_map["min_agent_target_dist_norm"] == pytest.approx(0.3)
    assert feature_map["task_object_fraction_at_target"] == pytest.approx(1.0)
    assert feature_map["task_target_visible_fraction"] == pytest.approx(1.0)
    assert feature_map["task_entity_count_norm"] == pytest.approx(0.2)
    assert feature_map["task_room_count_norm"] == pytest.approx(0.1)
    assert feature_map["task_proposition_count_norm"] == pytest.approx(0.2)
    assert feature_map["proposition_satisfied_fraction"] == pytest.approx(0.5)
    assert feature_map["action_history_count_norm"] == pytest.approx(0.1)

    state_vec = encoder.forward(world_state, info)
    assert state_vec.shape == (128,)


def test_construct_world_state_dict_handles_scalar_rot_and_spatial_anchors():
    runner = object.__new__(EvaluationRunner)
    runner.agents = {0: object()}
    runner._extract_task_context = lambda: {
        "instruction": "Put the mug on the table.",
        "objects": ["mug_0"],
        "receptacles": ["table_0"],
        "rooms": ["dining room"],
        "entities": ["mug_0"],
        "proposition_count": 1,
    }

    articulated_agent = SimpleNamespace(base_pos=[10.0, 20.0, 30.0], base_rot=1.25)
    sim = SimpleNamespace(
        agents_mgr={
            0: SimpleNamespace(articulated_agent=articulated_agent),
        }
    )

    object_node = SimpleNamespace(
        name="mug_0",
        properties={"translation": [1.0, 2.0, 3.0], "parent_receptacle": "unknown"},
    )
    furniture_node = SimpleNamespace(
        name="table_0",
        properties={"translation": [4.0, 5.0, 6.0]},
    )
    receptacle_node = SimpleNamespace(
        name="table_0_top",
        properties={"translation": [4.0, 5.5, 6.0]},
    )
    room_node = SimpleNamespace(
        name="dining_room",
        properties={"translation": [7.0, 8.0, 9.0]},
    )
    held_object = SimpleNamespace(name="mug_0")
    robot_node = SimpleNamespace(properties={"last_held_object": held_object})
    human_node = SimpleNamespace(properties={"last_held_object": None})

    world_graph = SimpleNamespace(
        get_all_objects=lambda: [object_node],
        get_all_furnitures=lambda: [furniture_node],
        get_all_receptacles=lambda: [receptacle_node],
        get_all_rooms=lambda: [room_node],
        get_spot_robot=lambda: robot_node,
        get_human=lambda: human_node,
        find_furniture_for_object=lambda obj: furniture_node,
        find_receptacle_for_object=lambda obj: receptacle_node,
    )

    runner.env_interface = SimpleNamespace(
        sim=sim,
        world_graph={0: world_graph},
        robot_agent_uid=0,
        human_agent_uid=1,
    )

    world_state = EvaluationRunner._construct_world_state_dict(runner)

    assert world_state["agent_poses"][0]["position"] == [30.0, -10.0, 20.0]
    assert world_state["agent_poses"][0]["yaw"] == pytest.approx(1.25)
    assert world_state["object_positions"]["mug_0"]["parent"] == "table_0"
    assert world_state["furniture_positions"]["table_0"]["position"] == [6.0, -4.0, 5.0]
    assert world_state["furniture_positions"]["table_0_top"]["position"] == [6.0, -4.0, 5.5]
    assert world_state["furniture_positions"]["dining_room"]["position"] == [9.0, -7.0, 8.0]
    assert world_state["agent_holdings"][0]["held_object"] == "mug_0"
