#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_target_graph_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "metrics"
        / "rebound"
        / "target_graph.py"
    )
    module_name = "rebound_target_graph_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


target_graph = _load_target_graph_module()


def _episode(propositions, dependencies=(), constraints=()):
    return SimpleNamespace(
        evaluation_propositions=propositions,
        evaluation_proposition_dependencies=dependencies,
        evaluation_constraints=constraints,
    )


def test_dataset_generation_predicate_schema_is_covered():
    propositions = [
        {
            "function_name": "is_on_top",
            "args": {
                "object_handles": ["cup_0"],
                "receptacle_handles": ["table_0"],
            },
        },
        {
            "function_name": "is_inside",
            "args": {
                "object_handles": ["apple_0"],
                "receptacle_handles": ["drawer_0"],
            },
        },
        {
            "function_name": "is_in_room",
            "args": {"object_handles": ["book_0"], "room_ids": ["bedroom_1"]},
        },
        {"function_name": "is_on_floor", "args": {"object_handles": ["toy_0"]}},
        {
            "function_name": "is_next_to",
            "args": {
                "entity_handles_a": ["cup_0"],
                "entity_handles_b": ["plate_0"],
            },
        },
        {
            "function_name": "is_clustered",
            "args": {
                "*args": [["cup_0"], ["plate_0"], ["fork_0"]],
                "number": [1, 1, 1],
            },
        },
        {"function_name": "is_clean", "args": {"object_handles": ["cup_0"]}},
        {"function_name": "is_dirty", "args": {"object_handles": ["plate_0"]}},
        {"function_name": "is_filled", "args": {"object_handles": ["cup_0"]}},
        {"function_name": "is_empty", "args": {"object_handles": ["bowl_0"]}},
        {"function_name": "is_powered_on", "args": {"object_handles": ["lamp_0"]}},
        {"function_name": "is_powered_off", "args": {"object_handles": ["fan_0"]}},
    ]

    graph = target_graph.build_episode_target_graph(_episode(propositions))

    assert {target.predicate for target in graph.placements} == {
        "is_on_top",
        "is_inside",
        "is_in_room",
        "is_on_floor",
        "is_next_to",
        "is_clustered",
    }
    assert {target.predicate for target in graph.state_flips} == {
        "is_clean",
        "is_dirty",
        "is_filled",
        "is_empty",
        "is_powered_on",
        "is_powered_off",
    }

    floor = next(
        target for target in graph.placements if target.predicate == "is_on_floor"
    )
    assert floor.matches("toy_0", "floor_kitchen_1")
    assert not floor.matches("toy_0", "table_0")

    room = next(
        target for target in graph.placements if target.predicate == "is_in_room"
    )
    next_to = next(
        target for target in graph.placements if target.predicate == "is_next_to"
    )
    clustered = next(
        target for target in graph.placements if target.predicate == "is_clustered"
    )
    assert not room.is_action_matchable
    assert next_to.receptacle_ids == ("plate_0",)
    assert not next_to.is_action_matchable
    assert clustered.receptacle_ids == ("plate_0", "fork_0")
    assert not clustered.is_action_matchable


def test_temporal_constraint_controls_stage_frontier_and_open_targets():
    propositions = [
        {
            "function_name": "is_on_top",
            "args": {
                "object_handles": [f"object_{idx}"],
                "receptacle_handles": [f"table_{idx}"],
            },
        }
        for idx in range(4)
    ]
    constraints = [
        {
            "type": "TemporalConstraint",
            "args": {"dag_edges": [[0, 1], [0, 2], [1, 3], [2, 3]]},
        }
    ]
    graph = target_graph.build_episode_target_graph(
        _episode(propositions, constraints=constraints)
    )

    assert [stage.proposition_indices for stage in graph.stages] == [
        (0,),
        (1, 2),
        (3,),
    ]
    assert [
        target.proposition_index for target in graph.open_placements([-1] * 4)
    ] == [0]
    assert [
        target.proposition_index
        for target in graph.open_placements([0, -1, -1, -1])
    ] == [1, 2]
    assert [
        target.proposition_index
        for target in graph.open_placements([0, 1, 1, -1])
    ] == [3]


def test_sequential_dependencies_are_a_stage_fallback():
    propositions = [
        {
            "function_name": "is_clean",
            "args": {"object_handles": [f"object_{idx}"]},
        }
        for idx in range(3)
    ]
    dependencies = [
        {
            "proposition_indices": [1],
            "depends_on": [0],
            "relation_type": "after_satisfied",
        },
        {
            "proposition_indices": [2],
            "depends_on": [0],
            "relation_type": "while_satisfied",
        },
    ]
    graph = target_graph.build_episode_target_graph(
        _episode(propositions, dependencies=dependencies)
    )

    assert [stage.proposition_indices for stage in graph.stages] == [(0, 2), (1,)]


def test_state_flips_are_limited_to_the_temporal_frontier():
    propositions = [
        {"function_name": "is_clean", "args": {"object_handles": ["cup_0"]}},
        {"function_name": "is_filled", "args": {"object_handles": ["cup_0"]}},
    ]
    constraints = [
        {
            "type": "TemporalConstraint",
            "args": {"dag_edges": [[0, 1]]},
        }
    ]
    graph = target_graph.build_episode_target_graph(
        _episode(propositions, constraints=constraints)
    )

    assert [state.predicate for state in graph.open_state_flips([-1, -1])] == [
        "is_clean"
    ]
    assert [state.predicate for state in graph.open_state_flips([0, -1])] == [
        "is_filled"
    ]


def test_real_planner_place_and_rearrange_actions_are_parsed():
    cases = [
        (("Place", "cup_0,on,table_0,none,none", None), "place"),
        ("Place[cup_0,on,table_0,none,none]", "place"),
        (("Rearrange", "cup_0,on,table_0,next_to,plate_0", None), "rearrange"),
        ({"name": "Place", "args": "cup_0,on,table_0,none,none"}, "place"),
    ]

    for action, expected_name in cases:
        frame = target_graph.parse_action_entry(action)
        assert frame is not None
        assert frame.name == expected_name
        assert frame.primary_arg == "cup_0"
        assert frame.secondary_arg == "table_0"
        assert frame.is_place


def test_runtime_aliases_and_floor_room_pairing_match_planner_actions():
    propositions = [
        {
            "function_name": "is_on_top",
            "args": {
                "object_handles": ["obj.handle.1"],
                "receptacle_handles": ["rec.handle.2"],
            },
        },
        {
            "function_name": "is_on_floor",
            "args": {"object_handles": ["toy.handle.3"]},
        },
        {
            "function_name": "is_in_room",
            "args": {
                "object_handles": ["toy.handle.3"],
                "room_ids": ["region.4"],
            },
        },
        {
            "function_name": "is_clean",
            "args": {"object_handles": ["obj.handle.1"]},
        },
    ]

    graph = target_graph.build_episode_target_graph(
        _episode(propositions),
        entity_aliases={
            "obj.handle.1": "cup_0",
            "rec.handle.2": "table_0",
            "toy.handle.3": "toy_0",
        },
        room_aliases={"region.4": "kitchen_1"},
    )

    assert graph.is_placement_target("cup_0", "table_0")
    assert graph.is_state_flip_target("cup_0")
    floor = next(p for p in graph.placements if p.predicate == "is_on_floor")
    assert floor.matches("toy_0", "floor_kitchen_1")
    assert not floor.matches("toy_0", "floor_bedroom_1")


def test_floor_room_pairing_follows_explicit_dependency_across_stages():
    propositions = [
        {
            "function_name": "is_in_room",
            "args": {"object_handles": ["toy"], "room_ids": ["kitchen_1"]},
        },
        {"function_name": "is_on_floor", "args": {"object_handles": ["toy"]}},
        {
            "function_name": "is_in_room",
            "args": {"object_handles": ["toy"], "room_ids": ["bedroom_1"]},
        },
        {"function_name": "is_on_floor", "args": {"object_handles": ["toy"]}},
    ]
    dependencies = [
        {
            "proposition_indices": [1],
            "depends_on": [0],
            "relation_type": "while_satisfied",
        },
        {
            "proposition_indices": [3],
            "depends_on": [2],
            "relation_type": "while_satisfied",
        },
    ]
    constraints = [
        {
            "type": "TemporalConstraint",
            "args": {"dag_edges": [[0, 2], [1, 3]]},
        }
    ]

    graph = target_graph.build_episode_target_graph(
        _episode(
            propositions,
            dependencies=dependencies,
            constraints=constraints,
        )
    )
    floors = [p for p in graph.placements if p.predicate == "is_on_floor"]

    assert floors[0].matches("toy", "floor_kitchen_1")
    assert not floors[0].matches("toy", "floor_bedroom_1")
    assert floors[1].matches("toy", "floor_bedroom_1")
    assert not floors[1].matches("toy", "floor_kitchen_1")
