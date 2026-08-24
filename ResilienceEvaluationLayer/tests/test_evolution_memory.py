#!/usr/bin/env python3

from pathlib import Path

from habitat_llm.planner.evolve.context_manager import EvolutionContextManager
from habitat_llm.planner.evolve.types import EvolutionEpisodeRecord
from habitat_llm.world_model import (
    Object,
    Receptacle,
    Room,
    SpotRobot,
    WorldGraph,
    build_task_signature,
    build_world_state_from_graphs,
    export_scene_graph_snapshot,
)


def _build_test_world_graph() -> WorldGraph:
    graph = WorldGraph()
    room = Room("kitchen", {"type": "room", "translation": [0.0, 0.0, 0.0]})
    table = Receptacle("table_surface", {"type": "receptacle", "translation": [1.0, 0.0, 1.0]})
    robot = SpotRobot("spot", {"type": "agent", "translation": [0.5, 0.0, 0.5]})
    bottle = Object(
        "water_bottle_1",
        {
            "type": "object",
            "translation": [1.0, 0.8, 1.0],
            "states": {"reachable": True},
            "confidence": 0.9,
        },
    )
    for node in (room, table, robot, bottle):
        graph.add_node(node)
    graph.add_edge(table, room, "inside", "contains")
    graph.add_edge(robot, room, "inside", "contains")
    graph.add_edge(bottle, table, "on", "under")
    return graph


def test_scene_graph_exporter_snapshot_is_stable():
    graph = _build_test_world_graph()
    snapshot_a = export_scene_graph_snapshot(graph, instruction="Bring me water")
    snapshot_b = export_scene_graph_snapshot(graph, instruction="Bring me water")

    assert snapshot_a.graph_snapshot_hash == snapshot_b.graph_snapshot_hash
    assert snapshot_a.task_signature == snapshot_b.task_signature
    assert snapshot_a.nodes
    assert snapshot_a.edges
    assert snapshot_a.to_dict()["task_context"]["instruction"] == "Bring me water"


def test_build_world_state_from_graphs_contains_expected_fields():
    graph = _build_test_world_graph()
    world_state = build_world_state_from_graphs(graph, instruction="Bring me water")

    assert "graph_snapshot_hash" in world_state
    assert "scene_graph_snapshot" in world_state
    assert "relation_edges" in world_state
    assert "uncertain_entities" in world_state
    assert "water_bottle_1" in world_state["object_positions"]
    assert world_state["object_positions"]["water_bottle_1"]["parent"] == "table_surface"


def test_evolution_manager_builds_lessons_and_overlay(tmp_path: Path):
    graph = _build_test_world_graph()
    world_model_before = build_world_state_from_graphs(graph, instruction="Bring me water")

    # Force an uncertainty to exercise graph-error lessons.
    world_model_after = build_world_state_from_graphs(graph, instruction="Bring me that bottle")
    world_model_after["scene_graph_snapshot"]["uncertainties"].append(
        {"kind": "room_conflict", "entity_id": "water_bottle_1", "rooms": ["kitchen", "living_room"]}
    )

    manager = EvolutionContextManager(output_root=tmp_path, enabled=True, max_history=2)
    record = EvolutionEpisodeRecord(
        record_id="rec-1",
        scene_id=world_model_after["scene_graph_snapshot"].get("scene_id", ""),
        episode_id="episode_1",
        episode_filename="episode_1_0",
        run_id=0,
        instruction="Bring me that bottle",
        task_signature=build_task_signature("Bring me that bottle"),
        graph_snapshot_hash_before=world_model_before["graph_snapshot_hash"],
        graph_snapshot_hash_after=world_model_after["graph_snapshot_hash"],
        world_model_before=world_model_before,
        world_model_after=world_model_after,
        action_history=[
            {"agent_uid": 0, "timestamp": 1, "action": ("Navigate", "table_surface", ""), "response": "navigated"},
            {"agent_uid": 0, "timestamp": 2, "action": ("Pick", "water_bottle_1", ""), "response": "failed: object not found"},
        ],
        response_history=[
            {"agent_uid": 0, "timestamp": 1, "response": "navigated"},
            {"agent_uid": 0, "timestamp": 2, "response": "failed: object not found"},
        ],
        skill_outcomes=[
            {"skill": "Navigate", "action_type": "Navigate", "success": True},
            {"skill": "Pick", "action_type": "Pick", "success": False},
        ],
        critic_summary={"mean_return": -0.2},
        adca_summary={"root_causes": ["ambiguity around target bottle"]},
        rebound_summary={"rebound_count": 1.0, "fault_distribution": {"graph_inconsistency": 1}},
        cyclevla_summary={"rollback_count": 1},
        failure_type="referential_ambiguity",
        recovery_mode="backtrack",
        success=False,
        human_feedback=[{"message": "The bottle is on the kitchen table", "resolution": "target=water_bottle_1"}],
    )

    manager.record_episode_record(record)
    lessons = manager.lessons_by_episode["episode_1"]
    lesson_types = {lesson.lesson_type for lesson in lessons}

    assert "ambiguity" in lesson_types
    assert "graph_error" in lesson_types
    assert "recovery" in lesson_types
    assert "skill_preference" in lesson_types

    overlay = manager.retrieve_overlay(
        instruction="Bring me that bottle",
        world_state=world_model_after,
        graph_snapshot=world_model_after["scene_graph_snapshot"],
    )
    assert not overlay.is_empty()
    assert overlay.current_task_guidance
    assert overlay.preferred_recovery
    assert overlay.lesson_refs

    lesson = lessons[0]
    before_confidence = lesson.confidence
    manager.record_human_feedback({"lesson_id": lesson.lesson_id, "message": "confirmed"})
    assert manager.lesson_store[lesson.lesson_id].confidence >= before_confidence
