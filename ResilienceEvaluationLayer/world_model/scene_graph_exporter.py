import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from habitat_llm.world_model.entity import Entity, Human, Object, Receptacle, Room, SpotRobot


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=_json_default, separators=(",", ":"))


def compute_stable_hash(data: Any) -> str:
    return hashlib.sha256(_stable_json(data).encode("utf-8")).hexdigest()


def build_task_signature(instruction: str, task_context: Optional[Dict[str, Any]] = None) -> str:
    task_context = task_context or {}
    normalized = " ".join((instruction or "").strip().lower().split())
    intent = task_context.get("intent") or ""
    target = task_context.get("candidate_target") or task_context.get("target") or ""
    signature_source = {"instruction": normalized, "intent": intent, "target": target}
    return compute_stable_hash(signature_source)[:16]


@dataclass
class SceneGraphSnapshot:
    scene_id: str = ""
    episode_id: str = ""
    timestamp: float = 0.0
    agent_states: Dict[str, Any] = field(default_factory=dict)
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    uncertainties: List[Dict[str, Any]] = field(default_factory=list)
    task_context: Dict[str, Any] = field(default_factory=dict)
    graph_snapshot_hash: str = ""
    task_signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_env_metadata(env_interface: Any) -> Tuple[str, str]:
    scene_id = ""
    episode_id = ""
    if env_interface is None:
        return scene_id, episode_id
    try:
        episode = env_interface.env.env.env._env.current_episode
        scene_id = str(getattr(episode, "scene_id", "") or "")
        episode_id = str(getattr(episode, "episode_id", "") or "")
    except Exception:
        pass
    return scene_id, episode_id


def _safe_position(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
        return [float(x) for x in value]
    try:
        if hasattr(value, "__len__") and len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        pass
    return None


def _node_labels(node: Entity) -> List[str]:
    labels: List[str] = [node.__class__.__name__.lower()]
    for key in ("type", "category", "semantic_class"):
        value = node.properties.get(key)
        if isinstance(value, str) and value:
            labels.append(value.lower())
    return sorted(set(labels))


def _node_geometry(node: Entity) -> Dict[str, Any]:
    geometry: Dict[str, Any] = {}
    translation = _safe_position(node.properties.get("translation"))
    if translation is not None:
        geometry["position"] = translation
    for key in ("bbox_min", "bbox_max", "bbox_extent"):
        value = node.properties.get(key)
        if value is not None:
            geometry[key] = _safe_position(value) or value
    return geometry


def _node_states(node: Entity, graph: Any) -> Dict[str, Any]:
    state = dict(node.properties.get("states", {}) or {})
    if isinstance(node, Object):
        holders = [
            neighbor.name
            for neighbor, edge in graph.graph.get(node, {}).items()
            if edge and isinstance(neighbor, (SpotRobot, Human))
        ]
        if holders:
            state["held_by"] = sorted(set(holders))
    update_ts = node.properties.get("time_of_update")
    if update_ts is not None:
        state["observed_at"] = update_ts
    return state


def _infer_source(node: Entity) -> str:
    if node.properties.get("time_of_update") is not None:
        return "detector"
    if node.sim_handle:
        return "concept_graph"
    return "world_model"


def _iter_edges(graph: Any) -> Iterable[Dict[str, Any]]:
    for node, neighbors in graph.graph.items():
        for neighbor, predicate in neighbors.items():
            if not predicate:
                continue
            yield {
                "subject": node.name,
                "predicate": predicate,
                "object": neighbor.name,
                "confidence": float(node.properties.get("confidence", 1.0) or 1.0),
            }


def _build_uncertainties(graph: Any, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uncertainties: List[Dict[str, Any]] = []
    room_membership: Dict[str, List[str]] = {}
    parent_membership: Dict[str, List[str]] = {}
    for edge in edges:
        if edge["predicate"] in {"inside", "in", "has", "contains"}:
            room_membership.setdefault(edge["subject"], []).append(edge["object"])
        if edge["predicate"] in {"on", "within", "in", "inside"}:
            parent_membership.setdefault(edge["subject"], []).append(edge["object"])

    for node in nodes:
        confidence = float(node.get("confidence", 1.0) or 1.0)
        if confidence < 0.5:
            uncertainties.append(
                {
                    "kind": "low_confidence_node",
                    "entity_id": node["id"],
                    "confidence": confidence,
                }
            )
        if node["type"] == "object":
            if not node.get("geometry", {}).get("position"):
                uncertainties.append(
                    {
                        "kind": "unknown_position",
                        "entity_id": node["id"],
                    }
                )
            rooms = room_membership.get(node["id"], [])
            if len(set(rooms)) > 1:
                uncertainties.append(
                    {
                        "kind": "room_conflict",
                        "entity_id": node["id"],
                        "rooms": sorted(set(rooms)),
                    }
                )
            parents = parent_membership.get(node["id"], [])
            if len(set(parents)) > 1:
                uncertainties.append(
                    {
                        "kind": "parent_conflict",
                        "entity_id": node["id"],
                        "parents": sorted(set(parents)),
                    }
                )
    return uncertainties


def _build_agent_states(graph_dict: Dict[int, Any], env_interface: Any) -> Dict[str, Any]:
    agent_states: Dict[str, Any] = {}
    for agent_id, graph in graph_dict.items():
        state: Dict[str, Any] = {"holding": None}
        agent_node = None
        try:
            if env_interface is not None and agent_id == getattr(env_interface, "robot_agent_uid", None):
                agent_node = graph.get_spot_robot()
            elif env_interface is not None and agent_id == getattr(env_interface, "human_agent_uid", None):
                agent_node = graph.get_human()
        except Exception:
            agent_node = None

        if agent_node is None:
            try:
                agents = graph.get_agents()
                if agents:
                    agent_node = agents[0]
            except Exception:
                agent_node = None

        if agent_node is not None:
            state["name"] = agent_node.name
            state["position"] = _safe_position(agent_node.properties.get("translation"))
            held_obj = agent_node.properties.get("last_held_object")
            if hasattr(held_obj, "name"):
                state["holding"] = held_obj.name
            elif held_obj:
                state["holding"] = str(held_obj)

            try:
                rooms = graph.get_neighbors_of_type(agent_node, Room)
                if rooms:
                    state["room"] = rooms[0].name
            except Exception:
                pass
        agent_states[str(agent_id)] = state
    return agent_states


def export_scene_graph_snapshot(
    world_graph: Any,
    env_interface: Any = None,
    instruction: str = "",
    task_context: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> SceneGraphSnapshot:
    task_context = dict(task_context or {})
    timestamp = float(timestamp if timestamp is not None else time.time())
    scene_id, episode_id = _get_env_metadata(env_interface)

    if isinstance(world_graph, dict):
        graph_dict = world_graph
        selected_graph = None
        if env_interface is not None:
            robot_uid = getattr(env_interface, "robot_agent_uid", None)
            if robot_uid in graph_dict:
                selected_graph = graph_dict[robot_uid]
        if selected_graph is None and graph_dict:
            selected_graph = next(iter(graph_dict.values()))
    else:
        graph_dict = {}
        selected_graph = world_graph

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    if selected_graph is not None:
        for node in sorted(selected_graph.graph.keys(), key=lambda item: item.name):
            node_payload = {
                "id": node.name,
                "type": node.__class__.__name__.lower(),
                "labels": _node_labels(node),
                "state": _node_states(node, selected_graph),
                "geometry": _node_geometry(node),
                "confidence": float(node.properties.get("confidence", 1.0) or 1.0),
                "source": _infer_source(node),
            }
            nodes.append(node_payload)
        edges = sorted(
            list(_iter_edges(selected_graph)),
            key=lambda item: (item["subject"], item["predicate"], item["object"]),
        )

    agent_graphs = graph_dict
    if not agent_graphs and selected_graph is not None:
        agent_graphs = {"0": selected_graph}
    agent_states = _build_agent_states(agent_graphs, env_interface)
    task_context.setdefault("instruction", instruction or "")
    task_context.setdefault("active_skills", [])
    task_signature = build_task_signature(instruction, task_context)
    uncertainties = _build_uncertainties(selected_graph, nodes, edges) if selected_graph is not None else []

    hash_payload = {
        "scene_id": scene_id,
        "episode_id": episode_id,
        "agent_states": agent_states,
        "nodes": nodes,
        "edges": edges,
        "task_context": task_context,
    }
    snapshot_hash = compute_stable_hash(hash_payload)

    return SceneGraphSnapshot(
        scene_id=scene_id,
        episode_id=episode_id,
        timestamp=timestamp,
        agent_states=agent_states,
        nodes=nodes,
        edges=edges,
        uncertainties=uncertainties,
        task_context=task_context,
        graph_snapshot_hash=snapshot_hash,
        task_signature=task_signature,
    )


def build_world_state_from_graphs(
    world_graph: Any,
    env_interface: Any = None,
    instruction: str = "",
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot = export_scene_graph_snapshot(
        world_graph=world_graph,
        env_interface=env_interface,
        instruction=instruction,
        task_context=task_context,
    )

    world_state: Dict[str, Any] = {
        "agent_poses": {},
        "object_positions": {},
        "furniture_positions": {},
        "agent_holdings": {},
        "relation_edges": snapshot.edges,
        "uncertain_entities": snapshot.uncertainties,
        "task_context": snapshot.task_context,
        "scene_graph_snapshot": snapshot.to_dict(),
        "graph_snapshot_hash": snapshot.graph_snapshot_hash,
        "task_signature": snapshot.task_signature,
        "agent_states": snapshot.agent_states,
    }

    for agent_id, state in snapshot.agent_states.items():
        pose: Dict[str, Any] = {}
        if state.get("position") is not None:
            pose["position"] = state["position"]
        if state.get("room"):
            pose["room"] = state["room"]
        world_state["agent_poses"][agent_id] = pose
        world_state["agent_holdings"][agent_id] = {"held_object": state.get("holding")}

    for node in snapshot.nodes:
        geometry = node.get("geometry", {})
        state = node.get("state", {})
        if node["type"] == "object":
            parent = None
            for edge in snapshot.edges:
                if edge["subject"] == node["id"] and edge["predicate"] in {"in", "inside", "on", "within"}:
                    parent = edge["object"]
                    break
            world_state["object_positions"][node["id"]] = {
                "position": geometry.get("position"),
                "parent": parent,
                "labels": node.get("labels", []),
                "state": state,
            }
        elif node["type"] in {"furniture", "receptacle", "room"}:
            world_state["furniture_positions"][node["id"]] = {
                "position": geometry.get("position"),
                "labels": node.get("labels", []),
                "state": state,
            }

    first_agent = next(iter(snapshot.agent_states.values()), {})
    world_state["agent_holding"] = first_agent.get("holding")
    world_state["current_room"] = first_agent.get("room")
    world_state["objects"] = {
        name: {
            "position": info.get("position"),
            "location": info.get("parent"),
            "labels": info.get("labels", []),
            "state": info.get("state", {}),
        }
        for name, info in world_state["object_positions"].items()
    }
    return world_state
