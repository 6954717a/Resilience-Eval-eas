#!/usr/bin/env python3

"""
World State Builder

Constructs critic-facing world state dictionaries from the environment interface.
Keeps task-context extraction and spatial state collection out of EvaluationRunner.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


class WorldStateBuilder:
    """
    Builds world state dictionaries from EnvironmentInterface.
    """

    def __init__(self, env_interface: Any):
        self.env_interface = env_interface

    def build(self, agents: Dict[int, Any], current_instruction: str = "") -> Dict[str, Any]:
        """
        Build the complete world state consumed by the critic.
        """
        world_state = {
            "agent_poses": self._extract_agent_poses(agents),
            "object_positions": {},
            "furniture_positions": {},
            "agent_holdings": self._extract_agent_holdings(),
            "object_state_snapshot": self._extract_object_state_snapshot(),
            "task_context": self._extract_task_context(current_instruction),
        }

        object_positions, furniture_positions = self._extract_spatial_entities()
        world_state["object_positions"] = object_positions
        world_state["furniture_positions"] = furniture_positions
        return world_state

    def _extract_agent_poses(self, agents: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
        agent_poses: Dict[int, Dict[str, Any]] = {}
        sim = self.env_interface.sim

        for agent_id in agents.keys():
            try:
                agent_pos = sim.agents_mgr[agent_id].articulated_agent.base_pos
                agent_rot = sim.agents_mgr[agent_id].articulated_agent.base_rot
                habitat_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
                standard_pos = self._habitat_to_standard_coords(habitat_pos)
                yaw = self._extract_agent_yaw(agent_rot)
                agent_poses[agent_id] = {
                    "position": standard_pos,
                    "rotation": (
                        [agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w]
                        if all(hasattr(agent_rot, attr) for attr in ("x", "y", "z", "w"))
                        else [float(yaw)]
                    ),
                    "yaw": float(yaw),
                }
            except Exception as exc:
                logger.debug("Could not extract pose for agent %s: %s", agent_id, exc)

        return agent_poses

    def _extract_spatial_entities(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        object_positions: Dict[str, Any] = {}
        furniture_positions: Dict[str, Any] = {}

        try:
            for _, world_graph in self.env_interface.world_graph.items():
                objects = (
                    world_graph.get_all_objects()
                    if hasattr(world_graph, "get_all_objects")
                    else []
                )
                for obj in objects:
                    if not hasattr(obj, "name") or not hasattr(obj, "properties"):
                        continue
                    standard_pos = self._extract_spatial_position(obj)
                    if standard_pos is None:
                        continue
                    object_positions[obj.name] = {
                        "position": standard_pos,
                        "parent": self._resolve_object_parent_name(world_graph, obj),
                    }

                seen_anchor_names = set()
                for getter_name in (
                    "get_all_furnitures",
                    "get_all_furniture",
                    "get_all_receptacles",
                    "get_all_rooms",
                ):
                    getter = getattr(world_graph, getter_name, None)
                    if not callable(getter):
                        continue

                    for node in getter() or []:
                        node_name = getattr(node, "name", None)
                        if not node_name or node_name in seen_anchor_names:
                            continue

                        standard_pos = self._extract_spatial_position(node)
                        if standard_pos is None:
                            continue

                        furniture_positions[node_name] = {"position": standard_pos}
                        seen_anchor_names.add(node_name)

                break
        except Exception as exc:
            logger.debug("Could not extract world graph state: %s", exc)

        return object_positions, furniture_positions

    def _extract_agent_holdings(self) -> Dict[int, Dict[str, Optional[str]]]:
        agent_holdings: Dict[int, Dict[str, Optional[str]]] = {}

        try:
            for agent_id, world_graph in self.env_interface.world_graph.items():
                held_name = None
                try:
                    if agent_id == getattr(self.env_interface, "robot_agent_uid", None):
                        agent_node = world_graph.get_spot_robot()
                    elif agent_id == getattr(self.env_interface, "human_agent_uid", None):
                        agent_node = world_graph.get_human()
                    else:
                        agent_node = None

                    if agent_node is not None:
                        held_obj = agent_node.properties.get("last_held_object")
                        if hasattr(held_obj, "name"):
                            held_name = held_obj.name
                        elif held_obj:
                            held_name = str(held_obj)
                except Exception:
                    held_name = None

                agent_holdings[agent_id] = {"held_object": held_name}
        except Exception as exc:
            logger.debug("Could not extract agent holdings: %s", exc)

        return agent_holdings

    def _extract_object_state_snapshot(self) -> Dict[str, Any]:
        try:
            sim = self.env_interface.sim
            snapshot_getter = getattr(sim, "object_state_machine", None)
            if snapshot_getter is not None and hasattr(snapshot_getter, "get_snapshot_dict"):
                return sim.object_state_machine.get_snapshot_dict(sim) or {}
        except Exception as exc:
            logger.debug("Could not extract object state snapshot: %s", exc)
        return {}

    def _extract_task_context(self, current_instruction: str) -> Dict[str, Any]:
        context = {
            "instruction": current_instruction or "",
            "objects": [],
            "object_handles": [],
            "receptacles": [],
            "receptacle_handles": [],
            "rooms": [],
            "entities": [],
            "entity_handles": [],
            "proposition_count": 0,
            "source_initial_state": [],
            "source_extra_info": {},
            "source_object_classes": [],
            "source_furniture_names": [],
            "source_allowed_regions": [],
            "object_labels": {},
            "episode_object_states": {},
        }

        try:
            episode = self.env_interface.env.env.env._env.current_episode
        except Exception:
            return context

        if not context["instruction"]:
            instruction = getattr(episode, "instruction", "")
            if isinstance(instruction, str):
                context["instruction"] = instruction

        episode_info = getattr(episode, "info", {}) or {}
        if isinstance(episode_info, dict):
            extra_info = episode_info.get("extra_info", {}) or {}
            context["source_extra_info"] = extra_info
            context["source_initial_state"] = extra_info.get("initial_state", []) or []
            context["object_labels"] = episode_info.get("object_labels", {}) or {}
            if not context["instruction"]:
                extra_instruction = extra_info.get("instruction")
                if isinstance(extra_instruction, str):
                    context["instruction"] = extra_instruction

        context["episode_object_states"] = getattr(episode, "object_states", {}) or {}

        source_object_classes = set()
        source_furniture_names = set()
        source_allowed_regions = set()
        for state_element in context["source_initial_state"]:
            if not isinstance(state_element, dict):
                continue
            for object_class in state_element.get("object_classes", []) or []:
                if object_class:
                    source_object_classes.add(str(object_class))
            for furniture_name in state_element.get("furniture_names", []) or []:
                if furniture_name:
                    source_furniture_names.add(str(furniture_name))
            for region_name in state_element.get("allowed_regions", []) or []:
                if region_name:
                    source_allowed_regions.add(str(region_name))
        context["source_object_classes"] = sorted(source_object_classes)
        context["source_furniture_names"] = sorted(source_furniture_names)
        context["source_allowed_regions"] = sorted(source_allowed_regions)

        propositions = getattr(episode, "evaluation_propositions", None)
        if not propositions:
            return context

        objects: set = set()
        receptacles: set = set()
        rooms: set = set()
        entities: set = set()

        def _add_handles(target_set: set, handles: Any) -> None:
            if isinstance(handles, (list, tuple)):
                for handle in handles:
                    if handle:
                        target_set.add(str(handle))
            elif handles:
                target_set.add(str(handles))

        for proposition in propositions:
            args = getattr(proposition, "args", {}) or {}
            _add_handles(objects, args.get("object_handles"))
            _add_handles(receptacles, args.get("receptacle_handles"))
            _add_handles(rooms, args.get("room_ids"))
            _add_handles(entities, args.get("entity_handles_a"))
            _add_handles(entities, args.get("entity_handles_b"))

            star_args = args.get("*args")
            if isinstance(star_args, (list, tuple)):
                for arg_list in star_args:
                    _add_handles(entities, arg_list)

        context["object_handles"] = sorted(objects)
        context["receptacle_handles"] = sorted(receptacles)
        context["entity_handles"] = sorted(entities)
        context["rooms"] = sorted(rooms)
        context["proposition_count"] = len(propositions)

        world_graph = self._get_world_graph_for_context()
        if world_graph is not None:

            def _map_handles(handles: List[str]) -> List[str]:
                names: List[str] = []
                for handle in handles:
                    if not world_graph.has_node_with_sim_handle(handle):
                        continue
                    try:
                        node = world_graph.get_node_from_sim_handle(handle)
                    except Exception:
                        continue
                    if getattr(node, "name", None):
                        names.append(node.name)
                return sorted(set(names))

            context["objects"] = _map_handles(context["object_handles"])
            context["receptacles"] = _map_handles(context["receptacle_handles"])
            context["entities"] = _map_handles(context["entity_handles"])

        if not context["rooms"] and context["source_allowed_regions"]:
            context["rooms"] = list(context["source_allowed_regions"])

        return context

    def _get_world_graph_for_context(self) -> Optional[Any]:
        world_graph_dict = getattr(self.env_interface, "world_graph", None)
        if isinstance(world_graph_dict, dict) and world_graph_dict:
            robot_uid = getattr(self.env_interface, "robot_agent_uid", None)
            if robot_uid is not None and robot_uid in world_graph_dict:
                return world_graph_dict[robot_uid]
            return next(iter(world_graph_dict.values()))
        return None

    @staticmethod
    def _habitat_to_standard_coords(pos: List[float]) -> List[float]:
        if len(pos) < 3:
            return pos
        return [pos[2], -pos[0], pos[1]]

    @staticmethod
    def _extract_agent_yaw(agent_rot: Any) -> float:
        if isinstance(agent_rot, (int, float, np.floating)):
            return float(agent_rot)

        if isinstance(agent_rot, (list, tuple)) and len(agent_rot) == 4:
            qx, qy, qz, qw = [float(v) for v in agent_rot]
        else:
            try:
                qw = float(agent_rot.w)
                qx = float(agent_rot.x)
                qy = float(agent_rot.y)
                qz = float(agent_rot.z)
            except Exception:
                try:
                    return float(agent_rot)
                except Exception:
                    return 0.0

        return float(
            np.arctan2(2.0 * (qw * qy - qx * qz), 1.0 - 2.0 * (qy**2 + qx**2))
        )

    def _extract_spatial_position(self, node: Any) -> Optional[List[float]]:
        properties = getattr(node, "properties", {}) or {}
        translation = properties.get("translation")
        if translation is None:
            return None
        if hasattr(translation, "tolist"):
            translation = translation.tolist()
        if not isinstance(translation, (list, tuple)) or len(translation) < 3:
            return None
        return self._habitat_to_standard_coords(list(translation))

    def _resolve_object_parent_name(self, world_graph: Any, obj: Any) -> str:
        parent_name = getattr(obj, "properties", {}).get("parent_receptacle", "unknown")

        try:
            if hasattr(world_graph, "find_receptacle_for_object"):
                receptacle = world_graph.find_receptacle_for_object(obj)
                if receptacle is not None and getattr(receptacle, "name", None):
                    parent_name = receptacle.name
        except Exception:
            pass

        try:
            if hasattr(world_graph, "find_furniture_for_object"):
                furniture = world_graph.find_furniture_for_object(obj)
                if furniture is not None and getattr(furniture, "name", None):
                    parent_name = furniture.name
        except Exception:
            pass

        return parent_name


__all__ = ["WorldStateBuilder"]
