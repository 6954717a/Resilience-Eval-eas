# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import importlib
from typing import Any

from habitat_llm.world_model.entity import (
    Entity,
    House,
    Human,
    Object,
    Receptacle,
    Room,
    SpotRobot,
    UncategorizedEntity,
)


def _optional_import(module_path: str, symbol: str) -> Any:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, symbol)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("habitat_llm"):
            raise
        return None


Floor = _optional_import("habitat_llm.world_model.entities.floor", "Floor")
Furniture = _optional_import("habitat_llm.world_model.entities.furniture", "Furniture")
from habitat_llm.world_model.graph import Graph
from habitat_llm.world_model.scene_graph_exporter import (
    SceneGraphSnapshot,
    build_task_signature,
    build_world_state_from_graphs,
    compute_stable_hash,
    export_scene_graph_snapshot,
)
WorldGraph = _optional_import("habitat_llm.world_model.world_graph", "WorldGraph")
DynamicWorldGraph = _optional_import(
    "habitat_llm.world_model.dynamic_world_graph",
    "DynamicWorldGraph",
)
