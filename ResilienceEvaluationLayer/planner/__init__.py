# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import importlib
from typing import Any


def _optional_import(module_path: str, symbol: str) -> Any:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, symbol)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("habitat_llm"):
            raise
        return None


CentralizedLLMPlanner = _optional_import(
    "habitat_llm.planner.centralized_llm_planner",
    "CentralizedLLMPlanner",
)
LLMPlanner = _optional_import("habitat_llm.planner.llm_planner", "LLMPlanner")
Planner = _optional_import("habitat_llm.planner.planner", "Planner")
ScriptedCentralizedPlanner = _optional_import(
    "habitat_llm.planner.scripted_centralized_planner",
    "ScriptedCentralizedPlanner",
)
ThoughtlessLLMPlanner = _optional_import(
    "habitat_llm.planner.thoughtless_llm_planner",
    "ThoughtlessLLMPlanner",
)
ZeroShotReactPlanner = _optional_import(
    "habitat_llm.planner.zero_shot_react_planner",
    "ZeroShotReactPlanner",
)
