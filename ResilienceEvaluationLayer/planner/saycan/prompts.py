"""
SayCan Prompts

Defines prompts that request multiple candidate actions with confidence scores.
"""

SAYCAN_SYSTEM_PROMPT = """You are a robot planner.
Instead of executing a single action, list the top {num_candidates} most useful actions for the instruction.
Assign a confidence score (0.0 to 1.0) to each, reflecting how much it advances the goal.

Format: JSON list.
Example:
[
  {{"action": "Navigate[kitchen_table]", "confidence": 0.9}},
  {{"action": "Pick[apple]", "confidence": 0.1}}
]
"""
