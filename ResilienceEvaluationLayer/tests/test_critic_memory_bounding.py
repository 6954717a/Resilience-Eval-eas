import json
from collections import deque

import numpy as np
import pytest

try:
    import habitat  # noqa: F401
except ImportError:
    habitat = None

if habitat is not None:
    from habitat_llm.evaluation.critic import A2CCritic
    from habitat_llm.evaluation.evaluation_runner import EvaluationRunner
else:
    A2CCritic = None
    EvaluationRunner = None


pytestmark = pytest.mark.skipif(
    habitat is None,
    reason="Habitat runtime is required for Critic integration tests",
)


def _bare_critic(tmp_path=None):
    critic = object.__new__(A2CCritic)
    critic.analysis_export_info_mode = "compact"
    critic.analysis_export_info_action_history_keep_last = 1
    critic.analysis_export_world_state_mode = "none"
    critic.analysis_export_include_text_embedding = False
    critic.analysis_export_keep_first_last = True
    critic.analysis_export_keep_progress_transitions = True
    critic.analysis_export_keep_response_transitions = True
    critic.analysis_export_keep_success_transition = True
    critic.analysis_export_keep_td_error_topk = 0
    critic.analysis_export_max_selected_transitions = 96
    critic.analysis_export_full_trajectory = True
    critic.analysis_export_progress_delta_threshold = 1.0e-6
    critic.analysis_focus_export_enabled = tmp_path is not None
    critic.analysis_focus_dir = str(tmp_path / "analyze") if tmp_path else ""
    critic.analysis_focus_mode = "replan_window"
    critic.analysis_replan_pre_sim_steps = 1
    critic.analysis_replan_post_sim_steps = 1
    critic.judge_model = "gpt-5.1"
    critic._episode_feature_schema = {}
    critic._focus_recent = deque(maxlen=1)
    critic._focus_stream = None
    critic._focus_stream_path = None
    critic._focus_written_indices = set()
    critic._focus_previous_planning_step = None
    critic._focus_previous_progress = None
    critic._focus_previous_success = None
    critic._focus_post_remaining = 0
    critic._focus_last_candidate = None
    critic.last_focus_export_path = None
    return critic


def test_critic_core_info_drops_tracker_and_action_world_graph():
    critic = _bare_critic()
    info = {
        "episode_id": "78",
        "episode_filename": "sample.json.gz",
        "task_percent_complete": 0.5,
        "task_state_success": 0.0,
        "sim_step_count": 42,
        "planning_step_count": 3,
        "auto_eval_proposition_tracker": {
            "proposition_satisfied_at": [12, -1]
        },
        "action_history": [
            {
                "action": ["Navigate", "table_1"],
                "response": "",
                "timestamp": 40,
                "agent_uid": 0,
                "world_graph": object(),
            }
        ],
    }

    compact = critic._compact_transition_info(info)

    assert "auto_eval_proposition_tracker" not in compact
    assert compact["proposition_satisfied_fraction"] == 0.5
    assert compact["proposition_satisfied_at"] == [12, -1]
    assert "world_graph" not in compact["action_history"][0]


def test_critic_core_artifacts_use_fixed_width_arrays_without_rich_text():
    critic = _bare_critic()
    artifacts = {
        "feature_schema_version": "v1",
        "feature_names": ["a", "b"],
        "feature_categories": ["x", "y"],
        "world_state_hash": "world",
        "text_desc": "large text that must not remain in the core buffer",
        "text_embedding": [0.1] * 384,
        "numerical_features": [1.0, 2.0],
        "state_vector": [3.0, 4.0, 5.0],
    }

    compact = critic._compact_encode_artifacts(artifacts)

    assert "text_desc" not in compact
    assert "text_embedding" not in compact
    assert compact["numerical_features"].dtype == np.float32
    assert compact["state_vector"].dtype == np.float32
    assert critic._episode_feature_schema["feature_names"] == ["a", "b"]


def test_full_export_selection_is_decoupled_from_offline_focus_selection():
    critic = _bare_critic()
    critic.analysis_export_keep_progress_transitions = False
    critic.analysis_export_keep_response_transitions = False
    critic.analysis_export_keep_success_transition = False
    transitions = [
        {
            "info": {
                "sim_step_count": index,
                "planning_step_count": 0,
                "task_percent_complete": 0.0,
                "task_state_success": 0.0,
            }
        }
        for index in range(10)
    ]

    focused, _, focused_summary = critic._select_analysis_indices(
        transitions,
        np.zeros(10, dtype=np.float32),
        include_full_trajectory=False,
    )
    full, _, full_summary = critic._select_analysis_indices(
        transitions,
        np.zeros(10, dtype=np.float32),
        include_full_trajectory=True,
    )

    assert focused == [0, 9]
    assert focused_summary["selected_transitions"] == 2
    assert full == list(range(10))
    assert full_summary["selected_transitions"] == 10


def test_replan_focus_records_are_streamed_to_results_analyze(tmp_path):
    critic = _bare_critic(tmp_path)
    planning_steps = [0, 0, 0, 1, 1, 1, 1, 1]
    for index, planning_step in enumerate(planning_steps):
        info = {
            "episode_id": "78",
            "episode_filename": "sample.json.gz",
            "sim_step_count": index,
            "planning_step_count": planning_step,
            "task_percent_complete": 0.0,
            "task_state_success": 0.0,
        }
        candidate = {
            "transition_index": index,
            "episode_id": "78",
            "episode_filename": "sample.json.gz",
            "sim_step_count": index,
            "planning_step_count": planning_step,
            "info_snapshot": dict(info),
        }
        critic._capture_focus_transition(
            index,
            candidate,
            info,
            done=index == len(planning_steps) - 1,
        )
    critic._close_focus_stream()

    focus_path = tmp_path / "analyze" / "critic_focus-episode_78_sample.json.gz.jsonl"
    payloads = [json.loads(line) for line in focus_path.read_text().splitlines()]
    records = [
        payload
        for payload in payloads
        if payload.get("record_type") == "focus_transition"
    ]

    assert payloads[0]["record_type"] == "manifest"
    assert [record["transition_index"] for record in records] == [0, 2, 3, 4, 7]
    assert len(critic._focus_recent) <= 1


def test_planner_history_compaction_keeps_metrics_and_drops_large_payloads():
    payload = {
        "sim_step_count": 10,
        "total_step_count": 8,
        "replanning_count": {0: 2, 1: 2},
        "replanned": {0: True, 1: True},
        "high_level_actions": {0: ("Pick", "cup_1")},
        "curr_graph": {0: "compact description"},
        "prompts": {0: "very large prompt"},
        "traces": {0: "very large trace"},
        "opaque_runtime_object": object(),
    }

    compact = EvaluationRunner._compact_planner_info_for_episode_history(payload)

    assert compact["replanning_count"] == {0: 2, 1: 2}
    assert compact["curr_graph"] == {0: "compact description"}
    assert "prompts" not in compact
    assert "traces" not in compact
    assert "opaque_runtime_object" not in compact
    assert EvaluationRunner._is_planner_history_event(compact)
