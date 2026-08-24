import ast
import threading
import time
from pathlib import Path

import pytest
import yaml

from habitat_llm.examples.resilience_scheduler import (
    BatchExecutionError,
    ExecutionSettings,
    chunk_episode_ids,
    create_attempt_dir,
    run_bounded_jobs,
)


def test_default_execution_profile_is_single_worker_with_host_reserve():
    settings = ExecutionSettings.from_mapping(None)

    assert settings.max_parallel_workers == 1
    assert settings.episodes_per_worker == 1
    assert settings.min_available_memory_gb == 48.0


def test_scheduler_never_exceeds_configured_worker_cap():
    lock = threading.Lock()
    active = 0
    observed_peak = 0

    def worker(value):
        nonlocal active, observed_peak
        with lock:
            active += 1
            observed_peak = max(observed_peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return value * 2

    jobs = [(f"job-{index}", index) for index in range(6)]
    report = run_bounded_jobs(
        jobs,
        worker,
        ExecutionSettings(
            max_parallel_workers=2,
            episodes_per_worker=1,
            min_available_memory_gb=24,
        ),
        memory_reader=lambda: 120.0,
    )

    assert observed_peak == 2
    assert report.max_active_workers == 2
    assert [outcome.result for outcome in report.outcomes] == [0, 2, 4, 6, 8, 10]


def test_scheduler_stops_new_jobs_after_worker_failure():
    started = []
    lock = threading.Lock()

    def worker(value):
        with lock:
            started.append(value)
        if value == 0:
            raise RuntimeError("boom")
        time.sleep(0.04)
        return value

    with pytest.raises(BatchExecutionError) as exc_info:
        run_bounded_jobs(
            [(f"job-{index}", index) for index in range(5)],
            worker,
            ExecutionSettings(max_parallel_workers=2, min_available_memory_gb=0),
            memory_reader=lambda: 120.0,
        )

    assert set(started).issubset({0, 1})
    assert any(outcome.status == "failed" for outcome in exc_info.value.report.outcomes)
    assert any(
        outcome.status == "not_started" for outcome in exc_info.value.report.outcomes
    )


def test_scheduler_can_record_failure_and_continue_later_jobs():
    started = []

    def worker(value):
        started.append(value)
        if value == 1:
            raise RuntimeError("bad episode")
        return value * 2

    report = run_bounded_jobs(
        [(f"job-{index}", index) for index in range(4)],
        worker,
        ExecutionSettings(max_parallel_workers=1, min_available_memory_gb=0),
        memory_reader=lambda: 120.0,
        continue_after_worker_failure=True,
    )

    assert started == [0, 1, 2, 3]
    assert [outcome.status for outcome in report.outcomes] == [
        "completed",
        "failed",
        "completed",
        "completed",
    ]
    assert "bad episode" in str(report.outcomes[1].error)


def test_scheduler_refuses_first_worker_below_memory_reserve():
    started = []
    with pytest.raises(BatchExecutionError) as exc_info:
        run_bounded_jobs(
            [("episode-78", "78")],
            lambda value: started.append(value),
            ExecutionSettings(max_parallel_workers=2, min_available_memory_gb=24),
            memory_reader=lambda: 12.0,
            continue_after_worker_failure=True,
        )

    assert started == []
    assert "below the 24.00 GiB reserve" in str(exc_info.value)


def test_episode_shards_are_single_episode_and_ordered():
    assert chunk_episode_ids([78, 127, 11], 1) == [("78",), ("127",), ("11",)]


def test_worker_attempt_directories_are_fresh(tmp_path):
    first = create_attempt_dir(tmp_path / "mp_workers")
    (first / "episode_metrics.csv").write_text("old evidence", encoding="utf-8")

    second = create_attempt_dir(tmp_path / "mp_workers")

    assert first != second
    assert first.parent == second.parent == tmp_path / "mp_workers"
    assert not (second / "episode_metrics.csv").exists()


def test_action_history_builds_one_world_graph_snapshot_per_replan_event():
    repo_root = Path(__file__).resolve().parents[2]
    runner_path = repo_root / "habitat_llm/evaluation/evaluation_runner.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "update_agent_action_history"
    )
    snapshot_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "snapshot"
    ]
    agent_loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "agent_id"
    ]

    assert len(snapshot_calls) == 1
    assert agent_loops
    assert snapshot_calls[0].lineno < agent_loops[0].lineno


def test_resilience_workers_use_bounded_planner_history():
    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root / "habitat_llm/examples/resilience_helpers.py"
    ).read_text(encoding="utf-8")

    assert 'cfg.evaluation.planner_history_mode = "replan_window"' in source
    assert "cfg.evaluation.retain_action_history_world_graph = False" in source
    assert "cfg.evaluation.log_detailed_traces = False" in source


def test_critic_buffer_uses_compact_info_and_disk_focus_stream():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "habitat_llm/evaluation/critic.py").read_text(
        encoding="utf-8"
    )

    assert "compact_info = self._compact_transition_info(info)" in source
    assert '"state_world": None' in source
    assert '"next_state_world": None' in source
    assert "critic_focus-episode_" in source
    assert 'include_full_trajectory=False' in source


def _load_yaml(relative_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((repo_root / relative_path).read_text(encoding="utf-8"))


def test_expq_beta_and_ge_configs_are_model_and_output_aligned():
    beta_qwen3 = _load_yaml("habitat_llm/conf/experiments/expq1_beta.yaml")
    beta_qwen35 = _load_yaml("habitat_llm/conf/experiments/expq1_beta_qwen35.yaml")
    ge_qwen3 = _load_yaml("habitat_llm/conf/experiments/expq1_ge.yaml")
    ge_qwen35 = _load_yaml(
        "habitat_llm/conf/experiments/expq1_ge_qwen35.yaml"
    )

    assert beta_qwen3["defaults"][0].startswith("/baselines/qwen3_")
    assert beta_qwen35["defaults"][0].startswith("/baselines/qwen35_")
    assert beta_qwen3["expq1"]["output_subdir"] == "expq1_beta_qwen3"
    assert beta_qwen35["expq1"]["output_subdir"] == "expq1_beta_qwen35"
    assert ge_qwen3["expq1"]["output_subdir"] == "expq1_ge_qwen3"
    assert ge_qwen35["expq1"]["output_subdir"] == "expq1_ge_qwen35"

    for config in (beta_qwen3, beta_qwen35):
        assert config["evaluation"]["critic"]["llm_model"] == "gpt-5.1"
        assert config["evaluation"]["adca"]["llm_model"] == "gpt-5.1"
        beta = config["expq1"]["shared"]["base_patch"]["resilience"]["beta"]
        assert "weight_vv" not in beta
        assert "weight_out" not in beta

    block = beta_qwen35["expq1"]["subexperiments"]["stability_execution_validity"]
    configured_metrics = block["support_metrics"] + block["summary_metrics"]
    assert not any(metric.endswith("_vv") for metric in configured_metrics)
    assert "stability_beta_perturbation_trajectory" in configured_metrics
    assert "sim_step_count" in block["summary_metrics"]


def test_joint_resilience_config_uses_one_shared_three_metric_grid():
    config = _load_yaml(
        "habitat_llm/conf/experiments/resilience_joint_qwen35.yaml"
    )
    suite = config["expq1"]
    blocks = suite["subexperiments"]

    assert list(blocks) == ["joint_resilience_evaluation"]
    block = blocks["joint_resilience_evaluation"]
    assert block["postprocess"] == ["stability", "margin"]
    assert {
        "rebound_c_rec",
        "stability_beta_neighborhood",
        "ge_margin_soft",
        "ge_audc_f",
    }.issubset(block["support_metrics"])

    phases = block["phases"]
    assert [phase["label"] for phase in phases] == ["stress_sweep"]
    resilience = phases[0]["patch"]["resilience"]
    assert resilience["mode"] == "joint"
    assert resilience["perturbations"] == [
        "clean",
        "llm_instruction_context_stress",
    ]
    assert resilience["seeds"] == [0, 1, 2]
    assert resilience["intensities"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    execution = suite["shared"]["base_patch"]["resilience"]["execution"]
    assert execution["max_parallel_workers"] == 1
    assert execution["episodes_per_worker"] == 1
    assert execution["min_available_memory_gb"] == 48


def test_standalone_joint_mode_runs_stability_before_margin():
    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root / "habitat_llm/examples/planner_demo_resilience.py"
    ).read_text(encoding="utf-8")

    assert 'elif mode == "joint":' in source
    assert 'requested_postprocess = ["stability", "margin"]' in source
