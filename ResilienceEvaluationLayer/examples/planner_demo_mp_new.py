#!/usr/bin/env python3
# isort: skip_file

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
# Fix for huggingface/tokenizers parallelism issue in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import os
import traceback
import json
import subprocess
import gc
import copy
import torch
from omegaconf import OmegaConf


# append the path of the
# parent directory
sys.path.append("..")

import hydra
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import multiprocessing as mp

from habitat_llm.agent.env.evaluation.evaluation_functions import (
    aggregate_measures,
)

from habitat_llm.utils import cprint, setup_config, fix_config
from habitat_llm.config_redaction import (
    redacted_config_yaml,
    write_redacted_config_copy,
)


from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.evaluation import (
    CentralizedEvaluationRunner,
    DecentralizedEvaluationRunner,
    EvaluationRunner,
)
from habitat_llm.evaluation.reporting import (
    append_csv_rows,
    build_episode_csv_metrics as _build_episode_csv_metrics,
    collect_episode_stats_keys as _collect_episode_stats_keys,
    primary_display_items,
    read_csv_rows,
    write_csv_rows_exact,
)
from habitat_llm.examples.resilience_scheduler import (
    BatchExecutionError,
    BatchRun,
    ExecutionSettings,
    create_attempt_dir,
    run_bounded_jobs,
)
# Use new planner-based evolution modules
from habitat_llm.planner.evolve import (
    EvolutionContextManager,
    DynamicEpisodeScheduler,
    apply_evolution_context,
    build_episode_summary,
    resolve_analysis_roots,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_baselines.utils.info_dict import extract_scalars_from_info


def clear_memory():
    """Release cyclic Python objects between Habitat episodes."""
    gc.collect()


def get_output_file(config, env_interface):
    dataset_file = env_interface.conf.habitat.dataset.data_path.split("/")[-1]
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    output_file = os.path.join(
        config.paths.results_dir,
        dataset_file,
        "stats",
        f"{episode_id}.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file


# Function to write data to the CSV file
def write_to_csv(file_name, result_dict):
    result_dict = dict(sorted(result_dict.items()))
    append_csv_rows(
        file_name,
        [result_dict],
        fieldnames=list(result_dict),
        schema_policy="expand",
        empty_policy="error",
    )


def save_exception_message(config, env_interface):
    output_file = get_output_file(config, env_interface)
    exc_string = traceback.format_exc()
    failure_dict = {"success": False, "info": str(exc_string)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


def save_success_message(config, env_interface, info):
    output_file = get_output_file(config, env_interface)
    failure_dict = {"success": True, "stats": json.dumps(info)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


# Write the config file into the results folders
def write_config(config):
    dataset_file = config.habitat.dataset.data_path.split("/")[-1]
    output_file = os.path.join(config.paths.results_dir, dataset_file)
    os.makedirs(output_file, exist_ok=True)
    with open(f"{output_file}/config.yaml", "w+") as f:
        f.write(redacted_config_yaml(config))

    # Copy over the RLM config
    planner_configs = []
    suffixes = []
    if "planner" in config.evaluation:
        # Centralized
        if "plan_config" in config.evaluation.planner is not None:
            planner_configs = [config.evaluation.planner.plan_config]
            suffixes = [""]
    else:
        for agent_name in config.evaluation.agents:
            suffixes.append(f"_{agent_name}")
            planner_configs.append(
                config.evaluation.agents[agent_name].planner.plan_config
            )

    for plan_config, suffix_rlm in zip(planner_configs, suffixes):
        if "llm" in plan_config and "serverdir" in plan_config.llm:
            yaml_rlm_path = plan_config.llm.serverdir
            if len(yaml_rlm_path) > 0:
                yaml_rlm_file = f"{yaml_rlm_path}/config.yaml"
                if os.path.isfile(yaml_rlm_file):
                    write_redacted_config_copy(
                        yaml_rlm_file,
                        f"{output_file}/config_rlm{suffix_rlm}.yaml",
                    )


# Allow selecting episodes by either dataset index or explicit episode id.
def _select_episode_subset(dataset: CollaborationDatasetV0, config) -> CollaborationDatasetV0:
    episode_indices = config.get("episode_indices", None)
    episode_ids = config.get("episode_ids", None)
    if episode_ids is None:
        episode_ids = config.get("evolve_episode_ids", None)

    if episode_indices is not None and episode_ids is not None:
        raise ValueError("Specify only one of episode_indices or episode_ids")

    if episode_indices is not None:
        indices = list(episode_indices)
        if any(ix < 0 or ix >= len(dataset.episodes) for ix in indices):
            raise ValueError("Episode index out of range.")
        episode_subset = [dataset.episodes[ix] for ix in indices]
        return CollaborationDatasetV0(config=config.habitat.dataset, episodes=episode_subset)

    if episode_ids is not None:
        wanted_ids = [str(eid) for eid in episode_ids]
        episode_subset = [ep for ep in dataset.episodes if str(ep.episode_id) in wanted_ids]
        missing = sorted(set(wanted_ids) - {str(ep.episode_id) for ep in episode_subset})
        if missing:
            raise ValueError(f"Could not find episode ids: {missing}")
        return CollaborationDatasetV0(config=config.habitat.dataset, episodes=episode_subset)

    return dataset


def _mp_execution_settings(config, requested_workers: int) -> ExecutionSettings:
    raw = OmegaConf.select(config, "resilience.execution")
    mapping: Dict[str, Any] = {}
    if raw is not None:
        converted = OmegaConf.to_container(raw, resolve=True)
        if isinstance(converted, dict):
            mapping = converted
    configured = ExecutionSettings.from_mapping(mapping)
    return ExecutionSettings(
        max_parallel_workers=min(
            max(1, requested_workers), configured.max_parallel_workers
        ),
        episodes_per_worker=1,
        min_available_memory_gb=configured.min_available_memory_gb,
    )


def _write_rows_exact(
    path: Path,
    rows: List[Dict[str, Any]],
    *,
    empty_fieldnames: Optional[Sequence[str]] = None,
) -> None:
    fieldnames = (
        sorted({str(key) for row in rows for key in row})
        if rows
        else [str(key) for key in empty_fieldnames or ()]
    )
    write_csv_rows_exact(
        path,
        rows,
        fieldnames=fieldnames,
        overwrite_policy="replace",
        empty_policy="write_header" if fieldnames else "error",
    )


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return list(read_csv_rows(path))


def _numeric_episode_metrics(row: Dict[str, Any]) -> Dict[str, float]:
    excluded = {
        "episode_id",
        "instruction",
        "run_id",
        "task_family",
        "action_sequence",
    }
    metrics: Dict[str, float] = {}
    for key, value in row.items():
        if key in excluded or value in (None, ""):
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def _write_mp_execution_manifest(
    output_path: Path,
    report: BatchRun,
    settings: ExecutionSettings,
    metadata: Dict[str, Dict[str, Any]],
) -> None:
    status_counts = {
        outcome_status: sum(
            1 for outcome in report.outcomes if outcome.status == outcome_status
        )
        for outcome_status in ("completed", "failed", "not_started")
    }
    batch_status = (
        "failed"
        if status_counts["not_started"]
        else "completed_with_failures"
        if status_counts["failed"]
        else "completed"
    )
    failed_episode_ids = [
        str(episode_id)
        for outcome in report.outcomes
        if outcome.status == "failed"
        for episode_id in metadata.get(outcome.job_id, {}).get("episode_ids", [])
    ]
    jobs = []
    for outcome in report.outcomes:
        job = dict(metadata.get(outcome.job_id, {}))
        resource_path = Path(str(job.get("resource_usage_path", "")))
        resource_usage = None
        if resource_path.is_file():
            try:
                resource_usage = json.loads(resource_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        jobs.append(
            {
                "job_id": outcome.job_id,
                "status": outcome.status,
                "error": outcome.error,
                "started_at": outcome.started_at,
                "finished_at": outcome.finished_at,
                **job,
                "resource_usage": resource_usage,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "execution": {
                    "max_parallel_workers": settings.max_parallel_workers,
                    "episodes_per_worker": 1,
                    "min_available_memory_gb": settings.min_available_memory_gb,
                    "worker_failure_policy": "continue",
                },
                "status": batch_status,
                "status_counts": status_counts,
                "failed_episode_ids": failed_episode_ids,
                "max_active_workers": report.max_active_workers,
                "jobs": jobs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_isolated_episode_batch(config, dataset: CollaborationDatasetV0) -> None:
    """Run each selected episode in a process-isolated, bounded worker."""

    requested_workers = int(config.get("num_proc", 1))
    settings = _mp_execution_settings(config, requested_workers)
    worker_root = Path(str(config.paths.results_dir)) / "mp_workers"
    attempt_root = create_attempt_dir(worker_root)
    attempt_id = attempt_root.name.removeprefix("attempt_")
    jobs = []
    metadata: Dict[str, Dict[str, Any]] = {}

    for order, episode in enumerate(dataset.episodes):
        episode_id = str(episode.episode_id)
        worker_dir = attempt_root / f"episode_{episode_id}"
        job_id = f"{attempt_id}::episode_{episode_id}"
        payload = {
            "order": order,
            "episode_id": episode_id,
            "worker_dir": worker_dir,
        }
        jobs.append((job_id, payload))
        metadata[job_id] = {
            "attempt_id": attempt_id,
            "episode_ids": [episode_id],
            "worker_dir": str(worker_dir),
            "worker_log_path": str(worker_dir / "worker.log"),
            "resource_usage_path": str(worker_dir / "resource_usage.json"),
        }

    def run_job(payload: Dict[str, Any]) -> Dict[str, Any]:
        worker_dir = Path(payload["worker_dir"])
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_cfg = copy.deepcopy(config)
        OmegaConf.set_struct(worker_cfg, False)
        worker_cfg.num_proc = 1
        for key in (
            "episode_indices",
            "evolve_episode_ids",
            "start_index",
            "end_index",
        ):
            if worker_cfg.get(key, None) is not None:
                del worker_cfg[key]
        worker_cfg.episode_ids = [payload["episode_id"]]
        worker_cfg.paths.results_dir = str(worker_dir)
        worker_cfg.paths.epi_result_file_path = str(worker_dir / "episode_metrics.csv")
        worker_cfg.paths.run_result_file_path = str(worker_dir / "run_result_log.csv")
        worker_cfg.paths.end_result_file_path = str(worker_dir / "end_result_log.csv")
        config_path = worker_dir / "_resolved_config.yaml"
        config_path.write_text(
            redacted_config_yaml(worker_cfg, resolve=True, replacement=None),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            "-m",
            "habitat_llm.examples.resilience_helpers",
            "--spec-config",
            str(config_path),
            "--resource-usage",
            str(worker_dir / "resource_usage.json"),
        ]
        with open(worker_dir / "worker.log", "w", encoding="utf-8") as worker_log:
            result = subprocess.run(
                command,
                check=False,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
            )
        csv_path = worker_dir / "episode_metrics.csv"
        if result.returncode != 0:
            raise RuntimeError(
                f"episode {payload['episode_id']} exited with code {result.returncode}"
            )
        if not csv_path.exists():
            raise RuntimeError(f"episode {payload['episode_id']} produced no metrics CSV")
        rows = _read_csv_rows(csv_path)
        written_episode_ids = {
            str(row.get("episode_id", "")) for row in rows
        }
        expected_episode_id = str(payload["episode_id"])
        if not rows or written_episode_ids != {expected_episode_id}:
            raise RuntimeError(
                f"episode {expected_episode_id} produced unexpected metrics evidence: "
                f"{sorted(written_episode_ids)}"
            )
        return {**payload, "csv_path": str(csv_path)}

    try:
        report = run_bounded_jobs(
            jobs,
            run_job,
            settings,
            continue_after_worker_failure=True,
        )
    except BatchExecutionError as exc:
        _write_mp_execution_manifest(
            worker_root / "execution_manifest.json",
            exc.report,
            settings,
            metadata,
        )
        raise RuntimeError(
            f"Episode worker batch failed; see {worker_root / 'execution_manifest.json'}"
        ) from exc

    ordered_results = sorted(
        (
            outcome.result
            for outcome in report.outcomes
            if outcome.status == "completed"
        ),
        key=lambda item: item["order"],
    )
    episode_rows: List[Dict[str, Any]] = []
    for result in ordered_results:
        episode_rows.extend(_read_csv_rows(Path(result["csv_path"])))
    _write_rows_exact(
        Path(str(config.paths.epi_result_file_path)),
        episode_rows,
        empty_fieldnames=("episode_id", "run_id", "instruction"),
    )

    stats_by_episode = {
        f"{row.get('episode_id', 'unknown')}::{row.get('run_id', '0')}::{index}": (
            _numeric_episode_metrics(row)
        )
        for index, row in enumerate(episode_rows)
    }
    aggregate = aggregate_measures(stats_by_episode) if stats_by_episode else {}
    if aggregate:
        cprint("Metrics Across Isolated Episode Workers:", "blue")
        for key, value in primary_display_items(aggregate):
            print(f"  {key}: {value}")
        _write_rows_exact(Path(str(config.paths.run_result_file_path)), [aggregate])
        _write_rows_exact(Path(str(config.paths.end_result_file_path)), [aggregate])

    _write_mp_execution_manifest(
        worker_root / "execution_manifest.json", report, settings, metadata
    )
    failed_count = sum(
        1 for outcome in report.outcomes if outcome.status == "failed"
    )
    print(
        f"Completed {len(episode_rows)} episode rows with {failed_count} failure(s) "
        f"and at most {report.max_active_workers} isolated workers."
    )


# Method to load agent planner from the config
@hydra.main(config_path="../conf", version_base=None)
def run_eval(config):
    fix_config(config)
    # Setup a seed
    # seed = 48212516
    seed = 47668090
    t0 = time.time()
    # Setup config
    config = setup_config(config, seed)
    dataset = CollaborationDatasetV0(config.habitat.dataset)

    write_config(config)
    if config.get("resume", False):
        dataset_file = config.habitat.dataset.data_path.split("/")[-1]
        # stats_dir = os.path.join(config.paths.results_dir, dataset_file, "stats")
        plan_log_dir = os.path.join(
            config.paths.results_dir, dataset_file, "planner-log"
        )

        # Find incomplete episodes
        incomplete_episodes = []
        for episode in dataset.episodes:
            episode_id = episode.episode_id
            # stats_file = os.path.join(stats_dir, f"{episode_id}.json")
            planlog_file = os.path.join(
                plan_log_dir, f"planner-log-episode_{episode_id}_0.json"
            )
            if not os.path.exists(planlog_file):
                incomplete_episodes.append(episode)
        print(
            f"Resuming with {len(incomplete_episodes)} incomplete episodes: {[e.episode_id for e in incomplete_episodes]}"
        )
        # Update dataset with only incomplete episodes
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=incomplete_episodes
        )

    # filter episodes by mod for running on multiple nodes
    if config.get("episode_mod_filter", None) is not None:
        rem, mod = config.episode_mod_filter
        episode_subset = [x for x in dataset.episodes if int(x.episode_id) % mod == rem]
        print(f"Mod filter: {rem}, {mod}")
        print(f"Episodes: {[e.episode_id for e in episode_subset]}")
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )

    # Allow running a specific subset of indices via command line overrides
    # This is used by the subprocess worker mode
    if config.get("start_index", None) is not None and config.get("end_index", None) is not None:
        start_idx = config.start_index
        end_idx = config.end_index
        print(f"Worker running indices [{start_idx}:{end_idx}]")
        episode_subset = dataset.episodes[start_idx:end_idx]
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )
        run_planner(config, dataset)
        return

    if (config.get("episode_indices", None) is not None or
            config.get("episode_ids", None) is not None or
            config.get("evolve_episode_ids", None) is not None):
        if config.get("resume", False):
            raise ValueError("episode selection and resume cannot be used together")
        dataset = _select_episode_subset(dataset, config)

    if int(config.get("num_proc", 1)) == 1:
        run_planner(config, dataset)
    else:
        _run_isolated_episode_batch(config, dataset)

    e_t = time.time() - t0
    print(f"Time elapsed since start of experiment: {e_t} seconds.")

def run_planner_wrapper(config, episode_indices, conn=None):
    """
    Wrapper to initialize dataset inside the child process.
    This avoids pickling large objects from the parent process.
    """
    # Re-load the full dataset metadata (lightweight)
    full_dataset = CollaborationDatasetV0(config.habitat.dataset)
    
    # Slice the episodes based on the passed indices
    subset_episodes = full_dataset.episodes[episode_indices]
    
    # Create the subset dataset for this process
    dataset = CollaborationDatasetV0(
        config=config.habitat.dataset, 
        episodes=subset_episodes
    )
    
    # Run the actual planner
    run_planner(config, dataset, conn)

def run_planner(config, dataset: CollaborationDatasetV0 = None, conn=None):
    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return

    # Setup interface with the simulator if the planner depends on it
    if config.env == "habitat":
        # Remove sensors if we are not saving video

        # TODO: have a flag for this, or some check
        keep_rgb = False
        if "use_rgb" in config.evaluation:
            keep_rgb = config.evaluation.use_rgb
        if not config.evaluation.save_video and not keep_rgb:
            remove_visual_sensors(config)

        # TODO: Can we move this inside the EnvironmentInterface?
        # We register the dynamic habitat sensors
        register_sensors(config)
        # We register custom actions
        register_actions(config)
        # We register custom measures
        register_measures(config)

        # Initialize the environment interface for the agent
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        try:
            env_interface.initialize_perception_and_world_graph()
        except Exception:
            print("Error initializing the environment")
            if config.evaluation.log_data:
                save_exception_message(config, env_interface)
    else:
        env_interface = None

    # Instantiate the agent planner
    eval_runner: EvaluationRunner = None
    if config.evaluation.type == "centralized":
        eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)
    elif config.evaluation.type == "decentralized":
        eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)
    else:
        cprint(
            "Invalid planner type. Please select between 'centralized' or 'decentralized'. Exiting",
            "red",
        )
        return

    # Print the planner
    cprint(f"Successfully constructed the '{config.evaluation.type}' planner!", "green")
    print(eval_runner)

    # Opt-in: attach a PerturbationInjector if the Hydra config requests one
    # via ``evaluation.runtime_perturbation`` (used by planner_demo_resilience).
    try:
        rp_cfg = OmegaConf.select(config, "evaluation.runtime_perturbation")
        if rp_cfg is not None:
            from habitat_llm.evaluation.perturbation import (
                PerturbationInjector,
                PerturbationSpec,
            )

            rp_dict = OmegaConf.to_container(rp_cfg, resolve=True) or {}
            spec = PerturbationSpec(
                kind=str(rp_dict.get("kind", "clean")),
                seed=int(rp_dict.get("seed", 0)),
                intensity=float(rp_dict.get("intensity", 1.0)),
                extras=dict(rp_dict.get("extras", {}) or {}),
            )
            eval_runner.perturbation_injector = PerturbationInjector(spec=spec)
            cprint(
                f"[resilience] Installed PerturbationInjector: {spec.tag()}",
                "cyan",
            )
    except Exception as _pi_exc:  # pragma: no cover
        print("Failed to install runtime perturbation injector:", _pi_exc)

    def bind_evolution_manager(manager) -> None:
        planners = eval_runner.planner
        if isinstance(planners, dict):
            for planner in planners.values():
                if hasattr(planner, "set_evolution_manager"):
                    planner.set_evolution_manager(manager)
                else:
                    planner.evolution_manager = manager
        else:
            if hasattr(planners, "set_evolution_manager"):
                planners.set_evolution_manager(manager)
            else:
                planners.evolution_manager = manager

    # Declare observability mode
    cprint(
        f"Partial observability is set to: '{config.world_model.partial_obs}'", "green"
    )

    # Print the agent list
    print("\nAgent List:")
    print(eval_runner.agent_list)

    # Print the agent description
    print("\nAgent Description:")
    print(eval_runner.agent_descriptions)

    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    # cprint(f"LLM model: {config.planner.llm.llm._target_}", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    cprint("---------------------------------------\n", "blue")

    os.makedirs(config.paths.results_dir, exist_ok=True)

    output_root = Path(eval_runner.output_dir)
    context_cfg = config.evaluation.get("context_evolve", {})
    evolve_llm_client = None
    if hasattr(eval_runner, "adca_analyzer") and getattr(eval_runner, "adca_analyzer", None):
        evolve_llm_client = eval_runner.adca_analyzer.llm_client
    if evolve_llm_client is None and env_interface is not None and hasattr(env_interface, "llm_client"):
        evolve_llm_client = env_interface.llm_client
    num_available_episodes = 0
    try:
        num_available_episodes = len(env_interface.env.episodes)
    except Exception:
        num_available_episodes = len(dataset.episodes) if dataset is not None else 0
    evolution_manager = EvolutionContextManager(
        output_root=output_root,
        enabled=bool(context_cfg.get("enabled", False)),
        batch_size=int(context_cfg.get("batch_size", num_available_episodes)),
        max_history=int(context_cfg.get("max_history", 3)),
        max_context_chars=int(context_cfg.get("max_context_chars", 2000)),
        llm_client=evolve_llm_client,
    )
    bind_evolution_manager(evolution_manager)

    # Run the planner
    if config.mode == "cli":
        instruction = "Go to the bed" if not config.instruction else config.instruction

        cprint(f'\nExecuting instruction: "{instruction}"', "blue")
        try:
            eval_runner.current_run_id = 0
            info = eval_runner.run_instruction(instruction)
        except Exception as e:
            print("An error occurred:", e)

    else:
        stats_episodes: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
        }
        run_aggregates: Dict[str, Dict[str, float]] = {}

        num_episodes = len(env_interface.env.episodes)
        analysis_roots = resolve_analysis_roots(config, output_root)
        
        # Initialize Dynamic Scheduler
        all_episodes = env_interface.env.env.env._env.episodes
        batch_size = int(context_cfg.get("batch_size", 15))
        scheduler = DynamicEpisodeScheduler(
            episodes=all_episodes,
            batch_size=batch_size,
            retry_ratio=float(context_cfg.get("retry_ratio", 0.5)),
            max_retries=int(context_cfg.get("max_retries", 3)),
        )

        for run_id in range(config.num_runs_per_episode):
            episode_failures: Dict[str, str] = {}
            
            # Check if evolution is enabled. If not, use standard sequential execution to avoid
            # simulator stability issues associated with manual episode injection.
            if not evolution_manager.enabled:
                for _ in range(num_episodes):
                    # Get episode id
                    episode_id = env_interface.env.env.env._env.current_episode.episode_id
                    instruction = env_interface.env.env.env._env.current_episode.instruction
                    print("\n\nEpisode", episode_id)

                    episode_id_str = str(episode_id)
                    episode_filename = f"episode_{episode_id}_{run_id}"

                    info: Dict[str, Any] = {}
                    episode_error: Optional[str] = None
                    task_success = False

                    try:
                        # Pass None/empty for log_subdirectory to maintain standard behavior
                        eval_runner.current_run_id = run_id
                        info = eval_runner.run_instruction( 
                            output_name=episode_filename
                        )
                        info["run_id"] = run_id
                        
                        task_success = bool(info.get("task_state_success", 0.0) == 1.0)
                        
                        info_episode = {
                            "run_id": run_id,
                            "episode_id": episode_id,
                            "instruction": instruction,
                        }
                        stats_keys = _collect_episode_stats_keys(info)
                        ignore_keys = (set(info.keys()) - stats_keys) | {k for k, v in info.items() if v is None}

                        stats_episode = extract_scalars_from_info(
                            info, ignore_keys=ignore_keys
                        )
                        stats_episodes[str(run_id)][episode_id] = stats_episode

                        eval_runner.print_primary_display_metrics(
                            f"Metrics For Run {run_id} Episode {episode_id}:",
                            stats_episodes[str(run_id)][episode_id],
                        )
                        
                        epi_metrics = _build_episode_csv_metrics(
                            stats_episodes[str(run_id)][episode_id], info
                        ) | info_episode
                        if config.evaluation.log_data:
                            save_success_message(config, env_interface, stats_episode)
                        write_to_csv(config.paths.epi_result_file_path, epi_metrics)
                    except Exception as e:
                        traceback.print_exc()
                        print("An error occurred while running the episode:", e)
                        print(f"Skipping evaluating episode: {episode_id}")
                        episode_error = str(e)
                        episode_failures[episode_id_str] = (
                            f"{type(e).__name__}: {e}"
                        )
                        if config.evaluation.log_data:
                            save_exception_message(config, env_interface)
                        if bool(config.get("fail_on_episode_error", False)):
                            raise RuntimeError(
                                f"Episode {episode_id_str} failed: "
                                f"{type(e).__name__}: {e}"
                            ) from e

                    try:
                        # Standard reset advances the internal iterator
                        env_interface.reset_environment()
                    except Exception as e:
                        traceback.print_exc()
                        print("An error occurred while resetting the env_interface:", e)
                        print("Skipping evaluating episode.")
                        if config.evaluation.log_data:
                            save_exception_message(config, env_interface)

                    eval_runner.reset()
                    clear_memory()
                    print(f"[Memory] Cleared cache after episode {episode_id}")
                    
            else:
                while scheduler.has_pending_work():
                    # 1. Get Next Batch
                    batch_episodes = scheduler.get_next_batch()
                    if not batch_episodes:
                        break
                    
                    batch_results: Dict[str, bool] = {}
                    batch_episode_ids: List[str] = []
                    
                    print(f"\n[Scheduler] Starting batch with {len(batch_episodes)} episodes. "
                          f"(Stats: {scheduler.get_stats()})")

                    # Update environment with batch episodes
                    # Note: We manually iterate through this batch
                    
                    for episode in batch_episodes:
                        # Override episode iterator
                        env_interface.env.env.env._env.episodes = [episode]
                        # This ensures next reset() picks this episode.
                        
                        # Get episode id
                        # Reset environment to load this episode
                        try:
                            env_interface.reset_environment()
                        except Exception as e:
                            print(f"Skipping episode {episode.episode_id} due to reset failure: {e}")
                            batch_results[str(episode.episode_id)] = False
                            continue

                        episode_id = env_interface.env.env.env._env.current_episode.episode_id
                        instruction = env_interface.env.env.env._env.current_episode.instruction
                        print("\n\nEpisode", episode_id)

                        episode_id_str = str(episode_id)
                        # Include batch index in filename to prevent overwrites during retries
                        batch_idx = evolution_manager.batch_counter if evolution_manager.enabled else 0
                        episode_filename = f"batch_{batch_idx}_episode_{episode_id}_{run_id}"

                        if evolution_manager.enabled:
                            evolution_context = evolution_manager.get_context(episode_id_str)
                            apply_evolution_context(
                                eval_runner, evolution_context, episode_id_str, run_id
                            )

                        info: Dict[str, Any] = {}
                        episode_error: Optional[str] = None
                        task_success = False
                        
                        try:
                            eval_runner.current_run_id = run_id
                            info = eval_runner.run_instruction(
                                output_name=episode_filename
                            )
                            info["run_id"] = run_id
                            
                            task_success = bool(info.get("task_state_success", 0.0) == 1.0)

                            info_episode = {
                                "run_id": run_id,
                                "episode_id": episode_id,
                                "instruction": instruction,
                            }
                            stats_keys = _collect_episode_stats_keys(info)
                            ignore_keys = (set(info.keys()) - stats_keys) | {k for k, v in info.items() if v is None}

                            stats_episode = extract_scalars_from_info(
                                info, ignore_keys=ignore_keys
                            )
                            stats_episodes[str(run_id)][episode_id] = stats_episode

                            eval_runner.print_primary_display_metrics(
                                f"Metrics For Run {run_id} Episode {episode_id}:",
                                stats_episodes[str(run_id)][episode_id],
                            )
                            # Log results onto a CSV
                            epi_metrics = _build_episode_csv_metrics(
                                stats_episodes[str(run_id)][episode_id], info
                            ) | info_episode
                            if config.evaluation.log_data:
                                save_success_message(config, env_interface, stats_episode)
                            write_to_csv(config.paths.epi_result_file_path, epi_metrics)
                        except Exception as e:
                            # print exception and trace
                            traceback.print_exc()
                            print("An error occurred while running the episode:", e)
                            print(f"Skipping evaluating episode: {episode_id}")
                            episode_error = str(e)
                            episode_failures[episode_id_str] = (
                                f"{type(e).__name__}: {e}"
                            )
                            if config.evaluation.log_data:
                                save_exception_message(config, env_interface)
                            if bool(config.get("fail_on_episode_error", False)):
                                raise RuntimeError(
                                    f"Episode {episode_id_str} failed: "
                                    f"{type(e).__name__}: {e}"
                                ) from e

                        if evolution_manager.enabled:
                            summary = build_episode_summary(
                                analysis_roots=analysis_roots,
                                episode_id=episode_id_str,
                                episode_filename=episode_filename,
                                instruction=instruction,
                                info=info,
                                run_id=run_id,
                            )
                            if episode_error:
                                summary["error"] = episode_error
                            evolution_manager.record_episode_summary(summary)
                            batch_episode_ids.append(episode_id_str)
                        
                        batch_results[episode_id_str] = task_success

                        # Reset evaluation runner
                        eval_runner.reset()

                        # Clear memory, prevent OOM
                        clear_memory()
                        print(f"[Memory] Cleared cache after episode {episode_id}")

                    # End of Batch Processing
                    if evolution_manager.enabled and batch_episode_ids:
                        evolution_manager.evolve_batch(batch_episode_ids)
                    
                    # Update Scheduler with results
                    scheduler.update_results(batch_results)

            # A run with no completed episode must fail with the original
            # episode context.  Writing an empty aggregate as ``[{}]`` would
            # otherwise raise TabularSchemaError and mask the real failure.
            completed_episode_metrics = stats_episodes[str(run_id)]
            if not completed_episode_metrics:
                failure_summary = "; ".join(
                    f"{episode_id}={reason}"
                    for episode_id, reason in sorted(episode_failures.items())
                )
                raise RuntimeError(
                    f"Run {run_id} produced no completed episode metrics"
                    + (f": {failure_summary}" if failure_summary else "")
                )

            # aggregate metrics across the current run.
            run_metrics = aggregate_measures(completed_episode_metrics)
            if not run_metrics:
                raise RuntimeError(
                    f"Run {run_id} completed episodes "
                    f"{sorted(str(key) for key in completed_episode_metrics)} "
                    "but produced no numeric metrics to aggregate"
                )
            run_aggregates[str(run_id)] = run_metrics
            eval_runner.print_primary_display_metrics(
                f"Metrics For Run {run_id}:",
                run_metrics,
            )

            # Write aggregated results across run
            write_to_csv(config.paths.run_result_file_path, run_metrics)


        # aggregate metrics across all runs.
        if conn is None:
            all_metrics = aggregate_measures(run_aggregates)
            if not all_metrics:
                raise RuntimeError(
                    "Completed runs produced no numeric experiment metrics"
                )
            eval_runner.print_primary_display_metrics(
                "Metrics Across All Runs:",
                all_metrics,
            )
            # Write aggregated results across experiment
            write_to_csv(config.paths.end_result_file_path, all_metrics)
        else:
            conn.send(stats_episodes)

    env_interface.env.close()
    del env_interface
    del eval_runner
    
    # Clear memory
    clear_memory()
    print("[Memory] Final cleanup completed")

    if conn is not None:
        # Potentially we may want to send something

        conn.close()


if __name__ == "__main__":
    cprint(
        "\nStart of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    # Run planner
    run_eval()

    cprint(
        "\nEnd of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )
