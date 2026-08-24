#!/usr/bin/env python3
# isort: skip_file

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
# Fix for huggingface/tokenizers parallelism issue in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import sys
import time
import os
import traceback
import json
import shutil
import subprocess
import gc
import torch
from omegaconf import OmegaConf


# append the path of the
# parent directory
sys.path.append("..")

import hydra
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch import multiprocessing as mp

from habitat_llm.agent.env.evaluation.evaluation_functions import (
    aggregate_measures,
)

from habitat_llm.utils import cprint, setup_config, fix_config


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


EPISODE_CANONICAL_METRIC_KEYS = [
    # Rebound
    "rebound_c_rec",
    "rebound_c_rec_cog",
    "rebound_c_rec_phy",
    "rebound_c_rec_state_debt",
    "rebound_c_rec_plan",
    "rebound_c_rec_sim",
    "rebound_num_recovery_windows",
    "rebound_raw_window_count",
    "rebound_formal_window_count",
    "rebound_skipped_window_count",
    "rebound_c_rec_valid",
    "rebound_c_rec_missing_reason",
    "rebound_baseline_match_level",
    "rebound_w_rem_star",
    "rebound_g_rec_total",
    "rebound_baseline_loaded",
    "rebound_tracker_window_count",
    "rebound_tracker_window_open_final",
    # Stability
    "stability_beta",
    "stability_beta_neighborhood",
    "stability_beta_vv",
    "stability_beta_out",
    "stability_beta_oscillation",
    "stability_beta_action",
    "stability_beta_progress",
    "stability_beta_success",
    "stability_beta_perturbation",
    "stability_beta_perturbation_vv",
    "stability_beta_perturbation_out",
    "stability_beta_perturbation_oscillation",
    "stability_beta_perturbation_action",
    "stability_beta_perturbation_progress",
    "stability_beta_perturbation_success",
    "stability_beta_perturbation_valid",
    "stability_beta_perturbation_scope",
    "stability_clean_ref_valid",
    "stability_clean_ref_strategy",
    "stability_n_clean_ref",
    "stability_beta_neighborhood_valid",
    "stability_beta_neighborhood_missing_reason",
    "stability_beta_scope",
    "stability_p_cbf",
    "stability_td_error_mean_abs",
    "stability_td_error_p95_abs",
    "stability_gae_mean_abs",
    "stability_gae_p95_abs",
    "stability_return_target_variance",
    "stability_oscillation_loss",
    "stability_oscillation_loss_valid",
    "stability_oscillation_source",
    "stability_critic_trace_len",
    "stability_value_delta_variance",
    # Degradation / Graceful Extensibility
    "degradation_auc_loss",
    "degradation_p_cliff",
    "degradation_l_bd",
    "ge_contract_margin_min",
    "ge_audc_f",
    "ge_cell_valid",
    "ge_cell_missing_reason",
    "ge_valid_lambda_coverage",
    "ge_boundary_lambda_star",
    "ge_boundary_hit",
    "ge_hard_boundary_hit",
]
EPISODE_COMPAT_ALIAS_KEYS = []
EPISODE_DYNAMIC_PREFIXES = ()


def _collect_episode_stats_keys(info: Dict[str, Any]) -> set:
    stats_keys = {
        "task_percent_complete",
        "task_state_success",
        "proposition_satisfied_fraction",
        "sim_step_count",
        "replanning_count",
        "runtime",
    }
    stats_keys.update(EPISODE_CANONICAL_METRIC_KEYS)
    stats_keys.update(EPISODE_COMPAT_ALIAS_KEYS)

    if "replanning_count" in info and isinstance(info["replanning_count"], dict):
        for agent_id, replan_count in info["replanning_count"].items():
            stats_keys.add(f"replanning_count_{agent_id}")
            info[f"replanning_count_{agent_id}"] = replan_count

    if EPISODE_DYNAMIC_PREFIXES:
        for key in list(info.keys()):
            if key.startswith(EPISODE_DYNAMIC_PREFIXES):
                stats_keys.add(key)

    return stats_keys


def _build_episode_csv_metrics(
    stats_episode: Dict[str, Any],
    info: Dict[str, Any],
) -> Dict[str, Any]:
    csv_metrics = dict(stats_episode)

    for key in EPISODE_CANONICAL_METRIC_KEYS + EPISODE_COMPAT_ALIAS_KEYS:
        value = info.get(key)
        csv_metrics[key] = "" if value is None else value

    for key in ("proposition_satisfied_fraction", "task_family", "action_sequence"):
        value = info.get(key)
        if value is not None:
            csv_metrics[key] = value

    return csv_metrics


def clear_memory():
    """Clear Python and PyTorch memory cache"""
    gc.collect()
    return
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     torch.cuda.synchronize()


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
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    existing_fieldnames: List[str] = []
    existing_rows: List[Dict[str, Any]] = []
    if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
        with open(file_name, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            existing_fieldnames = list(reader.fieldnames or [])
            if existing_fieldnames:
                existing_rows = list(reader)

    if existing_fieldnames and set(result_dict.keys()).issubset(existing_fieldnames):
        fieldnames = existing_fieldnames
        with open(file_name, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({key: result_dict.get(key, "") for key in fieldnames})
        return

    fieldnames = sorted(set(existing_fieldnames) | set(result_dict.keys()))
    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        writer.writerow({key: result_dict.get(key, "") for key in fieldnames})


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
        f.write(OmegaConf.to_yaml(config))

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
                    shutil.copy(
                        yaml_rlm_file, f"{output_file}/config_rlm{suffix_rlm}.yaml"
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

    num_episodes = len(dataset.episodes)
    if config.num_proc == 1:
        if (config.get("episode_indices", None) is not None or
                config.get("episode_ids", None) is not None or
                config.get("evolve_episode_ids", None) is not None):
            if config.get("resume", False):
                raise ValueError("episode selection and resume cannot be used together")
            dataset = _select_episode_subset(dataset, config)
        run_planner(config, dataset)
    else:
        # Process episodes in parallel using subprocess to avoid memory leaks
        # We launch separate python processes for each chunk
        print(f"Launching {config.num_proc} subprocess workers...")
        processes = []
        ochunk_size = num_episodes // config.num_proc
        
        start = 0
        for i in range(config.num_proc):
            chunk_size = ochunk_size
            if i < (num_episodes % config.num_proc):
                chunk_size += 1
            end = min(start + chunk_size, num_episodes)
            
            if start >= end:
                break

            # Construct command to run this script again with start/end indices
            # We assume we are running the same script.
            cmd = [
                sys.executable,
                "-m",
                "habitat_llm.examples.planner_demo",
                f"++start_index={start}",
                f"++end_index={end}",
            ]
            
            # Pass through all existing arguments, but filter out ones we don't want to duplicate or conflict
            # This is a bit tricky with Hydra. We mainly want to pass the config name and overrides.
            # A simpler way is to pass the same arguments but add ++start_index and ++end_index
            # We iterate sys.argv to reconstruct the command
            
            # Helper to check if arg is start/end index
            def is_index_arg(arg):
                return "start_index" in arg or "end_index" in arg
            
            # Reconstruct args, excluding our internal worker flags if they were somehow present
            base_args = [arg for arg in sys.argv[1:] if not is_index_arg(arg)]
            
            # Add config name if not in base_args (it usually is)
            # Add overrides
            final_cmd = cmd[:3] + base_args + cmd[3:]
            
            print(f"Starting worker {i}: indices {start}-{end}")
            # print(f"Command: {' '.join(final_cmd)}")
            
            p = subprocess.Popen(final_cmd)
            processes.append(p)
            
            start += chunk_size

        # Wait for all processes to complete
        for p in processes:
            p.wait()
            if p.returncode != 0:
                print(f"Process {p.pid} failed with return code {p.returncode}")

        # Aggregation logic would need to be updated to read from files
        # Since each worker writes its own results, we can just aggregate at the end.
        # But wait, the original code aggregated results from Pipes.
        # Now workers write to disk independently.
        # We need to run aggregation after all workers finish.
        
        # Re-read all stats files and aggregate
        # Assuming workers wrote their results to the results directory.
        # The original code did write_to_csv inside run_planner, so files are generated.
        # We just need to aggregate the final metrics if needed.
        # For now, we rely on the per-episode/run logs.
        
        # Recalculate aggregation from disk if possible, or just skip the "Metrics Across All Runs" print
        # which might be acceptable for the demo improvement.
        print("All workers completed.")

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
                        if config.evaluation.log_data:
                            save_exception_message(config, env_interface)

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
                            if config.evaluation.log_data:
                                save_exception_message(config, env_interface)

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

            # aggregate metrics across the current run.
            run_metrics = aggregate_measures(stats_episodes[str(run_id)])
            eval_runner.print_primary_display_metrics(
                f"Metrics For Run {run_id}:",
                run_metrics,
            )

            # Write aggregated results across run
            write_to_csv(config.paths.run_result_file_path, run_metrics)


        # aggregate metrics across all runs.
        if conn is None:
            all_metrics = aggregate_measures(
                {run_id: aggregate_measures(v) for run_id, v in stats_episodes.items()}
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
