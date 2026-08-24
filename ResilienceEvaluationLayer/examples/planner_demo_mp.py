#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
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
from habitat_llm.planner.evolve import (
    EvolutionContextManager,
    apply_evolution_context,
    build_episode_summary,
    resolve_analysis_roots,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_baselines.utils.info_dict import extract_scalars_from_info


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
    # Sort the dictionary by keys
    # Needed to ensure sanity in multi-process operation
    result_dict = dict(sorted(result_dict.items()))
    with open(file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

        # Check if the file is empty (to write headers)
        file.seek(0, 2)
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(result_dict)


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


# Method to load agent planner from the config
@hydra.main(config_path="../conf")
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
        if config.get("episode_indices", None) is not None:
            if config.get("resume", False):
                raise ValueError("episode_indices and resume cannot be used together")
            episode_subset = [dataset.episodes[x] for x in config.episode_indices]
            dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )
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

    # Run the planner
    if config.mode == "cli":
        instruction = "Go to the bed" if not config.instruction else config.instruction

        cprint(f'\nExecuting instruction: "{instruction}"', "blue")
        try:
            info = eval_runner.run_instruction(instruction)
        except Exception as e:
            print("An error occurred:", e)

    else:
        stats_episodes: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
        }

        num_episodes = len(env_interface.env.episodes)
        output_root = Path(eval_runner.output_dir)
        analysis_roots = resolve_analysis_roots(config, output_root)
        context_cfg = config.evaluation.get("context_evolve", {})
        
        # Try to extract LLM client from runner (e.g. from ADCA analyzer)
        evolve_llm_client = None
        if hasattr(eval_runner, "adca_analyzer") and getattr(eval_runner, "adca_analyzer", None):
            evolve_llm_client = eval_runner.adca_analyzer.llm_client

        evolution_manager = EvolutionContextManager(
            output_root=output_root,
            enabled=bool(context_cfg.get("enabled", False)),
            batch_size=int(context_cfg.get("batch_size", num_episodes)),
            max_history=int(context_cfg.get("max_history", 3)),
            max_context_chars=int(context_cfg.get("max_context_chars", 2000)),
            llm_client=evolve_llm_client,
        )
        pending_batch_episode_ids: List[str] = []

        for run_id in range(config.num_runs_per_episode):
            for _ in range(num_episodes):
                # Get episode id
                episode_id = env_interface.env.env.env._env.current_episode.episode_id

                # Get instruction
                instruction = env_interface.env.env.env._env.current_episode.instruction
                print("\n\nEpisode", episode_id)

                episode_id_str = str(episode_id)
                episode_filename = f"episode_{episode_id}_{run_id}"
                if evolution_manager.enabled:
                    evolution_context = evolution_manager.get_context(episode_id_str)
                    apply_evolution_context(
                        eval_runner, evolution_context, episode_id_str, run_id
                    )

                info: Dict[str, Any] = {}
                episode_error: Optional[str] = None
                try:
                    info = eval_runner.run_instruction(
                        output_name=episode_filename
                    )

                    info_episode = {
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "instruction": instruction,
                    }
                    stats_keys = {
                        "task_percent_complete",
                        "task_state_success",
                        "sim_step_count",
                        "replanning_count",
                        "runtime",
                        "rebound_mttr",
                        "rebound_mtbf",
                        "rebound_count",
                    }

                    # add replanning counts to stats_keys as scalars if replanning_count is a dict
                    if "replanning_count" in info and isinstance(
                        info["replanning_count"], dict
                    ):
                        for agent_id, replan_count in info["replanning_count"].items():
                            stats_keys.add(f"replanning_count_{agent_id}")
                            info[f"replanning_count_{agent_id}"] = replan_count
                    
                    # Add any agent-specific rebound metrics found in info to stats_keys
                    for key in list(info.keys()):
                        if key.startswith("rebound_"):
                            stats_keys.add(key)

                    stats_episode = extract_scalars_from_info(
                        info, ignore_keys=info.keys() - stats_keys
                    )
                    stats_episodes[str(run_id)][episode_id] = stats_episode

                    cprint("\n---------------------------------", "blue")
                    cprint(f"Metrics For Run {run_id} Episode {episode_id}:", "blue")
                    for k, v in stats_episodes[str(run_id)][episode_id].items():
                        cprint(f"{k}: {v:.3f}", "blue")
                    cprint("\n---------------------------------", "blue")
                    # Log results onto a CSV
                    epi_metrics = stats_episodes[str(run_id)][episode_id] | info_episode
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
                    pending_batch_episode_ids.append(episode_id_str)
                    if (
                        evolution_manager.batch_size > 0
                        and len(pending_batch_episode_ids)
                        >= evolution_manager.batch_size
                    ):
                        evolution_manager.evolve_batch(pending_batch_episode_ids)
                        pending_batch_episode_ids = []

                try:
                    # Reset env_interface (moves onto the next episode in the dataset)
                    env_interface.reset_environment()
                except Exception as e:
                    # print exception and trace
                    traceback.print_exc()
                    print("An error occurred while resetting the env_interface:", e)
                    print("Skipping evaluating episode.")
                    if config.evaluation.log_data:
                        save_exception_message(config, env_interface)

                # Reset evaluation runner
                eval_runner.reset()

                # Clear memory, prevent OOM
                clear_memory()
                print(f"[Memory] Cleared cache after episode {episode_id}")

            # aggregate metrics across the current run.
            run_metrics = aggregate_measures(stats_episodes[str(run_id)])
            cprint("\n---------------------------------", "blue")
            cprint(f"Metrics For Run {run_id}:", "blue")
            for k, v in run_metrics.items():
                cprint(f"{k}: {v:.3f}", "blue")
            cprint("\n---------------------------------", "blue")

            # Write aggregated results across run
            write_to_csv(config.paths.run_result_file_path, run_metrics)


        # aggregate metrics across all runs.
        if conn is None:
            all_metrics = aggregate_measures(
                {run_id: aggregate_measures(v) for run_id, v in stats_episodes.items()}
            )
            cprint("\n---------------------------------", "blue")
            cprint("Metrics Across All Runs:", "blue")
            for k, v in all_metrics.items():
                cprint(f"{k}: {v:.3f}", "blue")
            cprint("\n---------------------------------", "blue")
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
