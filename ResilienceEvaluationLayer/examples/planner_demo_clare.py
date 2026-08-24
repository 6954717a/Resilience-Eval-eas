#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
CLARE (Continual Learning with Autonomous Routing and Expansion) Demo Script

This script demonstrates CLARE's continual learning capabilities:
- Task routing based on semantic similarity
- OOD (Out-of-Distribution) detection for new tasks
- Context-based adapter creation and evolution
- Batch-based offline training

Usage:
    python -m habitat_llm.examples.planner_demo_clare \
        --config-name baselines/clare_config.yaml \
        habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
        num_proc=1 \
        ++evolve_episode_indices=[0,1,2]
"""

import os
# Fix for huggingface/tokenizers parallelism issue in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import OmegaConf

# append the path of the parent directory
sys.path.append("..")

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
from habitat_llm.planner.evolve import (
    EvolutionContextManager,
    apply_evolution_context,
    build_episode_summary,
    resolve_analysis_roots,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_baselines.utils.info_dict import extract_scalars_from_info


def clear_memory():
    """Clear Python and PyTorch memory cache."""
    gc.collect()
    return


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


def write_to_csv(file_name, result_dict):
    result_dict = dict(sorted(result_dict.items()))
    with open(file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

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


def write_config(config):
    dataset_file = config.habitat.dataset.data_path.split("/")[-1]
    output_file = os.path.join(config.paths.results_dir, dataset_file)
    os.makedirs(output_file, exist_ok=True)
    with open(f"{output_file}/config.yaml", "w+") as f:
        f.write(redacted_config_yaml(config))

    planner_configs = []
    suffixes = []
    if "planner" in config.evaluation:
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


def _select_episode_subset(dataset: CollaborationDatasetV0, config) -> CollaborationDatasetV0:
    evolve_episode_indices = config.get("evolve_episode_indices", None)
    evolve_episode_ids = config.get("evolve_episode_ids", None)
    if evolve_episode_indices is None:
        evolve_episode_indices = config.get("episode_indices", None)
    if evolve_episode_ids is None:
        evolve_episode_ids = config.get("episode_ids", None)

    if evolve_episode_indices is not None and evolve_episode_ids is not None:
        raise ValueError("Specify only one of evolve_episode_indices or evolve_episode_ids")

    if evolve_episode_indices is not None:
        episode_indices = list(evolve_episode_indices)
        if len(set(episode_indices)) != len(episode_indices):
            raise ValueError("Duplicate episode indices provided.")
        if any(ix < 0 or ix >= len(dataset.episodes) for ix in episode_indices):
            raise ValueError("Episode index out of range.")
        subset = [dataset.episodes[ix] for ix in episode_indices]
        return CollaborationDatasetV0(config=config.habitat.dataset, episodes=subset)

    if evolve_episode_ids is not None:
        episode_ids = [str(eid) for eid in evolve_episode_ids]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("Duplicate episode ids provided.")
        subset = [ep for ep in dataset.episodes if str(ep.episode_id) in episode_ids]
        missing = sorted(set(episode_ids) - {str(ep.episode_id) for ep in subset})
        if missing:
            raise ValueError(f"Could not find episode ids: {missing}")
        return CollaborationDatasetV0(config=config.habitat.dataset, episodes=subset)

    raise ValueError(
        "CLARE demo requires explicit episodes. "
        "Provide ++evolve_episode_indices=[...] or ++evolve_episode_ids=[...]."
    )


def _reset_clare_state(planner) -> None:
    """Reset CLARE state for fresh run."""
    if hasattr(planner, 'clare_components') and planner.clare_components:
        components = planner.clare_components
        if 'expansion_controller' in components:
            components['expansion_controller'].reset()
        if 'metrics_tracker' in components:
            # Reset metrics tracking for fresh evaluation
            components['metrics_tracker'].routing_log.clear()
    if hasattr(planner, 'clare_pending_episodes'):
        planner.clare_pending_episodes = []


@hydra.main(config_path="../conf")
def run_eval(config):
    fix_config(config)
    seed = 47668090
    t0 = time.time()
    config = setup_config(config, seed)
    dataset = CollaborationDatasetV0(config.habitat.dataset)

    write_config(config)

    if config.num_proc != 1:
        cprint("planner_demo_clare expects num_proc=1.", "yellow")

    dataset = _select_episode_subset(dataset, config)
    run_planner_clare(config, dataset)

    e_t = time.time() - t0
    print(f"Time elapsed since start of experiment: {e_t} seconds.")


def run_planner_clare(config, dataset: CollaborationDatasetV0):
    """Main CLARE evaluation loop."""
    if config is None:
        cprint("Failed to setup config. Exiting", "red")
        return

    if config.env == "habitat":
        keep_rgb = False
        if "use_rgb" in config.evaluation:
            keep_rgb = config.evaluation.use_rgb
        if not config.evaluation.save_video and not keep_rgb:
            remove_visual_sensors(config)

        register_sensors(config)
        register_actions(config)
        register_measures(config)

        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        try:
            env_interface.initialize_perception_and_world_graph()
        except Exception:
            print("Error initializing the environment")
            if config.evaluation.log_data:
                save_exception_message(config, env_interface)
    else:
        env_interface = None

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

    cprint(f"Successfully constructed the '{config.evaluation.type}' planner!", "green")
    print(eval_runner)

    # Check CLARE status
    clare_enabled = False
    if hasattr(eval_runner.planner, 'clare_enabled'):
        clare_enabled = eval_runner.planner.clare_enabled
    
    cprint(f"\n[CLARE] Continual Learning: {'ENABLED' if clare_enabled else 'DISABLED'}", 
           "green" if clare_enabled else "yellow")
    
    if clare_enabled and hasattr(eval_runner.planner, 'clare_components'):
        components = eval_runner.planner.clare_components
        adapter_count = len(components['adapter_manager'].adapters)
        cprint(f"[CLARE] Loaded {adapter_count} existing adapters", "blue")

    cprint(f"\nPartial observability is set to: '{config.world_model.partial_obs}'", "green")
    print("\nAgent List:")
    print(eval_runner.agent_list)
    print("\nAgent Description:")
    print(eval_runner.agent_descriptions)

    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    cprint(f"CLARE Continual Learning: {clare_enabled}", "blue")
    cprint("---------------------------------------\n", "blue")

    os.makedirs(config.paths.results_dir, exist_ok=True)

    if config.mode == "cli":
        instruction = "Go to the bed" if not config.instruction else config.instruction
        cprint(f'\nExecuting instruction: "{instruction}"', "blue")
        try:
            eval_runner.run_instruction(instruction)
        except Exception as exc:
            print("An error occurred:", exc)
        return

    # Get epoch and context evolution settings
    demo_cfg = config.get("evolve_demo", {})
    if "epochs" in demo_cfg:
        num_epochs = int(demo_cfg.get("epochs"))
    else:
        num_epochs = int(config.get("evolve_epochs", config.get("num_runs_per_episode", 1)))
    if num_epochs < 1:
        raise ValueError("evolve_demo.epochs must be >= 1")

    context_cfg = config.evaluation.get("context_evolve", {})
    num_episodes = len(env_interface.env.episodes)
    output_root = Path(eval_runner.output_dir)
    analysis_roots = resolve_analysis_roots(config, output_root)
    
    evolution_manager = EvolutionContextManager(
        output_root=output_root,
        enabled=bool(context_cfg.get("enabled", False)),
        batch_size=int(context_cfg.get("batch_size", num_episodes)),
        max_history=int(context_cfg.get("max_history", 3)),
        max_context_chars=int(context_cfg.get("max_context_chars", 2000)),
    )

    print_context = bool(demo_cfg.get("print_context", True))
    stats_episodes: Dict[str, Dict] = {str(i): {} for i in range(num_epochs)}

    for epoch_id in range(num_epochs):
        cprint(f"\n==== CLARE Epoch {epoch_id + 1}/{num_epochs} ====", "blue")
        
        if evolution_manager.enabled and print_context and evolution_manager.general_context:
            cprint("\n[Current Batch Guidance]\n" + evolution_manager.general_context, "blue")
        
        # Show CLARE status at epoch start
        if clare_enabled:
            components = eval_runner.planner.clare_components
            adapter_count = len(components['adapter_manager'].adapters)
            routing_stats = components['router'].get_routing_stats()
            cprint(f"[CLARE] Adapters: {adapter_count}, Routings: {routing_stats.get('total_routings', 0)}", "cyan")

        epoch_episode_ids: List[str] = []

        for _ in range(num_episodes):
            episode_id = env_interface.env.env.env._env.current_episode.episode_id
            instruction = env_interface.env.env.env._env.current_episode.instruction
            print("\n\nEpisode", episode_id)

            episode_id_str = str(episode_id)
            episode_filename = f"episode_{episode_id}_{epoch_id}"

            if evolution_manager.enabled:
                evolution_context = evolution_manager.get_context(episode_id_str)
                apply_evolution_context(
                    eval_runner, evolution_context, episode_id_str, epoch_id
                )

            info: Dict[str, Any] = {}
            episode_error: Optional[str] = None
            try:
                info = eval_runner.run_instruction(
                    output_name=episode_filename,
                    log_subdirectory=f"batch_{epoch_id}"
                )

                info_episode = {
                    "run_id": epoch_id,
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
                    # CLARE-specific keys
                    "resilience_clare_bwt",
                    "resilience_clare_fwt",
                    "resilience_clare_routing_accuracy",
                    "resilience_clare_adapter_count",
                    "resilience_clare_ood_rate",
                }

                if "replanning_count" in info and isinstance(
                    info["replanning_count"], dict
                ):
                    for agent_id, replan_count in info["replanning_count"].items():
                        stats_keys.add(f"replanning_count_{agent_id}")
                        info[f"replanning_count_{agent_id}"] = replan_count

                for key in list(info.keys()):
                    if key.startswith("rebound_") or key.startswith("resilience_clare_"):
                        stats_keys.add(key)

                stats_episode = extract_scalars_from_info(
                    info, ignore_keys=info.keys() - stats_keys
                )
                stats_episodes[str(epoch_id)][episode_id] = stats_episode

                cprint("\n---------------------------------", "blue")
                cprint(f"Metrics For Epoch {epoch_id} Episode {episode_id}:", "blue")
                for k, v in stats_episodes[str(epoch_id)][episode_id].items():
                    cprint(f"{k}: {v:.3f}", "blue")
                cprint("---------------------------------\n", "blue")
                epi_metrics = stats_episodes[str(epoch_id)][episode_id] | info_episode
                if config.evaluation.log_data:
                    save_success_message(config, env_interface, stats_episode)
                write_to_csv(config.paths.epi_result_file_path, epi_metrics)
            except Exception as exc:
                traceback.print_exc()
                print("An error occurred while running the episode:", exc)
                print(f"Skipping evaluating episode: {episode_id}")
                episode_error = str(exc)
                if config.evaluation.log_data:
                    save_exception_message(config, env_interface)

            if evolution_manager.enabled:
                summary = build_episode_summary(
                    analysis_roots=analysis_roots,
                    episode_id=episode_id_str,
                    episode_filename=episode_filename,
                    instruction=instruction,
                    info=info,
                    run_id=epoch_id,
                )
                if episode_error:
                    summary["error"] = episode_error
                evolution_manager.record_episode_summary(summary)
                epoch_episode_ids.append(episode_id_str)

            try:
                env_interface.reset_environment()
            except Exception as exc:
                traceback.print_exc()
                print("An error occurred while resetting the env_interface:", exc)
                print("Skipping evaluating episode.")
                if config.evaluation.log_data:
                    save_exception_message(config, env_interface)

            eval_runner.reset()
            clear_memory()
            print(f"[Memory] Cleared cache after episode {episode_id}")

        # === CLARE Batch Evolution ===
        if clare_enabled and epoch_episode_ids:
            cprint("\n[CLARE] Running batch evolution...", "cyan")
            try:
                from habitat_llm.planner.clare import clare_evolve_batch
                evolution_results = clare_evolve_batch(
                    eval_runner.planner, 
                    epoch_episode_ids
                )
                for result in evolution_results:
                    task_id = result.get('task_id', 'unknown')
                    is_new = result.get('is_new_adapter', False)
                    status = "NEW" if is_new else "UPDATED"
                    cprint(f"[CLARE] [{status}] Adapter: {task_id}", "green")
                
                # Log CLARE metrics
                from habitat_llm.planner.clare import get_clare_metrics
                clare_metrics = get_clare_metrics(eval_runner.planner)
                cprint(f"[CLARE] BWT: {clare_metrics.get('clare_bwt', 0):.3f}, "
                       f"FWT: {clare_metrics.get('clare_fwt', 0):.3f}, "
                       f"Adapters: {clare_metrics.get('clare_adapter_count', 0)}", "cyan")
            except Exception as e:
                cprint(f"[CLARE] Evolution error: {e}", "red")

        # Context Evolution (Evolve system)
        if evolution_manager.enabled and epoch_episode_ids:
            unique_episode_ids = list(dict.fromkeys(epoch_episode_ids))
            evolution_manager.evolve_batch(unique_episode_ids)
            if print_context and evolution_manager.general_context:
                cprint("\n[Updated Batch Guidance]\n" + evolution_manager.general_context, "blue")

        run_metrics = aggregate_measures(stats_episodes[str(epoch_id)])
        cprint("\n---------------------------------", "blue")
        cprint(f"Metrics For Epoch {epoch_id}:", "blue")
        for k, v in run_metrics.items():
            cprint(f"{k}: {v:.3f}", "blue")
        cprint("---------------------------------\n", "blue")
        write_to_csv(config.paths.run_result_file_path, run_metrics)

    # === Final CLARE Summary ===
    if clare_enabled:
        cprint("\n========================================", "green")
        cprint("CLARE Continual Learning Summary", "green")
        cprint("========================================", "green")
        try:
            from habitat_llm.planner.clare import get_clare_metrics
            final_metrics = get_clare_metrics(eval_runner.planner)
            cprint(f"Final Adapter Count: {final_metrics.get('clare_adapter_count', 0)}", "green")
            cprint(f"Backward Transfer (BWT): {final_metrics.get('clare_bwt', 0):.4f}", "green")
            cprint(f"Forward Transfer (FWT): {final_metrics.get('clare_fwt', 0):.4f}", "green")
            cprint(f"Routing Accuracy: {final_metrics.get('clare_routing_accuracy', 0):.4f}", "green")
            cprint(f"OOD Rate: {final_metrics.get('clare_ood_rate', 0):.4f}", "green")
            cprint(f"Total Routings: {final_metrics.get('total_routings', 0)}", "green")
            cprint(f"Total Expansions: {final_metrics.get('total_expansions', 0)}", "green")
        except Exception as e:
            cprint(f"Could not get final CLARE metrics: {e}", "yellow")

    all_metrics = aggregate_measures(
        {epoch_id: aggregate_measures(v) for epoch_id, v in stats_episodes.items()}
    )
    cprint("\n---------------------------------", "blue")
    cprint("Metrics Across All Epochs:", "blue")
    for k, v in all_metrics.items():
        cprint(f"{k}: {v:.3f}", "blue")
    cprint("---------------------------------\n", "blue")
    write_to_csv(config.paths.end_result_file_path, all_metrics)

    env_interface.env.close()
    del env_interface
    del eval_runner
    clear_memory()
    print("[Memory] Final cleanup completed")


if __name__ == "__main__":
    cprint(
        "\nStart of the example program to demonstrate CLARE planner demo.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    run_eval()

    cprint(
        "\nEnd of the example program to demonstrate CLARE planner demo.",
        "blue",
    )
