#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from habitat_llm.evaluation.reporting import RUNTIME_METRIC_KEYS
from habitat_llm.evaluation.evolve.context_evolve import (
    EvolutionContextManager,
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
        "Evolve demo requires explicit episodes. "
        "Provide ++evolve_episode_indices=[...] or ++evolve_episode_ids=[...]."
    )


def _reset_evolution_context(manager: EvolutionContextManager) -> None:
    manager.episode_history = {}
    manager.context_by_episode = {}
    manager.general_context = None
    manager.batch_counter = 0
    try:
        if manager.memory_path.exists():
            manager.memory_path.unlink()
    except Exception:
        pass


@hydra.main(config_path="../conf")
def run_eval(config):
    fix_config(config)
    seed = 47668090
    t0 = time.time()
    config = setup_config(config, seed)
    dataset = CollaborationDatasetV0(config.habitat.dataset)

    write_config(config)

    if config.num_proc != 1:
        cprint("planner_demo_evolve expects num_proc=1.", "yellow")

    dataset = _select_episode_subset(dataset, config)
    run_planner_evolve(config, dataset)

    e_t = time.time() - t0
    print(f"Time elapsed since start of experiment: {e_t} seconds.")


def run_planner_evolve(config, dataset: CollaborationDatasetV0):
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

    cprint(
        f"Partial observability is set to: '{config.world_model.partial_obs}'", "green"
    )
    print("\nAgent List:")
    print(eval_runner.agent_list)
    print("\nAgent Description:")
    print(eval_runner.agent_descriptions)

    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
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

    if demo_cfg.get("reset_context", False) and evolution_manager.enabled:
        _reset_evolution_context(evolution_manager)

    print_context = bool(demo_cfg.get("print_context", True))
    stats_episodes: Dict[str, Dict] = {str(i): {} for i in range(num_epochs)}

    for epoch_id in range(num_epochs):
        evaluated_evolve_step = int(
            getattr(evolution_manager, "batch_counter", epoch_id)
        )
        cprint(f"\n==== Evolve Epoch {epoch_id + 1}/{num_epochs} ====", "blue")
        if evolution_manager.enabled and print_context and evolution_manager.general_context:
            cprint("\n[Current Batch Guidance]\n" + evolution_manager.general_context, "blue")

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
                    output_name=episode_filename
                )

                info_episode = {
                    "run_id": epoch_id,
                    "episode_id": episode_id,
                    "instruction": instruction,
                    "evolve_step": evaluated_evolve_step,
                    "ge_evolve_step_e": evaluated_evolve_step,
                }
                stats_keys = {
                    "task_percent_complete",
                    "task_state_success",
                    "proposition_satisfied_fraction",
                    "sim_step_count",
                    "replanning_count",
                    "runtime",
                }
                stats_keys.update(RUNTIME_METRIC_KEYS)

                if "replanning_count" in info and isinstance(
                    info["replanning_count"], dict
                ):
                    for agent_id, replan_count in info["replanning_count"].items():
                        stats_keys.add(f"replanning_count_{agent_id}")
                        info[f"replanning_count_{agent_id}"] = replan_count

                stats_episode = extract_scalars_from_info(
                    info, ignore_keys=info.keys() - stats_keys
                )
                stats_episodes[str(epoch_id)][episode_id] = stats_episode

                cprint("\n---------------------------------", "blue")
                cprint(f"Metrics For Epoch {epoch_id} Episode {episode_id}:", "blue")
                for k, v in stats_episodes[str(epoch_id)][episode_id].items():
                    cprint(f"{k}: {v:.3f}", "blue")
                cprint("\n---------------------------------", "blue")
                epi_metrics = stats_episodes[str(epoch_id)][episode_id] | info_episode
                for key in ("task_family", "action_sequence"):
                    value = info.get(key)
                    if value is not None:
                        epi_metrics[key] = value
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

        if evolution_manager.enabled and epoch_episode_ids:
            unique_episode_ids = list(dict.fromkeys(epoch_episode_ids))
            evolution_manager.evolve_batch(unique_episode_ids)
            if print_context and evolution_manager.general_context:
                cprint("\n[Updated Batch Guidance]\n" + evolution_manager.general_context, "blue")
        next_evolve_step = int(
            getattr(evolution_manager, "batch_counter", evaluated_evolve_step)
        )

        run_metrics = aggregate_measures(stats_episodes[str(epoch_id)])
        cprint("\n---------------------------------", "blue")
        cprint(f"Metrics For Epoch {epoch_id}:", "blue")
        for k, v in run_metrics.items():
            cprint(f"{k}: {v:.3f}", "blue")
        cprint("\n---------------------------------", "blue")
        write_to_csv(config.paths.run_result_file_path, run_metrics)

        # Graceful Extensibility hook (§3 of resilience design doc)
        try:
            version_history_path = Path(config.paths.results_dir) / "version_history.jsonl"
            version_history_path.parent.mkdir(parents=True, exist_ok=True)
            version_entry = {
                "epoch_id": int(epoch_id),
                "evolve_step_e": int(evaluated_evolve_step),
                "evaluated_evolve_step": int(evaluated_evolve_step),
                "next_evolve_step": int(next_evolve_step),
                "context_enabled": bool(evolution_manager.enabled),
                "batch_size": int(context_cfg.get("batch_size", 0) or 0),
                "max_history": int(context_cfg.get("max_history", 0) or 0),
                "episode_count": len(stats_episodes.get(str(epoch_id), {})),
                "run_metrics": {str(k): float(v) for k, v in run_metrics.items()},
                "episode_ids": list(epoch_episode_ids),
            }
            with open(version_history_path, "a", encoding="utf-8") as _vh:
                _vh.write(json.dumps(version_entry) + "\n")
            cprint(
                f"[evolve] Wrote version snapshot evaluated_e={version_entry['evaluated_evolve_step']} "
                f"next_e={version_entry['next_evolve_step']} "
                f"-> {version_history_path}",
                "cyan",
            )
        except Exception as _vh_exc:  # pragma: no cover
            print("Failed to write version_history.jsonl:", _vh_exc)

    all_metrics = aggregate_measures(
        {epoch_id: aggregate_measures(v) for epoch_id, v in stats_episodes.items()}
    )
    cprint("\n---------------------------------", "blue")
    cprint("Metrics Across All Epochs:", "blue")
    for k, v in all_metrics.items():
        cprint(f"{k}: {v:.3f}", "blue")
    cprint("\n---------------------------------", "blue")
    write_to_csv(config.paths.end_result_file_path, all_metrics)

    env_interface.env.close()
    del env_interface
    del eval_runner
    clear_memory()
    print("[Memory] Final cleanup completed")


if __name__ == "__main__":
    cprint(
        "\nStart of the example program to demonstrate evolve planner demo.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    run_eval()

    cprint(
        "\nEnd of the example program to demonstrate evolve planner demo.",
        "blue",
    )
