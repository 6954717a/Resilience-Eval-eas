#!/usr/bin/env python3
# isort: skip_file
"""Subprocess worker for one resilience rollout config."""
from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from omegaconf import OmegaConf

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.examples import planner_demo_mp_new as base_runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = OmegaConf.load(args.config_yaml)
    OmegaConf.set_struct(cfg, False)

    dataset = CollaborationDatasetV0(cfg.habitat.dataset)
    if (
        cfg.get("episode_indices", None) is not None
        or cfg.get("episode_ids", None) is not None
        or cfg.get("evolve_episode_ids", None) is not None
    ):
        dataset = base_runner._select_episode_subset(dataset, cfg)

    try:
        base_runner.run_planner(cfg, dataset)
    finally:
        base_runner.clear_memory()


if __name__ == "__main__":
    main()
