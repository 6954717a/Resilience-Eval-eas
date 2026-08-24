#!/usr/bin/env python3
# isort: skip_file
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from habitat_llm.examples.expq_helpers import run_named_suite


@hydra.main(
    config_path="../conf",
    config_name="experiments/expq2_inner",
    version_base=None,
)
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_named_suite(config, "expq2_inner")


if __name__ == "__main__":
    main()
