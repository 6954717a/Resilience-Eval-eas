#!/usr/bin/env python3
# isort: skip_file
"""Run Exp-Q1 metric-validity experiments.

Exp-Q1 validates the three metric dimensions rather than testing an
optimization: Rebound cost, Stability neighborhood sensitivity, and GE
stress-response capacity. GE postprocess outputs are written under each
condition analysis directory, with a root-level ``capacity_summary.csv`` for
paper-facing analysis.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from habitat_llm.examples.expq_helpers import run_named_suite


@hydra.main(
    config_path="../conf",
    config_name="experiments/expq1",
    version_base=None,
)
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(
        "Running Exp-Q1: rebound validity, stability validity, and GE capacity validity."
    )
    run_named_suite(config, "expq1")


if __name__ == "__main__":
    main()
