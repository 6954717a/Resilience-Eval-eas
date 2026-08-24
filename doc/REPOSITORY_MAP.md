# Repository Map

## Project mode

This is a `software-git` research-artifact repository. It combines a Python
evaluation stack, packaged experiment evidence, analysis scripts, generated
figures, and reproduction material.

The GitHub repository slug is `Resilience-Eval-eas`; the display name is
**Resilience Evaluation**. Embodied Agent Systems (EAS) are the current
application target, not part of the core project name. On Windows, clone the
repository into the short local directory `ResEval` because packaged evidence
contains long relative paths.

## Primary deliverables and entry points

- `ResilienceEvaluationLayer/`: simulator integration, planners, runtime probes,
  resilience metrics, Hydra configuration, aggregation, examples, and tests.
- `Analysis_Code/main/run_all_expq1.py`: rebuilds the Exp-Q1 figures and tables.
- `Analysis_Code/main/plot_expq2_method_metric_heatmap.py`: rebuilds the benchmark
  profile view.
- `Analysis_Code/theory_process/run_analysis_2026-01-18_rebound.py`: legacy
  rebound extraction entry point.
- `Analysis_Data/`: packaged aggregate tables and selected runtime evidence.
- `Analysis_Results/`: precomputed result tables and media.
- `Reproduction/`: compact reproduction-oriented copy of planner and evaluation
  code.

## Source-of-truth surfaces

- Metric implementations: `ResilienceEvaluationLayer/evaluation/metrics/`.
- Runtime probes: `ResilienceEvaluationLayer/evaluation/monitors/`.
- Experiment configuration: `ResilienceEvaluationLayer/conf/experiments/`.
- Judge-scoped baseline artifacts:
  `ResilienceEvaluationLayer/evaluation/stage_baselines/`.
- Public overview and claims: `README.md`.
- Installation and first-run guidance: `INSTALLATION.md` and
  `doc/SIMPLE_REPRODUCE.md`.
- Packaged tables used by figures: `Analysis_Data/Aggregated_Tables/`.

## Dependencies and external systems

- Python 3.10+, PyTorch, Hydra, NumPy, Pandas, Matplotlib, and NetworkX.
- Habitat-Sim and the Habitat-LLM/PARTNR execution model.
- Optional LLM endpoints and W&B logging. Credentials are local-only and must
  never be committed.
- There is currently no lockfile, package manifest, CI workflow, or container
  definition; `requirements.txt` is the dependency inventory.

## Validation surfaces

- Seventeen `test_*.py` files cover dataset, planner, graph, evaluation, margin,
  and state-encoding behavior.
- Analysis smoke commands are documented in `README.md`.
- Repository checks: `git status`, scoped `git diff --cached --check` for
  migration-authored files, `git fsck --strict`, staged secret-pattern scans,
  and staged large-file scans.
- Full Habitat-Sim validation requires the target runtime and datasets; a local
  static check is not equivalent to a simulator run.

## Generated and local-only artifacts

Python caches, editor swap files, virtual environments, Hydra/W&B output,
checkpoints, and plaintext API-key files are excluded by `.gitignore`. Packaged
`Analysis_Data/` and `Analysis_Results/` remain tracked because they are part of
the research artifact rather than disposable local output.

## Risk areas and unknowns

- Many imports and commands still use the upstream package name `habitat_llm`,
  while this checkout's main directory is `ResilienceEvaluationLayer/`. A package
  migration must be designed and validated separately from the repository rename.
- `Reproduction/` duplicates portions of the main package and contains both exact
  copies and divergent files. The canonical implementation boundary is not yet
  documented.
- The README states MIT licensing, but no `LICENSE` file is present; the copyright
  holder and final license file require owner confirmation.
- The imported legacy sources contain pre-existing trailing whitespace, so an
  all-files initial-commit `git diff --cached --check` is not clean. Migration-
  authored files are checked separately to avoid an unrelated bulk rewrite.
- API keys present in the legacy history must be revoked or rotated at their
  providers even though the fresh repository excludes the local key files.
- Publishing the new remote, setting its visibility, and assigning GitHub About,
  Website, and topics are external owner decisions.
