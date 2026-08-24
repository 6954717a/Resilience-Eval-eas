# Runnable examples

This directory contains public entry points and the internal workers they use.
Generated trajectories, videos, checkpoints, and evaluation outputs are not source
files and must stay outside release archives.

## Planner demo

Run the standard multi-agent planner through Hydra:

```bash
python -m habitat_llm.examples.planner_demo \
  --config-name examples/planner_multi_agent_demo_config.yaml
```

`planner_demo_mp_new.py` is the current episode runner used by the experiment
orchestrators. It is kept as an internal compatibility module; new resilience runs
should use the wrappers below.

## Resilience evaluation

Run the joint Rebound, Stability, and Graceful Extensibility pipeline:

```bash
python -m habitat_llm.examples.planner_demo_resilience \
  --config-name experiments/resilience_joint_qwen35.yaml
```

The paper-facing experiment entry points are:

- `planner_demo_expq1.py`: metric-validity experiments.
- `planner_demo_expq2_inner.py`: internal-state stress experiments.
- `planner_demo_expq2_outer.py`: external perturbation experiments.
- `planner_demo_expq3.py`: metric-guided optimization experiments.

Each entry point composes its default config from `habitat_llm/conf/experiments/`.
Use a one-episode, one-seed smoke run before scheduling a complete stress grid.

## Utilities

- `scene_mapping.py` records RGB-D trajectories when trajectory logging is enabled.
- `skill_runner.py` runs explicit oracle-skill sequences in the sandbox.
- `fix_episode_placements.py` validates or repairs episode object placements.
- `verify_episodes.py` checks episode data before evaluation.

For example, validate one episode without applying corrections:

```bash
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.fix_episode_placements \
  hydra.run.dir=. \
  +validator_episode_indices=[0] \
  +validator_operations=[ep_obj_rec_inits] \
  +validator_correction_level=0
```
