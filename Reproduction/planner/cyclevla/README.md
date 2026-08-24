# CycleVLA Implementation for Habitat-LLM

This directory contains the implementation of **CycleVLA** (Proactive Self-Correcting Vision-Language-Action Models via Subtask Backtracking and Minimum Bayes Risk Decoding) adapted for the Habitat-LLM framework.

## Reference

- **Paper**: CycleVLA: Proactive Self-Correcting Vision-Language-Action Models via Subtask Backtracking and Minimum Bayes Risk Decoding
- **ArXiv**: https://arxiv.org/abs/2601.02295

## Core Concept

CycleVLA introduces **proactive self-correction** for robotic agents:

1. **Progress-Aware Execution**: Track progress within subtasks and detect critical transition points
2. **VLM Failure Prediction**: At subtask boundaries, use a VLM to predict potential failures before they occur
3. **Subtask Backtracking**: When failure is predicted, backtrack to an earlier subtask to retry
4. **MBR Decoding**: Use Minimum Bayes Risk decoding to select the most robust action after backtracking

## Architecture

```
habitat_llm/
├── evaluation/cyclevla/          # Subtask decomposition & progress
│   ├── __init__.py
│   └── subtask_manager.py        # SubtaskManager, Subtask dataclass
│
└── planner/cyclevla/             # Planning & execution logic
    ├── __init__.py
    ├── cycle_planner.py          # Main pipeline: run_cycle_pipeline()
    ├── vlm_predictor.py          # VLM-based failure prediction
    ├── mbr_sampler.py            # MBR action sampling
    └── rollback.py               # Cognitive backtracking
```

## Components

### 1. SubtaskManager (`evaluation/cyclevla/subtask_manager.py`)

Handles task decomposition and progress tracking:
- `decompose_task()`: Uses LLM to break instruction into atomic subtasks
- `estimate_progress()`: Heuristic-based progress estimation
- `is_boundary()`: Detects when to invoke VLM predictor

### 2. CycleVLMPredictor (`planner/cyclevla/vlm_predictor.py`)

VLM-based failure prediction at subtask boundaries:
- Constructs Chain-of-Thought prompts
- Returns `transit` (continue) or `backtrack` (retry earlier) decisions
- Includes confidence scoring for decision filtering

### 3. MBRActionSampler (`planner/cyclevla/mbr_sampler.py`)

Minimum Bayes Risk decoding for robust retry:
- Samples N action traces from LLM
- Computes pairwise distances (edit/jaccard)
- Selects consensus trajectory (minimum average distance)

### 4. CognitiveRollbackTool (`planner/cyclevla/rollback.py`)

Context-level backtracking (our adaptation):
- Saves checkpoints at subtask starts
- Restores planner context on backtrack
- Injects guidance to help LLM avoid previous mistakes

**Note**: Unlike CycleVLA's physical rollback (reversing delta actions), we use **cognitive backtracking** since Habitat-LLM is LLM-based rather than VLA-based.

### 5. run_cycle_pipeline (`planner/cyclevla/cycle_planner.py`)

Main entry point integrating all components:
```python
if planner.use_cycle_vla:
    response = run_cycle_pipeline(planner, instruction, env_interface, config)
```

## Usage

### Enable CycleVLA

In your planner configuration:

```yaml
evaluation:
  planner:
    use_cycle_vla: true
    cyclevla:
      progress_threshold: 0.8
      vlm_model: "gpt-4-turbo"
      mbr_num_samples: 5
      mbr_temperature: 0.7
```

Or use the provided config:
```bash
python planner_demo.py baseline=cyclevla_config
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `progress_threshold` | 0.8 | Progress level to trigger VLM check |
| `max_subtasks` | 10 | Maximum subtasks from decomposition |
| `vlm_model` | "gpt-4-turbo" | Model for failure prediction |
| `confidence_threshold` | "medium" | Min confidence for backtrack |
| `mbr_num_samples` | 5 | Samples for MBR decoding |
| `mbr_temperature` | 0.7 | Temperature for diverse sampling |
| `distance_metric` | "edit" | Distance metric: "edit" or "jaccard" |

## Execution Flow

```
1. Initialize (first call):
   - Decompose task into subtasks via LLM
   - Initialize components

2. Per-step execution:
   a. Save checkpoint if at subtask start
   b. Estimate progress within current subtask
   c. If progress >= threshold (boundary):
      - Call VLM predictor
      - If "backtrack": 
        → Restore context checkpoint
        → Clear future checkpoints
        → Use MBR to sample new actions
      - If "transit":
        → Mark subtask complete
        → Advance to next subtask
   d. Inject subtask guidance
   e. Generate LLM response

3. Repeat until all subtasks complete or failure
```

## Metrics

CycleVLA provides additional metrics for analysis:

```python
from habitat_llm.planner.cyclevla import get_cycle_statistics

stats = get_cycle_statistics(planner)
# Returns:
# - subtask_count: Number of subtasks
# - vlm_predictions: VLM prediction count
# - vlm_backtracks: Backtrack count
# - vlm_backtrack_rate: Backtrack frequency
# - mbr_sample_count: MBR usage count
# - checkpoint_count: Saved checkpoints
# - rollback_count: Rollback executions
```

## Differences from Original CycleVLA

| Aspect | Original CycleVLA | Our Implementation |
|--------|-------------------|-------------------|
| Action space | Continuous VLA (9-dim) | Discrete LLM actions |
| Progress signal | Trained model (action dim) | Heuristic estimation |
| Backtracking | Physical (reverse delta) | Cognitive (context reset) |
| MBR distance | Trajectory in R^6H | Edit distance on actions |
| VLM input | RGB images | Text state summary |

## Integration with Resilience Metrics

CycleVLA maps to our resilience framework:

- **Rebound**: Backtrack + MBR provides concrete recovery mechanism
- **Stability**: VLM boundary checks prevent error accumulation
- **Evolve**: Successful backtrack trajectories can be used for training

## Files Created

```
habitat_llm/
├── evaluation/cyclevla/
│   ├── __init__.py
│   └── subtask_manager.py
├── planner/cyclevla/
│   ├── __init__.py
│   ├── cycle_planner.py
│   ├── vlm_predictor.py
│   ├── mbr_sampler.py
│   └── rollback.py
├── conf/baselines/
│   └── cyclevla_config.yaml
└── planner/
    └── llm_planner.py (modified: added use_cycle_vla flag)
```
