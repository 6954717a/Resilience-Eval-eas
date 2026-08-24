# Simple Reproduction Guide

This document provides the instructions to reproduce the resilience evaluation results.

---

## Overview

The reproduction process consists of three main stages:

```
Stage 1: Run Evaluation     -->  Raw execution logs
Stage 2: Extract Metrics    -->  L1/L2/L3 JSON/CSV files
Stage 3: Visualize Results  -->  Figures and tables
```

---

## Prerequisites

Ensure you have completed the [installation](INSTALLATION.md) before proceeding.

---

## Stage 1: Run Evaluation

### 1.1 Quick Test (Single Episode)

```bash
cd Reproduction

# Run baseline with minimal data
python -m habitat_llm.examples.planner_demo_mp_new \
    --config-name baselines/centralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    evaluation.num_episodes=1
```

### 1.2 Full Evaluation (All Methods)

#### Baseline (ReAct)

```bash
python -m habitat_llm.examples.planner_demo_mp_new \
    --config-name baselines/centralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val.json.gz"
```

#### CycleVLA

```bash
python -m habitat_llm.examples.planner_demo_mp_new \
    --config-name baselines/cyclevla_config.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val.json.gz"
```

#### CoPAL

```bash
python -m habitat_llm.examples.planner_demo_mp_new \
    --config-name baselines/copal_config.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val.json.gz"
```

#### CLARE

```bash
python -m habitat_llm.examples.planner_demo_mp_new \
    --config-name baselines/clare_config.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val.json.gz"
```

#### SayCan

```bash
python -m habitat_llm.examples.planner_demo_mp_new \
    --config-name baselines/saycan_config.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val.json.gz"
```

#### Inner Monologue

```bash
python -m habitat_llm.examples.planner_demo_mp_new \
    --config-name baselines/inner_monologue_config.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val.json.gz"
```

### 1.3 Output Location

Evaluation outputs are saved to:

```
outputs/
+-- {dataset_name}/
    +-- analyses/
    |   +-- rebound/          # Rebound analysis logs
    |   +-- critic/           # Critic statistics
    |   +-- adca/             # ADCA trajectory analysis
    |   +-- resilience/       # Combined resilience metrics
    |   +-- reward_shaper/    # LLM reward shaping logs
    +-- prompts/              # LLM prompts
    +-- traces/               # Execution traces
    +-- planner-log/          # Detailed planner logs
```

---

## Stage 2: Extract Metrics

### 2.1 Run Metrics Extraction

```bash
cd Analysis_Code

# Full analysis pipeline
python run_analysis_2026-01-18_evolve.py

# Or run specific analysis
python run_analysis_2026-01-18_rebound.py
```

### 2.2 Manual Metrics Computation

```python
import pandas as pd
from metrics.episode_level import extract_episode_metrics
from metrics.task_family_level import compute_task_family_metrics
from metrics.evolve_level import compute_evolve_metrics

# Load raw data
raw_data = load_analysis_data("../outputs/analyses/")

# Step 1: Extract L1 (Episode-Level) metrics
l1_results = []
for episode in raw_data:
    l1 = extract_episode_metrics(episode)
    l1_results.append(l1)

# Save L1 metrics
l1_df = pd.DataFrame(l1_results)
l1_df.to_csv("../Outputs_v0/L1_metrics/L1_Method.csv", index=False)

# Step 2: Compute L2 (Task-Family-Level) metrics
l2_metrics = compute_task_family_metrics(l1_df)

# Save L2 metrics
import json
with open("../Outputs_v0/L2_metrics/L2_Method.json", "w") as f:
    json.dump(l2_metrics, f, indent=2)

# Step 3: Compute L3 (Evolve-Level) metrics
l3_metrics = compute_evolve_metrics(
    evolution_context=evolution_data,
    l2_metrics=l2_metrics,
    episodes_per_evolve=15
)

# Save L3 metrics
with open("../Outputs_v0/L3_metrics/L3_Method.json", "w") as f:
    json.dump(l3_metrics, f, indent=2)
```

### 2.3 Output Format Examples

#### L1 Metrics (CSV)

| episode_id | T_rec | B_epi | sigma_V | N_replan | GD | P_cliff |
|------------|-------|-------|---------|----------|-----|---------|
| ep_001 | 3.2 | 2 | 0.15 | 1 | 0.02 | 0 |
| ep_002 | 0.0 | 0 | 0.08 | 0 | 0.01 | 0 |
| ep_003 | 5.8 | 4 | 0.22 | 2 | 0.05 | 1 |

#### L2 Metrics (JSON)

```json
{
  "L2_Episode_Count": 309,
  "L2_MTTR_A": 4.08,
  "L2_MTTR_A_CI_Low": 4.07,
  "L2_MTTR_A_CI_High": 4.09,
  "L2_MTBF_A": 481.31,
  "L2_E_RR": 0.0026,
  "L2_Beta": 0.0024,
  "L2_LES": 0.106,
  "L2_Safety_Score": 0.113
}
```

#### L3 Metrics (JSON)

```json
{
  "L3_Total_Episodes": 309,
  "L3_NRR": 0.992,
  "L3_BWT": -0.071,
  "L3_FWT": 0.068,
  "L3_Lower_Bound": 0.633,
  "L3_Cumulative_Regret": 77.34
}
```

---

## Stage 3: Visualize Results

### 3.1 Generate Figures

```bash
cd Analysis_Code

# Generate radar charts
python visualize/fig5_visualize_signature_atlas.py

# Generate validation plots
python visualize/visualize_predictive_validity.py

# Generate method comparison
python visualize/fig1_visualize_metric_validation_by_method.py
```

### 3.2 Custom Visualization

```python
import matplotlib.pyplot as plt
import json

# Load L2 metrics for multiple methods
methods = ["Baseline", "CycleVLA", "CoPAL", "CLARE", "Reflexion"]
metrics_data = {}

for method in methods:
    with open(f"../Outputs_v0/L2_metrics/L2_{method}.json") as f:
        metrics_data[method] = json.load(f)

# Plot MTTR comparison
fig, ax = plt.subplots(figsize=(10, 6))
mttr_values = [metrics_data[m]["L2_MTTR_A"] for m in methods]
ax.bar(methods, mttr_values)
ax.set_ylabel("MTTR-A (planning steps)")
ax.set_title("Mean Time To Recovery Comparison")
plt.savefig("mttr_comparison.png", dpi=150)
```

---

## Using Pre-computed Results

If you want to skip Stage 1 and 2, use the pre-computed results in `Outputs_v0/`:

```python
import pandas as pd
import json

# Load pre-computed L1 metrics
l1_reflexion = pd.read_csv("Outputs_v0/L1_metrics/L1_Reflexion.csv")

# Load pre-computed L2 metrics
with open("Outputs_v0/L2_metrics/L2_Reflexion.json") as f:
    l2_reflexion = json.load(f)

# Load pre-computed L3 metrics
with open("Outputs_v0/L3_metrics/L3_Reflexion.json") as f:
    l3_reflexion = json.load(f)

# Analyze
print(f"Episodes: {l2_reflexion['L2_Episode_Count']}")
print(f"MTTR-A: {l2_reflexion['L2_MTTR_A']:.2f}")
print(f"Beta-Stability: {l2_reflexion['L2_Beta']:.4f}")
print(f"NRR: {l3_reflexion['L3_NRR']:.3f}")
```

---

## Reproducing Paper Results

### RQ2: Metric Validity

```bash
# Use RQ2 data
cd Analysis_Code
python run_rq2_validity_analysis.py \
    --data-dir ../Analysis_Data/RQ2_analyse_data/
```

### RQ3: Backbone Comparison

```bash
# Use RQ3 data
python run_rq3_backbone_analysis.py \
    --data-dir ../Analysis_Data/RQ3_backbone_full_data/
```

---

## Configuration Options

### Common YAML Parameters

```yaml
# baselines/cyclevla_config.yaml

evaluation:
  num_episodes: 100           # Number of episodes to run
  max_steps: 500              # Max steps per episode
  
  planner:
    plan_config:
      llm:
        inference_mode: hf    # hf, openai, vllm
        generation_params:
          engine: meta-llama/Meta-Llama-3-8B-Instruct
          temperature: 0.7
          max_tokens: 512

  # Resilience evaluation settings
  critic:
    enabled: true
    update_frequency: 1
  
  rebound:
    enabled: true
    detection_threshold: 3
  
  safety_monitor:
    enabled: true
```

### LLM Backend Options

| Backend | Config Key | Notes |
|---------|------------|-------|
| HuggingFace | `inference_mode: hf` | Local GPU required |
| OpenAI API | `inference_mode: openai` | Set OPENAI_API_KEY |
| vLLM | `inference_mode: vllm` | Start vLLM server first |

---

## Troubleshooting

### Evaluation Issues

**Problem**: Evaluation hangs on first episode

```bash
# Check if LLM is responding
python -c "from transformers import pipeline; p = pipeline('text-generation', model='gpt2'); print(p('Hello'))"
```

**Problem**: Missing analysis files

```bash
# Ensure all monitors are enabled in config
evaluation.critic.enabled: true
evaluation.rebound.enabled: true
```

### Metrics Issues

**Problem**: L1 metrics show all zeros

- Check if `analyses/` directory contains valid JSON files
- Verify episode summaries contain required fields

**Problem**: NaN values in L2/L3 metrics

- Ensure sufficient episode count (minimum 10)
- Check for division by zero in aggregation

---

## Expected Runtime

| Stage | Hardware | Time Estimate |
|-------|----------|---------------|
| Evaluation (100 eps) | RTX 3090 | 4-8 hours |
| Evaluation (100 eps) | CPU only | 12-24 hours |
| Metrics Extraction | Any | 5-10 minutes |
| Visualization | Any | 1-2 minutes |

---

## Next Steps

- Modify configurations in `Reproduction/baselines/` to test new settings
- Add new methods by creating planner implementations in `Reproduction/planner/`
- Extend metrics in `Analysis_Code/metrics/`

For questions, please open a GitHub issue.
