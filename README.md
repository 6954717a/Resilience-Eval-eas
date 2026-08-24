<p align="center">
  <img src="doc/imgs/static/logo.png" alt="Resilience Evaluation logo" width="130"/>
</p>

<h1 align="center">Resilience Evaluation for Embodied Agent Systems</h1>

<p align="center">
  <b>Process-aware evaluation across Rebound, Stability, and Graceful Extensibility.</b>
</p>

<p align="center">
  <a href="https://6954717a.github.io/resilience_evaluation_web/">Web</a> |
  <a href="https://6954717a.github.io/resilience_evaluation_web/run.html">Runbook</a> |
  <a href="https://github.com/6954717a/Resilience-Eval-eas">Code</a>
</p>

EAS Resilience Evaluation is based on the Habitat 3.0 stack (Habitat-Sim 0.3.3 in the pinned PARTNR runtime); its dataset and benchmark formats follow [PARTNR Planner](https://github.com/facebookresearch/partnr-planner). Resilience Evaluation uses an LLM as the critic and calculates episode-level stage baselines from EAS execution experience.

## Run

After completing [INSTALLATION.md](INSTALLATION.md), use the numbered [Web Runbook](https://6954717a.github.io/resilience_evaluation_web/run.html) as the canonical runtime guide. The commands below are its compact entry-point index.

> [!IMPORTANT]
> This repository is an evaluation overlay, not a standalone PARTNR distribution. It does not ship PARTNR's `third_party/` modules, Habitat assets, datasets, or the frozen resilience stress bank. Run these commands from a complete PARTNR-compatible checkout after syncing `ResilienceEvaluationLayer/` into its `habitat_llm/` package. See [INSTALLATION.md](INSTALLATION.md).

Follow both upstream installation sources before running:

- [PARTNR Planner pinned installation](https://github.com/facebookresearch/partnr-planner/blob/ddfff19f4b6c098a31edea4d19e7b75db72433c2/INSTALLATION.md)
- [Habitat-Sim installation](https://github.com/facebookresearch/habitat-sim#installation)

### 1. Configure the model endpoints

The served planner model is user supplied. `QWEN35_SERVED_MODEL` is the compatibility variable used by the current Qwen/ChatML planner config. Start the planner vLLM server on `127.0.0.1:8000` as shown in the Runbook, or override `evaluation.planner.plan_config.llm.host` and `.port`. A non-Qwen model also requires matching planner and instruction configs.

```bash
export SERVED_MODEL_NAME="<your-served-model-name>"
# Example (Qwen): SERVED_MODEL_NAME="Qwen3.5-9B-YOYO-Instruct"
export QWEN35_SERVED_MODEL="$SERVED_MODEL_NAME"
export DATASET="data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
```

The resilience run additionally uses a separate OpenAI-compatible Judge endpoint. The joint config currently selects `gpt-5.1`.

```bash
export HABITAT_LLM_BASE_URL="https://<judge-host>/v1"
export OPENAI_API_KEY="<judge-api-key>"
```

### 2. How to run EAS Case Analysis

Run one PARTNR episode with the served planner model:

```bash
EAS_RUN="outputs/habitat_llm/eas-78-$(date +%Y%m%d-%H%M%S)"
python -m habitat_llm.examples.planner_demo_mp_new \
  --config-name baselines/qwen35_centralized_zero_shot_react_summary_vllm \
  hydra.run.dir="$EAS_RUN" habitat.dataset.data_path="$DATASET" '++episode_ids=[78]'
```

### 3. How to run Resilience EAS Case Analysis

The canonical case analysis is a `10 episodes × 3 seeds × 6 stress levels = 180 cells` joint evaluation grid. If no compatible Judge-scoped StageBaseline exists, a separate clean calibration pass runs first (up to 30 additional rollouts) and is not counted in those 180 cells. Stability and Graceful Extensibility are then post-processed from the shared evaluation evidence.

The frozen bank is a required target-runtime asset and is not included in this repository:

```bash
test -f evaluation/perturbation_banks/instruction_context_v6/manifest.json
```

Then run the joint experiment:

```bash
RES_RUN="outputs/habitat_llm/resilience-case-$(date +%Y%m%d-%H%M%S)"
python -u -m habitat_llm.examples.planner_demo_expq1 \
  --config-name experiments/resilience_joint_qwen35 \
  hydra.run.dir="$RES_RUN" habitat.dataset.data_path="$DATASET"
```

### 4. How to run Resilience EAS Benchmark

Run the first 400 entries of the PARTNR `v0_0/val` split. These are dataset indices, not episode IDs.

```bash
export PARTNR_BENCHMARK="data/datasets/partnr_episodes/v0_0/val.json.gz"
BENCH_RUN="outputs/habitat_llm/benchmark-400-$(date +%Y%m%d-%H%M%S)"
EPISODES="[$(seq -s, 0 399)]"
python -m habitat_llm.examples.planner_demo_mp_new \
  --config-name baselines/qwen35_centralized_zero_shot_react_summary_vllm \
  hydra.run.dir="$BENCH_RUN" habitat.dataset.data_path="$PARTNR_BENCHMARK" \
  num_proc=4 "++episode_indices=$EPISODES"
```

`num_proc=4` enables isolated workers; the current safety scheduler still admits at most one worker at a time unless its memory-gated execution config is explicitly changed.

## Canonical configurations

Hydra config paths below are relative to `ResilienceEvaluationLayer/conf/` in this repository and to `habitat_llm/conf/` in the runnable PARTNR checkout.

| Purpose | Config | Role |
|:--|:--|:--|
| EAS case and 400-episode benchmark | [`baselines/qwen35_centralized_zero_shot_react_summary_vllm.yaml`](ResilienceEvaluationLayer/conf/baselines/qwen35_centralized_zero_shot_react_summary_vllm.yaml) | Served planner model and centralized motor-only baseline. |
| Joint resilience case | [`experiments/resilience_joint_qwen35.yaml`](ResilienceEvaluationLayer/conf/experiments/resilience_joint_qwen35.yaml) | Ten episodes, three seeds, six stress levels, and ordered Stability/GE post-processing. |
| Runtime resilience instrumentation | [`baselines/qwen35_centralized_zero_shot_react_summary_vllm_state.yaml`](ResilienceEvaluationLayer/conf/baselines/qwen35_centralized_zero_shot_react_summary_vllm_state.yaml) | Rebound metrics, deterministic critic traces, and compact state exports. |
| Resilience defaults | [`evaluation/resilience_config.yaml`](ResilienceEvaluationLayer/conf/evaluation/resilience_config.yaml) | Perturbations, StageBaseline lifecycle, execution limits, Stability, and GE thresholds. |
| Evaluation runner | [`evaluation/centralized_evaluation_runner_motortoolsonly_multiagent.yaml`](ResilienceEvaluationLayer/conf/evaluation/centralized_evaluation_runner_motortoolsonly_multiagent.yaml) | Centralized two-agent execution contract. |
| Planner instructions | [`instruct/qwen_few_shot_centralized_motoronly_new.yaml`](ResilienceEvaluationLayer/conf/instruct/qwen_few_shot_centralized_motoronly_new.yaml) | Current Qwen/ChatML planning and feedback contract. |

Specialized experiment configs remain available for focused studies:

- [`experiments/expq1_qwen35.yaml`](ResilienceEvaluationLayer/conf/experiments/expq1_qwen35.yaml): separate Rebound, Stability, and GE validity suites.
- [`experiments/expq1_beta_qwen35.yaml`](ResilienceEvaluationLayer/conf/experiments/expq1_beta_qwen35.yaml): Stability-focused diagnostics.
- [`experiments/expq1_ge_qwen35.yaml`](ResilienceEvaluationLayer/conf/experiments/expq1_ge_qwen35.yaml): GE capacity diagnostics.
- [`experiments/llm_judge_stability_compare.yaml`](ResilienceEvaluationLayer/conf/experiments/llm_judge_stability_compare.yaml): controlled Judge comparison.

## Resilience metrics

| Aspect | Main output | Interpretation |
|:--|:--|:--|
| Rebound | `C_rec` | Extra cognitive, physical, and state-debt work required after a disruption. |
| Stability | `beta` | Policy/value-trajectory sensitivity to controlled perturbations. |
| Graceful Extensibility | `M_f(lambda)`, `lambda_f^*` | Remaining margin and the largest stress level that satisfies the execution contract. |

<p align="center">
  <img src="doc/imgs/readme/pipeline_metrics.png" alt="Resilience evaluation pipeline and metric definitions" width="95%"/>
</p>

Stage baselines are aggregated by Judge model and episode ID. Runtime evidence remains separate from statistical post-processing. Runs save their resolved config; the resilience scheduler and isolated-worker benchmark additionally save execution manifests.

## Outputs

| Run | Primary evidence |
|:--|:--|
| EAS case | `$EAS_RUN/results/episode_result_log.csv` |
| Resilience case | `$RES_RUN/results/resilience_joint_qwen35/joint_resilience_evaluation/raw_rollouts.csv` |
| Resilience summaries | `$RES_RUN/results/resilience_joint_qwen35/joint_resilience_evaluation/summary.csv` and `boundary_capacity_summary.csv` |
| Resilience suite status | `$RES_RUN/results/resilience_joint_qwen35/suite_manifest.json` |
| PARTNR benchmark | `$BENCH_RUN/results/episode_result_log.csv` and `$BENCH_RUN/results/mp_workers/execution_manifest.json` |

Packaged analysis can be regenerated without launching Habitat-Sim:

```bash
python Analysis_Code/main/run_all_expq1.py
python Analysis_Code/main/plot_expq2_method_metric_heatmap.py
```

## Repository layout

| Path | Contents |
|:--|:--|
| `ResilienceEvaluationLayer/` | Current PARTNR-compatible runtime overlay, metrics, monitors, planners, configs, and tests. |
| `Analysis_Code/` | Offline aggregation, verification, tables, and figures. |
| `Analysis_Data/` | Packaged experiment inputs and selected execution evidence. |
| `Analysis_Results/` | Derived tables and media. |
| `Reproduction/` | Historical compact reproduction snapshot; not the source of truth for current runtime configs. |
| [`doc/`](doc/README.md) | Current figure/vector assets and documentation index. |

## References

- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- [PARTNR Planner](https://github.com/facebookresearch/partnr-planner)
