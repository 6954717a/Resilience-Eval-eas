<p align="center">
  <img src="doc/imgs/static/logo.png" alt="Resilience Evaluation logo" width="130"/>
</p>

<h1 align="center">Resilience Evaluation</h1>

<p align="center">
  <b>A process-aware evaluation layer applied to embodied agent systems: rebound, stability, and graceful extensibility.</b>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch 2.0+"></a>
</p>

<p align="center">
  <a href="#why-this-repository">Why</a> |
  <a href="#main-evaluation-line">Main Line</a> |
  <a href="#pipeline">Pipeline</a> |
  <a href="#results">Results</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#repository-layout">Layout</a>
</p>

---

## Why This Repository

**Motivation:** Outcome centric metrics as Success rate tells whether an embodied-agent episode ended well. It does not tell whether the agent recovered cheaply, stayed stable under small perturbations, or maintained bounded behavior as task stress increased. Two agents can reach the same outcome while taking very different resilience paths.

**Resilience Evaluation Framework:** We draw on *resilience-engineering* concepts to define a reusable, process-aware evaluation framework. This repository applies the framework to embodied agent systems (EAS), packaging the code, result tables, and figure assets for a **Resilience Evaluation Layer** that instruments embodied-agent execution and turns process traces into three complementary resilience aspects: Rebound, Stability, and Graceful Extensibility.

- **Rebound:** how costly it is to recover after a local disruption.
- **Stability:** how sensitive the policy and value trajectory are to small perturbations.
- **Graceful Extensibility:** how far the agent can operate as task stress increases.

<p align="center">
  <img src="doc/imgs/readme/overview.png" alt="Same outcome, different resilience behavior in embodied-agent execution" width="95%"/>
</p>

<p align="center">
  <b>Figure 1.</b> Outcome-centric evaluation hides process differences; the resilience layer exposes recovery, instability, and stress-response behavior.
</p>

---

## Main Evaluation Line

The public result surface is intentionally compact:

```text
execution traces
  -> synchronized event logs
  -> stage baselines and perturbation windows
  -> C_rec, beta, M_f(lambda), lambda_f^*
  -> method-level resilience profile
```

| Aspect | Symbol | What it measures | Main evidence |
|:-------|:-------|:-----------------|:--------------|
| Rebound | `C_rec` | Extra cognitive, physical, and state-debt work needed to recover after a disruption. | Recovery windows and paired perturbation rows. |
| Stability | `beta` | Change in policy/value behavior under semantic or execution perturbations. | Clean-vs-perturbed trajectory and critic/value diagnostics. |
| Graceful Extensibility | `M_f(lambda)` | Stress-response margin as task stress `lambda` increases. | Stress sweep and task-family heatmap. |
| Graceful Extensibility | `lambda_f^*` | Largest stress level that still satisfies the execution contract. | Family-level capacity summary. |

Legacy code and archived tables may still use older L1/L2/L3 names. The README, figures, and manuscript-facing summaries use the current Rebound/Stability/Graceful Extensibility framing.

---

## Pipeline

<p align="center">
  <img src="doc/imgs/readme/pipeline_metrics.png" alt="Resilience evaluation pipeline and metric definitions" width="95%"/>
</p>

<p align="center">
  <b>Figure 2.</b> Non-intrusive probes log execution traces, extract resilience events, and compute process-aware metrics.
</p>

| Layer | Role | Representative files |
|:------|:-----|:---------------------|
| Runtime evidence | Stores prompts, traces, planner logs, simulator stats, and per-episode analyses. | `ResilienceEvaluationLayer/conf/experiments/`, `Analysis_Data/Exp*` |
| Metric extraction | Converts logs and traces into rebound, stability, and stress-response signals. | `ResilienceEvaluationLayer/evaluation/metrics/`, `ResilienceEvaluationLayer/evaluation/monitors` |
| Aggregation | Builds Exp-Q1 validity, Exp-Q2 benchmark, and Exp-Q3 optimization tables samples. | `ResilienceEvaluationLayer/evaluation/offline_metrics_aggregator.py`, `ResilienceEvaluationLayer/evaluation/evaluation_runner` |
| Figures | Generates GitHub-renderable PNG previews and original paper PDFs. | `Analysis_Code/` |


---

## Results

### Metric Validity

Exp-Q1 (Validity Exp) checks whether the three metrics separate process behavior that success rate alone misses.

<p align="center">
  <img src="doc/imgs/fig_q1_metric_validity_map.png" alt="Metric validity map for rebound, stability, and graceful extensibility" width="95%"/>
</p>

<p align="center">
  <b>Figure 3.</b> Rebound, Stability, and Graceful Extensibility respond to different perturbation families.
</p>

The packaged validity surface supports three concrete readings:

- Rebound cost increases under physical perturbations such as surface rewrite and object-state toggle, even when episodes can still complete.
- Stability beta is low on clean runs and higher under filler injection or surface rewrite, showing sensitivity to small semantic changes.
- Graceful Extensibility is represented as a stress-response curve and a task-family heatmap, so the result is not only a single success-rate number.

### Benchmark Profiles

Exp-Q2 (Benchmark Exp) compares ten embodied-agent methods with the same resilience profile surface.

<p align="center">
  <img src="doc/imgs/readme/benchmark_profiles.png" alt="Method resilience profiles across rebound, stability, graceful extensibility, completion, and steps" width="95%"/>
</p>

<p align="center">
  <b>Figure 4.</b> Methods with similar task outcomes can have different process-level resilience signatures.
</p>

Packaged benchmark table from `Analysis_Data/Aggregated_Tables/calc_table_old/expq2_method_metric_heatmap_scores.csv`:

| Method | Profile | `C_rec` | `beta` | `lambda_f^*` | SR | Completion | Steps |
|:-------|--------:|--------:|-------:|-------------:|---:|-----------:|------:|
| Baseline | 0.780 | 25.0 | 0.200 | 0.79 | 53.0 | 69.7 | 3671 |
| CoPAL | 0.761 | 11.9 | 0.302 | 0.85 | 55.8 | 72.6 | 4846 |
| ReAct | 0.698 | 6.5 | 0.239 | 0.50 | 38.5 | 57.8 | 3374 |
| InnerMono | 0.591 | 43.5 | 0.149 | 0.53 | 49.5 | 67.5 | 3670 |
| CLARE | 0.574 | 21.6 | 0.344 | 0.72 | 28.4 | 46.0 | 4249 |
| SayCan | 0.544 | 38.5 | 0.283 | 0.72 | 57.0 | 72.6 | 4682 |
| AgentEvolver | 0.431 | 19.2 | 0.264 | 0.21 | 48.8 | 66.8 | 3084 |
| CycleVLA | 0.409 | 35.5 | 0.400 | 0.72 | 22.2 | 47.5 | 4688 |
| SMART-LLM | 0.371 | 28.2 | 0.339 | 0.40 | 55.7 | 74.3 | 5085 |
| Reflexion | 0.369 | 57.3 | 0.299 | 0.66 | 56.3 | 75.0 | 4390 |

One useful sanity check is CoPAL vs. Reflexion: their success rates are close (`55.8` vs. `56.3`), but their recovery costs are not (`11.9` vs. `57.3`). The resilience layer therefore exposes a process difference that task outcome metrics flatten.

### Targeted Optimization

Exp-Q3 (Optimization Exp) tests whether targeting one resilience aspect changes the corresponding metric.

| Target | Baseline | Optimized | Main observed change |
|:-------|:---------|:----------|:---------------------|
| Rebound | `C_rec = 54.31`, completion `0.967` | `C_rec = 16.48`, completion `1.000` | Recovery cost drops by `69.6%` without lowering completion. |
| Stability | `beta_step = 0.400`, completion `0.886` | `beta_step = 0.244`, completion `0.929` | Local replanning variance drops by `38.9%`. |
| Graceful Extensibility | completion `0.833`, formal GE incomplete | completion `0.917`, pilot GE | Completion improves under the GE pilot setting, but formal optimized-vs-baseline margin remains incomplete. |

The GE row is deliberately labeled as pilot evidence. The benchmark and validity claims are landed; the GE optimization claim should be read as a targeted pilot rather than a completed formal capacity proof.

---

## Quick Start

Install the package dependencies:

```bash
git clone -c core.longpaths=true https://github.com/6954717a/Resilience-Eval-eas.git ResEval
cd ResEval

conda create -n resilience python=3.10
conda activate resilience
pip install -r requirements.txt
```

See [INSTALLATION.md](INSTALLATION.md) and [doc/SIMPLE_REPRODUCE.md](doc/SIMPLE_REPRODUCE.md) for the longer setup notes.

Common analysis entry points:

```bash
# Rebuild Exp-Q1 plots and tables from packaged analysis inputs.
python Analysis_Code/main/run_all_expq1.py

# Rebuild the benchmark heatmap/profile view.
python Analysis_Code/main/plot_expq2_method_metric_heatmap.py

# Run a legacy rebound extractor from the theory-process layer.
python Analysis_Code/theory_process/run_analysis_2026-01-18_rebound.py
```

### Habitat-Sim Runtime Simulation Case

<p align="center">
  <img src="doc/imgs/readme/habitat_sim_case.png" alt="Habitat-Sim embodied agent evaluation case illustration" width="70%"/>
</p>

<p align="center">
  <b>Simulation Runtime Case.</b> Habitat-Sim rollout context for the embodied traces analyzed here. Vector source: <a href="doc/imgs/Habitat-Sim_Case.pdf">doc/imgs/Habitat-Sim_Case.pdf</a>.
</p>

---

## Repository Layout

We follow the Habitat-LLM infrastructure, and form our evaluation layer based on the simulator. The **inner** `ResilienceEvaluationLayer/` directory is the main package where environments setup, EAS executes tasks, resilience probes, metric code, Hydra experiment configs, and aggregation entry points live.

```text
Resilience-Eval-eas/                              # repository root (use ResEval locally on Windows)
|-- ResilienceEvaluationLayer/                     # core package: sim + planners + resilience evaluation
|   |-- evaluation/                                # process-aware evaluation layer (primary)
|   |   |-- metrics/                               # rebound/, stability/, extensibility/, degradation/
|   |   |-- monitors/                              # runtime probes and trace hooks
|   |   |-- perturbation/                          # perturbation specs and application
|   |   |-- methods/                               # per-method metric helpers (e.g., CoPAL, CLARE)
|   |   |-- core/                                  # shared evaluation core (critic, world-state helpers)
|   |   |-- llm_evaluator/                         # offline / LLM-assisted analyzers
|   |   |-- analysis/                              # aggregation helpers alongside top-level scripts
|   |   |-- offline_metrics_aggregator.py          # rolls traces into experiment tables
|   |   |-- evaluation_runner.py                   # evaluation / sweep driver
|   |   |-- stage_baselines/                       # baseline snapshots used by staging logic
|   |   `-- experiments/                           # experiment wiring (subpackage)
|   |-- conf/                                      # Hydra configs
|   |   |-- experiments/                           # Exp-Q1 / Exp-Q2 / Exp-Q3-style YAML
|   |   |-- baselines/                             # method and planner presets
|   |   |-- habitat_conf/                          # Habitat agent + task YAML
|   |   `-- instruct/, planner/, llm/, ...        # prompts, tools, training, logging
|   |-- planner/                                   # EAS LLM planners
|   |-- agent/                                     # environment binding and rollout glue
|   |-- tools/, sims/                              # skills, motor primitives, prompts, simulator integration
|   |-- examples/                                  # runnable demos and our experiments
|   |-- llm/, context/, perception/, world_model/  # model backends and supporting modules
|   |-- finetuning/, ...                           # further connectors and Habitat-LLM extension
|   `-- tests/                                     # unit tests for evaluation and planners
|-- Reproduction/                                  # typical EAS reproduction
|-- Analysis_Code/                                 # offline plots and table rebuilds (uses packaged data)
|   |-- main/                                      # current plotting and aggregation scripts
|   `-- theory_process/                            # legacy extractors and metric utilities
|-- Analysis_Data/
|   |-- Aggregated_Tables/                         # packaged Experiments CSV inputs
|   `-- Exp1_*/                                    # packaged runtime evidence snapshots
|-- Analysis_Results/                              # precomputed CSV/JSON and figure inputs
|-- doc/
|   |-- imgs/                                      # source figures, README PNG exports, paper PDFs
|   `-- SIMPLE_REPRODUCE.md
|-- INSTALLATION.md
|-- requirements.txt
`-- README.md
```

---

## License

MIT License.

## Acknowledgments

- Built on the embodied-agent evaluation stack around Habitat-LLM and PARTNR-style task execution.
- Resilience framing is inspired by process-aware resilience analysis and Woods' concepts of resilience.
