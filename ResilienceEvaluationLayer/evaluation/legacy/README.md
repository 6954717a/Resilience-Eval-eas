# Legacy Evaluation Code

This directory contains deprecated evaluation code that has been superseded by the dimension-based metrics system.

## Migration Guide

The evaluation system has been refactored from **implementation-method-based** organization to **analysis-dimension-based** organization.

### Old Structure (Deprecated)
```
evaluation/
├── eval_logging.py
│   ├── log_rebound_analysis()      # DEPRECATED
│   ├── log_saycan_analysis()       # DEPRECATED
│   └── log_resilience_summary()    # DEPRECATED
└── resilience_metrics_logging.py
    └── log_resilience_metrics()    # DEPRECATED
```

### New Structure (Current)
```
evaluation/
└── metrics/
    ├── rebound/
    │   ├── rebound_metrics.py      # Unified Rebound metrics
    │   ├── rebound_collector.py    # Collects from ReboundManager + CoPAL
    │   └── rebound_logging.py      # JSON persistence
    ├── stability/
    │   ├── stability_metrics.py    # Policy stability metrics
    │   ├── stability_collector.py  # Collects from StateEncoder + SafetyMonitor
    │   └── stability_logging.py    # JSON persistence
    └── degradation/
        ├── degradation_metrics.py  # Graceful degradation metrics
        ├── degradation_collector.py # Collects from DegradationMonitor
        └── degradation_logging.py  # JSON persistence
```

## Three Core Dimensions

### 1. Rebound (韧性恢复)
**Definition**: System's ability to recover from failures

**Metrics**:
- `B_epi`: Correction Cost = N_retry + γ×N_backtrack (γ=2.0)
- `MTTR`: Mean Time To Recovery (planning steps)
- `MTBF`: Mean Time Between Failures (planning steps)
- `RR`: Recovery Ratio = N_fixed / N_occurred
- `T_rec`: Cognitive Recovery Latency

**Migration**:
```python
# Old (deprecated)
from habitat_llm.evaluation.eval_logging import log_rebound_analysis
log_rebound_analysis(info, planner, output_dir, episode_filename, env_interface)

# New
from habitat_llm.evaluation.metrics import ReboundCollector, log_rebound_metrics
collector = ReboundCollector(config)
metrics = collector.collect_from_planner(planner, total_steps, degradation_monitor)
log_rebound_metrics(metrics, output_dir, episode_filename, env_interface)
```

### 2. Stability (策略稳定性)
**Definition**: Output consistency under perturbation/sampling variation

**Metrics**:
- `β`: Policy Stability Score = 1 - (σ_action / σ_max)
- `σ²_V`: Value Function Variance (from StateEncoder)
- `N_replan`: Replanning count
- `P_cbf`: CBF penalty from SafetyMonitor

**Migration**:
```python
# Old (deprecated)
from habitat_llm.evaluation.eval_logging import log_saycan_analysis
log_saycan_analysis(saycan_data, output_dir, episode_filename, env_interface)

# New
from habitat_llm.evaluation.metrics import StabilityCollector, log_stability_metrics
collector = StabilityCollector(config)
metrics = collector.collect(critic, planner_infos, safety_monitor, planner)
log_stability_metrics(metrics, output_dir, episode_filename, env_interface)
```

### 3. Degradation (优雅退化)
**Definition**: Version evolution without catastrophic regression

**Metrics**:
- `AUC_loss`: Resilience Loss Area = ∫[t_deg, t_rec] (θ_nom - P(t)) dt
- `P_cliff`: Cliff Probability = P(ΔP > θ_cliff) where θ_cliff = 0.3
- `T_rec`: Cognitive Recovery Latency
- `L_bd`: Evolve Lower Bound = min(P_v1, ..., P_vn) - ε
- `NRR`: Normalized Resilience Ratio = MTBF / (MTBF + MTTR)

**Migration**:
```python
# Old (deprecated)
from habitat_llm.evaluation.resilience_metrics_logging import log_resilience_metrics
log_resilience_metrics(planner, planner_infos, output_dir, episode_filename, env_interface, info)

# New
from habitat_llm.evaluation.metrics import DegradationCollector, log_degradation_metrics
collector = DegradationCollector(config)
metrics = collector.collect_from_monitor(degradation_monitor)
log_degradation_metrics(metrics, output_dir, episode_filename, env_interface)
```

## Configuration Migration

### Old Config Format
```yaml
planner:
  plan_config:
    rebound:
      enabled: true
      max_retries: 3

evaluation:
  context_evolve:
    enabled: false
```

### New Config Format
```yaml
evaluation:
  metrics:
    rebound:
      enabled: true
      gamma: 2.0
    stability:
      enabled: true
      use_saycan_scores: false
    degradation:
      enabled: true
```

## Backward Compatibility

The legacy functions are still available but will emit `DeprecationWarning`. They will be removed in a future version.

To suppress warnings during migration:
```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='habitat_llm.evaluation')
```

## Output Directory Structure

### Old
```
analyses/
├── rebound/
│   └── rebound_analysis-episode_*.json
├── saycan/
│   └── saycan_analysis-episode_*.json
└── resilience/
    └── resilience_metrics-episode_*.json
```

### New
```
analyses/
├── rebound/
│   └── rebound_metrics-episode_*.json
├── stability/
│   └── stability_metrics-episode_*.json
└── degradation/
    └── degradation_metrics-episode_*.json
```

## References

- Theory Document: `梳理论证逻辑.pdf`
- Plan Document: `~/.claude/plans/precious-drifting-catmull.md`
- Implementation: `habitat_llm/evaluation/metrics/`
