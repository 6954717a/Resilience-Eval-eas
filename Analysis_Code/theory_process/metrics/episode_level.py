import re
import numpy as np

def extract_episode_metrics(entry):
    """
    Extract Level 1 (Episode-Level) metrics from a single episode summary.
    
    Metrics:
    - T_rec (Cognitive Recovery Latency) -> rebound.rebound_mttr
    - B_epi (Correction Cost) -> rebound.rebound_count
    - sigma_V (Value Function Variance) -> critic.value_loss (Proxy)
    - AUC_loss (Resilience Loss Area) -> derived from progress events
    - P_cliff (Cliff Probability) -> derived from adca or abrupt failure
    - Execution Quality -> analysis.execution_quality_score
    - Safety -> Derived from rebound and penalties
    - CBF Penalty -> Derived from safety violations
    """
    metrics = {}
    
    # 1. Base Info
    metrics['episode_id'] = entry.get('episode_id', 'unknown')
    metrics['instruction'] = entry.get('task_instruction') or entry.get('instruction', '')
    
    # 2. Performance (Outcome)
    # Check multiple locations for success
    if 'task_state_success' in entry:
        metrics['task_success'] = float(entry['task_state_success'])
    elif 'performance' in entry:
        metrics['task_success'] = float(entry['performance'].get('success_rate', 0.0))
    else:
        metrics['task_success'] = 0.0

    if 'task_percent_complete' in entry:
        metrics['task_percent_complete'] = float(entry['task_percent_complete'])
    elif 'performance' in entry:
        metrics['task_percent_complete'] = float(entry['performance'].get('mean_completion', 0.0))
    else:
        metrics['task_percent_complete'] = 0.0
        
    metrics['sim_step_count'] = float(entry.get('sim_step_count', 0.0))
    if metrics['sim_step_count'] == 0 and 'performance' in entry:
        metrics['sim_step_count'] = float(entry['performance'].get('mean_steps', 0.0))

    metrics['runtime'] = float(entry.get('runtime', 0.0))
    
    # 3. Rebound Metrics (Level 1 - Rebound)
    rebound = entry.get('rebound', {})
    if 'rebound_mttr' in rebound:
        metrics['T_rec'] = float(rebound.get('rebound_mttr', {}).get('mean', 0.0) if isinstance(rebound.get('rebound_mttr'), dict) else rebound.get('rebound_mttr', 0.0))
    else:
        metrics['T_rec'] = 0.0
        
    if 'rebound_count' in rebound:
        metrics['B_epi'] = float(rebound.get('rebound_count', {}).get('mean', 0.0) if isinstance(rebound.get('rebound_count'), dict) else rebound.get('rebound_count', 0.0))
    else:
        metrics['B_epi'] = 0.0
    
    # B_epi with γ weight (from updated ReboundManager)
    if 'b_epi' in rebound:
        metrics['b_epi'] = float(rebound.get('b_epi', 0.0))
    else:
        metrics['b_epi'] = metrics['B_epi']  # Fallback
        
    if 'rebound_mtbf' in rebound:
        metrics['rebound_mtbf'] = float(rebound.get('rebound_mtbf', {}).get('mean', 0.0) if isinstance(rebound.get('rebound_mtbf'), dict) else rebound.get('rebound_mtbf', 0.0))
    else:
        metrics['rebound_mtbf'] = 0.0
    
    # RR (Recovery Ratio) - Eq.12 from Detailed_Metrics.tex
    if 'recovery_ratio' in rebound:
        metrics['recovery_ratio'] = float(rebound.get('recovery_ratio', 0.0))
    else:
        metrics['recovery_ratio'] = 0.0

    
    # 4. Stability Metrics (Level 1 - Stability)
    # Critic Value Loss as proxy for Value Variance (sigma_squared_V)
    critic = entry.get('critic', {})
    metrics['sigma_V_proxy'] = float(critic.get('value_loss', {}).get('mean', 0.0) if isinstance(critic.get('value_loss'), dict) else critic.get('value_loss', 0.0))
    
    metrics['replanning_count'] = float(entry.get('replanning_count', 0.0))
    if metrics['replanning_count'] == 0 and 'efficiency' in entry:
         metrics['replanning_count'] = float(entry['efficiency'].get('replanning_total', {}).get('mean', 0.0))

    # CBF Penalty Proxy
    # If using rebound data, safety violations can be estimated
    metrics['cbf_penalty_proxy'] = metrics['B_epi'] * 0.5  # Assumed cost
    
    # Safety Proxy: If no explicit cbf_penalty logging, infer from rebound & success
    metrics['is_safe_proxy'] = 1.0 if metrics['B_epi'] < 5 else 0.0
    
    # 5. Degradation Metrics (Level 1 - Degradation)
    analysis = entry.get('analysis', {})
    metrics['execution_quality'] = float(analysis.get('execution_quality_score', {}).get('mean', 0.0) if isinstance(analysis.get('execution_quality_score'), dict) else analysis.get('execution_quality_score', 0.0))
    
    metrics['failure_point_count'] = float(analysis.get('failure_point_count', {}).get('mean', 0.0) if isinstance(analysis.get('failure_point_count'), dict) else analysis.get('failure_point_count', 0.0))
    
    # AUC Loss Calculation
    # AUC = Integral of (1 - P(t)) dt
    progress_events = entry.get('state_encoder', {}).get('progress_events', [])
    metrics['AUC_loss'] = calculate_auc_loss(progress_events, metrics['sim_step_count'])
    
    # P_cliff Calculation (Cliff Probability)
    # Detect if progress stuck for long time then failed?
    # Proxy: If failure_point_count > threshold AND success = 0
    metrics['P_cliff_proxy'] = 1.0 if (metrics['failure_point_count'] > 5 and metrics['task_success'] < 1.0) else 0.0
    
    # GD (Degradation Slope)
    # Slope of performance drop under noise? 
    # Without explicit noise experiments, we use failure rate intensity
    metrics['GD_proxy'] = metrics['failure_point_count'] / (metrics['sim_step_count'] + 1e-6)

    return metrics

def calculate_auc_loss(progress_events, total_steps):
    """
    Calculate Area Under the Curve for Resilience Loss.
    Loss = 1 - Performance(t)
    AUC_loss = Integral(1 - P(t)) dt check range [0, total_steps]
    """
    if not progress_events or total_steps <= 0:
        return 1.0 # Max normalized loss
        
    # Parse events
    # Format: "step <int> -> <float>"
    events = []
    for evt in progress_events:
        try:
            if isinstance(evt, str):
                parts = evt.split('->')
                step = int(parts[0].replace('step', '').strip())
                val = float(parts[1].strip())
                events.append((step, val))
        except:
            continue
    
    if not events:
        return 1.0
        
    events.sort(key=lambda x: x[0])
    
    # Add start and end points
    if events[0][0] > 0:
        events.insert(0, (0, 0.0))
    if events[-1][0] < total_steps:
        events.append((total_steps, events[-1][1]))
        
    auc_perf = 0.0
    for i in range(len(events) - 1):
        t0, p0 = events[i]
        t1, p1 = events[i+1]
        dt = t1 - t0
        # Step function assumption
        auc_perf += p0 * dt
        
    total_area = total_steps * 1.0
    auc_loss = total_area - auc_perf
    
    # Normalized resilience loss (0 to 1)
    if total_steps > 0:
        return auc_loss / total_steps
    return 1.0
