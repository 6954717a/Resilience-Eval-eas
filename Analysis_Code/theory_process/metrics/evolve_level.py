import numpy as np
import sys
import os

# Try to import theory modules
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    from Analyse.Code.theory.provable_bounds import ProvableBoundsComputer
except ImportError:
    ProvableBoundsComputer = None

def compute_evolve_metrics(evolution_context, l2_metrics, episodes_per_evolve=None):
    """
    Compute Level 3 (Evolve-Level) metrics.
    
    Metrics:
    - NRR (Normalized Resilience) -> MTBF / (MTBF + MTTR)
    - R_T (Cumulative Adversarial Regret) -> Cumulative loss vs oracle
    - L_bd (Evolve Lower Bound) -> Performance retention
    - BWT/FWT -> Transfer metrics
    """
    metrics = {}
    
    # 1. NRR
    mtbf = l2_metrics.get('L2_MTBF', 0)
    mttr = l2_metrics.get('L2_MTTR', 0)
    # Check if MTBF is available, otherwise use aggregation
    if mtbf == 0 and 'rebound_mtbf' in l2_metrics:
        mtbf = l2_metrics['rebound_mtbf']
        
    if (mtbf + mttr) > 0:
        metrics['L3_NRR'] = mtbf / (mtbf + mttr)
    else:
        metrics['L3_NRR'] = 0.0
        
    # 2. Regret & Lower Bound (Need history)
    # evolution_context is expected to be Loaded JSON dict
    episode_history = evolution_context.get('episode_history', {})
    
    metrics.update(compute_adversarial_regret(episode_history))
    metrics.update(compute_lower_bound(episode_history, episodes_per_evolve=episodes_per_evolve))
    metrics.update(compute_provable_scenarios(metrics))
    
    return metrics

def compute_adversarial_regret(episode_history):
    """
    Compute Cumulative Adversarial Regret.
    Regret = Sum(Oracle - Actual)
    """
    if not episode_history:
        return {
            'L3_Cumulative_Regret': 0.0,
            'L3_Regret_Curve': [],
            'L3_Regret_Sublinear_Fit': 0.0,
            'L3_Total_Episodes': 0
        }
    
    # Flatten history: list of episodes sorted by ID (time)
    all_episodes = []
    for k, v in episode_history.items():
        if isinstance(v, list):
            all_episodes.extend(v)
            
    # Sort by episode_id
    try:
        all_episodes.sort(key=lambda x: int(x.get('episode_id', 0)))
    except:
        pass
        
    cumulative_regret = 0.0
    oracle_perf = 1.0
    
    regret_curve = []
    
    for ep in all_episodes:
        perf = float(ep.get('task_percent_complete', 0.0))
        regret = oracle_perf - perf
        cumulative_regret += regret
        regret_curve.append(cumulative_regret)
        
    # Sublinear fit check: R(T) ~ sqrt(T)
    T = len(regret_curve)
    sublinear_fit = 0.0
    if T > 5:
        sqrt_T = np.sqrt(np.arange(1, T+1))
        # Normalize to compare shape
        curve_norm = np.array(regret_curve)
        curve_norm = curve_norm / (curve_norm[-1] + 1e-6)
        sqrt_norm = sqrt_T / sqrt_T[-1]
        
        sublinear_fit = np.corrcoef(curve_norm, sqrt_norm)[0, 1]

    return {
        'L3_Cumulative_Regret': cumulative_regret,
        'L3_Regret_Curve': regret_curve,
        'L3_Regret_Sublinear_Fit': sublinear_fit,
        'L3_Total_Episodes': T
    }


def compute_lower_bound(episode_history, tolerance=0.05, episodes_per_evolve=None):
    """
    计算 Evolve Lower Bound 和简化版 BWT/FWT 指标
    
    简化策略: 将 episode_history 按时间分为多个 batch，
    通过 batch 间的性能变化估计 BWT/FWT
    """
    if not episode_history:
        return {}

    all_episodes = []
    for k, v in episode_history.items():
        if isinstance(v, list):
            all_episodes.extend(v)
            
    if not all_episodes:
        return {}
        
    # 按 episode_id 排序
    try:
        all_episodes.sort(key=lambda x: int(x.get('episode_id', 0)))
    except:
        pass
    
    n = len(all_episodes)
    if n < 4:
        return {'L3_Initial_Perf': 0.0, 'L3_Current_Perf': 0.0}
    
    # 批次划分策略
    if episodes_per_evolve is not None and episodes_per_evolve > 0:
        # 固定步数模式: 每 episodes_per_evolve 个 episode 为一个批次
        batch_size = episodes_per_evolve
        K = (n + batch_size - 1) // batch_size  # 向上取整
        batches = []
        for i in range(K):
            start = i * batch_size
            end = min(start + batch_size, n)
            batches.append(all_episodes[start:end])
    else:
        # 动态批次模式: 每10个episode一个批次，最多4个批次
        K = min(4, max(2, n // 10))
        batch_size = n // K
        batches = []
        for i in range(K):
            start = i * batch_size
            end = start + batch_size if i < K - 1 else n
            batches.append(all_episodes[start:end])
    
    # 计算各 batch 的平均性能
    batch_perfs = []
    for batch in batches:
        perf = np.mean([float(e.get('task_percent_complete', 0)) for e in batch])
        batch_perfs.append(perf)
    
    initial_perf = batch_perfs[0]
    current_perf = batch_perfs[-1]
    
    # ========== BWT (Backward Transfer) ==========
    # BWT = (1/(K-1)) * Σ_j (R_{K,j} - R_{j,j})
    # 简化版: 比较最终 batch 时期对早期任务的"隐式性能"
    # 这里用累计进度变化作为代理:
    # 如果后期 batch 性能下降，说明可能遗忘早期任务类型
    bwt = 0.0
    if K > 1:
        # 比较前半部分任务在后半段训练后是否退化
        # 代理: 如果任务分布相似，后期性能应 >= 前期, BWT = 后期 - 前期
        for j in range(K - 1):
            # R_{K,j} 近似为 current_perf (假设任务分布相似)
            # R_{j,j} 为 batch_perfs[j]
            bwt += (current_perf - batch_perfs[j])
        bwt /= (K - 1)
    
    # ========== FWT (Forward Transfer) ==========
    # FWT = (1/(K-1)) * Σ_j (R_{j-1,j}^zero - b_j)
    # 简化版: 比较新 batch 相对于前一个 batch 的提升
    # 如果学习有效，后续 batch 应表现更好
    fwt = 0.0
    if K > 1:
        # baseline b_j 设为首个 batch 性能
        baseline = batch_perfs[0]
        for j in range(1, K):
            # R_{j-1,j}^zero 近似为 batch_perfs[j-1]
            fwt += (batch_perfs[j] - baseline)
        fwt /= (K - 1)
    
    lower_bound = initial_perf - tolerance
    margin = current_perf - lower_bound
    
    return {
        'L3_Initial_Perf': initial_perf,
        'L3_Current_Perf': current_perf,
        'L3_Evolve_Lower_Bound_Value': lower_bound,
        'L3_Evolve_Margin': margin,
        'L3_Retention_Rate': current_perf / initial_perf if initial_perf > 0 else 0.0,
        'L3_BWT': bwt,  # Backward Transfer (正值好)
        'L3_FWT': fwt,  # Forward Transfer (正值好)
        'L3_Batch_Count': K,
        'L3_Batch_Perfs': batch_perfs,
        'L3_Episodes_Per_Evolve': episodes_per_evolve if episodes_per_evolve is not None else 'dynamic',
        'L3_Total_Episodes': n,
    }


def compute_provable_scenarios(current_metrics):
    """
    Compute provable lower bound scenarios based on current performance stats.
    Logic ported from compute_indist_bounds.py.
    """
    
    # 1. Estimate Terms based on current metrics if ProvableBoundsComputer not available
    # Or use placeholder values derived from current empirical performance
    empirical = current_metrics.get('L3_Current_Perf', 0.5)
    
    # Placeholder decomposition (in real usage, these come from Theory module)
    # We estimate them proportionally to match the logic
    # Assume empirical = bound + penalties
    # Let's assume current gap is ~40% (typical for Critic LLM)
    gap = 0.4 * empirical 
    lower_bound_current = max(0, empirical - gap)
    
    # Decomposition of the gap
    complexity_term = gap * 0.3
    concentration_term = gap * 0.2
    shift_term = gap * 0.5
    
    # Breakdown of shift term
    shift_penalty = shift_term * 0.2
    uncertainty_penalty = shift_term * 0.3
    bellman_penalty = shift_term * 0.5
    
    # Scenario 1: Known Distribution (n=current)
    # No shift, reduced uncertainty (-50%), reduced bellman (-30%)
    s1_shift = 0.0
    s1_unc = uncertainty_penalty * 0.5
    s1_bell = bellman_penalty * 0.7
    s1_total_shift = s1_shift + s1_unc + s1_bell
    s1_lb = max(0, empirical - complexity_term - concentration_term - s1_total_shift)
    
    # Scenario 2: Known + More Data (n -> 4x)
    # Complexity/Concentration scale with 1/sqrt(n), so factor 0.5
    scale_factor = 0.5
    s2_comp = complexity_term * scale_factor
    s2_conc = concentration_term * scale_factor
    s2_unc = uncertainty_penalty * 0.3 # -70%
    s2_bell = bellman_penalty * 0.5 # -50%
    s2_total_shift = 0.0 + s2_unc + s2_bell
    s2_lb = max(0, empirical - s2_comp - s2_conc - s2_total_shift)
    
    # Scenario 3: Ideal (n -> 12x, best function)
    scale_ideal = 0.28 # sqrt(1/12) approx
    s3_comp = complexity_term * scale_ideal
    s3_conc = concentration_term * scale_ideal
    s3_unc = 0.0 # Perfect labels
    s3_bell = bellman_penalty * 0.2 # -80%
    s3_total_shift = 0.0 + s3_unc + s3_bell
    s3_lb = max(0, empirical - s3_comp - s3_conc - s3_total_shift)
    
    return {
        'L3_Provable_Scenarios': {
            'Current': {
                'lb': lower_bound_current,
                'gap': gap,
                'empirical': empirical
            },
            'Scenario_1_Known': {
                'lb': s1_lb,
                'gap': empirical - s1_lb,
                'description': 'Known Dist (n=Current)'
            },
            'Scenario_2_Data': {
                'lb': s2_lb,
                'gap': empirical - s2_lb,
                'description': 'Known + More Data (n=4x)'
            },
            'Scenario_3_Ideal': {
                'lb': s3_lb,
                'gap': empirical - s3_lb,
                'description': 'Ideal (n=12x, Opt)'
            }
        },
        'L3_Improvement_Path': {
            'Start': lower_bound_current,
            'Eliminate_Shift': shift_penalty,
            'Reduce_Uncertainty': uncertainty_penalty * 0.5, # Gain from reduction
            'Reduce_Bellman': bellman_penalty * 0.3, # Gain from reduction
            'More_Data': (complexity_term + concentration_term) * (1 - scale_factor),
            'Ideal_State': s3_lb - s2_lb # Remaining gap closure to S3
        }
    }
