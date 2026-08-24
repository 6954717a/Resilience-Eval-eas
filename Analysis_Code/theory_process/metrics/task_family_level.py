"""
Task-Family-Level (L2) Metrics Computation

实现 Detailed_Metrics.tex 中 Task-Family 层的统计聚合指标:
- MTTR-A ± CI_95%
- MTBF-A ± CI_95%
- E[RR] (期望恢复率)
- P^F_cliff (任务族断崖概率)
- S_F (安全评分)
- β (算法稳定性)
- LES (局部弹性稳定性代理)
- Rademacher 复杂度 (如可用)
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple

# 尝试导入 scipy 用于 CI 计算
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 尝试导入 Rademacher 分析模块
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from Analyse.Code.theory.rademacher_complexity import RademacherComplexityAnalyzer
except ImportError:
    RademacherComplexityAnalyzer = None


def compute_ci_95(values: np.ndarray) -> Tuple[float, float]:
    """
    计算 95% 置信区间 (使用 t 分布)
    
    Returns:
        (lower, upper) 置信区间边界
    """
    values = np.array(values)
    values = values[~np.isnan(values)]
    n = len(values)
    
    if n < 2:
        return (0.0, 0.0)
    
    mean = np.mean(values)
    
    if SCIPY_AVAILABLE:
        sem = stats.sem(values)
        ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)
        return (float(ci[0]), float(ci[1]))
    else:
        # 简化版: 使用 1.96 * std / sqrt(n)
        std = np.std(values, ddof=1)
        margin = 1.96 * std / np.sqrt(n)
        return (mean - margin, mean + margin)


def compute_task_family_metrics(episode_df: pd.DataFrame) -> Dict[str, Any]:
    """
    计算 L2 (Task-Family-Level) 聚合指标
    
    Args:
        episode_df: 包含各 episode L1 指标的 DataFrame
        
    Returns:
        L2 指标字典
    """
    metrics = {}
    
    if episode_df.empty:
        return metrics
    
    n = len(episode_df)
    metrics['L2_Episode_Count'] = n
        
    # ========== 1. Rebound 聚合 ==========
    
    # MTTR-A with CI
    if 'T_rec' in episode_df.columns:
        values = episode_df['T_rec'].dropna().values
        if len(values) > 0:
            metrics['L2_MTTR'] = float(np.mean(values))
            metrics['L2_MTTR_Std'] = float(np.std(values))
            ci = compute_ci_95(values)
            metrics['L2_MTTR_CI_95_Low'] = ci[0]
            metrics['L2_MTTR_CI_95_High'] = ci[1]
    
    # MTBF-A with CI
    if 'rebound_mtbf' in episode_df.columns:
        values = episode_df['rebound_mtbf'].dropna().values
        if len(values) > 0:
            metrics['L2_MTBF'] = float(np.mean(values))
            metrics['L2_MTBF_Std'] = float(np.std(values))
            ci = compute_ci_95(values)
            metrics['L2_MTBF_CI_95_Low'] = ci[0]
            metrics['L2_MTBF_CI_95_High'] = ci[1]
    else:
        metrics['L2_MTBF'] = 0.0
    
    # E[RR] - 期望恢复率 (Algorithm 1, Line 15)
    if 'recovery_ratio' in episode_df.columns:
        values = episode_df['recovery_ratio'].dropna().values
        if len(values) > 0:
            metrics['L2_E_RR'] = float(np.mean(values))
            metrics['L2_E_RR_Std'] = float(np.std(values))
            ci = compute_ci_95(values)
            metrics['L2_E_RR_CI_95_Low'] = ci[0]
            metrics['L2_E_RR_CI_95_High'] = ci[1]
    else:
        metrics['L2_E_RR'] = 0.0
        
    # ========== 2. Stability 聚合 ==========
    
    if 'execution_quality' in episode_df.columns:
        metrics['L2_Stability_Var'] = float(episode_df['execution_quality'].std())
    
    # β 稳定性 (算法稳定性估计)
    metrics['L2_Beta_Stability'] = metrics.get('L2_Stability_Var', 0) / np.sqrt(n)
    
    # LES 代理
    metrics['L2_LES_Proxy'] = compute_les_proxy(episode_df)
    
    # ========== 3. Degradation 聚合 ==========
    
    # S_F (安全评分)
    metrics['L2_Safety_Score'] = compute_safety_score(episode_df)
    
    # P^F_cliff - 任务族断崖概率
    if 'P_cliff_proxy' in episode_df.columns:
        metrics['L2_P_cliff'] = float(episode_df['P_cliff_proxy'].mean())
    elif 'cliff_detected' in episode_df.columns:
        metrics['L2_P_cliff'] = float(episode_df['cliff_detected'].mean())
    else:
        metrics['L2_P_cliff'] = 0.0
    
    # ========== 4. 理论复杂度 ==========
    
    if RademacherComplexityAnalyzer and 'execution_quality' in episode_df.columns:
        try:
            vals = episode_df['execution_quality'].dropna().values
            if len(vals) > 10:
                analyzer = RademacherComplexityAnalyzer(bootstrap_samples=50)
                complexity = analyzer.analyze_all(vals)
                gen_bound = analyzer.compute_generalization_bound(complexity)
                
                metrics['L2_Empirical_RC'] = complexity.empirical_rc
                metrics['L2_Offset_RC'] = complexity.offset_rc
                metrics['L2_Generalization_Gap'] = gen_bound['bound']
        except Exception as e:
            print(f"Warning: Rademacher analysis failed: {e}")
            metrics['L2_Generalization_Gap'] = 0.0
    else:
        metrics['L2_Generalization_Gap'] = 0.0
            
    return metrics


def compute_les_proxy(df: pd.DataFrame) -> float:
    """
    计算 LES (Locally Elastic Stability) 代理
    
    通过任务类型分组，计算组内 execution_quality 方差的均值
    高方差 = 低稳定性 = 高 LES
    """
    if 'instruction' not in df.columns or 'execution_quality' not in df.columns:
        return 0.0
        
    def get_task_type(instr):
        if not isinstance(instr, str):
            return 'other'
        instr = instr.lower()
        if 'move' in instr or 'navigate' in instr: return 'navigate'
        if 'place' in instr: return 'place'
        if 'find' in instr: return 'find'
        if 'pick' in instr: return 'pick'
        return 'other'
        
    df = df.copy()
    df['task_type'] = df['instruction'].apply(get_task_type)
    
    group_vars = df.groupby('task_type')['execution_quality'].apply(
        lambda x: x.var() if len(x) > 1 else 0.0
    )
    
    return float(group_vars.mean()) if not group_vars.isnull().all() else 0.0


def compute_safety_score(df: pd.DataFrame) -> float:
    """
    计算分布安全评分 S_F (Eq.20)
    
    S_F = 1 - (1/N) * Σ I[unsafe_i]
    """
    if df.empty:
        return 0.0
    
    safe_count = 0
    total = len(df)
    
    # 安全标准: B_epi < 5 且 sigma_V < 1.0
    b_epi_threshold = 5
    sigma_v_threshold = 1.0
    
    for _, row in df.iterrows():
        b_epi = row.get('B_epi', row.get('b_epi', 0))
        s_v = row.get('sigma_V_proxy', 0)
        
        if b_epi < b_epi_threshold and s_v < sigma_v_threshold:
            safe_count += 1
            
    return safe_count / total
