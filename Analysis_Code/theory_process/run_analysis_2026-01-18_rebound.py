#!/usr/bin/env python3
"""
指标分析脚本 - 2026-01-18_Analyse_Rebound_0 数据集

Rebound 增强版运行，启用了 context_evolve (batch_size=15)
"""

import os
import sys
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Analyse.Code.metrics.episode_level import extract_episode_metrics
from Analyse.Code.metrics.task_family_level import compute_task_family_metrics
from Analyse.Code.metrics.evolve_level import compute_evolve_metrics


def load_evolve_summaries(data_dir: str):
    """加载所有 evolve summary JSON 文件"""
    evolve_dir = os.path.join(data_dir, "results", "val_mini.json.gz", "analyses", "evolve")
    summaries = []
    
    # 只加载 *_summary.json 文件
    for fp in glob.glob(os.path.join(evolve_dir, "*_summary.json")):
        with open(fp, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                summaries.append(data)
            except:
                pass
    
    return summaries


def filter_anomalous_episodes(summaries: list, step_threshold: float = 15000.0):
    """过滤异常 episode (completion=0 且 steps>=threshold)"""
    valid = []
    failed = []
    
    for ep in summaries:
        completion = float(ep.get('task_percent_complete', 0.0))
        steps = float(ep.get('sim_step_count', 0.0))
        
        if completion == 0.0 and steps >= step_threshold:
            failed.append(ep)
        else:
            valid.append(ep)
    
    return valid, failed


def run_analysis(data_dir: str, output_dir: str, step_threshold: float = 15000.0):
    """运行完整的三层指标分析"""
    print(f"[分析] 数据目录: {data_dir}")
    print(f"[分析] 输出目录: {output_dir}")
    print(f"[分析] 异常过滤阈值: sim_step >= {step_threshold} 且 completion=0")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    summaries = load_evolve_summaries(data_dir)
    print(f"[分析] 加载了 {len(summaries)} 个 episode")
    
    if not summaries:
        print("[错误] 未找到数据")
        return
    
    # 1.5 过滤异常 episode
    valid_episodes, failed_episodes = filter_anomalous_episodes(summaries, step_threshold)
    print(f"[过滤] 有效 episode: {len(valid_episodes)}, 异常/失败 episode: {len(failed_episodes)}")
    
    # 保存失败用例
    if failed_episodes:
        failed_output = os.path.join(output_dir, "failed_episodes.json")
        with open(failed_output, 'w', encoding='utf-8') as f:
            json.dump(failed_episodes, f, indent=2, ensure_ascii=False)
        print(f"[过滤] 失败用例保存到: {failed_output}")
    
    if not valid_episodes:
        print("[警告] 过滤后无有效 episode")
        return
    
    # 2. L1: Episode-Level 指标提取
    print("\n========== L1: Episode-Level 指标 ==========")
    l1_metrics_list = []
    for summary in valid_episodes:
        l1 = extract_episode_metrics(summary)
        l1_metrics_list.append(l1)
    
    l1_df = pd.DataFrame(l1_metrics_list)
    
    l1_output = os.path.join(output_dir, "L1_episode_metrics.csv")
    l1_df.to_csv(l1_output, index=False)
    print(f"[L1] 保存到: {l1_output}")
    
    # L1 汇总
    l1_summary = {
        'Episode Count': len(l1_df),
        'Mean Task Success': float(l1_df['task_success'].mean()) if 'task_success' in l1_df else 0,
        'Mean Task Completion': float(l1_df['task_percent_complete'].mean()) if 'task_percent_complete' in l1_df else 0,
        'Mean T_rec': float(l1_df['T_rec'].mean()) if 'T_rec' in l1_df else 0,
        'Mean B_epi': float(l1_df['B_epi'].mean()) if 'B_epi' in l1_df else 0,
        'Mean Replanning': float(l1_df['replanning_count'].mean()) if 'replanning_count' in l1_df else 0,
    }
    print("\n[L1] 指标汇总:")
    for k, v in l1_summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 3. L2: Task-Family-Level 指标
    print("\n========== L2: Task-Family-Level 指标 ==========")
    l2_metrics = compute_task_family_metrics(l1_df)
    
    l2_output = os.path.join(output_dir, "L2_task_family_metrics.json")
    with open(l2_output, 'w', encoding='utf-8') as f:
        json.dump(l2_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"[L2] 保存到: {l2_output}")
    
    print("\n[L2] 指标汇总:")
    for k, v in l2_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # 4. L3: Evolve-Level 指标
    print("\n========== L3: Evolve-Level 指标 ==========")
    # 固定步数: 15个episode为一步进化 (与运行配置一致)
    episodes_per_evolve = 15
    print(f"[L3] 进化步数配置: 每 {episodes_per_evolve} 个 episode 为一步进化")
    
    evolution_context = {
        'episode_history': {'all': valid_episodes}
    }
    l3_metrics = compute_evolve_metrics(evolution_context, l2_metrics)
    
    l3_output = os.path.join(output_dir, "L3_evolve_metrics.json")
    l3_serializable = {}
    for k, v in l3_metrics.items():
        if isinstance(v, (list, np.ndarray)):
            l3_serializable[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            l3_serializable[k] = float(v)
        else:
            l3_serializable[k] = v
    
    with open(l3_output, 'w', encoding='utf-8') as f:
        json.dump(l3_serializable, f, indent=2, ensure_ascii=False, default=str)
    print(f"[L3] 保存到: {l3_output}")
    
    print("\n[L3] 指标汇总:")
    for k, v in l3_metrics.items():
        if k.endswith('_Curve') or k.endswith('_Perfs'):
            print(f"  {k}: [length={len(v)}]")
        elif isinstance(v, dict):
            print(f"  {k}: <dict>")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # 5. 综合报告
    print("\n========== 生成综合报告 ==========")
    report = {
        'dataset': os.path.basename(data_dir),
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'run_config': {
            'rebound_enabled': True,
            'context_evolve_enabled': True,
            'context_evolve_batch_size': 15,
        },
        'filtered_count': len(failed_episodes),
        'valid_count': len(valid_episodes),
        'L1_summary': l1_summary,
        'L2_metrics': l2_metrics,
        'L3_metrics': l3_serializable,
    }
    
    report_path = os.path.join(output_dir, "metrics_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"[报告] 保存到: {report_path}")
    
    print("\n========== 分析完成 ==========")
    
    return l3_metrics


if __name__ == '__main__':
    data_dir = str(Path(__file__).parent.parent / "Data" / "2026-01-18_Analyse_Rebound_0")
    output_dir = str(Path(__file__).parent.parent / "Output" / "2026-01-18_Analyse_Rebound_0")
    
    l3 = run_analysis(data_dir, output_dir)
    
    if l3:
        print("\n" + "="*50)
        print("L3 EVOLVE-LEVEL 关键指标:")
        print("="*50)
        print(f"  NRR (归一化韧性):     {l3.get('L3_NRR', 0):.4f}")
        print(f"  BWT (后向迁移):       {l3.get('L3_BWT', 0):.4f}")
        print(f"  FWT (前向迁移):       {l3.get('L3_FWT', 0):.4f}")
        print(f"  Retention Rate:       {l3.get('L3_Retention_Rate', 0):.4f}")
        print(f"  Cumulative Regret:    {l3.get('L3_Cumulative_Regret', 0):.4f}")
        print(f"  Initial Perf:         {l3.get('L3_Initial_Perf', 0):.4f}")
        print(f"  Current Perf:         {l3.get('L3_Current_Perf', 0):.4f}")
