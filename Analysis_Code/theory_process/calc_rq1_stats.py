import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('temp_modify/Analyse/Output/2026-01-18_Evolve_0/L1_episode_metrics.csv')

# 1. Correlation: B_epi vs Task Completion
corr_b_comp = df['B_epi'].corr(df['task_percent_complete'])
print(f"Correlation B_epi vs Completion: {corr_b_comp:.4f}")

# 2. Episode Comparison Stats (SR=1 episodes only)
success_df = df[df['task_success'] == 1.0]
# High Cost (>200) vs Low Cost (<50)
high_cost = success_df[success_df['B_epi'] > 200]
low_cost = success_df[success_df['B_epi'] < 50]

print(f"SR=1 High Cost (>200) count: {len(high_cost)}")
print(f"SR=1 Low Cost (<50) count: {len(low_cost)}")
print(f"SR=1 High Cost Mean Replanning: {high_cost['replanning_count'].mean():.2f}")
print(f"SR=1 Low Cost Mean Replanning: {low_cost['replanning_count'].mean():.2f}")
print(f"SR=1 High Cost Mean Runtime: {high_cost['runtime'].mean():.2f}")
print(f"SR=1 Low Cost Mean Runtime: {low_cost['runtime'].mean():.2f}")

# 3. Safety Discrimination
safe_group = df[df['is_safe_proxy'] == 1]
unsafe_group = df[df['is_safe_proxy'] == 0]

print(f"Safe Group Count: {len(safe_group)}")
print(f"Unsafe Group Count: {len(unsafe_group)}")
print(f"Safe Group P_cliff Mean: {safe_group['P_cliff_proxy'].mean():.4f}")
print(f"Unsafe Group P_cliff Mean: {unsafe_group['P_cliff_proxy'].mean():.4f}")
print(f"Safe Group B_epi Mean: {safe_group['B_epi'].mean():.2f}")
print(f"Unsafe Group B_epi Mean: {unsafe_group['B_epi'].mean():.2f}")

# 4. Predictive Validity (Cliff Prediction)
# Thresholds: sigma_V > 0.05 and replanning_count > 40
high_risk = df[(df['sigma_V_proxy'] > 0.05) & (df['replanning_count'] > 40)]
low_risk = df[~((df['sigma_V_proxy'] > 0.05) & (df['replanning_count'] > 40))]

print(f"High Risk Group Count: {len(high_risk)}")
print(f"High Risk Cliff Prob: {high_risk['P_cliff_proxy'].mean():.4f}")
print(f"Low Risk Cliff Prob: {low_risk['P_cliff_proxy'].mean():.4f}")

# 5. Predictive Validity (B_epi vs Final Completion)
# Using total B_epi as proxy for "early" signals (assuming consistency)
high_b = df[df['B_epi'] > 100]
low_b = df[df['B_epi'] < 50]

print(f"B_epi > 100 Mean Completion: {high_b['task_percent_complete'].mean():.4f}")
print(f"B_epi < 50 Mean Completion: {low_b['task_percent_complete'].mean():.4f}")

# Check specific episodes
ep139 = df[df['episode_id'] == 139].iloc[0]
ep141 = df[df['episode_id'] == 141].iloc[0]
print(f"Ep 139: B_epi={ep139['B_epi']}, Replan={ep139['replanning_count']}, Runtime={ep139['runtime']:.2f}")
print(f"Ep 141: B_epi={ep141['B_epi']}, Replan={ep141['replanning_count']}, Runtime={ep141['runtime']:.2f}")
