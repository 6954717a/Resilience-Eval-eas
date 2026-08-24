"""Deep dive: discriminative analysis for SR=100% episodes across all runs."""
import pandas as pd
import numpy as np
import os

runs = {
    "Qwen3-8B": r"d:\Proj\Framework\Data_Exper\2026-04-20_12-33-13-val_mini.json\results\expq1",
    "Qwen3.5-9B": r"d:\Proj\Framework\Data_Exper\2026-04-20_15-09-51-val_mini.json\results\expq1",
    "Qwen2.5-3B-r1": r"d:\Proj\Framework\Data_Exper\Exp0428\2026-04-27_09-53-39-val_mini.json\results\expq1",
    "Qwen2.5-3B-r2": r"d:\Proj\Framework\Data_Exper\Exp0428\2026-04-27_15-52-44-val_mini.json\results\expq1",
}

all_data = []
for name, path in runs.items():
    for sub in ["rebound_validity", "stability_validity"]:
        rr = os.path.join(path, sub, "raw_rollouts.csv")
        if os.path.exists(rr):
            df = pd.read_csv(rr)
            df["model"] = name
            df["source_sub"] = sub
            all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Filter baseline_loaded == 1
df_valid = df_all[df_all["rebound_baseline_loaded"] == 1.0].copy()

print(f"Total valid rows: {len(df_valid)}")
print(f"Models: {df_valid['model'].unique()}")
print()

# --- Discriminative analysis: SR=100% episodes ---
success_eps = df_valid[df_valid["task_percent_complete"] == 1.0].copy()
print(f"=== Episodes with task_percent_complete == 1.0: {len(success_eps)} ===")
print()

# C_rec variation within SR=100%
for model in sorted(success_eps["model"].unique()):
    sub = success_eps[success_eps["model"] == model]
    print(f"  {model}: n={len(sub)}")
    print(f"    C_rec:     mean={sub['rebound_c_rec'].mean():.2f}, std={sub['rebound_c_rec'].std():.2f}, min={sub['rebound_c_rec'].min():.2f}, max={sub['rebound_c_rec'].max():.2f}")
    print(f"    C_rec_cog: mean={sub['rebound_c_rec_cog'].mean():.2f}")
    print(f"    C_rec_phy: mean={sub['rebound_c_rec_phy'].mean():.2f}")
    # By perturbation type
    for pt in sorted(sub["perturbation_type"].unique()):
        ps = sub[sub["perturbation_type"] == pt]
        print(f"      {pt}: n={len(ps)}, C_rec mean={ps['rebound_c_rec'].mean():.2f}, range=[{ps['rebound_c_rec'].min():.2f}, {ps['rebound_c_rec'].max():.2f}]")
    print()

# --- Overall: clean vs perturbed for SR=100% ---
print("=== Clean vs Perturbed (SR=100% only) ===")
clean100 = success_eps[success_eps["perturbation_type"] == "clean"]
pert100 = success_eps[success_eps["perturbation_type"] != "clean"]
print(f"  Clean (SR=100%): n={len(clean100)}, C_rec mean={clean100['rebound_c_rec'].mean():.2f}, std={clean100['rebound_c_rec'].std():.2f}")
print(f"  Perturbed (SR=100%): n={len(pert100)}, C_rec mean={pert100['rebound_c_rec'].mean():.2f}, std={pert100['rebound_c_rec'].std():.2f}")

from scipy import stats
if len(clean100) >= 3 and len(pert100) >= 3:
    u_stat, p_mw = stats.mannwhitneyu(clean100["rebound_c_rec"], pert100["rebound_c_rec"], alternative="two-sided")
    print(f"  Mann-Whitney U test: U={u_stat:.1f}, p={p_mw:.4f}")
    cd = (u_stat / (len(clean100) * len(pert100))) * 2 - 1  # Cliff's delta
    print(f"  Cliff's delta: {cd:.3f}")

print()

# --- Beta stability for SR=100% ---
print("=== Stability Beta for SR=100% Episodes ===")
stab_success = success_eps.dropna(subset=["stability_beta_neighborhood"])
for model in sorted(stab_success["model"].unique()):
    sub = stab_success[stab_success["model"] == model]
    print(f"  {model}: n={len(sub)}, beta_neigh mean={sub['stability_beta_neighborhood'].mean():.4f}, beta_out mean={sub['stability_beta_out'].mean():.4f}")

print()

# --- CV (Coefficient of Variation) of C_rec within SR=100% ---
print("=== Coefficient of Variation of C_rec within SR=100% ===")
print(f"  Overall CV: {success_eps['rebound_c_rec'].std() / (success_eps['rebound_c_rec'].mean() + 1e-9) * 100:.1f}%")
for model in sorted(success_eps["model"].unique()):
    sub = success_eps[success_eps["model"] == model]
    cv = sub["rebound_c_rec"].std() / (sub["rebound_c_rec"].mean() + 1e-9) * 100
    print(f"  {model}: CV={cv:.1f}%")

print()

# --- Rebound stats across all episodes ---
print("=== Overall Rebound Statistics (all episodes, baseline_loaded=1) ===")
for model in sorted(df_valid["model"].unique()):
    sub = df_valid[df_valid["model"] == model]
    for pt in sorted(sub["perturbation_type"].unique()):
        ps = sub[sub["perturbation_type"] == pt]
        print(f"  {model}/{pt}: n={len(ps)}, C_rec={ps['rebound_c_rec'].mean():.2f}+/-{ps['rebound_c_rec'].std():.2f}, "
              f"TC={ps['task_percent_complete'].mean():.3f}")
