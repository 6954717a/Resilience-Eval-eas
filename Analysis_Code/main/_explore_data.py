import sys
sys.path.insert(0, r"d:\Proj\Framework\Data_Exper\Analysis\scripts")
from expq1_data_loader import load_ge
from expq1_multirun_loader import load_all_rebound, load_all_stability
import pandas as pd

# Rebound data
r = load_all_rebound()
s_all = r[r["is_success"] == 1]
clean = s_all[s_all["is_perturbed"] == 0]["c_rec"]
pert = s_all[s_all["is_perturbed"] == 1]["c_rec"]
print("=== REBOUND (multirun, SR=100%) ===")
print(f"  Clean: n={len(clean)}, mu={clean.mean():.1f}, std={clean.std():.1f}")
print(f"  Perturbed: n={len(pert)}, mu={pert.mean():.1f}, std={pert.std():.1f}")

# Stability data
stab = load_all_stability()
print("\n=== STABILITY (multirun) ===")
for pt in sorted(stab["perturbation_type"].unique()):
    sub = stab[stab["perturbation_type"] == pt]
    print(f"  {pt}: n={len(sub)}, beta_neigh mu={sub['beta_neigh'].mean():.3f}")

# GE data
ge = load_ge()
valid = ge[(ge["ge_contract_margin_min"] != -1.0) & (ge["ge_contract_margin_min"] != 0.0)]
print(f"\n=== GE (valid margin rows: {len(valid)}) ===")
print(f"  Phases: {list(valid['phase_label'].unique())}")
print(f"  Families: {list(valid['task_family'].unique())}")
sweep = valid[valid["phase_label"] == "stress_sweep"]
clean_ref = valid[valid["phase_label"] == "clean_reference"]
print(f"  Clean ref: n={len(clean_ref)}, margin_mean={clean_ref['ge_contract_margin_min'].mean():.3f}")
for lam in sorted(sweep["stress_lambda"].unique()):
    sub = sweep[sweep["stress_lambda"] == lam]
    print(f"  lambda={lam}: n={len(sub)}, margin={sub['ge_contract_margin_min'].mean():.3f}, crec={sub['rebound_c_rec'].mean():.1f}")

# GE per family
for fam in sorted(valid["task_family"].unique()):
    fam_data = sweep[sweep["task_family"] == fam]
    fam_clean = clean_ref[clean_ref["task_family"] == fam]
    print(f"\n  Family: {fam}")
    print(f"    Clean: n={len(fam_clean)}, margin={fam_clean['ge_contract_margin_min'].mean():.3f}")
    for lam in sorted(fam_data["stress_lambda"].unique()):
        sub = fam_data[fam_data["stress_lambda"] == lam]
        print(f"    lambda={lam}: n={len(sub)}, margin={sub['ge_contract_margin_min'].mean():.3f}")
