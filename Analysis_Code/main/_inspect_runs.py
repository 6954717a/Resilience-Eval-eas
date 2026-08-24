"""Quick inspection of all 4 experimental runs."""
import pandas as pd
import os

runs = {
    "Qwen3-8B": r"d:\Proj\Framework\Data_Exper\2026-04-20_12-33-13-val_mini.json\results\expq1",
    "Qwen3.5-9B": r"d:\Proj\Framework\Data_Exper\2026-04-20_15-09-51-val_mini.json\results\expq1",
    "Qwen2.5-3B-r1": r"d:\Proj\Framework\Data_Exper\Exp0428\2026-04-27_09-53-39-val_mini.json\results\expq1",
    "Qwen2.5-3B-r2": r"d:\Proj\Framework\Data_Exper\Exp0428\2026-04-27_15-52-44-val_mini.json\results\expq1",
}

for name, path in runs.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for sub in ["rebound_validity", "stability_validity", "ge_validity"]:
        rr = os.path.join(path, sub, "raw_rollouts.csv")
        if os.path.exists(rr):
            df = pd.read_csv(rr)
            ptypes = list(df["perturbation_type"].unique())
            bl = df.get("rebound_baseline_loaded", pd.Series([]))
            tc = df["task_percent_complete"]
            success = df.get("task_state_success", pd.Series([]))
            
            print(f"\n  {sub}: {df.shape[0]} rows")
            print(f"    perturbation_type: {ptypes}")
            print(f"    task_percent_complete: mean={tc.mean():.3f}, n_full={int((tc==1.0).sum())}")
            if not success.empty:
                print(f"    task_state_success: mean={success.mean():.3f}, n_success={int(success.sum())}")
            if not bl.empty:
                bl_counts = bl.value_counts().to_dict()
                print(f"    baseline_loaded: {bl_counts}")
            
            # Key metrics
            if "rebound_c_rec" in df.columns:
                valid = df[df.get("rebound_baseline_loaded", 1) == 1.0]
                if len(valid) > 0:
                    print(f"    C_rec: mean={valid['rebound_c_rec'].mean():.2f}, std={valid['rebound_c_rec'].std():.2f}")
                    print(f"    C_rec_cog: mean={valid['rebound_c_rec_cog'].mean():.2f}")
                    print(f"    C_rec_phy: mean={valid['rebound_c_rec_phy'].mean():.2f}")
            if "stability_beta_neighborhood" in df.columns:
                print(f"    beta_neigh: mean={df['stability_beta_neighborhood'].mean():.4f}")
                print(f"    beta_out: mean={df['stability_beta_out'].mean():.4f}")
            if "stress_lambda" in df.columns:
                print(f"    stress_lambda levels: {sorted(df['stress_lambda'].unique())}")
                if "phase_label" in df.columns:
                    print(f"    phase_labels: {list(df['phase_label'].unique())}")
        else:
            print(f"\n  {sub}: NOT FOUND")
