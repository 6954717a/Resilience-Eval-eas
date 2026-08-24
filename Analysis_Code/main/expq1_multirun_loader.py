"""
expq1_multirun_loader.py
========================
Load and merge data from all 4 experimental runs for Exp-Q1.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"d:\Proj\Framework\Data_Exper")
OUT_DIR = BASE_DIR / "Analysis"
TABLE_DIR = OUT_DIR / "tables"
IMG_DIR = OUT_DIR / "images"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "Qwen3-8B":     BASE_DIR / "2026-04-20_12-33-13-val_mini.json" / "results" / "expq1",
    "Qwen3.5-9B":   BASE_DIR / "2026-04-20_15-09-51-val_mini.json" / "results" / "expq1",
    "Qwen2.5-3B-r1": BASE_DIR / "Exp0428" / "2026-04-27_09-53-39-val_mini.json" / "results" / "expq1",
    "Qwen2.5-3B-r2": BASE_DIR / "Exp0428" / "2026-04-27_15-52-44-val_mini.json" / "results" / "expq1",
}

def load_all_rebound(filter_baseline=True):
    dfs = []
    for model, path in RUNS.items():
        csv = path / "rebound_validity" / "raw_rollouts.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["model"] = model
            if filter_baseline:
                df = df[df["rebound_baseline_loaded"] == 1.0]
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out["c_rec"] = out["rebound_c_rec"]
    out["c_rec_cog"] = out["rebound_c_rec_cog"]
    out["c_rec_phy"] = out["rebound_c_rec_phy"]
    out["c_rec_debt"] = out["rebound_c_rec_state_debt"]
    out["is_success"] = (out["task_percent_complete"] == 1.0).astype(int)
    out["is_perturbed"] = (out["perturbation_type"] != "clean").astype(int)
    return out

def load_all_stability():
    dfs = []
    for model, path in RUNS.items():
        csv = path / "stability_validity" / "raw_rollouts.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["model"] = model
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out["beta_neigh"] = out["stability_beta_neighborhood"]
    out["beta_vv"] = out["stability_beta_vv"]
    out["beta_out"] = out["stability_beta_out"]
    out["is_success"] = (out["task_percent_complete"] == 1.0).astype(int)
    return out

def load_all_ge():
    dfs = []
    for model, path in RUNS.items():
        csv = path / "ge_validity" / "raw_rollouts.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["model"] = model
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    r = load_all_rebound()
    s = load_all_stability()
    g = load_all_ge()
    print(f"Rebound: {len(r)} rows, models={list(r['model'].unique())}")
    print(f"Stability: {len(s)} rows, models={list(s['model'].unique())}")
    print(f"GE: {len(g)} rows, models={list(g['model'].unique()) if len(g) else 'none'}")
