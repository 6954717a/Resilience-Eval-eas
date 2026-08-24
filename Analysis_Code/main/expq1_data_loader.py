"""
expq1_data_loader.py
====================
Unified data-loading and cleaning module for Exp-Q1 resilience-metrics analysis.

Data source: Exp0428/2026-04-27_09-53-39-val_mini.json
Subexperiments: rebound_validity, stability_validity, ge_validity
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"d:\Proj\Framework\Data_Exper")
DATA_ROOT = BASE_DIR / "Exp0428" / "2026-04-27_09-53-39-val_mini.json" / "results" / "expq1"
OUT_DIR = BASE_DIR / "Analysis"
TABLE_DIR = OUT_DIR / "tables"
IMG_DIR = OUT_DIR / "images"

MODEL_LABEL = "Qwen2.5-3B-Instruct"  # model used in this run

# Ensure output dirs
TABLE_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _completion_bucket(pc: float, n_bins: int = 5) -> str:
    """Map task_percent_complete ∈ [0,1] to a readable bucket label."""
    if pd.isna(pc):
        return "N/A"
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if pc <= hi + 1e-9:
            return f"[{lo:.1f},{hi:.1f}]"
    return f"[{edges[-2]:.1f},{edges[-1]:.1f}]"


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_rebound(filter_baseline: bool = True) -> pd.DataFrame:
    """Load rebound_validity/raw_rollouts.csv with cleaning."""
    csv = DATA_ROOT / "rebound_validity" / "raw_rollouts.csv"
    df = pd.read_csv(csv)
    df["model"] = MODEL_LABEL

    if filter_baseline:
        df = df[df["rebound_baseline_loaded"] == 1.0].copy()

    # Completion bucket
    df["completion_bucket"] = df["task_percent_complete"].apply(_completion_bucket)

    # Cliff tag: degradation_p_cliff == 1 → cliff episode
    df["is_cliff"] = (df["degradation_p_cliff"] == 1.0).astype(int)

    # Total recovery cost decomposition helper
    df["c_rec_total"] = df["rebound_c_rec"]
    df["c_rec_cog"] = df["rebound_c_rec_cog"]
    df["c_rec_phy"] = df["rebound_c_rec_phy"]
    df["c_rec_debt"] = df["rebound_c_rec_state_debt"]

    return df


def load_stability() -> pd.DataFrame:
    """Load stability_validity/raw_rollouts.csv with cleaning, including supplementary beta runs."""
    dfs = []
    
    # 1. New data (2026-04-30_16-55-01) - Complete beta run
    p30 = BASE_DIR / "Exp0430" / "Exp_Beta" / "2026-04-30_16-55-01-val_mini.json" / "results" / "expq1_beta" / "stability_execution_validity" / "raw_rollouts.csv"
    if p30.exists():
        df = pd.read_csv(p30)
        df["model"] = MODEL_LABEL
        
        # Apply specific beta aliases
        if "stability_beta_perturbation" in df.columns:
            df["beta_neigh"] = df["stability_beta_perturbation"]
            df["beta_vv"] = df.get("stability_beta_perturbation_vv", df.get("stability_beta_vv", 0.0))
            df["beta_out"] = df.get("stability_beta_perturbation_out", df.get("stability_beta_out", 0.0))
            df["beta_metric_scope"] = "perturbation_specific"

        # 2. Extract intrinsic clean baseline from old Exp0428 data
        old_csv = BASE_DIR / "Exp0428" / "2026-04-27_09-53-39-val_mini.json" / "results" / "expq1" / "stability_validity" / "raw_rollouts.csv"
        if old_csv.exists():
            old_df = pd.read_csv(old_csv)
            old_clean = old_df[old_df["perturbation_type"] == "clean"].set_index("anchor_id")
            
            # Map old baseline values to new "clean" rows
            clean_mask = df["perturbation_type"] == "clean"
            for idx, row in df[clean_mask].iterrows():
                anchor = row["anchor_id"]
                if anchor in old_clean.index:
                    df.at[idx, "beta_neigh"] = old_clean.at[anchor, "stability_beta_neighborhood"]
                    df.at[idx, "beta_vv"] = old_clean.at[anchor, "stability_beta_vv"]
                    df.at[idx, "beta_out"] = old_clean.at[anchor, "stability_beta_out"]
                else:
                    # Fallback to mean if specific anchor missing
                    df.at[idx, "beta_neigh"] = old_clean["stability_beta_neighborhood"].mean()
                    df.at[idx, "beta_vv"] = old_clean["stability_beta_vv"].mean()
                    df.at[idx, "beta_out"] = old_clean["stability_beta_out"].mean()

        df["beta_overall"] = df.get("stability_beta", 0.0)
        return df

    return pd.DataFrame()


def load_ge() -> pd.DataFrame:
    """Load GE data from supplementary experiments (Exp0430)."""
    import glob
    dfs = []
    
    # 1. New data (2026-04-30) - Contains fully calculated raw_rollouts.csv
    p30 = BASE_DIR / "Exp0430" / "2026-04-30_04-07-07-val_mini.json" / "results" / "expq1_ge" / "ge_capacity_validity" / "raw_rollouts.csv"
    family_map = {}
    if p30.exists():
        df30 = pd.read_csv(p30)
        family_map.update(dict(zip(df30['episode_id'], df30['task_family'])))
        dfs.append(df30)
        
    # 2. New data (2026-04-29) - Incomplete, load from episode_metrics.csv
    p29 = BASE_DIR / "Exp0430" / "2026-04-29_16-36-05-val_mini.json"
    eps29 = glob.glob(str(p29 / "**" / "episode_metrics.csv"), recursive=True)
    for ep in eps29:
        df29 = pd.read_csv(ep)
        # Extract lambda from path
        if "lambda" in ep:
            df29["stress_lambda"] = float(ep.split("lambda")[1][:4])
        else:
            df29["stress_lambda"] = 0.0
        df29["phase_label"] = "stress_sweep" if "stress_sweep" in ep else "clean_reference"
        df29["task_family"] = df29["episode_id"].map(lambda x: family_map.get(x, "unknown"))
        dfs.append(df29)

    # 3. Old data (fallback if needed, but we rely on new supplementary data primarily)
    csv = DATA_ROOT / "ge_validity" / "raw_rollouts.csv"
    if csv.exists():
        old_df = pd.read_csv(csv)
        dfs.append(old_df)

    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    df["model"] = MODEL_LABEL

    # Mark formal margin availability (new runs have valid margins != -1.0)
    if "ge_contract_margin_min" in df.columns:
        df["margin_available"] = True
    else:
        df["margin_available"] = False
        df["ge_contract_margin_min"] = -1.0

    # Proxy degradation: use task_percent_complete directly
    df["proxy_completion"] = df["task_percent_complete"]
    df["proxy_recovery_cost"] = df["rebound_c_rec"]

    return df


def load_ge_capacity_summary() -> pd.DataFrame:
    csv = DATA_ROOT / "ge_validity" / "capacity_summary.csv"
    return pd.read_csv(csv)


def load_ge_margin_cells() -> pd.DataFrame:
    csv = DATA_ROOT / "ge_validity" / "margin_cells_summary.csv"
    return pd.read_csv(csv)


# ── Aggregation ──────────────────────────────────────────────────────────────
def aggregate_rebound_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rebound cost stats by completion bucket."""
    agg = df.groupby("completion_bucket").agg(
        n=("c_rec_total", "count"),
        c_rec_mean=("c_rec_total", "mean"),
        c_rec_std=("c_rec_total", "std"),
        c_rec_median=("c_rec_total", "median"),
        c_rec_cog_mean=("c_rec_cog", "mean"),
        c_rec_phy_mean=("c_rec_phy", "mean"),
        c_rec_debt_mean=("c_rec_debt", "mean"),
        completion_mean=("task_percent_complete", "mean"),
    ).reset_index()
    return agg


def aggregate_rebound_paired(df: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison: clean vs each perturbation type per anchor."""
    clean = df[df["perturbation_type"] == "clean"][["anchor_id", "c_rec_total", "c_rec_cog", "c_rec_phy", "c_rec_debt", "task_percent_complete"]]
    clean = clean.rename(columns={c: f"{c}_clean" for c in clean.columns if c != "anchor_id"})

    rows = []
    for ptype in df["perturbation_type"].unique():
        if ptype == "clean":
            continue
        pert = df[df["perturbation_type"] == ptype][["anchor_id", "c_rec_total", "c_rec_cog", "c_rec_phy", "c_rec_debt", "task_percent_complete"]]
        pert = pert.rename(columns={c: f"{c}_pert" for c in pert.columns if c != "anchor_id"})
        merged = pd.merge(clean, pert, on="anchor_id", how="inner")
        merged["perturbation_type"] = ptype
        rows.append(merged)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def aggregate_stability_by_perturbation(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stability beta stats by perturbation type."""
    agg = df.groupby("perturbation_type").agg(
        n=("beta_neigh", "count"),
        beta_neigh_mean=("beta_neigh", "mean"),
        beta_neigh_std=("beta_neigh", "std"),
        beta_neigh_median=("beta_neigh", "median"),
        beta_vv_mean=("beta_vv", "mean"),
        beta_out_mean=("beta_out", "mean"),
        beta_out_std=("beta_out", "std"),
        completion_mean=("task_percent_complete", "mean"),
    ).reset_index()
    return agg


def aggregate_ge_stress_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GE metrics by stress_lambda (across task families)."""
    sweep = df[df["phase_label"] == "stress_sweep"].copy()
    agg = df.groupby("stress_lambda").agg(
        n=("proxy_completion", "count"),
        completion_mean=("proxy_completion", "mean"),
        completion_std=("proxy_completion", "std"),
        completion_se=("proxy_completion", lambda x: x.std() / np.sqrt(len(x))),
        c_rec_mean=("proxy_recovery_cost", "mean"),
        c_rec_std=("proxy_recovery_cost", "std"),
        c_rec_se=("proxy_recovery_cost", lambda x: x.std() / np.sqrt(len(x))),
        margin_mean=("ge_contract_margin_min", "mean"),
        margin_se=("ge_contract_margin_min", lambda x: x.std() / np.sqrt(len(x))),
        boundary_hit_mean=("ge_boundary_hit", "mean") if "ge_boundary_hit" in df.columns else ("task_state_success", "mean"),
        hard_boundary_hit_mean=("ge_hard_boundary_hit", "mean") if "ge_hard_boundary_hit" in df.columns else ("task_state_success", "mean"),
        success_rate=("task_state_success", "mean"),
    ).reset_index()
    return agg


def aggregate_ge_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GE stress sweep by task_family × stress_lambda."""
    sweep = df[df["phase_label"] == "stress_sweep"].copy()
    agg = sweep.groupby(["task_family", "stress_lambda"]).agg(
        n=("proxy_completion", "count"),
        completion_mean=("proxy_completion", "mean"),
        c_rec_mean=("proxy_recovery_cost", "mean"),
        margin_mean=("ge_contract_margin_min", "mean"),
        success_rate=("task_state_success", "mean"),
    ).reset_index()
    return agg


# ── Export ───────────────────────────────────────────────────────────────────
def export_all_tables():
    """Export all aggregated tables to Analysis/tables/."""
    df_reb = load_rebound()
    df_stab = load_stability()
    df_ge = load_ge()

    # Rebound
    agg_bucket = aggregate_rebound_by_bucket(df_reb)
    agg_bucket.to_csv(TABLE_DIR / "expq1_rebound_by_completion_bucket.csv", index=False)

    agg_paired = aggregate_rebound_paired(df_reb)
    agg_paired.to_csv(TABLE_DIR / "expq1_rebound_paired_comparison.csv", index=False)

    # Stability
    agg_stab = aggregate_stability_by_perturbation(df_stab)
    agg_stab.to_csv(TABLE_DIR / "expq1_stability_by_perturbation.csv", index=False)

    # GE
    agg_ge = aggregate_ge_stress_sweep(df_ge)
    agg_ge.to_csv(TABLE_DIR / "expq1_ge_stress_sweep.csv", index=False)

    agg_ge_fam = aggregate_ge_by_family(df_ge)
    agg_ge_fam.to_csv(TABLE_DIR / "expq1_ge_by_family.csv", index=False)

    # Global summary
    summary = {
        "rebound_n": len(df_reb),
        "rebound_c_rec_mean": df_reb["c_rec_total"].mean(),
        "rebound_c_rec_std": df_reb["c_rec_total"].std(),
        "rebound_cliff_pct": df_reb["is_cliff"].mean() * 100,
        "stability_n": len(df_stab),
        "stability_beta_neigh_mean": df_stab["beta_neigh"].mean(),
        "stability_beta_out_mean": df_stab["beta_out"].mean(),
        "ge_n": len(df_ge),
        "ge_stress_levels": sorted(df_ge["stress_lambda"].unique().tolist()),
        "ge_task_families": sorted(df_ge["task_family"].unique().tolist()),
        "ge_margin_available": df_ge["margin_available"].any(),
    }
    pd.DataFrame([summary]).to_csv(TABLE_DIR / "expq1_aggregated_stats.csv", index=False)

    print(f"[export] Wrote {len(os.listdir(TABLE_DIR))} table files to {TABLE_DIR}")
    return summary


if __name__ == "__main__":
    summary = export_all_tables()
    for k, v in summary.items():
        print(f"  {k}: {v}")
