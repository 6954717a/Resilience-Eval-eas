"""
run_all_expq1.py
================
Master runner: executes all Exp-Q1 analysis and figure-generation scripts.

Usage:
    python Analysis/scripts/run_all_expq1.py

Outputs:
    - Analysis/tables/  →  aggregated CSVs
    - Analysis/images/  →  5 publication-quality figures (PNG + PDF)
    - Analysis/analysis_manifest.json  →  updated manifest
"""

import sys, os, json, time, traceback
from pathlib import Path

# Add scripts dir to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from expq1_data_loader import export_all_tables, BASE_DIR, OUT_DIR, IMG_DIR, TABLE_DIR


def run_step(name, func):
    """Run a step with timing and error handling."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        func()
        elapsed = time.time() - t0
        print(f"  [OK] {name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] {name} FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  Exp-Q1 Resilience Metrics — Full Analysis Pipeline")
    print("=" * 60)
    print(f"  Data root:   {BASE_DIR / 'Exp0428'}")
    print(f"  Output dir:  {OUT_DIR}")
    print()

    t_start = time.time()
    results = {}

    # Step 0: Export aggregated tables
    results["tables"] = run_step(
        "Step 0: Export aggregated tables",
        export_all_tables,
    )

    # Main compact figure used by the Exp-Q1 paper text.
    from plot_fig_q1_metric_validity_map import plot_fig_q1_metric_validity_map
    results["fig_q1_metric_validity_map"] = run_step(
        "Step 1: Figure Q1 Metric Validity Map",
        plot_fig_q1_metric_validity_map,
    )

    # Step 1: Figure Q1-1
    from plot_fig_q1_1_construct_overview import plot_fig_q1_1
    results["fig_q1_1"] = run_step(
        "Step 1: Figure Q1-1 — Construct Validity Overview",
        plot_fig_q1_1,
    )

    # Step 2: Figure Q1-2
    from plot_fig_q1_2_rebound import plot_fig_q1_2
    results["fig_q1_2"] = run_step(
        "Step 2: Figure Q1-2 — Rebound Cost Decomposition",
        plot_fig_q1_2,
    )

    # Step 3: Figure Q1-3
    from plot_fig_q1_3_stability import plot_fig_q1_3
    results["fig_q1_3"] = run_step(
        "Step 3: Figure Q1-3 — Stability Semantic Rewrite",
        plot_fig_q1_3,
    )

    # Step 4: Figure Q1-4
    from plot_fig_q1_4_ge import fig_q1_4_ge_stress_response
    results["fig_q1_4"] = run_step(
        "Step 4: Figure Q1-4 — GE Stress Response",
        fig_q1_4_ge_stress_response,
    )

    # Step 5: Figure Q1-5
    from plot_fig_q1_5_reliability import plot_fig_q1_5
    results["fig_q1_5"] = run_step(
        "Step 5: Figure Q1-5 — Reliability & Sensitivity",
        plot_fig_q1_5,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)

    print(f"\n{'='*60}")
    print(f"  Pipeline Complete: {n_pass}/{n_total} steps passed  ({elapsed_total:.1f}s)")
    print(f"{'='*60}")

    # List generated files
    print("\n  Generated Tables:")
    for f in sorted(TABLE_DIR.glob("expq1_*.csv")):
        print(f"    - {f.name}  ({f.stat().st_size:,} bytes)")

    print("\n  Generated Figures:")
    for f in sorted(IMG_DIR.rglob("fig_q1_*")):
        print(f"    - {f.relative_to(IMG_DIR)}  ({f.stat().st_size:,} bytes)")

    # ── Update Manifest ──────────────────────────────────────────────────
    manifest = {
        "analysis_name": "expq1_resilience_metrics_v2",
        "data_source": str(BASE_DIR / "Exp0428" / "2026-04-27_09-53-39-val_mini.json"),
        "model": "Qwen2.5-3B-Instruct",
        "runs": [
            {
                "model": "Qwen2.5-3B-Instruct",
                "root": str(BASE_DIR / "Exp0428" / "2026-04-27_09-53-39-val_mini.json"),
                "exists": True,
            }
        ],
        "pipeline_results": {k: ("pass" if v else "fail") for k, v in results.items()},
        "outputs": {
            "tables": [str(f) for f in sorted(TABLE_DIR.glob("expq1_*.csv"))],
            "figures": [str(f) for f in sorted(IMG_DIR.rglob("fig_q1_*"))],
            "report": str(OUT_DIR / "expq1_resilience_metrics_report.md"),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    manifest_path = OUT_DIR / "analysis_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n  Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
