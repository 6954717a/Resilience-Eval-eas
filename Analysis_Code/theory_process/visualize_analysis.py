"""
Visualization Analysis Script
"""

import os
import json
import glob
import pandas as pd
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Analyse.Code.metrics.episode_level import extract_episode_metrics
from Analyse.Code.metrics.task_family_level import compute_task_family_metrics
from Analyse.Code.metrics.evolve_level import compute_evolve_metrics

from Analyse.Code.visualizers.style_utils import set_ieee_style
from Analyse.Code.visualizers.level1_plots import plot_level1_distributions, plot_rebound_analysis, plot_stability_analysis
from Analyse.Code.visualizers.level2_plots import plot_task_family_summary, plot_les_analysis
from Analyse.Code.visualizers.level3_plots import plot_regret_curve, plot_lower_bound_visual, plot_improvement_path_waterfall, plot_bounds_comparison
from Analyse.Code.visualizers.cross_level_plots import plot_cross_level_correlations

# Configuration
# Path to the evolve directory containing summary jsons
# Can be overridden via command line or environment variable
DEFAULT_DATA_DIR = os.path.join(project_root, "Debug", "analyses_01_13_v1", "evolve")
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "Visual", "output")

def load_data(data_dir):
    """Load episode data from summary JSON files."""
    data = []
    # Use summary files from evolve directory
    pattern = os.path.join(data_dir, "*_summary.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} summary files in {data_dir}.")

    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as json_file:
                entry = json.load(json_file)
                metrics = extract_episode_metrics(entry)
                data.append(metrics)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return pd.DataFrame(data)

def load_evolution_context(data_dir):
    """Load evolution context JSON file."""
    pattern = os.path.join(data_dir, "evolution_context_*.json")
    files = glob.glob(pattern)
    if not files:
        print("No evolution context found.")
        return {}
    
    # Load the first found context (assuming one per analysis run)
    f = files[0]
    print(f"Loading evolution context from {f}")
    with open(f, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)

def generate_report(l2, l3, output_dir):
    """Generate a text report."""
    report_path = os.path.join(output_dir, "hierarchical_metrics_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Hierarchical Resilience Metrics Report ===\n\n")
        
        f.write("--- Level 2: Task-Family Metrics ---\n")
        for k, v in l2.items():
            if isinstance(v, (int, float)):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")
        
        f.write("\n--- Level 3: Evolve Metrics ---\n")
        for k, v in l3.items():
            if isinstance(v, (int, float)):
                f.write(f"{k}: {v:.4f}\n")
            elif isinstance(v, dict):
                f.write(f"{k}:\n")
                for sub_k, sub_v in v.items():
                    f.write(f"  {sub_k}: {sub_v}\n")
            elif isinstance(v, list):
                f.write(f"{k}: [List length {len(v)}]\n")
            else:
                f.write(f"{k}: {v}\n")
                
    print(f"Report saved to {report_path}")

def main(data_dir=None, output_dir=None):
    """
    Main analysis pipeline.
    
    Args:
        data_dir: Path to directory containing summary JSON files
        output_dir: Path to output directory for visualizations
    """
    if data_dir is None:
        data_dir = os.environ.get('ANALYSIS_DATA_DIR', DEFAULT_DATA_DIR)
    if output_dir is None:
        output_dir = os.environ.get('ANALYSIS_OUTPUT_DIR', DEFAULT_OUTPUT_DIR)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize Style
    set_ieee_style()
        
    print("Loading Data...")
    df = load_data(data_dir)
    
    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} episodes.")
    
    print("Computing Level 2 Metrics...")
    l2_metrics = compute_task_family_metrics(df)
    print("L2 Metrics Computed")
    
    print("Loading Evolution Context...")
    evolution_context = load_evolution_context(data_dir)
    
    print("Computing Level 3 Metrics...")
    l3_metrics = compute_evolve_metrics(evolution_context, l2_metrics)
    print("L3 Metrics Computed")
    
    print("Generating Visualizations...")
    
    # Level 1 Plots
    plot_level1_distributions(df, output_dir)
    plot_rebound_analysis(df, output_dir)
    plot_stability_analysis(df, output_dir)
    
    # Level 2 Plots
    plot_task_family_summary(l2_metrics, output_dir)
    plot_les_analysis(df, output_dir)
    
    # Level 3 Plots
    plot_regret_curve(l3_metrics, output_dir)
    plot_lower_bound_visual(l3_metrics, output_dir)
    plot_improvement_path_waterfall(l3_metrics, output_dir)
    plot_bounds_comparison(l3_metrics, output_dir)
    
    # Cross Level
    plot_cross_level_correlations(df, output_dir)
    
    # Text Report
    generate_report(l2_metrics, l3_metrics, output_dir)
    
    print(f"Done. Visualizations saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate resilience metrics visualizations')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to directory containing summary JSON files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Path to output directory for visualizations')
    
    args = parser.parse_args()
    main(data_dir=args.data_dir, output_dir=args.output_dir)
