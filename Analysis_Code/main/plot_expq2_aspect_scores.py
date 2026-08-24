import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, r"d:\Proj\Framework\Data_Exper\Analysis\scripts")
from plot_style import setup_style, save_figure, PALETTE

def plot_aspect_scores():
    setup_style()
    
    csv_path = Path(r"d:\Proj\Framework\Data_Exper\Analysis\tables\expq2_method_metric_heatmap_scores.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    # Sort by ProfileScore descending for logical flow, or use original method order
    # Let's sort by ProfileScore to show the strongest methods first
    df = df.sort_values(by="ProfileScore", ascending=False)
    
    methods = df["Method"].tolist()
    
    # Extract the normalized [0,1] health scores
    rebound_scores = df["C_rec_health"].tolist()
    stability_scores = df["beta_health"].tolist()
    extensibility_scores = df["lambda_star_health"].tolist()
    
    # Use square figure
    fig, ax = plt.subplots(figsize=(5, 5))
    
    x = np.arange(len(methods))
    width = 0.25
    
    # Colors from original script
    c_reb = "#FF7F0E" # Orange for Rebound
    c_sta = "#2CA02C" # Green for Stability 
    c_ext = "#D62728" # Red for Degradation/Extensibility
    
    # Match the exact offset logic: x, x+width, x+2*width
    ax.bar(x, rebound_scores, width, label='Rebound', color=c_reb, alpha=0.8)
    ax.bar(x + width, stability_scores, width, label='Stability', color=c_sta, alpha=0.8)
    ax.bar(x + 2*width, extensibility_scores, width, label='Extensibility', color=c_ext, alpha=0.8)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('(b) Aspect Scores', fontsize=10)
    
    ax.legend(fontsize=7, loc='best')
    ax.set_ylim(0, 1.05)
    
    out_dir = Path(r"d:\Proj\Framework\paper_writing\Images\results")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    pdf_path = out_dir / "expq2_aspect_scores.pdf"
    png_path = out_dir / "expq2_aspect_scores.png"
    
    # Using save_figure from plot_style
    save_figure(fig, pdf_path)
    # Save PNG explicitly too
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    print(f"[save] {pdf_path}")
    print(f"[save] {png_path}")

if __name__ == "__main__":
    plot_aspect_scores()
