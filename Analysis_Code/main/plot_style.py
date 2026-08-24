"""
plot_style.py
=============
Shared plotting style configuration for Exp-Q1 figures.
Ensures consistent, publication-quality aesthetics across all figure scripts.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ── Color Palette ────────────────────────────────────────────────────────────
# A curated palette inspired by academic papers + modern design
PALETTE = {
    "primary":      "#2E86AB",   # Steel blue
    "secondary":    "#A23B72",   # Deep magenta
    "accent":       "#F18F01",   # Warm amber
    "success":      "#2CA58D",   # Teal green
    "danger":       "#E15554",   # Soft red
    "neutral":      "#7B7D7D",   # Medium gray
    "light_bg":     "#F8F9FA",   # Near-white
    "dark_text":    "#2C3E50",   # Dark navy
}

# Categorical palettes for different scenarios
PAL_PERTURBATION = ["#2E86AB", "#E15554", "#F18F01"]  # clean / perturbed / alt
PAL_DECOMP = ["#2E86AB", "#A23B72", "#F18F01"]        # cog / phy / debt
PAL_FAMILY = ["#2E86AB", "#2CA58D", "#A23B72"]         # task families
PAL_3TYPE = ["#2E86AB", "#E15554", "#F18F01"]          # 3-way comparison

# Sequential palette for stress levels
PAL_STRESS = sns.color_palette("YlOrRd", n_colors=6)


def setup_style():
    """Apply the global figure style. Call once at script start."""
    # Use a clean, minimal style
    sns.set_theme(style="whitegrid", font_scale=1.1)

    mpl.rcParams.update({
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Lines & markers
        "lines.linewidth": 1.8,
        "lines.markersize": 7,

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "grid.color": "#CCCCCC",

        # Spine & ticks
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=15, fontweight="bold"):
    """Add (a), (b), (c) style panel labels."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va="top", ha="left", color=PALETTE["dark_text"])


def save_figure(fig, path, also_pdf=True):
    """Save figure as PNG and optionally PDF."""
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    if also_pdf:
        pdf_path = str(path).replace(".png", ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [save] {path}")
