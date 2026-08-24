"""
plot_fig_q1_5_reliability.py
============================
Figure Q1-5: Reliability and Statistical Stability (3-panel)

(a) Bootstrap 95% CI for C_rec, β_neighborhood, and degradation_p_cliff.
(b) Monotonic sensitivity under increasing perturbation intensity (Spearman ρ).
(c) Heavy-tail evidence via CCDF with power-law fit.

Reference: Requirements.md §Figure Q1-5
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats as sp_stats

from expq1_data_loader import (load_rebound, load_stability, load_ge,
                                IMG_DIR, TABLE_DIR)
from plot_style import (setup_style, add_panel_label, save_figure,
                        PALETTE, PAL_DECOMP)


def bootstrap_ci(data, statistic=np.mean, n_boot=2000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))
    boot_stats = np.array(boot_stats)
    alpha = 1 - ci
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    center = statistic(data)
    return center, lo, hi


def ccdf(data):
    """Compute complementary CDF (survival function)."""
    data = np.sort(data)
    n = len(data)
    ccdf_vals = 1 - np.arange(1, n + 1) / n
    return data, ccdf_vals


def plot_fig_q1_5():
    setup_style()

    df_reb = load_rebound()
    df_stab = load_stability()
    df_ge = load_ge()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # ── Panel (a): Bootstrap 95% CI ──────────────────────────────────────
    ax = axes[0]

    metrics_info = [
        ("$C_{rec}$", df_reb["c_rec_total"].values, PALETTE["primary"]),
        ("$\\hat{\\beta}_{neigh}$", df_stab["beta_neigh"].values, PALETTE["secondary"]),
        ("$P_{cliff}$", df_reb["is_cliff"].astype(float).values, PALETTE["accent"]),
    ]

    y_positions = np.arange(len(metrics_info))
    ci_results = []

    for i, (label, data, color) in enumerate(metrics_info):
        center, lo, hi = bootstrap_ci(data, n_boot=5000)
        ci_results.append({"metric": label, "center": center, "ci_lo": lo, "ci_hi": hi})

        # Plot CI as horizontal error bar
        ax.errorbar(center, i, xerr=[[center - lo], [hi - center]],
                    fmt="o", color=color, markersize=9,
                    markeredgecolor="white", markeredgewidth=1.5,
                    capsize=6, capthick=2, elinewidth=2, zorder=3)
        ax.text(hi + (hi - lo) * 0.15, i + 0.05,
                f"{center:.3f} [{lo:.3f}, {hi:.3f}]",
                fontsize=8.5, va="center", color=PALETTE["dark_text"])

    ax.set_yticks(y_positions)
    ax.set_yticklabels([m[0] for m in metrics_info])
    ax.set_xlabel("Metric Value (95% Bootstrap CI)")
    ax.set_title("Bootstrap Reliability", fontweight="semibold", pad=8)
    ax.invert_yaxis()

    # Save bootstrap results
    pd.DataFrame(ci_results).to_csv(TABLE_DIR / "expq1_reliability_bootstrap.csv", index=False)

    add_panel_label(ax, "(a)")

    # ── Panel (b): Monotonic sensitivity ─────────────────────────────────
    ax = axes[1]

    # Use replanning_count as proxy for perturbation intensity
    # and recovery cost as the response
    replan_col = "replanning_count_0"
    response_col = "c_rec_total"

    if replan_col in df_reb.columns and response_col in df_reb.columns:
        valid = df_reb[[replan_col, response_col]].dropna()

        if len(valid) >= 5:
            # Bin into quartiles of replanning count
            valid["intensity_q"] = pd.qcut(valid[replan_col], q=4,
                                           duplicates="drop",
                                           labels=False)
            quartile_means = valid.groupby("intensity_q").agg(
                replan_mean=(replan_col, "mean"),
                crec_mean=(response_col, "mean"),
                crec_std=(response_col, "std"),
                n=(response_col, "count"),
            ).reset_index()
            quartile_means["crec_se"] = quartile_means["crec_std"] / np.sqrt(quartile_means["n"])

            x = quartile_means["replan_mean"].values
            y = quartile_means["crec_mean"].values
            yerr = quartile_means["crec_se"].values

            ax.errorbar(x, y, yerr=yerr, fmt="o-", color=PALETTE["primary"],
                        linewidth=2, markersize=9, markeredgecolor="white",
                        markeredgewidth=1.5, capsize=5, capthick=1.5, zorder=3)

            # Fill region
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=PALETTE["primary"])

            # Spearman on raw data
            rho, p = sp_stats.spearmanr(valid[replan_col], valid[response_col])
            ax.text(0.05, 0.95, f"Spearman ρ = {rho:.2f}\np = {p:.4f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

            ax.set_xlabel("Perturbation Intensity\n(Mean Replanning Count per Quartile)")
            ax.set_ylabel("Recovery Cost $C_{rec}$ (Mean ± SE)")
        else:
            # Fallback: use GE stress data
            sweep = df_ge[df_ge["phase_label"] == "stress_sweep"]
            agg = sweep.groupby("stress_lambda").agg(
                crec_mean=("rebound_c_rec", "mean"),
                crec_se=("rebound_c_rec", lambda x: x.std() / np.sqrt(len(x))),
            ).reset_index()

            x = agg["stress_lambda"].values
            y = agg["crec_mean"].values
            yerr = agg["crec_se"].values

            ax.errorbar(x, y, yerr=yerr, fmt="o-", color=PALETTE["primary"],
                        linewidth=2, markersize=9, markeredgecolor="white",
                        markeredgewidth=1.5, capsize=5, capthick=1.5, zorder=3)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=PALETTE["primary"])

            rho, p = sp_stats.spearmanr(x, y)
            ax.text(0.05, 0.95, f"Spearman ρ = {rho:.2f}\np = {p:.4f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

            ax.set_xlabel("Stress Severity λ")
            ax.set_ylabel("Recovery Cost $C_{rec}$ (Mean ± SE)")

    ax.set_title("Monotonic Sensitivity", fontweight="semibold", pad=8)
    add_panel_label(ax, "(b)")

    # ── Panel (c): Heavy-tail CCDF with power-law fit ────────────────────
    ax = axes[2]

    # Combine all recovery costs for CCDF analysis
    all_crec = np.concatenate([
        df_reb["c_rec_total"].dropna().values,
        df_ge[df_ge["phase_label"] == "stress_sweep"]["rebound_c_rec"].dropna().values,
    ])
    all_crec = all_crec[all_crec > 0]  # Remove zeros for log-log

    if len(all_crec) >= 5:
        x_ccdf, y_ccdf = ccdf(all_crec)

        ax.scatter(x_ccdf, y_ccdf, s=25, alpha=0.6, color=PALETTE["primary"],
                   edgecolors="white", linewidths=0.3, zorder=3, label="Empirical CCDF")

        # Power-law fit: log(CCDF) = -α * log(x) + const
        # Only fit to the upper portion
        mask = (x_ccdf > 0) & (y_ccdf > 0)
        log_x = np.log10(x_ccdf[mask])
        log_y = np.log10(y_ccdf[mask])

        if len(log_x) >= 5:
            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(log_x, log_y)
            alpha_pl = -slope

            # Fitted line
            x_fit = np.linspace(log_x.min(), log_x.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(10**x_fit, 10**y_fit, "--", color=PALETTE["danger"],
                    linewidth=2, alpha=0.8,
                    label=f"Power-law fit\nα = {alpha_pl:.2f}, R² = {r_value**2:.2f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Recovery Cost $C_{rec}$")
        ax.set_ylabel("P(X > x)  (CCDF)")
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    ax.set_title("Heavy-Tail Evidence", fontweight="semibold", pad=8)
    add_panel_label(ax, "(c)")

    fig.suptitle("Figure Q1-5: Reliability and Statistical Stability",
                 fontsize=13, fontweight="bold", y=1.03, color=PALETTE["dark_text"])
    fig.tight_layout()
    save_figure(fig, IMG_DIR / "fig_q1_5_reliability_sensitivity.png")


if __name__ == "__main__":
    plot_fig_q1_5()
