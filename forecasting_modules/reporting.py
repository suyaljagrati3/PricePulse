"""Plotting functions used by the batch forecasting workflow."""

import pandas as pd


def plot_cv_summary(cv_results, cv_folds, plt):
    """Render the SARIMA cross-validation summary chart."""
    if not cv_results:
        return
    cv_df = pd.DataFrame(cv_results)
    figure, axis = plt.subplots(figsize=(14, 5))
    colors = ["#2ecc71" if value < 10 else "#f39c12" if value < 20 else "#e74c3c" for value in cv_df["CV_MAPE"]]
    bars = axis.bar(range(len(cv_df)), cv_df["CV_MAPE"], color=colors, width=0.6, edgecolor="white")
    axis.set_xticks(range(len(cv_df)))
    axis.set_xticklabels(cv_df["Series"], rotation=30, ha="right", fontsize=9)
    axis.axhline(10, color="green", lw=1.5, ls="--", alpha=0.7, label="Excellent (10%)")
    axis.axhline(20, color="orange", lw=1.5, ls="--", alpha=0.7, label="Good (20%)")
    axis.set_title(f"Time-Series CV MAPE — SARIMA ({cv_folds} Folds)", fontsize=12, fontweight="bold")
    axis.set_ylabel("CV Mean MAPE (%)")
    for bar, value in zip(bars, cv_df["CV_MAPE"]):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{value:.1f}%", ha="center", fontsize=8)
    axis.legend(fontsize=9)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
