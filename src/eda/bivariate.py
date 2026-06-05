"""
Bivariate analysis: conversion rates, rejection reasons, bank profiles.

Produces:
  - Conversion rate by bank_type
  - income_type × bank_type heatmap
  - Rejection reason frequency
  - min_cibil_score distribution by bank_type
  - Approval rate range across banks
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import structlog

log = structlog.get_logger()


def plot_conversion_by_bank_type(
    apps_features: pd.DataFrame,
    banks: pd.DataFrame,
    save_dir: Path,
) -> pd.DataFrame:
    """
    Bar chart: conversion rate (mean of `converted`) by bank_type.
    Returns a summary DataFrame.
    """
    save_dir = Path(save_dir)

    merged = apps_features.merge(
        banks[["bank_id", "bank_type"]],
        on="bank_id",
        how="left",
    )
    summary = (
        merged.groupby("bank_type")["converted"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "conversion_rate", "sum": "conversions", "count": "total_pairs"})
        .sort_values("conversion_rate", ascending=False)
        .reset_index()
    )
    summary["conversion_rate_pct"] = summary["conversion_rate"] * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = sns.barplot(data=summary, x="bank_type", y="conversion_rate_pct",
                       palette="viridis", ax=ax)
    ax.set_title("Conversion Rate by Bank Type", fontsize=12)
    ax.set_xlabel("Bank Type")
    ax.set_ylabel("Conversion Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    for patch, (_, row) in zip(bars.patches, summary.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.1,
            f"{row['conversion_rate_pct']:.1f}%\nn={row['total_pairs']:,}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    out_path = save_dir / "bivariate_conversion_by_bank_type.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("conversion_by_bank_type_saved", path=str(out_path))
    return summary


def plot_conversion_heatmap(
    apps_features: pd.DataFrame,
    leads: pd.DataFrame,
    banks: pd.DataFrame,
    save_dir: Path,
) -> pd.DataFrame:
    """
    Heatmap: conversion rate by income_type × bank_type.
    Returns pivot table.
    """
    save_dir = Path(save_dir)

    merged = (
        apps_features[["lead_id", "bank_id", "converted"]]
        .merge(leads[["lead_id", "income_type"]], on="lead_id", how="left")
        .merge(banks[["bank_id", "bank_type"]], on="bank_id", how="left")
    )
    pivot = (
        merged.groupby(["income_type", "bank_type"])["converted"]
        .mean()
        .unstack("bank_type")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Conversion Rate (%)"},
    )
    ax.set_title("Conversion Rate (%) — income_type × bank_type", fontsize=12)
    ax.set_xlabel("Bank Type")
    ax.set_ylabel("Income Type")

    plt.tight_layout()
    out_path = save_dir / "bivariate_conversion_heatmap.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("conversion_heatmap_saved", path=str(out_path))
    return pivot


def plot_rejection_reasons(apps_raw: pd.DataFrame, save_dir: Path) -> pd.DataFrame:
    """
    Horizontal bar chart of rejection reason frequency (ineligible + rejected pairs).
    Returns a summary DataFrame.
    """
    save_dir = Path(save_dir)

    reason_counts = (
        apps_raw["rejection_reason"]
        .dropna()
        .value_counts()
        .reset_index()
        .rename(columns={"rejection_reason": "reason", "count": "n"})
    )
    reason_counts["pct"] = reason_counts["n"] / len(apps_raw) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=reason_counts,
        x="n",
        y="reason",
        palette="Reds_r",
        orient="h",
        ax=ax,
    )
    ax.set_title("Rejection / Ineligibility Reason Frequency", fontsize=12)
    ax.set_xlabel("Count of Pairs")
    ax.set_ylabel("")
    for i, (_, row) in enumerate(reason_counts.iterrows()):
        ax.text(row["n"] + reason_counts["n"].max() * 0.005, i,
                f"{row['pct']:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    out_path = save_dir / "bivariate_rejection_reasons.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("rejection_reasons_saved", path=str(out_path))
    return reason_counts


def plot_bank_profiles(banks: pd.DataFrame, save_dir: Path) -> None:
    """
    Two plots:
    1. min_cibil_score distribution by bank_type (box + strip)
    2. approval_base_rate range across all banks (sorted)
    """
    save_dir = Path(save_dir)

    # Plot 1: min_cibil_score by bank_type
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sns.boxplot(data=banks, x="bank_type", y="min_cibil_score",
                palette="Set2", ax=ax)
    sns.stripplot(data=banks, x="bank_type", y="min_cibil_score",
                  color="black", size=5, jitter=True, ax=ax, alpha=0.6)
    ax.set_title("min_cibil_score by Bank Type", fontsize=11)
    ax.set_xlabel("Bank Type")
    ax.set_ylabel("Min CIBIL Score")
    ax.tick_params(axis="x", rotation=15)

    # Plot 2: approval_base_rate sorted bar
    ax2 = axes[1]
    bank_sorted = banks.sort_values("approval_base_rate", ascending=False).reset_index(drop=True)
    colors = {"PSB": "#1f77b4", "private": "#ff7f0e", "NBFC": "#2ca02c",
              "fintech": "#d62728", "HFC": "#9467bd"}
    bar_colors = [colors.get(bt, "#7f7f7f") for bt in bank_sorted["bank_type"]]
    ax2.bar(range(len(bank_sorted)), bank_sorted["approval_base_rate"], color=bar_colors, alpha=0.8)
    ax2.set_title("Approval Base Rate — All Banks (sorted)", fontsize=11)
    ax2.set_xlabel("Bank (sorted by approval rate)")
    ax2.set_ylabel("Approval Base Rate")
    ax2.axhline(banks["approval_base_rate"].mean(), color="red", linestyle="--",
                linewidth=1.5, label=f"mean={banks['approval_base_rate'].mean():.2f}")
    ax2.legend(fontsize=9)

    # Legend for bank type colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in colors.items()]
    ax2.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="red", linestyle="--",
                   label=f"mean={banks['approval_base_rate'].mean():.2f}")
    ], fontsize=8)

    fig.suptitle("Bank Profile Analysis", fontsize=13)
    plt.tight_layout()
    out_path = save_dir / "bivariate_bank_profiles.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("bank_profiles_saved", path=str(out_path))


def plot_per_bank_conversion(
    apps_features: pd.DataFrame,
    banks: pd.DataFrame,
    save_dir: Path,
) -> pd.DataFrame:
    """Per-bank conversion rate bar chart — verifies std > 0.05 across banks."""
    save_dir = Path(save_dir)

    per_bank = (
        apps_features.groupby("bank_id")["converted"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "conversion_rate", "count": "pairs"})
        .merge(banks[["bank_id", "name", "bank_type"]], on="bank_id", how="left")
        .sort_values("conversion_rate", ascending=False)
        .reset_index()
    )
    std_val = per_bank["conversion_rate"].std()
    log.info("per_bank_conversion_std", std=round(std_val, 4),
             passes=std_val > 0.05)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"PSB": "#1f77b4", "private": "#ff7f0e", "NBFC": "#2ca02c",
              "fintech": "#d62728", "HFC": "#9467bd"}
    bar_colors = [colors.get(bt, "#7f7f7f") for bt in per_bank["bank_type"]]

    ax.bar(range(len(per_bank)), per_bank["conversion_rate"] * 100, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(per_bank)))
    ax.set_xticklabels(per_bank["name"], rotation=75, ha="right", fontsize=7)
    ax.set_title(f"Per-Bank Conversion Rate (std={std_val:.3f})", fontsize=11)
    ax.set_ylabel("Conversion Rate (%)")
    ax.axhline(per_bank["conversion_rate"].mean() * 100, color="red",
               linestyle="--", linewidth=1.5,
               label=f"mean={per_bank['conversion_rate'].mean() * 100:.1f}%")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in colors.items()]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = save_dir / "bivariate_per_bank_conversion.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("per_bank_conversion_saved", path=str(out_path))
    return per_bank
