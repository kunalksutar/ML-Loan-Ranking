"""
Univariate distribution analysis for leads and banks.

Produces:
  - Histograms (with KDE) for continuous lead features
  - Log-normal fit verification for income, savings, loan_amount
  - Frequency bar charts for categorical lead and bank features
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import structlog

warnings.filterwarnings("ignore", category=FutureWarning)

log = structlog.get_logger()

_CONTINUOUS_LEAD_COLS = [
    "age", "annual_income", "cibil_score", "foir", "dti_ratio",
    "loan_to_income_ratio", "work_experience_years", "current_employer_tenure_yrs",
    "credit_card_spend_monthly", "savings_balance", "loan_amount_requested",
    "loan_tenure_months", "credit_utilization", "monthly_obligations",
]
_LOG_NORMAL_COLS = ["annual_income", "savings_balance", "loan_amount_requested",
                    "credit_card_spend_monthly"]
_CATEGORICAL_LEAD_COLS = ["income_type", "employer_category", "loan_type",
                          "gender", "city_tier", "state"]
_CATEGORICAL_BANK_COLS = ["bank_type", "risk_appetite", "documentation_strictness"]


def plot_continuous_distributions(
    leads: pd.DataFrame,
    save_dir: Path,
    sample_n: int = 5000,
) -> None:
    """
    Histograms + KDE for continuous lead features.
    Uses a random sample for speed; stats computed on full data.
    """
    save_dir = Path(save_dir)
    rng = np.random.default_rng(42)

    available_cols = [c for c in _CONTINUOUS_LEAD_COLS if c in leads.columns]
    sample = leads.sample(min(sample_n, len(leads)), random_state=42)

    n_cols = 3
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes_flat = axes.flatten()

    for i, col in enumerate(available_cols):
        ax = axes_flat[i]
        data = sample[col].dropna()
        sns.histplot(data, kde=True, ax=ax, bins=40, color="#4C72B0", alpha=0.7)
        mean_val = leads[col].mean()
        median_val = leads[col].median()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.2, label=f"mean={mean_val:.1f}")
        ax.axvline(median_val, color="green", linestyle="-.", linewidth=1.2, label=f"median={median_val:.1f}")
        skew = leads[col].skew()
        ax.set_title(f"{col}\nskew={skew:.2f}", fontsize=9)
        ax.set_xlabel("")
        ax.legend(fontsize=7)

    for j in range(len(available_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Continuous Lead Feature Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = save_dir / "univariate_continuous_lead.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("univariate_continuous_saved", path=str(out_path))


def verify_log_normal_skew(leads: pd.DataFrame) -> dict[str, dict]:
    """
    Verify that income/savings/loan_amount are right-skewed and log-normally distributed.
    Returns stats dict per column.
    """
    results: dict[str, dict] = {}
    for col in _LOG_NORMAL_COLS:
        if col not in leads.columns:
            continue
        data = leads[col].dropna()
        log_data = np.log1p(data)
        skewness = float(data.skew())
        log_skewness = float(log_data.skew())
        _, shapiro_p = stats.shapiro(log_data.sample(min(500, len(log_data)), random_state=42))
        is_right_skewed = skewness > 0.5
        log_approx_normal = abs(log_skewness) < 1.0
        results[col] = {
            "skewness": round(skewness, 4),
            "log_skewness": round(log_skewness, 4),
            "shapiro_p_on_log": round(float(shapiro_p), 4),
            "is_right_skewed": is_right_skewed,
            "log_approx_normal": log_approx_normal,
        }
        status = "PASS" if is_right_skewed else "FAIL"
        log.info(
            "log_normal_check",
            col=col,
            skewness=round(skewness, 3),
            log_skewness=round(log_skewness, 3),
            status=status,
        )
    return results


def plot_log_normal_verification(leads: pd.DataFrame, save_dir: Path) -> None:
    """Side-by-side raw vs log-transformed histograms to verify log-normal fit."""
    save_dir = Path(save_dir)
    available_cols = [c for c in _LOG_NORMAL_COLS if c in leads.columns]
    n = len(available_cols)
    fig, axes = plt.subplots(n, 2, figsize=(12, n * 3))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, col in enumerate(available_cols):
        data = leads[col].dropna()
        log_data = np.log1p(data)

        sns.histplot(data, kde=True, ax=axes[i, 0], bins=50, color="#DD8452", alpha=0.7)
        axes[i, 0].set_title(f"{col} — raw (skew={data.skew():.2f})", fontsize=9)

        sns.histplot(log_data, kde=True, ax=axes[i, 1], bins=50, color="#55A868", alpha=0.7)
        axes[i, 1].set_title(f"log1p({col}) (skew={log_data.skew():.2f})", fontsize=9)

    fig.suptitle("Log-Normal Fit Verification (raw vs log-transformed)", fontsize=12)
    plt.tight_layout()
    out_path = save_dir / "univariate_log_normal_fit.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("log_normal_plot_saved", path=str(out_path))


def plot_categorical_distributions(
    leads: pd.DataFrame,
    banks: pd.DataFrame,
    save_dir: Path,
) -> None:
    """Frequency bar charts for categorical lead and bank features."""
    save_dir = Path(save_dir)

    lead_cats = [c for c in _CATEGORICAL_LEAD_COLS if c in leads.columns]
    bank_cats = [c for c in _CATEGORICAL_BANK_COLS if c in banks.columns]

    # Lead categoricals
    n = len(lead_cats)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(lead_cats):
        ax = axes_flat[i]
        counts = leads[col].value_counts().sort_values(ascending=False)
        if len(counts) > 12:
            counts = counts.head(12)
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax,
                    palette="Blues_d", orient="h")
        ax.set_title(f"Lead: {col}", fontsize=9)
        ax.set_xlabel("Count")
        for j, (val, count) in enumerate(zip(counts.index, counts.values)):
            ax.text(count * 0.02, j, f"{count:,}", va="center", fontsize=7)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Categorical Lead Feature Distributions", fontsize=13)
    plt.tight_layout()
    out_path = save_dir / "univariate_categorical_lead.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("categorical_lead_plot_saved", path=str(out_path))

    # Bank categoricals
    fig, axes = plt.subplots(1, len(bank_cats), figsize=(14, 4))
    if len(bank_cats) == 1:
        axes = [axes]

    for i, col in enumerate(bank_cats):
        ax = axes[i]
        counts = banks[col].value_counts().sort_values(ascending=False)
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                    palette="Oranges_d")
        ax.set_title(f"Bank: {col}", fontsize=9)
        ax.set_ylabel("Count")
        for j, (val, count) in enumerate(zip(counts.index, counts.values)):
            ax.text(j, count + 0.2, str(count), ha="center", fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Bank Feature Distributions", fontsize=13)
    plt.tight_layout()
    out_path = save_dir / "univariate_categorical_bank.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("categorical_bank_plot_saved", path=str(out_path))
