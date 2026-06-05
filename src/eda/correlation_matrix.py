"""
Correlation and multicollinearity analysis.

Produces:
  - Spearman correlation heatmap of LEAD_FEATURES
  - ALL_FEATURES vs `converted` ranked bar chart
  - VIF analysis for multicollinearity detection
  - Validation of expected correlation directions
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import structlog
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.features.feature_registry import ALL_FEATURES, LEAD_FEATURES, TARGET

log = structlog.get_logger()

# Expected sign of Spearman correlation with `converted`
_EXPECTED_DIRECTIONS: dict[str, str] = {
    "cibil_score": "positive",
    "enquiry_count_6m": "negative",
    "foir_headroom": "positive",
    "bureau_fatigue_flag": "negative",
    "income_type_match": "positive",
    "cibil_gap": "positive",
}


def compute_spearman_lead_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute full Spearman correlation matrix for LEAD_FEATURES.
    Uses a sample of 50K rows for speed (Spearman is O(n log n)).
    """
    available = [c for c in LEAD_FEATURES if c in df.columns]
    sample = df[available].sample(min(50_000, len(df)), random_state=42)
    corr_matrix = sample.corr(method="spearman")
    return corr_matrix


def plot_lead_feature_correlations(df: pd.DataFrame, save_dir: Path) -> pd.DataFrame:
    """Heatmap of Spearman correlations among LEAD_FEATURES."""
    save_dir = Path(save_dir)
    corr_matrix = compute_spearman_lead_features(df)

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={"size": 6},
        linewidths=0.3,
        square=True,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Spearman Correlation — Lead Features", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    out_path = save_dir / "correlation_lead_features.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("lead_feature_correlation_saved", path=str(out_path))
    return corr_matrix


def compute_feature_target_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Spearman correlation of all ALL_FEATURES with `converted`.
    Computed on a sample of 50K rows.
    """
    available = [c for c in ALL_FEATURES if c in df.columns]
    sample = df[available + [TARGET]].sample(min(50_000, len(df)), random_state=42)
    corrs = sample.corr(method="spearman")[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    return corrs


def plot_feature_target_correlations(df: pd.DataFrame, save_dir: Path) -> pd.Series:
    """Ranked bar chart: |Spearman corr| of each feature with `converted`."""
    save_dir = Path(save_dir)
    corrs = compute_feature_target_correlations(df)

    colors = ["#d62728" if v < 0 else "#2ca02c" for v in corrs.values]

    fig, ax = plt.subplots(figsize=(10, max(8, len(corrs) * 0.28)))
    ax.barh(range(len(corrs)), corrs.values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(corrs)))
    ax.set_yticklabels(corrs.index, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman Correlation with 'converted'")
    ax.set_title("Feature–Target Correlations (ALL_FEATURES vs converted)", fontsize=11)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#2ca02c", label="positive"),
            Patch(color="#d62728", label="negative"),
        ],
        fontsize=9,
    )

    # Annotate high-leakage warning line
    ax.axvline(0.95, color="orange", linestyle="--", linewidth=1.2, label="|corr|=0.95 leakage threshold")
    ax.axvline(-0.95, color="orange", linestyle="--", linewidth=1.2)

    plt.tight_layout()
    out_path = save_dir / "correlation_features_vs_target.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("feature_target_correlation_saved", path=str(out_path))
    return corrs


def validate_correlation_directions(df: pd.DataFrame) -> dict[str, dict]:
    """
    Check that expected correlation directions with `converted` hold.
    Returns pass/fail report per feature.
    """
    sample = df[list(_EXPECTED_DIRECTIONS.keys()) + [TARGET]].sample(
        min(50_000, len(df)), random_state=42
    )
    results: dict[str, dict] = {}
    all_pass = True

    for feat, expected_dir in _EXPECTED_DIRECTIONS.items():
        if feat not in sample.columns:
            results[feat] = {"status": "SKIP", "reason": "column missing"}
            continue
        corr_val = float(sample[feat].corr(sample[TARGET], method="spearman"))
        actual_dir = "positive" if corr_val >= 0 else "negative"
        passed = actual_dir == expected_dir
        if not passed:
            all_pass = False
        results[feat] = {
            "expected": expected_dir,
            "actual": actual_dir,
            "corr": round(corr_val, 4),
            "status": "PASS" if passed else "FAIL",
        }
        lvl = log.info if passed else log.warning
        lvl(
            "correlation_direction_check",
            feature=feat,
            expected=expected_dir,
            corr=round(corr_val, 4),
            status="PASS" if passed else "FAIL",
        )

    if all_pass:
        log.info("all_correlation_direction_checks_passed")
    else:
        failed = [k for k, v in results.items() if v.get("status") == "FAIL"]
        log.warning("correlation_direction_failures", features=failed)

    return results


def compute_vif(df: pd.DataFrame, max_features: int = 30) -> pd.DataFrame:
    """
    Compute VIF for numeric ALL_FEATURES (no encoded categoricals — treated as continuous
    here for multicollinearity detection only).
    Uses a sample of 10K rows to keep it fast.
    """
    numeric_cols = [
        c for c in ALL_FEATURES
        if c in df.columns and df[c].dtype in (np.float64, np.int64, float, int)
        and c.endswith("_enc") is False
    ][:max_features]

    sample = (
        df[numeric_cols]
        .sample(min(10_000, len(df)), random_state=42)
        .dropna()
    )

    # Guard: remove zero-variance columns
    sample = sample.loc[:, sample.std() > 0]

    vif_data = pd.DataFrame({
        "feature": sample.columns,
        "VIF": [
            variance_inflation_factor(sample.values, i)
            for i in range(sample.shape[1])
        ],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    high_vif = vif_data[vif_data["VIF"] > 10]
    if not high_vif.empty:
        log.warning(
            "high_vif_features_detected",
            features=high_vif["feature"].tolist(),
            max_vif=round(float(high_vif["VIF"].max()), 2),
        )
    else:
        log.info("vif_check_passed", max_vif=round(float(vif_data["VIF"].max()), 2))

    return vif_data


def plot_vif(vif_data: pd.DataFrame, save_dir: Path) -> None:
    """Horizontal bar chart of VIF values (flag > 10)."""
    save_dir = Path(save_dir)

    fig, ax = plt.subplots(figsize=(9, max(5, len(vif_data) * 0.35)))
    colors = ["#d62728" if v > 10 else "#1f77b4" for v in vif_data["VIF"]]
    ax.barh(range(len(vif_data)), vif_data["VIF"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(vif_data)))
    ax.set_yticklabels(vif_data["feature"], fontsize=8)
    ax.axvline(10, color="red", linestyle="--", linewidth=1.5, label="VIF=10 threshold")
    ax.axvline(5, color="orange", linestyle=":", linewidth=1.2, label="VIF=5 (moderate)")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title("VIF — Multicollinearity Analysis", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = save_dir / "correlation_vif.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("vif_plot_saved", path=str(out_path))
