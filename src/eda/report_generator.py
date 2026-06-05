"""
EDA Report Generator — Section 6.

Orchestrates all univariate, bivariate, and correlation analyses,
then emits a ydata-profiling HTML report to data/artifacts/data_report.html.

CLI usage:
  python -m src.eda.report_generator
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import structlog
import warnings

warnings.filterwarnings("ignore")

log = structlog.get_logger()

ARTIFACTS_DIR = Path("data/artifacts")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leads = pd.read_parquet(RAW_DIR / "leads.parquet")
    banks = pd.read_parquet(RAW_DIR / "banks.parquet")
    apps_raw = pd.read_parquet(PROCESSED_DIR / "applications_raw.parquet")
    apps_features = pd.read_parquet(PROCESSED_DIR / "applications_features.parquet")
    log.info("data_loaded",
             n_leads=len(leads), n_banks=len(banks),
             n_apps_raw=len(apps_raw), n_apps_features=len(apps_features))
    return leads, banks, apps_raw, apps_features


def run_univariate(leads: pd.DataFrame, banks: pd.DataFrame, save_dir: Path) -> dict:
    """Run all univariate analyses; return summary stats."""
    from src.eda.univariate import (
        plot_categorical_distributions,
        plot_continuous_distributions,
        plot_log_normal_verification,
        verify_log_normal_skew,
    )

    log.info("running_univariate_analysis")
    plot_continuous_distributions(leads, save_dir)
    log_normal_stats = verify_log_normal_skew(leads)
    plot_log_normal_verification(leads, save_dir)
    plot_categorical_distributions(leads, banks, save_dir)
    return {"log_normal_stats": log_normal_stats}


def run_bivariate(
    apps_features: pd.DataFrame,
    apps_raw: pd.DataFrame,
    leads: pd.DataFrame,
    banks: pd.DataFrame,
    save_dir: Path,
) -> dict:
    """Run all bivariate analyses; return summary DataFrames as dicts."""
    from src.eda.bivariate import (
        plot_bank_profiles,
        plot_conversion_by_bank_type,
        plot_conversion_heatmap,
        plot_per_bank_conversion,
        plot_rejection_reasons,
    )

    log.info("running_bivariate_analysis")

    conversion_by_type = plot_conversion_by_bank_type(apps_features, banks, save_dir)
    heatmap_pivot = plot_conversion_heatmap(apps_features, leads, banks, save_dir)
    rejection_summary = plot_rejection_reasons(apps_raw, save_dir)
    plot_bank_profiles(banks, save_dir)
    per_bank = plot_per_bank_conversion(apps_features, banks, save_dir)

    per_bank_std = per_bank["conversion_rate"].std()
    overall_rate = float(apps_features["converted"].mean())
    log.info(
        "bivariate_summary",
        overall_conversion_rate=round(overall_rate, 4),
        per_bank_conversion_std=round(per_bank_std, 4),
        per_bank_std_passes=per_bank_std > 0.05,
        overall_rate_in_range=0.10 <= overall_rate <= 0.22,
    )

    return {
        "overall_conversion_rate": overall_rate,
        "per_bank_std": per_bank_std,
        "conversion_by_bank_type": conversion_by_type.to_dict(orient="records"),
        "top_rejection_reasons": rejection_summary.head(5).to_dict(orient="records"),
    }


def run_correlation_analysis(apps_features: pd.DataFrame, save_dir: Path) -> dict:
    """Run correlation heatmap, feature-target correlations, VIF, and direction checks."""
    from src.eda.correlation_matrix import (
        compute_vif,
        plot_feature_target_correlations,
        plot_lead_feature_correlations,
        plot_vif,
        validate_correlation_directions,
    )

    log.info("running_correlation_analysis")

    plot_lead_feature_correlations(apps_features, save_dir)
    corrs = plot_feature_target_correlations(apps_features, save_dir)
    direction_results = validate_correlation_directions(apps_features)
    vif_data = compute_vif(apps_features)
    plot_vif(vif_data, save_dir)

    top_positive = corrs[corrs > 0].head(5).to_dict()
    top_negative = corrs[corrs < 0].head(5).to_dict()

    return {
        "direction_checks": direction_results,
        "top_positive_corr_with_converted": top_positive,
        "top_negative_corr_with_converted": top_negative,
        "high_vif_features": vif_data[vif_data["VIF"] > 10]["feature"].tolist(),
    }


def generate_profiling_report(
    apps_features: pd.DataFrame,
    save_dir: Path,
    sample_n: int = 20_000,
) -> None:
    """
    Generate ydata-profiling HTML report on a sample of the feature matrix.
    Saved to data/artifacts/data_report.html.
    """
    try:
        import ydata_profiling  # noqa: F401
        from ydata_profiling import ProfileReport
    except ImportError:
        log.warning("ydata_profiling_not_installed", msg="Skipping HTML report")
        return

    log.info("generating_profiling_report", sample_n=sample_n)
    sample = apps_features.sample(min(sample_n, len(apps_features)), random_state=42)

    profile = ProfileReport(
        sample,
        title="Lead-Bank Ranking: Feature Matrix EDA",
        explorative=True,
        minimal=False,
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
        },
        missing_diagrams={},
        progress_bar=False,
    )

    out_path = save_dir / "data_report.html"
    profile.to_file(out_path)
    log.info("profiling_report_saved", path=str(out_path))


def save_eda_summary(summary: dict, save_dir: Path) -> None:
    """Persist EDA summary JSON for reference in later phases."""
    out_path = save_dir / "eda_summary.json"

    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, (bool, int, str)):
            return obj
        return str(obj)

    with open(out_path, "w") as f:
        json.dump(_make_serializable(summary), f, indent=2)
    log.info("eda_summary_saved", path=str(out_path))


def run_eda(skip_profiling: bool = False) -> dict:
    """
    Full Section 6 EDA pipeline.

    Returns summary dict with all key findings.
    """
    t0 = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    leads, banks, apps_raw, apps_features = _load_data()

    univariate_summary = run_univariate(leads, banks, ARTIFACTS_DIR)
    bivariate_summary = run_bivariate(apps_features, apps_raw, leads, banks, ARTIFACTS_DIR)
    correlation_summary = run_correlation_analysis(apps_features, ARTIFACTS_DIR)

    if not skip_profiling:
        generate_profiling_report(apps_features, ARTIFACTS_DIR)

    summary = {
        "univariate": univariate_summary,
        "bivariate": bivariate_summary,
        "correlation": correlation_summary,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    save_eda_summary(summary, ARTIFACTS_DIR)
    log.info(
        "eda_pipeline_complete",
        elapsed_seconds=summary["elapsed_seconds"],
        artifacts_dir=str(ARTIFACTS_DIR),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Section 6 EDA pipeline")
    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="Skip ydata-profiling HTML report (faster for development)",
    )
    args = parser.parse_args()
    run_eda(skip_profiling=args.skip_profiling)


if __name__ == "__main__":
    main()
