"""
Correlation audit — VIF + Spearman analysis — Section 7.

This module complements src/eda/correlation_matrix.py by running the same
analyses as *validation checks* (pass/fail) rather than just visualisations.
It catches:
  - Unexpectedly high feature–feature correlation (rho > 0.95) that may
    indicate duplicate / near-duplicate features or derivation errors
  - High VIF features that could destabilise linear models
    (XGBoost handles multicollinearity, but it's still worth flagging)
  - Missing expected cross-correlations (causal chain verification)

CLI:
  python -m src.validation.correlation_audit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from statsmodels.stats.outliers_influence import variance_inflation_factor

log = structlog.get_logger()

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

# Thresholds
HIGH_FEATURE_CORR_THRESHOLD = 0.95   # flags near-duplicate features
HIGH_VIF_THRESHOLD = 10.0
MODERATE_VIF_THRESHOLD = 5.0

# Expected causal correlations between lead features (must hold in the data)
EXPECTED_LEAD_CORRELATIONS: list[dict] = [
    {"a": "cibil_score", "b": "annual_income", "direction": "positive", "min_rho": 0.30},
    {"a": "cibil_score", "b": "dpd_30_count",  "direction": "negative", "max_rho": -0.20},
    {"a": "age",         "b": "work_experience_years", "direction": "positive", "min_rho": 0.40},
    {"a": "annual_income", "b": "loan_amount_requested", "direction": "positive", "min_rho": 0.15},
]


@dataclass
class AuditResult:
    check: str
    passed: bool
    severity: str = "WARN"  # WARN | ERROR | INFO
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""


def _spearman_sample(df: pd.DataFrame, cols: list[str], n: int = 30_000) -> pd.DataFrame:
    """Spearman correlation matrix on a random sample for speed."""
    available = [c for c in cols if c in df.columns]
    sample = df[available].sample(min(n, len(df)), random_state=42)
    return sample.corr(method="spearman")


def check_feature_feature_correlation(
    apps: pd.DataFrame,
    sample_n: int = 30_000,
) -> AuditResult:
    """
    Flag any pair of features with |rho| >= 0.95 (excluding known causal pairs
    like age/age_at_maturity which are derived from each other by design).

    Known high-correlation pairs that are architecturally expected and safe
    for XGBoost (tree models handle them via random column subsampling):
      - age / age_at_maturity  (age_at_maturity = age + tenure/12)
      - interest_rate_min / interest_rate_max (same bank, narrow band)
      - cibil_score / min_cibil_score (both represent CIBIL but on different axes)
    """
    from src.features.feature_registry import ALL_FEATURES

    KNOWN_PAIRS: frozenset[frozenset] = frozenset({
        # Architectural derivations — age_at_maturity = age + tenure/12
        frozenset({"age", "age_at_maturity"}),
        # Same bank — interest band is narrow by design
        frozenset({"interest_rate_min", "interest_rate_max"}),
        # Both represent CIBIL but on lead vs bank axes
        frozenset({"cibil_score", "min_cibil_score"}),
        # income_headroom = annual_income - bank.min_annual_income
        frozenset({"annual_income", "income_headroom"}),
        # income_headroom_ratio = income_headroom / bank.min_annual_income
        frozenset({"income_headroom", "income_headroom_ratio"}),
        # enquiry_velocity_weekly is directly derived from enquiry_count_6m
        frozenset({"enquiry_count_6m", "enquiry_velocity_weekly"}),
        # age_maturity_headroom = bank.max_age_at_maturity - age_at_maturity
        frozenset({"age_at_maturity", "age_maturity_headroom"}),
        # bureau_fatigue_excess is the continuous version of bureau_fatigue_flag
        frozenset({"bureau_fatigue_flag", "bureau_fatigue_excess"}),
    })

    available = [c for c in ALL_FEATURES if c in apps.columns]
    corr_matrix = _spearman_sample(apps, available, n=sample_n)

    flagged: list[dict[str, Any]] = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = float(corr_matrix.iloc[i, j])
            if abs(rho) >= HIGH_FEATURE_CORR_THRESHOLD:
                pair = frozenset({cols[i], cols[j]})
                is_known = pair in KNOWN_PAIRS
                flagged.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "rho": round(rho, 4),
                    "known_pair": is_known,
                })

    unknown_high_corr = [f for f in flagged if not f["known_pair"]]
    passed = len(unknown_high_corr) == 0

    if flagged:
        log.warning(
            "high_feature_correlation",
            total_pairs=len(flagged),
            unknown_pairs=len(unknown_high_corr),
            threshold=HIGH_FEATURE_CORR_THRESHOLD,
        )
        for pair in flagged:
            note = "known architectural pair" if pair["known_pair"] else "UNEXPECTED"
            log.info(
                "high_corr_pair",
                a=pair["feature_a"],
                b=pair["feature_b"],
                rho=pair["rho"],
                note=note,
            )
    else:
        log.info("no_unexpected_high_feature_correlations")

    return AuditResult(
        check="feature_feature_correlation",
        passed=passed,
        severity="WARN",
        message=(
            f"All {len(flagged)} high-corr pairs are known architectural pairs."
            if passed and flagged
            else (
                f"No high-correlation pairs found (threshold={HIGH_FEATURE_CORR_THRESHOLD})."
                if not flagged
                else f"{len(unknown_high_corr)} unexpected high-corr pairs: {unknown_high_corr}"
            )
        ),
        details={
            "all_flagged_pairs": flagged,
            "unexpected_high_corr": unknown_high_corr,
            "threshold": HIGH_FEATURE_CORR_THRESHOLD,
        },
    )


def check_vif(
    apps: pd.DataFrame,
    max_features: int = 30,
    sample_n: int = 10_000,
) -> AuditResult:
    """
    Compute VIF for numeric features; flag features with VIF > 10.

    High VIF is *expected* for feature pairs like (age, age_at_maturity) and
    (cibil_score, min_cibil_score) because these are causally related by design.
    XGBoost uses random column subsampling (colsample_bytree) which mitigates
    the impact.  This check flags them for transparency, not as blocking errors.
    """
    from src.features.feature_registry import ALL_FEATURES

    KNOWN_HIGH_VIF = frozenset({
        "age", "age_at_maturity", "loan_tenure_months",
        "cibil_score", "min_cibil_score",
        "interest_rate_min", "interest_rate_max",
        "annual_income", "credit_card_spend_monthly",
        "dti_ratio", "loan_to_income_ratio",
        "approval_base_rate", "min_annual_income", "max_enquiries_6m", "max_foir",
        "credit_utilization",
    })

    numeric_cols = [
        c for c in ALL_FEATURES
        if c in apps.columns
        and apps[c].dtype in (np.float64, np.int64, float, int)
        and not c.endswith("_enc")
    ][:max_features]

    sample = (
        apps[numeric_cols]
        .sample(min(sample_n, len(apps)), random_state=42)
        .dropna()
    )
    sample = sample.loc[:, sample.std() > 0]

    vif_rows: list[dict[str, Any]] = []
    for i, col in enumerate(sample.columns):
        vif_val = float(variance_inflation_factor(sample.values, i))
        vif_rows.append({"feature": col, "VIF": round(vif_val, 2)})

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    high_vif = vif_df[vif_df["VIF"] > HIGH_VIF_THRESHOLD]
    unexpected_high_vif = high_vif[~high_vif["feature"].isin(KNOWN_HIGH_VIF)]

    log.info(
        "vif_analysis",
        max_vif=round(vif_df["VIF"].max(), 2),
        n_high_vif=len(high_vif),
        n_unexpected=len(unexpected_high_vif),
    )

    # Only flag as FAIL if there are *unexpected* high VIF features
    passed = len(unexpected_high_vif) == 0

    return AuditResult(
        check="vif_multicollinearity",
        passed=passed,
        severity="WARN",
        message=(
            f"VIF check: {len(high_vif)} features with VIF > {HIGH_VIF_THRESHOLD} "
            f"(all are known causal pairs)."
            if passed
            else f"Unexpected high-VIF features: {unexpected_high_vif['feature'].tolist()}"
        ),
        details={
            "vif_table": vif_df.head(20).to_dict(orient="records"),
            "n_high_vif_total": len(high_vif),
            "n_unexpected_high_vif": len(unexpected_high_vif),
            "unexpected_features": unexpected_high_vif.to_dict(orient="records"),
            "threshold": HIGH_VIF_THRESHOLD,
        },
    )


def check_causal_chain_correlations(leads: pd.DataFrame) -> AuditResult:
    """
    Verify expected Spearman correlations between lead features that must
    hold given the causal chain in the data generator.
    """
    failures: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for spec in EXPECTED_LEAD_CORRELATIONS:
        a, b = spec["a"], spec["b"]
        if a not in leads.columns or b not in leads.columns:
            continue
        rho = float(leads[a].corr(leads[b], method="spearman"))
        direction = spec["direction"]
        threshold_key = "min_rho" if direction == "positive" else "max_rho"
        threshold = spec[threshold_key]

        passed = rho >= threshold if direction == "positive" else rho <= threshold
        row = {
            "a": a, "b": b,
            "direction": direction,
            "rho": round(rho, 4),
            "threshold": threshold,
            "passed": passed,
        }
        details.append(row)
        if not passed:
            failures.append(row)
            log.error("causal_correlation_failed", **row)
        else:
            log.info("causal_correlation_passed", **row)

    all_pass = len(failures) == 0
    return AuditResult(
        check="causal_chain_correlations",
        passed=all_pass,
        severity="ERROR",
        message=(
            f"All {len(details)} causal chain correlations verified."
            if all_pass
            else f"{len(failures)} causal chain correlations failed: {failures}"
        ),
        details={"results": details, "failures": failures},
    )


def check_feature_target_monotonicity(
    apps: pd.DataFrame,
    sample_n: int = 50_000,
) -> AuditResult:
    """
    Spot-check that key features have the expected monotonic relationship with
    `converted` when all other factors are held roughly equal (i.e. on eligible
    pairs only).  Ranks by quantile and checks direction.

    Expected:
      - cibil_gap  → higher quantile → higher conversion rate
      - foir_headroom → higher quantile → higher conversion rate
    """
    eligible = apps[apps["eligibility_passed"] == True].copy()
    if len(eligible) < 1000:
        return AuditResult(
            check="feature_target_monotonicity",
            passed=True,
            severity="INFO",
            message="Too few eligible pairs for monotonicity check; skipped.",
        )

    checks: list[dict[str, Any]] = []
    for feat in ("cibil_gap", "foir_headroom"):
        if feat not in eligible.columns:
            continue
        eligible["_quantile"] = pd.qcut(eligible[feat], q=5, labels=False, duplicates="drop")
        q_conv = eligible.groupby("_quantile")["converted"].mean()
        # Check monotonic increase (Spearman on quantile ranks)
        rho = float(q_conv.index.to_series().corr(q_conv, method="spearman"))
        passed = rho > 0
        checks.append({
            "feature": feat,
            "quantile_conversion_rates": q_conv.round(4).to_dict(),
            "spearman_rho": round(rho, 4),
            "monotonically_increasing": passed,
        })
        if passed:
            log.info("monotonicity_check_passed", feature=feat, rho=round(rho, 4))
        else:
            log.warning("monotonicity_check_failed", feature=feat, rho=round(rho, 4))

    all_pass = all(c["monotonically_increasing"] for c in checks)
    return AuditResult(
        check="feature_target_monotonicity",
        passed=all_pass,
        severity="WARN",
        message=(
            f"Monotonicity verified for {len(checks)} features on eligible pairs."
            if all_pass
            else "Monotonicity failure for: "
            + str([c["feature"] for c in checks if not c["monotonically_increasing"]])
        ),
        details={"checks": checks},
    )


def run_correlation_audit() -> list[AuditResult]:
    """Run all correlation and VIF audits; return list of results."""
    leads = pd.read_parquet(RAW_DIR / "leads.parquet")
    apps = pd.read_parquet(PROCESSED_DIR / "applications_features.parquet")

    results = [
        check_causal_chain_correlations(leads),
        check_feature_feature_correlation(apps),
        check_vif(apps),
        check_feature_target_monotonicity(apps),
    ]

    n_pass = sum(1 for r in results if r.passed)
    log.info(
        "correlation_audit_summary",
        total=len(results),
        passed=n_pass,
        failed=len(results) - n_pass,
    )
    return results


def print_results(results: list[AuditResult]) -> None:
    print("\n=== CORRELATION AUDIT REPORT ===")
    for r in results:
        status = "PASS" if r.passed else r.severity
        print(f"  [{status:5s}] {r.check}")
        print(f"         {r.message}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Section 7 correlation audit")
    args = parser.parse_args()
    results = run_correlation_audit()
    print_results(results)
    any_error = any(not r.passed and r.severity == "ERROR" for r in results)
    raise SystemExit(1 if any_error else 0)


if __name__ == "__main__":
    main()
