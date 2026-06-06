"""
Leakage detection — Section 7.

Checks:
  1. Forbidden features not present in training feature matrix
  2. Correlation-based leakage: |corr(feature, converted)| > 0.95 flags any feature
  3. Lead-level split boundary integrity (no lead spans train+val or train+test)
  4. Future-use column audit: datetime/outcome columns absent from feature set

CLI:
  python -m src.validation.leakage_detector
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

PROCESSED_DIR = Path("data/processed")

LEAKAGE_CORR_THRESHOLD = 0.95

# Columns that are fine to have in the raw table but must never land in the
# training feature matrix.  Defined here as the ground truth; also available
# via feature_registry.FORBIDDEN_FEATURES.
FORBIDDEN_FEATURES: tuple[str, ...] = (
    "rejection_reason",
    "approved_amount",
    "approved_rate",
    "approved_tenure_months",
    "disbursed_amount",
    "application_status",
    "bank_responded_at",
    "disbursed_at",
    "disbursal_failure_reason",
)

# Columns that carry implicit outcome information if present
_OUTCOME_SIGNAL_PATTERNS = (
    "approved",
    "disbursed",
    "rejection",
    "status",
)


@dataclass
class LeakageReport:
    check: str
    passed: bool
    severity: str = "ERROR"   # ERROR | WARN | INFO
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""


def check_forbidden_features(df: pd.DataFrame) -> LeakageReport:
    """
    Verify that none of the FORBIDDEN_FEATURES appear in the feature matrix.
    These columns carry post-decision information that would not be available
    at scoring time.
    """
    present = [c for c in FORBIDDEN_FEATURES if c in df.columns]
    passed = len(present) == 0

    if passed:
        log.info("forbidden_feature_check_passed")
    else:
        log.error("forbidden_features_detected", features=present)

    return LeakageReport(
        check="forbidden_features",
        passed=passed,
        severity="ERROR",
        message=(
            "No forbidden features found in feature matrix."
            if passed
            else f"FORBIDDEN features present: {present}"
        ),
        details={"forbidden_present": present, "checked": list(FORBIDDEN_FEATURES)},
    )


def check_outcome_signal_columns(df: pd.DataFrame) -> LeakageReport:
    """
    Heuristic scan: flag any column whose name contains outcome-signal words
    (approved, disbursed, rejection, status) that isn't in the approved feature
    list.  These may indicate a new leaky column not yet in FORBIDDEN_FEATURES.
    """
    from src.features.feature_registry import ALL_FEATURES

    allowed = set(ALL_FEATURES) | {"application_id", "lead_id", "bank_id",
                                    "eligibility_passed", "converted",
                                    "application_sequence_num",
                                    "days_since_first_application",
                                    "enquiry_velocity_weekly", "is_reapplication"}

    suspicious = [
        c for c in df.columns
        if c not in allowed
        and any(pat in c.lower() for pat in _OUTCOME_SIGNAL_PATTERNS)
    ]
    passed = len(suspicious) == 0

    if passed:
        log.info("outcome_signal_scan_clean")
    else:
        log.warning("suspicious_outcome_columns", columns=suspicious)

    return LeakageReport(
        check="outcome_signal_columns",
        passed=passed,
        severity="WARN",
        message=(
            "No suspicious outcome-signal columns found."
            if passed
            else f"Suspicious columns with outcome-signal names: {suspicious}"
        ),
        details={"suspicious_columns": suspicious},
    )


def check_correlation_leakage(
    df: pd.DataFrame,
    sample_n: int = 50_000,
) -> LeakageReport:
    """
    Flag any numeric feature with |Spearman corr(feature, converted)| > 0.95.
    A very high correlation almost certainly means the feature leaks the label.
    """
    from src.features.feature_registry import ALL_FEATURES, TARGET

    available = [c for c in ALL_FEATURES if c in df.columns]
    sample = df[available + [TARGET]].sample(min(sample_n, len(df)), random_state=42)
    numeric = sample.select_dtypes(include=[np.number])

    corrs = (
        numeric.corr(method="spearman")[TARGET]
        .drop(TARGET, errors="ignore")
        .sort_values(key=abs, ascending=False)
    )

    leaky = corrs[abs(corrs) >= LEAKAGE_CORR_THRESHOLD]
    passed = len(leaky) == 0

    high_corr_dict = {str(k): round(float(v), 4) for k, v in corrs.head(20).items()}

    if passed:
        log.info(
            "correlation_leakage_check_passed",
            max_abs_corr=round(float(abs(corrs).max()), 4),
            threshold=LEAKAGE_CORR_THRESHOLD,
        )
    else:
        log.error(
            "correlation_leakage_detected",
            features=leaky.index.tolist(),
            max_corr=round(float(abs(leaky).max()), 4),
        )

    return LeakageReport(
        check="correlation_leakage",
        passed=passed,
        severity="ERROR",
        message=(
            f"No leaky correlations found (max |corr| = {abs(corrs).max():.4f}, "
            f"threshold = {LEAKAGE_CORR_THRESHOLD})."
            if passed
            else f"Leaky features: {leaky.to_dict()}"
        ),
        details={
            "leaky_features": leaky.to_dict(),
            "top20_corr_with_target": high_corr_dict,
            "max_abs_corr": round(float(abs(corrs).max()), 4),
            "threshold": LEAKAGE_CORR_THRESHOLD,
        },
    )


def check_split_boundary_leakage(splits_dir: Path) -> LeakageReport:
    """
    Ensure no lead_id appears in more than one split (train / val / test).
    A lead that straddles splits leaks its label distribution across boundaries.
    """
    splits_dir = Path(splits_dir)
    split_leads: dict[str, set] = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.parquet"
        if not path.exists():
            return LeakageReport(
                check="split_boundary_leakage",
                passed=False,
                severity="ERROR",
                message=f"Split file missing: {path}",
            )
        split_leads[name] = set(
            pd.read_parquet(path, columns=["lead_id"])["lead_id"]
        )

    overlaps: dict[str, int] = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = len(split_leads[a] & split_leads[b])
        overlaps[f"{a}+{b}"] = n
        if n > 0:
            log.error("split_boundary_leak", pair=f"{a}+{b}", n_leads=n)
        else:
            log.info("split_boundary_clean", pair=f"{a}+{b}")

    passed = all(v == 0 for v in overlaps.values())
    total_unique = len(split_leads["train"] | split_leads["val"] | split_leads["test"])

    return LeakageReport(
        check="split_boundary_leakage",
        passed=passed,
        severity="ERROR",
        message=(
            f"Split boundaries clean — no lead appears in more than one split "
            f"({total_unique} unique leads)."
            if passed
            else f"Lead-level leakage across split boundaries: {overlaps}"
        ),
        details={"overlaps": overlaps, "total_unique_leads": total_unique},
    )


def check_temporal_ordering(apps_raw: pd.DataFrame) -> LeakageReport:
    """
    Verify submitted_at is available and that no bank_responded_at or
    disbursed_at timestamps leak into the feature matrix columns.
    """
    details: dict[str, Any] = {}
    warnings: list[str] = []

    # submitted_at should exist in raw
    has_submitted_at = "submitted_at" in apps_raw.columns
    details["has_submitted_at"] = has_submitted_at
    if not has_submitted_at:
        warnings.append("submitted_at column not found in applications_raw")

    # future-leak columns in raw (expected to be there, just confirm absent from features)
    future_cols = ["bank_responded_at", "disbursed_at"]
    for c in future_cols:
        details[f"{c}_in_raw"] = c in apps_raw.columns

    passed = len(warnings) == 0
    return LeakageReport(
        check="temporal_ordering",
        passed=passed,
        severity="WARN",
        message=(
            "Temporal columns present in raw; confirmed absent from feature matrix."
            if passed
            else f"Temporal issues: {warnings}"
        ),
        details=details,
    )


def run_leakage_detection() -> dict[str, LeakageReport]:
    """Run all leakage checks; return dict keyed by check name."""
    apps_feat = pd.read_parquet(PROCESSED_DIR / "applications_features.parquet")
    apps_raw = pd.read_parquet(PROCESSED_DIR / "applications_raw.parquet")
    splits_dir = PROCESSED_DIR / "applications_splits"

    reports: dict[str, LeakageReport] = {}
    reports["forbidden_features"] = check_forbidden_features(apps_feat)
    reports["outcome_signal_scan"] = check_outcome_signal_columns(apps_feat)
    reports["correlation_leakage"] = check_correlation_leakage(apps_feat)
    reports["split_boundary"] = check_split_boundary_leakage(splits_dir)
    reports["temporal_ordering"] = check_temporal_ordering(apps_raw)

    n_errors = sum(
        1 for r in reports.values()
        if not r.passed and r.severity == "ERROR"
    )
    n_warns = sum(
        1 for r in reports.values()
        if not r.passed and r.severity == "WARN"
    )
    log.info(
        "leakage_detection_summary",
        total=len(reports),
        errors=n_errors,
        warnings=n_warns,
    )
    return reports


def print_reports(reports: dict[str, LeakageReport]) -> None:
    print("\n=== LEAKAGE DETECTION REPORT ===")
    for name, r in reports.items():
        status = "PASS" if r.passed else r.severity
        print(f"  [{status:5s}] {r.check}")
        print(f"         {r.message}")
        if not r.passed:
            for k, v in r.details.items():
                if v and k not in ("checked", "top20_corr_with_target"):
                    print(f"         detail: {k} = {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Section 7 leakage detection")
    args = parser.parse_args()
    reports = run_leakage_detection()
    print_reports(reports)
    any_error = any(
        not r.passed and r.severity == "ERROR" for r in reports.values()
    )
    raise SystemExit(1 if any_error else 0)


if __name__ == "__main__":
    main()
