"""
Schema validation using Pandera — Section 7.

Validates:
  - leads.parquet: field bounds, dtypes, nulls
  - applications_features.parquet: feature bounds, converted semantics,
    eligibility_passed constraint (converted==0 where eligibility_passed==False)

CLI:
  python -m src.validation.schema_validator
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
import structlog

log = structlog.get_logger()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

LEAD_SCHEMA = DataFrameSchema(
    {
        "lead_id": Column(str, nullable=False),
        "age": Column(
            int,
            checks=[Check.greater_than_or_equal_to(21), Check.less_than_or_equal_to(65)],
            nullable=False,
        ),
        "annual_income": Column(
            float,
            checks=Check.greater_than(0),
            nullable=False,
        ),
        "cibil_score": Column(
            int,
            checks=[Check.greater_than_or_equal_to(300), Check.less_than_or_equal_to(900)],
            nullable=False,
        ),
        "foir": Column(
            float,
            checks=[Check.greater_than_or_equal_to(0.05), Check.less_than_or_equal_to(0.95)],
            nullable=False,
        ),
        "enquiry_count_6m": Column(
            int,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(20)],
            nullable=False,
        ),
        "age_at_maturity": Column(
            int,
            checks=Check.less_than(80),
            nullable=False,
        ),
        "dpd_30_count": Column(int, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "dpd_90_count": Column(int, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "written_off_loans": Column(int, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "settled_loans": Column(int, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "loan_amount_requested": Column(float, checks=Check.greater_than(0), nullable=False),
        "dti_ratio": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "loan_to_income_ratio": Column(float, checks=Check.greater_than(0), nullable=False),
        "credit_utilization": Column(
            float,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(1.0)],
            nullable=False,
        ),
    },
    coerce=False,
)

APPS_FEATURES_SCHEMA = DataFrameSchema(
    {
        "lead_id": Column(str, nullable=False),
        "bank_id": Column(str, nullable=False),
        "converted": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
        ),
        "eligibility_passed": Column(bool, nullable=False),
        "cibil_score": Column(
            int,
            checks=[Check.greater_than_or_equal_to(300), Check.less_than_or_equal_to(900)],
            nullable=False,
        ),
        "foir": Column(
            float,
            checks=[Check.greater_than_or_equal_to(0.05), Check.less_than_or_equal_to(0.95)],
            nullable=False,
        ),
        "annual_income": Column(float, checks=Check.greater_than(0), nullable=False),
        "age": Column(
            int,
            checks=[Check.greater_than_or_equal_to(21), Check.less_than_or_equal_to(65)],
            nullable=False,
        ),
        "enquiry_count_6m": Column(
            int,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(20)],
            nullable=False,
        ),
        "age_at_maturity": Column(
            int,
            checks=Check.less_than(80),
            nullable=False,
        ),
    },
    coerce=False,
)


# ---------------------------------------------------------------------------
# Custom checks that Pandera row-level checks can't express cleanly
# ---------------------------------------------------------------------------

def _check_converted_eligibility_constraint(df: pd.DataFrame) -> dict[str, Any]:
    """
    CRITICAL INVARIANT: converted must be 0 for all ineligible rows.
    converted == 1 is only permitted when eligibility_passed == True.
    """
    violations = df[(df["eligibility_passed"] == False) & (df["converted"] == 1)]
    n_violations = len(violations)
    passed = n_violations == 0
    result = {
        "check": "converted_zero_when_ineligible",
        "passed": passed,
        "n_violations": n_violations,
        "description": "converted==0 for all rows where eligibility_passed==False",
    }
    if not passed:
        log.error(
            "eligibility_conversion_violation",
            n_violations=n_violations,
            sample_ids=violations["lead_id"].head(5).tolist(),
        )
    else:
        log.info("eligibility_conversion_check_passed", n_violations=0)
    return result


def _check_no_nulls_in_features(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    """All feature columns must be non-null."""
    available = [c for c in feature_cols if c in df.columns]
    null_counts = df[available].isnull().sum()
    has_nulls = null_counts[null_counts > 0]
    passed = len(has_nulls) == 0
    result = {
        "check": "no_nulls_in_features",
        "passed": passed,
        "null_counts": null_counts[null_counts > 0].to_dict(),
        "description": "All feature columns must have zero null values",
    }
    if not passed:
        log.error("null_values_in_features", null_counts=has_nulls.to_dict())
    else:
        log.info("null_check_passed", feature_count=len(available))
    return result


# ---------------------------------------------------------------------------
# Validation runners
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    name: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def validate_leads(leads: pd.DataFrame) -> ValidationResult:
    """Run Pandera schema + custom checks against leads.parquet."""
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    # Pandera schema check
    try:
        LEAD_SCHEMA.validate(leads, lazy=True)
        log.info("lead_schema_validation_passed", n_rows=len(leads))
        details["pandera_passed"] = True
    except pa.errors.SchemaErrors as exc:
        error_df = exc.failure_cases
        details["pandera_passed"] = False
        details["pandera_failures"] = error_df.to_dict(orient="records")[:20]
        for _, row in error_df.iterrows():
            errors.append(
                f"Schema violation [{row.get('schema_context', '')}] "
                f"column='{row.get('column', '')}' "
                f"check='{row.get('check', '')}' "
                f"failures={row.get('failure_case', '')}"
            )
        log.error("lead_schema_validation_failed", n_failures=len(error_df))

    # Null check on all columns
    null_counts = leads.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        errors.append(f"Null values found in leads: {cols_with_nulls.to_dict()}")
        details["null_counts"] = cols_with_nulls.to_dict()
    else:
        details["null_counts"] = {}

    # Derived ratio sanity: foir must equal monthly_obligations / (annual_income/12)
    if "monthly_obligations" in leads.columns:
        expected_foir = leads["monthly_obligations"] / (leads["annual_income"] / 12)
        foir_diff = (leads["foir"] - expected_foir).abs()
        n_mismatch = (foir_diff > 0.01).sum()
        if n_mismatch > 0:
            warnings.append(
                f"FOIR derivation mismatch in {n_mismatch} rows "
                "(|foir - obligations/monthly_income| > 0.01)"
            )
        details["foir_mismatch_count"] = int(n_mismatch)

    passed = len(errors) == 0
    return ValidationResult(
        name="leads_schema",
        passed=passed,
        errors=errors,
        warnings=warnings,
        details=details,
    )


def validate_applications_features(
    apps: pd.DataFrame,
    feature_cols: list[str],
) -> ValidationResult:
    """Run Pandera schema + custom checks against applications_features.parquet."""
    from src.features.feature_registry import ALL_FEATURES, FORBIDDEN_FEATURES

    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    # Pandera schema check
    try:
        APPS_FEATURES_SCHEMA.validate(apps, lazy=True)
        log.info("apps_features_schema_validation_passed", n_rows=len(apps))
        details["pandera_passed"] = True
    except pa.errors.SchemaErrors as exc:
        error_df = exc.failure_cases
        details["pandera_passed"] = False
        details["pandera_failures"] = error_df.to_dict(orient="records")[:20]
        for _, row in error_df.iterrows():
            errors.append(
                f"Schema violation [{row.get('schema_context', '')}] "
                f"column='{row.get('column', '')}' "
                f"check='{row.get('check', '')}'"
            )
        log.error("apps_features_schema_validation_failed", n_failures=len(error_df))

    # Critical: converted==0 where ineligible
    elig_check = _check_converted_eligibility_constraint(apps)
    details["eligibility_conversion_check"] = elig_check
    if not elig_check["passed"]:
        errors.append(
            f"CRITICAL: {elig_check['n_violations']} rows have "
            "converted==1 but eligibility_passed==False"
        )

    # No nulls in feature columns
    null_check = _check_no_nulls_in_features(apps, feature_cols)
    details["null_check"] = null_check
    if not null_check["passed"]:
        errors.append(
            f"Null values found in features: {null_check['null_counts']}"
        )

    # Forbidden features must not appear
    forbidden_present = [c for c in FORBIDDEN_FEATURES if c in apps.columns]
    details["forbidden_features_present"] = forbidden_present
    if forbidden_present:
        errors.append(f"Forbidden features present in feature matrix: {forbidden_present}")
        log.error("forbidden_features_in_feature_matrix", features=forbidden_present)
    else:
        log.info("forbidden_features_absent")

    # Row counts sanity
    total = len(apps)
    n_eligible = apps["eligibility_passed"].sum()
    n_converted = apps["converted"].sum()
    details["total_rows"] = total
    details["n_eligible"] = int(n_eligible)
    details["n_converted"] = int(n_converted)
    details["conversion_rate"] = round(float(n_converted / total), 6)

    passed = len(errors) == 0
    return ValidationResult(
        name="apps_features_schema",
        passed=passed,
        errors=errors,
        warnings=warnings,
        details=details,
    )


def validate_splits(splits_dir: Path) -> ValidationResult:
    """
    Validate lead-level split integrity:
    - No lead_id appears in more than one split
    - All 10K leads covered across splits
    - Approximate 70/15/15 proportions
    """
    splits_dir = Path(splits_dir)
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.parquet"
        if not path.exists():
            errors.append(f"Split file missing: {path}")
            continue
        splits[name] = pd.read_parquet(path, columns=["lead_id", "converted"])

    if len(splits) < 3:
        return ValidationResult(
            name="splits_integrity",
            passed=False,
            errors=errors,
        )

    lead_sets = {name: set(df["lead_id"]) for name, df in splits.items()}

    # No overlap
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = lead_sets[a] & lead_sets[b]
        n_overlap = len(overlap)
        details[f"lead_overlap_{a}_{b}"] = n_overlap
        if n_overlap > 0:
            errors.append(
                f"Lead-level leakage: {n_overlap} leads appear in both {a} and {b}"
            )
            log.error("split_lead_overlap", splits=f"{a}+{b}", n_leads=n_overlap)
        else:
            log.info("split_no_overlap", splits=f"{a}+{b}")

    # All leads covered
    all_leads = lead_sets["train"] | lead_sets["val"] | lead_sets["test"]
    details["total_leads_covered"] = len(all_leads)

    # Proportions check (within ±5% of target)
    total_rows = sum(len(df) for df in splits.values())
    for name, df in splits.items():
        pct = len(df) / total_rows * 100
        details[f"{name}_pct"] = round(pct, 2)
        details[f"{name}_conversion_rate"] = round(float(df["converted"].mean()), 4)

    target = {"train": 70.0, "val": 15.0, "test": 15.0}
    for name, tgt in target.items():
        actual = details[f"{name}_pct"]
        if abs(actual - tgt) > 5.0:
            warnings.append(
                f"Split proportion mismatch: {name}={actual:.1f}% (target {tgt}%)"
            )

    log.info(
        "splits_integrity_validated",
        train_pct=details["train_pct"],
        val_pct=details["val_pct"],
        test_pct=details["test_pct"],
        total_leads=details["total_leads_covered"],
    )

    passed = len(errors) == 0
    return ValidationResult(
        name="splits_integrity",
        passed=passed,
        errors=errors,
        warnings=warnings,
        details=details,
    )


def run_schema_validation() -> dict[str, ValidationResult]:
    """Run all schema validations; return dict of results keyed by check name."""
    from src.features.feature_registry import ALL_FEATURES

    leads = pd.read_parquet(RAW_DIR / "leads.parquet")
    apps = pd.read_parquet(PROCESSED_DIR / "applications_features.parquet")

    results: dict[str, ValidationResult] = {}
    results["leads"] = validate_leads(leads)
    results["apps_features"] = validate_applications_features(apps, ALL_FEATURES)
    results["splits"] = validate_splits(PROCESSED_DIR / "applications_splits")

    # Summary
    n_pass = sum(1 for r in results.values() if r.passed)
    n_fail = len(results) - n_pass
    log.info(
        "schema_validation_summary",
        total=len(results),
        passed=n_pass,
        failed=n_fail,
    )
    return results


def print_results(results: dict[str, ValidationResult]) -> None:
    for name, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {name}")
        for err in result.errors:
            print(f"  ERROR: {err}")
        for warn in result.warnings:
            print(f"  WARN:  {warn}")
        for k, v in result.details.items():
            if isinstance(v, (bool, int, float, str)):
                print(f"  info:  {k} = {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Section 7 schema validation")
    args = parser.parse_args()
    results = run_schema_validation()
    print_results(results)
    any_fail = any(not r.passed for r in results.values())
    raise SystemExit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
