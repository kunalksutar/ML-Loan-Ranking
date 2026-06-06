"""
Section 7 validation pipeline — orchestrates all four validators.

Runs:
  1. Schema validation (Pandera)
  2. Leakage detection (forbidden features + correlation)
  3. Distribution / realism checks
  4. Correlation audit (VIF + Spearman)

Exits with code 0 if all ERROR-severity checks pass (WARNs are allowed).
Exits with code 1 if any ERROR-severity check fails.

CLI:
  python -m src.validation.run_all
  python -m src.validation.run_all --save-report
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

ARTIFACTS_DIR = Path("data/artifacts")


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return round(obj, 6)
    return str(obj)


def run_all(save_report: bool = False) -> dict[str, Any]:
    from src.validation.schema_validator import run_schema_validation, ValidationResult
    from src.validation.leakage_detector import run_leakage_detection, LeakageReport
    from src.validation.distribution_checks import run_distribution_checks, AssertionResult
    from src.validation.correlation_audit import run_correlation_audit, AuditResult

    t0 = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Schema validation ----
    print("\n" + "=" * 60)
    print("STEP 1: SCHEMA VALIDATION (Pandera)")
    print("=" * 60)
    schema_results = run_schema_validation()
    for name, r in schema_results.items():
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}")
        for err in r.errors:
            print(f"    ERROR: {err}")
        for warn in r.warnings:
            print(f"    WARN:  {warn}")
        for k, v in r.details.items():
            if isinstance(v, (bool, int, float)) and k not in ("null_counts",):
                print(f"    info:  {k} = {v}")

    # ---- 2. Leakage detection ----
    print("\n" + "=" * 60)
    print("STEP 2: LEAKAGE DETECTION")
    print("=" * 60)
    leakage_reports = run_leakage_detection()
    for name, r in leakage_reports.items():
        status = "PASS" if r.passed else r.severity
        print(f"  [{status:5s}] {r.check}")
        print(f"          {r.message}")

    # ---- 3. Distribution checks ----
    print("\n" + "=" * 60)
    print("STEP 3: REALISM ASSERTIONS")
    print("=" * 60)
    dist_results = run_distribution_checks()
    for r in dist_results:
        status = r.status
        if r.direction == "in_range":
            thresh_str = f"in [{r.lower}, {r.upper}]"
        elif r.direction == "gte":
            thresh_str = f">= {r.threshold}"
        else:
            thresh_str = f"<= {r.threshold}"
        print(
            f"  [{status:5s}] {r.name:<50} "
            f"actual={r.actual:.5f}  ({thresh_str})"
        )
        if not r.passed and r.note:
            print(f"          note: {r.note[:100]}")
        for k, v in r.extra.items():
            print(f"          {k}: {v}")

    # ---- 4. Correlation audit ----
    print("\n" + "=" * 60)
    print("STEP 4: CORRELATION AUDIT")
    print("=" * 60)
    corr_results = run_correlation_audit()
    for r in corr_results:
        status = "PASS" if r.passed else r.severity
        print(f"  [{status:5s}] {r.check}")
        print(f"          {r.message}")

    elapsed = round(time.time() - t0, 1)

    # ---- Summary ----
    schema_errors = sum(1 for r in schema_results.values() if not r.passed)
    leakage_errors = sum(1 for r in leakage_reports.values() if not r.passed and r.severity == "ERROR")
    leakage_warns  = sum(1 for r in leakage_reports.values() if not r.passed and r.severity == "WARN")
    dist_errors    = sum(1 for r in dist_results if not r.passed and r.severity == "ERROR")
    dist_warns     = sum(1 for r in dist_results if not r.passed and r.severity == "WARN")
    corr_errors    = sum(1 for r in corr_results if not r.passed and r.severity == "ERROR")
    corr_warns     = sum(1 for r in corr_results if not r.passed and r.severity == "WARN")

    total_errors = schema_errors + leakage_errors + dist_errors + corr_errors
    total_warns  = leakage_warns + dist_warns + corr_warns

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Schema validation:   {'PASS' if schema_errors == 0 else 'FAIL'} ({schema_errors} errors)")
    print(f"  Leakage detection:   {'PASS' if leakage_errors == 0 else 'FAIL'} ({leakage_errors} errors, {leakage_warns} warnings)")
    print(f"  Realism assertions:  {'PASS' if dist_errors == 0 else 'FAIL'} ({dist_errors} errors, {dist_warns} warnings)")
    print(f"  Correlation audit:   {'PASS' if corr_errors == 0 else 'FAIL'} ({corr_errors} errors, {corr_warns} warnings)")
    print(f"  -----")
    print(f"  Total errors:  {total_errors}")
    print(f"  Total warnings: {total_warns}")
    print(f"  Elapsed: {elapsed}s")
    print()
    if total_errors == 0:
        print("  RESULT: ALL ERROR-LEVEL CHECKS PASSED")
    else:
        print(f"  RESULT: {total_errors} ERROR(S) FOUND — review before training")

    log.info(
        "validation_pipeline_complete",
        total_errors=total_errors,
        total_warnings=total_warns,
        elapsed_seconds=elapsed,
    )

    # Assemble report
    report: dict[str, Any] = {
        "elapsed_seconds": elapsed,
        "total_errors": total_errors,
        "total_warnings": total_warns,
        "schema": {
            name: {
                "passed": r.passed,
                "errors": r.errors,
                "warnings": r.warnings,
                "details": r.details,
            }
            for name, r in schema_results.items()
        },
        "leakage": {
            name: {
                "check": r.check,
                "passed": r.passed,
                "severity": r.severity,
                "message": r.message,
                "details": r.details,
            }
            for name, r in leakage_reports.items()
        },
        "distribution": [
            {
                "name": r.name,
                "passed": r.passed,
                "actual": r.actual,
                "threshold": r.threshold,
                "direction": r.direction,
                "lower": r.lower,
                "upper": r.upper,
                "severity": r.severity,
                "note": r.note,
                "extra": r.extra,
            }
            for r in dist_results
        ],
        "correlation_audit": [
            {
                "check": r.check,
                "passed": r.passed,
                "severity": r.severity,
                "message": r.message,
                "details": r.details,
            }
            for r in corr_results
        ],
    }

    if save_report:
        out_path = ARTIFACTS_DIR / "validation_report.json"
        with open(out_path, "w") as f:
            json.dump(_make_serializable(report), f, indent=2)
        log.info("validation_report_saved", path=str(out_path))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full Section 7 validation pipeline"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save validation report to data/artifacts/validation_report.json",
    )
    args = parser.parse_args()
    report = run_all(save_report=args.save_report)
    raise SystemExit(1 if report["total_errors"] > 0 else 0)


if __name__ == "__main__":
    main()
