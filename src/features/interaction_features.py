"""
Interaction feature computation and full feature-engineering pipeline.

Computes all 15 pair-level interaction features by combining lead and bank
attributes for each (lead × bank) row.

Usage (CLI — builds complete ML-ready dataset):
  python -m src.features.interaction_features

Outputs:
  data/processed/applications_features.parquet   — full feature dataset
  data/processed/applications_splits/train.parquet
  data/processed/applications_splits/val.parquet
  data/processed/applications_splits/test.parquet
  data/artifacts/feature_schema.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.bank_features import prepare_bank_features
from src.features.feature_registry import (
    ALL_FEATURES,
    BANK_FEATURES,
    FORBIDDEN_FEATURES,
    GROUP_KEY,
    INTERACTION_FEATURES,
    LEAD_FEATURES,
    TARGET,
    TEMPORAL_FEATURES,
)
from src.features.lead_features import prepare_lead_features
from src.features.temporal_features import compute_temporal_features

logger = logging.getLogger(__name__)

# Lead columns to carry into the merged frame
_LEAD_COLS = [
    "lead_id", "created_at",
    # raw categoricals needed for interaction features (dropped after)
    "income_type", "employer_category", "loan_type", "gender", "state",
    # LEAD_FEATURES (numeric + encoded)
    "age", "annual_income", "cibil_score", "foir", "dti_ratio",
    "loan_to_income_ratio", "enquiry_count_6m", "dpd_30_count", "dpd_90_count",
    "written_off_loans", "settled_loans", "existing_loan_count",
    "work_experience_years", "current_employer_tenure_yrs",
    "credit_card_spend_monthly", "savings_balance", "loan_amount_requested",
    "loan_tenure_months", "credit_utilization", "age_at_maturity", "city_tier",
    "income_type_enc", "employer_category_enc", "loan_type_enc", "gender_enc",
]

# Bank columns to carry into the merged frame
_BANK_COLS = [
    "bank_id",
    # BANK_FEATURES (numeric + encoded)
    "min_cibil_score", "max_foir", "min_annual_income", "approval_base_rate",
    "disbursal_speed_days", "interest_rate_min", "interest_rate_max",
    "max_enquiries_6m", "max_loan_amount", "min_loan_amount",
    "bank_type_enc", "risk_appetite_enc", "documentation_strictness_enc",
    # Extra columns for interaction feature formulas (dropped after)
    "accepted_income_types", "loan_types_offered", "states_covered",
    "preferred_cibil_min", "preferred_cibil_max",
    "max_age_at_maturity", "max_dpd_90_count",
]

# Columns used only as computation helpers — removed from final output
_HELPER_COLS = [
    "created_at", "submitted_at",
    "income_type", "employer_category", "loan_type", "gender", "state",
    "accepted_income_types", "loan_types_offered", "states_covered",
    "preferred_cibil_min", "preferred_cibil_max",
    "max_age_at_maturity", "max_dpd_90_count",
]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 15 pair-level interaction features.

    Expects a merged DataFrame with both lead and bank columns present.
    Adds the 15 INTERACTION_FEATURES columns in-place and returns df.

    Required columns:
      Lead side : cibil_score, foir, annual_income, loan_amount_requested,
                  enquiry_count_6m, dpd_90_count, age_at_maturity,
                  income_type, loan_type, state
      Bank side : min_cibil_score, max_foir, min_annual_income,
                  min_loan_amount, max_loan_amount, max_enquiries_6m,
                  max_dpd_90_count, preferred_cibil_min, preferred_cibil_max,
                  max_age_at_maturity, accepted_income_types (list),
                  loan_types_offered (list), states_covered (list)
    """
    cibil = df["cibil_score"].values.astype(float)
    foir = df["foir"].values.astype(float)
    income = df["annual_income"].values.astype(float)
    loan_amount = df["loan_amount_requested"].values.astype(float)
    enq_count = df["enquiry_count_6m"].values.astype(float)
    dpd_90 = df["dpd_90_count"].values.astype(float)
    age_at_mat = df["age_at_maturity"].values.astype(float)

    b_min_cibil = df["min_cibil_score"].values.astype(float)
    b_max_foir = df["max_foir"].values.astype(float)
    b_min_income = df["min_annual_income"].values.astype(float)
    b_min_loan = df["min_loan_amount"].values.astype(float)
    b_max_loan = df["max_loan_amount"].values.astype(float)
    b_max_enq = df["max_enquiries_6m"].values.astype(float)
    b_max_dpd90 = df["max_dpd_90_count"].values.astype(float)
    b_pref_cibil_min = df["preferred_cibil_min"].values.astype(float)
    b_pref_cibil_max = df["preferred_cibil_max"].values.astype(float)
    b_max_age_mat = df["max_age_at_maturity"].values.astype(float)

    # 1. cibil_gap = lead.cibil_score − bank.min_cibil_score
    df["cibil_gap"] = cibil - b_min_cibil

    # 2. foir_headroom = bank.max_foir − lead.foir
    df["foir_headroom"] = b_max_foir - foir

    # 3. income_headroom = lead.annual_income − bank.min_annual_income
    income_headroom = income - b_min_income
    df["income_headroom"] = income_headroom

    # 4. income_headroom_ratio = income_headroom / bank.min_annual_income
    safe_denom = np.where(b_min_income > 0, b_min_income, 1.0)
    df["income_headroom_ratio"] = income_headroom / safe_denom

    # 5. amount_fit_flag = 1 if loan_amount ∈ [bank.min, bank.max]
    df["amount_fit_flag"] = (
        (loan_amount >= b_min_loan) & (loan_amount <= b_max_loan)
    ).astype(int)

    # 6. amount_position = (amount − bank.min) / (bank.max − bank.min), clipped [0, 1]
    loan_range = b_max_loan - b_min_loan
    safe_range = np.where(loan_range > 0, loan_range, 1.0)
    df["amount_position"] = np.clip((loan_amount - b_min_loan) / safe_range, 0.0, 1.0)

    # 7. income_type_match — list membership per row
    income_types = df["income_type"].values
    accepted_sets = [set(v) for v in df["accepted_income_types"]]
    df["income_type_match"] = np.array(
        [int(it in s) for it, s in zip(income_types, accepted_sets)], dtype=int
    )

    # 8. loan_type_match — list membership per row
    loan_types = df["loan_type"].values
    offered_sets = [set(v) for v in df["loan_types_offered"]]
    df["loan_type_match"] = np.array(
        [int(lt in s) for lt, s in zip(loan_types, offered_sets)], dtype=int
    )

    # 9. geography_match — list membership per row
    states = df["state"].values
    covered_sets = [set(v) for v in df["states_covered"]]
    df["geography_match"] = np.array(
        [int(st in s) for st, s in zip(states, covered_sets)], dtype=int
    )

    # 10. bureau_fatigue_flag = 1 if enquiry_count_6m > bank.max_enquiries_6m
    df["bureau_fatigue_flag"] = (enq_count > b_max_enq).astype(int)

    # 11. bureau_fatigue_excess = max(0, enquiry_count_6m − bank.max_enquiries_6m)
    df["bureau_fatigue_excess"] = np.maximum(0.0, enq_count - b_max_enq)

    # 12. cibil_in_sweet_spot = 1 if cibil ∈ [preferred_cibil_min, preferred_cibil_max]
    df["cibil_in_sweet_spot"] = (
        (cibil >= b_pref_cibil_min) & (cibil <= b_pref_cibil_max)
    ).astype(int)

    # 13. cibil_vs_sweet_spot_dist = |cibil − sweet_spot_center|
    sweet_spot_center = (b_pref_cibil_min + b_pref_cibil_max) / 2.0
    df["cibil_vs_sweet_spot_dist"] = np.abs(cibil - sweet_spot_center)

    # 14. age_maturity_headroom = bank.max_age_at_maturity − lead.age_at_maturity
    df["age_maturity_headroom"] = b_max_age_mat - age_at_mat

    # 15. dpd90_exceeds_bank_max = 1 if lead.dpd_90_count > bank.max_dpd_90_count
    df["dpd90_exceeds_bank_max"] = (dpd_90 > b_max_dpd90).astype(int)

    return df


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def build_feature_dataset(
    apps_path: str = "data/processed/applications_raw.parquet",
    leads_path: str = "data/raw/leads.parquet",
    banks_path: str = "data/raw/banks.parquet",
) -> pd.DataFrame:
    """
    Build the complete ML-ready feature dataset.

    Loads raw data, adds encodings, computes interaction and temporal features,
    validates leakage and feature completeness, and returns the feature frame.

    Output columns:
      application_id, lead_id, bank_id, converted, eligibility_passed
      + ALL_FEATURES (57 columns)
    """
    logger.info("Loading raw data: apps=%s, leads=%s, banks=%s",
                apps_path, leads_path, banks_path)
    apps = pd.read_parquet(apps_path)
    leads = pd.read_parquet(leads_path)
    banks = pd.read_parquet(banks_path)

    logger.info("Preparing lead encodings (%d leads)", len(leads))
    leads_enc = prepare_lead_features(leads)

    logger.info("Preparing bank encodings (%d banks)", len(banks))
    banks_enc = prepare_bank_features(banks)

    # Keep only the columns we need from each table
    lead_join = leads_enc[[c for c in _LEAD_COLS if c in leads_enc.columns]]
    bank_join = banks_enc[[c for c in _BANK_COLS if c in banks_enc.columns]]

    # Retain key + metadata columns from applications
    base_cols = [
        "application_id", "lead_id", "bank_id",
        "submitted_at", "application_sequence_num",
        "eligibility_passed", TARGET,
    ]
    logger.info("Merging applications (%d pairs) with lead and bank tables", len(apps))
    merged = (
        apps[base_cols]
        .merge(lead_join, on="lead_id", how="left")
        .merge(bank_join, on="bank_id", how="left")
    )

    logger.info("Computing 15 interaction features")
    merged = compute_interaction_features(merged)

    logger.info("Computing 4 temporal features")
    merged = compute_temporal_features(merged)

    # Drop helper columns — keep only keys, metadata, and the 57 features
    cols_to_drop = [c for c in _HELPER_COLS if c in merged.columns]
    merged = merged.drop(columns=cols_to_drop)

    # ---- Validation ----
    _validate_feature_dataset(merged)

    logger.info(
        "Feature dataset ready | n_rows=%d | n_features=%d | "
        "conversion_rate=%.4f | null_feature_cells=%d",
        len(merged),
        len(ALL_FEATURES),
        float(merged[TARGET].mean()),
        int(merged[ALL_FEATURES].isnull().sum().sum()),
    )
    return merged


def _validate_feature_dataset(df: pd.DataFrame) -> None:
    """Assert leakage prevention and feature completeness."""
    # No forbidden feature columns
    forbidden_present = [c for c in FORBIDDEN_FEATURES if c in df.columns]
    if forbidden_present:
        raise ValueError(f"Forbidden features found in dataset: {forbidden_present}")

    # All 57 expected features present
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Leakage correlation check: no feature should be a near-perfect proxy for the target
    target = df[TARGET]
    for feat in ALL_FEATURES:
        if df[feat].nunique() < 2:
            continue
        try:
            corr = float(df[feat].corr(target))
        except Exception:
            continue
        if abs(corr) > 0.95:
            logger.warning(
                "High correlation detected: |corr(%s, converted)| = %.4f — inspect for leakage",
                feat, abs(corr),
            )

    # Expected correlation directions from CLAUDE.md §6
    _check_corr_direction(df, "cibil_gap", target, expected_positive=True)
    _check_corr_direction(df, "foir_headroom", target, expected_positive=True)
    _check_corr_direction(df, "bureau_fatigue_flag", target, expected_positive=False)


def _check_corr_direction(
    df: pd.DataFrame,
    feature: str,
    target: pd.Series,
    *,
    expected_positive: bool,
) -> None:
    """Log a warning if a feature's correlation direction contradicts expectations."""
    corr = float(df[feature].corr(target))
    if expected_positive and corr < 0:
        logger.warning(
            "Unexpected correlation direction: corr(%s, converted)=%.4f (expected positive)",
            feature, corr,
        )
    elif not expected_positive and corr > 0:
        logger.warning(
            "Unexpected correlation direction: corr(%s, converted)=%.4f (expected negative)",
            feature, corr,
        )


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lead-level 70/15/15 train/val/test split.

    All rows for a given lead_id land in the same partition — avoids
    target leakage via lead-level signal memorisation.
    """
    unique_leads = df[GROUP_KEY].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_leads))
    shuffled = unique_leads[perm]

    n = len(shuffled)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    train_ids = set(shuffled[:train_end])
    val_ids = set(shuffled[train_end:val_end])
    test_ids = set(shuffled[val_end:])

    train = df[df[GROUP_KEY].isin(train_ids)].reset_index(drop=True)
    val = df[df[GROUP_KEY].isin(val_ids)].reset_index(drop=True)
    test = df[df[GROUP_KEY].isin(test_ids)].reset_index(drop=True)

    logger.info(
        "Split complete | train=%d (%.0f%%) | val=%d (%.0f%%) | test=%d (%.0f%%)",
        len(train), 100 * len(train) / len(df),
        len(val), 100 * len(val) / len(df),
        len(test), 100 * len(test) / len(df),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Feature schema serialisation
# ---------------------------------------------------------------------------

def save_feature_schema(df: pd.DataFrame, output_path: str) -> None:
    """Save a feature schema JSON listing feature names, dtypes, and summary stats."""
    schema: dict = {
        "feature_count": len(ALL_FEATURES),
        "target": TARGET,
        "group_key": GROUP_KEY,
        "feature_groups": {
            "lead_features": LEAD_FEATURES,
            "bank_features": BANK_FEATURES,
            "interaction_features": INTERACTION_FEATURES,
            "temporal_features": TEMPORAL_FEATURES,
        },
        "features": {},
    }
    for feat in ALL_FEATURES:
        if feat not in df.columns:
            continue
        col = df[feat]
        schema["features"][feat] = {
            "dtype": str(col.dtype),
            "null_count": int(col.isnull().sum()),
            "min": float(col.min()) if col.dtype.kind in "fiu" else None,
            "max": float(col.max()) if col.dtype.kind in "fiu" else None,
            "mean": round(float(col.mean()), 4) if col.dtype.kind in "fiu" else None,
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info("Feature schema saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build the ML-ready feature dataset (Section 5)"
    )
    parser.add_argument("--apps",  default="data/processed/applications_raw.parquet")
    parser.add_argument("--leads", default="data/raw/leads.parquet")
    parser.add_argument("--banks", default="data/raw/banks.parquet")
    parser.add_argument("--out",   default="data/processed/applications_features.parquet")
    parser.add_argument("--splits-dir", default="data/processed/applications_splits")
    parser.add_argument("--schema", default="data/artifacts/feature_schema.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features_df = build_feature_dataset(
        apps_path=args.apps,
        leads_path=args.leads,
        banks_path=args.banks,
    )

    # Save full feature dataset
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(out_path, index=False)
    logger.info("Feature dataset saved to %s (%d rows)", out_path, len(features_df))

    # Save train / val / test splits
    train, val, test = split_dataset(features_df, seed=args.seed)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(splits_dir / "train.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)
    logger.info(
        "Splits saved | train=%d | val=%d | test=%d",
        len(train), len(val), len(test),
    )

    # Save feature schema
    save_feature_schema(features_df, args.schema)

    # Print summary statistics
    print("\n=== Feature Engineering Summary ===")
    print(f"Total rows        : {len(features_df):,}")
    print(f"Total features    : {len(ALL_FEATURES)}")
    print(f"  Lead features   : {len(LEAD_FEATURES)}")
    print(f"  Bank features   : {len(BANK_FEATURES)}")
    print(f"  Interaction     : {len(INTERACTION_FEATURES)}")
    print(f"  Temporal        : {len(TEMPORAL_FEATURES)}")
    print(f"Conversion rate   : {features_df[TARGET].mean():.4f}")
    print(f"Null feature cells: {features_df[ALL_FEATURES].isnull().sum().sum()}")
    print(f"Train rows        : {len(train):,}")
    print(f"Val rows          : {len(val):,}")
    print(f"Test rows         : {len(test):,}")
    print()
    print("Top 5 interaction features corr with converted:")
    corrs = features_df[INTERACTION_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET)
    print(corrs.abs().sort_values(ascending=False).head(5).to_string())


if __name__ == "__main__":
    _main()
