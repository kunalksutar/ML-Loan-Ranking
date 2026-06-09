"""
Model artifact registry for the Lead-to-Bank Ranking system (CLAUDE.md §17).

Manages the versioned model bundle saved to `models/v1/`:
  - metadata.json      : version, feature list, thresholds, training date, dataset stats
  - xgb_model.ubj      : XGBoost binary model (ubj = universal binary JSON)
  - preprocessor.pkl   : Fitted sklearn ColumnTransformer
  - eligibility_rules.json : Bank eligibility rules (one entry per bank)
  - feature_schema.json    : Feature names, types, expected ranges

Usage:
  from src.modeling.model_registry import save_model_bundle, load_model_bundle
  bundle = load_model_bundle("models/v1")
  model, preprocessor, metadata = bundle["model"], bundle["preprocessor"], bundle["metadata"]
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "models/v1"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model_bundle(
    model,
    preprocessor,
    metadata: dict[str, Any],
    feature_schema: dict[str, Any],
    eligibility_rules: list[dict] | None = None,
    output_dir: str = DEFAULT_MODEL_DIR,
) -> Path:
    """
    Persist the full model artifact bundle to `output_dir`.

    Parameters
    ----------
    model             : Fitted XGBClassifier.
    preprocessor      : Fitted sklearn ColumnTransformer.
    metadata          : Arbitrary metadata dict (version, metrics, dataset stats).
    feature_schema    : Feature names → {type, description, expected_range} mapping.
    eligibility_rules : List of per-bank eligibility rule dicts (optional).
    output_dir        : Output directory path.

    Returns
    -------
    Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # XGBoost model in ubj (universal binary JSON)
    model_path = out / "xgb_model.ubj"
    model.save_model(str(model_path))
    logger.info("XGBoost model saved to %s", model_path)

    # Sklearn preprocessor via pickle
    preprocessor_path = out / "preprocessor.pkl"
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Preprocessor saved to %s", preprocessor_path)

    # metadata.json — add saved_at timestamp
    metadata_with_ts = {
        **metadata,
        "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        "model_file": "xgb_model.ubj",
        "preprocessor_file": "preprocessor.pkl",
    }
    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata_with_ts, f, indent=2, default=str)
    logger.info("Metadata saved to %s", meta_path)

    # feature_schema.json
    schema_path = out / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(feature_schema, f, indent=2, default=str)
    logger.info("Feature schema saved to %s", schema_path)

    # eligibility_rules.json
    if eligibility_rules is not None:
        rules_path = out / "eligibility_rules.json"
        with open(rules_path, "w") as f:
            json.dump(eligibility_rules, f, indent=2, default=str)
        logger.info("Eligibility rules saved to %s (%d banks)", rules_path, len(eligibility_rules))

    logger.info("Model bundle saved to %s", out)
    return out


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_model_bundle(model_dir: str = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """
    Load all artifacts from a saved model bundle directory.

    Returns
    -------
    dict with keys: model, preprocessor, metadata, feature_schema,
                    eligibility_rules (may be None if not present).
    """
    from xgboost import XGBClassifier

    bundle_dir = Path(model_dir)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Model bundle directory not found: {bundle_dir}")

    # XGBoost model
    model = XGBClassifier()
    model.load_model(str(bundle_dir / "xgb_model.ubj"))
    logger.info("XGBoost model loaded from %s", bundle_dir / "xgb_model.ubj")

    # Preprocessor
    with open(bundle_dir / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("Preprocessor loaded from %s", bundle_dir / "preprocessor.pkl")

    # metadata
    with open(bundle_dir / "metadata.json") as f:
        metadata = json.load(f)

    # feature_schema
    schema_path = bundle_dir / "feature_schema.json"
    feature_schema = json.load(open(schema_path)) if schema_path.exists() else {}

    # eligibility_rules (optional)
    rules_path = bundle_dir / "eligibility_rules.json"
    eligibility_rules = json.load(open(rules_path)) if rules_path.exists() else None

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "feature_schema": feature_schema,
        "eligibility_rules": eligibility_rules,
    }


# ---------------------------------------------------------------------------
# Feature schema builder
# ---------------------------------------------------------------------------

def build_feature_schema(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Build a feature schema dict from a DataFrame — captures name, dtype,
    min, max, and mean for each feature column.
    """
    schema: dict[str, Any] = {}
    for col in feature_cols:
        series = df[col]
        schema[col] = {
            "dtype": str(series.dtype),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "null_count": int(series.isna().sum()),
        }
    return schema


def build_eligibility_rules(banks_df: pd.DataFrame) -> list[dict]:
    """
    Serialise bank eligibility rules from the banks DataFrame.

    Each row → one dict containing all hard-rule fields for that bank.
    """
    rule_columns = [
        "bank_id", "name", "bank_type",
        "min_cibil_score", "max_cibil_score",
        "min_annual_income", "max_annual_income",
        "max_foir", "max_dti_ratio",
        "min_age", "max_age_at_maturity",
        "max_enquiries_6m", "max_dpd_30_count", "max_dpd_90_count",
        "max_written_off_loans", "max_settled_loans",
        "accepted_income_types", "accepted_employer_categories",
        "min_employer_tenure_months", "min_work_experience_years",
        "loan_types_offered", "min_loan_amount", "max_loan_amount",
        "min_tenure_months", "max_tenure_months",
        "states_covered", "city_tiers_served",
    ]
    available = [c for c in rule_columns if c in banks_df.columns]
    return banks_df[available].to_dict(orient="records")
