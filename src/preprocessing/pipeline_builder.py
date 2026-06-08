"""
Assembles the sklearn ColumnTransformer preprocessing pipeline (CLAUDE.md §8).

Reads `configs/feature_config.yaml` to map each of the 57 ALL_FEATURES columns
to one of four transform groups:

  log_transform_then_scale -> median-impute -> log1p -> StandardScaler
  standard_scale           -> median-impute -> StandardScaler
  ordinal + passthrough    -> passthrough (already rank-ordered / binary ints)
  label_encoded            -> OneHotEncoder(handle_unknown="ignore")

Rule (CLAUDE.md §8): fit the preprocessor on training data only; transform
val/test separately. Never fit on the full dataset before splitting.

CLI (builds, fits on train, transforms val/test, saves the fitted artifact):
  python -m src.preprocessing.pipeline_builder
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.features.feature_registry import ALL_FEATURES, GROUP_KEY, TARGET
from src.preprocessing.encoders import ORDINAL_PASSTHROUGH, build_onehot_encoder
from src.preprocessing.imputers import build_numeric_imputer
from src.preprocessing.scalers import build_log1p_scaler, build_standard_scaler

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_CONFIG = "configs/feature_config.yaml"
DEFAULT_SPLITS_DIR = "data/processed/applications_splits"
DEFAULT_PREPROCESSOR_PATH = "data/artifacts/preprocessor.pkl"

_FEATURE_GROUP_KEYS = (
    "log_transform_then_scale",
    "standard_scale",
    "ordinal_features",
    "label_encoded_features",
    "passthrough_features",
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_feature_config(path: str = DEFAULT_FEATURE_CONFIG) -> dict:
    """Load the feature transform groups from `configs/feature_config.yaml`."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _validate_feature_groups(cfg: dict) -> None:
    """Assert the YAML transform groups partition ALL_FEATURES exactly (no gaps, no overlap)."""
    groups = {key: list(cfg[key]) for key in _FEATURE_GROUP_KEYS}

    seen: dict[str, str] = {}
    for group_name, cols in groups.items():
        for col in cols:
            if col in seen:
                raise ValueError(
                    f"Feature '{col}' appears in both '{seen[col]}' and '{group_name}' "
                    "transform groups — must belong to exactly one group"
                )
            seen[col] = group_name

    missing = [f for f in ALL_FEATURES if f not in seen]
    extra = [f for f in seen if f not in ALL_FEATURES]
    if missing:
        raise ValueError(f"Features missing from feature_config.yaml transform groups: {missing}")
    if extra:
        raise ValueError(f"feature_config.yaml lists unknown features not in ALL_FEATURES: {extra}")


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

def build_preprocessor(feature_config: dict | None = None) -> ColumnTransformer:
    """
    Assemble the full ColumnTransformer per the CLAUDE.md §8 encoding strategy.

    Returns an unfitted ColumnTransformer — call `.fit()` on the training
    split's feature matrix only (see `fit_preprocessor`).
    """
    cfg = feature_config if feature_config is not None else load_feature_config()
    _validate_feature_groups(cfg)

    log_cols = list(cfg["log_transform_then_scale"])
    scale_cols = list(cfg["standard_scale"])
    passthrough_cols = list(cfg["ordinal_features"]) + list(cfg["passthrough_features"])
    onehot_cols = list(cfg["label_encoded_features"])

    log_scale_pipeline = Pipeline(
        steps=[
            ("impute", build_numeric_imputer()),
            ("log1p_scale", build_log1p_scaler()),
        ]
    )
    standard_scale_pipeline = Pipeline(
        steps=[
            ("impute", build_numeric_imputer()),
            ("scale", build_standard_scaler()),
        ]
    )

    transformers = [
        ("log_transform_then_scale", log_scale_pipeline, log_cols),
        ("standard_scale", standard_scale_pipeline, scale_cols),
        ("ordinal_and_binary_passthrough", ORDINAL_PASSTHROUGH, passthrough_cols),
        ("nominal_onehot", build_onehot_encoder(), onehot_cols),
    ]

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    logger.info(
        "Preprocessor assembled | log_transform_then_scale=%d | standard_scale=%d | "
        "passthrough=%d | onehot=%d | total_input_features=%d",
        len(log_cols), len(scale_cols), len(passthrough_cols), len(onehot_cols),
        len(log_cols) + len(scale_cols) + len(passthrough_cols) + len(onehot_cols),
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Fit / transform helpers — enforce train-only fitting
# ---------------------------------------------------------------------------

def fit_preprocessor(
    preprocessor: ColumnTransformer,
    train_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Fit the preprocessor on the TRAINING split's feature matrix only.

    CLAUDE.md §8 rule: never fit on val/test or the full dataset — doing so
    leaks val/test distribution statistics (means, variances, categories)
    into the training-time transform.
    """
    cols = feature_cols if feature_cols is not None else ALL_FEATURES
    logger.info("Fitting preprocessor on training split (%d rows, %d features)", len(train_df), len(cols))
    preprocessor.fit(train_df[cols])
    return preprocessor


def transform_split(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Transform a split's feature matrix using an already-fitted preprocessor."""
    cols = feature_cols if feature_cols is not None else ALL_FEATURES
    return preprocessor.transform(df[cols])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_preprocessor(preprocessor: ColumnTransformer, path: str = DEFAULT_PREPROCESSOR_PATH) -> None:
    """Pickle a fitted preprocessor to disk (CLAUDE.md §17 model bundle: `preprocessor.pkl`)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Fitted preprocessor saved to %s", out_path)


def load_preprocessor(path: str = DEFAULT_PREPROCESSOR_PATH) -> ColumnTransformer:
    """Load a previously fitted preprocessor from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


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
        description="Build, fit, and persist the Section 8 preprocessing pipeline"
    )
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--out", default=DEFAULT_PREPROCESSOR_PATH)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    train = pd.read_parquet(splits_dir / "train.parquet")
    val = pd.read_parquet(splits_dir / "val.parquet")
    test = pd.read_parquet(splits_dir / "test.parquet")

    # ---- Section 9 split-integrity guard before fitting (no lead overlap) ----
    from src.preprocessing.splitting import compute_scale_pos_weight, validate_split_integrity

    validate_split_integrity(train, val, test)

    cfg = load_feature_config(args.feature_config)
    preprocessor = build_preprocessor(cfg)

    # ---- Train-only fit; transform val/test with the frozen preprocessor ----
    fit_preprocessor(preprocessor, train)
    X_train = transform_split(preprocessor, train)
    X_val = transform_split(preprocessor, val)
    X_test = transform_split(preprocessor, test)

    scale_pos_weight = compute_scale_pos_weight(train[TARGET])

    save_preprocessor(preprocessor, args.out)

    print("\n=== Preprocessing Pipeline Summary (Section 8 & 9) ===")
    print(f"Input features         : {len(ALL_FEATURES)}")
    print(f"Output dimensions      : {X_train.shape[1]}")
    print(f"Train rows             : {len(train):,}  -> transformed shape {X_train.shape}")
    print(f"Val rows               : {len(val):,}  -> transformed shape {X_val.shape}")
    print(f"Test rows              : {len(test):,}  -> transformed shape {X_test.shape}")
    print(f"Train unique leads     : {train[GROUP_KEY].nunique():,}")
    print(f"Val unique leads       : {val[GROUP_KEY].nunique():,}")
    print(f"Test unique leads      : {test[GROUP_KEY].nunique():,}")
    print(f"Train conversion rate  : {train[TARGET].mean():.4f}")
    print(f"scale_pos_weight (train): {scale_pos_weight:.4f}")
    print(f"Null cells after transform (train): {int(np.isnan(X_train).sum())}")
    print(f"Fitted preprocessor saved to: {args.out}")


if __name__ == "__main__":
    _main()
