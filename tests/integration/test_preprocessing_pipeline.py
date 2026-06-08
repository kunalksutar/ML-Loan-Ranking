"""
Integration test for the Section 8 + Section 9 preprocessing & split pipeline.

Runs the full simulation -> feature -> split pipeline on a small generated
dataset, then exercises the preprocessing pipeline end to end:
  - Lead-level split integrity (Section 9)
  - GroupKFold cross-validation grouping (Section 9)
  - Class imbalance handling via scale_pos_weight (Section 9)
  - ColumnTransformer assembly, train-only fitting, val/test transforms (Section 8)
  - No NaNs / forbidden leakage of val statistics into the train-fitted transform
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_registry import ALL_FEATURES, GROUP_KEY, TARGET
from src.features.interaction_features import build_feature_dataset, split_dataset
from src.preprocessing.pipeline_builder import (
    build_preprocessor,
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    transform_split,
)
from src.preprocessing.splitting import (
    class_balance_summary,
    compute_scale_pos_weight,
    iter_group_kfold_splits,
    validate_split_integrity,
)
from src.simulation.application_generator import generate_applications
from src.simulation.bank_generator import generate_banks
from src.simulation.lead_generator import generate_leads

ARCHETYPE_PATH = "configs/bank_archetypes.yaml"
SEED = 11
N_LEADS = 400
N_BANKS = 36


@pytest.fixture(scope="module")
def leads() -> pd.DataFrame:
    return generate_leads(n=N_LEADS, seed=SEED)


@pytest.fixture(scope="module")
def banks() -> pd.DataFrame:
    return generate_banks(seed=SEED, archetype_path=ARCHETYPE_PATH)


@pytest.fixture(scope="module")
def apps(leads, banks) -> pd.DataFrame:
    return generate_applications(leads, banks, seed=SEED)


@pytest.fixture(scope="module")
def features(leads, banks, apps, tmp_path_factory) -> pd.DataFrame:
    tmp = tmp_path_factory.mktemp("data")
    apps_path = str(tmp / "applications_raw.parquet")
    leads_path = str(tmp / "leads.parquet")
    banks_path = str(tmp / "banks.parquet")
    apps.to_parquet(apps_path, index=False)
    leads.to_parquet(leads_path, index=False)
    banks.to_parquet(banks_path, index=False)

    return build_feature_dataset(apps_path=apps_path, leads_path=leads_path, banks_path=banks_path)


@pytest.fixture(scope="module")
def splits(features):
    train, val, test = split_dataset(features, seed=SEED)
    return train, val, test


@pytest.fixture(scope="module")
def fitted_preprocessor(splits):
    train, _, _ = splits
    preprocessor = build_preprocessor()
    fit_preprocessor(preprocessor, train)
    return preprocessor


# ---------------------------------------------------------------------------
# Section 9 — split strategy
# ---------------------------------------------------------------------------

class TestSection9SplitStrategy:
    def test_split_integrity_validator_passes(self, splits):
        train, val, test = splits
        validate_split_integrity(train, val, test)  # must not raise

    def test_split_proportions_lead_level(self, splits, features):
        train, val, test = splits
        total = features[GROUP_KEY].nunique()
        assert 0.65 <= train[GROUP_KEY].nunique() / total <= 0.75
        assert 0.10 <= val[GROUP_KEY].nunique() / total <= 0.20
        assert 0.10 <= test[GROUP_KEY].nunique() / total <= 0.20

    def test_group_kfold_on_train_no_lead_leakage(self, splits):
        train, _, _ = splits
        n_folds = 0
        for train_idx, val_idx in iter_group_kfold_splits(train, n_splits=5):
            train_leads = set(train.iloc[train_idx][GROUP_KEY])
            val_leads = set(train.iloc[val_idx][GROUP_KEY])
            assert train_leads.isdisjoint(val_leads)
            n_folds += 1
        assert n_folds == 5

    def test_class_imbalance_scale_pos_weight_in_expected_band(self, splits):
        train, _, _ = splits
        weight = compute_scale_pos_weight(train[TARGET])
        # CLAUDE.md §1 requires overall conversion in [10%, 22%] -> n_neg/n_pos in ~[3.5, 9]
        assert 2.5 < weight < 12.0

    def test_class_balance_summary_consistent_across_splits(self, splits):
        train, val, test = splits
        for name, split in [("train", train), ("val", val), ("test", test)]:
            summary = class_balance_summary(split)
            assert summary["n_rows"] == len(split)
            assert summary["n_positive"] > 0, f"No positive samples in {name} split"
            assert 0.0 < summary["positive_rate"] < 1.0


# ---------------------------------------------------------------------------
# Section 8 — preprocessing pipeline
# ---------------------------------------------------------------------------

class TestSection8Preprocessing:
    def test_preprocessor_is_fitted_on_train_only(self, splits, fitted_preprocessor):
        train, val, test = splits

        X_train = transform_split(fitted_preprocessor, train)
        X_val = transform_split(fitted_preprocessor, val)
        X_test = transform_split(fitted_preprocessor, test)

        assert X_train.shape[0] == len(train)
        assert X_val.shape[0] == len(val)
        assert X_test.shape[0] == len(test)
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

    def test_no_nans_in_transformed_output(self, splits, fitted_preprocessor):
        train, val, test = splits
        for split in (train, val, test):
            X = transform_split(fitted_preprocessor, split)
            assert not np.isnan(X).any()
            assert np.isfinite(X).all()

    def test_train_split_is_centered_after_scaling(self, splits, fitted_preprocessor):
        """The fitted scalers' statistics come from train, so transform(train)
        should be (approximately) zero-mean for the scaled numeric block."""
        train, _, _ = splits
        X_train = transform_split(fitted_preprocessor, train)
        n_scaled = len(fitted_preprocessor.transformers_[0][2]) + len(fitted_preprocessor.transformers_[1][2])
        scaled_block = X_train[:, :n_scaled]
        assert np.abs(scaled_block.mean(axis=0)).max() < 0.05

    def test_refitting_on_full_data_changes_statistics(self, splits):
        """Sanity check that fitting on train vs. on train+val+test produces
        different statistics — i.e. the choice to fit on train only matters
        and is not a no-op (guards against accidentally fitting on everything)."""
        train, val, test = splits
        full = pd.concat([train, val, test], ignore_index=True)

        train_fitted = build_preprocessor()
        fit_preprocessor(train_fitted, train)

        full_fitted = build_preprocessor()
        fit_preprocessor(full_fitted, full)

        X_from_train_fit = transform_split(train_fitted, test)
        X_from_full_fit = transform_split(full_fitted, test)

        assert not np.allclose(X_from_train_fit, X_from_full_fit)

    def test_save_load_roundtrip_preserves_transform(self, splits, fitted_preprocessor, tmp_path):
        train, val, _ = splits
        path = tmp_path / "preprocessor.pkl"
        save_preprocessor(fitted_preprocessor, str(path))
        loaded = load_preprocessor(str(path))

        np.testing.assert_allclose(
            transform_split(loaded, val),
            transform_split(fitted_preprocessor, val),
        )

    def test_output_feature_count_matches_expected_expansion(self, splits, fitted_preprocessor):
        """Output width = numeric/passthrough columns (1:1) + one-hot expansion
        of the 5 nominal label-encoded columns."""
        from src.preprocessing.pipeline_builder import load_feature_config

        train, _, _ = splits
        cfg = load_feature_config()
        n_numeric_and_passthrough = (
            len(cfg["log_transform_then_scale"])
            + len(cfg["standard_scale"])
            + len(cfg["ordinal_features"])
            + len(cfg["passthrough_features"])
        )
        onehot_transformer = fitted_preprocessor.named_transformers_["nominal_onehot"]
        n_onehot_out = sum(len(cats) for cats in onehot_transformer.categories_)

        X_train = transform_split(fitted_preprocessor, train)
        assert X_train.shape[1] == n_numeric_and_passthrough + n_onehot_out
        assert len(ALL_FEATURES) == n_numeric_and_passthrough + len(cfg["label_encoded_features"])
