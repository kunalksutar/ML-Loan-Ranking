"""
Unit tests for the Section 8 preprocessing pipeline (`src.preprocessing`).

Covers: imputers, scalers, encoders, and the assembled ColumnTransformer
produced by `pipeline_builder.build_preprocessor`. Verifies transform
correctness, train-only fitting, and that the feature_config.yaml transform
groups exactly partition the 57 ALL_FEATURES columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.features.feature_registry import ALL_FEATURES
from src.preprocessing.encoders import build_onehot_encoder
from src.preprocessing.imputers import build_categorical_imputer, build_numeric_imputer
from src.preprocessing.pipeline_builder import (
    _validate_feature_groups,
    build_preprocessor,
    fit_preprocessor,
    load_feature_config,
    transform_split,
)
from src.preprocessing.scalers import build_log1p_scaler, build_standard_scaler


# ---------------------------------------------------------------------------
# scalers.py
# ---------------------------------------------------------------------------

class TestScalers:
    def test_standard_scaler_zero_mean_unit_variance(self):
        rng = np.random.default_rng(0)
        x = rng.normal(loc=50.0, scale=10.0, size=(500, 1))
        scaler = build_standard_scaler()
        out = scaler.fit_transform(x)
        assert out.mean() == pytest.approx(0.0, abs=1e-8)
        assert out.std() == pytest.approx(1.0, abs=1e-8)

    def test_log1p_scaler_reduces_skew(self):
        rng = np.random.default_rng(0)
        # Log-normal — strongly right-skewed, mirrors annual_income / loan_amount
        x = rng.lognormal(mean=13.0, sigma=0.8, size=(2000, 1))
        raw_skew = pd.Series(x.ravel()).skew()

        pipeline = build_log1p_scaler()
        out = pipeline.fit_transform(x)
        out_skew = pd.Series(out.ravel()).skew()

        assert abs(out_skew) < abs(raw_skew)
        assert out.mean() == pytest.approx(0.0, abs=1e-8)
        assert out.std() == pytest.approx(1.0, abs=1e-8)

    def test_log1p_scaler_handles_zero_input(self):
        x = np.array([[0.0], [10.0], [100.0], [1000.0]])
        pipeline = build_log1p_scaler()
        out = pipeline.fit_transform(x)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# imputers.py
# ---------------------------------------------------------------------------

class TestImputers:
    def test_numeric_imputer_fills_with_median(self):
        x = np.array([[1.0], [2.0], [np.nan], [4.0]])
        imputer = build_numeric_imputer()
        out = imputer.fit_transform(x)
        assert not np.isnan(out).any()
        assert out[2, 0] == pytest.approx(2.0)  # median of [1, 2, 4]

    def test_categorical_imputer_fills_with_mode(self):
        x = np.array([[0.0], [0.0], [np.nan], [1.0]])
        imputer = build_categorical_imputer()
        out = imputer.fit_transform(x)
        assert not np.isnan(out).any()
        assert out[2, 0] == pytest.approx(0.0)  # most frequent value


# ---------------------------------------------------------------------------
# encoders.py
# ---------------------------------------------------------------------------

class TestEncoders:
    def test_onehot_encoder_basic(self):
        x = np.array([[0], [1], [2], [0]])
        enc = build_onehot_encoder()
        out = enc.fit_transform(x)
        assert out.shape == (4, 3)
        assert (out.sum(axis=1) == 1).all()  # exactly one hot column per row

    def test_onehot_encoder_handles_unknown_category(self):
        train = np.array([[0], [1], [0]])
        test = np.array([[2]])  # unseen category at transform time
        enc = build_onehot_encoder()
        enc.fit(train)
        out = enc.transform(test)
        assert out.sum() == 0  # handle_unknown="ignore" -> all-zero row, no error


# ---------------------------------------------------------------------------
# pipeline_builder.py — config / assembly
# ---------------------------------------------------------------------------

class TestFeatureConfigPartition:
    def test_feature_config_loads(self):
        cfg = load_feature_config()
        for key in (
            "log_transform_then_scale",
            "standard_scale",
            "ordinal_features",
            "label_encoded_features",
            "passthrough_features",
        ):
            assert key in cfg

    def test_groups_partition_all_features_exactly(self):
        cfg = load_feature_config()
        _validate_feature_groups(cfg)  # must not raise

    def test_groups_detect_missing_feature(self):
        cfg = load_feature_config()
        broken = {k: list(v) for k, v in cfg.items()}
        broken["standard_scale"] = [c for c in broken["standard_scale"] if c != "age"]
        with pytest.raises(ValueError, match="missing"):
            _validate_feature_groups(broken)

    def test_groups_detect_duplicate_feature(self):
        cfg = load_feature_config()
        broken = {k: list(v) for k, v in cfg.items()}
        broken["log_transform_then_scale"] = list(broken["log_transform_then_scale"]) + ["age"]
        with pytest.raises(ValueError, match="appears in both"):
            _validate_feature_groups(broken)


class TestBuildPreprocessor:
    def test_returns_unfitted_column_transformer(self):
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_transformer_groups_cover_all_features(self):
        preprocessor = build_preprocessor()
        covered: list[str] = []
        for _, _, cols in preprocessor.transformers:
            covered.extend(cols)
        assert sorted(covered) == sorted(ALL_FEATURES)


# ---------------------------------------------------------------------------
# pipeline_builder.py — fit/transform behaviour on a synthetic frame
# ---------------------------------------------------------------------------

def _toy_frame(n: int, seed: int) -> pd.DataFrame:
    """Build a minimal frame with every ALL_FEATURES column populated with plausible values."""
    rng = np.random.default_rng(seed)
    cfg = load_feature_config()

    data: dict[str, np.ndarray] = {}
    for col in cfg["log_transform_then_scale"]:
        data[col] = rng.lognormal(mean=12.0, sigma=0.7, size=n)
    for col in cfg["standard_scale"]:
        data[col] = rng.normal(loc=10.0, scale=3.0, size=n)
    for col in cfg["ordinal_features"]:
        data[col] = rng.integers(0, 3, size=n)
    for col in cfg["passthrough_features"]:
        data[col] = rng.integers(0, 2, size=n)
    for col in cfg["label_encoded_features"]:
        data[col] = rng.integers(0, 4, size=n)

    return pd.DataFrame(data)[ALL_FEATURES]


class TestFitTransform:
    def test_fit_on_train_transform_val_no_nans(self):
        train = _toy_frame(400, seed=1)
        val = _toy_frame(120, seed=2)

        preprocessor = build_preprocessor()
        fit_preprocessor(preprocessor, train)

        X_train = transform_split(preprocessor, train)
        X_val = transform_split(preprocessor, val)

        assert X_train.shape[0] == len(train)
        assert X_val.shape[0] == len(val)
        assert X_train.shape[1] == X_val.shape[1]
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_val).any()

    def test_fit_is_train_only_not_refit_on_val(self):
        """Scaler statistics must come from train — transforming train should
        re-center it near zero, but the (differently distributed) val split
        should NOT also land at zero mean, proving stats weren't refit on it."""
        cfg = load_feature_config()
        scale_col = cfg["standard_scale"][0]

        rng = np.random.default_rng(3)
        train = _toy_frame(500, seed=10)
        val = _toy_frame(200, seed=11)
        # Shift one scaled column in val far away from the train distribution
        val[scale_col] = val[scale_col] + 100.0

        preprocessor = build_preprocessor(cfg)
        fit_preprocessor(preprocessor, train)

        X_train = transform_split(preprocessor, train)
        X_val = transform_split(preprocessor, val)

        # Locate the column's position in the transformed output. The
        # ColumnTransformer concatenates blocks in `transformers` order:
        # log_transform_then_scale columns come first, then standard_scale.
        n_log_cols = len(cfg["log_transform_then_scale"])
        col_idx = n_log_cols + list(cfg["standard_scale"]).index(scale_col)

        train_col = X_train[:, col_idx]
        val_col = X_val[:, col_idx]

        assert train_col.mean() == pytest.approx(0.0, abs=1e-6)
        # Val mean reflects the shift relative to train-fitted statistics —
        # i.e. the scaler was NOT refit on val.
        assert val_col.mean() > 5.0

    def test_save_and_load_roundtrip(self, tmp_path):
        from src.preprocessing.pipeline_builder import load_preprocessor, save_preprocessor

        train = _toy_frame(200, seed=20)
        preprocessor = build_preprocessor()
        fit_preprocessor(preprocessor, train)

        path = tmp_path / "preprocessor.pkl"
        save_preprocessor(preprocessor, str(path))
        loaded = load_preprocessor(str(path))

        np.testing.assert_allclose(
            transform_split(loaded, train),
            transform_split(preprocessor, train),
        )
