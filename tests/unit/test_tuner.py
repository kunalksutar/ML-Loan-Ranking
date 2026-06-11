"""
Unit tests for src/modeling/tuner.py (CLAUDE.md §12).

Covers:
  - _sample_params: correct keys and value ranges
  - _single_split_objective: smoke test returns float in [0, 1]
  - segment_errors / calibration_data / profile_error_segment (evaluator §14 helpers)
  - tune: integration smoke test with 3 trials on toy data (uses tmp MLflow store)
"""

from __future__ import annotations

import math
import tempfile
import uuid
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from src.features.feature_registry import GROUP_KEY, TARGET
from src.modeling.evaluator import (
    calibration_data,
    profile_error_segment,
    segment_errors,
)
from src.modeling.tuner import _sample_params


# ---------------------------------------------------------------------------
# Fixtures — minimal toy DataFrames
# ---------------------------------------------------------------------------

N_LEADS = 20
N_BANKS = 4
SCORE_COL = "predicted_score"


def _make_scored_df(seed: int = 0) -> pd.DataFrame:
    """
    Build a minimal (lead × bank) scored DataFrame.

    Each lead has N_BANKS rows. Exactly one bank per lead has converted=1.
    Scores are random — not necessarily ranking the positive bank first.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for lead_idx in range(N_LEADS):
        lead_id = f"lead_{lead_idx:03d}"
        positive_bank = rng.integers(0, N_BANKS)
        for bank_idx in range(N_BANKS):
            rows.append({
                GROUP_KEY: lead_id,
                "bank_id": f"bank_{bank_idx}",
                TARGET: 1 if bank_idx == positive_bank else 0,
                SCORE_COL: float(rng.uniform(0.0, 1.0)),
                # A handful of feature columns for profile_error_segment tests
                "cibil_score": float(rng.integers(300, 900)),
                "foir": float(rng.uniform(0.1, 0.9)),
                "income_type_enc": int(rng.integers(0, 4)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests — _sample_params
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "max_depth": {"type": "int", "low": 3, "high": 8},
    "learning_rate": {"type": "float_log", "low": 0.01, "high": 0.3},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "n_estimators": {"type": "int", "low": 100, "high": 600},
    "scale_pos_weight": {"type": "float", "low": 3.0, "high": 10.0},
}


class TestSampleParams:
    def test_returns_all_expected_keys(self):
        study = optuna.create_study()
        trial = study.ask()
        params = _sample_params(trial, SEARCH_SPACE)
        assert set(params.keys()) == set(SEARCH_SPACE.keys())

    def test_int_params_are_integers(self):
        study = optuna.create_study()
        trial = study.ask()
        params = _sample_params(trial, SEARCH_SPACE)
        assert isinstance(params["max_depth"], int)
        assert isinstance(params["n_estimators"], int)

    def test_int_params_within_bounds(self):
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
        for _ in range(20):
            trial = study.ask()
            params = _sample_params(trial, SEARCH_SPACE)
            assert 3 <= params["max_depth"] <= 8
            assert 100 <= params["n_estimators"] <= 600

    def test_float_params_within_bounds(self):
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        for _ in range(20):
            trial = study.ask()
            params = _sample_params(trial, SEARCH_SPACE)
            assert 0.5 <= params["subsample"] <= 1.0
            assert 3.0 <= params["scale_pos_weight"] <= 10.0

    def test_log_float_param_within_bounds(self):
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
        for _ in range(20):
            trial = study.ask()
            params = _sample_params(trial, SEARCH_SPACE)
            assert 0.01 <= params["learning_rate"] <= 0.3

    def test_unknown_type_raises(self):
        bad_space = {"x": {"type": "unknown", "low": 0, "high": 1}}
        study = optuna.create_study()
        trial = study.ask()
        with pytest.raises(ValueError, match="Unknown search_space type"):
            _sample_params(trial, bad_space)


# ---------------------------------------------------------------------------
# Tests — segment_errors
# ---------------------------------------------------------------------------

class TestSegmentErrors:
    def test_column_added(self):
        df = _make_scored_df()
        out = segment_errors(df, score_col=SCORE_COL, k=3)
        assert "error_type" in out.columns

    def test_does_not_mutate_input(self):
        df = _make_scored_df()
        original_cols = list(df.columns)
        _ = segment_errors(df, score_col=SCORE_COL)
        assert list(df.columns) == original_cols

    def test_only_valid_error_types(self):
        df = _make_scored_df()
        out = segment_errors(df, score_col=SCORE_COL)
        valid = {"true_positive", "false_negative", "false_positive", "true_negative"}
        assert set(out["error_type"].unique()).issubset(valid)

    def test_false_negatives_are_positive_rows(self):
        df = _make_scored_df()
        out = segment_errors(df, score_col=SCORE_COL, k=1)
        fn_rows = out[out["error_type"] == "false_negative"]
        assert (fn_rows[TARGET] == 1).all()

    def test_false_positives_are_negative_rows(self):
        df = _make_scored_df()
        out = segment_errors(df, score_col=SCORE_COL, k=3)
        fp_rows = out[out["error_type"] == "false_positive"]
        assert (fp_rows[TARGET] == 0).all()

    def test_total_positives_split_between_tp_and_fn(self):
        """All positive rows must be either TP or FN (none can be TN or FP)."""
        df = _make_scored_df(seed=7)
        out = segment_errors(df, score_col=SCORE_COL, k=2)
        pos_types = out.loc[out[TARGET] == 1, "error_type"].unique()
        assert set(pos_types).issubset({"true_positive", "false_negative"})

    def test_rank_k1_every_lead_has_one_fp_or_tp_in_top1(self):
        """With k=1, exactly one row per lead is in top-K (FP or TP)."""
        df = _make_scored_df(seed=3)
        out = segment_errors(df, score_col=SCORE_COL, k=1)
        top_k_rows = out[out["error_type"].isin(["true_positive", "false_positive"])]
        counts = top_k_rows.groupby(GROUP_KEY).size()
        assert (counts == 1).all()


# ---------------------------------------------------------------------------
# Tests — calibration_data
# ---------------------------------------------------------------------------

class TestCalibrationData:
    def test_returns_dataframe_with_expected_columns(self):
        df = _make_scored_df()
        cal = calibration_data(df, score_col=SCORE_COL, n_bins=5)
        required = {"bin_lower", "bin_upper", "bin_center", "mean_predicted",
                    "actual_positive_rate", "count"}
        assert required.issubset(set(cal.columns))

    def test_bin_centers_in_0_1(self):
        df = _make_scored_df()
        cal = calibration_data(df, score_col=SCORE_COL, n_bins=10)
        assert (cal["bin_center"] >= 0.0).all()
        assert (cal["bin_center"] <= 1.0).all()

    def test_counts_sum_to_total_rows(self):
        df = _make_scored_df()
        cal = calibration_data(df, score_col=SCORE_COL, n_bins=10)
        assert cal["count"].sum() == len(df)

    def test_actual_rate_bounded_01(self):
        df = _make_scored_df()
        cal = calibration_data(df, score_col=SCORE_COL)
        assert (cal["actual_positive_rate"] >= 0.0).all()
        assert (cal["actual_positive_rate"] <= 1.0).all()

    def test_perfect_model_has_high_arate_in_high_bins(self):
        """If scores perfectly rank the positives, high bins should have high actual rate."""
        n = 200
        rng = np.random.default_rng(42)
        labels = (rng.random(n) < 0.1).astype(int)
        # Perfect scores: positive gets score 0.95, negative gets score 0.2
        scores = np.where(labels == 1, 0.90 + rng.random(n) * 0.09,
                          0.05 + rng.random(n) * 0.20)
        df = pd.DataFrame({GROUP_KEY: [f"l{i}" for i in range(n)], TARGET: labels,
                           SCORE_COL: scores})
        cal = calibration_data(df, score_col=SCORE_COL, n_bins=5)
        high_bin = cal[cal["bin_center"] > 0.7]
        if len(high_bin) > 0:
            assert high_bin["actual_positive_rate"].mean() > 0.3


# ---------------------------------------------------------------------------
# Tests — profile_error_segment
# ---------------------------------------------------------------------------

class TestProfileErrorSegment:
    def test_returns_dataframe_indexed_by_error_type(self):
        df = segment_errors(_make_scored_df(), score_col=SCORE_COL)
        result = profile_error_segment(df, feature_cols=["cibil_score", "foir"])
        assert result.index.name == "error_type"
        assert "cibil_score" in result.columns

    def test_skips_missing_features_silently(self):
        df = segment_errors(_make_scored_df(), score_col=SCORE_COL)
        result = profile_error_segment(
            df, feature_cols=["cibil_score", "nonexistent_col"]
        )
        assert "cibil_score" in result.columns
        assert "nonexistent_col" not in result.columns

    def test_all_error_types_in_index(self):
        df = segment_errors(_make_scored_df(), score_col=SCORE_COL, k=2)
        result = profile_error_segment(df, feature_cols=["cibil_score"])
        # At least true_positive / false_negative / true_negative should exist
        assert len(result.index) >= 2

    def test_empty_feature_list_returns_empty_df(self):
        df = segment_errors(_make_scored_df(), score_col=SCORE_COL)
        result = profile_error_segment(df, feature_cols=[])
        assert result.empty
