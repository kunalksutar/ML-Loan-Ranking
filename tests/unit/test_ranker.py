"""
Unit tests for src/modeling/ranker.py (CLAUDE.md §16).

Covers:
  - rank_pairs: output sorted descending per lead, top-k length, scores in [0, 1]
  - Ranker.rank_lead: no-eligible-bank edge case, basic result shape
  - RankedBank dataclass field correctness
  - score_feature_matrix: output shape and probability bounds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.ranker import Ranker, RankedBank, RankResult, rank_pairs, score_feature_matrix


# ---------------------------------------------------------------------------
# Minimal stubs for model and preprocessor
# ---------------------------------------------------------------------------

class _StubPreprocessor:
    """Passthrough preprocessor that returns X unchanged (expects numeric input)."""

    def transform(self, X):
        if hasattr(X, "values"):
            return X.values.astype(float)
        return np.asarray(X, dtype=float)


class _StubModel:
    """Deterministic model: score = mean of all input features (normalised to [0,1])."""

    def predict_proba(self, X):
        raw = np.asarray(X, dtype=float).mean(axis=1)
        # Sigmoid so output is always in (0, 1)
        probs = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - probs, probs])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _feature_df(n_rows: int = 10, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    """Synthetic feature DataFrame with numeric columns."""
    from src.features.feature_registry import ALL_FEATURES
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, len(ALL_FEATURES)))
    df = pd.DataFrame(data, columns=ALL_FEATURES)
    df["lead_id"] = [f"lead-{i // 2}" for i in range(n_rows)]   # 2 rows per lead
    df["converted"] = rng.integers(0, 2, size=n_rows)
    return df


# ---------------------------------------------------------------------------
# score_feature_matrix
# ---------------------------------------------------------------------------

class TestScoreFeatureMatrix:
    def test_output_shape(self):
        df = _feature_df(n_rows=8)
        scores = score_feature_matrix(df, _StubModel(), _StubPreprocessor())
        assert scores.shape == (8,)

    def test_scores_in_unit_interval(self):
        df = _feature_df(n_rows=20)
        scores = score_feature_matrix(df, _StubModel(), _StubPreprocessor())
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_no_nans(self):
        df = _feature_df(n_rows=6)
        scores = score_feature_matrix(df, _StubModel(), _StubPreprocessor())
        assert not np.isnan(scores).any()


# ---------------------------------------------------------------------------
# rank_pairs
# ---------------------------------------------------------------------------

class TestRankPairs:
    def test_scores_added_as_column(self):
        df = _feature_df(n_rows=10)
        result = rank_pairs(df, _StubModel(), _StubPreprocessor(), top_k=None)
        assert "rank_score" in result.columns

    def test_scores_in_unit_interval(self):
        df = _feature_df(n_rows=10)
        result = rank_pairs(df, _StubModel(), _StubPreprocessor(), top_k=None)
        assert result["rank_score"].between(0.0, 1.0).all()

    def test_sorted_descending_per_lead(self):
        df = _feature_df(n_rows=20, seed=7)   # 10 leads, 2 rows each
        result = rank_pairs(df, _StubModel(), _StubPreprocessor(), top_k=None)
        for lead_id, grp in result.groupby("lead_id", sort=False):
            scores = grp["rank_score"].values
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), \
                f"Not sorted descending for lead {lead_id}: {scores}"

    def test_top_k_limits_rows_per_lead(self):
        n_rows_per_lead = 5
        n_leads = 4
        df = _feature_df(n_rows=n_leads * n_rows_per_lead, seed=3)
        df["lead_id"] = [f"lead-{i // n_rows_per_lead}" for i in range(len(df))]
        top_k = 3
        result = rank_pairs(df, _StubModel(), _StubPreprocessor(), top_k=top_k)
        for _, grp in result.groupby("lead_id", sort=False):
            assert len(grp) <= top_k

    def test_top_k_none_keeps_all_rows(self):
        df = _feature_df(n_rows=12)
        result = rank_pairs(df, _StubModel(), _StubPreprocessor(), top_k=None)
        assert len(result) == len(df)

    def test_top_k_1_returns_one_row_per_lead(self):
        n_leads = 5
        df = _feature_df(n_rows=n_leads * 4, seed=99)
        df["lead_id"] = [f"lead-{i // 4}" for i in range(len(df))]
        result = rank_pairs(df, _StubModel(), _StubPreprocessor(), top_k=1)
        assert len(result) == n_leads


# ---------------------------------------------------------------------------
# Ranker.rank_lead — no-eligible-bank edge case
# ---------------------------------------------------------------------------

def _make_ranker_fixtures():
    """Return (lead_df, banks_df) that always produce 0 eligible pairs."""
    lead = pd.DataFrame([{
        "lead_id": "lead-test",
        "income_type": "salaried",
        "state": "MH",
        "cibil_score": 750,
        "annual_income": 1_200_000.0,
        "foir": 0.30,
        "age_at_maturity": 55,
        "enquiry_count_6m": 1,
        "dpd_90_count": 0,
        "written_off_loans": 0,
        "loan_type": "personal",
        "loan_amount_requested": 500_000.0,
    }])
    # Bank that will reject everyone (impossibly high CIBIL minimum)
    bank_reject = pd.DataFrame([{
        "bank_id": "bank-hard",
        "name": "Impossible Bank",
        "bank_type": "PSB",
        "accepted_income_types": ["salaried"],
        "states_covered": ["MH"],
        "min_cibil_score": 960,
        "max_cibil_score": 900,
        "min_annual_income": 100_000.0,
        "max_annual_income": 50_000_000.0,
        "max_foir": 0.70,
        "max_age_at_maturity": 70,
        "max_enquiries_6m": 5,
        "max_dpd_90_count": 2,
        "max_written_off_loans": 1,
        "loan_types_offered": ["personal"],
        "min_loan_amount": 100_000.0,
        "max_loan_amount": 10_000_000.0,
        "interest_rate_min": 10.5,
        "disbursal_speed_days": 7,
    }])
    return lead, bank_reject


class TestRankerRankLead:
    def test_no_eligible_returns_empty_list(self):
        lead, banks = _make_ranker_fixtures()
        ranker = Ranker(_StubModel(), _StubPreprocessor(), top_k=5)
        result = ranker.rank_lead(lead, banks)
        assert isinstance(result, RankResult)
        assert result.n_eligible_banks == 0
        assert result.ranked_banks == []

    def test_lead_id_propagated(self):
        lead, banks = _make_ranker_fixtures()
        ranker = Ranker(_StubModel(), _StubPreprocessor(), top_k=5)
        result = ranker.rank_lead(lead, banks)
        assert result.lead_id == "lead-test"

    def test_latency_ms_is_positive(self):
        lead, banks = _make_ranker_fixtures()
        ranker = Ranker(_StubModel(), _StubPreprocessor(), top_k=5)
        result = ranker.rank_lead(lead, banks)
        assert result.latency_ms >= 0.0


def _make_eligible_fixtures():
    """Return (lead_df, banks_df, feature_df) where the bank is eligible."""
    from src.features.feature_registry import ALL_FEATURES

    lead = pd.DataFrame([{
        "lead_id": "lead-ok",
        "income_type": "salaried",
        "state": "MH",
        "cibil_score": 750,
        "annual_income": 1_200_000.0,
        "foir": 0.30,
        "age_at_maturity": 55,
        "enquiry_count_6m": 1,
        "dpd_90_count": 0,
        "written_off_loans": 0,
        "loan_type": "personal",
        "loan_amount_requested": 500_000.0,
    }])
    bank = pd.DataFrame([{
        "bank_id": "bank-easy",
        "name": "Easy Bank",
        "bank_type": "NBFC",
        "accepted_income_types": ["salaried", "self_employed"],
        "states_covered": ["MH", "DL"],
        "min_cibil_score": 600,
        "max_cibil_score": 900,
        "min_annual_income": 300_000.0,
        "max_annual_income": 50_000_000.0,
        "max_foir": 0.75,
        "max_age_at_maturity": 70,
        "max_enquiries_6m": 5,
        "max_dpd_90_count": 2,
        "max_written_off_loans": 1,
        "loan_types_offered": ["personal", "home"],
        "min_loan_amount": 100_000.0,
        "max_loan_amount": 10_000_000.0,
        "interest_rate_min": 11.0,
        "disbursal_speed_days": 5,
    }])
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        rng.standard_normal((1, len(ALL_FEATURES))),
        columns=ALL_FEATURES,
    )
    features["bank_id"] = "bank-easy"
    return lead, bank, features


class TestRankerWithEligibleBank:
    def test_ranked_bank_count_matches_n_eligible(self):
        lead, bank, features = _make_eligible_fixtures()

        def feature_builder(ld, bk):
            return features[features["bank_id"].isin(bk["bank_id"])]

        ranker = Ranker(_StubModel(), _StubPreprocessor(), top_k=5)
        result = ranker.rank_lead(lead, bank, feature_builder_fn=feature_builder)
        assert result.n_eligible_banks == 1
        assert len(result.ranked_banks) == 1

    def test_ranked_bank_score_in_unit_interval(self):
        lead, bank, features = _make_eligible_fixtures()

        def feature_builder(ld, bk):
            return features[features["bank_id"].isin(bk["bank_id"])]

        ranker = Ranker(_StubModel(), _StubPreprocessor(), top_k=5)
        result = ranker.rank_lead(lead, bank, feature_builder_fn=feature_builder)
        for rb in result.ranked_banks:
            assert 0.0 <= rb.rank_score <= 1.0

    def test_ranked_bank_rank_starts_at_one(self):
        lead, bank, features = _make_eligible_fixtures()

        def feature_builder(ld, bk):
            return features[features["bank_id"].isin(bk["bank_id"])]

        ranker = Ranker(_StubModel(), _StubPreprocessor(), top_k=5)
        result = ranker.rank_lead(lead, bank, feature_builder_fn=feature_builder)
        assert result.ranked_banks[0].rank == 1

    def test_ranked_banks_sorted_descending_by_score(self):
        from src.features.feature_registry import ALL_FEATURES

        lead, _, _ = _make_eligible_fixtures()
        # Build 3 banks, all eligible
        banks = pd.DataFrame([
            {
                "bank_id": f"bank-{j}",
                "name": f"Bank {j}",
                "bank_type": "NBFC",
                "accepted_income_types": ["salaried"],
                "states_covered": ["MH"],
                "min_cibil_score": 600,
                "max_cibil_score": 900,
                "min_annual_income": 100_000.0,
                "max_annual_income": 50_000_000.0,
                "max_foir": 0.80,
                "max_age_at_maturity": 75,
                "max_enquiries_6m": 10,
                "max_dpd_90_count": 5,
                "max_written_off_loans": 3,
                "loan_types_offered": ["personal"],
                "min_loan_amount": 50_000.0,
                "max_loan_amount": 20_000_000.0,
                "interest_rate_min": 10.0 + j,
                "disbursal_speed_days": 3,
            }
            for j in range(3)
        ])
        rng = np.random.default_rng(1)
        features = pd.DataFrame(
            rng.standard_normal((3, len(ALL_FEATURES))),
            columns=ALL_FEATURES,
        )
        features["bank_id"] = ["bank-0", "bank-1", "bank-2"]

        def feature_builder(ld, bk):
            return features[features["bank_id"].isin(bk["bank_id"])]

        ranker = Ranker(_StubModel(), _StubPreprocessor(), top_k=5)
        result = ranker.rank_lead(lead, banks, feature_builder_fn=feature_builder)
        scores = [rb.rank_score for rb in result.ranked_banks]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
