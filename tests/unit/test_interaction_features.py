"""
Unit tests for all 15 interaction feature formulas (CLAUDE.md §16).

Each test class targets one or more interaction features, verifies the exact
formula, and checks boundary / edge-case behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_registry import INTERACTION_FEATURES
from src.features.interaction_features import compute_interaction_features


# ---------------------------------------------------------------------------
# Test fixture helper
# ---------------------------------------------------------------------------

def _row(**overrides) -> pd.DataFrame:
    """Return a single-row merged DataFrame with sensible defaults."""
    defaults: dict = {
        # Lead fields
        "cibil_score": 750,
        "foir": 0.35,
        "annual_income": 600_000.0,
        "loan_amount_requested": 300_000.0,
        "enquiry_count_6m": 2,
        "dpd_90_count": 0,
        "age_at_maturity": 50,
        "income_type": "salaried",
        "loan_type": "personal",
        "state": "MH",
        # Bank fields
        "min_cibil_score": 700,
        "max_foir": 0.65,
        "min_annual_income": 200_000.0,
        "min_loan_amount": 50_000.0,
        "max_loan_amount": 2_000_000.0,
        "max_enquiries_6m": 3,
        "max_dpd_90_count": 0,
        "preferred_cibil_min": 730,
        "preferred_cibil_max": 800,
        "max_age_at_maturity": 68,
        "accepted_income_types": ["salaried", "business"],
        "loan_types_offered": ["personal", "home", "car"],
        "states_covered": ["MH", "DL", "KA"],
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def _compute(row_kwargs: dict | None = None, **overrides) -> pd.Series:
    """Compute all interaction features on a single row; return the row as a Series."""
    df = _row(**(row_kwargs or {}), **overrides)
    return compute_interaction_features(df).iloc[0]


# ---------------------------------------------------------------------------
# 1. cibil_gap
# ---------------------------------------------------------------------------

class TestCibilGap:
    def test_positive(self):
        r = _compute(cibil_score=750, min_cibil_score=700)
        assert r["cibil_gap"] == pytest.approx(50.0)

    def test_negative(self):
        r = _compute(cibil_score=680, min_cibil_score=700)
        assert r["cibil_gap"] == pytest.approx(-20.0)

    def test_zero(self):
        r = _compute(cibil_score=700, min_cibil_score=700)
        assert r["cibil_gap"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. foir_headroom
# ---------------------------------------------------------------------------

class TestFoirHeadroom:
    def test_positive(self):
        r = _compute(foir=0.35, max_foir=0.65)
        assert r["foir_headroom"] == pytest.approx(0.30)

    def test_negative(self):
        r = _compute(foir=0.75, max_foir=0.65)
        assert r["foir_headroom"] == pytest.approx(-0.10)

    def test_zero(self):
        r = _compute(foir=0.65, max_foir=0.65)
        assert r["foir_headroom"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. income_headroom + income_headroom_ratio
# ---------------------------------------------------------------------------

class TestIncomeHeadroom:
    def test_income_headroom_positive(self):
        r = _compute(annual_income=600_000.0, min_annual_income=200_000.0)
        assert r["income_headroom"] == pytest.approx(400_000.0)

    def test_income_headroom_negative(self):
        r = _compute(annual_income=150_000.0, min_annual_income=200_000.0)
        assert r["income_headroom"] == pytest.approx(-50_000.0)

    def test_income_headroom_ratio(self):
        # headroom = 400K, min = 200K → ratio = 2.0
        r = _compute(annual_income=600_000.0, min_annual_income=200_000.0)
        assert r["income_headroom_ratio"] == pytest.approx(2.0)

    def test_income_headroom_ratio_negative(self):
        r = _compute(annual_income=100_000.0, min_annual_income=200_000.0)
        assert r["income_headroom_ratio"] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# 4. amount_fit_flag + amount_position
# ---------------------------------------------------------------------------

class TestAmountFit:
    def test_flag_in_range(self):
        r = _compute(loan_amount_requested=300_000.0,
                     min_loan_amount=50_000.0, max_loan_amount=2_000_000.0)
        assert r["amount_fit_flag"] == 1

    def test_flag_below_range(self):
        r = _compute(loan_amount_requested=10_000.0,
                     min_loan_amount=50_000.0, max_loan_amount=2_000_000.0)
        assert r["amount_fit_flag"] == 0

    def test_flag_above_range(self):
        r = _compute(loan_amount_requested=5_000_000.0,
                     min_loan_amount=50_000.0, max_loan_amount=2_000_000.0)
        assert r["amount_fit_flag"] == 0

    def test_flag_at_boundary(self):
        # exactly at min and max → should be in range
        r = _compute(loan_amount_requested=50_000.0,
                     min_loan_amount=50_000.0, max_loan_amount=2_000_000.0)
        assert r["amount_fit_flag"] == 1
        r2 = _compute(loan_amount_requested=2_000_000.0,
                      min_loan_amount=50_000.0, max_loan_amount=2_000_000.0)
        assert r2["amount_fit_flag"] == 1

    def test_position_mid(self):
        # midpoint = (50K + 2M) / 2 = 1_025_000 → position ≈ 0.5
        r = _compute(loan_amount_requested=1_025_000.0,
                     min_loan_amount=50_000.0, max_loan_amount=2_000_000.0)
        assert r["amount_position"] == pytest.approx(0.5, abs=0.005)

    def test_position_clipped_above_1(self):
        r = _compute(loan_amount_requested=10_000_000.0,
                     min_loan_amount=50_000.0, max_loan_amount=2_000_000.0)
        assert r["amount_position"] == pytest.approx(1.0)

    def test_position_clipped_below_0(self):
        r = _compute(loan_amount_requested=0.0,
                     min_loan_amount=50_000.0, max_loan_amount=2_000_000.0)
        assert r["amount_position"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. income_type_match
# ---------------------------------------------------------------------------

class TestIncomeTypeMatch:
    def test_match_true(self):
        r = _compute(income_type="salaried",
                     accepted_income_types=["salaried", "business"])
        assert r["income_type_match"] == 1

    def test_match_false(self):
        r = _compute(income_type="freelance",
                     accepted_income_types=["salaried", "business"])
        assert r["income_type_match"] == 0

    def test_single_accepted_type(self):
        r = _compute(income_type="salaried", accepted_income_types=["salaried"])
        assert r["income_type_match"] == 1


# ---------------------------------------------------------------------------
# 6. loan_type_match
# ---------------------------------------------------------------------------

class TestLoanTypeMatch:
    def test_match_true(self):
        r = _compute(loan_type="personal",
                     loan_types_offered=["personal", "home"])
        assert r["loan_type_match"] == 1

    def test_match_false(self):
        r = _compute(loan_type="gold",
                     loan_types_offered=["personal", "home"])
        assert r["loan_type_match"] == 0

    def test_hfc_home_match(self):
        r = _compute(loan_type="home", loan_types_offered=["home", "lap"])
        assert r["loan_type_match"] == 1

    def test_hfc_personal_no_match(self):
        r = _compute(loan_type="personal", loan_types_offered=["home", "lap"])
        assert r["loan_type_match"] == 0


# ---------------------------------------------------------------------------
# 7. geography_match
# ---------------------------------------------------------------------------

class TestGeographyMatch:
    def test_match_true(self):
        r = _compute(state="MH", states_covered=["MH", "DL", "KA"])
        assert r["geography_match"] == 1

    def test_match_false(self):
        r = _compute(state="RJ", states_covered=["MH", "DL", "KA"])
        assert r["geography_match"] == 0

    def test_single_state_covered(self):
        r = _compute(state="DL", states_covered=["DL"])
        assert r["geography_match"] == 1


# ---------------------------------------------------------------------------
# 8. bureau_fatigue_flag + bureau_fatigue_excess
# ---------------------------------------------------------------------------

class TestBureauFatigue:
    def test_flag_triggered(self):
        r = _compute(enquiry_count_6m=5, max_enquiries_6m=3)
        assert r["bureau_fatigue_flag"] == 1

    def test_flag_not_triggered(self):
        r = _compute(enquiry_count_6m=2, max_enquiries_6m=3)
        assert r["bureau_fatigue_flag"] == 0

    def test_flag_at_limit_not_triggered(self):
        # rule is strict greater-than (>), not >=
        r = _compute(enquiry_count_6m=3, max_enquiries_6m=3)
        assert r["bureau_fatigue_flag"] == 0

    def test_excess_value(self):
        r = _compute(enquiry_count_6m=5, max_enquiries_6m=3)
        assert r["bureau_fatigue_excess"] == pytest.approx(2.0)

    def test_excess_zero_when_not_fatigued(self):
        r = _compute(enquiry_count_6m=1, max_enquiries_6m=3)
        assert r["bureau_fatigue_excess"] == pytest.approx(0.0)

    def test_excess_zero_at_limit(self):
        r = _compute(enquiry_count_6m=3, max_enquiries_6m=3)
        assert r["bureau_fatigue_excess"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. cibil_in_sweet_spot + cibil_vs_sweet_spot_dist
# ---------------------------------------------------------------------------

class TestCibilSweetSpot:
    def test_in_sweet_spot(self):
        r = _compute(cibil_score=760, preferred_cibil_min=730, preferred_cibil_max=800)
        assert r["cibil_in_sweet_spot"] == 1

    def test_below_sweet_spot(self):
        r = _compute(cibil_score=720, preferred_cibil_min=730, preferred_cibil_max=800)
        assert r["cibil_in_sweet_spot"] == 0

    def test_above_sweet_spot(self):
        r = _compute(cibil_score=820, preferred_cibil_min=730, preferred_cibil_max=800)
        assert r["cibil_in_sweet_spot"] == 0

    def test_at_boundary_min(self):
        r = _compute(cibil_score=730, preferred_cibil_min=730, preferred_cibil_max=800)
        assert r["cibil_in_sweet_spot"] == 1

    def test_at_boundary_max(self):
        r = _compute(cibil_score=800, preferred_cibil_min=730, preferred_cibil_max=800)
        assert r["cibil_in_sweet_spot"] == 1

    def test_distance_at_center(self):
        # center = (730 + 800) / 2 = 765 → dist = |765 - 765| = 0
        r = _compute(cibil_score=765, preferred_cibil_min=730, preferred_cibil_max=800)
        assert r["cibil_vs_sweet_spot_dist"] == pytest.approx(0.0)

    def test_distance_below_center(self):
        # center = 765, cibil = 750 → dist = 15
        r = _compute(cibil_score=750, preferred_cibil_min=730, preferred_cibil_max=800)
        assert r["cibil_vs_sweet_spot_dist"] == pytest.approx(15.0)

    def test_distance_above_center(self):
        # center = 765, cibil = 790 → dist = 25
        r = _compute(cibil_score=790, preferred_cibil_min=730, preferred_cibil_max=800)
        assert r["cibil_vs_sweet_spot_dist"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# 10. age_maturity_headroom
# ---------------------------------------------------------------------------

class TestAgeMaturityHeadroom:
    def test_positive_headroom(self):
        r = _compute(age_at_maturity=50, max_age_at_maturity=68)
        assert r["age_maturity_headroom"] == pytest.approx(18.0)

    def test_negative_headroom(self):
        r = _compute(age_at_maturity=75, max_age_at_maturity=68)
        assert r["age_maturity_headroom"] == pytest.approx(-7.0)

    def test_zero_headroom(self):
        r = _compute(age_at_maturity=68, max_age_at_maturity=68)
        assert r["age_maturity_headroom"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11. dpd90_exceeds_bank_max
# ---------------------------------------------------------------------------

class TestDpd90ExceedsBankMax:
    def test_exceeds(self):
        r = _compute(dpd_90_count=2, max_dpd_90_count=0)
        assert r["dpd90_exceeds_bank_max"] == 1

    def test_does_not_exceed(self):
        r = _compute(dpd_90_count=0, max_dpd_90_count=0)
        assert r["dpd90_exceeds_bank_max"] == 0

    def test_at_limit_not_exceeded(self):
        # rule is strict >, not >=
        r = _compute(dpd_90_count=1, max_dpd_90_count=1)
        assert r["dpd90_exceeds_bank_max"] == 0

    def test_at_limit_plus_one(self):
        r = _compute(dpd_90_count=2, max_dpd_90_count=1)
        assert r["dpd90_exceeds_bank_max"] == 1


# ---------------------------------------------------------------------------
# All 15 features present
# ---------------------------------------------------------------------------

class TestAllInteractionFeaturesPresent:
    def test_all_15_columns_computed(self):
        df = _row()
        result = compute_interaction_features(df)
        for feat in INTERACTION_FEATURES:
            assert feat in result.columns, f"Missing: {feat}"

    def test_multi_row_batch(self):
        """Verify vectorised computation works on multiple rows."""
        rows = pd.concat([_row(cibil_score=c) for c in [600, 700, 800]], ignore_index=True)
        result = compute_interaction_features(rows)
        assert len(result) == 3
        # cibil_gap should vary by row
        assert result["cibil_gap"].nunique() == 3

    def test_no_nulls_in_output(self):
        df = _row()
        result = compute_interaction_features(df)
        for feat in INTERACTION_FEATURES:
            assert not result[feat].isnull().any(), f"Null in {feat}"
