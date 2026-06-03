"""
Unit tests for src.simulation.lead_generator.

Tests cover:
  - Schema invariants (no nulls, column presence, type checks)
  - Domain bounds (age, CIBIL, FOIR, maturity, loan amount)
  - Causal correlations (CIBIL↑ with income, DPD↓ with CIBIL)
  - Business rules (age_at_maturity, loan tenure constraints)
  - Determinism (same seed → same output)
  - Edge cases (n=1, small n)
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation.lead_generator import generate_leads, validate_leads
from src.simulation.distributions import (
    INCOME_TYPES,
    CITY_TIERS,
    LOAN_TYPES,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SMALL = 500
N_LARGE = 5000
SEED = 42


@pytest.fixture(scope="module")
def leads_small() -> pd.DataFrame:
    return generate_leads(n=N_SMALL, seed=SEED)


@pytest.fixture(scope="module")
def leads_large() -> pd.DataFrame:
    return generate_leads(n=N_LARGE, seed=SEED)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    REQUIRED_COLUMNS = [
        "lead_id", "created_at", "age", "gender", "city_tier", "state",
        "pin_code", "income_type", "employer_category", "annual_income",
        "work_experience_years", "current_employer_tenure_yrs",
        "cibil_score", "dpd_30_count", "dpd_90_count", "enquiry_count_6m",
        "settled_loans", "written_off_loans", "existing_loan_count",
        "monthly_obligations", "credit_card_spend_monthly", "savings_balance",
        "fixed_deposits", "loan_type", "loan_amount_requested",
        "loan_tenure_months", "foir", "dti_ratio", "loan_to_income_ratio",
        "credit_utilization", "age_at_maturity",
    ]

    def test_all_columns_present(self, leads_small):
        for col in self.REQUIRED_COLUMNS:
            assert col in leads_small.columns, f"Missing column: {col}"

    def test_no_null_values(self, leads_small):
        null_counts = leads_small.isnull().sum()
        failing = null_counts[null_counts > 0]
        assert failing.empty, f"Null values found:\n{failing}"

    def test_row_count(self, leads_small):
        assert len(leads_small) == N_SMALL

    def test_lead_ids_unique(self, leads_small):
        assert leads_small["lead_id"].nunique() == N_SMALL

    def test_income_type_values(self, leads_small):
        assert set(leads_small["income_type"].unique()).issubset(set(INCOME_TYPES))

    def test_city_tier_values(self, leads_small):
        assert set(leads_small["city_tier"].unique()).issubset(set(CITY_TIERS))

    def test_loan_type_values(self, leads_small):
        assert set(leads_small["loan_type"].unique()).issubset(set(LOAN_TYPES))

    def test_gender_values(self, leads_small):
        assert set(leads_small["gender"].unique()).issubset({"M", "F", "Other"})


# ---------------------------------------------------------------------------
# Domain bounds tests
# ---------------------------------------------------------------------------

class TestDomainBounds:
    def test_age_range(self, leads_small):
        assert leads_small["age"].between(23, 62).all(), (
            f"Age out of [23,62]: min={leads_small['age'].min()}, max={leads_small['age'].max()}"
        )

    def test_cibil_range(self, leads_small):
        assert leads_small["cibil_score"].between(300, 900).all(), (
            f"CIBIL out of [300,900]: min={leads_small['cibil_score'].min()}, max={leads_small['cibil_score'].max()}"
        )

    def test_foir_bounds(self, leads_small):
        assert (leads_small["foir"] > 0.05).all(), f"FOIR min too low: {leads_small['foir'].min():.4f}"
        assert (leads_small["foir"] < 0.95).all(), f"FOIR max too high: {leads_small['foir'].max():.4f}"

    def test_age_at_maturity_cap(self, leads_small):
        assert leads_small["age_at_maturity"].max() < 80, (
            f"age_at_maturity exceeds 79: {leads_small['age_at_maturity'].max()}"
        )

    def test_annual_income_positive(self, leads_small):
        assert (leads_small["annual_income"] > 0).all()

    def test_loan_amount_positive(self, leads_small):
        assert (leads_small["loan_amount_requested"] > 0).all()

    def test_loan_tenure_minimum(self, leads_small):
        assert (leads_small["loan_tenure_months"] >= 12).all(), (
            f"Loan tenure below 12 months: min={leads_small['loan_tenure_months'].min()}"
        )

    def test_enquiry_count_non_negative(self, leads_small):
        assert (leads_small["enquiry_count_6m"] >= 0).all()

    def test_dpd_counts_non_negative(self, leads_small):
        assert (leads_small["dpd_30_count"] >= 0).all()
        assert (leads_small["dpd_90_count"] >= 0).all()

    def test_credit_utilization_range(self, leads_small):
        assert (leads_small["credit_utilization"] >= 0).all()
        assert (leads_small["credit_utilization"] <= 1).all()

    def test_savings_balance_positive(self, leads_small):
        assert (leads_small["savings_balance"] > 0).all()

    def test_monthly_obligations_positive(self, leads_small):
        assert (leads_small["monthly_obligations"] > 0).all()

    def test_work_experience_non_negative(self, leads_small):
        assert (leads_small["work_experience_years"] >= 0).all()

    def test_employer_tenure_lte_work_experience(self, leads_small):
        # Employer tenure should never exceed total work experience
        assert (
            leads_small["current_employer_tenure_yrs"] <=
            leads_small["work_experience_years"] + 0.1  # small tolerance for rounding
        ).all()


# ---------------------------------------------------------------------------
# Causal correlation tests (requires larger sample for stability)
# ---------------------------------------------------------------------------

class TestCausalCorrelations:
    def test_cibil_income_positive_correlation(self, leads_large):
        """CIBIL must be positively correlated with income (causal chain invariant)."""
        corr = leads_large["cibil_score"].corr(leads_large["annual_income"])
        assert corr > 0.30, f"CIBIL-income correlation too low: {corr:.3f} (expected > 0.30)"

    def test_cibil_dpd30_negative_correlation(self, leads_large):
        """DPD30 must be negatively correlated with CIBIL (delinquency ↑ as CIBIL ↓)."""
        corr = leads_large["dpd_30_count"].corr(leads_large["cibil_score"])
        assert corr < -0.25, f"CIBIL-DPD30 correlation wrong: {corr:.3f} (expected < -0.25)"

    def test_cibil_dpd90_negative_correlation(self, leads_large):
        corr = leads_large["dpd_90_count"].corr(leads_large["cibil_score"])
        assert corr < 0, f"CIBIL-DPD90 correlation should be negative: {corr:.3f}"

    def test_income_age_positive_correlation(self, leads_large):
        """Older leads should earn more on average (career progression)."""
        corr = leads_large["annual_income"].corr(leads_large["age"])
        assert corr > 0.10, f"Income-age correlation too low: {corr:.3f} (expected > 0.10)"

    def test_foir_derived_consistently(self, leads_small):
        """FOIR must equal monthly_obligations / (annual_income / 12) within rounding."""
        computed = leads_small["monthly_obligations"] / (leads_small["annual_income"] / 12)
        diff = (leads_small["foir"] - computed).abs()
        assert diff.max() < 1e-3, f"FOIR derived inconsistently: max diff={diff.max():.6f}"


# ---------------------------------------------------------------------------
# Business rule tests
# ---------------------------------------------------------------------------

class TestBusinessRules:
    def test_age_at_maturity_formula(self, leads_small):
        """age_at_maturity must equal age + floor(loan_tenure_months / 12)."""
        expected = leads_small["age"] + (leads_small["loan_tenure_months"] // 12)
        diff = (leads_small["age_at_maturity"] - expected).abs()
        assert diff.max() == 0, f"age_at_maturity formula mismatch: max diff={diff.max()}"

    def test_income_distribution_by_type(self, leads_large):
        """Business owners should have higher median income than freelancers."""
        medians = leads_large.groupby("income_type")["annual_income"].median()
        assert medians.get("business", 0) > medians.get("freelance", 0), (
            "Business income should exceed freelance income on average"
        )

    def test_high_cibil_lower_dpd(self, leads_large):
        """High-CIBIL leads (750+) must have fewer DPD events than low-CIBIL (<600)."""
        high_cibil = leads_large[leads_large["cibil_score"] >= 750]["dpd_30_count"].mean()
        low_cibil = leads_large[leads_large["cibil_score"] < 600]["dpd_30_count"].mean()
        assert high_cibil < low_cibil, (
            f"High-CIBIL DPD30 ({high_cibil:.2f}) should be < low-CIBIL DPD30 ({low_cibil:.2f})"
        )

    def test_income_type_distribution(self, leads_large):
        """Salaried should be the dominant income type (~55%)."""
        frac = (leads_large["income_type"] == "salaried").mean()
        assert 0.45 <= frac <= 0.65, f"Salaried fraction {frac:.2f} outside expected [0.45, 0.65]"

    def test_fixed_deposits_partial(self, leads_large):
        """Only ~40% of leads should have fixed deposits (the rest are zero)."""
        frac_with_fd = (leads_large["fixed_deposits"] > 0).mean()
        assert 0.30 <= frac_with_fd <= 0.55, (
            f"Fixed deposit prevalence {frac_with_fd:.2f} outside expected [0.30, 0.55]"
        )


# ---------------------------------------------------------------------------
# Determinism and edge-case tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_output(self):
        df1 = generate_leads(n=100, seed=42)
        df2 = generate_leads(n=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_output(self):
        df1 = generate_leads(n=100, seed=42)
        df2 = generate_leads(n=100, seed=99)
        assert not df1["cibil_score"].equals(df2["cibil_score"])

    def test_single_lead(self):
        df = generate_leads(n=1, seed=42)
        assert len(df) == 1
        assert df.isnull().sum().sum() == 0

    def test_large_n_no_crash(self):
        df = generate_leads(n=1000, seed=0)
        assert len(df) == 1000


# ---------------------------------------------------------------------------
# validate_leads integration test
# ---------------------------------------------------------------------------

class TestValidateLeads:
    def test_validate_passes_on_valid_data(self, leads_large):
        """validate_leads should not raise on correctly generated data."""
        validate_leads(leads_large)

    def test_validate_fails_on_null(self):
        df = generate_leads(n=200, seed=42)
        df.loc[0, "annual_income"] = np.nan
        with pytest.raises(AssertionError, match="Null values found"):
            validate_leads(df)

    def test_validate_fails_on_bad_cibil(self):
        df = generate_leads(n=200, seed=42)
        df.loc[0, "cibil_score"] = 950  # above 900
        with pytest.raises(AssertionError, match="CIBIL score"):
            validate_leads(df)
