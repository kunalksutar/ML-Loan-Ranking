"""
Unit tests for src.simulation.bank_generator.

Tests cover:
  - Schema: all required columns present, no nulls, correct dtypes
  - Archetype counts: correct number of banks per type
  - Domain bounds: CIBIL ranges, FOIR, income, approval rates per archetype
  - Business rules: sweet spot above minimum, loan range ordering
  - Differentiation: unique intercepts, variance across banks
  - List fields: non-empty, valid elements
  - Determinism: same seed → identical output
  - validate_banks integration
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.simulation.bank_generator import generate_banks, validate_banks
from src.simulation.distributions import (
    INCOME_TYPES,
    LOAN_TYPES,
    STATE_WEIGHTS,
)

ARCHETYPE_PATH = "configs/bank_archetypes.yaml"
SEED = 42

EXPECTED_COUNTS = {
    "PSB":     8,
    "private": 10,
    "NBFC":    8,
    "fintech": 6,
    "HFC":     4,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())  # 36

VALID_INCOME_TYPES     = set(INCOME_TYPES)
VALID_LOAN_TYPES       = set(LOAN_TYPES)
VALID_STATES           = set(STATE_WEIGHTS.keys())
VALID_RISK_APPETITES   = {"conservative", "moderate", "aggressive"}
VALID_DOC_STRICTNESS   = {"low", "medium", "high"}
VALID_EMPLOYER_CATS    = {"PSU", "private_listed", "private_unlisted", "MNC", "govt"}

REQUIRED_COLUMNS = [
    "bank_id", "name", "bank_type",
    "states_covered", "city_tiers_served", "digital_only",
    "loan_types_offered",
    "min_cibil_score", "max_cibil_score",
    "min_annual_income", "max_annual_income",
    "max_foir", "max_dti_ratio",
    "min_age", "max_age_at_maturity",
    "max_enquiries_6m",
    "max_dpd_30_count", "max_dpd_90_count",
    "max_written_off_loans", "max_settled_loans",
    "accepted_income_types", "accepted_employer_categories",
    "premium_employer_categories",
    "min_employer_tenure_months", "min_work_experience_years",
    "min_loan_amount", "max_loan_amount",
    "min_tenure_months", "max_tenure_months",
    "interest_rate_min", "interest_rate_max", "processing_fee_pct",
    "risk_appetite", "approval_base_rate",
    "disbursal_success_rate", "disbursal_speed_days",
    "documentation_strictness",
    "preferred_cibil_min", "preferred_cibil_max",
    "preferred_loan_size_min", "preferred_loan_size_max",
    "preferred_loan_size_midpoint", "preferred_loan_size_range",
    "cibil_weight", "dti_weight", "amount_fit_weight",
    "intercept",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def banks() -> pd.DataFrame:
    return generate_banks(seed=SEED, archetype_path=ARCHETYPE_PATH)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_all_required_columns_present(self, banks):
        missing = [c for c in REQUIRED_COLUMNS if c not in banks.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_null_values(self, banks):
        # List columns (object) are allowed; scalar columns must have no NaN
        scalar_cols = [c for c in REQUIRED_COLUMNS
                       if c not in ("states_covered", "loan_types_offered",
                                    "accepted_income_types", "accepted_employer_categories",
                                    "premium_employer_categories", "city_tiers_served")]
        null_counts = banks[scalar_cols].isnull().sum()
        failing = null_counts[null_counts > 0]
        assert failing.empty, f"Null values found:\n{failing}"

    def test_total_bank_count(self, banks):
        assert len(banks) == EXPECTED_TOTAL

    def test_bank_id_uniqueness(self, banks):
        assert banks["bank_id"].nunique() == EXPECTED_TOTAL

    def test_bank_name_uniqueness(self, banks):
        assert banks["name"].nunique() == EXPECTED_TOTAL

    def test_bank_type_values(self, banks):
        assert set(banks["bank_type"].unique()) == set(EXPECTED_COUNTS.keys())

    def test_risk_appetite_values(self, banks):
        assert set(banks["risk_appetite"].unique()).issubset(VALID_RISK_APPETITES)

    def test_documentation_strictness_values(self, banks):
        assert set(banks["documentation_strictness"].unique()).issubset(VALID_DOC_STRICTNESS)


# ---------------------------------------------------------------------------
# Archetype count tests
# ---------------------------------------------------------------------------

class TestArchetypeCounts:
    def test_psb_count(self, banks):
        assert (banks["bank_type"] == "PSB").sum() == EXPECTED_COUNTS["PSB"]

    def test_private_count(self, banks):
        assert (banks["bank_type"] == "private").sum() == EXPECTED_COUNTS["private"]

    def test_nbfc_count(self, banks):
        assert (banks["bank_type"] == "NBFC").sum() == EXPECTED_COUNTS["NBFC"]

    def test_fintech_count(self, banks):
        assert (banks["bank_type"] == "fintech").sum() == EXPECTED_COUNTS["fintech"]

    def test_hfc_count(self, banks):
        assert (banks["bank_type"] == "HFC").sum() == EXPECTED_COUNTS["HFC"]


# ---------------------------------------------------------------------------
# Domain bounds tests
# ---------------------------------------------------------------------------

class TestDomainBounds:
    def test_cibil_score_range(self, banks):
        assert banks["min_cibil_score"].between(550, 760).all()
        assert (banks["max_cibil_score"] >= banks["min_cibil_score"]).all()
        assert (banks["max_cibil_score"] <= 900).all()

    def test_foir_range(self, banks):
        assert (banks["max_foir"] > 0.40).all()
        assert (banks["max_foir"] < 1.0).all()

    def test_dti_exceeds_foir(self, banks):
        # DTI limit must be at least as large as FOIR limit
        assert (banks["max_dti_ratio"] >= banks["max_foir"]).all()

    def test_income_positive(self, banks):
        assert (banks["min_annual_income"] > 0).all()
        assert (banks["max_annual_income"] >= banks["min_annual_income"]).all()

    def test_approval_rate_range(self, banks):
        assert banks["approval_base_rate"].between(0.10, 0.90).all()

    def test_disbursal_rate_range(self, banks):
        assert banks["disbursal_success_rate"].between(0.50, 1.0).all()

    def test_disbursal_speed_positive(self, banks):
        assert (banks["disbursal_speed_days"] >= 1).all()

    def test_loan_amount_ordering(self, banks):
        assert (banks["max_loan_amount"] > banks["min_loan_amount"]).all()

    def test_tenure_ordering(self, banks):
        assert (banks["max_tenure_months"] >= banks["min_tenure_months"]).all()

    def test_interest_rate_ordering(self, banks):
        assert (banks["interest_rate_max"] > banks["interest_rate_min"]).all()

    def test_interest_rate_positive(self, banks):
        assert (banks["interest_rate_min"] > 0).all()

    def test_processing_fee_positive(self, banks):
        assert (banks["processing_fee_pct"] > 0).all()

    def test_age_at_maturity_ge_min_age(self, banks):
        assert (banks["max_age_at_maturity"] > banks["min_age"]).all()

    def test_enquiry_limit_positive(self, banks):
        assert (banks["max_enquiries_6m"] >= 1).all()


# ---------------------------------------------------------------------------
# Business rules tests (critical pitfalls from CLAUDE.md)
# ---------------------------------------------------------------------------

class TestBusinessRules:
    def test_preferred_cibil_above_floor(self, banks):
        """
        CLAUDE.md pitfall: preferred_cibil_band must be ABOVE min_cibil_score.
        This ensures the sweet spot is reachable only by better-than-minimum leads.
        """
        bad = banks[banks["preferred_cibil_min"] <= banks["min_cibil_score"]]
        assert bad.empty, (
            f"{len(bad)} banks have preferred_cibil_min <= min_cibil_score:\n"
            f"{bad[['bank_type','name','min_cibil_score','preferred_cibil_min']].to_string()}"
        )

    def test_preferred_cibil_max_exceeds_min(self, banks):
        assert (banks["preferred_cibil_max"] > banks["preferred_cibil_min"]).all()

    def test_preferred_cibil_cap_at_900(self, banks):
        assert (banks["preferred_cibil_max"] <= 900).all()

    def test_preferred_loan_size_ordering(self, banks):
        assert (banks["preferred_loan_size_max"] > banks["preferred_loan_size_min"]).all()

    def test_preferred_loan_size_midpoint_formula(self, banks):
        expected_mid = (banks["preferred_loan_size_min"] + banks["preferred_loan_size_max"]) / 2
        diff = (banks["preferred_loan_size_midpoint"] - expected_mid).abs()
        assert diff.max() < 1.0, f"preferred_loan_size_midpoint formula mismatch: {diff.max()}"

    def test_preferred_loan_size_range_formula(self, banks):
        expected_rng = banks["preferred_loan_size_max"] - banks["preferred_loan_size_min"]
        diff = (banks["preferred_loan_size_range"] - expected_rng).abs()
        assert diff.max() < 1.0, f"preferred_loan_size_range formula mismatch: {diff.max()}"

    def test_preferred_loan_within_bank_range(self, banks):
        assert (banks["preferred_loan_size_min"] >= banks["min_loan_amount"]).all()
        assert (banks["preferred_loan_size_max"] <= banks["max_loan_amount"]).all()

    def test_psb_conservative_cibil(self, banks):
        """PSBs must have higher CIBIL floors than fintechs (market structure)."""
        psb_min_cibil    = banks[banks["bank_type"] == "PSB"]["min_cibil_score"].mean()
        fintech_min_cibil = banks[banks["bank_type"] == "fintech"]["min_cibil_score"].mean()
        assert psb_min_cibil > fintech_min_cibil, (
            f"PSB avg min_cibil ({psb_min_cibil:.0f}) should exceed "
            f"fintech avg ({fintech_min_cibil:.0f})"
        )

    def test_fintech_higher_approval_rate(self, banks):
        """Fintechs approve more liberally than PSBs on average."""
        psb_rate    = banks[banks["bank_type"] == "PSB"]["approval_base_rate"].mean()
        fintech_rate = banks[banks["bank_type"] == "fintech"]["approval_base_rate"].mean()
        assert fintech_rate > psb_rate, (
            f"Fintech avg approval ({fintech_rate:.3f}) should exceed PSB ({psb_rate:.3f})"
        )

    def test_hfc_only_offers_housing_loans(self, banks):
        """HFCs are housing-only per CLAUDE.md archetype spec."""
        hfc = banks[banks["bank_type"] == "HFC"]
        for _, row in hfc.iterrows():
            assert set(row["loan_types_offered"]).issubset({"home", "lap"}), (
                f"HFC '{row['name']}' offers non-housing loans: {row['loan_types_offered']}"
            )

    def test_fintech_digital_only(self, banks):
        """All fintechs are digital-only per archetype config."""
        fintechs = banks[banks["bank_type"] == "fintech"]
        assert fintechs["digital_only"].all(), "Some fintechs are not digital_only"


# ---------------------------------------------------------------------------
# List field validation tests
# ---------------------------------------------------------------------------

class TestListFields:
    def test_states_covered_non_empty(self, banks):
        for _, row in banks.iterrows():
            assert isinstance(row["states_covered"], list) and len(row["states_covered"]) > 0

    def test_states_covered_valid_codes(self, banks):
        for _, row in banks.iterrows():
            invalid = set(row["states_covered"]) - VALID_STATES
            assert not invalid, f"Bank '{row['name']}' has invalid state codes: {invalid}"

    def test_city_tiers_served_non_empty(self, banks):
        for _, row in banks.iterrows():
            assert isinstance(row["city_tiers_served"], list) and len(row["city_tiers_served"]) > 0

    def test_city_tiers_valid_values(self, banks):
        for _, row in banks.iterrows():
            assert set(row["city_tiers_served"]).issubset({1, 2, 3})

    def test_loan_types_offered_non_empty(self, banks):
        for _, row in banks.iterrows():
            assert isinstance(row["loan_types_offered"], list) and len(row["loan_types_offered"]) > 0

    def test_loan_types_valid(self, banks):
        for _, row in banks.iterrows():
            invalid = set(row["loan_types_offered"]) - VALID_LOAN_TYPES
            assert not invalid, f"Bank '{row['name']}' has invalid loan types: {invalid}"

    def test_accepted_income_types_non_empty(self, banks):
        for _, row in banks.iterrows():
            assert isinstance(row["accepted_income_types"], list) and len(row["accepted_income_types"]) > 0

    def test_accepted_income_types_valid(self, banks):
        for _, row in banks.iterrows():
            invalid = set(row["accepted_income_types"]) - VALID_INCOME_TYPES
            assert not invalid, f"Bank '{row['name']}' has invalid income types: {invalid}"

    def test_accepted_employer_categories_non_empty(self, banks):
        for _, row in banks.iterrows():
            assert isinstance(row["accepted_employer_categories"], list) \
                   and len(row["accepted_employer_categories"]) > 0

    def test_accepted_employer_categories_valid(self, banks):
        for _, row in banks.iterrows():
            invalid = set(row["accepted_employer_categories"]) - VALID_EMPLOYER_CATS
            assert not invalid, (
                f"Bank '{row['name']}' has invalid employer categories: {invalid}"
            )


# ---------------------------------------------------------------------------
# Bank differentiation tests
# ---------------------------------------------------------------------------

class TestDifferentiation:
    def test_unique_intercepts(self, banks):
        """CLAUDE.md pitfall: every bank must have a unique intercept."""
        assert banks["intercept"].nunique() == len(banks), (
            f"Duplicate intercepts found: {len(banks) - banks['intercept'].nunique()} duplicates"
        )

    def test_approval_rate_variance(self, banks):
        """Banks must differ sufficiently in approval behaviour."""
        std = banks["approval_base_rate"].std()
        assert std > 0.05, (
            f"approval_base_rate std too low: {std:.4f} (need > 0.05). "
            "Banks are too similar — ranking will be non-trivial only with differentiation."
        )

    def test_min_cibil_variance(self, banks):
        """Each bank type sets a different CIBIL floor."""
        std = banks["min_cibil_score"].std()
        assert std > 10, f"min_cibil_score std too low: {std:.1f}"

    def test_unique_cibil_weight_dti_weight_pairs(self, banks):
        """Each bank should have a unique (cibil_weight, dti_weight) pair."""
        pairs = banks[["cibil_weight", "dti_weight"]].apply(tuple, axis=1)
        assert pairs.nunique() == len(banks), "Non-unique (cibil_weight, dti_weight) pairs"

    def test_psb_fintechs_conflict(self, banks):
        """
        A lead with CIBIL 640 is fine for fintechs but rejected by PSBs.
        Validates that bank types genuinely conflict — key to non-trivial ranking.
        """
        psb_min = banks[banks["bank_type"] == "PSB"]["min_cibil_score"].min()
        fintech_max_min = banks[banks["bank_type"] == "fintech"]["min_cibil_score"].max()
        assert psb_min > fintech_max_min, (
            f"No CIBIL conflict: PSB floor ({psb_min}) must exceed "
            f"highest fintech floor ({fintech_max_min})"
        )


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_output(self):
        df1 = generate_banks(seed=42, archetype_path=ARCHETYPE_PATH)
        df2 = generate_banks(seed=42, archetype_path=ARCHETYPE_PATH)
        # Compare scalar columns only (list equality works in pandas but is slower)
        scalar_cols = [c for c in df1.columns
                       if c not in ("states_covered", "loan_types_offered",
                                    "accepted_income_types", "accepted_employer_categories",
                                    "premium_employer_categories", "city_tiers_served")]
        pd.testing.assert_frame_equal(df1[scalar_cols], df2[scalar_cols])

    def test_different_seed_different_output(self):
        df1 = generate_banks(seed=42, archetype_path=ARCHETYPE_PATH)
        df2 = generate_banks(seed=99, archetype_path=ARCHETYPE_PATH)
        # Intercepts must differ (they are seed-dependent)
        assert not df1["intercept"].equals(df2["intercept"])


# ---------------------------------------------------------------------------
# validate_banks integration tests
# ---------------------------------------------------------------------------

class TestValidateBanks:
    def test_validate_passes_on_valid_data(self, banks):
        validate_banks(banks)

    def test_validate_fails_on_null(self, banks):
        bad = banks.copy()
        bad.loc[0, "min_cibil_score"] = np.nan
        with pytest.raises(AssertionError, match="Null values found"):
            validate_banks(bad)

    def test_validate_fails_on_sweet_spot_violation(self, banks):
        """preferred_cibil_min must strictly exceed min_cibil_score."""
        bad = banks.copy()
        bad.loc[0, "preferred_cibil_min"] = bad.loc[0, "min_cibil_score"]
        with pytest.raises(AssertionError, match="preferred_cibil_min"):
            validate_banks(bad)

    def test_validate_fails_on_duplicate_intercepts(self, banks):
        bad = banks.copy()
        bad.loc[1, "intercept"] = bad.loc[0, "intercept"]
        with pytest.raises(AssertionError, match="unique intercept"):
            validate_banks(bad)
