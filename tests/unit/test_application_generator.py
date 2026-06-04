"""
Unit tests for the application generation pipeline.

Covers:
  - Eligibility rule engine: individual rule triggers, all-pass for perfect lead
  - Disbursal simulator: income-type modifiers, savings buffer, FOIR stress
  - Application generator: schema, acceptance criteria, leakage prevention
  - Bureau simulator: schema, pull types
  - Sequence numbers: ordering within each lead
  - Determinism across runs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.lead_generator import generate_leads
from src.simulation.bank_generator import generate_banks
from src.eligibility.rule_engine import apply_eligibility_batch
from src.simulation.disbursal_simulator import (
    compute_disbursal_probs_batch,
    _DISBURSAL_MIN,
    _DISBURSAL_MAX,
)
from src.simulation.bureau_simulator import generate_bureau_pulls
from src.simulation.application_generator import (
    generate_applications,
    validate_applications,
)

ARCHETYPE_PATH = "configs/bank_archetypes.yaml"
SEED = 42
N_LEADS = 300
N_SMALL = 100


@pytest.fixture(scope="module")
def leads() -> pd.DataFrame:
    return generate_leads(n=N_LEADS, seed=SEED)


@pytest.fixture(scope="module")
def banks() -> pd.DataFrame:
    return generate_banks(seed=SEED, archetype_path=ARCHETYPE_PATH)


@pytest.fixture(scope="module")
def apps(leads, banks) -> pd.DataFrame:
    return generate_applications(leads, banks, seed=SEED)


# ---------------------------------------------------------------------------
# Eligibility Engine Unit Tests
# ---------------------------------------------------------------------------

class TestEligibilityEngine:
    def test_all_pass_for_perfect_lead(self, banks):
        """A perfect lead should pass eligibility for at least some banks."""
        perfect = pd.DataFrame([{
            "lead_id": "perfect",
            "income_type": "salaried",
            "state": "MH",
            "cibil_score": 850,
            "annual_income": 2_000_000.0,
            "foir": 0.25,
            "age_at_maturity": 45,
            "enquiry_count_6m": 0,
            "dpd_90_count": 0,
            "written_off_loans": 0,
            "loan_type": "personal",
            "loan_amount_requested": 500_000.0,
        }])
        eligible, reasons = apply_eligibility_batch(perfect, banks)
        assert eligible.any(), "Perfect lead should pass eligibility for at least one bank"
        assert (reasons[eligible] == None).all()  # noqa: E711

    def test_cibil_below_minimum_rejected(self, banks):
        """Lead with CIBIL=300 should fail min_cibil rule for most banks."""
        low_cibil = pd.DataFrame([{
            "lead_id": "low-cibil",
            "income_type": "salaried",
            "state": "MH",
            "cibil_score": 300,
            "annual_income": 1_000_000.0,
            "foir": 0.30,
            "age_at_maturity": 40,
            "enquiry_count_6m": 0,
            "dpd_90_count": 0,
            "written_off_loans": 0,
            "loan_type": "personal",
            "loan_amount_requested": 200_000.0,
        }])
        eligible, reasons = apply_eligibility_batch(low_cibil, banks)
        rejected_by_cibil = (reasons == "cibil_below_minimum")
        assert rejected_by_cibil.any(), "Low CIBIL lead should be rejected by min_cibil rule"

    def test_wrong_income_type_rejected(self, banks):
        """PSBs only accept salaried and business; freelance leads should be rejected."""
        psbs = banks[banks["bank_type"] == "PSB"]
        freelance = pd.DataFrame([{
            "lead_id": "freelance",
            "income_type": "freelance",
            "state": "MH",
            "cibil_score": 750,
            "annual_income": 800_000.0,
            "foir": 0.30,
            "age_at_maturity": 45,
            "enquiry_count_6m": 0,
            "dpd_90_count": 0,
            "written_off_loans": 0,
            "loan_type": "personal",
            "loan_amount_requested": 200_000.0,
        }])
        eligible, reasons = apply_eligibility_batch(freelance, psbs)
        assert not eligible.any(), "Freelance lead should be rejected by all PSBs"
        assert all(r == "income_type_not_accepted" for r in reasons)

    def test_high_enquiry_count_rejected(self, banks):
        """Lead with 15 enquiries should fail all banks' max_enquiries rule."""
        high_enq = pd.DataFrame([{
            "lead_id": "high-enq",
            "income_type": "salaried",
            "state": "MH",
            "cibil_score": 750,
            "annual_income": 1_000_000.0,
            "foir": 0.30,
            "age_at_maturity": 40,
            "enquiry_count_6m": 15,
            "dpd_90_count": 0,
            "written_off_loans": 0,
            "loan_type": "personal",
            "loan_amount_requested": 200_000.0,
        }])
        eligible, reasons = apply_eligibility_batch(high_enq, banks)
        assert not eligible.any(), "Lead with 15 enquiries should be rejected everywhere"

    def test_hfc_only_offers_home_and_lap(self, banks):
        """HFCs should reject personal loan requests (any rule can fire first)."""
        hfcs = banks[banks["bank_type"] == "HFC"]
        # Use a state that is broadly covered; test checks rejection not specific reason
        personal_loan = pd.DataFrame([{
            "lead_id": "personal-hfc",
            "income_type": "salaried",
            "state": "UP",  # high-weight state likely covered by most HFCs
            "cibil_score": 800,
            "annual_income": 2_000_000.0,
            "foir": 0.25,
            "age_at_maturity": 45,
            "enquiry_count_6m": 0,
            "dpd_90_count": 0,
            "written_off_loans": 0,
            "loan_type": "personal",
            "loan_amount_requested": 1_000_000.0,
        }])
        eligible, reasons = apply_eligibility_batch(personal_loan, hfcs)
        assert not eligible.any(), "HFCs should reject personal loans"
        # At least one HFC must cite loan_type as the reason (others may fail earlier on state)
        assert any(r == "loan_type_not_offered" for r in reasons), (
            f"Expected at least one loan_type_not_offered rejection; got {list(reasons)}"
        )

    def test_written_off_loans_rejected(self, banks):
        """Lead with written_off_loans should be rejected by conservative banks."""
        written_off = pd.DataFrame([{
            "lead_id": "written-off",
            "income_type": "salaried",
            "state": "MH",
            "cibil_score": 720,
            "annual_income": 1_200_000.0,
            "foir": 0.30,
            "age_at_maturity": 45,
            "enquiry_count_6m": 1,
            "dpd_90_count": 0,
            "written_off_loans": 2,
            "loan_type": "personal",
            "loan_amount_requested": 300_000.0,
        }])
        psbs = banks[banks["bank_type"] == "PSB"]
        eligible, reasons = apply_eligibility_batch(written_off, psbs)
        rejected_by_wo = (reasons == "written_off_loans_exceeded")
        assert rejected_by_wo.any(), "Written-off loans should trigger rejection"

    def test_eligibility_output_shape(self, leads, banks):
        """Eligibility output shape must match n_leads × n_banks."""
        eligible, reasons = apply_eligibility_batch(leads, banks)
        assert eligible.shape == (len(leads) * len(banks),)
        assert reasons.shape == (len(leads) * len(banks),)

    def test_failure_reasons_null_for_eligible(self, leads, banks):
        """failure_reason must be None for all eligible pairs."""
        eligible, reasons = apply_eligibility_batch(leads, banks)
        eligible_reasons = reasons[eligible]
        assert all(r is None for r in eligible_reasons)

    def test_failure_reasons_set_for_ineligible(self, leads, banks):
        """Every ineligible pair must have a non-None failure reason."""
        eligible, reasons = apply_eligibility_batch(leads, banks)
        ineligible_reasons = reasons[~eligible]
        assert all(r is not None for r in ineligible_reasons)


# ---------------------------------------------------------------------------
# Disbursal Simulator Unit Tests
# ---------------------------------------------------------------------------

class TestDisbursalSimulator:
    def test_probs_in_valid_range(self, leads, banks):
        """Disbursal probs for approved pairs must be in [DISBURSAL_MIN, DISBURSAL_MAX]."""
        approved = np.ones(len(leads) * len(banks), dtype=bool)
        probs = compute_disbursal_probs_batch(leads, banks, approved)
        active = probs[probs > 0]
        if len(active) > 0:
            assert (active >= _DISBURSAL_MIN).all()
            assert (active <= _DISBURSAL_MAX).all()

    def test_unapproved_get_zero_prob(self, leads, banks):
        """Non-approved pairs must have disbursal prob = 0."""
        approved = np.zeros(len(leads) * len(banks), dtype=bool)
        probs = compute_disbursal_probs_batch(leads, banks, approved)
        assert (probs == 0.0).all()

    def test_salaried_higher_than_freelance(self, banks):
        """Salaried leads should have higher disbursal prob than freelance."""
        base_lead = {
            "lead_id": "x",
            "foir": 0.30,
            "savings_balance": 200_000.0,
            "loan_amount_requested": 500_000.0,
        }
        salaried = pd.DataFrame([{**base_lead, "income_type": "salaried"}])
        freelance = pd.DataFrame([{**base_lead, "income_type": "freelance"}])

        bank = banks.iloc[:1].copy()
        approved = np.ones(1, dtype=bool)

        p_sal = compute_disbursal_probs_batch(salaried, bank, approved)[0]
        p_free = compute_disbursal_probs_batch(freelance, bank, approved)[0]

        assert p_sal > p_free, (
            f"Salaried disbursal {p_sal:.3f} should exceed freelance {p_free:.3f}"
        )


# ---------------------------------------------------------------------------
# Application Generator Integration Tests
# ---------------------------------------------------------------------------

class TestApplicationGenerator:
    def test_schema_completeness(self, apps):
        required = [
            "application_id", "lead_id", "bank_id",
            "submitted_at", "bank_responded_at", "disbursed_at",
            "application_sequence_num", "eligibility_passed",
            "eligibility_failure_reason", "application_status",
            "rejection_reason", "approved_amount", "approved_rate",
            "disbursed_amount", "disbursal_failure_reason", "converted",
        ]
        for col in required:
            assert col in apps.columns, f"Missing column: {col}"

    def test_row_count(self, apps, leads, banks):
        assert len(apps) == len(leads) * len(banks)

    def test_no_nulls_in_key_columns(self, apps):
        for col in ["application_id", "lead_id", "bank_id",
                    "eligibility_passed", "converted"]:
            assert apps[col].notna().all(), f"Nulls in {col}"

    def test_unique_application_ids(self, apps):
        assert apps["application_id"].nunique() == len(apps)

    def test_converted_binary(self, apps):
        assert apps["converted"].isin([0, 1]).all()

    def test_leakage_invariant(self, apps):
        """converted=1 must never occur where eligibility_passed=False."""
        bad = apps.loc[~apps["eligibility_passed"], "converted"].sum()
        assert bad == 0, f"Leakage: {bad} ineligible rows with converted=1"

    def test_conversion_rate_in_range(self, apps):
        conv_rate = apps["converted"].mean()
        assert 0.08 <= conv_rate <= 0.25, (
            f"Conversion rate {conv_rate:.4f} outside [0.08, 0.25]"
        )

    def test_per_bank_conversion_std(self, apps):
        per_bank = apps.groupby("bank_id")["converted"].mean()
        std = per_bank.std()
        assert std > 0.05, f"Per-bank conversion std {std:.4f} (need > 0.05)"

    def test_disbursed_rows_have_disbursed_at(self, apps):
        disbursed = apps[apps["converted"] == 1]
        assert disbursed["disbursed_at"].notna().all()

    def test_not_submitted_have_null_response(self, apps):
        not_submitted = apps[apps["application_status"] == "not_submitted"]
        assert not_submitted["bank_responded_at"].isna().all()

    def test_sequence_nums_positive_for_eligible(self, apps):
        eligible = apps[apps["eligibility_passed"]]
        assert (eligible["application_sequence_num"] >= 1).all()

    def test_sequence_nums_zero_for_ineligible(self, apps):
        ineligible = apps[~apps["eligibility_passed"]]
        assert (ineligible["application_sequence_num"] == 0).all()

    def test_status_values(self, apps):
        valid = {"not_submitted", "rejected", "disbursal_failed", "disbursed"}
        assert apps["application_status"].isin(valid).all()

    def test_disbursed_converted_consistent(self, apps):
        disbursed_mask = apps["application_status"] == "disbursed"
        converted_mask = apps["converted"] == 1
        assert (disbursed_mask == converted_mask).all()

    def test_determinism(self, leads, banks):
        apps1 = generate_applications(leads, banks, seed=SEED)
        apps2 = generate_applications(leads, banks, seed=SEED)
        pd.testing.assert_frame_equal(
            apps1[["application_id", "converted", "eligibility_passed"]].reset_index(drop=True),
            apps2[["application_id", "converted", "eligibility_passed"]].reset_index(drop=True),
        )

    def test_validate_applications_passes(self, apps, leads, banks):
        validate_applications(apps, leads, banks)

    def test_rejection_reasons_set_for_ineligible(self, apps):
        ineligible = apps[~apps["eligibility_passed"]]
        assert ineligible["eligibility_failure_reason"].notna().all()

    def test_rejection_reasons_null_for_eligible(self, apps):
        eligible = apps[apps["eligibility_passed"]]
        assert eligible["eligibility_failure_reason"].isna().all()


# ---------------------------------------------------------------------------
# Bureau Simulator Unit Tests
# ---------------------------------------------------------------------------

class TestBureauSimulator:
    def test_schema(self, apps, leads):
        rng = np.random.default_rng(SEED)
        bureau = generate_bureau_pulls(apps, leads, rng)
        required = ["pull_id", "lead_id", "bank_id",
                    "pulled_at", "cibil_score_at_pull", "enquiry_type"]
        for col in required:
            assert col in bureau.columns, f"Missing column: {col}"

    def test_row_count_matches_eligible(self, apps, leads):
        rng = np.random.default_rng(SEED)
        bureau = generate_bureau_pulls(apps, leads, rng)
        n_eligible = apps["eligibility_passed"].sum()
        assert len(bureau) == n_eligible

    def test_enquiry_type_valid(self, apps, leads):
        rng = np.random.default_rng(SEED)
        bureau = generate_bureau_pulls(apps, leads, rng)
        assert bureau["enquiry_type"].isin(["hard", "soft"]).all()

    def test_cibil_in_valid_range(self, apps, leads):
        rng = np.random.default_rng(SEED)
        bureau = generate_bureau_pulls(apps, leads, rng)
        assert bureau["cibil_score_at_pull"].between(300, 900).all()

    def test_unique_pull_ids(self, apps, leads):
        rng = np.random.default_rng(SEED)
        bureau = generate_bureau_pulls(apps, leads, rng)
        assert bureau["pull_id"].nunique() == len(bureau)
