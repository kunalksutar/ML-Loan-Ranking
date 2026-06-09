"""
Unit tests for src/eligibility/rule_engine.py (CLAUDE.md §16).

Covers:
  - Geography rejection (state not covered)
  - All-pass for a perfect lead against a permissive bank
  - Each of the 12 rules triggered individually
  - Ordering: first-failing rule wins (earlier rules take precedence)
  - Zero eligible on hardest possible bank
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eligibility.rule_engine import apply_eligibility_batch


# ---------------------------------------------------------------------------
# Fixtures — base "perfect" lead and "permissive" bank
# ---------------------------------------------------------------------------

def _make_lead(**overrides) -> pd.DataFrame:
    base = {
        "lead_id": "lead-001",
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
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _make_bank(**overrides) -> pd.DataFrame:
    base = {
        "bank_id": "bank-001",
        "accepted_income_types": ["salaried", "self_employed", "business", "freelance"],
        "states_covered": ["MH", "DL", "KA", "TN", "GJ"],
        "min_cibil_score": 600,
        "max_cibil_score": 900,
        "min_annual_income": 300_000.0,
        "max_annual_income": 50_000_000.0,
        "max_foir": 0.70,
        "max_age_at_maturity": 70,
        "max_enquiries_6m": 5,
        "max_dpd_90_count": 2,
        "max_written_off_loans": 1,
        "loan_types_offered": ["personal", "home", "car", "education", "business", "gold", "lap"],
        "min_loan_amount": 100_000.0,
        "max_loan_amount": 10_000_000.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# All-pass: perfect lead, permissive bank
# ---------------------------------------------------------------------------

class TestAllPass:
    def test_eligible_flag_is_true(self):
        eligible, reasons = apply_eligibility_batch(_make_lead(), _make_bank())
        assert eligible[0] is True or eligible[0] == True  # noqa: E712

    def test_failure_reason_is_none(self):
        _, reasons = apply_eligibility_batch(_make_lead(), _make_bank())
        assert reasons[0] is None


# ---------------------------------------------------------------------------
# Rule 1: income_type_not_accepted
# ---------------------------------------------------------------------------

class TestRule1IncomeType:
    def test_freelance_rejected_by_salaried_only_bank(self):
        lead = _make_lead(income_type="freelance")
        bank = _make_bank(accepted_income_types=["salaried"])
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "income_type_not_accepted"

    def test_matching_income_type_passes(self):
        lead = _make_lead(income_type="self_employed")
        bank = _make_bank(accepted_income_types=["salaried", "self_employed"])
        eligible, _ = apply_eligibility_batch(lead, bank)
        assert eligible[0]


# ---------------------------------------------------------------------------
# Rule 2: state_not_covered
# ---------------------------------------------------------------------------

class TestRule2Geography:
    def test_state_not_covered_rejected(self):
        lead = _make_lead(state="NE")
        bank = _make_bank(states_covered=["MH", "DL"])
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "state_not_covered"

    def test_covered_state_passes(self):
        lead = _make_lead(state="MH")
        bank = _make_bank(states_covered=["MH", "DL"])
        eligible, _ = apply_eligibility_batch(lead, bank)
        assert eligible[0]


# ---------------------------------------------------------------------------
# Rule 3: cibil_below_minimum
# ---------------------------------------------------------------------------

class TestRule3CibilBelow:
    def test_cibil_below_minimum_rejected(self):
        lead = _make_lead(cibil_score=599)
        bank = _make_bank(min_cibil_score=600)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "cibil_below_minimum"

    def test_cibil_exactly_at_minimum_passes(self):
        lead = _make_lead(cibil_score=600)
        bank = _make_bank(min_cibil_score=600)
        eligible, _ = apply_eligibility_batch(lead, bank)
        assert eligible[0]


# ---------------------------------------------------------------------------
# Rule 4: cibil_above_maximum
# ---------------------------------------------------------------------------

class TestRule4CibilAbove:
    def test_cibil_above_maximum_rejected(self):
        lead = _make_lead(cibil_score=901)
        bank = _make_bank(min_cibil_score=600, max_cibil_score=900)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "cibil_above_maximum"


# ---------------------------------------------------------------------------
# Rule 5: income bounds
# ---------------------------------------------------------------------------

class TestRule5Income:
    def test_income_below_minimum_rejected(self):
        lead = _make_lead(annual_income=299_000.0)
        bank = _make_bank(min_annual_income=300_000.0)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "income_below_minimum"

    def test_income_above_maximum_rejected(self):
        lead = _make_lead(annual_income=60_000_000.0)
        bank = _make_bank(max_annual_income=50_000_000.0)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "income_above_maximum"


# ---------------------------------------------------------------------------
# Rule 6: foir_exceeds_maximum
# ---------------------------------------------------------------------------

class TestRule6Foir:
    def test_foir_too_high_rejected(self):
        lead = _make_lead(foir=0.75)
        bank = _make_bank(max_foir=0.70)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "foir_exceeds_maximum"

    def test_foir_at_limit_passes(self):
        lead = _make_lead(foir=0.70)
        bank = _make_bank(max_foir=0.70)
        eligible, _ = apply_eligibility_batch(lead, bank)
        assert eligible[0]


# ---------------------------------------------------------------------------
# Rule 7: age_at_maturity_exceeded
# ---------------------------------------------------------------------------

class TestRule7AgeAtMaturity:
    def test_age_maturity_exceeded_rejected(self):
        lead = _make_lead(age_at_maturity=71)
        bank = _make_bank(max_age_at_maturity=70)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "age_at_maturity_exceeded"


# ---------------------------------------------------------------------------
# Rule 8: enquiry_count_exceeded
# ---------------------------------------------------------------------------

class TestRule8Enquiry:
    def test_too_many_enquiries_rejected(self):
        lead = _make_lead(enquiry_count_6m=6)
        bank = _make_bank(max_enquiries_6m=5)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "enquiry_count_exceeded"


# ---------------------------------------------------------------------------
# Rule 9: dpd_90_exceeded
# ---------------------------------------------------------------------------

class TestRule9Dpd90:
    def test_dpd90_exceeds_bank_max_rejected(self):
        lead = _make_lead(dpd_90_count=3)
        bank = _make_bank(max_dpd_90_count=2)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "dpd_90_exceeded"


# ---------------------------------------------------------------------------
# Rule 10: written_off_loans_exceeded
# ---------------------------------------------------------------------------

class TestRule10WrittenOff:
    def test_written_off_exceeds_max_rejected(self):
        lead = _make_lead(written_off_loans=2)
        bank = _make_bank(max_written_off_loans=1)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "written_off_loans_exceeded"


# ---------------------------------------------------------------------------
# Rule 11: loan_type_not_offered
# ---------------------------------------------------------------------------

class TestRule11LoanType:
    def test_loan_type_not_offered_rejected(self):
        lead = _make_lead(loan_type="education")
        bank = _make_bank(loan_types_offered=["personal", "home"])
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "loan_type_not_offered"


# ---------------------------------------------------------------------------
# Rule 12: loan_amount_out_of_range
# ---------------------------------------------------------------------------

class TestRule12LoanAmount:
    def test_loan_amount_too_low_rejected(self):
        lead = _make_lead(loan_amount_requested=50_000.0)
        bank = _make_bank(min_loan_amount=100_000.0)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "loan_amount_out_of_range"

    def test_loan_amount_too_high_rejected(self):
        lead = _make_lead(loan_amount_requested=15_000_000.0)
        bank = _make_bank(max_loan_amount=10_000_000.0)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "loan_amount_out_of_range"


# ---------------------------------------------------------------------------
# Rule ordering: first-failing rule label wins
# ---------------------------------------------------------------------------

class TestRuleOrdering:
    def test_income_type_rule_takes_priority_over_cibil(self):
        """Rule 1 (income_type) fires before Rule 3 (cibil) for the same pair."""
        lead = _make_lead(income_type="freelance", cibil_score=400)
        bank = _make_bank(
            accepted_income_types=["salaried"],
            min_cibil_score=700,
        )
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "income_type_not_accepted"

    def test_state_rule_takes_priority_over_cibil(self):
        """Rule 2 (state) fires before Rule 3 (cibil)."""
        lead = _make_lead(state="ZZ", cibil_score=400)
        bank = _make_bank(states_covered=["MH"], min_cibil_score=700)
        eligible, reasons = apply_eligibility_batch(lead, bank)
        assert not eligible[0]
        assert reasons[0] == "state_not_covered"


# ---------------------------------------------------------------------------
# Batch: multiple leads × multiple banks
# ---------------------------------------------------------------------------

class TestBatch:
    def test_output_shape_is_n_leads_times_n_banks(self):
        leads = pd.concat([_make_lead(lead_id=f"lead-{i}") for i in range(3)], ignore_index=True)
        banks = pd.concat([_make_bank(bank_id=f"bank-{j}") for j in range(2)], ignore_index=True)
        eligible, reasons = apply_eligibility_batch(leads, banks)
        assert eligible.shape == (6,)
        assert reasons.shape == (6,)

    def test_all_eligible_when_all_pass(self):
        leads = pd.concat([_make_lead(lead_id=f"lead-{i}") for i in range(4)], ignore_index=True)
        banks = _make_bank()
        eligible, _ = apply_eligibility_batch(leads, banks)
        assert eligible.all()

    def test_zero_eligible_for_impossible_bank(self):
        """Bank that rejects everyone: CIBIL min = 950."""
        leads = pd.concat([_make_lead(lead_id=f"lead-{i}") for i in range(5)], ignore_index=True)
        bank = _make_bank(min_cibil_score=950)
        eligible, _ = apply_eligibility_batch(leads, bank)
        assert not eligible.any()
