"""
Hard-rule eligibility engine for the Lead-to-Bank Ranking system.

Applies 12 ordered rules to every (lead × bank) pair. Rules are evaluated in
order of rejection frequency — cheapest / most common rejections first — so
that expensive list-membership checks are skipped for already-rejected pairs.

The batch function operates on numpy arrays derived from the leads/banks
DataFrames to avoid row-by-row Python overhead on the full cross-join.

Usage (library):
  from src.eligibility.rule_engine import apply_eligibility_batch
  eligible, reasons = apply_eligibility_batch(leads_df, banks_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ordered failure-reason labels (matches Section 10.1 rule order)
REJECTION_REASONS = [
    "income_type_not_accepted",
    "state_not_covered",
    "cibil_below_minimum",
    "cibil_above_maximum",
    "income_below_minimum",
    "income_above_maximum",
    "foir_exceeds_maximum",
    "age_at_maturity_exceeded",
    "enquiry_count_exceeded",
    "dpd_90_exceeded",
    "written_off_loans_exceeded",
    "loan_type_not_offered",
    "loan_amount_out_of_range",
]


def apply_eligibility_batch(
    leads: pd.DataFrame,
    banks: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply all 12 hard eligibility rules to every (lead × bank) pair.

    Output ordering: row-major over the (n_leads × n_banks) matrix.
    Position i maps to lead (i // n_banks) and bank (i % n_banks).

    Parameters
    ----------
    leads  : leads DataFrame (n_leads rows)
    banks  : banks DataFrame (n_banks rows)

    Returns
    -------
    eligible        : bool ndarray of shape (n_leads * n_banks,)
    failure_reasons : object ndarray of same shape; None where eligible
    """
    n_leads = len(leads)
    n_banks = len(banks)
    n_total = n_leads * n_banks

    eligible = np.ones(n_total, dtype=bool)
    failure_reasons = np.full(n_total, None, dtype=object)

    # Precompute sets for list-membership rules (one set per bank)
    accepted_it_sets = [set(v) for v in banks["accepted_income_types"]]
    states_sets = [set(v) for v in banks["states_covered"]]
    lt_sets = [set(v) for v in banks["loan_types_offered"]]

    # Lead value arrays (n_leads,)
    l_income_type = leads["income_type"].values
    l_state = leads["state"].values
    l_cibil = leads["cibil_score"].values.astype(float)
    l_income = leads["annual_income"].values
    l_foir = leads["foir"].values
    l_age_mat = leads["age_at_maturity"].values.astype(float)
    l_enquiry = leads["enquiry_count_6m"].values.astype(float)
    l_dpd90 = leads["dpd_90_count"].values.astype(float)
    l_written_off = leads["written_off_loans"].values.astype(float)
    l_loan_type = leads["loan_type"].values
    l_loan_amount = leads["loan_amount_requested"].values

    # Bank value arrays (n_banks,)
    b_min_cibil = banks["min_cibil_score"].values.astype(float)
    b_max_cibil = banks["max_cibil_score"].values.astype(float)
    b_min_income = banks["min_annual_income"].values
    b_max_income = banks["max_annual_income"].values
    b_max_foir = banks["max_foir"].values
    b_max_age_mat = banks["max_age_at_maturity"].values.astype(float)
    b_max_enq = banks["max_enquiries_6m"].values.astype(float)
    b_max_dpd90 = banks["max_dpd_90_count"].values.astype(float)
    b_max_wo = banks["max_written_off_loans"].values.astype(float)
    b_min_loan = banks["min_loan_amount"].values
    b_max_loan = banks["max_loan_amount"].values

    # ------------------------------------------------------------------ #
    # Rules 1 & 2: list-membership checks (Python loop over banks)
    # Each bank's column is set independently; costs 36 × n_leads iterations.
    # ------------------------------------------------------------------ #

    # Rule 1: income_type not in accepted_income_types
    it_matrix = np.zeros((n_leads, n_banks), dtype=bool)
    for b_i, accepted in enumerate(accepted_it_sets):
        it_matrix[:, b_i] = [it not in accepted for it in l_income_type]
    _apply_rule(eligible, failure_reasons, it_matrix.ravel(), "income_type_not_accepted")

    # Rule 2: state not in states_covered
    state_matrix = np.zeros((n_leads, n_banks), dtype=bool)
    for b_i, covered in enumerate(states_sets):
        state_matrix[:, b_i] = [s not in covered for s in l_state]
    _apply_rule(eligible, failure_reasons, state_matrix.ravel(), "state_not_covered")

    # ------------------------------------------------------------------ #
    # Rules 3–10: scalar comparisons (pure numpy broadcasting)
    # ------------------------------------------------------------------ #

    # Rule 3: cibil < min_cibil_score
    _apply_rule(
        eligible, failure_reasons,
        (l_cibil[:, None] < b_min_cibil[None, :]).ravel(),
        "cibil_below_minimum",
    )

    # Rule 4: cibil > max_cibil_score
    _apply_rule(
        eligible, failure_reasons,
        (l_cibil[:, None] > b_max_cibil[None, :]).ravel(),
        "cibil_above_maximum",
    )

    # Rule 5a: annual_income < min_annual_income
    _apply_rule(
        eligible, failure_reasons,
        (l_income[:, None] < b_min_income[None, :]).ravel(),
        "income_below_minimum",
    )

    # Rule 5b: annual_income > max_annual_income
    _apply_rule(
        eligible, failure_reasons,
        (l_income[:, None] > b_max_income[None, :]).ravel(),
        "income_above_maximum",
    )

    # Rule 6: foir > max_foir
    _apply_rule(
        eligible, failure_reasons,
        (l_foir[:, None] > b_max_foir[None, :]).ravel(),
        "foir_exceeds_maximum",
    )

    # Rule 7: age_at_maturity > max_age_at_maturity
    _apply_rule(
        eligible, failure_reasons,
        (l_age_mat[:, None] > b_max_age_mat[None, :]).ravel(),
        "age_at_maturity_exceeded",
    )

    # Rule 8: enquiry_count_6m > max_enquiries_6m
    _apply_rule(
        eligible, failure_reasons,
        (l_enquiry[:, None] > b_max_enq[None, :]).ravel(),
        "enquiry_count_exceeded",
    )

    # Rule 9: dpd_90_count > max_dpd_90_count
    _apply_rule(
        eligible, failure_reasons,
        (l_dpd90[:, None] > b_max_dpd90[None, :]).ravel(),
        "dpd_90_exceeded",
    )

    # Rule 10: written_off_loans > max_written_off_loans
    _apply_rule(
        eligible, failure_reasons,
        (l_written_off[:, None] > b_max_wo[None, :]).ravel(),
        "written_off_loans_exceeded",
    )

    # ------------------------------------------------------------------ #
    # Rules 11 & 12: list/range checks
    # ------------------------------------------------------------------ #

    # Rule 11: loan_type not in loan_types_offered
    lt_matrix = np.zeros((n_leads, n_banks), dtype=bool)
    for b_i, offered in enumerate(lt_sets):
        lt_matrix[:, b_i] = [lt not in offered for lt in l_loan_type]
    _apply_rule(eligible, failure_reasons, lt_matrix.ravel(), "loan_type_not_offered")

    # Rule 12: loan_amount outside [min_loan_amount, max_loan_amount]
    amt_out = (
        (l_loan_amount[:, None] < b_min_loan[None, :])
        | (l_loan_amount[:, None] > b_max_loan[None, :])
    ).ravel()
    _apply_rule(eligible, failure_reasons, amt_out, "loan_amount_out_of_range")

    n_eligible = int(eligible.sum())
    logger.info(
        "Eligibility complete | total=%d | eligible=%d (%.1f%%) | "
        "top_rejection=%s",
        n_total,
        n_eligible,
        100.0 * n_eligible / n_total,
        _top_rejection(failure_reasons),
    )

    return eligible, failure_reasons


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_rule(
    eligible: np.ndarray,
    failure_reasons: np.ndarray,
    violated: np.ndarray,
    reason: str,
) -> None:
    """Mark newly-failing pairs; skip pairs already failed by an earlier rule."""
    newly_failed = eligible & violated
    eligible[newly_failed] = False
    failure_reasons[newly_failed] = reason


def _top_rejection(failure_reasons: np.ndarray) -> str:
    """Return the most common failure reason for logging."""
    reasons = failure_reasons[failure_reasons != None]  # noqa: E711
    if len(reasons) == 0:
        return "none"
    unique, counts = np.unique(reasons, return_counts=True)
    return str(unique[counts.argmax()])
