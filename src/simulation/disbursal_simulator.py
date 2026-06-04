"""
Disbursal probability simulator for the Lead-to-Bank Ranking system.

Computes P(disbursed | approved, lead, bank) using the bank's base disbursal
success rate adjusted for lead-level risk signals:

  p = base_rate
    + income_type_modifier       (salaried +0.05, business ±0, SE −0.05, freelance −0.10)
    − max(0, foir − 0.5) × 0.2  (high-obligation stress)
    + min(1.0, savings / (loan × 0.1)) × 0.05  (liquidity buffer)
  p = clip(p, 0.50, 0.98)

All operations are vectorised over the flat (n_leads × n_banks) array.

Usage (library):
  from src.simulation.disbursal_simulator import compute_disbursal_probs_batch
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Income-type disbursal adjustments (Section 4.5)
_INCOME_TYPE_MODIFIER: dict[str, float] = {
    "salaried":      +0.05,
    "business":       0.00,
    "self_employed": -0.05,
    "freelance":     -0.10,
}

_DISBURSAL_MIN = 0.50
_DISBURSAL_MAX = 0.98


def compute_disbursal_probs_batch(
    leads: pd.DataFrame,
    banks: pd.DataFrame,
    approved: np.ndarray,
) -> np.ndarray:
    """
    Compute P(disbursed) for all (lead × bank) pairs in row-major flat order.

    Only approved pairs receive a non-zero probability; unapproved pairs
    (including ineligible) return 0.0.

    Parameters
    ----------
    leads    : leads DataFrame (n_leads rows)
    banks    : banks DataFrame (n_banks rows)
    approved : bool array (n_leads * n_banks,); True only for approved pairs

    Returns
    -------
    probs : float array (n_leads * n_banks,); 0.0 for non-approved pairs
    """
    n_leads = len(leads)
    n_banks = len(banks)
    n_total = n_leads * n_banks

    probs = np.zeros(n_total, dtype=float)

    if not approved.any():
        return probs

    # Lead arrays  (n_leads,)
    l_income_type = leads["income_type"].values
    l_foir = leads["foir"].values
    l_savings = leads["savings_balance"].values
    l_loan_amount = leads["loan_amount_requested"].values

    # Bank arrays  (n_banks,)
    b_disbursal_rate = banks["disbursal_success_rate"].values

    # ------------------------------------------------------------------ #
    # Income-type modifier matrix (n_leads, n_banks)
    # All banks apply the same income-type modifier, so it's lead-only.
    # ------------------------------------------------------------------ #
    it_mod = np.array(
        [_INCOME_TYPE_MODIFIER.get(it, 0.0) for it in l_income_type]
    )  # (n_leads,)

    # FOIR liquidity stress: penalise high-obligation leads
    foir_stress = -np.maximum(0.0, l_foir - 0.5) * 0.2  # (n_leads,)

    # Savings buffer: reward leads with a cushion relative to loan size
    # Avoid division by zero for tiny loan amounts
    relative_savings = l_savings / np.maximum(l_loan_amount * 0.1, 1.0)
    savings_bonus = np.minimum(1.0, relative_savings) * 0.05  # (n_leads,)

    # Lead-level adjustment (n_leads,) → broadcast to (n_leads, n_banks)
    lead_adj = it_mod + foir_stress + savings_bonus  # (n_leads,)

    # Base rate (n_banks,) + lead adjustment (n_leads,) → (n_leads, n_banks)
    prob_mat = b_disbursal_rate[None, :] + lead_adj[:, None]
    prob_mat = np.clip(prob_mat, _DISBURSAL_MIN, _DISBURSAL_MAX)

    probs_flat = prob_mat.ravel()
    probs[approved] = probs_flat[approved]

    return probs


def assign_disbursal_failure_reasons(
    leads: pd.DataFrame,
    banks: pd.DataFrame,
    approved: np.ndarray,
    disbursed: np.ndarray,
) -> np.ndarray:
    """
    Assign human-readable disbursal failure reasons for approved-but-not-disbursed pairs.

    Returns
    -------
    reasons : object array (n_leads * n_banks,); None except for disbursal-failed pairs
    """
    n_leads = len(leads)
    n_banks = len(banks)

    reasons = np.full(n_leads * n_banks, None, dtype=object)

    disbursal_failed = approved & ~disbursed
    if not disbursal_failed.any():
        return reasons

    # Bank arrays
    b_doc_strict = banks["documentation_strictness"].values

    # Lead arrays
    l_income_type = leads["income_type"].values
    l_loan_type = leads["loan_type"].values

    # Determine dominant failure cause for each disbursal-failed pair
    for flat_i in np.where(disbursal_failed)[0]:
        l_i = flat_i // n_banks
        b_i = flat_i % n_banks

        doc = b_doc_strict[b_i]
        it = l_income_type[l_i]
        lt = l_loan_type[l_i]

        if doc == "high":
            reasons[flat_i] = "documentation_incomplete"
        elif it in ("self_employed", "freelance"):
            reasons[flat_i] = "income_verification_failed"
        elif lt in ("home", "lap"):
            reasons[flat_i] = "property_valuation_failed"
        else:
            reasons[flat_i] = "processing_failure"

    return reasons
