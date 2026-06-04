"""
Approval probability simulator for the Lead-to-Bank Ranking system.

Computes P(approved | lead, bank) via a bank-specific sigmoid scoring function:

  score = intercept
        + cibil_weight × tanh((cibil − min_cibil) / 100)
        + dti_weight   × (max_foir − foir) × 3
        − 2.5          × [enquiry > max_enquiries]
        − 0.25×dpd30 − 0.90×dpd90 − 4.0×written_off − 0.8×settled
        + amount_fit_weight × clip(amount_fit_score, −1, 1)
        + 0.3          × [employer_category ∈ premium_employer_categories]
        + Normal(0, σ_noise)

  P(approved) = sigmoid(score)

Hard override: returns 0.0 immediately for income_type ∉ accepted_income_types
(used during standalone calibration; in the main pipeline the eligibility engine
already filters these pairs before the simulator runs).

Calibration:
  _calibrate_intercepts() adjusts each bank's intercept so that
  mean(P(approved)) over a random lead sample (using the hard override) matches
  bank.approval_base_rate ± 0.02. This is done before the full simulation.

Usage (library):
  from src.simulation.approval_simulator import (
      compute_approval_probs_batch,
      calibrate_intercepts,
  )
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sigmoid noise std — must not exceed 0.3 (see CLAUDE.md §4.4)
_NOISE_STD = 0.25

# Coefficient on the bureau fatigue hard-penalty
_BUREAU_FATIGUE_PENALTY = 2.5


def compute_approval_probs_batch(
    leads: pd.DataFrame,
    banks: pd.DataFrame,
    eligible: np.ndarray,
    rng: np.random.Generator,
    intercepts: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute P(approved) for all (lead × bank) pairs in row-major flat order.

    Parameters
    ----------
    leads      : leads DataFrame (n_leads rows)
    banks      : banks DataFrame (n_banks rows)
    eligible   : bool array of shape (n_leads * n_banks,); ineligible pairs → 0
    rng        : reproducible numpy RNG
    intercepts : optional (n_banks,) array of calibrated intercepts;
                 falls back to banks['intercept'] if None

    Returns
    -------
    probs : float array (n_leads * n_banks,); 0.0 for ineligible pairs
    """
    n_leads = len(leads)
    n_banks = len(banks)
    n_total = n_leads * n_banks

    probs = np.zeros(n_total, dtype=float)

    if not eligible.any():
        return probs

    b_intercepts = intercepts if intercepts is not None else banks["intercept"].values

    # Lead arrays  (n_leads,)
    l_cibil = leads["cibil_score"].values.astype(float)
    l_foir = leads["foir"].values
    l_enquiry = leads["enquiry_count_6m"].values.astype(float)
    l_dpd30 = leads["dpd_30_count"].values.astype(float)
    l_dpd90 = leads["dpd_90_count"].values.astype(float)
    l_written_off = leads["written_off_loans"].values.astype(float)
    l_settled = leads["settled_loans"].values.astype(float)
    l_employer = leads["employer_category"].values
    l_loan_amount = leads["loan_amount_requested"].values

    # Bank arrays  (n_banks,)
    b_min_cibil = banks["min_cibil_score"].values.astype(float)
    b_max_foir = banks["max_foir"].values
    b_max_enq = banks["max_enquiries_6m"].values.astype(float)
    b_cibil_w = banks["cibil_weight"].values
    b_dti_w = banks["dti_weight"].values
    b_amt_fit_w = banks["amount_fit_weight"].values
    b_pref_mid = banks["preferred_loan_size_midpoint"].values
    b_pref_rng = banks["preferred_loan_size_range"].values

    # Premium employer sets (one set per bank)
    premium_sets = [set(v) for v in banks["premium_employer_categories"]]

    # ------------------------------------------------------------------ #
    # Score components as (n_leads, n_banks) matrices
    # ------------------------------------------------------------------ #

    # Intercept
    intercept_mat = b_intercepts[None, :]  # broadcast → (n_leads, n_banks)

    # CIBIL: tanh normalised distance above bank's minimum floor
    cibil_norm = (l_cibil[:, None] - b_min_cibil[None, :]) / 100.0
    cibil_mat = b_cibil_w[None, :] * np.tanh(cibil_norm)

    # FOIR headroom: how far the lead is below the bank's FOIR cap
    foir_headroom = (b_max_foir[None, :] - l_foir[:, None]) * 3.0
    dti_mat = b_dti_w[None, :] * foir_headroom

    # Bureau fatigue penalty (−2.5 if enquiry > max_enquiries)
    bureau_mat = -_BUREAU_FATIGUE_PENALTY * (
        l_enquiry[:, None] > b_max_enq[None, :]
    ).astype(float)

    # Delinquency penalties (lead-level, broadcast over banks)
    dpd_penalty = (
        0.25 * l_dpd30
        + 0.90 * l_dpd90
        + 4.00 * l_written_off
        + 0.80 * l_settled
    )
    dpd_mat = -dpd_penalty[:, None]

    # Loan amount fit: linear distance from bank's preferred size midpoint
    half_range = np.maximum(b_pref_rng[None, :] / 2.0, 1.0)
    dist = np.abs(l_loan_amount[:, None] - b_pref_mid[None, :])
    amount_fit = np.clip(1.0 - 2.0 * dist / half_range, -1.0, 1.0)
    amt_mat = b_amt_fit_w[None, :] * amount_fit

    # Premium employer bonus — requires per-bank list lookup
    premium_mat = np.zeros((n_leads, n_banks), dtype=float)
    for b_i, pset in enumerate(premium_sets):
        if pset:
            premium_mat[:, b_i] = np.array(
                [0.3 if ec in pset else 0.0 for ec in l_employer]
            )

    # Idiosyncratic noise (one draw per pair)
    noise_mat = rng.normal(0.0, _NOISE_STD, (n_leads, n_banks))

    # Total score → sigmoid
    score_mat = (
        intercept_mat + cibil_mat + dti_mat + bureau_mat
        + dpd_mat + amt_mat + premium_mat + noise_mat
    )
    prob_mat = _sigmoid(score_mat)

    # Write eligible pairs only (ineligible remain 0.0)
    probs_flat = prob_mat.ravel()
    probs[eligible] = probs_flat[eligible]

    return probs


def calibrate_intercepts(
    banks: pd.DataFrame,
    leads: pd.DataFrame,
    rng: np.random.Generator,
    n_sample: int = 2_000,
    max_iter: int = 30,
    tol: float = 0.02,
) -> np.ndarray:
    """
    Iteratively adjust each bank's sigmoid intercept so that
    mean(P(approved)) ≈ bank.approval_base_rate ± tol over a lead sample.

    The hard income-type override (P=0 for non-accepted income types) is
    applied during calibration to match how the bank generator calibrates.

    Returns
    -------
    intercepts : (n_banks,) float array of calibrated intercepts
    """
    n_banks = len(banks)
    intercepts = banks["intercept"].values.copy().astype(float)

    # Sample leads for calibration (use the same leads subset for all banks)
    idx = rng.choice(len(leads), size=min(n_sample, len(leads)), replace=False)
    cal_leads = leads.iloc[idx].reset_index(drop=True)

    # Precompute lead arrays for calibration
    l_cibil = cal_leads["cibil_score"].values.astype(float)
    l_foir = cal_leads["foir"].values
    l_enquiry = cal_leads["enquiry_count_6m"].values.astype(float)
    l_dpd30 = cal_leads["dpd_30_count"].values.astype(float)
    l_dpd90 = cal_leads["dpd_90_count"].values.astype(float)
    l_written_off = cal_leads["written_off_loans"].values.astype(float)
    l_settled = cal_leads["settled_loans"].values.astype(float)
    l_employer = cal_leads["employer_category"].values
    l_loan_amount = cal_leads["loan_amount_requested"].values
    l_income_type = cal_leads["income_type"].values

    for b_i, (_, bank) in enumerate(banks.iterrows()):
        target = float(bank["approval_base_rate"])
        accepted_it = set(bank["accepted_income_types"])
        premium_set = set(bank["premium_employer_categories"])

        # Income-type hard override mask
        it_mask = np.array([it in accepted_it for it in l_income_type])

        min_cibil = float(bank["min_cibil_score"])
        max_foir = float(bank["max_foir"])
        max_enq = float(bank["max_enquiries_6m"])
        cibil_w = float(bank["cibil_weight"])
        dti_w = float(bank["dti_weight"])
        amt_fit_w = float(bank["amount_fit_weight"])
        pref_mid = float(bank["preferred_loan_size_midpoint"])
        pref_rng = float(bank["preferred_loan_size_range"])

        for _ in range(max_iter):
            # Deterministic score (no noise during calibration)
            cibil_comp = cibil_w * np.tanh((l_cibil - min_cibil) / 100.0)
            foir_comp = dti_w * (max_foir - l_foir) * 3.0
            bureau_pen = -_BUREAU_FATIGUE_PENALTY * (l_enquiry > max_enq).astype(float)
            dpd_pen = -(0.25 * l_dpd30 + 0.90 * l_dpd90 + 4.0 * l_written_off + 0.8 * l_settled)
            half_rng = max(pref_rng / 2.0, 1.0)
            dist = np.abs(l_loan_amount - pref_mid)
            amt_fit = np.clip(1.0 - 2.0 * dist / half_rng, -1.0, 1.0)
            amt_comp = amt_fit_w * amt_fit
            premium_bonus = np.array([0.3 if ec in premium_set else 0.0 for ec in l_employer])

            score = (intercepts[b_i] + cibil_comp + foir_comp
                     + bureau_pen + dpd_pen + amt_comp + premium_bonus)
            prob = _sigmoid(score)

            # Apply income-type hard override
            prob_with_override = np.where(it_mask, prob, 0.0)
            current_rate = float(prob_with_override.mean())

            if abs(current_rate - target) < tol:
                break

            # Newton step in logit space
            logit_target = _logit(target)
            logit_current = _logit(max(0.001, min(0.999, current_rate)))
            intercepts[b_i] += 0.5 * (logit_target - logit_current)

        logger.debug(
            "Bank %s calibrated: target=%.3f achieved=%.3f intercept=%.4f",
            bank["bank_id"], target, current_rate, intercepts[b_i],
        )

    logger.info("Intercept calibration complete for %d banks", n_banks)
    return intercepts


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1.0 - p)))
