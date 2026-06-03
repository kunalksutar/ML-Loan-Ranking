"""
Lead generator for the Lead-to-Bank Ranking system.

Generates synthetic loan applicant (lead) records using a causal generation
chain so that feature correlations mirror those observed in real Indian lending
markets — without using any real personal data.

Causal order enforced:
  demographics → income → credit_score → delinquency → obligations →
  enquiries → behavioral → loan_request → derived_ratios

Usage (CLI):
  python -m src.simulation.lead_generator --config configs/data_config.yaml

Usage (library):
  from src.simulation.lead_generator import generate_leads
  df = generate_leads(n=10_000, seed=42)
"""

from __future__ import annotations

import argparse
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.simulation.distributions import (
    CITY_TIER_PROBS,
    CITY_TIERS,
    EMPLOYER_CATEGORY_BY_INCOME_TYPE,
    INCOME_LOGNORMAL_PARAMS,
    INCOME_TYPE_PROBS,
    INCOME_TYPES,
    LOAN_AMOUNT_BOUNDS,
    LOAN_TYPE_MONTHLY_RATES,
    LOAN_TYPE_TENURES,
    LOAN_TYPE_WEIGHTS_BY_INCOME,
    LOAN_TYPES,
    STATE_WEIGHTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_leads(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate n synthetic lead records using a causal chain.

    Parameters
    ----------
    n    : Number of leads to generate.
    seed : Random seed for full reproducibility.

    Returns
    -------
    pd.DataFrame with all lead fields populated and causally consistent.
    """
    logger.info("Starting lead generation: n=%d, seed=%d", n, seed)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Step 1: Demographics
    # ------------------------------------------------------------------ #
    ages = rng.normal(38, 10, n).clip(23, 62).astype(int)

    income_type_idx = rng.choice(len(INCOME_TYPES), n, p=INCOME_TYPE_PROBS)
    income_types = np.array(INCOME_TYPES)[income_type_idx]

    city_tier_idx = rng.choice(len(CITY_TIERS), n, p=CITY_TIER_PROBS)
    city_tiers = np.array(CITY_TIERS)[city_tier_idx]

    state_codes = list(STATE_WEIGHTS.keys())
    state_probs = list(STATE_WEIGHTS.values())
    states = rng.choice(state_codes, n, p=state_probs)

    genders = rng.choice(["M", "F", "Other"], n, p=[0.62, 0.35, 0.03])

    # Simple 6-digit pin codes (no real geographic encoding needed for simulation)
    pin_codes = rng.integers(100_000, 1_000_000, n).astype(str)

    # ------------------------------------------------------------------ #
    # Step 2: Income (depends on income_type and age)
    # ------------------------------------------------------------------ #
    base_incomes = np.zeros(n)
    for itype, (mu, sigma) in INCOME_LOGNORMAL_PARAMS.items():
        mask = income_types == itype
        if mask.any():
            base_incomes[mask] = rng.lognormal(mu, sigma, int(mask.sum()))

    # Career progression: income grows ~2% per year after age 25
    career_multiplier = 1.0 + 0.02 * np.maximum(0, ages - 25)
    annual_incomes = base_incomes * career_multiplier

    # ------------------------------------------------------------------ #
    # Step 3: CIBIL score (depends on income and age)
    # ------------------------------------------------------------------ #
    # Higher income and older age → better credit history → higher CIBIL
    cibil_means = 620.0 + (ages - 25) * 0.8 + (annual_incomes / 100_000) * 4.5
    cibil_scores = rng.normal(cibil_means, 55).clip(300, 900).astype(int)

    # ------------------------------------------------------------------ #
    # Step 4: Delinquency (inversely driven by CIBIL)
    # ------------------------------------------------------------------ #
    # dpd_prob = 0 for CIBIL=750, increases linearly as CIBIL drops
    dpd_probs = np.maximum(0.0, (750 - cibil_scores) / 500.0)

    dpd_30_counts       = rng.poisson(dpd_probs * 3.0).astype(int)
    dpd_90_counts       = rng.poisson(dpd_probs * 0.5).astype(int)
    written_off_loans   = rng.poisson(dpd_probs * 0.1).astype(int)
    settled_loans       = rng.poisson(dpd_probs * 0.2).astype(int)

    # ------------------------------------------------------------------ #
    # Step 5: Monthly obligations (derived from income via target FOIR)
    # ------------------------------------------------------------------ #
    # Beta(2, 4): mode ~0.25, mean ~0.33; most leads between 15%–60% FOIR
    target_foir = rng.beta(2, 4, n).clip(0.10, 0.90)
    monthly_incomes = annual_incomes / 12.0
    monthly_obligations = monthly_incomes * target_foir

    # ------------------------------------------------------------------ #
    # Step 6: Bureau enquiry count (correlated with financial distress)
    # ------------------------------------------------------------------ #
    # Distressed borrowers shop more aggressively; clipped to realistic max
    enquiry_base = dpd_probs * 4.0
    enquiry_counts = rng.poisson(enquiry_base).clip(0, 10).astype(int)

    # ------------------------------------------------------------------ #
    # Step 7: Behavioral and employment features (from income / age)
    # ------------------------------------------------------------------ #
    cc_spends = rng.lognormal(np.log(annual_incomes * 0.12), 0.4)
    savings_balances = rng.lognormal(np.log(annual_incomes * 0.15), 0.6)

    # Only ~40% of leads hold fixed deposits
    fd_log_means = np.log(np.maximum(annual_incomes * 0.05, 1.0))
    fd_raw = rng.lognormal(fd_log_means, 0.8)
    fixed_deposits = np.where(rng.random(n) < 0.40, fd_raw, 0.0)

    # Work experience: uniformly between 0.5 years and (age - 22) years
    work_exp_max = np.maximum(1.0, (ages - 22).astype(float))
    work_experience = np.round(rng.uniform(0.5, work_exp_max), 1)

    # Current employer tenure: between 0.5 and min(work_experience, 10) years
    tenure_max = np.maximum(np.minimum(work_experience, 10.0), 0.6)
    employer_tenure = np.round(rng.uniform(0.5, tenure_max), 1)

    # Existing loan count: rises with age and income
    loan_count_lam = 0.5 + (ages - 23) * 0.05 + (annual_incomes / 1_000_000) * 0.3
    existing_loan_counts = rng.poisson(loan_count_lam).clip(0, 8).astype(int)

    # Employer category (matches realistic distribution per income_type)
    employer_categories = np.empty(n, dtype=object)
    for itype, (cats, probs) in EMPLOYER_CATEGORY_BY_INCOME_TYPE.items():
        mask = income_types == itype
        if mask.any():
            employer_categories[mask] = rng.choice(cats, p=probs, size=int(mask.sum()))

    # ------------------------------------------------------------------ #
    # Step 8: Loan request (consistent with income type and income level)
    # ------------------------------------------------------------------ #
    loan_types = np.empty(n, dtype=object)
    for itype, weights in LOAN_TYPE_WEIGHTS_BY_INCOME.items():
        mask = income_types == itype
        if mask.any():
            loan_types[mask] = rng.choice(LOAN_TYPES, p=weights, size=int(mask.sum()))

    loan_amounts = _generate_loan_amounts(rng, loan_types, annual_incomes)

    loan_tenures = np.zeros(n, dtype=int)
    for lt, tenure_opts in LOAN_TYPE_TENURES.items():
        mask = loan_types == lt
        if mask.any():
            loan_tenures[mask] = rng.choice(tenure_opts, size=int(mask.sum()))

    # Enforce age_at_maturity < 80 (schema invariant)
    max_tenure_by_age = (79 - ages) * 12
    loan_tenures = np.minimum(loan_tenures, max_tenure_by_age)
    loan_tenures = np.maximum(loan_tenures, 12)  # minimum 12 months

    # ------------------------------------------------------------------ #
    # Step 9: Derived ratios
    # ------------------------------------------------------------------ #
    foir = monthly_obligations / monthly_incomes  # equals target_foir by construction

    # DTI includes estimated EMI for the new loan being requested
    monthly_rates = np.array(
        [LOAN_TYPE_MONTHLY_RATES[lt] for lt in loan_types], dtype=float
    )
    new_emis = _compute_emi(loan_amounts, monthly_rates, loan_tenures)
    dti_ratio = (monthly_obligations + new_emis) / monthly_incomes

    loan_to_income_ratio = loan_amounts / annual_incomes

    # age_at_maturity: age + full years of loan tenure
    age_at_maturity = ages + (loan_tenures // 12)

    # Credit utilization: CC spend relative to estimated revolving credit limit
    # Limit estimated as 6× monthly income (common Indian bank heuristic)
    cc_limit_estimates = monthly_incomes * 6.0
    credit_utilization = (cc_spends / cc_limit_estimates).clip(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Identifiers and timestamps
    # ------------------------------------------------------------------ #
    # Generate UUIDs deterministically from the numpy RNG (uuid4 uses OS entropy)
    uuid_bytes = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    uuid_bytes[:, 6] = (uuid_bytes[:, 6] & 0x0F) | 0x40  # version 4
    uuid_bytes[:, 8] = (uuid_bytes[:, 8] & 0x3F) | 0x80  # variant bits
    lead_ids = [str(uuid.UUID(bytes=bytes(row))) for row in uuid_bytes]

    base_date = datetime(2024, 1, 1)
    created_ats = [
        base_date + timedelta(days=int(d))
        for d in rng.integers(0, 365, n)
    ]

    # ------------------------------------------------------------------ #
    # Assemble DataFrame
    # ------------------------------------------------------------------ #
    df = pd.DataFrame({
        # Identity
        "lead_id":                     lead_ids,
        "created_at":                  created_ats,
        # Demographics
        "age":                         ages,
        "gender":                      genders,
        "city_tier":                   city_tiers,
        "state":                       states,
        "pin_code":                    pin_codes,
        # Employment
        "income_type":                 income_types,
        "employer_category":           employer_categories,
        "annual_income":               annual_incomes.round(2),
        "work_experience_years":       work_experience,
        "current_employer_tenure_yrs": employer_tenure,
        # Credit profile
        "cibil_score":                 cibil_scores,
        "dpd_30_count":                dpd_30_counts,
        "dpd_90_count":                dpd_90_counts,
        "enquiry_count_6m":            enquiry_counts,
        "settled_loans":               settled_loans,
        "written_off_loans":           written_off_loans,
        "existing_loan_count":         existing_loan_counts,
        # Financial behavior
        "monthly_obligations":         monthly_obligations.round(2),
        "credit_card_spend_monthly":   cc_spends.round(2),
        "savings_balance":             savings_balances.round(2),
        "fixed_deposits":              fixed_deposits.round(2),
        # Loan request
        "loan_type":                   loan_types,
        "loan_amount_requested":       loan_amounts,
        "loan_tenure_months":          loan_tenures,
        # Derived ratios
        "foir":                        foir.round(4),
        "dti_ratio":                   dti_ratio.round(4),
        "loan_to_income_ratio":        loan_to_income_ratio.round(4),
        "credit_utilization":          credit_utilization.round(4),
        "age_at_maturity":             age_at_maturity,
    })

    logger.info("Lead generation complete: %d leads generated", len(df))
    return df


def validate_leads(df: pd.DataFrame, config: dict | None = None) -> None:
    """
    Run schema and causal correlation checks on a generated leads DataFrame.

    Raises AssertionError on any failed check. Logs a summary on success.
    """
    cfg = (config or {}).get("validation", {})

    _check(df.isnull().sum().sum() == 0,
           f"Null values found: {df.isnull().sum()[df.isnull().sum() > 0].to_dict()}")

    _check(df["cibil_score"].between(300, 900).all(),
           f"CIBIL score outside [300, 900]: min={df['cibil_score'].min()}, max={df['cibil_score'].max()}")

    foir_min = cfg.get("foir_min", 0.05)
    foir_max = cfg.get("foir_max", 0.95)
    _check((df["foir"] > foir_min).all() and (df["foir"] < foir_max).all(),
           f"FOIR outside ({foir_min}, {foir_max}): min={df['foir'].min():.4f}, max={df['foir'].max():.4f}")

    age_min = cfg.get("age_min", 23)
    age_max = cfg.get("age_max", 62)
    _check(df["age"].between(age_min, age_max).all(),
           f"Age outside [{age_min}, {age_max}]")

    max_maturity = cfg.get("max_age_at_maturity", 79)
    _check(df["age_at_maturity"].max() <= max_maturity,
           f"age_at_maturity exceeds {max_maturity}: max={df['age_at_maturity'].max()}")

    _check((df["annual_income"] > 0).all(), "Non-positive annual_income found")
    _check((df["loan_amount_requested"] > 0).all(), "Non-positive loan_amount_requested found")
    _check((df["enquiry_count_6m"] >= 0).all(), "Negative enquiry_count_6m found")
    _check((df["loan_tenure_months"] >= 12).all(), "Loan tenure below 12 months")

    # Causal correlation checks
    cibil_income_corr = df["cibil_score"].corr(df["annual_income"])
    min_corr = cfg.get("min_cibil_income_corr", 0.30)
    _check(cibil_income_corr > min_corr,
           f"CIBIL-income correlation too low: {cibil_income_corr:.3f} (expected > {min_corr})")

    cibil_dpd_corr = df["dpd_30_count"].corr(df["cibil_score"])
    max_corr = cfg.get("max_cibil_dpd30_corr", -0.25)
    _check(cibil_dpd_corr < max_corr,
           f"CIBIL-DPD30 correlation wrong: {cibil_dpd_corr:.3f} (expected < {max_corr})")

    logger.info(
        "Validation passed | n=%d | mean_cibil=%.0f | mean_income=%.0f | "
        "mean_foir=%.3f | cibil_income_corr=%.3f | cibil_dpd_corr=%.3f",
        len(df),
        df["cibil_score"].mean(),
        df["annual_income"].mean(),
        df["foir"].mean(),
        cibil_income_corr,
        cibil_dpd_corr,
    )


def save_leads(df: pd.DataFrame, output_dir: str) -> Path:
    """Save leads DataFrame to parquet and return the output path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "leads.parquet"
    df.to_parquet(path, index=False)
    logger.info("Leads saved to %s (%d rows)", path, len(df))
    return path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _check(condition: bool, message: str) -> None:
    """Raise AssertionError with message if condition is False."""
    if not condition:
        raise AssertionError(message)


def _generate_loan_amounts(
    rng: np.random.Generator,
    loan_types: np.ndarray,
    annual_incomes: np.ndarray,
) -> np.ndarray:
    """
    Sample loan amounts using log-normal distributions clipped to income-
    scaled bounds. Amounts are rounded to the nearest 1,000 INR.
    """
    amounts = np.zeros(len(loan_types))

    for lt, (min_a, max_abs, income_mult) in LOAN_AMOUNT_BOUNDS.items():
        mask = loan_types == lt
        if not mask.any():
            continue

        income = annual_incomes[mask]
        effective_max = np.minimum(max_abs, income * income_mult)
        # Ensure effective_max is always strictly greater than min_a
        effective_max = np.maximum(effective_max, min_a * 1.1)

        log_min = np.log(min_a)
        log_max = np.log(effective_max)
        log_mid = (log_min + log_max) / 2.0
        # std devs chosen so ~95% of samples fall within [min_a, effective_max]
        log_std = (log_max - log_min) / 4.0

        sampled = np.exp(rng.normal(log_mid, log_std, int(mask.sum())))
        amounts[mask] = np.clip(sampled, min_a, effective_max)

    return np.round(amounts, -3)  # round to nearest 1,000 INR


def _compute_emi(
    principals: np.ndarray,
    monthly_rates: np.ndarray,
    tenures: np.ndarray,
) -> np.ndarray:
    """
    Vectorised standard EMI formula:
        EMI = P × r × (1+r)^n / ((1+r)^n − 1)

    Falls back to simple division when rate is effectively zero.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = (1.0 + monthly_rates) ** tenures
        emis = np.where(
            monthly_rates > 1e-10,
            principals * monthly_rates * factor / np.maximum(factor - 1.0, 1e-10),
            principals / np.maximum(tenures, 1),
        )
    return emis


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate synthetic loan leads.")
    parser.add_argument("--config", default="configs/data_config.yaml",
                        help="Path to data_config.yaml")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate existing parquet without regenerating")
    args = parser.parse_args()

    config = _load_config(args.config)
    sim_cfg = config["simulation"]
    out_cfg = config["output"]

    leads_path = Path(out_cfg["raw_dir"]) / "leads.parquet"

    if args.validate_only:
        if not leads_path.exists():
            raise FileNotFoundError(f"No leads file at {leads_path}. Run without --validate-only first.")
        df = pd.read_parquet(leads_path)
        logger.info("Loaded %d leads from %s for validation", len(df), leads_path)
    else:
        df = generate_leads(n=sim_cfg["n_leads"], seed=sim_cfg["seed"])

    validate_leads(df, config)

    if not args.validate_only:
        save_leads(df, out_cfg["raw_dir"])

    logger.info("Done.")


if __name__ == "__main__":
    main()
