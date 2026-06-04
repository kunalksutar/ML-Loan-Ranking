"""
Bank generator for the Lead-to-Bank Ranking system.

Generates synthetic lender (bank) records from archetype definitions stored in
configs/bank_archetypes.yaml. Each bank gets unique behavioral parameters
sampled within its archetype's ranges, ensuring realistic market heterogeneity.

Key design choices:
  - All sampling is archetype-driven from YAML config (no hardcoded magic numbers)
  - Intercepts are analytically calibrated so each bank's mean approval probability
    matches its target approval_base_rate on a typical lead population
  - preferred_cibil_min is enforced to exceed min_cibil_score (sweet spot above floor)
  - Every bank has a unique (intercept, cibil_weight, dti_weight) triple

Usage (CLI):
  python -m src.simulation.bank_generator --archetypes configs/bank_archetypes.yaml

Usage (library):
  from src.simulation.bank_generator import generate_banks
  df = generate_banks(seed=42)
"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.simulation.distributions import STATE_WEIGHTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typical lead population statistics (from Section 4.1 validation, n=10 000)
# Used for analytical intercept calibration — avoids needing leads at bank
# generation time.
# ---------------------------------------------------------------------------
_TYPICAL_LEAD = {
    "mean_cibil":          663.5,
    "mean_foir":           0.335,
    "mean_dpd_30":         0.54,
    "mean_dpd_90":         0.087,
    "mean_written_off":    0.017,
    "mean_settled":        0.035,
}

# Average delinquency penalty for a typical lead (constants from approval simulator)
_AVG_DPD_PENALTY = (
    _TYPICAL_LEAD["mean_dpd_30"]    * 0.25
    + _TYPICAL_LEAD["mean_dpd_90"]  * 0.90
    + _TYPICAL_LEAD["mean_written_off"] * 4.0
    + _TYPICAL_LEAD["mean_settled"] * 0.8
)  # ≈ 0.309

# Bank name components per type — constructed to sound realistic but clearly synthetic
_BANK_NAME_PARTS: dict[str, dict[str, list[str]]] = {
    "PSB": {
        "first":  ["State", "National", "Regional", "Central", "Allied",
                   "Heritage", "Premier", "Pioneer"],
        "second": ["Bank of India", "National Bank", "Public Bank",
                   "People's Bank", "Banking Corporation"],
    },
    "private": {
        "first":  ["Pinnacle", "Horizon", "Vertex", "Nexus", "Primus",
                   "Apex", "Crest", "Nova", "Orion", "Titan",
                   "Zenith", "Polaris", "Meridian", "Summit"],
        "second": ["Private Bank", "Finance Bank", "Commercial Bank",
                   "Banking Ltd", "Financial Services Bank"],
    },
    "NBFC": {
        "first":  ["Capital", "Metro", "Prime", "Paramount",
                   "Frontier", "Landmark", "Anchor", "Vanguard"],
        "second": ["Finance", "Capital Ltd", "Credit Corp",
                   "Financial Services", "Lending Solutions"],
    },
    "fintech": {
        "first":  ["Rapid", "Swift", "Zest", "Flash", "Nimble", "Agile"],
        "second": ["Credit", "Fintech", "Lend", "Pay", "Cash"],
    },
    "HFC": {
        "first":  ["Griha", "Avas", "Home", "Shelter", "Niwas"],
        "second": ["Housing Finance Ltd", "Home Finance Corp", "Housing Finance"],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_banks(
    seed: int = 42,
    archetype_path: str = "configs/bank_archetypes.yaml",
) -> pd.DataFrame:
    """
    Generate synthetic bank records driven by archetype definitions.

    Parameters
    ----------
    seed          : Random seed for full reproducibility.
    archetype_path: Path to bank_archetypes.yaml.

    Returns
    -------
    pd.DataFrame with one row per bank, all fields populated.
    """
    archetypes = _load_yaml(archetype_path)
    rng = np.random.default_rng(seed)
    all_states = list(STATE_WEIGHTS.keys())

    rows: list[dict] = []
    used_names: set[str] = set()

    for bank_type, arch in archetypes.items():
        count = int(arch["count"])
        names = _unique_bank_names(rng, bank_type, count, used_names)
        used_names.update(names)

        for name in names:
            row = _generate_one_bank(rng, bank_type, arch, name, all_states)
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "Bank generation complete: %d banks across %d types",
        len(df), df["bank_type"].nunique(),
    )
    return df


def validate_banks(df: pd.DataFrame, archetypes: dict | None = None) -> None:
    """
    Run schema and business-rule checks on a generated banks DataFrame.

    Raises AssertionError on any failed check. Logs a summary on success.
    """
    _check(df.isnull().sum().sum() == 0,
           f"Null values found: {df.isnull().sum()[df.isnull().sum() > 0].to_dict()}")

    _check(df["bank_id"].nunique() == len(df), "Duplicate bank_ids detected")
    _check(df["name"].nunique() == len(df), "Duplicate bank names detected")

    # CIBIL floor must be within realistic range
    _check(df["min_cibil_score"].between(550, 760).all(),
           f"min_cibil_score out of [550,760]: {df['min_cibil_score'].describe().to_dict()}")

    # Sweet spot must be above the eligibility floor (critical pitfall from CLAUDE.md)
    bad = df["preferred_cibil_min"] <= df["min_cibil_score"]
    _check(not bad.any(),
           f"{bad.sum()} banks have preferred_cibil_min <= min_cibil_score")

    _check((df["preferred_cibil_max"] > df["preferred_cibil_min"]).all(),
           "preferred_cibil_max must exceed preferred_cibil_min")

    _check((df["approval_base_rate"].between(0.10, 0.90)).all(),
           f"approval_base_rate out of bounds: {df['approval_base_rate'].describe().to_dict()}")

    _check((df["disbursal_success_rate"].between(0.50, 1.0)).all(),
           "disbursal_success_rate out of [0.50, 1.0]")

    _check((df["max_foir"] > 0.40).all() and (df["max_foir"] < 1.0).all(),
           f"max_foir out of (0.40, 1.0): {df['max_foir'].describe().to_dict()}")

    _check((df["min_loan_amount"] > 0).all(), "Non-positive min_loan_amount")
    _check((df["max_loan_amount"] > df["min_loan_amount"]).all(),
           "max_loan_amount must exceed min_loan_amount")

    _check((df["preferred_loan_size_max"] > df["preferred_loan_size_min"]).all(),
           "preferred_loan_size_max must exceed preferred_loan_size_min")

    # All list fields must be non-empty lists
    for col in ("states_covered", "loan_types_offered",
                "accepted_income_types", "accepted_employer_categories"):
        _check(df[col].apply(lambda x: isinstance(x, list) and len(x) > 0).all(),
               f"Column '{col}' has empty or non-list values")

    # Banks must be sufficiently differentiated
    diff_std = df["approval_base_rate"].std()
    _check(diff_std > 0.03,
           f"Banks not differentiated enough: approval_base_rate std={diff_std:.4f} (need >0.03)")

    _check(df["intercept"].nunique() == len(df),
           "All banks must have unique intercept values")

    logger.info(
        "Bank validation passed | n=%d | archetypes=%s | "
        "approval_rate: mean=%.3f std=%.3f | "
        "cibil_min: mean=%.0f std=%.0f | "
        "unique_intercepts=%d",
        len(df),
        dict(df["bank_type"].value_counts().to_dict()),
        df["approval_base_rate"].mean(),
        df["approval_base_rate"].std(),
        df["min_cibil_score"].mean(),
        df["min_cibil_score"].std(),
        df["intercept"].nunique(),
    )


def save_banks(df: pd.DataFrame, output_dir: str) -> Path:
    """Save banks DataFrame to parquet and return the output path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "banks.parquet"
    df.to_parquet(path, index=False)
    logger.info("Banks saved to %s (%d rows)", path, len(df))
    return path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _sample(rng: np.random.Generator, spec) -> float:
    """
    Sample a scalar from a YAML spec:
      - [lo, hi] list → uniform sample in [lo, hi]
      - scalar       → return as-is
    """
    if isinstance(spec, list) and len(spec) == 2:
        lo, hi = float(spec[0]), float(spec[1])
        return float(rng.uniform(lo, hi))
    return float(spec)


def _sample_int(rng: np.random.Generator, spec) -> int:
    """Like _sample but returns an integer (rounds [lo, hi] to integers first)."""
    if isinstance(spec, list) and len(spec) == 2:
        lo, hi = int(spec[0]), int(spec[1])
        return int(rng.integers(lo, hi + 1))
    return int(spec)


def _unique_bank_names(
    rng: np.random.Generator,
    bank_type: str,
    count: int,
    used: set[str],
) -> list[str]:
    """
    Generate `count` unique bank names for `bank_type`.
    Appends a roman-numeral suffix if combinations run out.
    """
    parts = _BANK_NAME_PARTS.get(bank_type, {
        "first":  [bank_type],
        "second": ["Financial Institution"],
    })
    firsts  = parts["first"]
    seconds = parts["second"]

    # Build candidate pool (cross product)
    candidates = [f"{f} {s}" for f in firsts for s in seconds]
    rng.shuffle(candidates)

    names: list[str] = []
    for candidate in candidates:
        if len(names) == count:
            break
        if candidate not in used:
            names.append(candidate)

    # If pool exhausted (shouldn't happen with current sizes), add suffix
    suffix_idx = 1
    while len(names) < count:
        candidate = f"{firsts[0]} {seconds[0]} {suffix_idx}"
        if candidate not in used and candidate not in names:
            names.append(candidate)
        suffix_idx += 1

    return names


def _sample_states(
    rng: np.random.Generator,
    all_states: list[str],
    coverage_spec,
) -> list[str]:
    """Sample a subset of states based on a coverage fraction [lo, hi]."""
    fraction = _sample(rng, coverage_spec)
    n = max(1, round(len(all_states) * fraction))
    chosen = rng.choice(all_states, size=n, replace=False).tolist()
    return sorted(chosen)


def _calibrate_intercept(
    approval_base_rate: float,
    min_cibil_score: float,
    max_foir: float,
    cibil_weight: float,
    dti_weight: float,
) -> float:
    """
    Analytically estimate the sigmoid intercept so that the approval probability
    for a typical lead equals approval_base_rate.

    Derivation:
      E[score] = intercept
               + cibil_weight * tanh((mean_cibil - min_cibil) / 100)
               + dti_weight   * (max_foir - mean_foir) * 3.0
               - avg_dpd_penalty

      sigmoid(E[score]) ≈ approval_base_rate
      ⟹ intercept = logit(approval_base_rate) - E[additional_contributions]

    This is an approximation; the application generator may apply a fine-grained
    iterative calibration pass once both leads and banks are available.
    """
    mean_cibil = _TYPICAL_LEAD["mean_cibil"]
    mean_foir  = _TYPICAL_LEAD["mean_foir"]

    cibil_norm   = (mean_cibil - min_cibil_score) / 100.0
    cibil_contrib = cibil_weight * float(np.tanh(cibil_norm))
    foir_contrib  = dti_weight * (max_foir - mean_foir) * 3.0

    expected_additional = cibil_contrib + foir_contrib - _AVG_DPD_PENALTY

    # logit(p) = log(p / (1 - p))
    p = float(np.clip(approval_base_rate, 0.01, 0.99))
    target_logit = float(np.log(p / (1.0 - p)))

    return target_logit - expected_additional


def _generate_one_bank(
    rng: np.random.Generator,
    bank_type: str,
    arch: dict,
    name: str,
    all_states: list[str],
) -> dict:
    """Sample all fields for a single bank from its archetype definition."""

    # -- Eligibility thresholds --
    min_cibil    = _sample_int(rng, arch["min_cibil_score"])
    max_cibil    = _sample_int(rng, arch.get("max_cibil_score", 900))
    min_income   = _sample(rng, arch["min_annual_income"])
    max_income   = float(arch.get("max_annual_income", 100_000_000))
    max_foir_val = _sample(rng, arch["max_foir"])
    max_dti      = _sample(rng, arch["max_dti_ratio"])
    min_age      = _sample_int(rng, arch.get("min_age", 21))
    max_mat_age  = _sample_int(rng, arch.get("max_age_at_maturity", 65))
    max_enq      = _sample_int(rng, arch["max_enquiries_6m"])
    max_dpd30    = _sample_int(rng, arch.get("max_dpd_30_count", 0))
    max_dpd90    = _sample_int(rng, arch.get("max_dpd_90_count", 0))
    max_wo       = _sample_int(rng, arch.get("max_written_off_loans", 0))
    max_settled  = _sample_int(rng, arch.get("max_settled_loans", 0))
    min_emp_ten  = _sample_int(rng, arch.get("min_employer_tenure_months", 6))
    min_work_exp = _sample(rng, arch.get("min_work_experience_years", 1.0))

    accepted_it  = list(arch["accepted_income_types"])
    accepted_ec  = list(arch["accepted_employer_categories"])
    premium_ec   = list(arch.get("premium_employer_categories", []))

    # -- Loan products --
    loan_types   = list(arch["loan_types_offered"])

    # -- Loan terms --
    min_loan     = float(arch.get("min_loan_amount", 50_000))
    max_loan     = _sample(rng, arch["max_loan_amount"])
    min_tenure   = int(arch.get("min_tenure_months", 12))
    max_tenure   = int(arch.get("max_tenure_months", 360))
    ir_min       = _sample(rng, arch["interest_rate_min"])
    ir_max       = _sample(rng, arch["interest_rate_max"])
    proc_fee     = _sample(rng, arch["processing_fee_pct"])

    # -- Coverage --
    states       = _sample_states(rng, all_states, arch["states_coverage_fraction"])
    city_tiers   = list(arch["city_tiers_served"])
    digital_only = bool(arch.get("digital_only", False))

    # -- Behavioral parameters --
    appr_rate    = _sample(rng, arch["approval_base_rate"])
    disb_rate    = _sample(rng, arch["disbursal_success_rate"])
    disb_speed   = _sample_int(rng, arch["disbursal_speed_days"])
    doc_strict   = str(arch["documentation_strictness"])
    risk_app     = str(arch["risk_appetite"])

    # -- Approval model weights --
    cibil_w      = _sample(rng, arch["cibil_weight"])
    dti_w        = _sample(rng, arch["dti_weight"])
    amt_fit_w    = _sample(rng, arch["amount_fit_weight"])

    # -- Sweet spot CIBIL band --
    pref_offset  = _sample_int(rng, arch["preferred_cibil_offset"])
    pref_width   = _sample_int(rng, arch["preferred_cibil_width"])
    pref_cibil_min = min_cibil + pref_offset        # strictly above floor
    pref_cibil_max = min(900, pref_cibil_min + pref_width)

    # -- Preferred loan size --
    pref_loan_min = _sample(rng, arch["preferred_loan_size_min"])
    pref_loan_max = _sample(rng, arch["preferred_loan_size_max"])
    # Ensure max > min and both within bank's actual range
    pref_loan_min = max(min_loan, pref_loan_min)
    pref_loan_max = max(pref_loan_min * 1.5, min(max_loan, pref_loan_max))
    pref_loan_mid = (pref_loan_min + pref_loan_max) / 2.0
    pref_loan_rng = pref_loan_max - pref_loan_min

    # -- Analytically calibrated sigmoid intercept --
    intercept = _calibrate_intercept(appr_rate, min_cibil, max_foir_val, cibil_w, dti_w)

    # -- Deterministic UUID from RNG --
    uuid_b = rng.integers(0, 256, size=16, dtype=np.uint8)
    uuid_b[6] = (uuid_b[6] & 0x0F) | 0x40
    uuid_b[8] = (uuid_b[8] & 0x3F) | 0x80
    bank_id = str(uuid.UUID(bytes=bytes(uuid_b)))

    return {
        # Identity
        "bank_id":                     bank_id,
        "name":                        name,
        "bank_type":                   bank_type,
        # Coverage
        "states_covered":              states,
        "city_tiers_served":           city_tiers,
        "digital_only":                digital_only,
        # Loan products
        "loan_types_offered":          loan_types,
        # Eligibility rules
        "min_cibil_score":             min_cibil,
        "max_cibil_score":             max_cibil,
        "min_annual_income":           round(min_income, 2),
        "max_annual_income":           max_income,
        "max_foir":                    round(max_foir_val, 4),
        "max_dti_ratio":               round(max_dti, 4),
        "min_age":                     min_age,
        "max_age_at_maturity":         max_mat_age,
        "max_enquiries_6m":            max_enq,
        "max_dpd_30_count":            max_dpd30,
        "max_dpd_90_count":            max_dpd90,
        "max_written_off_loans":       max_wo,
        "max_settled_loans":           max_settled,
        "accepted_income_types":       accepted_it,
        "accepted_employer_categories":accepted_ec,
        "premium_employer_categories": premium_ec,
        "min_employer_tenure_months":  min_emp_ten,
        "min_work_experience_years":   round(min_work_exp, 1),
        # Loan terms
        "min_loan_amount":             round(min_loan, 2),
        "max_loan_amount":             round(max_loan, 2),
        "min_tenure_months":           min_tenure,
        "max_tenure_months":           max_tenure,
        "interest_rate_min":           round(ir_min, 2),
        "interest_rate_max":           round(ir_max, 2),
        "processing_fee_pct":          round(proc_fee, 2),
        # Behavioral parameters
        "risk_appetite":               risk_app,
        "approval_base_rate":          round(appr_rate, 4),
        "disbursal_success_rate":      round(disb_rate, 4),
        "disbursal_speed_days":        disb_speed,
        "documentation_strictness":    doc_strict,
        # Sweet spot (hidden — used only in approval simulator)
        "preferred_cibil_min":         pref_cibil_min,
        "preferred_cibil_max":         pref_cibil_max,
        "preferred_loan_size_min":     round(pref_loan_min, 2),
        "preferred_loan_size_max":     round(pref_loan_max, 2),
        "preferred_loan_size_midpoint":round(pref_loan_mid, 2),
        "preferred_loan_size_range":   round(pref_loan_rng, 2),
        # Approval model weights
        "cibil_weight":                round(cibil_w, 4),
        "dti_weight":                  round(dti_w, 4),
        "amount_fit_weight":           round(amt_fit_w, 4),
        "intercept":                   round(intercept, 6),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate synthetic lending banks.")
    parser.add_argument(
        "--archetypes", default="configs/bank_archetypes.yaml",
        help="Path to bank_archetypes.yaml",
    )
    parser.add_argument(
        "--config", default="configs/data_config.yaml",
        help="Path to data_config.yaml (for seed and output dir)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate existing parquet without regenerating",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed     = cfg["simulation"]["seed"]
    out_dir  = cfg["output"]["raw_dir"]

    banks_path = Path(out_dir) / "banks.parquet"

    if args.validate_only:
        if not banks_path.exists():
            raise FileNotFoundError(f"No banks file at {banks_path}.")
        df = pd.read_parquet(banks_path)
        logger.info("Loaded %d banks from %s for validation", len(df), banks_path)
    else:
        df = generate_banks(seed=seed, archetype_path=args.archetypes)

    archetypes = _load_yaml(args.archetypes)
    validate_banks(df, archetypes)

    if not args.validate_only:
        save_banks(df, out_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
