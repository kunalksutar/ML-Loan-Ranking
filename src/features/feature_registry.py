"""
Central feature registry for the Lead-to-Bank Ranking system.

All feature group lists and key constants are defined here. Import from this
module rather than hardcoding feature names anywhere else in the codebase.
"""

from __future__ import annotations

# ---- Lead features (25) ----
# Numeric columns are used as-is; *_enc columns are label-encoded integers.
LEAD_FEATURES: list[str] = [
    "age",
    "annual_income",
    "cibil_score",
    "foir",
    "dti_ratio",
    "loan_to_income_ratio",
    "enquiry_count_6m",
    "dpd_30_count",
    "dpd_90_count",
    "written_off_loans",
    "settled_loans",
    "existing_loan_count",
    "work_experience_years",
    "current_employer_tenure_yrs",
    "credit_card_spend_monthly",
    "savings_balance",
    "loan_amount_requested",
    "loan_tenure_months",
    "credit_utilization",
    "age_at_maturity",
    "income_type_enc",
    "employer_category_enc",
    "loan_type_enc",
    "city_tier",
    "gender_enc",
]

# ---- Bank features (13) ----
BANK_FEATURES: list[str] = [
    "min_cibil_score",
    "max_foir",
    "min_annual_income",
    "approval_base_rate",
    "disbursal_speed_days",
    "interest_rate_min",
    "interest_rate_max",
    "max_enquiries_6m",
    "max_loan_amount",
    "min_loan_amount",
    "bank_type_enc",
    "risk_appetite_enc",
    "documentation_strictness_enc",
]

# ---- Pair-level interaction features (15) ----
INTERACTION_FEATURES: list[str] = [
    "cibil_gap",
    "foir_headroom",
    "income_headroom",
    "income_headroom_ratio",
    "amount_fit_flag",
    "amount_position",
    "income_type_match",
    "loan_type_match",
    "geography_match",
    "bureau_fatigue_flag",
    "bureau_fatigue_excess",
    "cibil_in_sweet_spot",
    "cibil_vs_sweet_spot_dist",
    "age_maturity_headroom",
    "dpd90_exceeds_bank_max",
]

# ---- Temporal features (4) ----
TEMPORAL_FEATURES: list[str] = [
    "application_sequence_num",
    "days_since_first_application",
    "enquiry_velocity_weekly",
    "is_reapplication",
]

# ---- Combined feature list (57 total) ----
ALL_FEATURES: list[str] = (
    LEAD_FEATURES + BANK_FEATURES + INTERACTION_FEATURES + TEMPORAL_FEATURES
)

# ---- Features that must NEVER appear in the training matrix ----
FORBIDDEN_FEATURES: list[str] = [
    "rejection_reason",
    "approved_amount",
    "approved_rate",
    "approved_tenure_months",
    "disbursed_amount",
    "application_status",
    "bank_responded_at",
    "disbursed_at",
    "disbursal_failure_reason",
]

TARGET: str = "converted"
GROUP_KEY: str = "lead_id"
