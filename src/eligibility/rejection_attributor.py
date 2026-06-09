"""
Rejection reason attribution for the Lead-to-Bank Ranking system.

Maps raw `eligibility_failure_reason` codes to human-readable explanations
and computes frequency distributions for reporting.
"""

from __future__ import annotations

import pandas as pd

REASON_LABELS: dict[str, str] = {
    "income_type_not_accepted": "Income type not accepted by bank",
    "state_not_covered": "Applicant state not in bank's coverage",
    "cibil_below_minimum": "CIBIL score below bank minimum",
    "cibil_above_maximum": "CIBIL score above bank maximum",
    "income_below_minimum": "Annual income below bank minimum",
    "income_above_maximum": "Annual income above bank maximum",
    "foir_exceeds_maximum": "FOIR exceeds bank's maximum allowed",
    "age_at_maturity_exceeded": "Age at loan maturity exceeds bank limit",
    "enquiry_count_exceeded": "Bureau enquiry count exceeds bank limit",
    "dpd_90_exceeded": "DPD-90 count exceeds bank's maximum",
    "written_off_loans_exceeded": "Written-off loans exceed bank's threshold",
    "loan_type_not_offered": "Loan type not offered by this bank",
    "loan_amount_out_of_range": "Loan amount outside bank's min/max range",
}


def rejection_frequency(applications: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted DataFrame of rejection reason frequencies.

    Parameters
    ----------
    applications : DataFrame containing an `eligibility_failure_reason` column.

    Returns
    -------
    DataFrame with columns [reason, label, count, pct_of_all_pairs].
    """
    ineligible = applications[applications["eligibility_failure_reason"].notna()].copy()
    counts = ineligible["eligibility_failure_reason"].value_counts().reset_index()
    counts.columns = ["reason", "count"]
    counts["label"] = counts["reason"].map(REASON_LABELS).fillna(counts["reason"])
    counts["pct_of_all_pairs"] = 100.0 * counts["count"] / len(applications)
    return counts[["reason", "label", "count", "pct_of_all_pairs"]].reset_index(drop=True)
