"""
Lead feature preparation for the Lead-to-Bank Ranking system.

Adds label-encoded integer columns for categorical lead fields.
Fixed encoding maps ensure stable codes across train/val/test splits and
across different dataset sizes.
"""

from __future__ import annotations

import pandas as pd

# Fixed label encoding maps — must not change between runs
_INCOME_TYPE_MAP: dict[str, int] = {
    "salaried": 0,
    "self_employed": 1,
    "business": 2,
    "freelance": 3,
}

_EMPLOYER_CATEGORY_MAP: dict[str, int] = {
    "PSU": 0,
    "private_listed": 1,
    "private_unlisted": 2,
    "MNC": 3,
    "govt": 4,
}

_LOAN_TYPE_MAP: dict[str, int] = {
    "personal": 0,
    "home": 1,
    "car": 2,
    "education": 3,
    "business": 4,
    "gold": 5,
    "lap": 6,
}

_GENDER_MAP: dict[str, int] = {
    "M": 0,
    "F": 1,
    "Other": 2,
}


def prepare_lead_features(leads: pd.DataFrame) -> pd.DataFrame:
    """
    Add label-encoded columns to a leads DataFrame.

    Adds four new columns:
      income_type_enc, employer_category_enc, loan_type_enc, gender_enc

    Unknown values are encoded as -1 (should not occur with well-formed data).
    Returns a copy; the input is not modified.
    """
    df = leads.copy()
    df["income_type_enc"] = (
        df["income_type"].map(_INCOME_TYPE_MAP).fillna(-1).astype(int)
    )
    df["employer_category_enc"] = (
        df["employer_category"].map(_EMPLOYER_CATEGORY_MAP).fillna(-1).astype(int)
    )
    df["loan_type_enc"] = (
        df["loan_type"].map(_LOAN_TYPE_MAP).fillna(-1).astype(int)
    )
    df["gender_enc"] = (
        df["gender"].map(_GENDER_MAP).fillna(-1).astype(int)
    )
    return df
