"""
Bank feature preparation for the Lead-to-Bank Ranking system.

Adds ordinal/label-encoded integer columns for categorical bank fields.
Ordinal maps preserve the natural ordering (conservative < moderate < aggressive).
"""

from __future__ import annotations

import pandas as pd

# Ordinal maps — ordering is semantically meaningful for XGBoost splits
_RISK_APPETITE_MAP: dict[str, int] = {
    "conservative": 0,
    "moderate": 1,
    "aggressive": 2,
}

_DOCUMENTATION_STRICTNESS_MAP: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

# Label encoding for bank_type (no natural ordering)
_BANK_TYPE_MAP: dict[str, int] = {
    "PSB": 0,
    "private": 1,
    "NBFC": 2,
    "fintech": 3,
    "HFC": 4,
    "cooperative": 5,
}


def prepare_bank_features(banks: pd.DataFrame) -> pd.DataFrame:
    """
    Add ordinal/label-encoded columns to a banks DataFrame.

    Adds three new columns:
      bank_type_enc, risk_appetite_enc, documentation_strictness_enc

    Unknown values are encoded as -1. Returns a copy; input is not modified.
    """
    df = banks.copy()
    df["bank_type_enc"] = (
        df["bank_type"].map(_BANK_TYPE_MAP).fillna(-1).astype(int)
    )
    df["risk_appetite_enc"] = (
        df["risk_appetite"].map(_RISK_APPETITE_MAP).fillna(-1).astype(int)
    )
    df["documentation_strictness_enc"] = (
        df["documentation_strictness"]
        .map(_DOCUMENTATION_STRICTNESS_MAP)
        .fillna(-1)
        .astype(int)
    )
    return df
