"""
Scaling transformers for the preprocessing pipeline (CLAUDE.md §8).

Two strategies are used:
  - log1p -> StandardScaler : right-skewed financial features
                              (annual_income, loan_amount_requested,
                               savings_balance, credit_card_spend_monthly)
  - StandardScaler          : all other continuous numeric features
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def build_standard_scaler() -> StandardScaler:
    """Zero-mean, unit-variance scaling for approximately symmetric continuous features."""
    return StandardScaler()


def build_log1p_scaler() -> Pipeline:
    """log1p transform followed by StandardScaler — for right-skewed log-normal features.

    log1p (rather than log) safely handles zero-valued inputs since all source
    features (income, loan amount, savings, card spend) are non-negative.
    """
    return Pipeline(
        steps=[
            ("log1p", FunctionTransformer(func=np.log1p, inverse_func=np.expm1)),
            ("scale", StandardScaler()),
        ]
    )
