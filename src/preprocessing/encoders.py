"""
Categorical encoders for the preprocessing pipeline (CLAUDE.md §8).

Encoding strategy:
  - Ordinal features  (risk_appetite_enc, documentation_strictness_enc, city_tier)
    are already integer-coded in their natural rank order by the Section 5
    feature pipeline, so they pass through the ColumnTransformer unchanged.
  - Nominal features  (income_type_enc, employer_category_enc, loan_type_enc,
    gender_enc, bank_type_enc) have no inherent order, so they are one-hot
    encoded — `handle_unknown="ignore"` keeps inference robust to category
    codes unseen during training (e.g. a new bank_type added post-launch).
"""

from __future__ import annotations

from sklearn.preprocessing import OneHotEncoder

# Ordinal-encoded columns are already rank-ordered integers — pass through as-is.
ORDINAL_PASSTHROUGH = "passthrough"


def build_onehot_encoder() -> OneHotEncoder:
    """One-hot encoder for nominal label-encoded categoricals."""
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
