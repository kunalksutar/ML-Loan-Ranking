"""
Imputation transformers for the preprocessing pipeline (CLAUDE.md §8).

The Section 5 feature pipeline produces zero null feature cells (validated in
`_validate_feature_dataset`), so these imputers act as a defensive guard
against nulls that may appear in future data slices (new banks, partial
applications) rather than a corrective step on the current dataset.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer


def build_numeric_imputer(strategy: str = "median") -> SimpleImputer:
    """Median imputation for continuous numeric features — robust to outliers/skew."""
    return SimpleImputer(strategy=strategy)


def build_categorical_imputer(strategy: str = "most_frequent") -> SimpleImputer:
    """Most-frequent imputation for label-encoded / ordinal categorical features."""
    return SimpleImputer(strategy=strategy)
