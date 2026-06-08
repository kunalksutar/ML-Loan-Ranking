"""
Train/validation/test split strategy and cross-validation utilities (CLAUDE.md §9).

The lead-level 70/15/15 split itself is implemented in
`src.features.interaction_features.split_dataset` (built alongside the
Section 5 feature pipeline and exercised by `TestSplitIntegrity`). This module
adds the remaining Section 9 concerns that sit on top of that split:

  - GroupKFold cross-validation (groups = lead_id) for hyperparameter tuning
  - Class-imbalance handling via `scale_pos_weight`
  - Split-integrity validation (no lead_id spans more than one partition)

Rule: a `lead_id` must never appear in more than one partition — whether that
is train/val/test or a CV train/validation fold — otherwise the model can
memorise lead-level signal, which is equivalent to target leakage.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.features.feature_registry import GROUP_KEY, TARGET

logger = logging.getLogger(__name__)

DEFAULT_N_SPLITS = 5


# ---------------------------------------------------------------------------
# Split-integrity validation (train / val / test)
# ---------------------------------------------------------------------------

def validate_split_integrity(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    group_col: str = GROUP_KEY,
) -> None:
    """Assert that no `lead_id` spans more than one of train/val/test.

    Raises ValueError on any overlap — a lead spanning splits would let the
    model memorise lead-level signal across partitions (CLAUDE.md §9).
    """
    train_ids = set(train[group_col].unique())
    val_ids = set(val[group_col].unique())
    test_ids = set(test[group_col].unique())

    overlaps = {
        "train/val": train_ids & val_ids,
        "train/test": train_ids & test_ids,
        "val/test": val_ids & test_ids,
    }
    bad = {pair: ids for pair, ids in overlaps.items() if ids}
    if bad:
        sizes = {pair: len(ids) for pair, ids in bad.items()}
        raise ValueError(f"Lead-level split leakage detected — overlapping lead_ids: {sizes}")

    logger.info(
        "Split integrity OK | train_leads=%d | val_leads=%d | test_leads=%d | no overlap",
        len(train_ids), len(val_ids), len(test_ids),
    )


# ---------------------------------------------------------------------------
# GroupKFold cross-validation (CLAUDE.md §9, §12)
# ---------------------------------------------------------------------------

def make_group_kfold(n_splits: int = DEFAULT_N_SPLITS) -> GroupKFold:
    """Return a `GroupKFold` splitter — required for all CV during hyperparameter tuning."""
    return GroupKFold(n_splits=n_splits)


def iter_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    group_col: str = GROUP_KEY,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield `(train_idx, val_idx)` positional-index pairs for GroupKFold CV.

    Each yielded pair is verified to contain no overlapping `lead_id` values
    between the train and validation folds before being returned — this is
    the CV-time analogue of `validate_split_integrity`.
    """
    gkf = make_group_kfold(n_splits)
    groups = df[group_col].to_numpy()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups & val_groups
        if overlap:
            raise ValueError(
                f"GroupKFold fold {fold}: {len(overlap)} lead_ids leak across train/val folds"
            )
        logger.info(
            "GroupKFold fold %d | train_rows=%d (%d leads) | val_rows=%d (%d leads)",
            fold, len(train_idx), len(train_groups), len(val_idx), len(val_groups),
        )
        yield train_idx, val_idx


# ---------------------------------------------------------------------------
# Class imbalance — scale_pos_weight (CLAUDE.md §9, §12)
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Compute XGBoost's `scale_pos_weight = n_negative / n_positive`.

    CLAUDE.md §9 mandates `scale_pos_weight` (typically 5-8x given the
    12-18% disbursal rate) over SMOTE, since SMOTE would synthesise
    (lead, bank) pairs that break the pairwise row structure required for
    ranking.
    """
    arr = np.asarray(y)
    n_pos = int((arr == 1).sum())
    n_neg = int((arr == 0).sum())
    if n_pos == 0:
        raise ValueError("compute_scale_pos_weight: no positive samples in `y`")

    weight = n_neg / n_pos
    logger.info(
        "Class imbalance | n_negative=%d | n_positive=%d | positive_rate=%.4f | scale_pos_weight=%.4f",
        n_neg, n_pos, n_pos / (n_pos + n_neg), weight,
    )
    return weight


def class_balance_summary(df: pd.DataFrame, target: str = TARGET) -> dict:
    """Return a small dict summarising class balance — used for logging/reporting."""
    y = df[target]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {
        "n_rows": len(df),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": n_pos / len(df),
        "scale_pos_weight": n_neg / n_pos,
    }
