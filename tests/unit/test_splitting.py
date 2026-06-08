"""
Unit tests for the Section 9 train/val/test strategy utilities (`src.preprocessing.splitting`).

Covers: split-integrity validation, GroupKFold lead-level grouping, and
class-imbalance (`scale_pos_weight`) computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_registry import GROUP_KEY, TARGET
from src.preprocessing.splitting import (
    DEFAULT_N_SPLITS,
    class_balance_summary,
    compute_scale_pos_weight,
    iter_group_kfold_splits,
    make_group_kfold,
    validate_split_integrity,
)


def _frame_for_leads(lead_ids: list[str], rows_per_lead: int = 3, pos_rate: float = 0.2) -> pd.DataFrame:
    """Build a toy (lead x bank)-like frame: `rows_per_lead` rows per lead_id."""
    rng = np.random.default_rng(0)
    rows = []
    for lid in lead_ids:
        for _ in range(rows_per_lead):
            rows.append({GROUP_KEY: lid, TARGET: int(rng.random() < pos_rate)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# validate_split_integrity
# ---------------------------------------------------------------------------

class TestValidateSplitIntegrity:
    def test_passes_on_disjoint_leads(self):
        train = _frame_for_leads([f"L{i}" for i in range(0, 10)])
        val = _frame_for_leads([f"L{i}" for i in range(10, 13)])
        test = _frame_for_leads([f"L{i}" for i in range(13, 16)])
        validate_split_integrity(train, val, test)  # must not raise

    def test_detects_train_val_overlap(self):
        train = _frame_for_leads(["L0", "L1", "L2"])
        val = _frame_for_leads(["L2", "L3"])  # L2 leaks into val
        test = _frame_for_leads(["L4"])
        with pytest.raises(ValueError, match="train/val"):
            validate_split_integrity(train, val, test)

    def test_detects_train_test_overlap(self):
        train = _frame_for_leads(["L0", "L1"])
        val = _frame_for_leads(["L2"])
        test = _frame_for_leads(["L0"])  # L0 leaks into test
        with pytest.raises(ValueError, match="train/test"):
            validate_split_integrity(train, val, test)

    def test_detects_val_test_overlap(self):
        train = _frame_for_leads(["L0"])
        val = _frame_for_leads(["L1", "L2"])
        test = _frame_for_leads(["L2"])  # L2 leaks into test
        with pytest.raises(ValueError, match="val/test"):
            validate_split_integrity(train, val, test)


# ---------------------------------------------------------------------------
# GroupKFold
# ---------------------------------------------------------------------------

class TestGroupKFold:
    def test_make_group_kfold_default_splits(self):
        gkf = make_group_kfold()
        assert gkf.get_n_splits() == DEFAULT_N_SPLITS

    def test_make_group_kfold_custom_splits(self):
        gkf = make_group_kfold(n_splits=3)
        assert gkf.get_n_splits() == 3

    def test_iter_group_kfold_no_lead_overlap_per_fold(self):
        lead_ids = [f"L{i}" for i in range(40)]
        df = _frame_for_leads(lead_ids, rows_per_lead=4)

        n_folds = 0
        for train_idx, val_idx in iter_group_kfold_splits(df, n_splits=5):
            train_leads = set(df.iloc[train_idx][GROUP_KEY])
            val_leads = set(df.iloc[val_idx][GROUP_KEY])
            assert train_leads.isdisjoint(val_leads)
            assert len(train_idx) + len(val_idx) == len(df)
            n_folds += 1
        assert n_folds == 5

    def test_iter_group_kfold_covers_every_row_exactly_once_as_val(self):
        lead_ids = [f"L{i}" for i in range(30)]
        df = _frame_for_leads(lead_ids, rows_per_lead=2)

        seen_as_val = np.zeros(len(df), dtype=int)
        for _, val_idx in iter_group_kfold_splits(df, n_splits=5):
            seen_as_val[val_idx] += 1

        assert (seen_as_val == 1).all()


# ---------------------------------------------------------------------------
# Class imbalance — scale_pos_weight
# ---------------------------------------------------------------------------

class TestScalePosWeight:
    def test_basic_ratio(self):
        y = np.array([1, 0, 0, 0, 0, 1, 0, 0])  # 2 positive, 6 negative
        assert compute_scale_pos_weight(y) == pytest.approx(3.0)

    def test_realistic_disbursal_rate_yields_expected_range(self):
        # ~14% positive rate -> scale_pos_weight in CLAUDE.md's typical 5-8x band...
        # actually n_neg/n_pos at 14% positive = 0.86/0.14 ≈ 6.14
        rng = np.random.default_rng(1)
        y = (rng.random(100_000) < 0.14).astype(int)
        weight = compute_scale_pos_weight(y)
        assert 4.0 < weight < 9.0

    def test_raises_on_no_positive_samples(self):
        y = np.zeros(10)
        with pytest.raises(ValueError, match="no positive samples"):
            compute_scale_pos_weight(y)

    def test_class_balance_summary_keys_and_consistency(self):
        df = pd.DataFrame({TARGET: [1, 0, 0, 0, 1, 0]})
        summary = class_balance_summary(df)
        assert summary["n_rows"] == 6
        assert summary["n_positive"] == 2
        assert summary["n_negative"] == 4
        assert summary["positive_rate"] == pytest.approx(2 / 6)
        assert summary["scale_pos_weight"] == pytest.approx(4 / 2)
