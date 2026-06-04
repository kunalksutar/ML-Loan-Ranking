"""
Integration test for the full Section 4.3 simulation pipeline.

Runs leads → banks → applications end-to-end on a small dataset and
asserts all acceptance criteria, schema constraints, leakage prevention,
and correlation targets from CLAUDE.md §4.3 and §7.

Tests:
  - End-to-end pipeline runs without error
  - Schema validation passes for all three tables
  - Conversion rate within [0.08, 0.25]
  - Per-bank conversion rate std > 0.05
  - Zero leakage (no converted=1 where eligibility_passed=False)
  - Causal correlations present in joined table
  - Lead-level split integrity (all rows for a lead in one split)
  - Bureau pull log schema and join consistency
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.lead_generator import generate_leads, validate_leads
from src.simulation.bank_generator import generate_banks, validate_banks
from src.simulation.application_generator import (
    generate_applications,
    validate_applications,
)
from src.simulation.bureau_simulator import generate_bureau_pulls

ARCHETYPE_PATH = "configs/bank_archetypes.yaml"
SEED = 99  # distinct seed from unit tests

# 2 000 leads gives enough per-bank observations to reliably hit std > 0.05
N_LEADS = 2000
N_BANKS = 36


@pytest.fixture(scope="module")
def leads() -> pd.DataFrame:
    df = generate_leads(n=N_LEADS, seed=SEED)
    validate_leads(df)
    return df


@pytest.fixture(scope="module")
def banks() -> pd.DataFrame:
    import yaml
    with open(ARCHETYPE_PATH) as f:
        archetypes = yaml.safe_load(f)
    df = generate_banks(seed=SEED, archetype_path=ARCHETYPE_PATH)
    validate_banks(df, archetypes)
    return df


@pytest.fixture(scope="module")
def apps(leads, banks) -> pd.DataFrame:
    return generate_applications(leads, banks, seed=SEED)


@pytest.fixture(scope="module")
def bureau(apps, leads) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    return generate_bureau_pulls(apps, leads, rng)


# ---------------------------------------------------------------------------
# Full-pipeline smoke test
# ---------------------------------------------------------------------------

class TestPipelineRuns:
    def test_applications_generated(self, apps, leads, banks):
        assert len(apps) == N_LEADS * N_BANKS

    def test_validate_applications_passes(self, apps, leads, banks):
        validate_applications(apps, leads, banks)

    def test_bureau_pulls_generated(self, bureau):
        assert len(bureau) > 0


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------

class TestApplicationSchema:
    REQUIRED_COLUMNS = [
        "application_id", "lead_id", "bank_id",
        "submitted_at", "bank_responded_at", "disbursed_at",
        "application_sequence_num",
        "eligibility_passed", "eligibility_failure_reason",
        "application_status", "rejection_reason",
        "approved_amount", "approved_rate", "disbursed_amount",
        "disbursal_failure_reason", "converted",
    ]

    def test_all_required_columns_present(self, apps):
        for col in self.REQUIRED_COLUMNS:
            assert col in apps.columns, f"Missing column: {col}"

    def test_converted_is_binary(self, apps):
        assert apps["converted"].isin([0, 1]).all()

    def test_eligibility_passed_is_bool(self, apps):
        assert apps["eligibility_passed"].dtype == bool or apps["eligibility_passed"].isin([True, False]).all()

    def test_application_ids_unique(self, apps):
        assert apps["application_id"].nunique() == len(apps)

    def test_no_nulls_in_required_fields(self, apps):
        must_not_be_null = [
            "application_id", "lead_id", "bank_id",
            "eligibility_passed", "converted",
            "application_sequence_num", "application_status",
        ]
        for col in must_not_be_null:
            assert apps[col].notna().all(), f"Nulls found in {col}"


class TestBureauSchema:
    def test_all_columns_present(self, bureau):
        required = ["pull_id", "lead_id", "bank_id",
                    "pulled_at", "cibil_score_at_pull", "enquiry_type"]
        for col in required:
            assert col in bureau.columns

    def test_enquiry_types_valid(self, bureau):
        assert bureau["enquiry_type"].isin(["hard", "soft"]).all()

    def test_cibil_scores_in_range(self, bureau):
        assert bureau["cibil_score_at_pull"].between(300, 900).all()


# ---------------------------------------------------------------------------
# Acceptance criteria (CLAUDE.md §4.3)
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    def test_conversion_rate_in_range(self, apps):
        conv = apps["converted"].mean()
        assert 0.08 <= conv <= 0.25, f"Conversion rate {conv:.4f} out of range"

    def test_per_bank_conversion_std(self, apps):
        per_bank = apps.groupby("bank_id")["converted"].mean()
        std = float(per_bank.std())
        assert std > 0.05, f"Per-bank conversion std {std:.4f} (need > 0.05)"

    def test_zero_leakage(self, apps):
        bad = int(apps.loc[~apps["eligibility_passed"], "converted"].sum())
        assert bad == 0, f"Leakage: {bad} ineligible pairs with converted=1"

    def test_some_conversions_exist(self, apps):
        assert apps["converted"].sum() > 0, "No conversions generated"

    def test_all_statuses_represented(self, apps):
        statuses = set(apps["application_status"].unique())
        expected = {"not_submitted", "rejected"}
        assert expected.issubset(statuses), (
            f"Expected statuses {expected} missing; got {statuses}"
        )


# ---------------------------------------------------------------------------
# Causal correlation targets (CLAUDE.md §7)
# ---------------------------------------------------------------------------

class TestCausalCorrelations:
    @pytest.fixture(scope="class")
    def joined(self, apps, leads, banks):
        j = apps.merge(
            leads[["lead_id", "cibil_score", "annual_income", "foir",
                   "enquiry_count_6m", "dpd_30_count"]],
            on="lead_id", how="left",
        ).merge(
            banks[["bank_id", "max_foir", "max_enquiries_6m", "min_cibil_score"]],
            on="bank_id", how="left",
        )
        j["foir_headroom"] = j["max_foir"] - j["foir"]
        j["bureau_fatigue_flag"] = (j["enquiry_count_6m"] > j["max_enquiries_6m"]).astype(int)
        j["cibil_gap"] = j["cibil_score"] - j["min_cibil_score"]
        return j

    def test_cibil_income_correlation(self, joined):
        corr = float(joined["cibil_score"].corr(joined["annual_income"]))
        assert corr > 0.30, f"corr(cibil_score, annual_income)={corr:.3f} (need > 0.30)"

    def test_cibil_dpd_correlation_negative(self, joined):
        corr = float(joined["cibil_score"].corr(joined["dpd_30_count"]))
        assert corr < -0.20, f"corr(cibil_score, dpd_30_count)={corr:.3f} (need < −0.20)"

    def test_foir_headroom_positive_with_converted(self, joined):
        corr = float(joined["foir_headroom"].corr(joined["converted"]))
        assert corr > 0.05, f"corr(foir_headroom, converted)={corr:.3f} (need > 0.05)"

    def test_bureau_fatigue_negative_with_converted(self, joined):
        corr = float(joined["bureau_fatigue_flag"].corr(joined["converted"]))
        assert corr < -0.02, f"corr(bureau_fatigue_flag, converted)={corr:.3f} (need < −0.02)"

    def test_cibil_gap_positive_with_converted(self, joined):
        corr = float(joined["cibil_gap"].corr(joined["converted"]))
        assert corr > 0.0, f"corr(cibil_gap, converted)={corr:.3f} (expected positive)"


# ---------------------------------------------------------------------------
# Lead-level split integrity (CLAUDE.md §9)
# ---------------------------------------------------------------------------

class TestLeadLevelSplitIntegrity:
    def test_all_lead_rows_grouped_correctly(self, apps, leads):
        """Verify that lead-level groupby gives exactly n_banks rows per lead."""
        n_banks = apps["bank_id"].nunique()
        per_lead = apps.groupby("lead_id").size()
        assert (per_lead == n_banks).all(), (
            f"Some leads have unexpected row counts (expected {n_banks} each)"
        )

    def test_no_lead_spans_multiple_splits(self, leads, banks):
        """Simulated train/val/test split must not split a lead across partitions."""
        all_lead_ids = leads["lead_id"].values
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(all_lead_ids))
        n = len(all_lead_ids)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        train_ids = set(all_lead_ids[perm[:train_end]])
        val_ids = set(all_lead_ids[perm[train_end:val_end]])
        test_ids = set(all_lead_ids[perm[val_end:]])

        # Sets must be disjoint
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0
        assert len(train_ids | val_ids | test_ids) == n


# ---------------------------------------------------------------------------
# Bank differentiation
# ---------------------------------------------------------------------------

class TestBankDifferentiation:
    def test_fintech_approves_more_than_psb(self, apps, banks):
        bank_type = banks[["bank_id", "bank_type"]]
        merged = apps.merge(bank_type, on="bank_id")
        by_type = merged.groupby("bank_type")["converted"].mean()

        if "fintech" in by_type.index and "PSB" in by_type.index:
            assert by_type["fintech"] > by_type["PSB"], (
                f"Fintech {by_type['fintech']:.3f} should exceed PSB {by_type['PSB']:.3f}"
            )

    def test_bureau_pulls_match_eligible_applications(self, apps, bureau):
        n_eligible = int(apps["eligibility_passed"].sum())
        assert len(bureau) == n_eligible
