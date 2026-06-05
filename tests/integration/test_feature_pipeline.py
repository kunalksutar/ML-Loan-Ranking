"""
Integration test for the Section 5 feature engineering pipeline.

Runs the full feature pipeline on a small generated dataset and asserts:
  - All 57 expected features are present
  - No forbidden features appear in the output
  - Expected correlation directions hold
  - Lead-level split integrity (no lead spans two splits)
  - No nulls in the ML feature columns
  - Leakage invariant preserved
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.lead_generator import generate_leads
from src.simulation.bank_generator import generate_banks
from src.simulation.application_generator import generate_applications
from src.features.interaction_features import build_feature_dataset, split_dataset
from src.features.feature_registry import (
    ALL_FEATURES,
    BANK_FEATURES,
    FORBIDDEN_FEATURES,
    GROUP_KEY,
    INTERACTION_FEATURES,
    LEAD_FEATURES,
    TARGET,
    TEMPORAL_FEATURES,
)

ARCHETYPE_PATH = "configs/bank_archetypes.yaml"
SEED = 7
N_LEADS = 500
N_BANKS = 36


@pytest.fixture(scope="module")
def leads() -> pd.DataFrame:
    return generate_leads(n=N_LEADS, seed=SEED)


@pytest.fixture(scope="module")
def banks() -> pd.DataFrame:
    return generate_banks(seed=SEED, archetype_path=ARCHETYPE_PATH)


@pytest.fixture(scope="module")
def apps(leads, banks) -> pd.DataFrame:
    return generate_applications(leads, banks, seed=SEED)


@pytest.fixture(scope="module")
def features(leads, banks, apps, tmp_path_factory) -> pd.DataFrame:
    """Build the feature dataset in a temporary directory."""
    tmp = tmp_path_factory.mktemp("data")

    # Write raw parquets to temp paths
    apps_path = str(tmp / "applications_raw.parquet")
    leads_path = str(tmp / "leads.parquet")
    banks_path = str(tmp / "banks.parquet")
    apps.to_parquet(apps_path, index=False)
    leads.to_parquet(leads_path, index=False)
    banks.to_parquet(banks_path, index=False)

    return build_feature_dataset(
        apps_path=apps_path,
        leads_path=leads_path,
        banks_path=banks_path,
    )


# ---------------------------------------------------------------------------
# Schema and completeness
# ---------------------------------------------------------------------------

class TestFeatureSchema:
    def test_row_count(self, features, leads, banks):
        assert len(features) == N_LEADS * N_BANKS

    def test_all_57_features_present(self, features):
        missing = [f for f in ALL_FEATURES if f not in features.columns]
        assert missing == [], f"Missing features: {missing}"

    def test_lead_feature_count(self, features):
        present = [f for f in LEAD_FEATURES if f in features.columns]
        assert len(present) == len(LEAD_FEATURES)

    def test_bank_feature_count(self, features):
        present = [f for f in BANK_FEATURES if f in features.columns]
        assert len(present) == len(BANK_FEATURES)

    def test_interaction_feature_count(self, features):
        present = [f for f in INTERACTION_FEATURES if f in features.columns]
        assert len(present) == len(INTERACTION_FEATURES)

    def test_temporal_feature_count(self, features):
        present = [f for f in TEMPORAL_FEATURES if f in features.columns]
        assert len(present) == len(TEMPORAL_FEATURES)

    def test_key_columns_present(self, features):
        for col in ["application_id", "lead_id", "bank_id", TARGET, "eligibility_passed"]:
            assert col in features.columns, f"Missing key column: {col}"

    def test_no_nulls_in_features(self, features):
        null_counts = features[ALL_FEATURES].isnull().sum()
        null_features = null_counts[null_counts > 0]
        assert len(null_features) == 0, f"Nulls in features: {null_features.to_dict()}"

    def test_converted_binary(self, features):
        assert features[TARGET].isin([0, 1]).all()


# ---------------------------------------------------------------------------
# Leakage prevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    def test_no_forbidden_features(self, features):
        present = [f for f in FORBIDDEN_FEATURES if f in features.columns]
        assert present == [], f"Forbidden features found: {present}"

    def test_leakage_invariant(self, features):
        bad = features.loc[~features["eligibility_passed"], TARGET].sum()
        assert bad == 0, f"Leakage: {bad} ineligible pairs have converted=1"

    def test_no_near_perfect_correlation(self, features):
        target = features[TARGET]
        for feat in ALL_FEATURES:
            if features[feat].nunique() < 2:
                continue
            corr = abs(float(features[feat].corr(target)))
            assert corr < 0.95, (
                f"|corr({feat}, converted)| = {corr:.4f} — potential leakage"
            )


# ---------------------------------------------------------------------------
# Expected correlation directions (CLAUDE.md §6)
# ---------------------------------------------------------------------------

class TestCorrelationDirections:
    def test_cibil_gap_positive(self, features):
        corr = float(features["cibil_gap"].corr(features[TARGET]))
        assert corr > 0, f"cibil_gap corr={corr:.4f} (expected positive)"

    def test_foir_headroom_positive(self, features):
        corr = float(features["foir_headroom"].corr(features[TARGET]))
        assert corr > 0, f"foir_headroom corr={corr:.4f} (expected positive)"

    def test_bureau_fatigue_negative(self, features):
        corr = float(features["bureau_fatigue_flag"].corr(features[TARGET]))
        assert corr < 0, f"bureau_fatigue_flag corr={corr:.4f} (expected negative)"

    def test_income_type_match_positive(self, features):
        corr = float(features["income_type_match"].corr(features[TARGET]))
        assert corr > 0, f"income_type_match corr={corr:.4f} (expected positive)"

    def test_dpd90_exceeds_negative(self, features):
        corr = float(features["dpd90_exceeds_bank_max"].corr(features[TARGET]))
        assert corr < 0, f"dpd90_exceeds_bank_max corr={corr:.4f} (expected negative)"

    def test_cibil_score_positive(self, features):
        corr = float(features["cibil_score"].corr(features[TARGET]))
        assert corr > 0, f"cibil_score corr={corr:.4f} (expected positive)"


# ---------------------------------------------------------------------------
# Encoding correctness
# ---------------------------------------------------------------------------

class TestEncodings:
    def test_income_type_enc_no_unknown(self, features):
        assert (features["income_type_enc"] >= 0).all(), \
            "income_type_enc has -1 (unknown value)"

    def test_bank_type_enc_no_unknown(self, features):
        assert (features["bank_type_enc"] >= 0).all(), \
            "bank_type_enc has -1 (unknown value)"

    def test_risk_appetite_enc_range(self, features):
        # conservative=0, moderate=1, aggressive=2
        assert features["risk_appetite_enc"].isin([0, 1, 2]).all()

    def test_documentation_strictness_enc_range(self, features):
        # low=0, medium=1, high=2
        assert features["documentation_strictness_enc"].isin([0, 1, 2]).all()

    def test_income_type_enc_range(self, features):
        # salaried=0, self_employed=1, business=2, freelance=3
        assert features["income_type_enc"].isin([0, 1, 2, 3]).all()


# ---------------------------------------------------------------------------
# Interaction feature value ranges
# ---------------------------------------------------------------------------

class TestInteractionFeatureRanges:
    def test_amount_fit_flag_binary(self, features):
        assert features["amount_fit_flag"].isin([0, 1]).all()

    def test_amount_position_clipped(self, features):
        assert (features["amount_position"] >= 0.0).all()
        assert (features["amount_position"] <= 1.0).all()

    def test_bureau_fatigue_excess_non_negative(self, features):
        assert (features["bureau_fatigue_excess"] >= 0.0).all()

    def test_cibil_in_sweet_spot_binary(self, features):
        assert features["cibil_in_sweet_spot"].isin([0, 1]).all()

    def test_income_type_match_binary(self, features):
        assert features["income_type_match"].isin([0, 1]).all()

    def test_loan_type_match_binary(self, features):
        assert features["loan_type_match"].isin([0, 1]).all()

    def test_geography_match_binary(self, features):
        assert features["geography_match"].isin([0, 1]).all()

    def test_dpd90_exceeds_binary(self, features):
        assert features["dpd90_exceeds_bank_max"].isin([0, 1]).all()


# ---------------------------------------------------------------------------
# Temporal feature correctness
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    def test_days_since_first_app_non_negative(self, features):
        assert (features["days_since_first_application"] >= 0).all()

    def test_enquiry_velocity_non_negative(self, features):
        assert (features["enquiry_velocity_weekly"] >= 0.0).all()

    def test_is_reapplication_binary(self, features):
        assert features["is_reapplication"].isin([0, 1]).all()

    def test_sequence_num_zero_for_ineligible(self, features):
        ineligible = features[~features["eligibility_passed"]]
        assert (ineligible["application_sequence_num"] == 0).all()

    def test_sequence_num_positive_for_eligible(self, features):
        eligible = features[features["eligibility_passed"]]
        assert (eligible["application_sequence_num"] >= 1).all()


# ---------------------------------------------------------------------------
# Lead-level split integrity
# ---------------------------------------------------------------------------

class TestSplitIntegrity:
    def test_lead_level_split_no_overlap(self, features):
        train, val, test = split_dataset(features, seed=SEED)

        train_leads = set(train[GROUP_KEY].unique())
        val_leads = set(val[GROUP_KEY].unique())
        test_leads = set(test[GROUP_KEY].unique())

        assert len(train_leads & val_leads) == 0, "Train and val share leads"
        assert len(train_leads & test_leads) == 0, "Train and test share leads"
        assert len(val_leads & test_leads) == 0, "Val and test share leads"

    def test_split_covers_all_leads(self, features):
        all_leads = set(features[GROUP_KEY].unique())
        train, val, test = split_dataset(features, seed=SEED)
        split_leads = (
            set(train[GROUP_KEY].unique())
            | set(val[GROUP_KEY].unique())
            | set(test[GROUP_KEY].unique())
        )
        assert split_leads == all_leads

    def test_split_proportions_approximate(self, features):
        train, val, test = split_dataset(features, seed=SEED)
        total_leads = features[GROUP_KEY].nunique()
        train_frac = train[GROUP_KEY].nunique() / total_leads
        val_frac = val[GROUP_KEY].nunique() / total_leads
        test_frac = test[GROUP_KEY].nunique() / total_leads
        assert 0.65 <= train_frac <= 0.75, f"train fraction={train_frac:.3f}"
        assert 0.10 <= val_frac <= 0.20, f"val fraction={val_frac:.3f}"
        assert 0.10 <= test_frac <= 0.20, f"test fraction={test_frac:.3f}"

    def test_target_present_in_all_splits(self, features):
        train, val, test = split_dataset(features, seed=SEED)
        for name, split in [("train", train), ("val", val), ("test", test)]:
            assert TARGET in split.columns, f"'{TARGET}' missing in {name} split"
            assert split[TARGET].sum() > 0, f"No positives in {name} split"
