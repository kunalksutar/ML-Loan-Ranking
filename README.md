## Lead Generation (Section 4.1) — `data/raw/leads.parquet`

| Category | Metric | Actual | Threshold / Target | Pass |
|---|---|---|---|---|
| **Schema** | Rows generated | 10,000 | 10,000 | ✓ |
| **Schema** | Columns | 31 | 31 | ✓ |
| **Schema** | Null values | 0 | 0 | ✓ |
| **Schema** | Unique lead IDs | 10,000 | 10,000 | ✓ |
| **Constraints** | Age range | [23, 62] | [23, 62] | ✓ |
| **Constraints** | CIBIL score range | [458, 900] | [300, 900] | ✓ |
| **Constraints** | FOIR range | [0.10, 0.90] | (0.05, 0.95) | ✓ |
| **Constraints** | `age_at_maturity` max | 79 | < 80 | ✓ |
| **Constraints** | `loan_tenure_months` min | 12 | ≥ 12 | ✓ |
| **Correlations** | CIBIL vs annual income | 0.491 | > 0.30 | ✓ |
| **Correlations** | DPD-30 vs CIBIL | −0.401 | < −0.25 | ✓ |
| **Correlations** | Age vs annual income | 0.119 | > 0 | ✓ |
| **Distributions** | Salaried share | 54.1 % | ~55 % | ✓ |
| **Distributions** | Self-employed share | 25.8 % | ~25 % | ✓ |
| **Distributions** | Business share | 15.1 % | ~15 % | ✓ |
| **Distributions** | Freelance share | 5.0 % | ~5 % | ✓ |
| **Distributions** | Leads with fixed deposits | 39.1 % | ~40 % | ✓ |
| **Distributions** | Top loan type (personal) | 29.9 % | — | — |
| **Distributions** | 2nd loan type (home) | 21.6 % | — | — |
| **Distributions** | 3rd loan type (car) | 16.8 % | — | — |
| **Key Metrics** | Mean CIBIL score | 663 | — | — |
| **Key Metrics** | Mean annual income (INR) | 778,843 | — | — |
| **Key Metrics** | Mean FOIR | 0.335 | — | — |
| **Key Metrics** | Mean enquiry count (6 m) | 0.727 | — | — |
| **Key Metrics** | Mean DPD-30 count | 0.543 | — | — |
| **Tests** | Unit tests passed | 39 / 39 | 39 / 39 | ✓ |

---

## Bank Generation (Section 4.2) — `data/raw/banks.parquet`

| Category | Metric | Actual | Threshold / Target | Pass |
|---|---|---|---|---|
| **Schema** | Rows generated | 36 | 36 | ✓ |
| **Schema** | Columns | 47 | — | ✓ |
| **Schema** | Null values | 0 | 0 | ✓ |
| **Schema** | Unique bank IDs | 36 | 36 | ✓ |
| **Schema** | Unique bank names | 36 | 36 | ✓ |
| **Schema** | Unique sigmoid intercepts | 36 | 36 | ✓ |
| **Archetype Counts** | PSB | 8 | 8 | ✓ |
| **Archetype Counts** | Private | 10 | 10 | ✓ |
| **Archetype Counts** | NBFC | 8 | 8 | ✓ |
| **Archetype Counts** | Fintech | 6 | 6 | ✓ |
| **Archetype Counts** | HFC | 4 | 4 | ✓ |
| **Archetype Checks** | `preferred_cibil_min` > `min_cibil_score` | 36 / 36 | 36 / 36 | ✓ |
| **Archetype Checks** | HFCs offer only [home, lap] | 4 / 4 | 4 / 4 | ✓ |
| **Archetype Checks** | Fintechs are `digital_only` | 6 / 6 | 6 / 6 | ✓ |
| **Differentiation** | `approval_base_rate` std | 0.122 | > 0.05 | ✓ |
| **Differentiation** | `min_cibil_score` std | 38.3 | > 10 | ✓ |
| **Differentiation** | PSB min floor vs fintech max floor | 700 vs 640 (gap = 60 pts) | > 0 | ✓ |
| **Differentiation** | `cibil_weight` range | [0.586, 1.970] | — | — |
| **Differentiation** | `dti_weight` range | [0.501, 1.196] | — | — |
| **PSB** | `min_cibil_score` range | [700, 723] | [700, 725] | ✓ |
| **PSB** | `approval_base_rate` range | [0.292, 0.419] | [0.28, 0.42] | ✓ |
| **PSB** | `disbursal_success_rate` range | [0.828, 0.918] | [0.82, 0.92] | ✓ |
| **PSB** | `disbursal_speed_days` range | [11, 19] days | [10, 25] days | ✓ |
| **PSB** | Intercept range | [−0.514, 0.064] | — | — |
| **Private** | `min_cibil_score` range | [680, 715] | [680, 715] | ✓ |
| **Private** | `approval_base_rate` range | [0.258, 0.398] | [0.25, 0.40] | ✓ |
| **Private** | `disbursal_success_rate` range | [0.855, 0.929] | [0.85, 0.93] | ✓ |
| **Private** | `disbursal_speed_days` range | [5, 15] days | [5, 15] days | ✓ |
| **Private** | Intercept range | [−1.404, −0.468] | — | — |
| **NBFC** | `min_cibil_score` range | [637, 678] | [620, 680] | ✓ |
| **NBFC** | `approval_base_rate` range | [0.412, 0.577] | [0.38, 0.58] | ✓ |
| **NBFC** | `disbursal_success_rate` range | [0.751, 0.878] | [0.75, 0.88] | ✓ |
| **NBFC** | `disbursal_speed_days` range | [3, 8] days | [3, 8] days | ✓ |
| **NBFC** | Intercept range | [−1.116, −0.256] | — | — |
| **Fintech** | `min_cibil_score` range | [583, 640] | [580, 650] | ✓ |
| **Fintech** | `approval_base_rate` range | [0.522, 0.688] | [0.48, 0.70] | ✓ |
| **Fintech** | `disbursal_success_rate` range | [0.824, 0.893] | [0.80, 0.92] | ✓ |
| **Fintech** | `disbursal_speed_days` range | [1, 3] days | [1, 4] days | ✓ |
| **Fintech** | Intercept range | [−0.798, 0.239] | — | — |
| **HFC** | `min_cibil_score` range | [682, 718] | [680, 720] | ✓ |
| **HFC** | `approval_base_rate` range | [0.270, 0.306] | [0.25, 0.38] | ✓ |
| **HFC** | `disbursal_success_rate` range | [0.805, 0.833] | [0.78, 0.90] | ✓ |
| **HFC** | `disbursal_speed_days` range | [23, 35] days | [15, 45] days | ✓ |
| **HFC** | Intercept range | [−1.392, −0.790] | — | — |
| **Tests** | Unit tests passed | 59 / 59 | 59 / 59 | ✓ |
| **Tests** | Combined suite (leads + banks) | 98 / 98 | 98 / 98 | ✓ |

---

## Feature Engineering (Section 5) — `data/processed/applications_features.parquet`

| Category | Metric | Actual | Threshold / Target | Pass |
|---|---|---|---|---|
| **Schema** | Total rows (10K leads × 36 banks) | 360,000 | 360,000 | ✓ |
| **Schema** | Total ML features | 57 | 57 | ✓ |
| **Schema** | Null feature cells | 0 | 0 | ✓ |
| **Feature Groups** | Lead features | 25 | 25 | ✓ |
| **Feature Groups** | Bank features | 13 | 13 | ✓ |
| **Feature Groups** | Interaction features | 15 | 15 | ✓ |
| **Feature Groups** | Temporal features | 4 | 4 | ✓ |
| **Leakage Prevention** | Forbidden features in output | 0 | 0 | ✓ |
| **Leakage Prevention** | `converted=1` where ineligible | 0 | 0 | ✓ |
| **Leakage Prevention** | Max `\|corr(feature, converted)\|` | 0.321 | < 0.95 | ✓ |
| **Encodings** | `income_type_enc` unknown values | 0 | 0 | ✓ |
| **Encodings** | `bank_type_enc` unknown values | 0 | 0 | ✓ |
| **Encodings** | `risk_appetite_enc` range | [0, 2] | {0,1,2} | ✓ |
| **Encodings** | `documentation_strictness_enc` range | [0, 2] | {0,1,2} | ✓ |
| **Interaction** | `amount_position` clipped to [0, 1] | True | [0, 1] | ✓ |
| **Interaction** | `bureau_fatigue_excess` non-negative | True | ≥ 0 | ✓ |
| **Interaction** | `geography_match` binary | True | {0, 1} | ✓ |
| **Correlations** | `corr(cibil_gap, converted)` | +0.320 | positive | ✓ |
| **Correlations** | `corr(foir_headroom, converted)` | +0.064 | positive | ✓ |
| **Correlations** | `corr(bureau_fatigue_flag, converted)` | −0.042 | negative | ✓ |
| **Correlations** | `corr(income_type_match, converted)` | +0.061 | positive | ✓ |
| **Correlations** | `corr(cibil_score, converted)` | +0.110 | positive | ✓ |
| **Top Interaction** | #1 by `\|corr\|` | `cibil_vs_sweet_spot_dist` (0.321) | — | — |
| **Top Interaction** | #2 by `\|corr\|` | `cibil_gap` (0.320) | — | — |
| **Top Interaction** | #3 by `\|corr\|` | `cibil_in_sweet_spot` (0.235) | — | — |
| **Splits** | Train rows (70 % of leads) | 252,000 | ~70 % | ✓ |
| **Splits** | Val rows (15 % of leads) | 54,000 | ~15 % | ✓ |
| **Splits** | Test rows (15 % of leads) | 54,000 | ~15 % | ✓ |
| **Splits** | Lead-level no-overlap guarantee | True | True | ✓ |
| **Artifacts** | `feature_schema.json` generated | True | True | ✓ |
| **Tests** | Unit tests (interaction features) | 51 / 51 | 51 / 51 | ✓ |
| **Tests** | Integration tests (feature pipeline) | 40 / 40 | 40 / 40 | ✓ |
| **Tests** | Combined suite (all phases) | 260 / 260 | 260 / 260 | ✓ |

---

## Preprocessing & Train/Val/Test Strategy (Sections 8 & 9) — `src/preprocessing/`

| Category | Metric | Actual | Threshold / Target | Pass |
|---|---|---|---|---|
| **Pipeline** | Input features (ALL_FEATURES) | 57 | 57 | ✓ |
| **Pipeline** | Transform groups partition ALL_FEATURES | 57 / 57, no overlap | exact partition | ✓ |
| **Pipeline** | `log1p → StandardScaler` columns | 4 | 4 | ✓ |
| **Pipeline** | `StandardScaler` columns | 37 | 37 | ✓ |
| **Pipeline** | Passthrough (ordinal + binary) columns | 11 | 11 | ✓ |
| **Pipeline** | `OneHotEncoder` columns (nominal `_enc`) | 5 → 24 expanded | `handle_unknown="ignore"` | ✓ |
| **Pipeline** | Output dimensions after transform | 76 | 52 + 24 | ✓ |
| **Pipeline** | Null cells after transform | 0 | 0 | ✓ |
| **Fitting Rule** | Preprocessor fit on train split only | True | train-only | ✓ |
| **Fitting Rule** | val/test transformed with frozen train statistics | True | True | ✓ |
| **Artifacts** | `preprocessor.pkl` saved & round-trips | True | True | ✓ |
| **Split Strategy** | Method | Lead-level 70/15/15 | lead-level | ✓ |
| **Split Strategy** | Train / val / test unique leads | 7,000 / 1,500 / 1,500 | 70 / 15 / 15 % | ✓ |
| **Split Strategy** | Train / val / test rows | 252,000 / 54,000 / 54,000 | — | ✓ |
| **Split Strategy** | Lead-level overlap across splits | 0 | 0 | ✓ |
| **Cross-Validation** | `GroupKFold` folds | 5 | 5 | ✓ |
| **Cross-Validation** | Lead overlap between CV train/val folds | 0 | 0 | ✓ |
| **Cross-Validation** | Each row appears as CV-validation exactly once | True | True | ✓ |
| **Class Imbalance** | Train conversion rate | 10.56 % | [10 %, 22 %] | ✓ |
| **Class Imbalance** | `scale_pos_weight` (train) | 8.47 | ~5–8× (typical) | ✓ |
| **Class Imbalance** | Strategy | `scale_pos_weight`, no SMOTE | per CLAUDE.md §9 | ✓ |
| **Tests** | Unit tests (`test_pipeline_builder`) | 16 / 16 | 16 / 16 | ✓ |
| **Tests** | Unit tests (`test_splitting`) | 12 / 12 | 12 / 12 | ✓ |
| **Tests** | Integration tests (`test_preprocessing_pipeline`) | 11 / 11 | 11 / 11 | ✓ |
| **Tests** | Combined suite (all phases) | 299 / 299 | 299 / 299 | ✓ |

---

## Application Generation (Section 4.3) — `data/processed/applications_raw.parquet`

| Category | Metric | Actual | Threshold / Target | Pass |
|---|---|---|---|---|
| **Schema** | Total pairs (10K leads × 36 banks) | 360,000 | 360,000 | ✓ |
| **Schema** | Columns | 16 | 16 | ✓ |
| **Schema** | Unique application IDs | 360,000 | 360,000 | ✓ |
| **Schema** | Unique lead IDs covered | 10,000 | 10,000 | ✓ |
| **Schema** | Unique bank IDs covered | 36 | 36 | ✓ |
| **Schema** | Nullable fields (by design) | 2,037,807 | Expected | ✓ |
| **Acceptance** | Overall conversion rate | 10.60 % | [10 %, 22 %] | ✓ |
| **Acceptance** | Per-bank conversion rate std | 0.0508 | > 0.05 | ✓ |
| **Acceptance** | Per-bank conversion rate range | [1.6 %, 23.4 %] | — | — |
| **Acceptance** | Leakage (converted=1 where ineligible) | 0 | 0 | ✓ |
| **Eligibility** | Eligibility pass rate | 13.25 % | — | — |
| **Eligibility** | Top rejection: `cibil_below_minimum` | 39.5 % of pairs | — | — |
| **Eligibility** | 2nd rejection: `state_not_covered` | 23.3 % of pairs | — | — |
| **Eligibility** | 3rd rejection: `income_type_not_accepted` | 11.3 % of pairs | — | — |
| **Eligibility** | 4th rejection: `loan_type_not_offered` | 6.7 % of pairs | — | — |
| **Application Status** | `not_submitted` (ineligible) | 86.75 % | — | — |
| **Application Status** | `disbursed` (converted=1) | 10.60 % | — | — |
| **Application Status** | `rejected` (eligible, not approved) | 1.58 % | — | — |
| **Application Status** | `disbursal_failed` | 1.07 % | — | — |
| **Bank Type Conversion** | NBFC | 16.23 % | Highest (aggressive) | ✓ |
| **Bank Type Conversion** | Private | 12.37 % | — | — |
| **Bank Type Conversion** | PSB | 8.45 % | — | — |
| **Bank Type Conversion** | Fintech | 8.69 % | — | — |
| **Bank Type Conversion** | HFC | 2.07 % | Lowest (home/lap only) | ✓ |
| **Correlations** | `corr(cibil_score, annual_income)` | 0.491 | > 0.30 | ✓ |
| **Correlations** | `corr(cibil_score, dpd_30_count)` | −0.401 | < −0.20 | ✓ |
| **Correlations** | `corr(foir_headroom, converted)` | 0.064 | > 0.05 | ✓ |
| **Correlations** | `corr(bureau_fatigue_flag, converted)` | −0.042 | < −0.02 | ✓ |
| **Correlations** | `corr(cibil_gap, converted)` | 0.320 | > 0 (positive) | ✓ |
| **Bureau Pulls** | Total pull records | 47,695 | = eligible pairs | ✓ |
| **Bureau Pulls** | Hard enquiry share | 85.2 % | ~85 % | ✓ |
| **Bureau Pulls** | Soft enquiry share | 14.8 % | ~15 % | ✓ |
| **Tests** | Unit tests (approval + application) | 46 / 46 | 46 / 46 | ✓ |
| **Tests** | Integration test (full pipeline) | 25 / 25 | 25 / 25 | ✓ |
| **Tests** | Combined suite (all phases) | 169 / 169 | 169 / 169 | ✓ |
