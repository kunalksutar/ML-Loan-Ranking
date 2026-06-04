# CLAUDE.md — Lead-to-Bank Ranking System
## Project Specification & Blueprint

> **Framing**: Learning-to-Rank (Pointwise → LambdaMART) over pairwise (lead × bank) records
> **Target**: `P(disbursed=1 | lead, bank)` — scalar per pair, used to rank banks per lead
> **Data**: 100% synthetic, Indian lending market distributions, no PII

---

## 1. Project Scope

### Problem
Given a loan lead's financial/credit profile, rank all candidate banks by their likelihood to approve **and** disburse the loan.

### Runtime Architecture (3 Stages)
```
[Lead arrives]
      │
      ▼
Stage 1 — Eligibility Engine (rule-based, deterministic)
      │   Hard-reject ineligible banks; output shortlist of K eligible banks (3–12 of N)
      ▼
Stage 2 — Scoring Model (XGBoost pointwise)
      │   Score each (lead, bank) pair → P(disbursed)
      ▼
Stage 3 — Ranking Layer
      │   Sort by score; apply business overrides
      ▼
[API response: ordered bank list with scores]
```

### Boundaries
- **In scope**: Synthetic data generation, feature engineering, ML model, ranking evaluation, FastAPI stub
- **Out of scope**: Real bureau API integration, bank APIs, payments, UI
- **Upgrade path**: Phase 1 = XGBoost pointwise; Phase 2 = LightGBM LambdaMART once conversion logs exist

### Hard Constraints
- All (lead × bank) pairs included in dataset — not just submitted applications (avoids collider bias)
- Features must be available at time of application submission — no future leakage
- CIBIL, income, and obligations must be causally generated, not independently sampled
- Overall disbursal rate must land in 12–18% range
- Train/val/test split must be **lead-level**, not row-level

---

## 2. Folder Structure

```
lead_bank_ranking/
├── CLAUDE.md
├── README.md
├── pyproject.toml                     # Poetry or uv
├── .env.example
│
├── configs/
│   ├── data_config.yaml               # n_leads, n_banks, random seed
│   ├── bank_archetypes.yaml           # Bank type definitions and eligibility rule ranges
│   ├── model_config.yaml              # Hyperparameters, training settings
│   └── feature_config.yaml           # Feature groups, encodings, log-transform list
│
├── data/
│   ├── raw/
│   │   ├── leads.parquet
│   │   ├── banks.parquet
│   │   └── bureau_pulls.parquet
│   ├── processed/
│   │   ├── applications_raw.parquet   # Full cross-join with outcomes
│   │   ├── applications_features.parquet
│   │   └── applications_splits/
│   │       ├── train.parquet
│   │       ├── val.parquet
│   │       └── test.parquet
│   └── artifacts/
│       ├── feature_schema.json
│       └── data_report.html           # ydata-profiling output
│
├── src/
│   ├── simulation/
│   │   ├── lead_generator.py          # Causal chain generation
│   │   ├── bank_generator.py          # Archetype-based generation
│   │   ├── application_generator.py   # Cross-join + eligibility + outcomes
│   │   ├── approval_simulator.py      # Sigmoid approval probability engine
│   │   ├── disbursal_simulator.py     # Post-approval disbursal success/failure
│   │   ├── bureau_simulator.py        # Enquiry history, bureau fatigue
│   │   └── distributions.py           # Shared statistical helpers
│   ├── features/
│   │   ├── lead_features.py
│   │   ├── bank_features.py
│   │   ├── interaction_features.py    # cibil_gap, foir_headroom, etc.
│   │   ├── temporal_features.py
│   │   └── feature_registry.py        # Central feature name/group/type registry
│   ├── preprocessing/
│   │   ├── encoders.py
│   │   ├── scalers.py
│   │   ├── imputers.py
│   │   └── pipeline_builder.py        # Assembles sklearn ColumnTransformer pipeline
│   ├── validation/
│   │   ├── schema_validator.py        # Pandera checks
│   │   ├── distribution_checks.py     # Realism assertions
│   │   ├── leakage_detector.py        # Forbidden feature + correlation checks
│   │   └── correlation_audit.py       # VIF + Spearman analysis
│   ├── eda/
│   │   ├── univariate.py
│   │   ├── bivariate.py
│   │   ├── correlation_matrix.py
│   │   └── report_generator.py        # ydata-profiling HTML report
│   ├── modeling/
│   │   ├── trainer.py                 # Training loop with GroupKFold CV
│   │   ├── ranker.py                  # Per-lead ranking logic
│   │   ├── evaluator.py               # AUC, NDCG@K, Recall@K, MRR
│   │   ├── tuner.py                   # Optuna search (optimize NDCG@3)
│   │   └── model_registry.py          # Save/load versioned artifacts
│   ├── eligibility/
│   │   ├── rule_engine.py             # Hard-rule evaluator (ordered by rejection frequency)
│   │   └── rejection_attributor.py
│   └── api/
│       ├── main.py                    # FastAPI app: POST /rank, GET /health
│       ├── schemas.py                 # Pydantic request/response models
│       └── predictor.py               # Inference pipeline wrapper
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_error_analysis.ipynb
│
├── tests/
│   ├── unit/
│   │   ├── test_lead_generator.py
│   │   ├── test_bank_generator.py
│   │   ├── test_approval_simulator.py
│   │   ├── test_interaction_features.py
│   │   ├── test_eligibility_engine.py
│   │   └── test_ranker.py
│   └── integration/
│       ├── test_full_simulation_pipeline.py
│       ├── test_feature_pipeline.py
│       └── test_api_endpoint.py
│
├── experiments/mlflow/
├── models/v1/
│   ├── metadata.json
│   ├── xgb_model.ubj
│   ├── preprocessor.pkl
│   ├── eligibility_rules.json
│   └── feature_schema.json
└── logs/pipeline.log
```

---

## 3. Schema Design

### 3.1 Lead
**File**: `data/raw/leads.parquet` — all fields must be available at application time

| Field | Type | Constraints / Notes |
|---|---|---|
| `lead_id` | UUID | PK |
| `created_at` | datetime | |
| `age` | int | 23–62 |
| `gender` | str | M \| F \| Other |
| `city_tier` | int | 1 (metro) \| 2 \| 3 (rural) |
| `state` | str | Indian state code |
| `income_type` | str | salaried \| self_employed \| business \| freelance |
| `employer_category` | str | PSU \| private_listed \| private_unlisted \| MNC \| govt |
| `annual_income` | float | INR; log-normal; causally derived from age + income_type |
| `work_experience_years` | float | correlated with age |
| `current_employer_tenure_yrs` | float | |
| `cibil_score` | int | 300–900; causally derived from income + age |
| `dpd_30_count` | int | inversely correlated with cibil_score |
| `dpd_90_count` | int | inversely correlated with cibil_score; weighted higher than dpd_30 |
| `enquiry_count_6m` | int | 0–10+; correlated with delinquency risk |
| `settled_loans` | int | negative credit signal |
| `written_off_loans` | int | strong negative credit signal |
| `existing_loan_count` | int | |
| `monthly_obligations` | float | total existing EMIs; derived from target FOIR |
| `credit_card_spend_monthly` | float | |
| `savings_balance` | float | avg quarterly balance |
| `loan_type` | str | personal \| home \| car \| education \| business \| gold \| lap |
| `loan_amount_requested` | float | INR; constrained by income and loan_type |
| `loan_tenure_months` | int | 12 \| 24 \| 36 \| 48 \| 60 \| 84 \| 120 \| 180 \| 240 |
| `foir` | float | monthly_obligations / (annual_income / 12); must be in (0.10, 0.90) |
| `dti_ratio` | float | (obligations + new EMI estimate) / monthly income |
| `loan_to_income_ratio` | float | loan_amount / annual_income |
| `credit_utilization` | float | cc_spend / estimated cc_limit |
| `age_at_maturity` | int | age + tenure_months/12; must be < 80 |

**Generation invariants**:
- `cibil_score` ↔ `annual_income`: ρ must be 0.40–0.50
- `dpd_30_count` ↔ `cibil_score`: ρ must be negative (< −0.20)
- `foir` ∈ (0.10, 0.90) — enforce during generation, not post-hoc
- `loan_amount_requested` ≤ income × 5 for personal; ≤ income × 80 for home

---

### 3.2 Bank
**File**: `data/raw/banks.parquet`

| Field Group | Fields |
|---|---|
| **Identity** | `bank_id` (UUID PK), `name`, `bank_type` (PSB \| private \| NBFC \| fintech \| HFC \| cooperative) |
| **Coverage** | `states_covered` (list), `city_tiers_served` (list), `digital_only` (bool) |
| **Products** | `loan_types_offered` (list) |
| **Hard Eligibility Rules** | `min_cibil_score`, `max_cibil_score`, `min_annual_income`, `max_annual_income`, `max_foir`, `max_dti_ratio`, `min_age`, `max_age_at_maturity`, `max_enquiries_6m`, `max_dpd_30_count`, `max_dpd_90_count`, `max_written_off_loans`, `max_settled_loans`, `accepted_income_types`, `accepted_employer_categories`, `min_employer_tenure_months`, `min_work_experience_years` |
| **Loan Terms** | `min_loan_amount`, `max_loan_amount`, `min_tenure_months`, `max_tenure_months`, `interest_rate_min`, `interest_rate_max`, `processing_fee_pct` |
| **Behavioral** (simulator only) | `risk_appetite` (conservative \| moderate \| aggressive), `approval_base_rate`, `disbursal_success_rate`, `disbursal_speed_days`, `documentation_strictness` (low \| medium \| high) |
| **Sweet Spot** (simulator only) | `preferred_cibil_min`, `preferred_cibil_max`, `preferred_loan_size_min`, `preferred_loan_size_max`, `cibil_weight`, `dti_weight`, `amount_fit_weight`, `intercept` |

---

### 3.3 Application (Core Fact Table)
**File**: `data/processed/applications_raw.parquet`

| Field | Type | Notes |
|---|---|---|
| `application_id` | UUID | PK |
| `lead_id` | UUID | FK → Lead |
| `bank_id` | UUID | FK → Bank |
| `submitted_at` | datetime | |
| `bank_responded_at` | datetime \| null | |
| `disbursed_at` | datetime \| null | |
| `application_sequence_num` | int | order of bank application per lead |
| `eligibility_passed` | bool | Stage 1 result |
| `eligibility_failure_reason` | str \| null | |
| `application_status` | str | not_submitted \| submitted \| under_review \| approved \| rejected \| withdrawn \| disbursed \| disbursal_failed |
| `rejection_reason` | str \| null | **forbidden in training features** |
| `approved_amount` | float \| null | **forbidden in training features** |
| `approved_rate` | float \| null | **forbidden in training features** |
| `disbursed_amount` | float \| null | **forbidden in training features** |
| `disbursal_failure_reason` | str \| null | **forbidden in training features** |
| **`converted`** | int | **TARGET: 1 if disbursed, else 0** |

**Critical**: All (lead × bank) pairs are included — ineligible pairs have `converted = 0`, `eligibility_passed = False`. Never drop ineligible pairs from training data.

---

### 3.4 Bureau Pull Log
**File**: `data/raw/bureau_pulls.parquet`

Fields: `pull_id` (UUID), `lead_id`, `bank_id`, `pulled_at` (datetime), `cibil_score_at_pull` (int), `enquiry_type` (hard \| soft)

Used to compute rolling `effective_enquiry_count` at any application timestamp — enables temporal bureau fatigue simulation.

---

## 4. Synthetic Data Generation

### 4.1 Lead Generator — Causal Chain
**Must follow this exact order.** Each step depends on previous outputs.

```
Step 1  Demographics      age ~ Normal(38,10).clip(23,62)
                          income_type ~ Cat([0.55, 0.25, 0.15, 0.05])
                          city_tier ~ Cat([0.35, 0.40, 0.25])
                          state ~ Cat(state_population_weights)

Step 2  Income            base ~ LogNormal(μ by income_type, σ=0.6–1.0)
                          annual_income = base × (1 + 0.02 × max(0, age−25))

Step 3  CIBIL             cibil_mean = 620 + (age−25)×0.8 + (income/100k)×4.5
                          cibil_score ~ Normal(cibil_mean, 55).clip(300, 900)

Step 4  Delinquency       dpd_prob = max(0, (750−cibil) / 500)
                          dpd_30_count ~ Poisson(dpd_prob × 3)
                          dpd_90_count ~ Poisson(dpd_prob × 0.5)
                          written_off_loans ~ Poisson(dpd_prob × 0.1)
                          settled_loans ~ Poisson(dpd_prob × 0.2)

Step 5  Obligations       target_foir ~ Beta(2, 4)  [mode ≈ 0.28]
                          monthly_obligations = (income/12) × target_foir

Step 6  Enquiries         enquiry_count ~ Poisson(dpd_prob × 4).clip(0, 10)
                          [correlated with delinquency — desperate borrowers shop more]

Step 7  Behavior          cc_spend ~ LogNormal(log(income×0.12), 0.4)
                          savings_balance ~ LogNormal(log(income×0.15), 0.6)

Step 8  Loan Request      loan_type: weighted by income_type and income level
                          loan_amount: sampled from income-scaled range by loan_type
                          tenure: categorical, consistent with loan_type norms

Step 9  Derived Ratios    foir, dti_ratio, loan_to_income_ratio,
                          credit_utilization, age_at_maturity — all computed
```

**Pitfalls**:
- Use LogNormal for income — never Normal (produces negatives)
- Apply `.clip(300, 900)` to CIBIL only after computing `dpd_prob`
- Never sample `loan_amount` uniformly — must depend on `annual_income` and `loan_type`
- Do not compute derived ratios independently — derive from the already-generated primitives

---

### 4.2 Bank Generator — Archetypes
**File**: `configs/bank_archetypes.yaml` — defines parameter ranges per type; each bank samples from its archetype range.

| Attribute | PSB | Private | NBFC | Fintech | HFC |
|---|---|---|---|---|---|
| Count | 8 | 10 | 8 | 6 | 4 |
| `min_cibil_score` | 700–725 | 680–715 | 620–680 | 580–650 | 680–720 |
| `max_foir` | 0.55–0.65 | 0.60–0.70 | 0.65–0.78 | 0.70–0.85 | 0.55–0.65 |
| `max_enquiries_6m` | 2–3 | 3–5 | 4–6 | 5–8 | 2–4 |
| `approval_base_rate` | 0.28–0.42 | 0.25–0.40 | 0.38–0.58 | 0.48–0.70 | 0.25–0.38 |
| `disbursal_success_rate` | 0.82–0.92 | 0.85–0.93 | 0.75–0.88 | 0.80–0.92 | 0.78–0.90 |
| `disbursal_speed_days` | 10–25 | 5–15 | 3–8 | 1–4 | 15–45 |
| `documentation_strictness` | high | medium | low | low | high |
| `accepted_income_types` | salaried, business, govt | salaried, self_employed, business | all | salaried, self_employed, freelance | salaried, business |

**Requirement**: Each bank must have a unique behavioral signature. `preferred_cibil_band` must sit above `min_cibil_score`. Some banks must create natural conflicts (lead ideal for NBFC = borderline for PSB) — this is what makes ranking non-trivial.

---

### 4.3 Application Generator — Logic Flow

```
1. Cross-join all leads × all banks → N_leads × N_banks rows
2. Apply eligibility engine → mark eligibility_passed + failure_reason
3. For eligible pairs only → run approval_simulator → sample approved = Bernoulli(p)
4. For approved pairs → run disbursal_simulator → sample disbursed = Bernoulli(p)
5. Assign timestamps: submitted_at, response times by bank_type, application_sequence_num
6. Assign rejection reasons for non-approved pairs (priority-ordered — see §10.1)
7. Set converted = 1 if status == "disbursed" else 0
8. Save ALL pairs including ineligible ones
```

**Acceptance criteria**:
- Overall `converted.mean()` ∈ [0.10, 0.22]
- Per-bank approval rate std > 0.05 (banks must behave differently)
- Zero converted=1 rows where `eligibility_passed=False`

---

### 4.4 Approval Simulator — Specification

**Algorithm**: sigmoid(score) where score is a bank-specific linear combination.

**Score components** (each bank has its own weight vector):
- `bank.intercept` — calibrated base propensity
- `cibil_weight × tanh((cibil − bank.min_cibil) / 100)` — nonlinear, diminishing returns above sweet spot
- `dti_weight × (bank.max_foir − lead.foir) × 3` — FOIR headroom
- `−2.5` if `enquiry_count_6m > bank.max_enquiries_6m` — hard bureau fatigue penalty
- `−0.25 × dpd_30_count`, `−0.90 × dpd_90_count`, `−4.0 × written_off_loans`, `−0.8 × settled_loans`
- `amount_fit_weight × clip(amount_fit_score, −1, 1)` — loan size in bank's preferred range
- `+0.3` if employer_category in bank's premium employer list
- `+ Normal(0, 0.25)` — idiosyncratic bank noise

**Hard overrides**: return 0.0 immediately if `income_type not in accepted_income_types`

**Calibration requirement**: For each bank, simulate 10K leads and verify `mean(approved)` matches `bank.approval_base_rate ± 0.05`. Adjust `intercept` until calibrated. Noise std must not exceed 0.3 or bank differentiation breaks.

---

### 4.5 Disbursal Simulator — Specification

**Base**: `bank.disbursal_success_rate`

**Adjustments**:
- Income type modifier: salaried +0.05, business ±0, self_employed −0.05, freelance −0.10
- Liquidity stress: `−max(0, foir − 0.5) × 0.2`
- Savings buffer: `+min(1.0, savings / (loan_amount × 0.1)) × 0.05`
- Output clipped to [0.50, 0.98]

**Expected output**: disbursal rate 80–92% conditional on approval.

---

## 5. Feature Engineering

### Feature Registry (`src/features/feature_registry.py`)

**LEAD_FEATURES** (25 features):
`age, annual_income, cibil_score, foir, dti_ratio, loan_to_income_ratio, enquiry_count_6m, dpd_30_count, dpd_90_count, written_off_loans, settled_loans, existing_loan_count, work_experience_years, current_employer_tenure_yrs, credit_card_spend_monthly, savings_balance, loan_amount_requested, loan_tenure_months, credit_utilization, age_at_maturity, income_type_enc, employer_category_enc, loan_type_enc, city_tier, gender_enc`

**BANK_FEATURES** (13 features):
`min_cibil_score, max_foir, min_annual_income, approval_base_rate, disbursal_speed_days, interest_rate_min, interest_rate_max, max_enquiries_6m, max_loan_amount, min_loan_amount, bank_type_enc, risk_appetite_enc, documentation_strictness_enc`

**INTERACTION_FEATURES** (15 features — computed after joining lead × bank):
| Feature | Formula |
|---|---|
| `cibil_gap` | `lead.cibil_score − bank.min_cibil_score` |
| `foir_headroom` | `bank.max_foir − lead.foir` |
| `income_headroom` | `lead.annual_income − bank.min_annual_income` |
| `income_headroom_ratio` | `income_headroom / bank.min_annual_income` |
| `amount_fit_flag` | `1 if loan_amount ∈ [bank.min, bank.max]` |
| `amount_position` | `(amount − bank.min) / (bank.max − bank.min)` clipped [0,1] |
| `income_type_match` | bool |
| `loan_type_match` | bool |
| `geography_match` | bool |
| `bureau_fatigue_flag` | `1 if enquiry_count > bank.max_enquiries_6m` |
| `bureau_fatigue_excess` | `max(0, enquiry_count − bank.max_enquiries_6m)` |
| `cibil_in_sweet_spot` | `1 if cibil ∈ [bank.preferred_cibil_min, bank.preferred_cibil_max]` |
| `cibil_vs_sweet_spot_dist` | `abs(cibil − sweet_spot_center)` |
| `age_maturity_headroom` | `bank.max_age_at_maturity − lead.age_at_maturity` |
| `dpd90_exceeds_bank_max` | bool |

**TEMPORAL_FEATURES** (4 features):
`application_sequence_num, days_since_first_application, enquiry_velocity_weekly, is_reapplication`

**Constants**: `TARGET = "converted"`, `GROUP_KEY = "lead_id"`

**Rule**: Compute interaction features only after joining lead and bank tables. Never store bank attributes on the lead table.

---

## 6. EDA & Statistical Analysis

**Required analyses** (output to `notebooks/01_data_exploration.ipynb` + `data/artifacts/data_report.html`):

- Univariate distributions: histograms for continuous, frequency bars for categorical
- Expected skew: income, savings, loan_amount (right-skewed — verify log-normal fit)
- Bank profile: `min_cibil_score` distribution by `bank_type`; approval rate range across banks
- Application analysis: conversion rate by `bank_type`; by `income_type × bank_type` (heatmap); rejection reason frequency
- Spearman correlation heatmap of LEAD_FEATURES; ALL_FEATURES vs `converted`
- VIF analysis for multicollinearity detection

**Expected correlation directions** (must hold — if not, re-check generation):

| Feature | Expected direction with `converted` |
|---|---|
| `cibil_score` | positive |
| `enquiry_count_6m` | negative |
| `foir_headroom` | positive |
| `bureau_fatigue_flag` | strong negative |
| `income_type_match` | positive |
| `cibil_gap` | positive |

---

## 7. Data Validation & Leakage Prevention

### Schema Constraints (Pandera — `src/validation/schema_validator.py`)
- `cibil_score`: int, [300, 900]
- `foir`: float, [0.05, 0.95]
- `annual_income`: float, > 0
- `age`: int, [21, 65]
- `enquiry_count_6m`: int, [0, 20]
- `age_at_maturity`: int, < 80
- `converted`: int, ∈ {0, 1}
- `converted == 0` for all rows where `eligibility_passed == False`

### Forbidden Features (must never appear in training feature matrix)
`rejection_reason`, `approved_amount`, `approved_rate`, `approved_tenure_months`, `disbursed_amount`, `application_status`, `bank_responded_at`, `disbursed_at`, `disbursal_failure_reason`

### Correlation-Based Leakage Check
Any feature with `|corr(feature, converted)| > 0.95` must be flagged and investigated before training.

### Simulation Realism Assertions
- `converted.mean()` ∈ [0.10, 0.22]
- Per-bank `converted` rate std > 0.05
- `corr(cibil_score, annual_income)` > 0.30
- `corr(cibil_score, dpd_30_count)` < −0.20
- `corr(foir_headroom, converted)` > 0.10
- `corr(bureau_fatigue_flag, converted)` < −0.05

---

## 8. Preprocessing Pipeline

**File**: `src/preprocessing/pipeline_builder.py` — assembles a `sklearn ColumnTransformer`

**Encoding strategy** (`configs/feature_config.yaml`):

| Transform | Features |
|---|---|
| `log1p → StandardScaler` | `annual_income, loan_amount_requested, savings_balance, credit_card_spend_monthly, monthly_obligations` |
| `StandardScaler` | All other continuous numeric features |
| `OrdinalEncoder` (ordered) | `risk_appetite` [conservative→aggressive], `documentation_strictness` [low→high], `city_tier` [1→3] |
| `OneHotEncoder` (handle_unknown=ignore) | `income_type, employer_category, loan_type, bank_type, gender` |
| Passthrough | All boolean / already-binary interaction features |

**Rule**: Fit the preprocessor on training data only. Transform val and test separately. Never fit on full dataset before splitting.

---

## 9. Train / Validation / Test Strategy

### Split Method: Lead-Level (not row-level)
All rows for a given `lead_id` must land in the same split. A lead that spans train and test allows the model to memorize lead-level signals, equivalent to target leakage.

**Proportions**: 70% train / 15% val / 15% test (by unique lead count)

### Cross-Validation
Use `GroupKFold(n_splits=5)` with `groups = lead_id` for all CV during hyperparameter tuning.

### Class Imbalance
Use `scale_pos_weight = n_negative / n_positive` in XGBoost (typically 5–8×). Do **not** use SMOTE — it disrupts the pairwise structure of the data by creating synthetic (lead, bank) pairs that are inconsistent.

---

## 10. Model Training & Ranking Pipeline

### Stage 1: Eligibility Engine (`src/eligibility/rule_engine.py`)

Check hard rules in this order (cheapest / most common rejections first):
1. `income_type` not in `accepted_income_types`
2. `state` not in `states_covered`
3. `cibil_score < min_cibil_score`
4. `cibil_score > max_cibil_score`
5. `annual_income < min_annual_income` or `> max_annual_income`
6. `foir > max_foir`
7. `age_at_maturity > max_age_at_maturity`
8. `enquiry_count_6m > max_enquiries_6m`
9. `dpd_90_count > max_dpd_90_count`
10. `written_off_loans > max_written_off_loans`
11. `loan_type` not in `loan_types_offered`
12. `loan_amount_requested` not in `[min_loan_amount, max_loan_amount]`

Returns: `(eligible: bool, failure_reason: str | None)`

### Stage 2: XGBoost Pointwise Scorer (`src/modeling/trainer.py`)
- `objective = binary:logistic`
- `eval_metric = [auc, logloss]`
- `tree_method = hist`
- `early_stopping_rounds = 50`
- Preprocessor fitted on train, applied to val/test
- Full sklearn Pipeline: `[preprocessor → XGBClassifier]`

### Stage 3: Ranker (`src/modeling/ranker.py`)
- Input: eligible bank candidates for a lead (post-Stage 1)
- Score each pair with `model.predict_proba(X)[:, 1]`
- Sort descending by score, return top-K banks with scores
- `top_k` default = 5

### Stage 2 Upgrade Path: LambdaMART (`src/modeling/trainer.py`)
When real conversion logs exist, upgrade to LightGBM with `objective = lambdarank`, `metric = ndcg`, `ndcg_eval_at = [3, 5]`, `label_gain = [0, 1]`. Use `group` parameter = applications per lead.

---

## 11. Evaluation Metrics

### Pointwise (per application)
- AUC-ROC, F1 (class 1), Precision, Recall

### Ranking (per lead group) — **Primary**
- **NDCG@K** (K=1,3,5): was the correct bank ranked high?
- **Recall@K** (K=1,3,5): was at least one converting bank in top-K?
- **MRR** (Mean Reciprocal Rank): reciprocal rank of first converting bank

### Minimum Acceptance Thresholds

| Metric | Threshold |
|---|---|
| AUC-ROC | ≥ 0.82 |
| NDCG@3 | ≥ 0.70 |
| Recall@3 | ≥ 0.75 |
| MRR | ≥ 0.60 |
| F1 (class 1) | ≥ 0.65 |

**Note**: Optimize hyperparameters for **NDCG@3**, not AUC. AUC is a pointwise proxy; NDCG@3 is the actual business objective.

---

## 12. Hyperparameter Tuning

**Framework**: Optuna (`src/modeling/tuner.py`)
**Objective function**: maximize `NDCG@3` on validation set
**CV strategy**: `GroupKFold(n_splits=5)` with `groups = lead_id`

**Search space**:

| Parameter | Range | Type |
|---|---|---|
| `max_depth` | 3–8 | int |
| `learning_rate` | 0.01–0.3 | float (log) |
| `n_estimators` | 100–600 | int |
| `min_child_weight` | 1–20 | int |
| `subsample` | 0.5–1.0 | float |
| `colsample_bytree` | 0.5–1.0 | float |
| `gamma` | 0–2.0 | float |
| `reg_alpha` | 0–2.0 | float |
| `reg_lambda` | 0.5–5.0 | float |
| `scale_pos_weight` | 3.0–10.0 | float |

**Minimum trials**: 100

---

## 13. Experiment Tracking

**Framework**: MLflow (local store at `experiments/mlflow/`)
**Naming convention**: `{model_type}_{feature_group}_{YYYYMMDD}_{run_id}`

**Log per run**:
- All hyperparameters + dataset parameters (`n_leads`, `n_banks`, `feature_count`)
- Metrics: `val_auc`, `val_ndcg_3`, `val_ndcg_5`, `val_recall_3`, `val_mrr`, `val_f1`
- Artifacts: `feature_schema.json`, `data_report.html`, feature importance plot
- Model artifact: `xgb_model.ubj` + `preprocessor.pkl`

---

## 14. Error Analysis

**File**: `notebooks/04_error_analysis.ipynb`

**Required analyses**:
1. **False Negatives** (converted=1 but ranked > 3): profile these leads and banks — expect borderline eligibility, low cibil_gap, high enquiry_count
2. **False Positives** (converted=0 but ranked ≤ 3): passed eligibility but soft-rejected; investigate which features misled the model
3. **Per-bank AUC**: flag any bank where model AUC < 0.70 — that bank may need a separate model or rule override
4. **Per-income-type ranking quality**: compare NDCG@3 across salaried / self_employed / business / freelance — expect freelance to underperform
5. **Top feature importance**: `cibil_gap`, `foir_headroom`, `bureau_fatigue_flag`, `income_type_match`, `amount_fit_flag` must appear in top 10

---

## 15. Logging & Monitoring

**Library**: `structlog` (JSON output) — module-level logger in every `src/` file

**Mandatory log points**:

| Stage | Level | Event |
|---|---|---|
| Lead generation start/end | INFO | `n_leads`, `seed`, `time_seconds` |
| CIBIL-income correlation low | WARN | `correlation` value if < 0.30 |
| Application generation end | INFO | `total_pairs`, `conversion_rate` |
| Conversion rate out of range | ERROR | `actual`, `expected=[0.10,0.22]` |
| Feature engineering end | INFO | `feature_count`, `null_count` |
| High feature-feature correlation | WARN | `f1`, `f2`, `corr` |
| Training epoch | INFO | `epoch`, `val_auc`, `val_ndcg3` |
| NDCG@3 below threshold | ERROR | `actual`, `threshold=0.70` |
| API ranking request | INFO | `lead_id`, `n_eligible_banks`, `latency_ms` |
| No eligible banks | WARN | `lead_id` |
| Model inference failure | ERROR | `lead_id`, `error` |

---

## 16. Testing Strategy

### Unit Tests (`tests/unit/`) — must cover
- `test_lead_generator`: CIBIL-income correlation ≥ 0.30; FOIR bounds; zero nulls
- `test_bank_generator`: each archetype produces correct parameter ranges
- `test_approval_simulator`: ineligible income_type returns 0.0; perfect lead returns > 0.70
- `test_interaction_features`: formula correctness for all 15 interaction features
- `test_eligibility_engine`: geography rejection; all-pass for perfect lead; each rule triggered individually
- `test_ranker`: ranking sorts descending; top-K length correct; score ∈ [0, 1]

### Integration Tests (`tests/integration/`) — must cover
- `test_full_simulation_pipeline`: run end-to-end generation; assert schema, conversion rate, no leakage, lead-boundary split integrity
- `test_feature_pipeline`: all interaction features present after join; no forbidden features in output
- `test_api_endpoint`: POST `/rank` returns 200 with `ranked_banks` list; GET `/health` returns 200

### Tooling
- Framework: `pytest`
- Coverage: `pytest --cov=src --cov-report=html` — target ≥ 80% line coverage on `src/simulation/` and `src/eligibility/`

---

## 17. Productionization

### API (`src/api/`)
- **Framework**: FastAPI
- **Endpoints**: `POST /rank` (returns ranked bank list), `GET /health`
- **Input validation**: Pydantic models with field-level constraints matching Lead schema bounds
- **Response**: `ranked_banks: list[{bank_id, bank_name, bank_type, rank_score, interest_rate_min, disbursal_speed_days}]`
- **Error handling**: return 200 with empty list if no eligible banks; return 500 with structured error on inference failure
- **Latency target**: < 50ms p99 for 50-bank candidate set

### Model Artifact Bundle (`models/v1/`)
| File | Contents |
|---|---|
| `metadata.json` | version, feature list, thresholds, training date, dataset stats |
| `xgb_model.ubj` | XGBoost binary model |
| `preprocessor.pkl` | Fitted sklearn ColumnTransformer |
| `eligibility_rules.json` | Bank eligibility rules (serialized from bank entity) |
| `feature_schema.json` | Feature names, types, expected ranges |

---

## 18. Scalability & Future Improvements

### Phase 2 Upgrades
- **LambdaMART**: LightGBM `rank:ndcg` once real conversion logs are available
- **Calibration**: Platt scaling or isotonic regression on model outputs
- **Real bureau data**: Replace synthetic CIBIL with live bureau API pull (CIBIL/Experian) under consent
- **Feedback loop**: Capture real disbursal outcomes → monthly retraining
- **Cold-start**: New bank defaults to archetype parameters until 500+ applications accumulate
- **Temporal CIBIL**: Track score changes over time; model bureau fatigue from actual pull timestamps
- **Multi-objective ranking**: Pareto-rank on (P(disbursal), interest_rate, disbursal_speed) with user preference weights

### Scale Targets
| Dimension | Target |
|---|---|
| Dataset | 10M leads × 50 banks = 500M pairs |
| Training | ~45 min (LightGBM, GPU) |
| Inference latency | < 50ms per lead (p99) |
| API throughput | 2000 RPS |

---

## Quick Reference — Execution Order

```bash
python -m src.simulation.lead_generator       --config configs/data_config.yaml
python -m src.simulation.bank_generator       --config configs/bank_archetypes.yaml
python -m src.simulation.application_generator --config configs/data_config.yaml
python -m src.validation.schema_validator
python -m src.validation.leakage_detector
python -m src.validation.distribution_checks
python -m src.features.interaction_features
python -m src.eda.report_generator
python -m src.modeling.trainer                --config configs/model_config.yaml
python -m src.modeling.tuner                  --n_trials 100
python -m src.modeling.evaluator              --split test
uvicorn src.api.main:app                      --reload --port 8000
pytest tests/ -v --cov=src
```

## Agent Instructions (Reusable — Apply to All Future Sessions)

- Follow modular and reusable coding practices.
- Maintain a clean and minimal repository structure.
- Avoid unnecessary files, folders, and abstractions.
- Use a root-level `.venv`, `Makefile`, `.gitignore`, and `requirements.txt`.
- Use Python 3.11 consistently across the project.
- Use meaningful Git branches and descriptive commit messages.
- Run validations and tests before completing any task.
- Validate synthetic datasets against schema constraints, business rules, and causal correlations.
- Use MLflow for all experiment tracking and model versioning.
- Keep implementations production-oriented and maintainable.
- Complete and fully validate one implementation phase before moving to the next.
- After each phase, summarize: what was implemented, what validations passed, and recommended next steps.
