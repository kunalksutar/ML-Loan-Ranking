# CLAUDE.md — Lead-to-Bank Ranking System
## Project Blueprint & Implementation Guide

> **For**: Claude Code Agent (VSCode)
> **Purpose**: End-to-end implementation guideline — from synthetic data generation to production-ready ML ranking pipeline
> **Framing**: Learning-to-Rank (Pointwise → LambdaMART) over pairwise (lead × bank) application records
> **Target**: Predict P(disbursed | lead, bank), rank banks per lead to maximize loan conversion

---

## Table of Contents

1. [Project Scope & Constraints](#1-project-scope--constraints)
2. [Folder Structure](#2-folder-structure)
3. [Schema Design](#3-schema-design)
4. [Synthetic Data Generation Pipeline](#4-synthetic-data-generation-pipeline)
5. [Feature Engineering Pipeline](#5-feature-engineering-pipeline)
6. [EDA & Statistical Analysis](#6-eda--statistical-analysis)
7. [Data Validation & Leakage Prevention](#7-data-validation--leakage-prevention)
8. [Preprocessing Pipeline](#8-preprocessing-pipeline)
9. [Train / Validation / Test Strategy](#9-train--validation--test-strategy)
10. [Model Training & Ranking Pipeline](#10-model-training--ranking-pipeline)
11. [Evaluation & Metrics](#11-evaluation--metrics)
12. [Hyperparameter Tuning](#12-hyperparameter-tuning)
13. [Experiment Tracking](#13-experiment-tracking)
14. [Error Analysis](#14-error-analysis)
15. [Logging & Monitoring](#15-logging--monitoring)
16. [Testing Strategy](#16-testing-strategy)
17. [Productionization](#17-productionization)
18. [Scalability & Future Improvements](#18-scalability--future-improvements)

---

## 1. Project Scope & Constraints

### Problem Statement
Given a loan lead with known financial and credit attributes, rank a set of lending banks in order of their likelihood to **approve and disburse** the loan. The system must handle many-to-many relationships: each lead may be eligible for multiple banks; each bank has distinct eligibility rules and risk appetite.

### ML Framing
```
Input:  (lead_features, bank_features, interaction_features) for each candidate pair
Output: P(disbursed=1 | lead, bank) — a scalar per pair, used to rank banks per lead
Target: Binary — 1 if application reached "disbursed" status, 0 otherwise
```

### Three-Stage Runtime Architecture
```
[Lead arrives]
      │
      ▼
Stage 1: Eligibility Engine (rule-based, deterministic)
      │   Hard-reject ineligible banks from candidate set
      │   Output: Shortlist of K eligible banks (typically 3–12 of N)
      ▼
Stage 2: Scoring Model (XGBoost / LightGBM pointwise)
      │   Score each (lead, bank) pair → P(disbursed)
      ▼
Stage 3: Ranking Layer (sort by score, apply business overrides)
      │   Output: Ordered bank list
      ▼
[API response: ranked bank list with scores]
```

### Scope Boundaries
- **In scope**: Synthetic data generation, feature engineering, ML model, ranking evaluation, REST API stub
- **Out of scope**: Real bureau integration, bank API calls, payment processing, UI
- **Data**: 100% synthetic; no PII; distributions calibrated to Indian lending market

### Key Constraints
- No real-world training data — all data is programmatically simulated
- Approval rates must reflect realistic market ratios (~12–18% overall disbursal rate)
- Features must be available at time of application (no future leakage)
- Every (lead, bank) pair must be evaluated, not just submitted applications
- CIBIL scores, income, and obligations must be **causally correlated**, not independently sampled

---

## 2. Folder Structure

```
lead_bank_ranking/
│
├── CLAUDE.md                          # This file
├── README.md                          # Project overview for humans
├── pyproject.toml                     # Dependencies (Poetry or uv)
├── .env.example                       # Environment variable template
├── .gitignore
│
├── configs/
│   ├── data_config.yaml               # Simulation parameters (n_leads, n_banks, seed)
│   ├── bank_archetypes.yaml           # Bank type definitions and eligibility rules
│   ├── model_config.yaml              # Model hyperparameters and training settings
│   └── feature_config.yaml           # Feature groups, encodings, transformations
│
├── data/
│   ├── raw/                           # Generated raw entities (leads, banks)
│   │   ├── leads.parquet
│   │   ├── banks.parquet
│   │   └── bureau_pulls.parquet
│   ├── processed/                     # Application pairs with features
│   │   ├── applications_raw.parquet   # All pairs before feature engineering
│   │   ├── applications_features.parquet
│   │   └── applications_splits/
│   │       ├── train.parquet
│   │       ├── val.parquet
│   │       └── test.parquet
│   └── artifacts/
│       ├── feature_schema.json        # Feature names, types, expected ranges
│       └── data_report.html           # ydata-profiling report
│
├── src/
│   ├── __init__.py
│   │
│   ├── simulation/                    # Module 1: Synthetic data generation
│   │   ├── __init__.py
│   │   ├── lead_generator.py          # Lead entity generation (causal chain)
│   │   ├── bank_generator.py          # Bank entity generation (archetype-based)
│   │   ├── application_generator.py   # Cross-join + eligibility filter + simulation
│   │   ├── approval_simulator.py      # Sigmoid-based approval probability engine
│   │   ├── disbursal_simulator.py     # Post-approval disbursal success/failure
│   │   ├── bureau_simulator.py        # Enquiry history + bureau fatigue
│   │   └── distributions.py           # Reusable statistical distribution helpers
│   │
│   ├── features/                      # Module 2: Feature engineering
│   │   ├── __init__.py
│   │   ├── lead_features.py           # Lead-level feature computation
│   │   ├── bank_features.py           # Bank-level feature computation
│   │   ├── interaction_features.py    # Cross features: cibil_gap, foir_headroom, etc.
│   │   ├── temporal_features.py       # Bureau fatigue, application sequence features
│   │   └── feature_registry.py        # Central feature name/type/group registry
│   │
│   ├── preprocessing/                 # Module 3: Preprocessing pipeline
│   │   ├── __init__.py
│   │   ├── encoders.py                # Categorical encoding (OHE, ordinal, target)
│   │   ├── scalers.py                 # StandardScaler / RobustScaler wrappers
│   │   ├── imputers.py                # Imputation strategies per feature group
│   │   └── pipeline_builder.py        # Assemble sklearn Pipeline
│   │
│   ├── validation/                    # Module 4: Data validation
│   │   ├── __init__.py
│   │   ├── schema_validator.py        # Pandera schema checks
│   │   ├── distribution_checks.py     # Statistical sanity checks
│   │   ├── leakage_detector.py        # Feature-target leakage detection
│   │   └── correlation_audit.py       # Feature correlation & VIF analysis
│   │
│   ├── eda/                           # Module 5: EDA & visualization
│   │   ├── __init__.py
│   │   ├── univariate.py              # Distribution plots per feature
│   │   ├── bivariate.py               # Feature vs. target analysis
│   │   ├── correlation_matrix.py      # Spearman/Pearson heatmaps
│   │   └── report_generator.py        # Auto-generate EDA HTML report
│   │
│   ├── modeling/                      # Module 6: Model training & ranking
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Training loop with CV
│   │   ├── ranker.py                  # Ranking logic per lead group
│   │   ├── evaluator.py               # AUC, NDCG@K, Recall@K, MRR
│   │   ├── tuner.py                   # Optuna hyperparameter search
│   │   └── model_registry.py          # Save/load model artifacts
│   │
│   ├── eligibility/                   # Module 7: Rule-based eligibility engine
│   │   ├── __init__.py
│   │   ├── rule_engine.py             # Hard-rule evaluator
│   │   └── rejection_attributor.py    # Assign rejection reason
│   │
│   └── api/                           # Module 8: Production API stub
│       ├── __init__.py
│       ├── main.py                    # FastAPI app
│       ├── schemas.py                 # Pydantic request/response models
│       └── predictor.py               # Inference pipeline
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
├── experiments/
│   └── mlflow/                        # MLflow tracking store (local)
│
└── logs/
    └── pipeline.log
```

---

## 3. Schema Design

### 3.1 Lead Entity

**File**: `data/raw/leads.parquet`
**Generated by**: `src/simulation/lead_generator.py`

```python
# All fields available at application time (no future leakage)
Lead:
  # Identity
  lead_id:                    UUID (primary key)
  created_at:                 datetime

  # Demographics
  age:                        int        # range: 23–62
  gender:                     str        # "M" | "F" | "Other"
  city_tier:                  int        # 1 (metro) | 2 (tier-2) | 3 (rural)
  state:                      str        # Indian state code
  pin_code:                   str        # 6-digit

  # Employment
  income_type:                str        # "salaried" | "self_employed" | "business" | "freelance"
  employer_category:          str        # "PSU" | "private_listed" | "private_unlisted" | "MNC" | "govt"
  annual_income:              float      # INR; log-normal distribution
  work_experience_years:      float      # years; correlated with age
  current_employer_tenure_yrs: float    # years at current employer

  # Credit Profile
  cibil_score:                int        # 300–900; correlated with income/age
  dpd_30_count:               int        # delinquencies ≥30 DPD in 12m; inversely ~ cibil
  dpd_90_count:               int        # severe delinquencies; inversely ~ cibil
  enquiry_count_6m:           int        # hard bureau pulls; 0–8+
  settled_loans:              int        # negotiated settlements (negative signal)
  written_off_loans:          int        # write-offs (strong negative signal)
  existing_loan_count:        int        # active loans

  # Financial Behavior
  monthly_obligations:        float      # total existing EMIs; derived from FOIR
  credit_card_spend_monthly:  float      # avg monthly CC spend
  savings_balance:            float      # avg quarterly balance
  fixed_deposits:             float      # total FD value

  # Loan Request
  loan_type:                  str        # "personal" | "home" | "car" | "education" | "business" | "gold" | "lap"
  loan_amount_requested:      float      # INR; constrained by income
  loan_tenure_months:         int        # 12 | 24 | 36 | 48 | 60 | 84 | 120 | 180 | 240

  # Derived Ratios (computed in generation, not engineered later)
  foir:                       float      # monthly_obligations / (annual_income / 12)
  dti_ratio:                  float      # (monthly_obligations + new_emi_estimate) / monthly_income
  loan_to_income_ratio:       float      # loan_amount_requested / annual_income
  credit_utilization:         float      # cc_spend / estimated_cc_limit
  age_at_maturity:            int        # age + loan_tenure_months / 12
```

**Generation invariants** (enforced during generation, not post-hoc):
- `cibil_score` must be positively correlated with `annual_income` (ρ ≈ 0.40–0.50)
- `dpd_30_count` must be inversely correlated with `cibil_score`
- `foir` must be in (0.10, 0.90) — values outside this range are not realistic
- `loan_amount_requested` ≤ `annual_income × 5` for personal loans; ≤ 80× for home loans

---

### 3.2 Bank Entity

**File**: `data/raw/banks.parquet`
**Generated by**: `src/simulation/bank_generator.py`

```python
Bank:
  # Identity
  bank_id:                    UUID (primary key)
  name:                       str        # Faker-generated institution name
  bank_type:                  str        # "PSB" | "private" | "NBFC" | "fintech" | "HFC" | "cooperative"

  # Coverage
  states_covered:             List[str]  # subset of Indian states
  city_tiers_served:          List[int]  # [1] | [1,2] | [1,2,3]
  digital_only:               bool       # no branch required

  # Loan Products
  loan_types_offered:         List[str]

  # Eligibility Rules (hard rules for Stage 1 filter)
  min_cibil_score:            int        # 580–750 depending on type
  max_cibil_score:            int        # usually 900; some fintechs cap at 800
  min_annual_income:          float
  max_annual_income:          float      # some banks don't lend to very high income (niche)
  max_foir:                   float      # 0.55–0.85
  max_dti_ratio:              float
  min_age:                    int        # usually 21–25
  max_age_at_maturity:        int        # 60–70
  max_enquiries_6m:           int        # 2–6; banks differ significantly
  max_dpd_30_count:           int        # 0–3
  max_dpd_90_count:           int        # 0–1
  max_written_off_loans:      int        # 0 for conservative banks
  max_settled_loans:          int
  accepted_income_types:      List[str]
  accepted_employer_categories: List[str]
  min_employer_tenure_months: int
  min_work_experience_years:  float

  # Loan Terms
  min_loan_amount:            float
  max_loan_amount:            float
  min_tenure_months:          int
  max_tenure_months:          int
  interest_rate_min:          float      # APR %
  interest_rate_max:          float
  processing_fee_pct:         float

  # Behavioral Parameters (for simulator)
  risk_appetite:              str        # "conservative" | "moderate" | "aggressive"
  approval_base_rate:         float      # calibrated target approval rate
  disbursal_success_rate:     float      # P(disbursed | approved)
  disbursal_speed_days:       int
  documentation_strictness:   str        # "low" | "medium" | "high"

  # Sweet Spot Profile (hidden; used only in simulator)
  preferred_cibil_min:        int        # CIBIL band where bank is most competitive
  preferred_cibil_max:        int
  preferred_loan_size_min:    float
  preferred_loan_size_max:    float
  cibil_weight:               float      # how much CIBIL moves approval probability
  dti_weight:                 float
  amount_fit_weight:          float
  intercept:                  float      # sigmoid intercept for approval sim
```

---

### 3.3 Application Entity (Core Fact Table)

**File**: `data/processed/applications_raw.parquet`
**Generated by**: `src/simulation/application_generator.py`

```python
Application:
  # Keys
  application_id:             UUID (primary key)
  lead_id:                    UUID (FK → Lead)
  bank_id:                    UUID (FK → Bank)

  # Timing
  submitted_at:               datetime
  bank_responded_at:          datetime (nullable)
  disbursed_at:               datetime (nullable)
  application_sequence_num:   int        # 1st bank applied to, 2nd, etc. (per lead)

  # Stage 1: Eligibility
  eligibility_passed:         bool
  eligibility_failure_reason: str        # null if passed

  # Stage 2: Underwriting Decision
  application_status:         str
    # "not_submitted" (failed eligibility — still included in training data)
    # "submitted" | "under_review" | "approved" | "rejected" | "withdrawn"
    # "disbursed" | "disbursal_failed"
  rejection_reason:           str        # null if approved; see enum in schema design doc

  # Approval Details
  approved_amount:            float (nullable)
  approved_rate:              float (nullable)
  approved_tenure_months:     int (nullable)

  # Disbursal Details
  disbursed_amount:           float (nullable)
  disbursal_failure_reason:   str (nullable)

  # ── TARGET VARIABLE ──
  converted:                  int        # 1 if disbursed, 0 otherwise
```

**Critical design note**: Include ALL lead-bank pairs in the dataset — not just submitted applications. Pairs that failed eligibility have `converted = 0` and `eligibility_passed = False`. Training on only submitted applications creates selection bias (collider bias).

---

### 3.4 Bureau Pull Log (Temporal)

**File**: `data/raw/bureau_pulls.parquet`

```python
BureauPull:
  pull_id:                    UUID
  lead_id:                    UUID
  bank_id:                    UUID
  pulled_at:                  datetime
  cibil_score_at_pull:        int        # may differ from lead's static score
  enquiry_type:               str        # "hard" | "soft"
```

Used to compute rolling `effective_enquiry_count` at any point in time. Enables temporal simulation of bureau fatigue.

---

## 4. Synthetic Data Generation Pipeline

### Objective
Generate statistically realistic, causally coherent synthetic data representing the Indian lending market without any PII.

### 4.1 Lead Generator

**File**: `src/simulation/lead_generator.py`

**Causal Generation Chain** (must follow this order — each step depends on the previous):

```
Step 1: Sample demographics
        age ~ Normal(38, 10).clip(23, 62)
        income_type ~ Categorical([0.55, 0.25, 0.15, 0.05])
        city_tier ~ Categorical([0.35, 0.40, 0.25])
        state ~ Categorical(state_population_weights)

Step 2: Derive income from demographics
        base_income ~ LogNormal(μ by income_type, σ=0.6–1.0)
        career_multiplier = 1 + 0.02 * max(0, age - 25)
        annual_income = base_income * career_multiplier

Step 3: Derive credit score from income + age
        cibil_mean = 620 + (age-25)*0.8 + (annual_income/100000)*4.5
        cibil_score ~ Normal(cibil_mean, 55).clip(300, 900)

Step 4: Derive delinquency from credit score
        dpd_prob = max(0, (750 - cibil_score) / 500)
        dpd_30_count ~ Poisson(dpd_prob * 3)
        dpd_90_count ~ Poisson(dpd_prob * 0.5)
        written_off_loans ~ Poisson(dpd_prob * 0.1)
        settled_loans ~ Poisson(dpd_prob * 0.2)

Step 5: Derive obligations from income
        target_foir ~ Beta(2, 4)  # mode ~0.28
        monthly_obligations = (annual_income / 12) * target_foir

Step 6: Derive enquiry count
        # Correlated with delinquency (desperate borrowers shop more)
        enquiry_base = dpd_prob * 4
        enquiry_count_6m ~ Poisson(enquiry_base).clip(0, 10)

Step 7: Derive behavioral features from income
        cc_spend ~ LogNormal(log(annual_income * 0.12), 0.4)
        savings_balance ~ LogNormal(log(annual_income * 0.15), 0.6)

Step 8: Generate loan request consistent with profile
        loan_type: weighted by income_type and income level
        loan_amount: sampled from income-scaled range
        loan_tenure: categorical, consistent with loan_type norms

Step 9: Compute derived ratios
        foir = monthly_obligations / (annual_income / 12)
        loan_to_income_ratio = loan_amount / annual_income
        age_at_maturity = age + loan_tenure_months / 12
        credit_utilization = cc_spend / estimated_limit
```

**Expected Output**: DataFrame of N leads with all fields populated and causally consistent.

**Validation Checks**:
- `assert df['cibil_score'].corr(df['annual_income']) > 0.30`
- `assert df['dpd_30_count'].corr(df['cibil_score']) < -0.25`
- `assert (df['foir'] > 0.05).all() and (df['foir'] < 0.95).all()`
- `assert df['age_at_maturity'].max() < 80`
- No nulls in any required field

**Pitfalls**:
- Do NOT use `np.random.normal` for income — it produces negatives. Always use `lognormal`.
- Do NOT cap CIBIL to 900 before computing dpd_prob — apply `.clip()` only at the end.
- Do NOT sample `loan_amount` uniformly — it should depend on `annual_income` and `loan_type`.

---

### 4.2 Bank Generator

**File**: `src/simulation/bank_generator.py`

**Archetype Definitions** (`configs/bank_archetypes.yaml`):

```yaml
PSB:
  count: 8
  risk_appetite: conservative
  min_cibil_score: [700, 725]       # sampled uniformly within range
  min_annual_income: [200000, 350000]
  max_foir: [0.55, 0.65]
  max_enquiries_6m: [2, 3]
  accepted_income_types: [salaried, business, govt]
  approval_base_rate: [0.28, 0.42]
  disbursal_success_rate: [0.82, 0.92]
  disbursal_speed_days: [10, 25]
  documentation_strictness: high

private:
  count: 10
  risk_appetite: moderate
  min_cibil_score: [680, 715]
  min_annual_income: [180000, 320000]
  max_foir: [0.60, 0.70]
  max_enquiries_6m: [3, 5]
  accepted_income_types: [salaried, self_employed, business]
  approval_base_rate: [0.25, 0.40]
  disbursal_success_rate: [0.85, 0.93]
  disbursal_speed_days: [5, 15]
  documentation_strictness: medium

NBFC:
  count: 8
  risk_appetite: aggressive
  min_cibil_score: [620, 680]
  min_annual_income: [120000, 200000]
  max_foir: [0.65, 0.78]
  max_enquiries_6m: [4, 6]
  accepted_income_types: [salaried, self_employed, business, freelance]
  approval_base_rate: [0.38, 0.58]
  disbursal_success_rate: [0.75, 0.88]
  disbursal_speed_days: [3, 8]
  documentation_strictness: low

fintech:
  count: 6
  risk_appetite: aggressive
  min_cibil_score: [580, 650]
  min_annual_income: [100000, 180000]
  max_foir: [0.70, 0.85]
  max_enquiries_6m: [5, 8]
  accepted_income_types: [salaried, self_employed, freelance]
  approval_base_rate: [0.48, 0.70]
  disbursal_success_rate: [0.80, 0.92]
  disbursal_speed_days: [1, 4]
  documentation_strictness: low

HFC:
  count: 4
  risk_appetite: moderate
  min_cibil_score: [680, 720]
  min_annual_income: [250000, 400000]
  max_foir: [0.55, 0.65]
  loan_types_offered: [home, lap]
  approval_base_rate: [0.25, 0.38]
  disbursal_success_rate: [0.78, 0.90]
  disbursal_speed_days: [15, 45]
  documentation_strictness: high
```

**Pitfalls**:
- Each bank must have a unique behavioral signature — avoid all banks having identical `intercept` values in the approval simulator.
- `preferred_cibil_band` must be _above_ `min_cibil_score` to create a realistic sweet spot.
- Ensure some banks conflict (e.g., a lead that's perfect for NBFC is borderline for PSB) — this is what makes ranking non-trivial.

---

### 4.3 Application Generator

**File**: `src/simulation/application_generator.py`

```
Step 1: Cross-join all leads × all banks
        → total pairs = n_leads × n_banks (e.g., 10,000 × 32 = 320,000 rows)

Step 2: Apply eligibility engine (Stage 1)
        → mark each pair: eligibility_passed = True/False
        → assign eligibility_failure_reason for failed pairs

Step 3: For eligible pairs only — run approval simulator
        → compute approval_probability per pair
        → sample approved = Bernoulli(approval_probability)

Step 4: For approved pairs — run disbursal simulator
        → compute disbursal_probability (bank-level + lead behavioral factors)
        → sample disbursed = Bernoulli(disbursal_probability)

Step 5: Assign temporal attributes
        → submitted_at: lead.created_at + offset (hours to days)
        → response times by bank type (fintech: 1–3 days; PSB: 7–21 days)
        → application_sequence_num: order of submission per lead

Step 6: Assign rejection reasons for non-approved pairs

Step 7: Compute target variable
        → converted = 1 if status == "disbursed" else 0

Step 8: Save full cross-join (including ineligible pairs)
```

**Expected Output**: `applications_raw.parquet` with 320K+ rows, ~12–18% positive rate.

**Validation Checks**:
- `assert 0.10 <= df['converted'].mean() <= 0.22`
- `assert df[df['eligibility_passed']==False]['converted'].sum() == 0`
- `assert df['lead_id'].nunique() == n_leads`
- Check per-bank approval rates match archetype `approval_base_rate` within ±10%

---

### 4.4 Approval Simulator

**File**: `src/simulation/approval_simulator.py`

```python
def compute_approval_probability(lead: Lead, bank: Bank) -> float:
    """
    Sigmoid-based approval probability. Each bank has its own weight vector.
    The sigmoid ensures output is always in (0, 1).
    """
    score = bank.intercept

    # CIBIL contribution (nonlinear — tanh captures diminishing returns above sweet spot)
    cibil_normalized = (lead.cibil_score - bank.min_cibil_score) / 100.0
    score += bank.cibil_weight * np.tanh(cibil_normalized)

    # FOIR headroom (positive = lead is under bank's maximum)
    foir_headroom = bank.max_foir - lead.foir
    score += bank.dti_weight * foir_headroom * 3.0

    # Bureau fatigue (hard-coded penalty, not soft)
    if lead.enquiry_count_6m > bank.max_enquiries_6m:
        score -= 2.5

    # Delinquency penalties (DPD 90 weighted much more than DPD 30)
    score -= lead.dpd_30_count * 0.25
    score -= lead.dpd_90_count * 0.90
    score -= lead.written_off_loans * 4.0
    score -= lead.settled_loans * 0.8

    # Loan size fit within bank's preferred range
    amount_fit = 1.0 - (
        abs(lead.loan_amount_requested - bank.preferred_loan_size_midpoint)
        / bank.preferred_loan_size_range
    )
    score += bank.amount_fit_weight * np.clip(amount_fit, -1, 1)

    # Employer category bonus
    if lead.employer_category in bank.premium_employer_categories:
        score += 0.3

    # Bank-specific idiosyncratic noise (each bank is not perfectly rational)
    score += np.random.normal(0, 0.25)

    # Hard reject conditions (eligibility should catch these, but double-check)
    if lead.income_type not in bank.accepted_income_types:
        return 0.0

    return sigmoid(score)  # 1 / (1 + exp(-score))
```

**Pitfalls**:
- Set `np.random.seed` at the application level (not inside the function) for reproducibility.
- Calibrate intercepts by running 10K simulated leads through each bank and checking mean approval rate against archetype `approval_base_rate`. Adjust intercept until calibrated.
- Do NOT add noise with std > 0.5 — it will destroy bank differentiation.

---

### 4.5 Disbursal Simulator

**File**: `src/simulation/disbursal_simulator.py`

```python
def compute_disbursal_probability(lead: Lead, bank: Bank) -> float:
    """
    P(disbursed | approved). Independent of approval probability.
    Affected by documentation strictness, lead profile stability, and bank speed.
    """
    base = bank.disbursal_success_rate

    # Documentation quality proxy (income type + employer)
    doc_quality = {
        "salaried": 0.05,
        "business": 0.0,
        "self_employed": -0.05,
        "freelance": -0.10
    }[lead.income_type]

    # Lead volatility (high DTI + low savings = higher disbursal failure)
    liquidity_stress = max(0, lead.foir - 0.5) * 0.2
    savings_buffer = min(1.0, lead.savings_balance / (lead.loan_amount_requested * 0.1))

    prob = base + doc_quality - liquidity_stress + savings_buffer * 0.05
    return np.clip(prob, 0.50, 0.98)
```

**Expected Output**: Disbursal rate of 80–92% conditional on approval.

---

## 5. Feature Engineering Pipeline

### Objective
Transform raw entities into a flat feature matrix ready for ML. The feature matrix must contain **only features available at the time of application**.

### 5.1 Three Feature Groups

**File**: `src/features/feature_registry.py`

```python
LEAD_FEATURES = [
    "age", "annual_income", "cibil_score", "foir", "dti_ratio",
    "loan_to_income_ratio", "enquiry_count_6m", "dpd_30_count",
    "dpd_90_count", "written_off_loans", "settled_loans",
    "existing_loan_count", "work_experience_years",
    "current_employer_tenure_yrs", "credit_card_spend_monthly",
    "savings_balance", "loan_amount_requested", "loan_tenure_months",
    "credit_utilization", "age_at_maturity",
    # Encoded categoricals
    "income_type_enc", "employer_category_enc",
    "loan_type_enc", "city_tier", "gender_enc"
]

BANK_FEATURES = [
    "min_cibil_score", "max_foir", "min_annual_income",
    "approval_base_rate", "disbursal_speed_days",
    "interest_rate_min", "interest_rate_max",
    "max_enquiries_6m", "max_loan_amount", "min_loan_amount",
    # Encoded categoricals
    "bank_type_enc", "risk_appetite_enc", "documentation_strictness_enc"
]

INTERACTION_FEATURES = [
    "cibil_gap",                   # lead.cibil - bank.min_cibil
    "foir_headroom",               # bank.max_foir - lead.foir
    "income_headroom",             # lead.income - bank.min_income
    "income_headroom_ratio",       # income_headroom / bank.min_income
    "amount_fit_flag",             # loan_amount in [bank.min, bank.max]
    "amount_position",             # (amount - bank.min) / (bank.max - bank.min)
    "income_type_match",           # bool
    "loan_type_match",             # bool
    "geography_match",             # bool
    "bureau_fatigue_flag",         # enquiry_count > bank.max_enquiries
    "bureau_fatigue_excess",       # max(0, enquiry_count - bank.max_enquiries)
    "cibil_in_sweet_spot",         # bool: bank.pref_min <= cibil <= bank.pref_max
    "cibil_vs_sweet_spot_dist",    # distance from sweet spot center
    "age_maturity_headroom",       # bank.max_age_at_maturity - lead.age_at_maturity
    "dpd90_exceeds_bank_max",      # bool
    "written_off_exceeds_max",     # bool
]

TEMPORAL_FEATURES = [
    "application_sequence_num",    # 1st, 2nd, 3rd bank applied to
    "days_since_first_application",
    "enquiry_velocity_weekly",     # enquiries / weeks in market
    "is_reapplication",            # bool: applied after a prior rejection
]

ALL_FEATURES = LEAD_FEATURES + BANK_FEATURES + INTERACTION_FEATURES + TEMPORAL_FEATURES
TARGET = "converted"
GROUP_KEY = "lead_id"              # For ranking evaluation
```

### 5.2 Interaction Feature Computation

**File**: `src/features/interaction_features.py`

```python
def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must contain all lead and bank columns joined together.
    Returns df with new interaction columns appended.
    """
    df = df.copy()

    df['cibil_gap'] = df['cibil_score'] - df['min_cibil_score']
    df['foir_headroom'] = df['max_foir'] - df['foir']
    df['income_headroom'] = df['annual_income'] - df['min_annual_income']
    df['income_headroom_ratio'] = df['income_headroom'] / df['min_annual_income']

    df['amount_fit_flag'] = (
        (df['loan_amount_requested'] >= df['min_loan_amount']) &
        (df['loan_amount_requested'] <= df['max_loan_amount'])
    ).astype(int)

    df['amount_position'] = (
        (df['loan_amount_requested'] - df['min_loan_amount']) /
        (df['max_loan_amount'] - df['min_loan_amount'])
    ).clip(0, 1)

    df['bureau_fatigue_flag'] = (
        df['enquiry_count_6m'] > df['max_enquiries_6m']
    ).astype(int)
    df['bureau_fatigue_excess'] = (
        df['enquiry_count_6m'] - df['max_enquiries_6m']
    ).clip(lower=0)

    df['cibil_sweet_spot_center'] = (
        (df['preferred_cibil_min'] + df['preferred_cibil_max']) / 2
    )
    df['cibil_in_sweet_spot'] = (
        (df['cibil_score'] >= df['preferred_cibil_min']) &
        (df['cibil_score'] <= df['preferred_cibil_max'])
    ).astype(int)
    df['cibil_vs_sweet_spot_dist'] = (
        df['cibil_score'] - df['cibil_sweet_spot_center']
    ).abs()

    df['age_maturity_headroom'] = (
        df['max_age_at_maturity'] - df['age_at_maturity']
    )

    return df
```

**Pitfall**: Do not compute interaction features before joining lead and bank tables. Join first, then compute. Never store bank attributes on the lead table.

---

## 6. EDA & Statistical Analysis

**Files**: `src/eda/`, `notebooks/01_data_exploration.ipynb`

### 6.1 Required Analyses

```
A. Lead Distribution Analysis
   ├─ Univariate histograms for all continuous features
   ├─ Frequency bars for all categorical features
   ├─ Check skewness: income, savings (expect right-skew)
   └─ Outlier detection: Z-score on income, loan_amount

B. Bank Profile Analysis
   ├─ Distribution of min_cibil_score by bank_type
   ├─ Approval rate distribution across banks
   └─ Loan size range overlap visualization

C. Application-Level Analysis
   ├─ Conversion rate by bank_type
   ├─ Conversion rate by income_type × bank_type (heatmap)
   ├─ Rejection reason frequency (bar chart)
   ├─ Distribution of converted=1 by CIBIL decile
   └─ Applications per lead distribution

D. Correlation Analysis
   ├─ Spearman correlation heatmap of LEAD_FEATURES
   ├─ Spearman correlation of ALL_FEATURES vs. "converted"
   ├─ VIF analysis (Variance Inflation Factor) for multicollinearity
   └─ Key expected correlations to validate:
       - cibil_score ↑ → converted ↑ (positive)
       - enquiry_count_6m ↑ → converted ↓ (negative)
       - foir_headroom ↑ → converted ↑ (positive)
       - bureau_fatigue_flag → converted ↓ (strong negative)
       - income_type_match = True → converted ↑

E. Target Distribution
   ├─ Overall conversion rate (expect 12–18%)
   ├─ Per-lead: min/max/mean number of banks that convert
   └─ Histogram: how many leads have 0, 1, 2, 3+ converting banks
```

### 6.2 Automated Report

```python
# src/eda/report_generator.py
from ydata_profiling import ProfileReport

def generate_data_report(df: pd.DataFrame, output_path: str):
    profile = ProfileReport(df, title="Applications Feature Report", explorative=True)
    profile.to_file(output_path)
    # Also log key stats to MLflow
```

---

## 7. Data Validation & Leakage Prevention

**Files**: `src/validation/`

### 7.1 Schema Validation

**File**: `src/validation/schema_validator.py`

```python
import pandera as pa

LeadSchema = pa.DataFrameSchema({
    "cibil_score": pa.Column(int, pa.Check.in_range(300, 900)),
    "foir": pa.Column(float, pa.Check.in_range(0.05, 0.95)),
    "annual_income": pa.Column(float, pa.Check.greater_than(0)),
    "age": pa.Column(int, pa.Check.in_range(21, 65)),
    "enquiry_count_6m": pa.Column(int, pa.Check.in_range(0, 20)),
    "loan_to_income_ratio": pa.Column(float, pa.Check.greater_than(0)),
    "age_at_maturity": pa.Column(int, pa.Check.less_than(80)),
})

ApplicationSchema = pa.DataFrameSchema({
    "converted": pa.Column(int, pa.Check.isin([0, 1])),
    # Ineligible applications must have converted=0
    "converted": pa.Column(int, pa.Check(
        lambda s: not (s[df['eligibility_passed']==False] == 1).any()
    )),
})
```

### 7.2 Leakage Prevention

**File**: `src/validation/leakage_detector.py`

**Rules**:
```
FORBIDDEN features (future data — must NEVER appear in training):
  - rejection_reason        (only known after bank decision)
  - approved_amount         (only known after approval)
  - approved_rate           (only known after approval)
  - disbursed_amount        (is the target, not a feature)
  - bank_responded_at       (future relative to application)
  - disbursed_at            (future)
  - disbursal_failure_reason (future)
  - application_status      (IS the target, encoding)

ALLOWED features (available at application submission time):
  - All lead features (known before application)
  - All bank features (public knowledge)
  - All interaction features (computed from lead × bank)
  - application_sequence_num (known at submission)
  - effective_enquiry_count_at_submission (computed from bureau log)
```

```python
def check_for_leakage(feature_df: pd.DataFrame, target: str) -> None:
    FORBIDDEN = [
        "rejection_reason", "approved_amount", "approved_rate",
        "disbursed_amount", "application_status", "bank_responded_at",
        "disbursed_at", "disbursal_failure_reason"
    ]
    leaking = [f for f in FORBIDDEN if f in feature_df.columns]
    assert len(leaking) == 0, f"Leaking features found: {leaking}"

    # Correlation-based leakage check
    # Any feature with |correlation| > 0.95 with target is suspicious
    corrs = feature_df.corrwith(feature_df[target]).abs()
    suspicious = corrs[corrs > 0.95].drop(target)
    if len(suspicious) > 0:
        raise ValueError(f"Suspicious high-correlation features: {suspicious.index.tolist()}")
```

### 7.3 Distribution Sanity Checks

```python
def validate_simulation_realism(df: pd.DataFrame) -> None:
    overall_conversion = df['converted'].mean()
    assert 0.10 <= overall_conversion <= 0.22, f"Conversion rate {overall_conversion:.3f} out of expected range"

    # Bank-level approval rate variance (banks must behave differently)
    per_bank = df.groupby('bank_id')['converted'].mean()
    assert per_bank.std() > 0.05, "Banks too similar — insufficient differentiation"

    # Causal correlation checks
    assert df['cibil_score'].corr(df['annual_income']) > 0.30, "Income-CIBIL correlation too low"
    assert df['cibil_score'].corr(df['dpd_30_count']) < -0.20, "CIBIL-DPD correlation direction wrong"
    assert df['foir_headroom'].corr(df['converted']) > 0.10, "FOIR headroom not predictive"
    assert df['bureau_fatigue_flag'].corr(df['converted']) < -0.05, "Bureau fatigue not penalizing"
```

---

## 8. Preprocessing Pipeline

**Files**: `src/preprocessing/`

### 8.1 Encoding Strategy

```python
# configs/feature_config.yaml

ordinal_features:
  risk_appetite: [conservative, moderate, aggressive]
  documentation_strictness: [low, medium, high]
  city_tier: [1, 2, 3]

onehot_features:
  - income_type
  - employer_category
  - loan_type
  - bank_type
  - gender

passthrough_features:
  # All numeric features pass through as-is after scaling
  - cibil_score, annual_income, foir, dti_ratio, ...

log_transform_features:
  # Right-skewed — apply log1p before scaling
  - annual_income
  - loan_amount_requested
  - savings_balance
  - credit_card_spend_monthly
  - fixed_deposits
  - monthly_obligations
```

### 8.2 Pipeline Assembly

**File**: `src/preprocessing/pipeline_builder.py`

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def build_preprocessing_pipeline(feature_config: dict) -> ColumnTransformer:
    log_transformer = FunctionTransformer(np.log1p, validate=True)

    preprocessor = ColumnTransformer(transformers=[
        ("log_scale", Pipeline([
            ("log", log_transformer),
            ("scale", StandardScaler())
        ]), feature_config["log_transform_features"]),

        ("scale", StandardScaler(), feature_config["scale_features"]),

        ("ordinal", OrdinalEncoder(
            categories=[feature_config["ordinal_categories"][f]
                        for f in feature_config["ordinal_features"]]
        ), feature_config["ordinal_features"]),

        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         feature_config["onehot_features"]),

        ("passthrough", "passthrough", feature_config["bool_features"]),
    ])
    return preprocessor
```

**Pitfall**: Fit the preprocessor **only on training data**. Apply (transform) to val and test. Never fit on the full dataset before splitting.

---

## 9. Train / Validation / Test Strategy

### 9.1 Split Logic

**File**: `src/modeling/trainer.py`

The split must be **lead-based** (not row-based) to prevent data leakage across the lead group boundary:

```python
def split_by_lead(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Split at the LEAD level, not the row level.
    All applications from a lead go into the same split.
    This prevents: training on Bank_A's decision for lead_X and
    testing on Bank_B's decision for the same lead_X.
    """
    unique_leads = df['lead_id'].unique()
    n = len(unique_leads)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_leads)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_leads = unique_leads[:n_test]
    val_leads = unique_leads[n_test:n_test + n_val]
    train_leads = unique_leads[n_test + n_val:]

    train = df[df['lead_id'].isin(train_leads)]
    val = df[df['lead_id'].isin(val_leads)]
    test = df[df['lead_id'].isin(test_leads)]

    return train, val, test
```

**Why lead-level splits matter**: If lead_X appears in both train and test (different banks), the model can memorize lead-level patterns that don't generalize. This is the ranking equivalent of target leakage.

### 9.2 Cross-Validation

Use `GroupKFold` with `groups = lead_id` for CV during hyperparameter tuning:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=lead_ids)):
    ...
```

### 9.3 Class Imbalance

```python
# XGBoost: use scale_pos_weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos   # typically 5–8x

# LightGBM: use is_unbalance=True or class_weight
# Do NOT use SMOTE — it disrupts pairwise structure of the data
```

---

## 10. Model Training & Ranking Pipeline

### 10.1 Stage 1: Eligibility Engine

**File**: `src/eligibility/rule_engine.py`

```python
def check_eligibility(lead: dict, bank: dict) -> tuple[bool, str | None]:
    """
    Returns (eligible: bool, failure_reason: str | None).
    Order matters: check cheapest/most common rejections first.
    """
    if lead['income_type'] not in bank['accepted_income_types']:
        return False, "income_type_mismatch"
    if lead['state'] not in bank['states_covered']:
        return False, "geography_not_covered"
    if lead['cibil_score'] < bank['min_cibil_score']:
        return False, "low_cibil"
    if lead['cibil_score'] > bank.get('max_cibil_score', 900):
        return False, "cibil_too_high"
    if lead['annual_income'] < bank['min_annual_income']:
        return False, "income_insufficient"
    if lead['annual_income'] > bank.get('max_annual_income', float('inf')):
        return False, "income_too_high"
    if lead['foir'] > bank['max_foir']:
        return False, "high_foir"
    if lead['age_at_maturity'] > bank['max_age_at_maturity']:
        return False, "age_at_maturity_exceeded"
    if lead['enquiry_count_6m'] > bank['max_enquiries_6m']:
        return False, "bureau_fatigue"
    if lead['dpd_90_count'] > bank['max_dpd_90_count']:
        return False, "delinquency_history"
    if lead['written_off_loans'] > bank['max_written_off_loans']:
        return False, "delinquency_history"
    if lead['loan_type'] not in bank['loan_types_offered']:
        return False, "loan_type_not_offered"
    if not (bank['min_loan_amount'] <= lead['loan_amount_requested'] <= bank['max_loan_amount']):
        return False, "loan_amount_out_of_range"
    return True, None
```

### 10.2 Stage 2: Pointwise Scorer (XGBoost)

**File**: `src/modeling/trainer.py`

```python
import xgboost as xgb
from sklearn.pipeline import Pipeline

def train_pointwise_model(
    X_train, y_train, X_val, y_val,
    preprocessing_pipeline, params: dict
) -> Pipeline:

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=["auc", "logloss"],
        tree_method="hist",
        scale_pos_weight=params["scale_pos_weight"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        early_stopping_rounds=50,
        random_state=42
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessing_pipeline),
        ("model", model)
    ])

    X_train_prep = preprocessing_pipeline.fit_transform(X_train)
    X_val_prep = preprocessing_pipeline.transform(X_val)

    model.fit(
        X_train_prep, y_train,
        eval_set=[(X_val_prep, y_val)],
        verbose=50
    )

    return full_pipeline
```

### 10.3 Stage 3: Ranking

**File**: `src/modeling/ranker.py`

```python
def rank_banks_for_lead(
    lead_id: str,
    candidate_banks: list[str],
    model,
    feature_df: pd.DataFrame,
    top_k: int = 5
) -> list[dict]:
    """
    Given a lead and a list of eligible bank IDs (post-Stage 1),
    return a ranked list of banks with scores.
    """
    pairs = feature_df[
        (feature_df['lead_id'] == lead_id) &
        (feature_df['bank_id'].isin(candidate_banks))
    ]

    X = pairs[ALL_FEATURES]
    scores = model.predict_proba(X)[:, 1]

    pairs = pairs.copy()
    pairs['rank_score'] = scores
    ranked = pairs.sort_values('rank_score', ascending=False).head(top_k)

    return ranked[['bank_id', 'bank_name', 'rank_score']].to_dict('records')
```

### 10.4 Optional: LightGBM LambdaMART (Upgrade Path)

```python
import lightgbm as lgb

def train_lambdamart(X_train, y_train, groups_train, X_val, y_val, groups_val):
    """
    groups = number of applications per lead (for LambdaMART's group structure).
    """
    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 5],
        "label_gain": [0, 1],   # binary relevance
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "verbose": -1
    }

    model = lgb.train(
        params, train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    return model
```

---

## 11. Evaluation & Metrics

**File**: `src/modeling/evaluator.py`

### 11.1 Point-Level Metrics (per application)
```python
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def evaluate_pointwise(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "auc_roc":   roc_auc_score(y_true, y_pred_proba),
        "f1":        f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
    }
```

### 11.2 Ranking Metrics (per lead group) — Primary Metrics

```python
from sklearn.metrics import ndcg_score

def evaluate_ranking(df: pd.DataFrame, score_col: str, target_col: str,
                      k_values=[1, 3, 5]) -> dict:
    """
    Evaluate ranking quality across all leads.
    For each lead, compute NDCG@K and Recall@K, then average.
    """
    results = {}
    leads = df['lead_id'].unique()

    ndcg_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    mrr_scores = []

    for lead_id in leads:
        group = df[df['lead_id'] == lead_id].copy()
        if group[target_col].sum() == 0:
            continue  # Skip leads with no positive outcome

        y_true = group[target_col].values
        y_score = group[score_col].values

        for k in k_values:
            # NDCG@K
            ndcg_k = ndcg_score([y_true], [y_score], k=k)
            ndcg_scores[k].append(ndcg_k)

            # Recall@K: was at least one positive in top-K?
            top_k_idx = np.argsort(-y_score)[:k]
            recall_k = int(y_true[top_k_idx].any())
            recall_scores[k].append(recall_k)

        # MRR: reciprocal rank of first relevant result
        ranked_idx = np.argsort(-y_score)
        for rank, idx in enumerate(ranked_idx, 1):
            if y_true[idx] == 1:
                mrr_scores.append(1.0 / rank)
                break

    for k in k_values:
        results[f"ndcg@{k}"] = np.mean(ndcg_scores[k])
        results[f"recall@{k}"] = np.mean(recall_scores[k])

    results["mrr"] = np.mean(mrr_scores)
    return results
```

### 11.3 Minimum Acceptable Performance Thresholds

```
AUC-ROC:     ≥ 0.82
NDCG@3:      ≥ 0.70
Recall@3:    ≥ 0.75  (correct bank in top-3 for 75%+ of leads)
MRR:         ≥ 0.60
F1 (class 1): ≥ 0.65
```

---

## 12. Hyperparameter Tuning

**File**: `src/modeling/tuner.py`
**Framework**: Optuna

```python
import optuna
import mlflow

def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val, lead_ids_val):
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma":            trial.suggest_float("gamma", 0, 2.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0, 2.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.0, 10.0),
    }

    model = train_pointwise_model(X_train, y_train, X_val, y_val, params)
    scores = model.predict_proba(X_val)[:, 1]

    val_df = pd.DataFrame({
        'lead_id': lead_ids_val,
        'score': scores,
        'converted': y_val
    })

    metrics = evaluate_ranking(val_df, score_col='score', target_col='converted')
    return metrics['ndcg@3']   # Optimize for NDCG@3, not AUC


def run_tuning(n_trials=100, ...):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_params
```

**Note**: Optimize for `NDCG@3`, not AUC. AUC is a pointwise metric; NDCG@3 is the actual business objective.

---

## 13. Experiment Tracking

**Framework**: MLflow (local store at `experiments/mlflow/`)
**File**: `src/modeling/trainer.py` (integrated)

```python
import mlflow
import mlflow.xgboost

mlflow.set_experiment("lead_bank_ranking_v1")

with mlflow.start_run(run_name=f"xgb_{timestamp}"):
    # Log parameters
    mlflow.log_params(params)
    mlflow.log_param("n_leads", n_leads)
    mlflow.log_param("n_banks", n_banks)
    mlflow.log_param("feature_count", len(ALL_FEATURES))

    # Train model
    model = train_pointwise_model(...)

    # Log metrics
    mlflow.log_metrics({
        "val_auc": val_metrics["auc_roc"],
        "val_ndcg_3": ranking_metrics["ndcg@3"],
        "val_recall_3": ranking_metrics["recall@3"],
        "val_mrr": ranking_metrics["mrr"],
        "val_f1": val_metrics["f1"],
    })

    # Log artifacts
    mlflow.log_artifact("data/artifacts/feature_schema.json")
    mlflow.log_artifact("data/artifacts/data_report.html")
    mlflow.xgboost.log_model(model, "model")
    mlflow.log_figure(feature_importance_fig, "feature_importance.png")
```

**Experiment naming convention**: `{model_type}_{feature_group}_{date}_{experiment_id}`
- Example: `xgb_all_features_20250601_001`

---

## 14. Error Analysis

**File**: `notebooks/04_error_analysis.ipynb`
**File**: `src/modeling/evaluator.py`

### Analysis Checklist

```python
# 1. False Negatives (FN) — disbursed but ranked low
fn_df = test_df[
    (test_df['converted'] == 1) &
    (test_df['rank_position'] > 3)  # rank_position: bank's position in ranked list
]
# Analyze: what's different about these leads/banks?
# Expected: low cibil_gap, borderline foir_headroom, high enquiry_count

# 2. False Positives (FP) — ranked high but not disbursed
fp_df = test_df[
    (test_df['converted'] == 0) &
    (test_df['rank_position'] <= 3)
]
# Expected: eligibility passed but rejected on soft criteria

# 3. Per-bank error analysis
for bank_id in test_df['bank_id'].unique():
    bank_results = test_df[test_df['bank_id'] == bank_id]
    bank_auc = roc_auc_score(bank_results['converted'], bank_results['score'])
    # Flag banks where model performs < 0.70 AUC

# 4. Per-income-type analysis
for income_type in ["salaried", "self_employed", "business", "freelance"]:
    subset = test_df[test_df['income_type'] == income_type]
    print(income_type, evaluate_ranking(subset, ...))

# 5. Feature importance analysis
importance_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': model['model'].feature_importances_
}).sort_values('importance', ascending=False)

# Top features should include: cibil_gap, foir_headroom, bureau_fatigue_flag,
# income_type_match, amount_fit_flag
```

---

## 15. Logging & Monitoring

**File**: `src/` (module-level logging in every module)
**Library**: Python `logging` + `structlog` for structured output

### Logger Setup

```python
# src/__init__.py
import logging
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()
```

### Logging Points (mandatory)

```
Data Generation:
  - [INFO]  "Starting lead generation" {n_leads: N, seed: S}
  - [INFO]  "Lead generation complete" {actual_count: N, time_seconds: T}
  - [WARN]  "CIBIL-income correlation low" {correlation: 0.XX}  # if < 0.30
  - [INFO]  "Application generation complete" {total_pairs: N, conversion_rate: X.XX}
  - [ERROR] "Conversion rate out of expected range" {actual: X, expected: [0.10, 0.22]}

Feature Engineering:
  - [INFO]  "Feature engineering complete" {feature_count: N, null_count: N}
  - [WARN]  "High-correlation feature pair detected" {f1: X, f2: Y, corr: Z}

Training:
  - [INFO]  "Training started" {model: XGB, params: {...}}
  - [INFO]  "Epoch {N}: val_auc={X}, val_ndcg3={Y}"
  - [INFO]  "Training complete" {best_iteration: N, test_ndcg3: X}
  - [ERROR] "NDCG@3 below threshold" {actual: X, threshold: 0.70}

API (production):
  - [INFO]  "Ranking request" {lead_id: X, n_eligible_banks: N, latency_ms: T}
  - [WARN]  "No eligible banks for lead" {lead_id: X}
  - [ERROR] "Model inference failed" {lead_id: X, error: E}
```

---

## 16. Testing Strategy

### 16.1 Unit Tests

**Directory**: `tests/unit/`

```python
# test_lead_generator.py
def test_cibil_income_correlation():
    leads = generate_leads(n=5000, seed=42)
    corr = leads['cibil_score'].corr(leads['annual_income'])
    assert corr > 0.30, f"Expected correlation > 0.30, got {corr:.3f}"

def test_foir_bounds():
    leads = generate_leads(n=1000, seed=42)
    assert (leads['foir'] > 0.05).all()
    assert (leads['foir'] < 0.95).all()

def test_no_nulls():
    leads = generate_leads(n=100, seed=42)
    assert leads.isnull().sum().sum() == 0

# test_approval_simulator.py
def test_ineligible_always_returns_zero():
    lead = make_test_lead(income_type="freelance")
    bank = make_test_bank(accepted_income_types=["salaried"])
    prob = compute_approval_probability(lead, bank)
    assert prob == 0.0

def test_perfect_lead_high_probability():
    lead = make_test_lead(cibil_score=800, foir=0.30, enquiry_count_6m=0)
    bank = make_test_bank(min_cibil_score=650, max_foir=0.70, max_enquiries_6m=5)
    prob = compute_approval_probability(lead, bank)
    assert prob > 0.70, f"Expected high probability, got {prob:.3f}"

# test_interaction_features.py
def test_cibil_gap_computation():
    df = pd.DataFrame({'cibil_score': [720], 'min_cibil_score': [650]})
    result = compute_interaction_features(df)
    assert result['cibil_gap'].iloc[0] == 70

def test_bureau_fatigue_flag():
    df = pd.DataFrame({'enquiry_count_6m': [5], 'max_enquiries_6m': [3]})
    result = compute_interaction_features(df)
    assert result['bureau_fatigue_flag'].iloc[0] == 1
    assert result['bureau_fatigue_excess'].iloc[0] == 2

# test_eligibility_engine.py
def test_geography_rejection():
    lead = make_test_lead(state="KL")
    bank = make_test_bank(states_covered=["MH", "GJ", "DL"])
    eligible, reason = check_eligibility(lead, bank)
    assert not eligible
    assert reason == "geography_not_covered"

def test_eligible_lead_passes_all_rules():
    lead = make_perfect_lead()
    bank = make_permissive_bank()
    eligible, reason = check_eligibility(lead, bank)
    assert eligible
    assert reason is None
```

### 16.2 Integration Tests

```python
# test_full_simulation_pipeline.py
def test_full_pipeline_produces_valid_dataset():
    leads = generate_leads(n=500, seed=42)
    banks = generate_banks(seed=42)
    apps = generate_applications(leads, banks)
    features = compute_all_features(apps)

    # Schema check
    assert 'converted' in features.columns
    assert 'cibil_gap' in features.columns
    assert features['converted'].mean() < 0.30

    # No leakage
    check_for_leakage(features, 'converted')

    # Splits maintain lead boundary
    train, val, test = split_by_lead(features)
    train_leads = set(train['lead_id'])
    test_leads = set(test['lead_id'])
    assert len(train_leads & test_leads) == 0

# test_api_endpoint.py
def test_ranking_endpoint_returns_valid_response():
    response = client.post("/rank", json={
        "lead_id": "test-lead-001",
        "age": 35, "annual_income": 800000, "cibil_score": 720, ...
    })
    assert response.status_code == 200
    data = response.json()
    assert "ranked_banks" in data
    assert len(data["ranked_banks"]) > 0
    assert all("bank_id" in b and "score" in b for b in data["ranked_banks"])
```

### 16.3 Test Execution

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run with markers
pytest tests/ -m "not slow"
```

---

## 17. Productionization

### 17.1 FastAPI Inference Endpoint

**File**: `src/api/main.py`

```python
from fastapi import FastAPI, HTTPException
from .schemas import LeadRequest, RankingResponse
from .predictor import RankingPredictor

app = FastAPI(title="Lead-Bank Ranking API", version="1.0.0")
predictor = RankingPredictor.load_from_registry("models/v1/")

@app.post("/rank", response_model=RankingResponse)
async def rank_banks(request: LeadRequest):
    try:
        # Stage 1: Eligibility
        eligible_banks = predictor.get_eligible_banks(request.dict())
        if not eligible_banks:
            return RankingResponse(lead_id=request.lead_id, ranked_banks=[],
                                   message="No eligible banks found")

        # Stage 2 + 3: Score + Rank
        ranked = predictor.rank(request.dict(), eligible_banks)
        return RankingResponse(lead_id=request.lead_id, ranked_banks=ranked)

    except Exception as e:
        logger.error("ranking_failed", lead_id=request.lead_id, error=str(e))
        raise HTTPException(status_code=500, detail="Ranking failed")

@app.get("/health")
async def health():
    return {"status": "ok", "model_version": predictor.version}
```

### 17.2 Model Artifact Structure

```
models/
└── v1/
    ├── metadata.json           # version, feature list, thresholds, training date
    ├── xgb_model.ubj           # XGBoost binary model
    ├── preprocessor.pkl        # Fitted sklearn preprocessor
    ├── eligibility_rules.json  # Bank eligibility rules (serialized)
    └── feature_schema.json     # Feature names and expected types
```

### 17.3 Request/Response Schema

```python
# src/api/schemas.py
from pydantic import BaseModel, Field

class LeadRequest(BaseModel):
    lead_id: str
    age: int = Field(ge=21, le=65)
    annual_income: float = Field(gt=0)
    cibil_score: int = Field(ge=300, le=900)
    foir: float = Field(ge=0.0, le=1.0)
    income_type: str
    loan_type: str
    loan_amount_requested: float = Field(gt=0)
    loan_tenure_months: int
    enquiry_count_6m: int = Field(ge=0)
    dpd_30_count: int = Field(ge=0)
    dpd_90_count: int = Field(ge=0)
    written_off_loans: int = Field(ge=0)
    city_tier: int = Field(ge=1, le=3)
    state: str
    # ... all required lead features

class RankedBank(BaseModel):
    bank_id: str
    bank_name: str
    bank_type: str
    rank_score: float
    interest_rate_min: float
    disbursal_speed_days: int

class RankingResponse(BaseModel):
    lead_id: str
    ranked_banks: list[RankedBank]
    eligible_bank_count: int
    message: str = "OK"
```

---

## 18. Scalability & Future Improvements

### Phase 2 Improvements

```
1. Upgrade ML framing
   ├─ LambdaMART (LightGBM rank:ndcg) for true listwise ranking
   └─ Calibrated probability outputs (Platt scaling / isotonic regression)

2. Real data integration
   ├─ Bureau API integration (CIBIL, Experian) — replace synthetic CIBIL
   ├─ Bank policy API — replace static bank rules with real-time feeds
   └─ Feedback loop — capture real disbursal outcomes → retrain monthly

3. Cold-start handling
   ├─ New bank: use archetype defaults until 500+ applications
   └─ New lead profile type: fallback to conservative ranking

4. Temporal modeling
   ├─ Track CIBIL score changes over time
   ├─ Model bureau fatigue with actual timestamps (not static count)
   └─ Session-level features: how long has lead been in market?

5. Multi-objective ranking
   ├─ Pareto-rank on (P(disbursal), interest_rate, disbursal_speed)
   └─ User preference weights (speed vs. rate vs. approval prob)

6. Infrastructure
   ├─ Feature store (Feast / Hopsworks) for real-time feature serving
   ├─ Model serving: Triton or BentoML for GPU inference
   └─ A/B testing framework: route % of traffic to v2 model
```

### Performance Targets at Scale

```
Dataset:     10M leads × 50 banks = 500M application pairs
Generation:  ~2 hours on 32-core machine (parallelized)
Training:    ~45 min with LightGBM on 400M rows (GPU)
Inference:   < 50ms per lead (100 bank candidates, batch score)
API:         2000 RPS, p99 < 100ms
```

---

## Quick Start Commands

```bash
# 1. Setup
poetry install
cp .env.example .env

# 2. Generate synthetic data
python -m src.simulation.lead_generator --config configs/data_config.yaml
python -m src.simulation.bank_generator --config configs/bank_archetypes.yaml
python -m src.simulation.application_generator --config configs/data_config.yaml

# 3. Validate data
python -m src.validation.schema_validator
python -m src.validation.leakage_detector
python -m src.validation.distribution_checks

# 4. Feature engineering
python -m src.features.interaction_features

# 5. EDA report
python -m src.eda.report_generator

# 6. Train model
python -m src.modeling.trainer --config configs/model_config.yaml

# 7. Tune hyperparameters
python -m src.modeling.tuner --n_trials 100

# 8. Evaluate
python -m src.modeling.evaluator --split test

# 9. Start API
uvicorn src.api.main:app --reload --port 8000

# 10. Run tests
pytest tests/ -v --cov=src
```

---

*Last updated: See git log. All implementation decisions are final unless overridden by a new CLAUDE.md revision tagged as `[UPDATED]`.*

---

## Agent Instructions (Reusable — Apply to All Future Sessions)

- Review all project documentation before making changes.
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
