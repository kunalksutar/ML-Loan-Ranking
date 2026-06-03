# Lead-to-Bank Ranking System: Full Design & Architecture Guide

> **Scope**: Schema design, synthetic data generation, ML framing, feature engineering, and production considerations for a many-leads-to-many-banks loan ranking system.

---

## Part 1: Notebook Analysis — What's Useful and What Isn't

### ✅ Directly Reusable

| Notebook Component | Relevance to Your Project |
|---|---|
| **Imbalanced dataset handling** | Loan approvals are rare events (~8–15% in real markets). Use `scale_pos_weight` in XGBoost, stratified splits. |
| **Stratified train/test split** | Critical — your positive class (approved + disbursed) will be sparse. |
| **Model zoo** (RF, XGBoost, GBM) | Tree-based ensemble models are the right family for tabular ranking. |
| **F1-score as primary metric** | Correct framing for imbalanced targets; for ranking, extend to NDCG. |
| **Drop-column feature importance** | Use this to validate synthetic feature-target correlations. |
| **Hyperparameter grid patterns** | XGBoost grids (`scale_pos_weight`, `min_child_weight`, `gamma`) directly apply. |
| **Spearman correlation analysis** | Use for detecting feature collinearity in your synthetic data. |

### ❌ Not Applicable / Needs Adaptation

| Notebook Assumption | Why It Breaks for You |
|---|---|
| **One bank, one loan type** | You have N banks × M leads — the problem is pairwise. |
| **Static customer features only** | You need bank-lead *interaction* features (compatibility scores, eligibility flags). |
| **Binary classification** | Your target is a ranked list per lead, not a binary outcome per customer. |
| **No eligibility pre-filtering** | Real banks hard-reject before scoring; skip this and you'll have impossible positives. |
| **No temporal dimension** | Real lending has application sequences, bureau pull timestamps, and rate windows. |
| **ZIP Code as a feature** | You need geography as a bank coverage/branch variable, not a lead variable. |

---

## Part 2: Problem Framing — Classification vs. Ranking vs. Recommendation

### The Right Answer: **Learning-to-Rank (LTR)**

Your problem is not a single classification — it is generating a **ranked list of banks per lead** that maximizes conversion probability. Here's how the three framings compare:

```
Classification:   P(approved | lead) → one probability per lead, ignores banks
Recommendation:   "Banks like this lead" → collaborative filtering, needs history
Learning-to-Rank: P(approved | lead, bank) → pairwise score, rank banks per lead ✅
```

### Recommended Hybrid Architecture

```
Stage 1 — Eligibility Filter (Rule-Based)
    ├─ Hard rules: CIBIL cutoff, income minimum, age range
    └─ Output: candidate bank set per lead (reduces N banks to K eligible)

Stage 2 — Pointwise Scoring (XGBoost / LightGBM)
    ├─ Input: (lead_features + bank_features + interaction_features)
    ├─ Target: P(approval AND disbursal)
    └─ Output: probability score per lead-bank pair

Stage 3 — Ranking Layer (optional LambdaMART / LightGBM with rank objective)
    ├─ Input: scored candidates from Stage 2
    └─ Output: ordered list of banks per lead

Stage 4 — Business Rules Override
    ├─ Boost: partner banks, preferred loan types
    └─ Cap: cooling-off periods, bureau pull limits
```

**Why pointwise scoring works well here**: You likely don't have listwise training signal initially. Pointwise XGBoost trained on (lead, bank, outcome) triplets gives you a probability you can directly rank by. Upgrade to LambdaMART once you have enough conversion logs.

---

## Part 3: Improved Schema Design

### 3.1 Entity: Lead (Loan Applicant)

```python
Lead:
  # Identifiers
  lead_id: UUID
  created_at: datetime
  source_channel: Enum["organic", "partner", "referral", "ad_campaign", "aggregator"]

  # Demographics
  age: int                          # 21–65 (lending age range)
  gender: Enum["M", "F", "Other"]   # Some banks have gender-based products
  city_tier: Enum[1, 2, 3]          # Tier 1 = metro, Tier 3 = rural
  state: str                        # Regulatory differences by state
  pin_code: str                     # Bank branch coverage proxy

  # Financial Profile
  annual_income: float              # Gross, in INR
  income_type: Enum["salaried", "self_employed", "business", "freelance"]
  employer_category: Enum["PSU", "private_listed", "private_unlisted", "MNC", "govt"]
  monthly_obligations: float        # Total existing EMIs
  existing_loan_count: int          # Number of active loans
  credit_card_spend_monthly: float  # Avg monthly CC spend
  savings_balance: float            # Average quarterly balance
  fixed_deposits: float             # Total FD value

  # Credit Profile
  cibil_score: int                  # 300–900
  dpd_30_count: int                 # Days-past-due events in last 12m
  dpd_90_count: int                 # Severe delinquencies
  enquiry_count_6m: int             # Hard pulls in last 6 months (bureau fatigue)
  settled_loans: int                # Count of settled (bad signal)
  written_off_loans: int            # Count of write-offs (very bad signal)

  # Employment
  work_experience_years: float
  current_employer_tenure_years: float
  is_currently_employed: bool

  # Loan Request
  loan_type: Enum["personal", "home", "car", "education", "business", "gold", "lap"]
  loan_amount_requested: float
  loan_tenure_months: int           # Requested duration
  purpose: str                      # Self-declared purpose

  # Derived / Engineered (computed at generation time)
  dti_ratio: float                  # monthly_obligations / (annual_income/12)
  foir: float                       # Fixed Obligations to Income Ratio
  loan_to_income_ratio: float       # loan_amount / annual_income
  credit_utilization: float         # CC spend / CC limit estimate
```

**Why these additions matter:**
- `enquiry_count_6m` is one of the strongest rejection signals — too many hard pulls kills eligibility
- `dpd_30_count / dpd_90_count` distinguish mild from severe delinquency; banks weight these very differently
- `dti_ratio` / `foir` are the actual underwriting ratios banks compute — generating raw income without these leads to unrealistic approval patterns
- `income_type` dramatically changes which banks will consider a lead (PSBs love salaried; NBFCs serve self-employed)
- `city_tier` proxies branch coverage and product availability

---

### 3.2 Entity: Bank (Lender Profile)

```python
Bank:
  # Identifiers
  bank_id: UUID
  name: str
  bank_type: Enum["PSB", "private", "NBFC", "cooperative", "fintech", "microfinance", "HFC"]

  # Operational Scope
  states_covered: List[str]          # Geographic coverage
  city_tiers_served: List[int]       # [1], [1,2], [1,2,3]
  digital_only: bool                 # No branch requirement for application

  # Loan Products Offered
  loan_types_offered: List[str]

  # Eligibility Criteria (per loan type, but simplified here)
  min_cibil_score: int               # Hard floor, e.g. 650, 700, 720
  min_annual_income: float
  max_dti_ratio: float               # E.g. 0.50 means max 50% DTI
  max_foir: float                    # E.g. 0.65
  min_age: int
  max_age: int
  max_loan_to_income_ratio: float
  max_enquiries_6m: int              # E.g., reject if >4 pulls in 6m
  accepted_income_types: List[str]   # E.g., PSBs exclude freelance
  accepted_employer_categories: List[str]
  min_employer_tenure_months: int
  min_work_experience_years: float

  # Loan Terms (per bank's typical offering)
  interest_rate_min: float           # APR range
  interest_rate_max: float
  processing_fee_pct: float
  max_loan_amount: float
  min_loan_amount: float
  max_tenure_months: int

  # Behavioral Profile (for simulation)
  risk_appetite: Enum["conservative", "moderate", "aggressive"]
  approval_base_rate: float          # Calibrated base approval rate (0.05–0.35)
  disbursal_speed_days: int          # Days from approval to disbursal
  documentation_strictness: Enum["low", "medium", "high"]

  # Business Preferences (hidden behavior)
  preferred_loan_size_segment: Enum["small", "mid", "large"]
  preferred_cibil_band: Tuple[int, int]   # E.g. (700, 780) = sweet spot
  sensitivity_to_employer_type: float     # How much employer type shifts approval
```

**Key additions explained:**
- `risk_appetite` drives the approval probability simulation — aggressive banks (NBFCs, fintechs) approve lower CIBIL; conservative banks (PSBs) don't
- `max_enquiries_6m` is a real hard rule — many banks auto-reject leads with bureau fatigue
- `preferred_cibil_band` captures that banks don't just want ≥700; they have a sweet spot where they're most competitive
- `documentation_strictness` affects disbursal success independent of approval

---

### 3.3 Entity: Application (Lead-Bank Interaction)

This is the core fact table — every row is one lead submitted to one bank.

```python
Application:
  application_id: UUID
  lead_id: UUID (FK → Lead)
  bank_id: UUID (FK → Bank)

  # Timing
  submitted_at: datetime
  bank_responded_at: datetime
  disbursed_at: datetime (nullable)

  # Outcome (your target variables)
  eligibility_passed: bool          # Stage 1: did lead pass hard rules?
  application_status: Enum[
    "submitted",
    "under_review",
    "approved",
    "rejected",
    "withdrawn",
    "disbursed",
    "disbursal_failed"
  ]

  # If Rejected
  rejection_reason: Enum[
    "low_cibil",
    "high_dti",
    "bureau_fatigue",
    "income_insufficient",
    "employer_category_mismatch",
    "geography_not_covered",
    "loan_amount_out_of_range",
    "delinquency_history",
    "tenure_mismatch",
    "incomplete_docs",
    "policy_decline",
    "other"
  ]

  # If Approved
  approved_amount: float (nullable)
  approved_rate: float (nullable)
  approved_tenure_months: int (nullable)

  # If Disbursed
  disbursed_amount: float (nullable)
  disbursal_failure_reason: Enum["doc_failure", "lead_withdrew", "bank_cancelled", null]

  # Interaction Features (computed at submission time)
  cibil_vs_bank_min: float           # lead.cibil - bank.min_cibil (gap feature)
  dti_vs_bank_max: float             # bank.max_dti - lead.dti (headroom feature)
  loan_amount_fit_score: float       # How well requested amount fits bank's range
  income_type_match: bool
  geography_match: bool
```

---

### 3.4 Entity: Bureau Pull Log (Temporal History)

```python
BureauPull:
  pull_id: UUID
  lead_id: UUID
  bank_id: UUID
  pulled_at: datetime
  cibil_score_at_pull: int          # Score may change between pulls
  enquiry_type: Enum["soft", "hard"]
```

This enables you to simulate **bureau fatigue** — a critical real-world effect where a lead who gets pulled by 5 banks in 2 weeks becomes radioactive to the 6th bank.

---

### 3.5 Revised Schema Relationships

```
Lead ──────────────────────────────────────┐
  │                                        │
  │ 1:N                                    │ 1:N
  ▼                                        ▼
Application ◄──────── Bank          BureauPull
  │
  │ (contains outcome)
  ▼
application_status + rejection_reason + disbursed_amount
```

**Removed from your original schema:**
- `LeadSentToBank` table — merged into `Application` (it was doing the same job)
- `LeadStatus` table — the status belongs on the Application, not the Lead

---

## Part 4: Target Variable Design

### What to predict

For ranking, your target should be a **composite conversion signal**, not just approval:

```python
# Option A: Binary (simplest, good starting point)
converted = (application_status == "disbursed")  # 1 = full success

# Option B: Ordinal (captures partial value)
outcome_score = {
  "rejected_eligibility": 0,
  "policy_decline": 0,
  "rejected_underwriting": 1,
  "approved_not_disbursed": 2,
  "disbursed": 3
}

# Option C: Probability (best for ranking)
# Train to predict P(disbursed | lead, bank)
# Use this probability to rank banks per lead
```

**Recommendation**: Start with **Option A** (binary, disbursed vs. not). It's cleanest for XGBoost and avoids ordinal encoding complexity. Upgrade to a calibrated probability model later.

---

## Part 5: Synthetic Data Generation Strategy

### 5.1 Architecture of the Simulator

```python
class LoanMarketSimulator:
    def __init__(self, n_leads, n_banks, n_applications_per_lead):
        self.leads = self.generate_leads(n_leads)
        self.banks = self.generate_banks(n_banks)
        self.applications = self.simulate_applications()

    def generate_leads(self, n):
        # Step 1: Sample demographics
        # Step 2: Derive financial profile from demographics
        # Step 3: Derive credit profile from financial history
        # Step 4: Compute derived ratios (DTI, FOIR, etc.)
        # Step 5: Generate loan request consistent with profile
        pass

    def generate_banks(self, n):
        # Assign bank_type first
        # Derive eligibility rules from bank_type
        # Assign behavioral parameters
        pass

    def simulate_applications(self):
        # For each lead: determine which banks to apply to
        # For each lead-bank pair: run eligibility check
        # For eligible pairs: simulate approval decision
        # For approved pairs: simulate disbursal decision
        pass
```

---

### 5.2 Realistic Lead Generation

The key principle: **derive attributes in causal order**, not independently.

```python
import numpy as np
from faker import Faker

fake = Faker('en_IN')

def generate_lead():
    # Step 1: Anchor demographics
    age = int(np.random.normal(38, 10).clip(23, 62))
    income_type = np.random.choice(
        ["salaried", "self_employed", "business", "freelance"],
        p=[0.55, 0.25, 0.15, 0.05]
    )

    # Step 2: Income depends on age + income_type
    base_income = {
        "salaried": np.random.lognormal(13.0, 0.6),   # ~440K median
        "self_employed": np.random.lognormal(12.8, 0.8),
        "business": np.random.lognormal(13.5, 1.0),
        "freelance": np.random.lognormal(12.5, 0.7),
    }[income_type]
    # Income grows with age (career progression)
    income_multiplier = 1 + 0.02 * max(0, age - 25)
    annual_income = base_income * income_multiplier

    # Step 3: CIBIL score depends on age, income, and history
    # Older + higher income → better score distribution
    cibil_mean = 680 + (age - 25) * 0.8 + (annual_income / 100000) * 5
    cibil_score = int(np.random.normal(cibil_mean, 55).clip(300, 900))

    # Step 4: Delinquency depends inversely on CIBIL
    # Low CIBIL → more DPD events
    dpd_prob = max(0, (750 - cibil_score) / 500)
    dpd_30_count = np.random.poisson(dpd_prob * 3)
    dpd_90_count = np.random.poisson(dpd_prob * 0.5)

    # Step 5: Existing obligations from income
    foir_target = np.random.beta(2, 4)  # Most people 20–50% FOIR
    monthly_obligations = (annual_income / 12) * foir_target

    # Step 6: Enquiry count (bureau fatigue)
    # More active loan seekers → more enquiries
    enquiry_count_6m = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        p=[0.25, 0.25, 0.20, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01]
    )

    return {...}
```

**Critical pitfalls to avoid:**
1. **Don't sample income and CIBIL independently** — they're correlated (ρ ≈ 0.45)
2. **Don't sample DTI without income** — DTI = obligations / income; both must be generated first
3. **Don't generate extreme outliers uniformly** — use log-normal for income, beta for ratios

---

### 5.3 Realistic Bank Generation

```python
BANK_ARCHETYPES = {
    "PSB": {
        "risk_appetite": "conservative",
        "min_cibil_score": (700, 720),      # Range across banks of this type
        "min_annual_income": (200000, 300000),
        "accepted_income_types": [["salaried", "business"]],
        "approval_base_rate": (0.30, 0.45),
        "disbursal_speed_days": (10, 25),
        "max_foir": (0.55, 0.65),
    },
    "private": {
        "risk_appetite": "moderate",
        "min_cibil_score": (680, 710),
        "min_annual_income": (180000, 350000),
        "accepted_income_types": [["salaried", "self_employed", "business"]],
        "approval_base_rate": (0.25, 0.40),
        "disbursal_speed_days": (5, 15),
        "max_foir": (0.60, 0.70),
    },
    "NBFC": {
        "risk_appetite": "aggressive",
        "min_cibil_score": (600, 680),
        "min_annual_income": (120000, 200000),
        "accepted_income_types": [["salaried", "self_employed", "business", "freelance"]],
        "approval_base_rate": (0.40, 0.65),
        "disbursal_speed_days": (2, 7),
        "max_foir": (0.65, 0.80),
    },
    "fintech": {
        "risk_appetite": "aggressive",
        "min_cibil_score": (580, 650),
        "min_annual_income": (100000, 180000),
        "accepted_income_types": [["salaried", "self_employed", "freelance"]],
        "approval_base_rate": (0.50, 0.75),
        "disbursal_speed_days": (1, 3),
        "max_foir": (0.70, 0.85),
    }
}
```

---

### 5.4 Approval Probability Simulation

This is the hardest and most important part. Don't use a hard threshold — use a **sigmoid-based probability** to simulate realistic bank decision-making:

```python
def simulate_approval_probability(lead, bank):
    """
    Each bank has a slightly different utility function.
    We compute a score and pass through sigmoid.
    """
    score = bank.intercept  # Bank's base approval propensity

    # CIBIL effect (nonlinear — marginal returns above 750)
    cibil_normalized = (lead.cibil_score - bank.min_cibil_score) / 100
    score += bank.cibil_weight * np.tanh(cibil_normalized)

    # DTI headroom (positive = under bank's max)
    dti_headroom = bank.max_foir - lead.foir
    score += bank.dti_weight * dti_headroom * 3

    # Bureau fatigue (sharp penalty above threshold)
    if lead.enquiry_count_6m > bank.max_enquiries_6m:
        score -= 2.0  # Near-certain rejection

    # Delinquency (DPD 90 is very bad)
    score -= lead.dpd_30_count * 0.2
    score -= lead.dpd_90_count * 0.8
    score -= lead.written_off_loans * 3.0

    # Income type match
    if lead.income_type not in bank.accepted_income_types:
        return 0.0  # Hard reject

    # Loan amount fit
    amount_fit = 1 - abs(lead.loan_amount_requested - bank.preferred_loan_size_midpoint) / bank.preferred_loan_size_range
    score += bank.amount_fit_weight * amount_fit

    # Add bank-specific noise (each bank has idiosyncratic behavior)
    score += np.random.normal(0, 0.3)

    return sigmoid(score)
```

---

### 5.5 Rejection Reason Attribution

```python
def assign_rejection_reason(lead, bank, approved):
    if approved:
        return None
    
    # Priority-ordered reason assignment
    if lead.income_type not in bank.accepted_income_types:
        return "income_type_mismatch"
    if lead.cibil_score < bank.min_cibil_score:
        return "low_cibil"
    if lead.enquiry_count_6m > bank.max_enquiries_6m:
        return "bureau_fatigue"
    if lead.foir > bank.max_foir:
        return "high_dti"
    if lead.annual_income < bank.min_annual_income:
        return "income_insufficient"
    if lead.written_off_loans > 0:
        return "delinquency_history"
    if lead.dpd_90_count > 0:
        return "delinquency_history"
    # Soft rejection — passed rules but bank declined
    return "policy_decline"
```

---

## Part 6: Feature Engineering for the ML Model

### 6.1 Three Feature Groups

**Group A — Lead Features (independent of bank)**
```
cibil_score, age, annual_income, foir, dti_ratio,
loan_to_income_ratio, enquiry_count_6m, dpd_30_count,
dpd_90_count, work_experience_years, loan_amount_requested,
loan_tenure_months, income_type_encoded, city_tier
```

**Group B — Bank Features (independent of lead)**
```
bank_type_encoded, risk_appetite_encoded,
min_cibil_score, max_foir, approval_base_rate,
disbursal_speed_days, interest_rate_min,
documentation_strictness_encoded
```

**Group C — Interaction Features (the critical ones)**
```python
# These are the features that distinguish your model from a simple classifier
cibil_gap = lead.cibil_score - bank.min_cibil_score
foir_headroom = bank.max_foir - lead.foir
income_headroom = lead.annual_income - bank.min_annual_income
amount_in_range = (bank.min_loan_amount <= lead.loan_amount_requested <= bank.max_loan_amount)
income_type_match = (lead.income_type in bank.accepted_income_types)
geography_match = (lead.state in bank.states_covered)
bureau_fatigue_flag = (lead.enquiry_count_6m > bank.max_enquiries_6m)
loan_type_match = (lead.loan_type in bank.loan_types_offered)
cibil_in_sweet_spot = (bank.preferred_cibil_min <= lead.cibil_score <= bank.preferred_cibil_max)
```

**Why interaction features are essential**: A CIBIL of 680 means very different things to a PSB (hard reject) vs. a fintech (approved easily). Without interaction features, your model cannot learn this bank-specific context.

---

### 6.2 Hidden Correlations Real Banks Use

These are rarely documented but consistently matter in real underwriting:

| Feature | Signal | Notes |
|---|---|---|
| **Income / CCAvg spend ratio** | Lifestyle inflation indicator | High spenders relative to income = risk |
| **Employer tenure / total experience ratio** | Job stability | Frequent job changers = risk |
| **Loan amount / current savings** | Repayment buffer | Low savings relative to EMI = risk |
| **Age at loan maturity** | `age + tenure_years` | Banks want this < 60 (retirement age) |
| **Enquiries per existing loan** | Shopping behavior | Many enquiries + few loans = desperate |
| **Settled loans ratio** | Credit character | Settlements are negotiated write-downs |
| **CIBIL trajectory** | Improvement or decline | Hard to simulate but worth noting |
| **Credit card utilization** | `cc_spend / cc_limit` | >80% utilization is a red flag |
| **City tier × bank type** | Coverage match | PSBs in tier 2/3; fintechs in tier 1 |

---

## Part 7: Training Data Structure for Ranking

### 7.1 Data Format

For **pointwise ranking** (recommended starting point):
```
Each row = one (lead, bank) application pair
Features = lead_features + bank_features + interaction_features
Target = 1 if disbursed, 0 otherwise
Group_id = lead_id (for ranking evaluation)
```

For **pairwise ranking** (LambdaMART):
```
Each row = one (lead, bank_A, bank_B) triple
Target = 1 if bank_A > bank_B for this lead, else 0
```

### 7.2 Evaluation Metrics

Don't just use F1. For a ranking system, use:

```python
from sklearn.metrics import roc_auc_score, ndcg_score

# Per-lead evaluation
for lead_id in test_leads:
    lead_apps = test_df[test_df.lead_id == lead_id]
    predicted_scores = model.predict_proba(lead_apps[features])[:, 1]
    actual_outcomes = lead_apps['converted'].values
    
    # NDCG@K: did you put the right banks at the top?
    ndcg_k = ndcg_score([actual_outcomes], [predicted_scores], k=3)
    
    # Recall@K: was the best bank in your top-K?
    recall_k = int(actual_outcomes[np.argsort(-predicted_scores)[:3]].any())
```

---

## Part 8: Biases and Pitfalls to Avoid

### 8.1 Data Leakage Traps

| Trap | Description | Fix |
|---|---|---|
| **Outcome-derived features** | Using `disbursed_amount` in features when target is `disbursed` | Only use features available at time of application |
| **Future bureau data** | Using CIBIL from after the application date | Snapshot CIBIL at application timestamp |
| **Bank acceptance as feature** | Including whether other banks approved this lead | This is future data relative to the current bank's decision |
| **Rejection reason as feature** | Rejection reason is only known after decision | Remove from training features |

### 8.2 Synthetic Data Pitfalls

| Pitfall | Description | Fix |
|---|---|---|
| **Independent feature sampling** | CIBIL, income, obligations sampled independently | Use causal generation chain |
| **Too clean distributions** | No outliers, no noise, perfectly calibrated | Add 5–10% noise and deliberate outliers |
| **Uniform bank behavior** | All banks approve at the same rate | Use archetypes with different base rates |
| **No impossible positives** | High CIBIL lead rejected by lenient bank | Enforce hard eligibility rules before probability simulation |
| **Label imbalance ignored** | Disbursed = 12%, but model sees 50/50 | Stratified splits, `scale_pos_weight` |
| **Collider bias** | Conditioning on "applied to bank" creates false correlations | Be careful about which leads appear in your training set |

### 8.3 Collider Bias — The Hardest Pitfall

If your training data only contains leads who were *actually submitted* to banks (not all leads), you have selection bias. Leads who were pre-filtered out by the first bank don't appear for that bank in your training set. Mitigation:

```python
# Include ALL lead-bank pairs in training, not just submitted ones
# Mark non-submitted pairs as "not eligible" rather than excluding them
all_pairs = cross_join(all_leads, all_banks)
all_pairs['eligible'] = all_pairs.apply(check_eligibility, axis=1)
all_pairs['converted'] = all_pairs.apply(simulate_outcome, axis=1)
# Train on all pairs; the model learns eligibility AND approval
```

---

## Part 9: Temporal Data Design

Real lending has time — model it:

```python
# Lead application timeline
t0: Lead created, CIBIL pulled (enquiry_count becomes n+1 for future checks)
t1: Application submitted to Bank A (enquiry registered)
t2: Bank A responds (3–21 days depending on bank type)
t3: Application submitted to Bank B (6 days after t0 — bureau now shows 2 enquiries)
t4: Bank B responds

# Key temporal features
days_since_first_enquiry: int        # How long has this lead been shopping?
enquiry_velocity: float              # Enquiries per week
application_sequence_number: int     # 1st bank, 2nd bank, etc.
is_sequential_application: bool      # Did they reapply after a rejection?
```

**Simulating bureau fatigue over time:**
```python
def get_effective_enquiry_count(lead, bank, application_date):
    """Enquiries decay out of the 6-month window."""
    recent_enquiries = [p for p in lead.bureau_pulls 
                        if (application_date - p.pulled_at).days <= 180]
    return len(recent_enquiries)
```

---

## Part 10: Implementation Roadmap

```
Phase 1 — Data Foundation (Week 1–2)
├─ Implement Lead generator with causal chain
├─ Implement Bank generator with archetypes
├─ Implement eligibility checker (hard rules)
└─ Generate 10K leads × 20 banks = 200K application pairs

Phase 2 — Simulation (Week 3)
├─ Implement approval probability simulator
├─ Implement rejection reason attributor
├─ Implement disbursal success/failure simulator
└─ Validate distributions against real-world benchmarks

Phase 3 — Feature Engineering (Week 4)
├─ Compute all interaction features
├─ Validate feature-target correlations with Spearman analysis
├─ Check for leakage using a time-aware split
└─ Build feature pipeline (sklearn Pipeline)

Phase 4 — Model Training (Week 5–6)
├─ Stage 1: XGBoost pointwise (lead × bank → P(disbursed))
├─ Evaluate with NDCG@3, Recall@3, AUC
├─ Stage 2: LightGBM with rank:ndcg objective
└─ A/B test ranking quality between Stage 1 and Stage 2

Phase 5 — Productionization (Week 7–8)
├─ Build eligibility API (fast rule engine)
├─ Build scoring API (model inference)
├─ Add monitoring: approval rate drift, score distribution
└─ Add feedback loop: capture real conversion outcomes
```

---

## Summary: Key Design Decisions

| Decision | Recommendation | Rationale |
|---|---|---|
| **ML framing** | Pointwise LTR → upgrade to LambdaMART | Simplest that works; upgrade path clear |
| **Target variable** | Binary: `disbursed = 1` | Clean, unambiguous, matches business value |
| **Interaction features** | Mandatory: cibil_gap, foir_headroom, income_type_match | Without these, bank-specific patterns are invisible |
| **Imbalance handling** | `scale_pos_weight` in XGBoost, stratified split | Preserves training signal without synthetic oversampling |
| **Data generation** | Causal chain generation (not independent sampling) | Avoids spurious independence between correlated features |
| **Temporal modeling** | Snapshot CIBIL + enquiry count at application time | Prevents leakage; models bureau fatigue correctly |
| **Bank heterogeneity** | Archetype-based behavior (PSB/private/NBFC/fintech) | Each bank type has genuinely different risk appetites |
| **Evaluation metric** | NDCG@3 + Recall@3 (ranking), AUC (scoring) | Matches business objective: top-3 bank suggestions |
