"""
Statistical constants and lookup tables for synthetic data generation.

All probability arrays are normalized to sum to 1.0.
All range tuples represent inclusive [min, max] unless noted.
"""

# --- Indian state codes with approximate lending-market population weights ---
STATE_WEIGHTS = {
    "MH": 0.112,  # Maharashtra
    "UP": 0.180,  # Uttar Pradesh (largest population)
    "BR": 0.086,  # Bihar
    "WB": 0.075,  # West Bengal
    "MP": 0.060,  # Madhya Pradesh
    "TN": 0.060,  # Tamil Nadu
    "RJ": 0.057,  # Rajasthan
    "KA": 0.053,  # Karnataka
    "GJ": 0.050,  # Gujarat
    "AP": 0.049,  # Andhra Pradesh
    "OR": 0.035,  # Odisha
    "TG": 0.029,  # Telangana
    "KL": 0.027,  # Kerala
    "JH": 0.027,  # Jharkhand
    "AS": 0.025,  # Assam
    "PB": 0.023,  # Punjab
    "HR": 0.021,  # Haryana
    "DL": 0.016,  # Delhi
    "UK": 0.009,  # Uttarakhand
    "HP": 0.006,  # Himachal Pradesh
}  # sums to 1.000

# --- Income types ---
INCOME_TYPES = ["salaried", "self_employed", "business", "freelance"]
INCOME_TYPE_PROBS = [0.55, 0.25, 0.15, 0.05]

# --- City tiers (1=metro, 2=tier-2, 3=rural) ---
CITY_TIERS = [1, 2, 3]
CITY_TIER_PROBS = [0.35, 0.40, 0.25]

# --- LogNormal income parameters (mu, sigma) per income_type ---
# Resulting median annual incomes (approx): salaried ~440K, SE ~360K, business ~730K, freelance ~270K
INCOME_LOGNORMAL_PARAMS = {
    "salaried":      (13.0, 0.6),
    "self_employed": (12.8, 0.8),
    "business":      (13.5, 1.0),
    "freelance":     (12.5, 0.7),
}

# --- Employer category distribution per income_type: (categories, probs) ---
EMPLOYER_CATEGORY_BY_INCOME_TYPE = {
    "salaried": (
        ["PSU", "private_listed", "private_unlisted", "MNC", "govt"],
        [0.10, 0.25, 0.30, 0.15, 0.20],
    ),
    "self_employed": (
        ["private_unlisted", "private_listed", "MNC"],
        [0.70, 0.20, 0.10],
    ),
    "business": (
        ["private_unlisted", "private_listed"],
        [0.70, 0.30],
    ),
    "freelance": (
        ["private_unlisted", "MNC"],
        [0.85, 0.15],
    ),
}

# --- Loan types ---
LOAN_TYPES = ["personal", "home", "car", "education", "business", "gold", "lap"]

# --- Loan type probability weights per income_type (aligns with LOAN_TYPES order) ---
LOAN_TYPE_WEIGHTS_BY_INCOME = {
    "salaried":      [0.35, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02],
    "self_employed": [0.25, 0.15, 0.15, 0.05, 0.25, 0.08, 0.07],
    "business":      [0.15, 0.20, 0.10, 0.05, 0.35, 0.05, 0.10],
    "freelance":     [0.45, 0.15, 0.15, 0.10, 0.10, 0.03, 0.02],
}

# --- Loan tenure options (months) per loan_type ---
LOAN_TYPE_TENURES = {
    "personal":  [12, 24, 36, 48, 60],
    "home":      [60, 84, 120, 180, 240],
    "car":       [12, 24, 36, 48, 60, 84],
    "education": [12, 24, 36, 48, 60, 84],
    "business":  [12, 24, 36, 48, 60],
    "gold":      [6, 12, 18, 24],
    "lap":       [12, 24, 36, 60, 84, 120, 180],
}

# --- Approximate monthly interest rates (annual rate / 12) per loan_type ---
LOAN_TYPE_MONTHLY_RATES = {
    "personal":  0.14 / 12,
    "home":      0.085 / 12,
    "car":       0.10 / 12,
    "education": 0.09 / 12,
    "business":  0.15 / 12,
    "gold":      0.09 / 12,
    "lap":       0.11 / 12,
}

# --- Loan amount bounds: (min_amount_INR, max_absolute_INR, max_income_multiple) ---
# max effective cap = min(max_absolute, annual_income * max_income_multiple)
LOAN_AMOUNT_BOUNDS = {
    "personal":  (50_000,     2_500_000,   5.0),
    "home":      (1_000_000,  30_000_000,  80.0),
    "car":       (300_000,    3_000_000,   2.0),
    "education": (100_000,    2_000_000,   3.0),
    "business":  (100_000,    10_000_000,  5.0),
    "gold":      (10_000,     2_000_000,   1.0),
    "lap":       (500_000,    15_000_000,  10.0),
}
