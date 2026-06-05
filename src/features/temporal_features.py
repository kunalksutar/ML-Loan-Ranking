"""
Temporal feature computation for the Lead-to-Bank Ranking system.

Operates on a merged DataFrame that already contains:
  - submitted_at, created_at  (from applications + leads join)
  - enquiry_count_6m          (from leads join)
  - application_sequence_num  (from applications_raw)
"""

from __future__ import annotations

import pandas as pd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4 temporal features in-place on a merged applications DataFrame.

    Required input columns:
      submitted_at, created_at, enquiry_count_6m, application_sequence_num

    New columns added:
      days_since_first_application  — days from lead creation to application date
      enquiry_velocity_weekly       — enquiry_count_6m / 26 weeks
      is_reapplication              — 1 if application_sequence_num > 1 else 0

    application_sequence_num is passed through unchanged (already present).
    """
    # days_since_first_application:
    # In the simulation, all banks for a lead share the same submitted_at
    # (= created_at + Uniform(1, 5) days). This captures the lead's urgency —
    # how quickly they submitted their application after being created.
    submitted = pd.to_datetime(df["submitted_at"])
    created = pd.to_datetime(df["created_at"])
    days_delta = (submitted - created).dt.days.fillna(0).clip(lower=0)
    df["days_since_first_application"] = days_delta.astype(int)

    # enquiry_velocity_weekly: normalize enquiry count to a per-week rate
    df["enquiry_velocity_weekly"] = (df["enquiry_count_6m"] / 26.0).round(4)

    # is_reapplication: this is not the lead's first eligible bank application
    df["is_reapplication"] = (df["application_sequence_num"] > 1).astype(int)

    return df
