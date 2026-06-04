"""
Bureau pull log simulator for the Lead-to-Bank Ranking system.

Generates a `bureau_pulls.parquet` record for every eligible (lead × bank)
application pair. Each pull represents the bank querying the lead's credit
report at the time of application submission.

Fields (per CLAUDE.md §3.4):
  pull_id             : UUID
  lead_id             : FK → leads
  bank_id             : FK → banks
  pulled_at           : datetime (= submitted_at of the application)
  cibil_score_at_pull : int (current CIBIL score at pull time)
  enquiry_type        : hard | soft

All hard enquiries are generated for submitted applications. The bureau pull
log can later be used to compute rolling effective_enquiry_count at any
application timestamp to simulate bureau fatigue.

Usage (library):
  from src.simulation.bureau_simulator import generate_bureau_pulls
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_bureau_pulls(
    applications: pd.DataFrame,
    leads: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate bureau pull records for all submitted (eligible) applications.

    Parameters
    ----------
    applications : applications_raw DataFrame with at least:
                   lead_id, bank_id, eligibility_passed, submitted_at
    leads        : leads DataFrame with lead_id and cibil_score
    rng          : reproducible numpy RNG

    Returns
    -------
    pd.DataFrame with bureau pull log schema
    """
    submitted = applications[applications["eligibility_passed"]].copy()

    if submitted.empty:
        logger.warning("No eligible applications found; bureau pull log is empty.")
        return pd.DataFrame(columns=[
            "pull_id", "lead_id", "bank_id",
            "pulled_at", "cibil_score_at_pull", "enquiry_type",
        ])

    n_pulls = len(submitted)
    logger.info("Generating %d bureau pull records", n_pulls)

    # Join cibil_score from leads
    cibil_map = leads.set_index("lead_id")["cibil_score"].to_dict()

    # Deterministic UUIDs from RNG
    uuid_bytes = rng.integers(0, 256, size=(n_pulls, 16), dtype=np.uint8)
    uuid_bytes[:, 6] = (uuid_bytes[:, 6] & 0x0F) | 0x40  # version 4
    uuid_bytes[:, 8] = (uuid_bytes[:, 8] & 0x3F) | 0x80  # variant bits
    pull_ids = [str(uuid.UUID(bytes=bytes(row))) for row in uuid_bytes]

    # Most applications generate hard pulls; ~15% may be soft (pre-screening)
    enquiry_types = rng.choice(
        ["hard", "soft"],
        size=n_pulls,
        p=[0.85, 0.15],
    )

    cibil_at_pull = [
        int(cibil_map.get(lid, 0)) for lid in submitted["lead_id"]
    ]

    df = pd.DataFrame({
        "pull_id":             pull_ids,
        "lead_id":             submitted["lead_id"].values,
        "bank_id":             submitted["bank_id"].values,
        "pulled_at":           submitted["submitted_at"].values,
        "cibil_score_at_pull": cibil_at_pull,
        "enquiry_type":        enquiry_types,
    })

    logger.info(
        "Bureau pull log complete | n=%d | hard=%.1f%% | soft=%.1f%%",
        n_pulls,
        100.0 * (df["enquiry_type"] == "hard").mean(),
        100.0 * (df["enquiry_type"] == "soft").mean(),
    )
    return df


def save_bureau_pulls(df: pd.DataFrame, output_dir: str) -> Path:
    """Save bureau pulls DataFrame to parquet and return the output path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bureau_pulls.parquet"
    df.to_parquet(path, index=False)
    logger.info("Bureau pulls saved to %s (%d rows)", path, len(df))
    return path
