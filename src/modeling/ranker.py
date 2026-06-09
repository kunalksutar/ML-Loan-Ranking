"""
Stage 3 — Ranking Layer for the Lead-to-Bank Ranking system (CLAUDE.md §10).

Consumes eligible bank candidates (post Stage 1) and model scores (Stage 2)
to produce a sorted, top-K bank recommendation list for each lead.

Runtime flow:
  Stage 1  Eligibility Engine  → shortlist of K eligible banks
  Stage 2  XGBoost scorer      → P(disbursed | lead, bank) per pair
  Stage 3  This module         → sort descending, return top_k banks

Usage (batch inference from saved bundle):
  from src.modeling.ranker import Ranker
  ranker = Ranker.from_bundle("models/v1")
  ranked = ranker.rank_lead(lead_row, banks_df, top_k=5)

Usage (library, model already in memory):
  from src.modeling.ranker import rank_pairs
  scores_df = rank_pairs(feature_df, model, preprocessor, top_k=5)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.eligibility.rule_engine import apply_eligibility_batch
from src.features.feature_registry import ALL_FEATURES, GROUP_KEY, TARGET

logger = logging.getLogger(__name__)

_SCORE_COL = "rank_score"


# ---------------------------------------------------------------------------
# Low-level scoring helper
# ---------------------------------------------------------------------------

def score_feature_matrix(
    feature_df: pd.DataFrame,
    model,
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """
    Transform `feature_df[ALL_FEATURES]` and return predicted probabilities.

    Returns
    -------
    1-D ndarray of shape (n_rows,) with P(converted=1) for each row.
    """
    X = preprocessor.transform(feature_df[ALL_FEATURES])
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Batch ranking over a pre-built feature DataFrame (evaluation / offline use)
# ---------------------------------------------------------------------------

def rank_pairs(
    feature_df: pd.DataFrame,
    model,
    preprocessor: ColumnTransformer,
    top_k: int = 5,
    group_col: str = GROUP_KEY,
    score_col: str = _SCORE_COL,
) -> pd.DataFrame:
    """
    Score every row in `feature_df` and return a DataFrame sorted
    descending by score within each lead group.

    This is the batch-evaluation form used for offline evaluation. It
    operates on the already-built feature DataFrame (applications_splits/).

    Parameters
    ----------
    feature_df  : DataFrame with ALL_FEATURES columns, group_col, and optionally TARGET.
    model       : Fitted XGBClassifier.
    preprocessor: Fitted sklearn ColumnTransformer.
    top_k       : Number of top banks to keep per lead.
    group_col   : Lead identifier column.
    score_col   : Name of the output score column added to the result.

    Returns
    -------
    DataFrame with a `score_col` column added, sorted descending by
    score within each lead, limited to top_k rows per lead.
    """
    scores = score_feature_matrix(feature_df, model, preprocessor)
    result = feature_df.copy()
    result[score_col] = scores

    if top_k is not None:
        result = (
            result
            .sort_values([group_col, score_col], ascending=[True, False])
            .groupby(group_col, sort=False)
            .head(top_k)
            .reset_index(drop=True)
        )
    else:
        result = result.sort_values([group_col, score_col], ascending=[True, False]).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Online / inference-time Ranker class
# ---------------------------------------------------------------------------

@dataclass
class RankedBank:
    """A single bank in a ranked recommendation list."""
    rank: int
    bank_id: str
    bank_name: str
    bank_type: str
    rank_score: float
    interest_rate_min: float
    disbursal_speed_days: float


@dataclass
class RankResult:
    """Output of the ranker for a single lead."""
    lead_id: str
    n_eligible_banks: int
    ranked_banks: list[RankedBank]
    latency_ms: float
    eligibility_failure_summary: dict[str, int] = field(default_factory=dict)


class Ranker:
    """
    Three-stage inference pipeline: eligibility → scoring → ranking.

    Instantiate via `Ranker(model, preprocessor)` or
    `Ranker.from_bundle("models/v1")` for bundle-based loading.
    """

    def __init__(self, model, preprocessor: ColumnTransformer, top_k: int = 5):
        self.model = model
        self.preprocessor = preprocessor
        self.top_k = top_k

    @classmethod
    def from_bundle(cls, model_dir: str = "models/v1", top_k: int = 5) -> "Ranker":
        from src.modeling.model_registry import load_model_bundle
        bundle = load_model_bundle(model_dir)
        return cls(bundle["model"], bundle["preprocessor"], top_k=top_k)

    def rank_lead(
        self,
        lead_row: pd.Series | pd.DataFrame,
        banks_df: pd.DataFrame,
        feature_builder_fn=None,
        top_k: int | None = None,
    ) -> RankResult:
        """
        Rank all banks for a single lead.

        Parameters
        ----------
        lead_row        : Single-row DataFrame or Series for the lead.
        banks_df        : All candidate banks DataFrame (raw bank attributes).
        feature_builder_fn : Optional callable(leads_df, banks_df) → features_df.
                           If None, banks_df is assumed to already contain ALL_FEATURES.
        top_k           : Override default top_k.

        Returns
        -------
        RankResult with ranked_banks sorted descending by rank_score.
        """
        t_start = time.perf_counter()
        effective_k = top_k if top_k is not None else self.top_k

        # Normalise lead_row to a single-row DataFrame
        if isinstance(lead_row, pd.Series):
            lead_df = lead_row.to_frame().T.reset_index(drop=True)
        else:
            lead_df = lead_row.head(1).reset_index(drop=True)

        lead_id = str(lead_df["lead_id"].iloc[0]) if "lead_id" in lead_df.columns else "unknown"

        # Stage 1: eligibility engine
        eligible_mask, failure_reasons = apply_eligibility_batch(lead_df, banks_df)
        eligible_bank_indices = np.where(eligible_mask)[0]  # indices into banks_df rows

        n_eligible = len(eligible_bank_indices)
        failure_summary: dict[str, int] = {}
        for reason in failure_reasons[~eligible_mask]:
            if reason is not None:
                failure_summary[str(reason)] = failure_summary.get(str(reason), 0) + 1

        if n_eligible == 0:
            logger.warning("No eligible banks for lead_id=%s", lead_id)
            latency_ms = (time.perf_counter() - t_start) * 1000
            return RankResult(
                lead_id=lead_id,
                n_eligible_banks=0,
                ranked_banks=[],
                latency_ms=round(latency_ms, 2),
                eligibility_failure_summary=failure_summary,
            )

        eligible_banks = banks_df.iloc[eligible_bank_indices].reset_index(drop=True)

        # Build feature matrix for eligible (lead × bank) pairs
        if feature_builder_fn is not None:
            features_df = feature_builder_fn(lead_df, eligible_banks)
        else:
            # Fallback: assume eligible_banks contains ALL_FEATURES already
            features_df = eligible_banks

        # Stage 2: score
        scores = score_feature_matrix(features_df, self.model, self.preprocessor)

        # Stage 3: sort descending and take top_k
        order = np.argsort(scores)[::-1][:effective_k]
        ranked_banks: list[RankedBank] = []
        for rank, idx in enumerate(order, start=1):
            bank_row = eligible_banks.iloc[idx]
            ranked_banks.append(RankedBank(
                rank=rank,
                bank_id=str(bank_row.get("bank_id", "")),
                bank_name=str(bank_row.get("name", "")),
                bank_type=str(bank_row.get("bank_type", "")),
                rank_score=float(scores[idx]),
                interest_rate_min=float(bank_row.get("interest_rate_min", 0.0)),
                disbursal_speed_days=float(bank_row.get("disbursal_speed_days", 0.0)),
            ))

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Ranked lead_id=%s | n_eligible=%d | top_k=%d | latency_ms=%.2f",
            lead_id, n_eligible, min(n_eligible, effective_k), latency_ms,
        )

        return RankResult(
            lead_id=lead_id,
            n_eligible_banks=n_eligible,
            ranked_banks=ranked_banks,
            latency_ms=round(latency_ms, 2),
            eligibility_failure_summary=failure_summary,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="Rank banks for a random lead (demo)")
    parser.add_argument("--model-dir", default="models/v1")
    parser.add_argument("--leads", default="data/raw/leads.parquet")
    parser.add_argument("--banks", default="data/raw/banks.parquet")
    parser.add_argument("--features", default="data/processed/applications_features.parquet")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lead-idx", type=int, default=0, help="Lead row index to rank for")
    args = parser.parse_args()

    leads = pd.read_parquet(args.leads)
    banks = pd.read_parquet(args.banks)
    features = pd.read_parquet(args.features)

    ranker = Ranker.from_bundle(args.model_dir, top_k=args.top_k)
    lead_row = leads.iloc[[args.lead_idx]]
    lead_id = lead_row["lead_id"].iloc[0]

    # Build a feature lookup for this lead's pairs
    lead_features = features[features["lead_id"] == lead_id].copy()

    def feature_builder(ld, bk):
        return lead_features[lead_features["bank_id"].isin(bk["bank_id"])]

    result = ranker.rank_lead(lead_row, banks, feature_builder_fn=feature_builder)

    print(f"\nLead ID: {result.lead_id}")
    print(f"Eligible banks: {result.n_eligible_banks}")
    print(f"Latency: {result.latency_ms:.2f}ms")
    print(f"\nTop-{args.top_k} ranked banks:")
    for rb in result.ranked_banks:
        print(f"  #{rb.rank} {rb.bank_name} ({rb.bank_type}) | score={rb.rank_score:.4f} | "
              f"rate={rb.interest_rate_min:.1f}% | speed={rb.disbursal_speed_days:.0f}d")


if __name__ == "__main__":
    _main()
