"""
Evaluation metrics for the Lead-to-Bank Ranking system (CLAUDE.md §11).

Pointwise metrics (per application):
  - AUC-ROC, F1 (class 1), Precision (class 1), Recall (class 1)

Ranking metrics (per lead group) — primary business metrics:
  - NDCG@K  : was the correct bank ranked high?
  - Recall@K: was at least one converting bank in top-K?
  - MRR     : reciprocal rank of first converting bank

Minimum acceptance thresholds (CLAUDE.md §11):
  AUC-ROC ≥ 0.82, NDCG@3 ≥ 0.70, Recall@3 ≥ 0.75, MRR ≥ 0.60, F1 ≥ 0.65

Usage (library):
  from src.modeling.evaluator import evaluate_all
  metrics = evaluate_all(df, score_col="predicted_score", ks=[1, 3, 5])
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.feature_registry import GROUP_KEY, TARGET

logger = logging.getLogger(__name__)

# Minimum acceptance thresholds (CLAUDE.md §11)
THRESHOLDS = {
    "auc_roc": 0.82,
    "ndcg_at_3": 0.70,
    "recall_at_3": 0.75,
    "mrr": 0.60,
    "f1_class1": 0.65,
}


# ---------------------------------------------------------------------------
# Per-lead ranking metric helpers
# ---------------------------------------------------------------------------

def _dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at K for a single ranked list."""
    top_k = relevance[:k]
    if len(top_k) == 0:
        return 0.0
    gains = top_k / np.log2(np.arange(2, len(top_k) + 2))
    return float(gains.sum())


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@K for a single lead's bank list.

    Returns nan when the lead has no positive banks so the caller can
    exclude it from the mean — consistent with Recall@K and MRR treatment.
    """
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(y_score)[::-1]
    ranked_relevance = y_true[order]
    ideal_relevance = np.sort(y_true)[::-1]

    dcg = _dcg_at_k(ranked_relevance, k)
    idcg = _dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0.0 else 0.0


def _recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """1 if at least one converting bank appears in top-K predictions, else 0."""
    if y_true.sum() == 0:
        return float("nan")  # lead has no converting bank — exclude from mean
    order = np.argsort(y_score)[::-1]
    return float(y_true[order[:k]].sum() > 0)


def _reciprocal_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Reciprocal rank of the first converting bank in the ranked list."""
    if y_true.sum() == 0:
        return float("nan")  # no converting bank — exclude from MRR
    order = np.argsort(y_score)[::-1]
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Aggregate ranking metrics over all leads
# ---------------------------------------------------------------------------

def compute_ranking_metrics(
    df: pd.DataFrame,
    score_col: str = "predicted_score",
    ks: list[int] | None = None,
    group_col: str = GROUP_KEY,
    target_col: str = TARGET,
) -> dict[str, float]:
    """
    Compute NDCG@K, Recall@K, and MRR aggregated over all lead groups.

    Parameters
    ----------
    df         : DataFrame with group_col, target_col, and score_col columns.
    score_col  : Column name for predicted probability scores.
    ks         : List of cutoff values; defaults to [1, 3, 5].
    group_col  : Lead identifier column.
    target_col : Binary target column (0/1 converted).

    Returns
    -------
    dict mapping metric names to float values.
    """
    if ks is None:
        ks = [1, 3, 5]

    ndcg_per_lead: dict[int, list[float]] = {k: [] for k in ks}
    recall_per_lead: dict[int, list[float]] = {k: [] for k in ks}
    rr_per_lead: list[float] = []

    for lead_id, group in df.groupby(group_col, sort=False):
        y_true = group[target_col].to_numpy().astype(float)
        y_score = group[score_col].to_numpy().astype(float)

        for k in ks:
            ndcg = _ndcg_at_k(y_true, y_score, k)
            if not np.isnan(ndcg):
                ndcg_per_lead[k].append(ndcg)
            r = _recall_at_k(y_true, y_score, k)
            if not np.isnan(r):
                recall_per_lead[k].append(r)

        rr = _reciprocal_rank(y_true, y_score)
        if not np.isnan(rr):
            rr_per_lead.append(rr)

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"ndcg_at_{k}"] = float(np.mean(ndcg_per_lead[k])) if ndcg_per_lead[k] else 0.0
        metrics[f"recall_at_{k}"] = float(np.mean(recall_per_lead[k])) if recall_per_lead[k] else 0.0
    metrics["mrr"] = float(np.mean(rr_per_lead)) if rr_per_lead else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Pointwise classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute AUC-ROC, F1 (class 1), Precision (class 1), Recall (class 1).

    Parameters
    ----------
    y_true     : Binary ground-truth labels.
    y_score    : Predicted probabilities for class 1.
    threshold  : Decision threshold for binary predictions.
    """
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "f1_class1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_class1": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_class1": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Combined evaluation entrypoint
# ---------------------------------------------------------------------------

def evaluate_all(
    df: pd.DataFrame,
    score_col: str = "predicted_score",
    ks: list[int] | None = None,
    group_col: str = GROUP_KEY,
    target_col: str = TARGET,
    threshold: float = 0.5,
    split_label: str = "",
) -> dict[str, float]:
    """
    Run all pointwise + ranking metrics and log results.

    Parameters
    ----------
    df          : DataFrame with predictions and targets.
    score_col   : Predicted probability column.
    ks          : NDCG/Recall cutoffs; defaults to [1, 3, 5].
    group_col   : Lead identifier column.
    target_col  : Binary target column.
    threshold   : Classification threshold.
    split_label : Label for logging (e.g., "val", "test").

    Returns
    -------
    Combined dict of all metrics.
    """
    if ks is None:
        ks = [1, 3, 5]

    clf = compute_classification_metrics(
        df[target_col].to_numpy(),
        df[score_col].to_numpy(),
        threshold=threshold,
    )
    rank = compute_ranking_metrics(df, score_col=score_col, ks=ks, group_col=group_col, target_col=target_col)
    metrics = {**clf, **rank}

    prefix = f"[{split_label}] " if split_label else ""
    logger.info(
        "%sAUC=%.4f | F1=%.4f | NDCG@1=%.4f | NDCG@3=%.4f | NDCG@5=%.4f | "
        "Recall@3=%.4f | MRR=%.4f",
        prefix,
        metrics.get("auc_roc", 0.0),
        metrics.get("f1_class1", 0.0),
        metrics.get("ndcg_at_1", 0.0),
        metrics.get("ndcg_at_3", 0.0),
        metrics.get("ndcg_at_5", 0.0),
        metrics.get("recall_at_3", 0.0),
        metrics.get("mrr", 0.0),
    )

    return metrics


# ---------------------------------------------------------------------------
# Threshold gate
# ---------------------------------------------------------------------------

def check_thresholds(metrics: dict[str, float]) -> dict[str, bool]:
    """
    Check computed metrics against CLAUDE.md §11 minimum thresholds.

    Returns a dict mapping each threshold name to True (pass) or False (fail).
    Logs an ERROR for each metric that falls below its threshold.
    """
    results: dict[str, bool] = {}
    for metric_key, threshold_val in THRESHOLDS.items():
        actual = metrics.get(metric_key, 0.0)
        passed = actual >= threshold_val
        results[metric_key] = passed
        if not passed:
            logger.error(
                "Threshold FAIL: %s=%.4f below threshold=%.2f",
                metric_key, actual, threshold_val,
            )
        else:
            logger.info(
                "Threshold PASS: %s=%.4f >= %.2f",
                metric_key, actual, threshold_val,
            )
    return results


# ---------------------------------------------------------------------------
# Per-bank and per-income-type sliced AUC (error analysis support)
# ---------------------------------------------------------------------------

def per_bank_auc(
    df: pd.DataFrame,
    score_col: str = "predicted_score",
    bank_col: str = "bank_id",
    target_col: str = TARGET,
    min_auc_threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Compute AUC per bank. Flag any bank with AUC < min_auc_threshold.

    Returns a DataFrame sorted by AUC ascending with a `flagged` column.
    """
    rows = []
    for bank_id, grp in df.groupby(bank_col, sort=False):
        y = grp[target_col].to_numpy()
        if y.sum() == 0 or (y == 0).sum() == 0:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y, grp[score_col].to_numpy()))
        rows.append({"bank_id": bank_id, "auc": auc, "n_pairs": len(grp), "n_positive": int(y.sum())})

    result = pd.DataFrame(rows).sort_values("auc")
    result["flagged"] = result["auc"] < min_auc_threshold
    for _, row in result[result["flagged"]].iterrows():
        logger.warning("Per-bank AUC below %.2f: bank_id=%s auc=%.4f", min_auc_threshold, row["bank_id"], row["auc"])
    return result.reset_index(drop=True)


def per_income_type_ndcg(
    df: pd.DataFrame,
    score_col: str = "predicted_score",
    income_col: str = "income_type_enc",
    k: int = 3,
) -> pd.DataFrame:
    """NDCG@K broken down by income_type — expect freelance to underperform."""
    rows = []
    for enc_val, grp in df.groupby(income_col, sort=False):
        rank_metrics = compute_ranking_metrics(grp, score_col=score_col, ks=[k])
        rows.append({"income_type_enc": enc_val, f"ndcg_at_{k}": rank_metrics[f"ndcg_at_{k}"], "n_leads": grp["lead_id"].nunique()})
    return pd.DataFrame(rows).sort_values(f"ndcg_at_{k}")


# ---------------------------------------------------------------------------
# §14 Error Analysis helpers
# ---------------------------------------------------------------------------

def segment_errors(
    df: pd.DataFrame,
    score_col: str = "predicted_score",
    k: int = 3,
) -> pd.DataFrame:
    """
    Classify each (lead × bank) row into one of four error types based on
    rank position vs. ground-truth conversion label.

    Banks are ranked within each lead group by score_col descending.
    Error types assigned:
      true_positive   converted=1 AND rank <= k  (correct top-K recommendation)
      false_negative  converted=1 AND rank >  k  (missed disbursed bank — §14 #1)
      false_positive  converted=0 AND rank <= k  (wasted recommendation slot — §14 #2)
      true_negative   converted=0 AND rank >  k  (correctly excluded)

    Leads with zero converted=1 rows cannot produce false negatives (no bank
    ever disbursed) — their rows stay true_negative / false_positive only.

    Returns a copy of df with added 'error_type' column.
    """
    out = df.copy()

    # Rank banks within each lead (rank 1 = highest predicted score)
    out["_rank"] = (
        out.groupby(GROUP_KEY)[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Leads that have at least one positive example (FN is only meaningful here)
    leads_with_positive = out.loc[out[TARGET] == 1, GROUP_KEY].unique()
    has_positive = out[GROUP_KEY].isin(leads_with_positive)

    in_top_k = out["_rank"] <= k
    positive = out[TARGET] == 1

    # Vectorised assignment (np.select evaluates conditions in order)
    conditions = [
        positive & in_top_k,
        positive & ~in_top_k & has_positive,
        ~positive & in_top_k,
    ]
    choices = ["true_positive", "false_negative", "false_positive"]
    out["error_type"] = np.select(conditions, choices, default="true_negative")

    out = out.drop(columns=["_rank"])
    return out


def calibration_data(
    df: pd.DataFrame,
    score_col: str = "predicted_score",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute reliability-diagram data: predicted probability vs. actual positive rate.

    Bins the [0, 1] score range into n_bins equal-width buckets and returns,
    per occupied bin:
      bin_lower, bin_upper, bin_center — bucket boundaries and midpoint
      mean_predicted                   — mean predicted score in the bin
      actual_positive_rate             — empirical positive rate in the bin
      count                            — number of samples in the bin

    A well-calibrated model should have mean_predicted ≈ actual_positive_rate
    (points near the diagonal in a reliability diagram).
    """
    scores = df[score_col].to_numpy()
    labels = df[TARGET].to_numpy()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize returns 1..n_bins; clip to 0-based index
    bin_idx = np.clip(np.digitize(scores, edges[:-1]) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append({
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "bin_center": float((edges[b] + edges[b + 1]) / 2),
            "mean_predicted": float(scores[mask].mean()),
            "actual_positive_rate": float(labels[mask].mean()),
            "count": int(mask.sum()),
        })

    return pd.DataFrame(rows)


def profile_error_segment(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "error_type",
) -> pd.DataFrame:
    """
    Summarise mean feature values by error_type (or any group column).

    Only features that are present in df are included; silently skips missing
    columns so the caller need not filter first.

    Returns a DataFrame indexed by group values, columns = feature means.
    Useful for answering: "how do false negatives differ from true negatives
    in terms of cibil_gap, foir_headroom, etc.?"
    """
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    return df.groupby(group_col)[available].mean().round(4)
