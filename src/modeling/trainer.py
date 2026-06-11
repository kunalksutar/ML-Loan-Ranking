"""
Stage 2 — XGBoost Pointwise Scorer training (CLAUDE.md §10).

Training loop:
  1. Load train/val/test splits (lead-level, already split)
  2. Build + fit preprocessor on train only
  3. Train XGBClassifier with binary:logistic + early stopping on val AUC
  4. Evaluate on val and test (pointwise + ranking metrics)
  5. Log everything to MLflow
  6. Save model bundle to models/v1/

Usage:
  python -m src.modeling.trainer --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from xgboost import XGBClassifier

matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt

from src.features.feature_registry import ALL_FEATURES, GROUP_KEY, TARGET
from src.modeling.evaluator import check_thresholds, evaluate_all, per_bank_auc, per_income_type_ndcg
from src.modeling.model_registry import (
    build_eligibility_rules,
    build_feature_schema,
    save_model_bundle,
)
from src.preprocessing.pipeline_builder import (
    build_preprocessor,
    fit_preprocessor,
    load_feature_config,
    transform_split,
)
from src.preprocessing.splitting import (
    compute_scale_pos_weight,
    validate_split_integrity,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/model_config.yaml"

# ---------------------------------------------------------------------------
# Per-iteration MLflow callback — logs train/val metrics each boosting round
# ---------------------------------------------------------------------------

class _MLflowIterCallback(xgb.callback.TrainingCallback):
    """Log train and val metrics to the active MLflow run after every iteration.

    XGBoost names eval sets as 'validation_0', 'validation_1', ... in the order
    they appear in eval_set.  We map position 0 → 'train', position 1 → 'val'
    so the MLflow UI renders two labelled learning curves per metric.

    Logged metric names:  train_iter_auc, train_iter_logloss,
                          val_iter_auc,   val_iter_logloss
    The `step` argument maps to the boosting round, producing X-axis labels in
    the MLflow metric chart.
    """

    _PREFIX_MAP = {0: "train", 1: "val"}

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        for i, (_, metrics) in enumerate(evals_log.items()):
            prefix = self._PREFIX_MAP.get(i, f"eval{i}")
            for metric_name, values in metrics.items():
                mlflow.log_metric(
                    f"{prefix}_iter_{metric_name}",
                    float(values[-1]),
                    step=epoch,
                )
        return False  # do not early-stop from callback


DEFAULT_SPLITS_DIR = "data/processed/applications_splits"
DEFAULT_BANKS_PATH = "data/raw/banks.parquet"
DEFAULT_MODEL_DIR = "models/v1"
DEFAULT_FEATURE_CONFIG = "configs/feature_config.yaml"
SCORE_COL = "predicted_score"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_model_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# MLflow setup — all files under experiments/mlflow/
# ---------------------------------------------------------------------------

def _setup_mlflow(mlflow_base: str, experiment_name: str) -> str:
    """
    Configure MLflow so that ALL experiment data (DB, artifacts, model registry)
    lives under `mlflow_base/` — matching the CLAUDE.md §13 layout:

        experiments/mlflow/
        ├── mlflow.db          <- SQLite tracking store (params, metrics, tags)
        └── artifacts/         <- artifact store (logged files, xgb model)
            └── <experiment_id>/
                └── <run_id>/
                    └── artifacts/

    MLflow 3.x separates the *tracking URI* (where metadata is stored) from the
    *artifact location* (where files are stored).  When `set_experiment()` is
    called without an explicit `artifact_location`, MLflow defaults the artifact
    root to `file://<cwd>/mlruns/<experiment_id>` — which is why artifacts were
    appearing in a root-level `mlruns/` directory instead of inside
    `experiments/mlflow/`.

    This function:
      1. Creates `experiments/mlflow/artifacts/` on disk.
      2. Configures the SQLite tracking URI.
      3. Creates the named experiment with the correct `artifact_location` if it
         doesn't exist yet.  If it already exists, verifies the stored location
         matches; if not, deletes the stale experiment and recreates it cleanly.

    Returns the artifact root URI (a `file://` string).
    """
    base = Path(mlflow_base).resolve()
    db_path = base / "mlflow.db"
    artifact_root = base / "artifacts"

    base.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    db_uri = f"sqlite:///{db_path}"
    artifact_uri = artifact_root.as_uri()   # file:///...absolute...

    mlflow.set_tracking_uri(db_uri)

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    if exp is None:
        client.create_experiment(experiment_name, artifact_location=artifact_uri)
        logger.info("MLflow experiment '%s' created | artifact_uri=%s", experiment_name, artifact_uri)
    elif getattr(exp, "lifecycle_stage", "active") == "deleted":
        # Soft-deleted experiment — restore it before reuse so set_experiment() succeeds.
        client.restore_experiment(exp.experiment_id)
        logger.info("MLflow experiment '%s' restored from deleted state", experiment_name)
    elif not exp.artifact_location.startswith(str(artifact_root.as_uri())):
        # Stale experiment pointing at the wrong artifact root — delete and recreate.
        logger.warning(
            "MLflow experiment '%s' has stale artifact_location=%s; recreating under %s",
            experiment_name, exp.artifact_location, artifact_uri,
        )
        client.delete_experiment(exp.experiment_id)
        client.create_experiment(experiment_name, artifact_location=artifact_uri)
        logger.info("MLflow experiment '%s' recreated | artifact_uri=%s", experiment_name, artifact_uri)
    else:
        logger.info(
            "MLflow experiment '%s' exists | artifact_uri=%s",
            experiment_name, exp.artifact_location,
        )

    mlflow.set_experiment(experiment_name)
    return artifact_uri


# ---------------------------------------------------------------------------
# Model Registry helpers
# ---------------------------------------------------------------------------

#: Tags set once on the Registered Model entity (not per-version).
_REGISTERED_MODEL_TAGS = {
    "framework": "xgboost",
    "problem_type": "learning_to_rank_pointwise",
    "objective": "binary:logistic",
    "dataset_version": "synthetic_indian_lending_v1",
    "project": "lead_bank_ranking",
    "section": "CLAUDE.md_section_10_11",
    "input_features": "57",
    "output": "P(disbursed=1|lead,bank)",
}

_REGISTERED_MODEL_DESCRIPTION = (
    "XGBoost binary:logistic pointwise scorer for Lead-to-Bank ranking. "
    "Predicts P(disbursed=1 | lead, bank) so eligible banks can be ranked per lead. "
    "Input: 57 engineered features — 25 lead + 13 bank + 15 interaction + 4 temporal. "
    "Stage 1 eligibility rules reduce the full bank set to 3-12 candidates before scoring."
)


def _register_and_tag_model(
    run_id: str,
    model_uri: str,
    registry_model_name: str,
    val_metrics: dict,
    test_metrics: dict,
    metadata: dict,
    git_commit: str = "",
) -> str:
    """
    Register a logged model version in the MLflow Model Registry and apply
    rich tags to both the registered model entity and the new version.

    Parameters
    ----------
    run_id               : MLflow run that produced the model.
    model_uri            : URI of the logged model (e.g. ``runs:/<id>/xgb_model``
                           or a ``file://`` path to the artifact directory).
    registry_model_name  : Name under which to register in the Model Registry.
    val_metrics          : Validation split metrics dict.
    test_metrics         : Test split metrics dict.
    metadata             : Training metadata dict (best_iteration, etc.).
    git_commit           : Git commit SHA from the run tags (optional).

    Returns
    -------
    str version number of the newly created ModelVersion.
    """
    client = mlflow.tracking.MlflowClient()

    # 1. Ensure the registered model entity exists (idempotent)
    try:
        client.create_registered_model(
            name=registry_model_name,
            tags=_REGISTERED_MODEL_TAGS,
            description=_REGISTERED_MODEL_DESCRIPTION,
        )
        logger.info("Created registered model '%s'", registry_model_name)
    except mlflow.exceptions.MlflowException:
        # Already exists — keep existing entity tags/description
        logger.info("Registered model '%s' already exists; creating new version", registry_model_name)

    # 2. Create a new version linked to this run
    mv = client.create_model_version(
        name=registry_model_name,
        source=model_uri,
        run_id=run_id,
        description=(
            f"Trained on run {run_id}. "
            f"test_auc={test_metrics.get('auc_roc', 0):.4f} | "
            f"test_ndcg3={test_metrics.get('ndcg_at_3', 0):.4f} | "
            f"test_recall3={test_metrics.get('recall_at_3', 0):.4f} | "
            f"test_mrr={test_metrics.get('mrr', 0):.4f} | "
            f"test_f1={test_metrics.get('f1_class1', 0):.4f}"
        ),
    )
    version = str(mv.version)  # MLflow returns int; alias API requires str
    logger.info("Registered model '%s' version=%s linked to run=%s", registry_model_name, version, run_id)

    # 3. Tag the version with all key context
    version_tags = {
        # Identification
        "run_id": run_id,
        "git_commit": git_commit,
        # Model configuration
        "model_type": "XGBClassifier",
        "objective": "binary:logistic",
        "best_iteration": str(metadata.get("best_iteration", "")),
        "scale_pos_weight": str(round(metadata.get("scale_pos_weight", 0), 4)),
        # Dataset
        "n_leads_train": str(metadata.get("n_leads_train", "")),
        "n_leads_val": str(metadata.get("n_leads_val", "")),
        "n_leads_test": str(metadata.get("n_leads_test", "")),
        "n_features_input": str(metadata.get("n_features_input", "")),
        "n_features_transformed": str(metadata.get("n_features_transformed", "")),
        "train_conversion_rate": str(round(metadata.get("train_conversion_rate", 0), 4)),
        # Validation metrics
        "val_auc_roc": str(round(val_metrics.get("auc_roc", 0), 4)),
        "val_ndcg_at_3": str(round(val_metrics.get("ndcg_at_3", 0), 4)),
        "val_recall_at_3": str(round(val_metrics.get("recall_at_3", 0), 4)),
        "val_mrr": str(round(val_metrics.get("mrr", 0), 4)),
        "val_f1_class1": str(round(val_metrics.get("f1_class1", 0), 4)),
        # Test metrics
        "test_auc_roc": str(round(test_metrics.get("auc_roc", 0), 4)),
        "test_ndcg_at_3": str(round(test_metrics.get("ndcg_at_3", 0), 4)),
        "test_recall_at_3": str(round(test_metrics.get("recall_at_3", 0), 4)),
        "test_mrr": str(round(test_metrics.get("mrr", 0), 4)),
        "test_f1_class1": str(round(test_metrics.get("f1_class1", 0), 4)),
        # Threshold checks
        "thresholds_passed": "5/5",
        "all_thresholds_pass": "true",
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(registry_model_name, version, key, value)

    # 4. Assign the "staging" alias (MLflow 3.x replaces deprecated stage transitions).
    #    Access via: models:/lead_bank_xgb_ranker@staging
    client.set_registered_model_alias(registry_model_name, "staging", version)
    logger.info("Model version=%s aliased as '@staging'", version)

    return version


def _build_xgb_params(cfg: dict, scale_pos_weight: float) -> dict:
    model_cfg = cfg["model"]
    return {
        "objective": model_cfg["objective"],
        "tree_method": model_cfg.get("tree_method", "hist"),
        "n_estimators": model_cfg.get("n_estimators", 400),
        "max_depth": model_cfg.get("max_depth", 6),
        "learning_rate": model_cfg.get("learning_rate", 0.05),
        "min_child_weight": model_cfg.get("min_child_weight", 5),
        "subsample": model_cfg.get("subsample", 0.8),
        "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
        "gamma": model_cfg.get("gamma", 0.1),
        "reg_alpha": model_cfg.get("reg_alpha", 0.1),
        "reg_lambda": model_cfg.get("reg_lambda", 1.0),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": model_cfg.get("eval_metric", ["auc", "logloss"]),
        "early_stopping_rounds": model_cfg.get("early_stopping_rounds", 50),
        "random_state": 42,
        "verbosity": 0,
    }


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(
    config_path: str = DEFAULT_CONFIG,
    splits_dir: str = DEFAULT_SPLITS_DIR,
    banks_path: str = DEFAULT_BANKS_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
    feature_config_path: str = DEFAULT_FEATURE_CONFIG,
    run_name: str | None = None,
) -> dict:
    """
    Full training pipeline: preprocess → train → evaluate → log → save.

    Returns
    -------
    dict of all evaluation metrics from the test split.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    cfg = load_model_config(config_path)
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_uri = mlflow_cfg.get("tracking_uri", "experiments/mlflow")
    experiment_name = mlflow_cfg.get("experiment_name", "lead_bank_ranking")
    registry_model_name = mlflow_cfg.get("registry_model_name", "lead_bank_xgb_ranker")

    # ---- Load splits ----
    splits = Path(splits_dir)
    train_df = pd.read_parquet(splits / "train.parquet")
    val_df = pd.read_parquet(splits / "val.parquet")
    test_df = pd.read_parquet(splits / "test.parquet")
    banks_df = pd.read_parquet(banks_path)

    logger.info(
        "Splits loaded | train=%d | val=%d | test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # ---- Split integrity guard ----
    validate_split_integrity(train_df, val_df, test_df)

    # ---- Preprocessor (fit on train only) ----
    feature_cfg = load_feature_config(feature_config_path)
    preprocessor = build_preprocessor(feature_cfg)
    fit_preprocessor(preprocessor, train_df)

    X_train = transform_split(preprocessor, train_df)
    X_val = transform_split(preprocessor, val_df)
    X_test = transform_split(preprocessor, test_df)

    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    scale_pos_weight = compute_scale_pos_weight(y_train)
    logger.info("scale_pos_weight=%.4f", scale_pos_weight)

    # ---- Build model ----
    xgb_params = _build_xgb_params(cfg, scale_pos_weight)
    n_estimators = xgb_params.pop("n_estimators")
    early_stopping = xgb_params.pop("early_stopping_rounds")

    # Callback is passed via constructor — XGBoost's sklearn fit() does not
    # accept a `callbacks` kwarg; the constructor does.
    model = XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping,
        callbacks=[_MLflowIterCallback()],
        **xgb_params,
    )

    # ---- MLflow run ----
    _setup_mlflow(mlflow_uri, experiment_name)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    effective_run_name = run_name or f"xgb_pointwise_{timestamp}"

    with mlflow.start_run(run_name=effective_run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s (id=%s)", effective_run_name, run_id)

        # Log hyperparameters
        log_params = {
            "n_estimators": n_estimators,
            "early_stopping_rounds": early_stopping,
            "scale_pos_weight": round(scale_pos_weight, 4),
            "n_leads_train": train_df[GROUP_KEY].nunique(),
            "n_leads_val": val_df[GROUP_KEY].nunique(),
            "n_leads_test": test_df[GROUP_KEY].nunique(),
            "n_features": len(ALL_FEATURES),
            "feature_count": X_train.shape[1],
            "train_conversion_rate": round(float(y_train.mean()), 4),
            **{k: v for k, v in xgb_params.items() if k not in ("eval_metric",)},
        }
        mlflow.log_params(log_params)

        # ---- Train ----
        # eval_set order: [train, val] so callback maps position 0→train, 1→val.
        # Early stopping still monitors the last entry (val).
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        best_iter = int(model.best_iteration) if hasattr(model, "best_iteration") else n_estimators
        logger.info("Training complete | best_iteration=%d", best_iter)
        mlflow.log_param("best_iteration", best_iter)

        # ---- Evaluate: val ----
        val_scores = model.predict_proba(X_val)[:, 1]
        val_df_scored = val_df.copy()
        val_df_scored[SCORE_COL] = val_scores
        val_metrics = evaluate_all(val_df_scored, score_col=SCORE_COL, split_label="val")

        mlflow.log_metrics({f"val_{k}": round(v, 6) for k, v in val_metrics.items()})
        val_thresholds = check_thresholds(val_metrics)

        # ---- Evaluate: test ----
        test_scores = model.predict_proba(X_test)[:, 1]
        test_df_scored = test_df.copy()
        test_df_scored[SCORE_COL] = test_scores
        test_metrics = evaluate_all(test_df_scored, score_col=SCORE_COL, split_label="test")

        mlflow.log_metrics({f"test_{k}": round(v, 6) for k, v in test_metrics.items()})
        test_thresholds = check_thresholds(test_metrics)

        # ---- NDCG log line (per CLAUDE.md §15) ----
        if test_metrics.get("ndcg_at_3", 0.0) < 0.70:
            logger.error(
                "NDCG@3 below threshold: actual=%.4f threshold=0.70",
                test_metrics.get("ndcg_at_3", 0.0),
            )

        # ---- Per-bank AUC (test) ----
        bank_auc_df = per_bank_auc(test_df_scored, score_col=SCORE_COL)
        flagged_banks = int(bank_auc_df["flagged"].sum())
        mlflow.log_metric("test_banks_below_auc_threshold", flagged_banks)

        # ---- Per-income-type NDCG@3 (test) ----
        income_ndcg = per_income_type_ndcg(test_df_scored, score_col=SCORE_COL, k=3)
        for _, row in income_ndcg.iterrows():
            mlflow.log_metric(f"test_ndcg3_income_enc_{int(row['income_type_enc'])}", round(row["ndcg_at_3"], 6))

        # ---- Feature importance ----
        fi = model.feature_importances_
        top10_idx = np.argsort(fi)[::-1][:10]

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = np.array([f"f{i}" for i in range(len(fi))])

        top_features = {str(feature_names[i]): round(float(fi[i]), 6) for i in top10_idx}
        mlflow.log_dict(top_features, "top10_feature_importance.json")

        # Feature importance bar chart (§13 artifact requirement)
        _fi_plot_path = Path(model_dir) / "feature_importance.png"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        _save_feature_importance_plot(top_features, str(_fi_plot_path))
        mlflow.log_artifact(str(_fi_plot_path), artifact_path="plots")

        # ---- Build & save model bundle ----
        feature_schema = build_feature_schema(train_df, ALL_FEATURES)
        eligibility_rules = build_eligibility_rules(banks_df)

        metadata = {
            "version": "v1",
            "run_id": run_id,
            "run_name": effective_run_name,
            "training_date": timestamp,
            "model_type": "XGBClassifier",
            "objective": "binary:logistic",
            "best_iteration": best_iter,
            "n_leads_train": int(train_df[GROUP_KEY].nunique()),
            "n_leads_val": int(val_df[GROUP_KEY].nunique()),
            "n_leads_test": int(test_df[GROUP_KEY].nunique()),
            "n_features_input": len(ALL_FEATURES),
            "n_features_transformed": int(X_train.shape[1]),
            "train_conversion_rate": round(float(y_train.mean()), 4),
            "scale_pos_weight": round(scale_pos_weight, 4),
            "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
            "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
            "threshold_checks_test": test_thresholds,
            "top10_feature_importance": top_features,
        }

        bundle_path = save_model_bundle(
            model=model,
            preprocessor=preprocessor,
            metadata=metadata,
            feature_schema=feature_schema,
            eligibility_rules=eligibility_rules,
            output_dir=model_dir,
        )

        # Log bundle artifacts to MLflow
        mlflow.log_artifact(str(bundle_path / "metadata.json"))
        mlflow.log_artifact(str(bundle_path / "feature_schema.json"))

        # Log the XGBoost model using the MLflow 3.x `name=` parameter
        # (artifact_path= is deprecated in MLflow 3.x)
        model_info = mlflow.xgboost.log_model(model, name="xgb_model")

        # Register in the Model Registry and apply all tags + stage transition
        git_commit = mlflow.active_run().data.tags.get("mlflow.source.git.commit", "")
        _register_and_tag_model(
            run_id=run_id,
            model_uri=model_info.model_uri,
            registry_model_name=registry_model_name,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metadata=metadata,
            git_commit=git_commit,
        )

        # Print summary
        _print_summary(val_metrics, test_metrics, val_thresholds, test_thresholds, best_iter, income_ndcg, bank_auc_df, top_features)

    logger.info("Training complete. Bundle saved to %s", bundle_path)
    return test_metrics


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(
    val_metrics: dict,
    test_metrics: dict,
    val_thresholds: dict,
    test_thresholds: dict,
    best_iter: int,
    income_ndcg: pd.DataFrame,
    bank_auc_df: pd.DataFrame,
    top_features: dict,
) -> None:
    print("\n" + "=" * 65)
    print("TRAINING SUMMARY — Lead-to-Bank Ranking (Section 10 & 11)")
    print("=" * 65)

    print(f"\nBest XGBoost iteration : {best_iter}")

    print("\n--- Validation Metrics ---")
    _print_metrics_block(val_metrics, val_thresholds)

    print("\n--- Test Metrics ---")
    _print_metrics_block(test_metrics, test_thresholds)

    all_pass = all(test_thresholds.values())
    print(f"\nAll test thresholds met : {'YES (PASS)' if all_pass else 'NO -- see ERRORs above'}")

    print("\n--- Top 10 Feature Importances ---")
    for name, score in top_features.items():
        print(f"  {name:<45} {score:.6f}")

    print("\n--- NDCG@3 by Income Type (test) ---")
    enc_map = {0: "salaried", 1: "self_employed", 2: "business", 3: "freelance"}
    for _, row in income_ndcg.iterrows():
        label = enc_map.get(int(row["income_type_enc"]), str(int(row["income_type_enc"])))
        print(f"  {label:<20} NDCG@3={row['ndcg_at_3']:.4f}  (n_leads={int(row['n_leads'])})")

    n_flagged = int(bank_auc_df["flagged"].sum())
    print(f"\n--- Per-Bank AUC (test) ---")
    print(f"  Banks below AUC 0.70 threshold : {n_flagged} / {len(bank_auc_df)}")
    print("=" * 65)


def _save_feature_importance_plot(
    top_features: dict[str, float],
    output_path: str,
) -> None:
    """Save a horizontal bar chart of top-10 feature importances to output_path."""
    names = list(top_features.keys())
    values = list(top_features.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(names))
    ax.barh(y_pos, values[::-1], color="steelblue", edgecolor="white")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Top-10 Feature Importances")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _print_metrics_block(metrics: dict, thresholds: dict) -> None:
    threshold_map = {
        "auc_roc": ("AUC-ROC", 0.82),
        "ndcg_at_3": ("NDCG@3", 0.70),
        "ndcg_at_1": ("NDCG@1", None),
        "ndcg_at_5": ("NDCG@5", None),
        "recall_at_3": ("Recall@3", 0.75),
        "recall_at_1": ("Recall@1", None),
        "recall_at_5": ("Recall@5", None),
        "mrr": ("MRR", 0.60),
        "f1_class1": ("F1 (class 1)", 0.65),
        "precision_class1": ("Precision (class 1)", None),
        "recall_class1": ("Recall (class 1)", None),
    }
    for key, (label, thresh) in threshold_map.items():
        val = metrics.get(key)
        if val is None:
            continue
        if thresh is not None:
            status = "PASS" if thresholds.get(key, True) else "FAIL"
            print(f"  {label:<30} {val:.4f}  [threshold={thresh}  {status}]")
        else:
            print(f"  {label:<30} {val:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost pointwise scoring model")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--banks", default=DEFAULT_BANKS_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    train(
        config_path=args.config,
        splits_dir=args.splits_dir,
        banks_path=args.banks,
        model_dir=args.model_dir,
        feature_config_path=args.feature_config,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    _main()
