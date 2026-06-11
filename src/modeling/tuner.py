"""
§12 Hyperparameter Tuning — Optuna + MLflow.

Searches XGBoost hyperparameters by maximising NDCG@3 on a held-out
validation set, using Optuna's TPE sampler with MedianPruner.

Two evaluation modes (controlled by --use-cv):

  default (fast)  — train on train.parquet, score val.parquet → NDCG@3.
                    ~25 s/trial; suitable for interactive tuning runs.

  --use-cv (slow) — 5-fold GroupKFold CV within train.parquet → mean
                    NDCG@3.  Each trial = 5 model fits. Use for final
                    production runs per CLAUDE.md §12 spec.

After optimisation the best hyperparameters are used to retrain a fresh
model on the full training split.  That model is evaluated on val + test,
logged to MLflow with per-iteration curves, and registered in the Model
Registry as the next version with alias @staging.  The previous version
(baseline) receives alias @baseline.

Usage:
  python -m src.modeling.tuner --config configs/model_config.yaml \\
      --n-trials 30 [--use-cv]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import yaml
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features.feature_registry import ALL_FEATURES, GROUP_KEY, TARGET
from src.modeling.evaluator import check_thresholds, evaluate_all
from src.modeling.model_registry import (
    build_eligibility_rules,
    build_feature_schema,
    save_model_bundle,
)
from src.modeling.trainer import (
    _MLflowIterCallback,
    _register_and_tag_model,
    _setup_mlflow,
    _save_feature_importance_plot,
)
from src.preprocessing.pipeline_builder import (
    build_preprocessor,
    fit_preprocessor,
    load_feature_config,
    transform_split,
)
from src.preprocessing.splitting import (
    compute_scale_pos_weight,
    iter_group_kfold_splits,
    validate_split_integrity,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/model_config.yaml"
DEFAULT_SPLITS_DIR = "data/processed/applications_splits"
DEFAULT_BANKS_PATH = "data/raw/banks.parquet"
DEFAULT_MODEL_DIR = "models/v1"
DEFAULT_FEATURE_CONFIG = "configs/feature_config.yaml"
SCORE_COL = "predicted_score"
OPTUNA_SEED = 42


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Hyperparameter sampling
# ---------------------------------------------------------------------------

def _sample_params(trial: optuna.Trial, search_space_cfg: dict) -> dict[str, Any]:
    """
    Sample one hyperparameter set from the Optuna trial, driven entirely
    by the search_space block in model_config.yaml.  Supports int, float,
    and float_log (log-uniform) types.
    """
    params: dict[str, Any] = {}
    for name, spec in search_space_cfg.items():
        kind = spec["type"]
        lo, hi = spec["low"], spec["high"]
        if kind == "int":
            params[name] = trial.suggest_int(name, lo, hi)
        elif kind == "float":
            params[name] = trial.suggest_float(name, lo, hi)
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, lo, hi, log=True)
        else:
            raise ValueError(f"Unknown search_space type '{kind}' for '{name}'")
    return params


# ---------------------------------------------------------------------------
# Trial objective — single val-split (fast, default)
# ---------------------------------------------------------------------------

def _single_split_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_df: pd.DataFrame,
    search_space_cfg: dict,
    model_cfg: dict,
) -> float:
    """
    Objective for one trial: train on X_train, early-stop on X_val,
    return val NDCG@3.  One model fit per trial.
    """
    params = _sample_params(trial, search_space_cfg)
    n_estimators = params.pop("n_estimators", model_cfg.get("n_estimators", 400))
    early_stopping = model_cfg.get("early_stopping_rounds", 50)

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric=["auc", "logloss"],
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping,
        random_state=OPTUNA_SEED,
        verbosity=0,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    scores = model.predict_proba(X_val)[:, 1]
    val_scored = val_df.copy()
    val_scored[SCORE_COL] = scores
    metrics = evaluate_all(val_scored, score_col=SCORE_COL, split_label=f"t{trial.number}")
    return float(metrics.get("ndcg_at_3", 0.0))


# ---------------------------------------------------------------------------
# Trial objective — GroupKFold CV (slow, production-grade)
# ---------------------------------------------------------------------------

def _cv_objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    feature_cfg: dict,
    search_space_cfg: dict,
    model_cfg: dict,
    n_folds: int,
) -> float:
    """
    Objective for one trial: GroupKFold CV on train_df, return mean NDCG@3.
    Fits the preprocessor independently on each fold's training portion to
    prevent leakage of held-out fold statistics (CLAUDE.md §9).
    Reports per-fold NDCG@3 as intermediate Optuna values for pruning.
    """
    params = _sample_params(trial, search_space_cfg)
    n_estimators = params.pop("n_estimators", model_cfg.get("n_estimators", 400))
    early_stopping = model_cfg.get("early_stopping_rounds", 50)

    ndcg3_scores: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        iter_group_kfold_splits(train_df, n_splits=n_folds)
    ):
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        fold_preprocessor = build_preprocessor(feature_cfg)
        fit_preprocessor(fold_preprocessor, fold_train)
        X_fold_tr = transform_split(fold_preprocessor, fold_train)
        X_fold_val = transform_split(fold_preprocessor, fold_val)
        y_fold_tr = fold_train[TARGET].to_numpy()
        y_fold_val = fold_val[TARGET].to_numpy()

        model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            eval_metric=["auc", "logloss"],
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping,
            random_state=OPTUNA_SEED,
            verbosity=0,
            **params,
        )
        model.fit(X_fold_tr, y_fold_tr, eval_set=[(X_fold_val, y_fold_val)], verbose=False)

        scores = model.predict_proba(X_fold_val)[:, 1]
        fold_scored = fold_val.copy()
        fold_scored[SCORE_COL] = scores
        fold_metrics = evaluate_all(
            fold_scored, score_col=SCORE_COL,
            split_label=f"t{trial.number}_f{fold_idx}",
        )
        ndcg3 = float(fold_metrics.get("ndcg_at_3", 0.0))
        ndcg3_scores.append(ndcg3)

        # Report intermediate value; allows MedianPruner to prune bad trials
        trial.report(float(np.mean(ndcg3_scores)), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(ndcg3_scores))


# ---------------------------------------------------------------------------
# Retrain best model on full training data + register
# ---------------------------------------------------------------------------

def _retrain_and_register(
    best_params: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    banks_df: pd.DataFrame,
    feature_cfg: dict,
    model_cfg: dict,
    mlflow_cfg: dict,
    model_dir: str,
    best_cv_ndcg3: float,
    study_run_id: str,
) -> dict[str, float]:
    """
    Retrain a fresh XGBClassifier with best_params on the full training split,
    log with per-iteration learning curves, evaluate on val + test, and register
    in the Model Registry.

    Sets @baseline on the previous version and @staging on the new version.

    Returns the test-split metrics dict.
    """
    registry_model_name = mlflow_cfg.get("registry_model_name", "lead_bank_xgb_ranker")
    experiment_name = mlflow_cfg.get("experiment_name", "lead_bank_ranking")

    # ---- Preprocessing (fit on train only) ----
    preprocessor = build_preprocessor(feature_cfg)
    fit_preprocessor(preprocessor, train_df)
    X_train = transform_split(preprocessor, train_df)
    X_val = transform_split(preprocessor, val_df)
    X_test = transform_split(preprocessor, test_df)
    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    scale_pos_weight = compute_scale_pos_weight(y_train)

    # ---- Build model with best hyperparameters ----
    params = dict(best_params)
    n_estimators = int(params.pop("n_estimators", model_cfg.get("n_estimators", 400)))
    early_stopping = model_cfg.get("early_stopping_rounds", 50)
    # Use data-derived scale_pos_weight if not in tuned params
    if "scale_pos_weight" not in params:
        params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric=["auc", "logloss"],
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping,
        callbacks=[_MLflowIterCallback()],
        random_state=OPTUNA_SEED,
        verbosity=0,
        **params,
    )

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"xgb_tuned_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow retrain run started: %s (id=%s)", run_name, run_id)

        # Log all hyperparameters + dataset context
        mlflow.log_params({
            "n_estimators": n_estimators,
            "early_stopping_rounds": early_stopping,
            "scale_pos_weight": round(float(params.get("scale_pos_weight", scale_pos_weight)), 4),
            "n_leads_train": train_df[GROUP_KEY].nunique(),
            "n_leads_val": val_df[GROUP_KEY].nunique(),
            "n_leads_test": test_df[GROUP_KEY].nunique(),
            "n_features": len(ALL_FEATURES),
            "feature_count": X_train.shape[1],
            "train_conversion_rate": round(float(y_train.mean()), 4),
            "tuning_study_run_id": study_run_id,
            "best_cv_ndcg3": round(float(best_cv_ndcg3), 6),
            **{k: v for k, v in params.items() if k not in ("scale_pos_weight",)},
        })
        mlflow.set_tag("model_variant", "tuned")

        # ---- Train ----
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        best_iter = int(model.best_iteration) if hasattr(model, "best_iteration") else n_estimators
        mlflow.log_param("best_iteration", best_iter)
        logger.info("Retrain complete | best_iteration=%d", best_iter)

        # ---- Evaluate ----
        val_scores = model.predict_proba(X_val)[:, 1]
        val_scored = val_df.copy()
        val_scored[SCORE_COL] = val_scores
        val_metrics = evaluate_all(val_scored, score_col=SCORE_COL, split_label="val")
        mlflow.log_metrics({f"val_{k}": round(v, 6) for k, v in val_metrics.items()})
        check_thresholds(val_metrics)

        test_scores = model.predict_proba(X_test)[:, 1]
        test_scored = test_df.copy()
        test_scored[SCORE_COL] = test_scores
        test_metrics = evaluate_all(test_scored, score_col=SCORE_COL, split_label="test")
        mlflow.log_metrics({f"test_{k}": round(v, 6) for k, v in test_metrics.items()})
        test_thresholds = check_thresholds(test_metrics)

        if test_metrics.get("ndcg_at_3", 0.0) < 0.70:
            logger.error(
                "NDCG@3 below threshold: actual=%.4f threshold=0.70",
                test_metrics["ndcg_at_3"],
            )

        # ---- Feature importance ----
        fi = model.feature_importances_
        top10_idx = np.argsort(fi)[::-1][:10]
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = np.array([f"f{i}" for i in range(len(fi))])
        top_features = {str(feature_names[i]): round(float(fi[i]), 6) for i in top10_idx}
        mlflow.log_dict(top_features, "top10_feature_importance.json")

        fi_plot_path = Path(model_dir) / "feature_importance_tuned.png"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        _save_feature_importance_plot(top_features, str(fi_plot_path))
        mlflow.log_artifact(str(fi_plot_path), artifact_path="plots")

        # ---- Save bundle ----
        feature_schema = build_feature_schema(train_df, ALL_FEATURES)
        eligibility_rules = build_eligibility_rules(banks_df)
        metadata = {
            "version": "v1",
            "run_id": run_id,
            "run_name": run_name,
            "training_date": timestamp,
            "model_type": "XGBClassifier",
            "objective": "binary:logistic",
            "best_iteration": best_iter,
            "tuned": True,
            "tuning_study_run_id": study_run_id,
            "best_cv_ndcg3": round(float(best_cv_ndcg3), 6),
            "n_leads_train": int(train_df[GROUP_KEY].nunique()),
            "n_leads_val": int(val_df[GROUP_KEY].nunique()),
            "n_leads_test": int(test_df[GROUP_KEY].nunique()),
            "n_features_input": len(ALL_FEATURES),
            "n_features_transformed": int(X_train.shape[1]),
            "train_conversion_rate": round(float(y_train.mean()), 4),
            "scale_pos_weight": round(float(params.get("scale_pos_weight", scale_pos_weight)), 4),
            "best_params": best_params,
            "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
            "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
            "threshold_checks_test": test_thresholds,
        }
        bundle_path = save_model_bundle(
            model=model,
            preprocessor=preprocessor,
            metadata=metadata,
            feature_schema=feature_schema,
            eligibility_rules=eligibility_rules,
            output_dir=model_dir,
        )
        mlflow.log_artifact(str(bundle_path / "metadata.json"))
        mlflow.log_artifact(str(bundle_path / "feature_schema.json"))

        model_info = mlflow.xgboost.log_model(model, name="xgb_model")

        git_commit = mlflow.active_run().data.tags.get("mlflow.source.git.commit", "")
        new_version = _register_and_tag_model(
            run_id=run_id,
            model_uri=model_info.model_uri,
            registry_model_name=registry_model_name,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metadata=metadata,
            git_commit=git_commit,
        )

        # Tag additional tuning context on the new version
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(
            registry_model_name, new_version, "model_variant", "tuned",
        )
        client.set_model_version_tag(
            registry_model_name, new_version, "tuning_study_run_id", study_run_id,
        )
        client.set_model_version_tag(
            registry_model_name, new_version, "best_cv_ndcg3",
            str(round(float(best_cv_ndcg3), 6)),
        )

        # Set @baseline on the previous version, @staging stays on new version
        versions = client.search_model_versions(f"name='{registry_model_name}'")
        prev_versions = [v for v in versions if str(v.version) != str(new_version)]
        if prev_versions:
            # Most recent prior version gets @baseline
            latest_prev = max(prev_versions, key=lambda v: int(v.version))
            try:
                client.set_registered_model_alias(
                    registry_model_name, "baseline", str(latest_prev.version),
                )
                logger.info(
                    "Alias @baseline set on version=%s", latest_prev.version,
                )
            except Exception as e:
                logger.warning("Could not set @baseline alias: %s", e)

        logger.info(
            "Tuned model registered | version=%s | @staging | "
            "test_ndcg3=%.4f | test_auc=%.4f",
            new_version,
            test_metrics.get("ndcg_at_3", 0.0),
            test_metrics.get("auc_roc", 0.0),
        )

    return test_metrics


# ---------------------------------------------------------------------------
# Study visualisation helper
# ---------------------------------------------------------------------------

def _save_study_plots(
    study: optuna.Study,
    output_dir: str,
) -> list[str]:
    """
    Save Optuna study summary plots to output_dir.
    Returns list of saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        return saved

    # 1. Optimisation history
    trial_nums = [t.number for t in completed]
    trial_vals = [t.value for t in completed]
    best_so_far = np.maximum.accumulate(trial_vals)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(trial_nums, trial_vals, alpha=0.5, label="Trial NDCG@3")
    ax.plot(trial_nums, best_so_far, color="red", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Val NDCG@3")
    ax.set_title("Optuna optimisation history")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    path = out / "optuna_history.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))

    # 2. Hyperparameter vs NDCG@3 scatter (one subplot per param)
    param_names = list(study.best_params.keys())
    ncols = min(3, len(param_names))
    nrows = (len(param_names) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, pname in enumerate(param_names):
        ax = axes[idx // ncols][idx % ncols]
        xs = [t.params.get(pname) for t in completed if pname in t.params]
        ys = [t.value for t in completed if pname in t.params]
        ax.scatter(xs, ys, alpha=0.6, s=20)
        ax.axvline(study.best_params[pname], color="red", linestyle="--", linewidth=1)
        ax.set_xlabel(pname)
        ax.set_ylabel("NDCG@3")
        ax.set_title(pname)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for idx in range(len(param_names), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle("Hyperparameter importance — NDCG@3 vs param value", y=1.01)
    plt.tight_layout()
    path2 = out / "optuna_param_scatter.png"
    fig.savefig(str(path2), dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path2))

    return saved


# ---------------------------------------------------------------------------
# Main tuning entry point
# ---------------------------------------------------------------------------

def tune(
    config_path: str = DEFAULT_CONFIG,
    n_trials: int = 30,
    splits_dir: str = DEFAULT_SPLITS_DIR,
    banks_path: str = DEFAULT_BANKS_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
    feature_config_path: str = DEFAULT_FEATURE_CONFIG,
    use_cv: bool = False,
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter search and retrain the best model.

    Parameters
    ----------
    config_path          : Path to model_config.yaml.
    n_trials             : Number of Optuna trials to run.
    splits_dir           : Directory containing train/val/test parquet files.
    banks_path           : Path to banks.parquet.
    model_dir            : Directory for saving the best model bundle.
    feature_config_path  : Path to feature_config.yaml.
    use_cv               : If True, use GroupKFold CV within train.parquet
                           (5-fold, ~5× slower per trial).

    Returns
    -------
    dict with keys: best_params, best_ndcg3, val_metrics, test_metrics
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    cfg = _load_config(config_path)
    mlflow_cfg = cfg.get("mlflow", {})
    tuning_cfg = cfg.get("tuning", {})
    model_cfg = cfg.get("model", {})

    mlflow_uri = mlflow_cfg.get("tracking_uri", "experiments/mlflow")
    experiment_name = mlflow_cfg.get("experiment_name", "lead_bank_ranking")
    registry_model_name = mlflow_cfg.get("registry_model_name", "lead_bank_xgb_ranker")
    search_space_cfg = tuning_cfg.get("search_space", {})
    n_folds = tuning_cfg.get("cv_n_splits", 5) if use_cv else 1

    # ---- Load data ----
    splits = Path(splits_dir)
    train_df = pd.read_parquet(splits / "train.parquet")
    val_df = pd.read_parquet(splits / "val.parquet")
    test_df = pd.read_parquet(splits / "test.parquet")
    banks_df = pd.read_parquet(banks_path)
    validate_split_integrity(train_df, val_df, test_df)

    logger.info(
        "Data loaded | train=%d rows | val=%d rows | test=%d rows",
        len(train_df), len(val_df), len(test_df),
    )

    feature_cfg = load_feature_config(feature_config_path)

    # ---- Preprocess (single fit for fast mode; per-fold fit handled in cv_objective) ----
    if not use_cv:
        preprocessor = build_preprocessor(feature_cfg)
        fit_preprocessor(preprocessor, train_df)
        X_train = transform_split(preprocessor, train_df)
        X_val = transform_split(preprocessor, val_df)
        y_train = train_df[TARGET].to_numpy()
        y_val = val_df[TARGET].to_numpy()
        scale_pos_weight = compute_scale_pos_weight(y_train)
    else:
        scale_pos_weight = compute_scale_pos_weight(train_df[TARGET].to_numpy())
        X_train = X_val = y_train = y_val = None  # not used in CV mode

    # ---- MLflow setup ----
    _setup_mlflow(mlflow_uri, experiment_name)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Silence Optuna's verbose per-trial logging (MLflow captures the detail)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ---- Optuna study ----
    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    # MedianPruner prunes a trial if its intermediate value falls below the
    # median of completed trials after the first warm-up phase (n_startup_trials=5)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner if use_cv else optuna.pruners.NopPruner(),
        study_name=f"lead_bank_tuning_{timestamp}",
    )

    study_run_name = f"tuning_study_{timestamp}"

    with mlflow.start_run(run_name=study_run_name) as study_run:
        study_run_id = study_run.info.run_id
        mlflow.log_params({
            "n_trials": n_trials,
            "use_cv": use_cv,
            "cv_n_folds": n_folds if use_cv else "N/A",
            "sampler": "TPE",
            "pruner": "MedianPruner" if use_cv else "NopPruner",
            "optuna_seed": OPTUNA_SEED,
            "optimize_metric": "val_ndcg_3",
        })
        mlflow.set_tag("run_type", "tuning_study")

        # ---- Objective closure capturing MLflow parent run context ----
        def objective(trial: optuna.Trial) -> float:
            trial_run_name = f"trial_{trial.number:03d}"

            with mlflow.start_run(run_name=trial_run_name, nested=True) as trial_run:
                # Log sampled hyperparameters immediately
                sampled = _sample_params(trial, search_space_cfg)
                mlflow.log_params(sampled)
                mlflow.log_param("trial_number", trial.number)
                mlflow.set_tag("run_type", "tuning_trial")

                if use_cv:
                    ndcg3 = _cv_objective(
                        trial, train_df, feature_cfg,
                        search_space_cfg, model_cfg, n_folds,
                    )
                else:
                    ndcg3 = _single_split_objective(
                        trial, X_train, y_train, X_val, y_val, val_df,
                        search_space_cfg, model_cfg,
                    )

                mlflow.log_metric("cv_ndcg3", round(ndcg3, 6))
                mlflow.set_tag("trial_state", "COMPLETE")
                return ndcg3

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # ---- Log study summary to parent run ----
        best_trial = study.best_trial
        best_ndcg3 = float(study.best_value)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_ndcg3", round(best_ndcg3, 6))
        mlflow.log_metric("n_completed_trials", sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ))
        mlflow.log_metric("n_pruned_trials", sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ))
        mlflow.set_tag("best_trial_number", str(best_trial.number))

        # Persist best params as JSON artifact
        best_params_path = Path(model_dir) / "best_hyperparams.json"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        with open(best_params_path, "w") as f:
            json.dump({"best_params": study.best_params, "best_cv_ndcg3": best_ndcg3}, f, indent=2)
        mlflow.log_artifact(str(best_params_path))

        # Study visualisations
        plot_paths = _save_study_plots(study, str(Path(model_dir) / "study_plots"))
        for p in plot_paths:
            mlflow.log_artifact(p, artifact_path="study_plots")

        logger.info(
            "Tuning complete | best_trial=%d | best_cv_ndcg3=%.4f | params=%s",
            best_trial.number, best_ndcg3, study.best_params,
        )

    # ---- Retrain best model on full train, register ----
    test_metrics = _retrain_and_register(
        best_params=study.best_params,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        banks_df=banks_df,
        feature_cfg=feature_cfg,
        model_cfg=model_cfg,
        mlflow_cfg=mlflow_cfg,
        model_dir=model_dir,
        best_cv_ndcg3=best_ndcg3,
        study_run_id=study_run_id,
    )

    return {
        "best_params": study.best_params,
        "best_ndcg3": best_ndcg3,
        "val_metrics": None,   # populated in _retrain_and_register logs
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Lead-to-Bank Ranking (§12)"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--banks", default=DEFAULT_BANKS_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument(
        "--use-cv", action="store_true",
        help="Use 5-fold GroupKFold CV (production-grade, ~5x slower per trial)",
    )
    args = parser.parse_args()

    result = tune(
        config_path=args.config,
        n_trials=args.n_trials,
        splits_dir=args.splits_dir,
        banks_path=args.banks,
        model_dir=args.model_dir,
        feature_config_path=args.feature_config,
        use_cv=args.use_cv,
    )

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Best CV NDCG@3 : {result['best_ndcg3']:.4f}")
    print(f"Best params:")
    for k, v in result["best_params"].items():
        print(f"  {k:<25} = {v}")
    if result.get("test_metrics"):
        print("\nFinal test metrics (tuned model):")
        for k, v in result["test_metrics"].items():
            print(f"  {k:<35} {v:.4f}")


if __name__ == "__main__":
    _main()
