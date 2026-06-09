"""
One-time script: register the existing trained model (run 833c79d2...)
in the MLflow Model Registry as lead_bank_xgb_ranker v1.

Run once from the project root:
  python scripts/_register_existing_run.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from src.modeling.trainer import _register_and_tag_model, _setup_mlflow

EXISTING_RUN_ID = "833c79d21f154414b9a7ca41722590a7"
REGISTRY_MODEL_NAME = "lead_bank_xgb_ranker"
MLFLOW_BASE = "experiments/mlflow"
EXPERIMENT_NAME = "lead_bank_ranking"

_setup_mlflow(MLFLOW_BASE, EXPERIMENT_NAME)
client = mlflow.tracking.MlflowClient()

run = client.get_run(EXISTING_RUN_ID)
metrics = run.data.metrics

# The XGBoost model was logged via mlflow.xgboost.log_model(artifact_path="xgb_model")
# In MLflow 3.x this stores it in experiments/mlflow/artifacts/models/<model_id>/
# Retrieve the logged model to get its artifact_location
import sqlite3, urllib.parse
conn = sqlite3.connect(f"{MLFLOW_BASE}/mlflow.db")
cur = conn.cursor()
cur.execute(
    "SELECT model_id, artifact_location FROM logged_models WHERE source_run_id = ?",
    (EXISTING_RUN_ID,)
)
rows = cur.fetchall()
conn.close()

if not rows:
    print("No logged model found for run", EXISTING_RUN_ID)
    sys.exit(1)

model_id, artifact_location = rows[0]
print(f"Found logged model: id={model_id}")
print(f"  artifact_location={artifact_location}")

# Decode %20 -> space for use as a file URI
model_uri = artifact_location

val_metrics = {k.replace("val_", ""): v for k, v in metrics.items() if k.startswith("val_")}
test_metrics = {k.replace("test_", ""): v for k, v in metrics.items() if k.startswith("test_")}

import sqlite3, json
conn = sqlite3.connect(f"{MLFLOW_BASE}/mlflow.db")
cur = conn.cursor()
cur.execute("SELECT key, value FROM params WHERE run_uuid = ?", (EXISTING_RUN_ID,))
params = dict(cur.fetchall())
conn.close()

metadata = {
    "best_iteration": int(params.get("best_iteration", 374)),
    "scale_pos_weight": float(params.get("scale_pos_weight", 8.4676)),
    "n_leads_train": int(params.get("n_leads_train", 7000)),
    "n_leads_val": int(params.get("n_leads_val", 1500)),
    "n_leads_test": int(params.get("n_leads_test", 1500)),
    "n_features_input": int(params.get("n_features", 57)),
    "n_features_transformed": int(params.get("feature_count", 76)),
    "train_conversion_rate": float(params.get("train_conversion_rate", 0.1056)),
}

git_commit = run.data.tags.get("mlflow.source.git.commit", "")

# Patch alias on the existing v1 that was registered with the old stage API
import mlflow as _mlflow
_client = _mlflow.tracking.MlflowClient()
try:
    _client.set_registered_model_alias(REGISTRY_MODEL_NAME, "staging", "1")
    print("Alias '@staging' set on version=1 (overrides deprecated stage)")
except Exception as e:
    print(f"Alias already set or error: {e}")

version = _register_and_tag_model(
    run_id=EXISTING_RUN_ID,
    model_uri=model_uri,
    registry_model_name=REGISTRY_MODEL_NAME,
    val_metrics=val_metrics,
    test_metrics=test_metrics,
    metadata=metadata,
    git_commit=git_commit,
)

print(f"\nRegistered model '{REGISTRY_MODEL_NAME}' version={version}")
print(f"Run ID    : {EXISTING_RUN_ID}")
print(f"Model URI : {model_uri}")
print(f"Stage     : Staging")
