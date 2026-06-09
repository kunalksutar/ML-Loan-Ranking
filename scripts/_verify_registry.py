"""Verify the MLflow Model Registry state."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
mlflow.set_tracking_uri("sqlite:///experiments/mlflow/mlflow.db")
client = mlflow.tracking.MlflowClient()

print("=== Registered Models ===")
for m in client.search_registered_models():
    print(f"  name        : {m.name}")
    print(f"  description : {(m.description or '')[:90]}")
    print(f"  tags        : {dict(m.tags)}")
    print()

print("=== Model Versions ===")
versions = client.search_model_versions("name='lead_bank_xgb_ranker'")
for mv in versions:
    print(f"  version     : {mv.version}")
    print(f"  run_id      : {mv.run_id}")
    print(f"  source      : {mv.source}")
    print(f"  description : {(mv.description or '')[:110]}")
    tags = {k: v for k, v in mv.tags.items() if k in (
        "run_id", "git_commit", "test_auc_roc", "test_ndcg_at_3",
        "test_mrr", "test_recall_at_3", "test_f1_class1",
        "thresholds_passed", "best_iteration", "n_leads_train",
    )}
    print(f"  tags (key subset):")
    for k, v in tags.items():
        print(f"    {k:<30} = {v}")
    print()

print("=== Aliases ===")
rm = client.get_registered_model("lead_bank_xgb_ranker")
print(f"  aliases: {rm.aliases}")
