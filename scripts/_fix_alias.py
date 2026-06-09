"""One-time fix: set @staging alias to version 2 (the latest auto-registered version)."""
import mlflow
mlflow.set_tracking_uri("sqlite:///experiments/mlflow/mlflow.db")
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias("lead_bank_xgb_ranker", "staging", "2")
rm = client.get_registered_model("lead_bank_xgb_ranker")
print(f"Aliases after fix: {rm.aliases}")
