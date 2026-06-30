import os
import json
import logging
from typing import List, Dict, Any

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class FineTuneTracker:
    def __init__(self):
        self.mlflow_available = False
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow_available = True
        except ImportError:
            logger.warning("MLflow not available, tracking will be mocked.")

    def export_to_jsonl(self, messages_list: List[Dict[str, Any]], job_id: str, is_dpo: bool = False) -> str:
        with tracer.start_as_current_span("finetune.track.export") as span:
            os.makedirs("training_data", exist_ok=True)
            suffix = "_dpo" if is_dpo else ""
            file_path = f"training_data/{job_id}{suffix}.jsonl"
            
            with open(file_path, "w", encoding="utf-8") as f:
                for item in messages_list:
                    f.write(json.dumps(item) + "\n")
            
            span.set_attribute("file_path", file_path)
            span.set_attribute("message_count", len(messages_list))
            return file_path

    def start_run(self, run_name: str) -> str:
        with tracer.start_as_current_span("finetune.track.start_run") as span:
            if self.mlflow_available:
                try:
                    # Resolve MLflow tracking URI from configuration settings
                    try:
                        from backend.config import settings
                        self.mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
                    except Exception:
                        pass
                    
                    self.mlflow.set_experiment("neuroflow-finetuning")
                    run = self.mlflow.start_run(run_name=run_name)
                    run_id = run.info.run_id
                    span.set_attribute("mlflow_run_id", run_id)
                    return run_id
                except Exception as e:
                    logger.warning(f"Failed to start MLflow run: {e}")
            return "mock_mlflow_run_id"

    def log_training_params(self, run_id: str, base_model: str, pairs: list, avg_score: float, date_range: str):
        with tracer.start_as_current_span("finetune.track.log_params") as span:
            if self.mlflow_available and run_id != "mock_mlflow_run_id":
                try:
                    self.mlflow.log_params({
                        "base_model": base_model,
                        "training_pair_count": len(pairs),
                        "avg_quality_score": avg_score,
                        "date_range": date_range
                    })
                except Exception as e:
                    logger.warning(f"Failed to log MLflow parameters: {e}")

    def log_jsonl_artifact(self, run_id: str, file_path: str):
        with tracer.start_as_current_span("finetune.track.log_artifact") as span:
            if self.mlflow_available and run_id != "mock_mlflow_run_id":
                try:
                    self.mlflow.log_artifact(file_path)
                except Exception as e:
                    logger.warning(f"Failed to log MLflow artifact: {e}")

    def log_metrics(self, run_id: str, training_loss: float, validation_loss: float, training_token_count: int):
        with tracer.start_as_current_span("finetune.track.log_metrics") as span:
            if self.mlflow_available and run_id != "mock_mlflow_run_id":
                try:
                    self.mlflow.log_metrics({
                        "training_loss": training_loss,
                        "validation_loss": validation_loss,
                        "training_token_count": training_token_count
                    })
                except Exception as e:
                    logger.warning(f"Failed to log MLflow metrics: {e}")

    def end_run(self):
        if self.mlflow_available:
            try:
                self.mlflow.end_run()
            except Exception:
                pass
