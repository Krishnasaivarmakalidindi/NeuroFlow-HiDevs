import json
import logging
import redis.asyncio as redis
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class ModelRegistry:
    def __init__(self, redis_url: str = "redis://:redis123@localhost:6379"):
        self.redis_url = redis_url

    async def register_model_redis(self, model_name: str, task_type: str):
        with tracer.start_as_current_span("finetune.register.redis") as span:
            span.set_attribute("model", model_name)
            span.set_attribute("task_type", task_type)

            try:
                redis_client = redis.from_url(self.redis_url, decode_responses=True)
                models_json = await redis_client.get("router:models")
                
                # Default models fallback to ensure registry has seed records if empty
                models = json.loads(models_json) if models_json else [
                    {"model": "gpt-4o-mini", "cost": 0.00015, "fine_tuned": False},
                    {"model": "gpt-4o", "cost": 0.0025, "is_judge": True},
                    {"model": "llama-3.3-70b-versatile", "cost": 0.00059, "fine_tuned": False}
                ]

                new_model = {
                    "model": model_name,
                    "task_type": task_type,
                    "fine_tuned": True,
                    "cost": 0.00010
                }

                # Remove any existing version of this model and append the updated details
                models = [m for m in models if m.get("model") != model_name]
                models.append(new_model)

                await redis_client.set("router:models", json.dumps(models))
                await redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to register model in Redis: {e}")

    def register_model_mlflow(self, run_id: str, model_name: str):
        with tracer.start_as_current_span("finetune.register.mlflow") as span:
            span.set_attribute("mlflow_run_id", run_id)
            span.set_attribute("model", model_name)

            try:
                import mlflow
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/model",
                    name=model_name
                )
            except Exception as e:
                logger.warning(f"Failed to register model in MLflow registry: {e}")
