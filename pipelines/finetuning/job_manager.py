import uuid
import logging
import asyncio
import time
import json
from typing import List, Dict, Any, Optional

try:
    from db.pool import DatabasePool
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
    from db.pool import DatabasePool

from .extractor import TrainingDataExtractor
from .validator import TrainingDataValidator
from .tracker import FineTuneTracker
from .registry import ModelRegistry

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# ARQ task definition for simulation
async def simulate_finetune_job(ctx, job_id: str):
    manager = FineTuneJobManager()
    await manager.simulate_finetune_job_loop(job_id)

class FineTuneJobManager:
    def __init__(self):
        self.extractor = TrainingDataExtractor()
        self.validator = TrainingDataValidator()
        self.tracker = FineTuneTracker()
        self.registry = ModelRegistry()

    async def submit_mock_finetune_job(self, base_model: str, task_type: str) -> str:
        start_time = time.time()
        with tracer.start_as_current_span("finetune.submit") as span:
            span.set_attribute("base_model", base_model)
            span.set_attribute("task_type", task_type)

            # 1. Extract Training Pairs
            pairs = await self.extractor.extract_pairs()
            if not pairs:
                raise ValueError("No training pairs found in database for extraction.")

            # 2. Validate Pairs
            validated_pairs = []
            for p in pairs:
                is_valid, reason = self.validator.validate_pair(p)
                if is_valid:
                    validated_pairs.append(p)
                else:
                    logger.info(f"Skipping pair {p.get('id')} due to validation failure: {reason}")

            if not validated_pairs:
                raise ValueError("No valid training pairs remained after validation filtering.")

            # 3. Format as messages
            formatted_messages = [self.extractor.format_as_chat_messages(p) for p in validated_pairs]

            # 4. Generate Job ID & Export to JSONL
            job_db_id = uuid.uuid4()
            provider_job_id = f"ft-job-{job_db_id}"
            
            # Export valid training data
            jsonl_file = self.tracker.export_to_jsonl(formatted_messages, provider_job_id)

            # 5. Start MLflow tracking run
            avg_score = sum(p["quality_score"] for p in validated_pairs) / len(validated_pairs)
            date_range = "N/A" # Simple default
            mlflow_run_id = self.tracker.start_run(run_name=provider_job_id)
            
            # Log params & JSONL file to MLflow
            self.tracker.log_training_params(mlflow_run_id, base_model, validated_pairs, avg_score, date_range)
            self.tracker.log_jsonl_artifact(mlflow_run_id, jsonl_file)

            # 6. Insert fine-tune job record in DB
            pool = await DatabasePool.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO finetune_jobs (id, provider_job_id, base_model, status, mlflow_run_id, training_pair_count)
                    VALUES ($1, $2, $3, $4, $5, $6);
                    """,
                    job_db_id,
                    provider_job_id,
                    base_model,
                    "queued",
                    mlflow_run_id,
                    len(validated_pairs)
                )

                # Mark selected training pairs as included in this job
                pair_uuids = [uuid.UUID(p["id"]) for p in validated_pairs]
                await conn.execute(
                    """
                    UPDATE training_pairs
                    SET included_in_job = $1
                    WHERE id = ANY($2);
                    """,
                    job_db_id,
                    pair_uuids
                )

            # 7. Enqueue ARQ task to simulate transitions
            try:
                from arq import create_pool
                from arq.connections import RedisSettings
                # Check for password authentication in redis url
                try:
                    from backend.config import settings
                    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
                except Exception:
                    redis_settings = RedisSettings(host="localhost", port=6379, password="redis123")
                
                redis_pool = await create_pool(redis_settings)
                await redis_pool.enqueue_job("simulate_finetune_job", str(job_db_id))
                logger.info(f"Enqueued fine-tune simulation task {provider_job_id} in ARQ.")
            except Exception as e:
                logger.warning(f"Could not enqueue simulation in ARQ queue ({e}). Running inline in asyncio background task.")
                # Run in background via asyncio
                asyncio.create_task(self.simulate_finetune_job_loop(str(job_db_id), task_type))

            latency = (time.time() - start_time) * 1000
            span.set_attribute("latency", latency)
            span.set_attribute("provider_job_id", provider_job_id)
            
            return provider_job_id

    async def simulate_finetune_job_loop(self, job_id: str, task_type: str = "legal"):
        # Simulated states: queued -> running -> training -> validating -> succeeded
        statuses = ["queued", "running", "training", "validating", "succeeded"]
        
        pool = await DatabasePool.get_pool()
        
        # Load run details to retrieve mlflow_run_id and training_pair_count
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT mlflow_run_id, training_pair_count, base_model FROM finetune_jobs WHERE id = $1 or provider_job_id = $2;",
                uuid.UUID(job_id) if len(job_id) == 36 else None,
                job_id
            )
        
        if not row:
            logger.error(f"Cannot find finetune job {job_id} in DB.")
            return

        db_id = uuid.UUID(job_id) if len(job_id) == 36 else None
        mlflow_run_id = row["mlflow_run_id"]
        pair_count = row["training_pair_count"] or 0
        base_model = row["base_model"]
        
        for status in statuses:
            # Update status in PostgreSQL
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE finetune_jobs SET status = $1 WHERE id = $2 or provider_job_id = $3;",
                    status,
                    db_id,
                    job_id
                )
            
            logger.info(f"Finetune job {job_id} status updated to: {status}")
            
            # Brief sleep between states to simulate transitions
            if status != "succeeded":
                await asyncio.sleep(0.2)

        # Job completed successfully - finalize tracking and registry
        with tracer.start_as_current_span("finetune.register") as span:
            span.set_attribute("provider_job_id", job_id)
            
            # Model naming logic, e.g. finetuned-legal-v1
            model_name = f"finetuned-{task_type}-v1"
            span.set_attribute("registered_model", model_name)
            
            # 1. Log completion metrics to MLflow
            training_loss = 0.05
            validation_loss = 0.06
            training_token_count = pair_count * 150 # estimate 150 tokens per conversation
            
            self.tracker.log_metrics(mlflow_run_id, training_loss, validation_loss, training_token_count)
            self.tracker.end_run()

            # 2. Register model in MLflow Model Registry
            self.registry.register_model_mlflow(mlflow_run_id, model_name)

            # 3. Register model in Redis router configurations
            await self.registry.register_model_redis(model_name, task_type)

            # 4. Save final completion metrics to PostgreSQL
            metrics = {
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "training_token_count": training_token_count
            }
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE finetune_jobs
                    SET completed_at = NOW(), metrics = $1::jsonb
                    WHERE id = $2 or provider_job_id = $3;
                    """,
                    json.dumps(metrics),
                    db_id,
                    job_id
                )
