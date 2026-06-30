import uuid
import json
import logging
import time
import re
from opentelemetry import trace
from db.pool import DatabasePool
from models.pipeline import PipelineConfig

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class PipelineManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PipelineManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True

    async def ensure_db_schema(self):
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            try:
                # Drop unique constraint on name since we have versions (e.g. legal-v1, legal-v2)
                await conn.execute("ALTER TABLE pipelines DROP CONSTRAINT IF EXISTS pipelines_name_key;")
                
                # Add columns for versioning
                await conn.execute("ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 1;")
                await conn.execute("ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'active';")
                await conn.execute("ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS parent_pipeline_id UUID REFERENCES pipelines(id);")
            except Exception as e:
                logger.warning(f"Database migration for pipelines table failed or already applied: {e}")

    def _extract_base_name(self, name: str, version: int) -> str:
        # e.g., legal-v1 -> legal
        suffix = f"-v{version}"
        if name.endswith(suffix):
            return name[:-len(suffix)]
        match = re.search(r"^(.*?)-v\d+$", name)
        if match:
            return match.group(1)
        return name

    async def create_pipeline(self, config: PipelineConfig) -> dict:
        await self.ensure_db_schema()
        
        base_name = config.name
        version = 1
        name = f"{base_name}-v{version}"
        
        # Serialize the config schema to dict/json
        config_dict = config.model_dump()
        # Override the name in config to reflect the versioned name
        config_dict["name"] = name

        pool = await DatabasePool.get_pool()
        
        start_time = time.time()
        with tracer.start_as_current_span("pipeline.create") as span:
            async with pool.acquire() as conn:
                pipeline_id = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO pipelines (id, name, version, config, status, parent_pipeline_id)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6);
                    """,
                    pipeline_id,
                    name,
                    version,
                    json.dumps(config_dict),
                    "active",
                    None
                )
                
                # Fetch created row
                row = await conn.fetchrow(
                    "SELECT id, name, version, config, status, created_at, parent_pipeline_id FROM pipelines WHERE id = $1;",
                    pipeline_id
                )
                
                latency = (time.time() - start_time) * 1000
                span.set_attribute("pipeline_id", str(pipeline_id))
                span.set_attribute("version", version)
                span.set_attribute("latency", latency)
                
                result = dict(row)
                result["config"] = json.loads(result["config"]) if isinstance(result["config"], str) else result["config"]
                return result

    async def get_pipeline(self, pipeline_id: str) -> dict:
        await self.ensure_db_schema()
        
        try:
            pipeline_uuid = uuid.UUID(pipeline_id) if isinstance(pipeline_id, str) else pipeline_id
        except ValueError:
            raise ValueError(f"Invalid pipeline ID format: {pipeline_id}")
            
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, name, version, config, status, created_at, parent_pipeline_id FROM pipelines WHERE id = $1;",
                pipeline_uuid
            )
            if not row:
                return None
                
            result = dict(row)
            result["config"] = json.loads(result["config"]) if isinstance(result["config"], str) else result["config"]
            return result

    async def list_pipelines(self) -> list:
        await self.ensure_db_schema()
        
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, version, config, status, created_at, parent_pipeline_id FROM pipelines WHERE status != 'archived' ORDER BY created_at DESC;"
            )
            
            results = []
            for r in rows:
                res = dict(r)
                res["config"] = json.loads(res["config"]) if isinstance(res["config"], str) else res["config"]
                results.append(res)
            return results

    async def update_pipeline(self, pipeline_id: str, new_config: PipelineConfig) -> dict:
        await self.ensure_db_schema()
        
        try:
            pipeline_uuid = uuid.UUID(pipeline_id) if isinstance(pipeline_id, str) else pipeline_id
        except ValueError:
            raise ValueError(f"Invalid pipeline ID format: {pipeline_id}")

        pool = await DatabasePool.get_pool()
        
        start_time = time.time()
        with tracer.start_as_current_span("pipeline.update") as span:
            async with pool.acquire() as conn:
                # Fetch parent
                parent = await conn.fetchrow(
                    "SELECT name, version, status FROM pipelines WHERE id = $1;",
                    pipeline_uuid
                )
                if not parent:
                    raise ValueError(f"Pipeline with ID {pipeline_id} not found.")
                if parent["status"] == "archived":
                    raise ValueError(f"Cannot update archived pipeline {pipeline_id}.")
                    
                version = parent["version"] + 1
                base_name = self._extract_base_name(parent["name"], parent["version"])
                name = f"{base_name}-v{version}"
                
                config_dict = new_config.model_dump()
                config_dict["name"] = name
                
                new_id = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO pipelines (id, name, version, config, status, parent_pipeline_id)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6);
                    """,
                    new_id,
                    name,
                    version,
                    json.dumps(config_dict),
                    "active",
                    pipeline_uuid
                )
                
                row = await conn.fetchrow(
                    "SELECT id, name, version, config, status, created_at, parent_pipeline_id FROM pipelines WHERE id = $1;",
                    new_id
                )
                
                latency = (time.time() - start_time) * 1000
                span.set_attribute("pipeline_id", str(new_id))
                span.set_attribute("version", version)
                span.set_attribute("latency", latency)
                
                result = dict(row)
                result["config"] = json.loads(result["config"]) if isinstance(result["config"], str) else result["config"]
                return result

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        await self.ensure_db_schema()
        
        try:
            pipeline_uuid = uuid.UUID(pipeline_id) if isinstance(pipeline_id, str) else pipeline_id
        except ValueError:
            return False

        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            # Soft delete: update status to archived
            res = await conn.execute(
                "UPDATE pipelines SET status = $1 WHERE id = $2;",
                "archived",
                pipeline_uuid
            )
            return "UPDATE 1" in res
