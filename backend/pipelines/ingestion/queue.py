import os
from arq import create_pool
from arq.connections import RedisSettings

async def get_redis_pool():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    # Parse standard redis URL if needed, or just use defaults for local
    return await create_pool(RedisSettings())

async def enqueue_document_job(document_id: str, file_path: str, source_type: str):
    redis = await get_redis_pool()
    await redis.enqueue_job(
        "process_document",
        {
            "document_id": document_id,
            "file_path": file_path,
            "source_type": source_type,
        },
    )
