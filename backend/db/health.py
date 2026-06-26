import asyncpg
import redis.asyncio as redis
import httpx
from config import settings

from db.pool import DatabasePool

async def check_postgres() -> bool:
    try:
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:
        return False

async def check_redis() -> bool:
    try:
        r = redis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.aclose()
        return True
    except Exception:
        return False

async def check_mlflow() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            # mlflow usually hosts a health endpoint or just checking the base URL
            response = await client.get(f"{settings.MLFLOW_TRACKING_URI}/health")
            return response.status_code == 200
    except Exception:
        return False
