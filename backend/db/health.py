import logging
import asyncpg
import redis.asyncio as redis
import httpx
from config import settings
from db.pool import DatabasePool
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

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
        redis_url = settings.REDIS_URL
        if "redis:6379" in redis_url:
            redis_url = redis_url.replace("redis:6379", "localhost:6379")
        r = redis.from_url(redis_url)
        await r.ping()
        await r.aclose()
        return True
    except Exception:
        return False

async def check_mlflow() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.MLFLOW_TRACKING_URI}/health")
            return response.status_code == 200
    except Exception:
        return False

async def check_health_extended() -> dict:
    with tracer.start_as_current_span("resilience.health") as span:
        postgres_healthy = await check_postgres()
        redis_healthy = await check_redis()
        mlflow_healthy = await check_mlflow()

        checks = {
            "postgres": {"status": "ok" if postgres_healthy else "unhealthy"},
            "redis": {"status": "ok" if redis_healthy else "unhealthy"},
            "mlflow": {"status": "ok" if mlflow_healthy else "unhealthy"},
            "circuit_breakers": {},
            "queue_depth": {"status": "ok", "depth": 0},
            "worker_count": {"status": "ok", "count": 1}
        }

        any_circuit_open = False
        
        if redis_healthy:
            try:
                redis_url = settings.REDIS_URL
                if "redis:6379" in redis_url:
                    redis_url = redis_url.replace("redis:6379", "localhost:6379")
                r = redis.from_url(redis_url, decode_responses=True)
                
                # Fetch queue depth from queue:ingest
                depth = await r.llen("queue:ingest") or 0
                checks["queue_depth"] = {"status": "ok", "depth": depth}
                
                # Fetch arq active workers (default to 1 if none found)
                workers = await r.scard("arq:queue:active-workers") or 0
                checks["worker_count"] = {"status": "ok", "count": max(1, workers)}
                
                # Retrieve all circuit breaker states stored in Redis
                cb_keys = await r.keys("circuit:*:state")
                for key in cb_keys:
                    name = key.split(":")[1]
                    state = await r.get(key)
                    checks["circuit_breakers"][name] = state
                    if state == "OPEN":
                        any_circuit_open = True
                        
                await r.aclose()
            except Exception as e:
                logger.warning(f"Error querying extended health details from Redis: {e}")

        # Set status classification
        if not postgres_healthy or not redis_healthy:
            status = "critical"
        elif any_circuit_open:
            status = "degraded"
        else:
            status = "ok"

        span.set_attribute("health.status", status)
        return {
            "status": status,
            "checks": checks
        }
