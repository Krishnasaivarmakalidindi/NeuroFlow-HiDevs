import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from db.health import check_postgres, check_redis, check_mlflow

from db.pool import DatabasePool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await DatabasePool.get_pool()
    yield
    # Shutdown
    await DatabasePool.close()

app = FastAPI(lifespan=lifespan)

from api.ingest import router as ingest_router
from api.query import router as query_router
from api.feedback import router as feedback_router
from api.pipelines import router as pipelines_router
from api.compare import router as compare_router

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(feedback_router)
app.include_router(pipelines_router)
app.include_router(compare_router)

FastAPIInstrumentor.instrument_app(app)


@app.get("/health")
async def health():
    postgres_ok = await check_postgres()
    redis_ok = await check_redis()
    mlflow_ok = await check_mlflow()
    
    status = "ok" if (postgres_ok and redis_ok and mlflow_ok) else "error"
    return {
        "status": status,
        "checks": {
            "postgres": postgres_ok,
            "redis": redis_ok,
            "mlflow": mlflow_ok
        }
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
