import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from db.health import check_postgres, check_redis, check_mlflow, check_health_extended

from db.pool import DatabasePool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await DatabasePool.get_pool()
    yield
    # Shutdown
    await DatabasePool.close()

app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.ingest import router as ingest_router
from api.query import router as query_router
from api.feedback import router as feedback_router
from api.pipelines import router as pipelines_router
from api.compare import router as compare_router
from api.finetune import router as finetune_router
from api.evaluations import router as evaluations_router

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(feedback_router)
app.include_router(pipelines_router)
app.include_router(compare_router)
app.include_router(finetune_router)
app.include_router(evaluations_router)

FastAPIInstrumentor.instrument_app(app)


@app.get("/health")
async def health():
    return await check_health_extended()

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
