"""
Horo RAG — FastAPI entry point
Run: uvicorn main:app --reload --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.database import close_pool, init_pool, init_schema
from routers.routers import chat_router, files_router, prompts_router, upload_router
from routers.conversations import conversations_router

logging.basicConfig(level=get_settings().log_level)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Horo RAG API…")
    pool = await init_pool()
    await init_schema(pool)
    log.info("DB pool ready, schema initialised.")
    yield
    await close_pool()
    log.info("Shutdown complete.")


cfg = get_settings()

app = FastAPI(
    title="Horo RAG API",
    version="1.0.0",
    description="Private knowledge-base RAG for founders — Azure OpenAI + pgvector + Claude",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )

app.include_router(upload_router)
app.include_router(files_router)
app.include_router(chat_router)
app.include_router(prompts_router)
app.include_router(conversations_router)


@app.get("/health", tags=["meta"])
async def health():
    return {
        "status": "ok",
        "chat_deployment": cfg.azure_chat_deployment,
        "embed_deployment": cfg.azure_embed_deployment,
        "sim_threshold": cfg.sim_threshold,
    }