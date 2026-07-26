#!/usr/bin/env python3
"""
FootballIQ Server — v4.0.0
===========================
Thin entry point. Mounts the API router. That's it.

server.py → api/routes.py → PipelineManager → Everything
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes      import router
from config.settings import settings
from utils.file_utils import ensure_dir
from utils.logger    import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dir(settings.work_dir_path)
    log.info("FootballIQ API v4.0.0 starting")
    log.info("Work dir : %s", settings.work_dir_path)
    log.info("LLM      : %s / %s", settings.LLM_PROVIDER, settings.active_llm_model)
    yield


app = FastAPI(
    title    = "FootballIQ Analysis API",
    version  = "4.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"] if settings.ALLOW_ALL_ORIGINS else [settings.FRONTEND_URL],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Mount all routes under /api
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "service":     "FootballIQ Analysis API",
        "version":     "4.0.0",
        "status":      "running",
        "entry_point": "POST /api/upload-video → PipelineManager → Everything",
        "config":      settings.summary(),
    }
