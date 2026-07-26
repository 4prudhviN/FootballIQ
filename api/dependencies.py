#!/usr/bin/env python3
"""
API Dependencies
================
FastAPI dependency injection — shared singletons injected into route handlers.

Usage in routes::

    from api.dependencies import get_pipeline, get_session_manager
    from fastapi import Depends

    @router.post("/upload-video")
    async def upload(pipeline=Depends(get_pipeline)):
        ...
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from config.settings          import settings
from pipeline.pipeline_manager import PipelineManager
from reports.report_writer    import ReportWriter
from session.session_manager  import SessionManager


@lru_cache(maxsize=1)
def get_pipeline() -> PipelineManager:
    """Shared PipelineManager instance."""
    return PipelineManager(
        frame_stride          = settings.FRAME_STRIDE,
        player_threshold      = 0.10,
        pose_model_complexity = settings.POSE_MODEL_COMPLEXITY,
        use_ai                = True,
    )


@lru_cache(maxsize=1)
def get_report_writer() -> ReportWriter:
    """Shared ReportWriter instance."""
    return ReportWriter()


@lru_cache(maxsize=1)
def get_session_manager() -> SessionManager:
    """Shared SessionManager instance."""
    return SessionManager(writer=get_report_writer())
