#!/usr/bin/env python3
"""
FootballIQ Analysis Backend — FastAPI Server  (v3)
===================================================

Delegates all analysis to PipelineManager which runs the full 9-stage pipeline:

  Video → Player Detection → Ball Detection → Pose Estimation
  → Activity Detection → Analyzer Selection → Metric Calculation
  → Skill Classification → Feedback Engine → Dashboard

Endpoints
---------
  POST /api/upload-video          Upload video → run pipeline → FootballSession JSON
  GET  /api/video/{job_id}/{file} Stream processed video
  GET  /api/pipeline-status/{id}  Poll current pipeline stage
  GET  /api/sessions              List saved session IDs
  GET  /api/sessions/{id}         Load a saved session JSON
  GET  /                          Health check

Startup
-------
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config.settings       import settings
from pipeline.pipeline_manager import PipelineManager
from reports.report_writer import ReportWriter
from session.session_manager import SessionManager
from utils.file_utils      import ensure_dir, save_bytes
from utils.logger          import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dir(settings.work_dir_path)
    log.info("Work directory: %s", settings.work_dir_path)
    log.info("LLM provider:   %s / %s", settings.LLM_PROVIDER, settings.active_llm_model)
    yield

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title    = "FootballIQ Analysis API",
    version  = "3.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"] if settings.ALLOW_ALL_ORIGINS else [settings.FRONTEND_URL],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Singletons
_pipeline = PipelineManager(
    frame_stride          = settings.FRAME_STRIDE,
    player_threshold      = 0.10,
    pose_model_complexity = settings.POSE_MODEL_COMPLEXITY,
)
_writer  = ReportWriter()
_session = SessionManager(writer=_writer)

# In-memory job status store (replace with Redis in production)
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# POST /api/upload-video
# ---------------------------------------------------------------------------

@app.post("/api/upload-video")
async def upload_video(
    file: UploadFile = File(..., description="MP4 video of a football movement drill."),
):
    """Run the full 9-stage pipeline and return a FootballSession-compatible payload."""

    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are accepted.")

    # Save upload to a unique job directory.
    job_id  = uuid.uuid4().hex[:12]
    job_dir = ensure_dir(settings.work_dir_path / job_id)

    input_path  = str(job_dir / "input.mp4")
    output_path = str(job_dir / "analyzed.mp4")

    content = await file.read()
    save_bytes(content, input_path)
    log.info("Job %s — saved %.1f MB: %s", job_id, len(content) / 1e6, file.filename)

    _jobs[job_id] = {"stage": "player_detection", "status": "running"}

    # Run pipeline synchronously (move to background task for production).
    output = _pipeline.run(input_path)

    if not output.success:
        _jobs[job_id]["status"] = "failed"
        raise HTTPException(status_code=422, detail=output.error or "Pipeline failed.")

    _jobs[job_id]["stage"]  = "complete"
    _jobs[job_id]["status"] = "complete"

    # Build response payload.
    payload: dict[str, Any] = {
        "status":    "complete",
        "job_id":    job_id,
        "video_url": f"/api/video/{job_id}/analyzed.mp4",

        # FootballSession fields
        "detectedActivities": output.detected_activities,
        "playerLevel":        output.player_level,
        "metrics":            output.metrics,
        "aiFeedback":         output.ai_feedback,
        "drills":             output.drills,

        # New structured fields from full pipeline
        "timeline":          output.timeline,
        "skillProfile":      output.skill_profile,
        "focusThisWeek":     output.focus_areas,
        "trainingPlan":      output.training_plan,
        "weeklyPlan":        output.weekly_plan,
        "recoveryAdvice":    output.recovery_advice,
        "coachingFeedback":  output.coaching_feedback,

        "_pipeline": output.diagnostics,
    }

    # Persist JSON report and create session.
    session = _session.create(output, file_name=file.filename or "upload.mp4")
    payload["session_id"] = session.id
    log.info("Job %s complete — level=%s activities=%s",
             job_id, output.player_level, output.detected_activities)

    return payload


# ---------------------------------------------------------------------------
# GET /api/video/{job_id}/{filename}
# ---------------------------------------------------------------------------

@app.get("/api/video/{job_id}/{filename}")
async def stream_video(job_id: str, filename: str):
    allowed = {"analyzed.mp4", "input.mp4"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="File not found.")

    path = settings.work_dir_path / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(str(path), media_type="video/mp4", filename=filename)


# ---------------------------------------------------------------------------
# GET /api/pipeline-status/{job_id}
# ---------------------------------------------------------------------------

@app.get("/api/pipeline-status/{job_id}")
async def pipeline_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": job_id, **job}


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions():
    """List all saved session IDs (newest first)."""
    return {"sessions": _writer.list_sessions()}


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}
# ---------------------------------------------------------------------------

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Load a saved session JSON report."""
    data = _writer.load_json(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found.")
    return data


# ---------------------------------------------------------------------------
# GET /  — health check
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service":  "FootballIQ Analysis API",
        "version":  "3.0.0",
        "status":   "running",
        "config":   settings.summary(),
        "endpoints": {
            "POST": "/api/upload-video",
            "GET":  "/api/video/{job_id}/{filename}",
            "GET":  "/api/pipeline-status/{job_id}",
            "GET":  "/api/sessions",
            "GET":  "/api/sessions/{session_id}",
        },
    }
