#!/usr/bin/env python3
"""
FootballIQ Analysis Backend
============================
Single entry point. Everything goes through PipelineManager.

server.py
  ↓
PipelineManager.run_with_context(video_path)
  ↓
Everything (Video → Detection → Pose → Activity → Metrics → Coaching → LLM → Report)

Endpoints
---------
  POST /api/upload-video          Upload video → PipelineManager → response
  GET  /api/video/{job_id}/{file} Stream processed video
  GET  /api/pipeline-status/{id}  Poll current pipeline stage
  GET  /api/sessions              List all saved sessions
  GET  /api/sessions/{id}         Load a session
  GET  /api/progress              Progress across all sessions
  GET  /                          Health check

Startup
-------
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config.settings          import settings
from pipeline.pipeline_manager import PipelineManager
from reports.report_writer    import ReportWriter
from session.session_manager  import SessionManager
from utils.file_utils         import ensure_dir, save_bytes
from utils.logger             import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dir(settings.work_dir_path)
    log.info("Work directory : %s", settings.work_dir_path)
    log.info("LLM provider   : %s / %s", settings.LLM_PROVIDER, settings.active_llm_model)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Singletons — created once at startup
# ---------------------------------------------------------------------------

_pipeline = PipelineManager(
    frame_stride          = settings.FRAME_STRIDE,
    player_threshold      = 0.10,
    pose_model_complexity = settings.POSE_MODEL_COMPLEXITY,
    use_ai                = True,
)
_writer  = ReportWriter()
_session = SessionManager(writer=_writer)

# In-memory job status store.
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# POST /api/upload-video
# ---------------------------------------------------------------------------

@app.post("/api/upload-video")
async def upload_video(
    file: UploadFile = File(..., description="MP4 video of a football drill."),
):
    """
    Single pipeline entry point.

    Every upload goes through PipelineManager.run_with_context()
    which orchestrates all stages and returns a PipelineContext.
    """
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are accepted.")

    # Save upload.
    job_id    = uuid.uuid4().hex[:12]
    job_dir   = ensure_dir(settings.work_dir_path / job_id)
    input_path = str(job_dir / "input.mp4")

    content = await file.read()
    save_bytes(content, input_path)
    log.info("Job %s — saved %.1f MB: %s", job_id, len(content) / 1e6, file.filename)

    _jobs[job_id] = {"stage": "starting", "status": "running"}

    # ── Single pipeline call ──────────────────────────────────────────────
    ctx = _pipeline.run_with_context(input_path, session_id=job_id)

    if not ctx.is_complete():
        _jobs[job_id]["status"] = "failed"
        raise HTTPException(status_code=422, detail=ctx.error or "Pipeline failed.")

    _jobs[job_id]["stage"]  = "complete"
    _jobs[job_id]["status"] = "complete"

    # Persist session.
    _session.create(
        _make_pipeline_output(ctx),
        file_name=file.filename or "upload.mp4",
    )

    payload = ctx.to_api_response()
    payload["job_id"]    = job_id
    payload["video_url"] = f"/api/video/{job_id}/analyzed.mp4"

    log.info(
        "Job %s complete — level=%s activities=%s",
        job_id, ctx.player_level, ctx.detected_activities,
    )
    return payload


# ---------------------------------------------------------------------------
# GET /api/video/{job_id}/{filename}
# ---------------------------------------------------------------------------

@app.get("/api/video/{job_id}/{filename}")
async def stream_video(job_id: str, filename: str):
    """Stream a processed video."""
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
    """Poll current pipeline stage for a job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": job_id, **job}


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions():
    """List all saved session IDs, newest first."""
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
# GET /api/progress
# ---------------------------------------------------------------------------

@app.get("/api/progress")
async def get_progress():
    """Progress comparison across all saved sessions."""
    from session.session_history import SessionHistory
    history  = SessionHistory(manager=_session)
    sessions = history.all_sessions()

    if len(sessions) < 2:
        return {"message": "Upload at least 2 sessions to see progress.", "sessions": len(sessions)}

    timeline   = history.progress_timeline(sessions)
    comparison = history.compare(sessions[-1], sessions[0])

    metric_history: dict[str, list[dict]] = {}
    for point in timeline:
        for metric, val in [
            ("torso_lean",     point.torso_lean),
            ("knee_stability", point.knee_stability),
            ("gait_symmetry",  point.gait_symmetry),
        ]:
            if metric not in metric_history:
                metric_history[metric] = []
            metric_history[metric].append({
                "session_id": point.session_id,
                "date":       point.date[:10],
                "value":      round(val, 1),
                "level":      point.player_level,
            })

    trend = comparison.get("overall_trend", "stable")
    summary = {
        "improving":  f"Your performance has improved across your last {len(sessions)} sessions.",
        "stable":     f"Your performance has been consistent across {len(sessions)} sessions.",
        "regressing": "Some metrics dipped. Review your training plan.",
    }.get(trend, "Keep training consistently.")

    return {
        "session_count":         len(sessions),
        "overall_trend":         trend,
        "summary":               summary,
        "comparison":            comparison,
        "metric_history":        metric_history,
        "persistent_weaknesses": history.persistent_weaknesses(sessions),
    }


# ---------------------------------------------------------------------------
# GET /  — health check
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service":  "FootballIQ Analysis API",
        "version":  "4.0.0",
        "status":   "running",
        "entry_point": "POST /api/upload-video → PipelineManager → Everything",
        "config":   settings.summary(),
        "endpoints": {
            "POST": "/api/upload-video",
            "GET":  "/api/video/{job_id}/{filename}",
            "GET":  "/api/pipeline-status/{job_id}",
            "GET":  "/api/sessions",
            "GET":  "/api/sessions/{session_id}",
            "GET":  "/api/progress",
        },
    }


# ---------------------------------------------------------------------------
# Helper — adapt PipelineContext to SessionManager interface
# ---------------------------------------------------------------------------

def _make_pipeline_output(ctx: Any) -> Any:
    """Wrap PipelineContext fields into an object SessionManager.create() accepts."""
    class _Adapter:
        def __init__(self, ctx):
            self.success             = ctx.is_complete()
            self.error               = ctx.error
            self.detected_activities = ctx.detected_activities
            self.player_level        = ctx.player_level
            self.metrics             = ctx.analysis.metrics if ctx.analysis else {}
            self.ai_feedback         = ctx.report.report   if ctx.report   else {}
            self.drills              = ctx.coaching.drills if ctx.coaching  else []
            self.timeline            = [
                {
                    "label":      seg.label,
                    "action":     seg.action,
                    "startTime":  seg.start_time_s,
                    "endTime":    seg.end_time_s,
                    "duration":   seg.duration_s,
                    "confidence": seg.confidence,
                }
                for seg in (ctx.activity.timeline if ctx.activity else [])
            ]
            self.diagnostics = {}
    return _Adapter(ctx)
