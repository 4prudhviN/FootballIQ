#!/usr/bin/env python3
"""
API Routes
==========
All FastAPI route handlers. server.py only mounts this router.

Every route uses dependency injection from api/dependencies.py.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies  import get_pipeline, get_session_manager, get_report_writer
from config.settings   import settings
from utils.file_utils  import ensure_dir, save_bytes
from utils.logger      import get_logger

log = get_logger(__name__)

router = APIRouter()

# In-memory job store (replace with Redis in production).
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# POST /api/upload-video
# ---------------------------------------------------------------------------

@router.post("/upload-video")
async def upload_video(
    file:     UploadFile = File(..., description="MP4 video of a football drill."),
    pipeline  = Depends(get_pipeline),
    sessions  = Depends(get_session_manager),
):
    """Single pipeline entry point — everything goes through PipelineManager."""
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are accepted.")

    job_id    = uuid.uuid4().hex[:12]
    job_dir   = ensure_dir(settings.work_dir_path / job_id)
    input_path = str(job_dir / "input.mp4")

    content = await file.read()
    save_bytes(content, input_path)
    log.info("Job %s — %.1f MB: %s", job_id, len(content) / 1e6, file.filename)

    _jobs[job_id] = {"stage": "starting", "status": "running"}

    ctx = pipeline.run_with_context(input_path, session_id=job_id)

    if not ctx.is_complete():
        _jobs[job_id]["status"] = "failed"
        raise HTTPException(status_code=422, detail=ctx.error or "Pipeline failed.")

    _jobs[job_id] = {"stage": "complete", "status": "complete"}

    sessions.create(_AdapterOutput(ctx), file_name=file.filename or "upload.mp4")

    payload             = ctx.to_api_response()
    payload["job_id"]   = job_id
    payload["video_url"] = f"/api/video/{job_id}/analyzed.mp4"

    log.info("Job %s — level=%s activities=%s", job_id, ctx.player_level, ctx.detected_activities)
    return payload


# ---------------------------------------------------------------------------
# GET /api/video/{job_id}/{filename}
# ---------------------------------------------------------------------------

@router.get("/video/{job_id}/{filename}")
async def stream_video(job_id: str, filename: str):
    from fastapi.responses import FileResponse
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

@router.get("/pipeline-status/{job_id}")
async def pipeline_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": job_id, **job}


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def list_sessions(writer=Depends(get_report_writer)):
    return {"sessions": writer.list_sessions()}


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}
# ---------------------------------------------------------------------------

@router.get("/sessions/{session_id}")
async def get_session(session_id: str, writer=Depends(get_report_writer)):
    data = writer.load_json(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found.")
    return data


# ---------------------------------------------------------------------------
# GET /api/progress
# ---------------------------------------------------------------------------

@router.get("/progress")
async def get_progress(sessions=Depends(get_session_manager)):
    from session.session_history import SessionHistory
    history  = SessionHistory(manager=sessions)
    all_sess = history.all_sessions()

    if len(all_sess) < 2:
        return {"message": "Upload at least 2 sessions to see progress.", "sessions": len(all_sess)}

    timeline   = history.progress_timeline(all_sess)
    comparison = history.compare(all_sess[-1], all_sess[0])

    metric_history: dict[str, list] = {}
    for point in timeline:
        for metric, val in [
            ("torso_lean", point.torso_lean),
            ("knee_stability", point.knee_stability),
            ("gait_symmetry", point.gait_symmetry),
        ]:
            metric_history.setdefault(metric, []).append({
                "session_id": point.session_id,
                "date":       point.date[:10],
                "value":      round(val, 1),
                "level":      point.player_level,
            })

    trend = comparison.get("overall_trend", "stable")
    return {
        "session_count":         len(all_sess),
        "overall_trend":         trend,
        "summary":               _progress_summary(trend, len(all_sess)),
        "comparison":            comparison,
        "metric_history":        metric_history,
        "persistent_weaknesses": history.persistent_weaknesses(all_sess),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress_summary(trend: str, n: int) -> str:
    return {
        "improving":  f"Your performance has improved across your last {n} sessions.",
        "stable":     f"Consistent performance across {n} sessions.",
        "regressing": "Some metrics dipped. Review your training plan.",
    }.get(trend, "Keep training consistently.")


class _AdapterOutput:
    """Adapts PipelineContext to the SessionManager.create() interface."""
    def __init__(self, ctx: Any) -> None:
        self.success             = ctx.is_complete()
        self.error               = ctx.error
        self.detected_activities = ctx.detected_activities
        self.player_level        = ctx.player_level
        self.metrics             = getattr(ctx.analysis, "metrics", {})
        self.ai_feedback         = getattr(ctx.report,   "report",   {})
        self.drills              = getattr(ctx.coaching, "drills",   [])
        self.timeline            = [
            {"label": s.label, "action": s.action, "startTime": s.start_time_s,
             "endTime": s.end_time_s, "duration": s.duration_s, "confidence": s.confidence}
            for s in (getattr(ctx.activity, "timeline", []) or [])
        ]
        self.diagnostics = {}
