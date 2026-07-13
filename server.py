#!/usr/bin/env python3
"""
FootballIQ Analysis Backend — FastAPI Server
=============================================

A lightweight production‑ready API that connects the FootballIQ frontend to
the MediaPipe movement‑analysis pipeline.

Endpoints
---------
  POST /api/upload-video
      Accept an MP4 video file → run the multi‑metric analysis →
      return the processed video URL + detected warning flags.

  GET  /api/video/{filename}
      Stream a processed video file (e.g. 'analyzed_movement.mp4')
      back to the frontend <video> player.

Startup
-------
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Dependencies (install once)
---------------------------
  pip install "fastapi[standard]" uvicorn python-multipart
  pip install opencv-python mediapipe numpy
"""

import io
import os
import re
import sys
import uuid
from contextlib import redirect_stdout, asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Local analysis engine — import the core module from the same directory
# ---------------------------------------------------------------------------
# The server imports the existing analyze_movement.py module and calls its
# top‑level `analyze_movement()` function.  We capture the printed warning
# output by redirecting stdout into a string buffer.

import analyze_movement as am

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

# Where uploaded and processed videos live.  Created automatically on startup.
WORK_DIR = Path(__file__).resolve().parent / "backend_temp"

# Map from "torso" / "knee" / "gait" to the human‑readable warning labels
# that the client UI expects.
WARNING_LABELS: dict[str, list[str]] = {
    "torso": ["POOR POSTURE / LEANING BACK"],
    "knee":  ["KNEE ALIGNMENT RISK"],
    "gait":  ["ASYMMETRIC GAIT DETECTED"],
}

# ---------------------------------------------------------------------------
# Lifespan handler — create the working directory on start
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the temp directory when the server starts (no clean‑up needed)."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Work directory: {WORK_DIR}")
    yield


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FootballIQ Movement Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
# Allow the React dev server (and any common alternative ports) to call the
# API without being blocked by the browser's same‑origin policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",       # Vite (React) default
        "http://localhost:3000",       # Create‑React‑App / Next.js alternative
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        # Allow all origins in development (remove in production).
        # In production, replace with your frontend's actual domain.
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper — parse warning flags from the analysis script's stdout
# ---------------------------------------------------------------------------

def _parse_warnings(captured_stdout: str) -> list[str]:
    """
    Scan the printed output of `analyze_movement()` for the [RESULT] section
    and return a list of warning strings the frontend can display.

    The analysis script prints lines like::

        [RESULT] Warnings triggered:
            Torso lean:     ⚠ FLAGGED
            Knee alignment: ✓ OK (or no data)
            Gait symmetry:  ⚠ FLAGGED

    We map "FLAGGED" lines back to the human‑readable labels.
    """
    warnings: list[str] = []

    # Map metric names (as they appear in the stdout) to warning labels.
    metric_map = {
        "torso lean":     WARNING_LABELS["torso"],
        "knee alignment": WARNING_LABELS["knee"],
        "gait symmetry":  WARNING_LABELS["gait"],
    }

    for line in captured_stdout.splitlines():
        line_lower = line.strip().lower()

        # Look for lines that contain "flagged".
        if "flagged" not in line_lower:
            continue

        for keyword, labels in metric_map.items():
            if keyword in line_lower:
                warnings.extend(labels)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique_warnings: list[str] = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            unique_warnings.append(w)

    return unique_warnings


# ---------------------------------------------------------------------------
# POST /api/upload-video
# ---------------------------------------------------------------------------

@app.post("/api/upload-video")
async def upload_video(
    file: UploadFile = File(..., description="MP4 video file of a football movement drill."),
):
    """
    Accept an MP4 file from the frontend, run the full biomechanical analysis
    pipeline, and return:

      - ``video_url``  — URL to stream the processed video
      - ``warnings``   — list of alert strings (e.g. "POOR POSTURE / LEANING BACK")
      - ``status``     — "complete" or "error"
    """

    # ── 1. Validate file type ──────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail="Only MP4 files are accepted.  Please upload a .mp4 video.",
        )

    # ── 2. Save the uploaded file ──────────────────────────────────────────
    # Use a UUID sub‑directory to avoid collisions between concurrent uploads.
    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input_movement.mp4"
    output_path = job_dir / "analyzed_movement.mp4"

    try:
        content = await file.read()
        input_path.write_bytes(content)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )

    file_size_mb = len(content) / (1024 * 1024)
    print(f"[INFO] Saved upload  →  {input_path}  ({file_size_mb:.1f} MB)")

    # ── 3. Run the analysis pipeline ───────────────────────────────────────
    # We redirect stdout so we can capture the [RESULT] block and extract
    # warning flags without modifying the existing script.
    print(f"[INFO] Running analysis →  {output_path}")
    stdout_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture):
            exit_code = am.analyze_movement(
                input_path=str(input_path),
                output_path=str(output_path),
            )
        captured_text = stdout_capture.getvalue()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline error: {exc}",
        )

    # ── 4. Parse warnings from the captured output ─────────────────────────
    warnings = _parse_warnings(captured_text)

    # Log what happened for debugging.
    print(captured_text)  # preserve the original output in the server log

    # ── 5. Verify the output video exists ──────────────────────────────────
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail="Analysis completed but no output video was produced.",
        )

    # ── 6. Build the streaming URL ─────────────────────────────────────────
    video_url = f"/api/video/{job_id}/analyzed_movement.mp4"

    return {
        "status": "complete" if exit_code == 0 else "error",
        "video_url": video_url,
        "warnings": warnings,
        "job_id": job_id,
    }


# ---------------------------------------------------------------------------
# GET /api/video/{job_id}/{filename}
# ---------------------------------------------------------------------------

@app.get("/api/video/{job_id}/{filename}")
async def stream_video(job_id: str, filename: str):
    """
    Stream a processed video file from the working directory.

    Path construction::

        backend_temp/<job_id>/<filename>

    The ``job_id`` and ``filename`` are the values returned in the
    ``upload-video`` response, so the frontend can plug them directly
    into a ``<video src="...">`` tag.
    """
    # Security: only allow known filenames to prevent path traversal.
    allowed = {"analyzed_movement.mp4", "input_movement.mp4"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="File not found.")

    file_path = WORK_DIR / job_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    # FastAPI's FileResponse handles streaming & proper Content-Type headers.
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# GET /  — health check
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Simple health‑check / landing page."""
    return {
        "service": "FootballIQ Movement Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "POST":  "/api/upload-video",
            "GET":   "/api/video/{job_id}/{filename}",
        },
        "status": "running",
    }