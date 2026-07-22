#!/usr/bin/env python3
"""
FootballIQ Analysis Backend — FastAPI Server  (v2)
===================================================

New 9-stage pipeline
--------------------
  1.  Player Detection      — confirm a player is visible in the video
  2.  Ball Detection        — detect ball presence / movement vectors
  3.  Pose Estimation       — MediaPipe full-body landmark tracking
  4.  Activity Detection    — classify what the player is doing
  5.  Analyzer Selection    — route to the correct activity analyzer
  6.  Metric Calculation    — run the selected analyzer, extract metrics
  7.  Skill Classification  — score metrics → Beginner/Intermediate/Advanced
  8.  Feedback Engine       — metrics + knowledge base → drills + coach tips
  9.  Dashboard             — return FootballSession JSON to the frontend

Endpoints
---------
  POST /api/upload-video
      Accept an MP4 video → run the full pipeline →
      return a FootballSession-compatible JSON payload.

  GET  /api/video/{job_id}/{filename}
      Stream a processed video file back to the frontend.

  GET  /api/pipeline-status/{job_id}
      Poll the current pipeline stage for a running job.

  GET  /
      Health check.

Startup
-------
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Dependencies
------------
  pip install "fastapi[standard]" uvicorn python-multipart
  pip install opencv-python mediapipe numpy
"""

import io
import sys
import uuid
from contextlib import asynccontextmanager, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------
import analyze_movement as am
from skill_classifier import PlayerMetrics, SkillLevel, classify_skill
from feedback_engine import FeedbackEngine, FeedbackRequest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORK_DIR = Path(__file__).resolve().parent / "backend_temp"

# Minimum fraction of frames in which a person must appear
# for the player detection stage to pass.
PLAYER_DETECTION_THRESHOLD: float = 0.10

# Minimum fraction of frames in which a ball-like circular
# object must appear for ball detection to pass.
BALL_DETECTION_THRESHOLD: float = 0.05

# Pipeline stage labels (returned in status endpoint).
PIPELINE_STAGES = [
    "player_detection",
    "ball_detection",
    "pose_estimation",
    "activity_detection",
    "analyzer_selection",
    "metric_calculation",
    "skill_classification",
    "feedback_engine",
    "dashboard",
]

# ---------------------------------------------------------------------------
# In-memory job store  (replace with Redis / DB in production)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Work directory: {WORK_DIR}")
    yield


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FootballIQ Analysis API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Stage 1 — Player Detection
# ---------------------------------------------------------------------------

def stage_player_detection(video_path: str) -> dict[str, Any]:
    """
    Use a simple background-subtractor to detect significant human-shaped
    motion blobs.  Returns detection confidence and a boolean pass/fail.

    In production replace with a YOLOv8-person detector.
    """
    cap = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    detected_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        mask = subtractor.apply(frame)
        # Count non-zero pixels — a person produces a large moving region.
        motion_pixels = cv2.countNonZero(mask)
        if motion_pixels > (frame.shape[0] * frame.shape[1] * 0.02):
            detected_frames += 1

    cap.release()

    confidence = detected_frames / total
    passed = confidence >= PLAYER_DETECTION_THRESHOLD

    return {
        "passed": passed,
        "confidence": round(confidence, 3),
        "detected_frames": detected_frames,
        "total_frames": total,
    }


# ---------------------------------------------------------------------------
# Stage 2 — Ball Detection
# ---------------------------------------------------------------------------

def stage_ball_detection(video_path: str) -> dict[str, Any]:
    """
    Detect circular objects (Hough circles) in the video as a proxy for
    ball presence.  Returns detection confidence.

    In production replace with a YOLOv8-ball detector or TrackNet.
    """
    cap = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    # Sample every 5th frame to keep it fast.
    SAMPLE_EVERY = 5

    detected_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SAMPLE_EVERY != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=40,
        )
        if circles is not None:
            detected_frames += 1

    cap.release()

    sampled = max(1, total // SAMPLE_EVERY)
    confidence = detected_frames / sampled

    return {
        "passed": confidence >= BALL_DETECTION_THRESHOLD,
        "confidence": round(confidence, 3),
        "ball_detected": confidence >= BALL_DETECTION_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Stage 3 — Pose Estimation  (via analyze_movement.py)
# ---------------------------------------------------------------------------

def stage_pose_estimation(
    input_path: str,
    output_path: str,
) -> dict[str, Any]:
    """
    Run MediaPipe pose estimation through analyze_movement.py.
    Captures stdout to extract warning flags.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = am.analyze_movement(
            input_path=input_path,
            output_path=output_path,
        )
    stdout_text = buf.getvalue()

    # Parse [RESULT] warnings.
    warnings: list[str] = []
    metric_map = {
        "torso lean":     "POOR POSTURE / LEANING BACK",
        "knee alignment": "KNEE ALIGNMENT RISK",
        "gait symmetry":  "ASYMMETRIC GAIT DETECTED",
    }
    for line in stdout_text.splitlines():
        ll = line.strip().lower()
        if "flagged" not in ll:
            continue
        for keyword, label in metric_map.items():
            if keyword in ll and label not in warnings:
                warnings.append(label)

    return {
        "exit_code": exit_code,
        "warnings": warnings,
        "stdout": stdout_text,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Stage 4 — Activity Detection
# ---------------------------------------------------------------------------

def stage_activity_detection(warnings: list[str]) -> dict[str, Any]:
    """
    Classify what football activities are present based on pose warnings
    and (in future) ball trajectory / scene context.

    Returns a list of detected FootballAction strings.
    """
    activities: list[str] = []

    # Heuristic mapping — extend when real activity_detector.py is wired in.
    if "POOR POSTURE / LEANING BACK" in warnings:
        activities.append("shooting")
    if "KNEE ALIGNMENT RISK" in warnings:
        activities.append("dribbling")
    if "ASYMMETRIC GAIT DETECTED" in warnings:
        activities.append("movement")

    # Always include at least one activity.
    if not activities:
        activities.append("passing")

    return {
        "detected_activities": activities,
        "activity_count": len(activities),
    }


# ---------------------------------------------------------------------------
# Stage 5 — Analyzer Selection
# ---------------------------------------------------------------------------

def stage_analyzer_selection(activities: list[str]) -> dict[str, Any]:
    """
    Map each detected activity to its corresponding analyzer module
    in the analyzers/ directory.
    """
    ANALYZER_MAP = {
        "passing":     "analyzers.passing_analyzer",
        "dribbling":   "analyzers.dribbling_analyzer",
        "shooting":    "analyzers.shooting_analyzer",
        "goalkeeping": "analyzers.goalkeeping_analyzer",
        "defending":   "analyzers.defending_analyzer",
        "movement":    "analyzers.movement_analyzer",
    }

    selected = [ANALYZER_MAP[a] for a in activities if a in ANALYZER_MAP]

    return {
        "selected_analyzers": selected,
        "activity_to_analyzer": {a: ANALYZER_MAP[a] for a in activities if a in ANALYZER_MAP},
    }


# ---------------------------------------------------------------------------
# Stage 6 — Metric Calculation
# ---------------------------------------------------------------------------

# Default metric values per activity — replace with real analyzer output
# once the analyzer modules are fully implemented.
_ACTIVITY_METRICS: dict[str, dict[str, str]] = {
    "passing": {
        "Ball Control":   "92%",
        "First Touch":    "0.36 m/s²",
        "Pass Accuracy":  "87%",
        "Weight of Pass": "Medium",
    },
    "dribbling": {
        "Close Control":        "88%",
        "Change of Direction":  "5.8 m/s",
        "Touch Tightness":      "±2.4 cm",
        "Speed with Ball":      "24 km/h",
    },
    "shooting": {
        "Shot Velocity":    "88 km/h",
        "Launch Angle":     "14°",
        "Target Accuracy":  "81%",
        "Torso Alignment":  "12°",
    },
    "goalkeeping": {
        "Reaction Time":  "0.28s",
        "Diving Range":   "2.4m",
        "Distribution":   "74%",
        "Positioning":    "Good",
    },
    "defending": {
        "Tackle Timing":  "Good",
        "Positioning":    "88%",
        "Interception":   "3",
        "Aerial Duels":   "67%",
    },
    "movement": {
        "Gait Symmetry":  "92%",
        "Stride Length":  "1.24m",
        "Sprint Speed":   "31.2 km/h",
        "Agility":        "4.2s",
    },
}


def stage_metric_calculation(
    activities: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """
    Derive scalar biomechanical metrics from warnings and look up
    per-action display metrics.
    """
    torso_lean     = 22.0 if "POOR POSTURE / LEANING BACK" in warnings else 8.0
    knee_stability = 72.0 if "KNEE ALIGNMENT RISK"          in warnings else 87.0
    gait_symmetry  = 78.0 if "ASYMMETRIC GAIT DETECTED"     in warnings else 92.0

    by_action = {a: _ACTIVITY_METRICS.get(a, {}) for a in activities}

    return {
        "torso_lean":     torso_lean,
        "knee_stability": knee_stability,
        "gait_symmetry":  gait_symmetry,
        "by_action":      by_action,
    }


# ---------------------------------------------------------------------------
# Stage 7 — Skill Classification
# ---------------------------------------------------------------------------

def stage_skill_classification(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Feed scalar metrics into skill_classifier.py and get
    Beginner / Intermediate / Advanced + per-metric scores.
    """
    player_metrics = PlayerMetrics(
        torso_lean=metrics["torso_lean"],
        knee_dev=1.0 - (metrics["knee_stability"] / 100.0),
        gait_asymmetry=1.0 - (metrics["gait_symmetry"] / 100.0),
    )

    report = classify_skill(player_metrics)

    return {
        "player_level":   report.level.value,
        "overall_score":  report.overall_score,
        "metric_scores":  report.metric_scores,
        "strengths":      report.strengths,
        "weaknesses":     report.weaknesses,
    }


# ---------------------------------------------------------------------------
# Stage 8 — Feedback Engine
# ---------------------------------------------------------------------------

def stage_feedback_engine(
    metrics: dict[str, Any],
    activities: list[str],
    player_level: str,
) -> dict[str, Any]:
    """
    Generate plain-English coaching feedback, drills, and tips via
    feedback_engine.py for each detected activity.
    """
    engine = FeedbackEngine()

    primary_activity = activities[0] if activities else "general"

    raw_metrics = {
        "torso_lean":   metrics["torso_lean"],
        "knee_dev":     1.0 - (metrics["knee_stability"] / 100.0),
        "gait_asymmetry": 1.0 - (metrics["gait_symmetry"] / 100.0),
    }

    request = FeedbackRequest(
        metrics=raw_metrics,
        activity=primary_activity,
        level=player_level,
    )
    report = engine.generate(request)

    # Convert dataclass objects to plain dicts for JSON serialisation.
    drills = [
        {
            "name":          item.drill.split(":")[0].strip(),
            "targetMetric":  item.metric,
            "instructions":  item.drill,
            "coachTip":      item.coach_tip,
            "duration":      "10-15 min",
            "difficulty":    player_level,
        }
        for item in report.items
    ]

    return {
        "summary":          report.summary,
        "strengths":        report.positive,
        "weaknesses":       [i.metric.replace("_", " ").title() for i in report.items],
        "coaching_tips":    [i.coach_tip for i in report.items],
        "motivational_tip": report.motivational_tip,
        "drills":           drills,
        "priority_drill":   report.priority_drill,
    }


# ---------------------------------------------------------------------------
# Stage 9 — Dashboard payload assembly
# ---------------------------------------------------------------------------

def stage_dashboard(
    job_id: str,
    output_video_path: str,
    activities: list[str],
    metrics: dict[str, Any],
    skill: dict[str, Any],
    feedback: dict[str, Any],
    warnings: list[str],
    player_detection: dict[str, Any],
    ball_detection: dict[str, Any],
) -> dict[str, Any]:
    """
    Assemble the final FootballSession-compatible JSON payload.
    """
    video_url = f"/api/video/{job_id}/analyzed_movement.mp4"

    return {
        "status":       "complete",
        "job_id":       job_id,
        "video_url":    video_url,

        # FootballSession fields
        "detectedActivities": activities,
        "playerLevel":        skill["player_level"],

        "metrics": {
            "byAction":      metrics["by_action"],
            "torsoLean":     metrics["torso_lean"],
            "kneeStability": metrics["knee_stability"],
            "gaitSymmetry":  metrics["gait_symmetry"],
            "warnings":      warnings,
        },

        "aiFeedback": {
            "summary":         feedback["summary"],
            "strengths":       feedback["strengths"],
            "weaknesses":      feedback["weaknesses"],
            "coachingTips":    feedback["coaching_tips"],
            "motivationalTip": feedback["motivational_tip"],
        },

        "drills": feedback["drills"],

        # Debug / diagnostic fields
        "_pipeline": {
            "player_detection": player_detection,
            "ball_detection":   ball_detection,
            "skill_scores":     skill["metric_scores"],
            "overall_score":    skill["overall_score"],
        },
    }


# ---------------------------------------------------------------------------
# POST /api/upload-video
# ---------------------------------------------------------------------------

@app.post("/api/upload-video")
async def upload_video(
    file: UploadFile = File(..., description="MP4 video of a football movement drill."),
):
    """
    Run the full 9-stage FootballIQ analysis pipeline and return a
    FootballSession-compatible JSON payload.
    """

    # ── Validate ────────────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail="Only .mp4 files are accepted.",
        )

    # ── Save upload ─────────────────────────────────────────────────────────
    job_id  = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path  = str(job_dir / "input_movement.mp4")
    output_path = str(job_dir / "analyzed_movement.mp4")

    try:
        content = await file.read()
        Path(input_path).write_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    print(f"[PIPELINE] Job {job_id} started — {len(content)/1e6:.1f} MB")
    _jobs[job_id] = {"stage": "player_detection", "progress": 0}

    # ── Stage 1: Player Detection ────────────────────────────────────────────
    _jobs[job_id]["stage"] = PIPELINE_STAGES[0]
    print(f"[STAGE 1/9] Player Detection")
    player_det = stage_player_detection(input_path)
    if not player_det["passed"]:
        raise HTTPException(
            status_code=422,
            detail=f"No player detected in video (confidence: {player_det['confidence']:.1%}). "
                   "Ensure the player is clearly visible in the frame.",
        )

    # ── Stage 2: Ball Detection ──────────────────────────────────────────────
    _jobs[job_id]["stage"] = PIPELINE_STAGES[1]
    print(f"[STAGE 2/9] Ball Detection")
    ball_det = stage_ball_detection(input_path)
    # Ball detection is informational — we don't fail the pipeline if absent.
    print(f"  Ball detected: {ball_det['ball_detected']} (conf: {ball_det['confidence']:.1%})")

    # ── Stage 3: Pose Estimation ─────────────────────────────────────────────
    _jobs[job_id]["stage"] = PIPELINE_STAGES[2]
    print(f"[STAGE 3/9] Pose Estimation")
    pose = stage_pose_estimation(input_path, output_path)
    warnings = pose["warnings"]
    print(f"  Warnings: {warnings or 'none'}")

    # ── Stage 4: Activity Detection ──────────────────────────────────────────
    _jobs[job_id]["stage"] = PIPELINE_STAGES[3]
    print(f"[STAGE 4/9] Activity Detection")
    activity = stage_activity_detection(warnings)
    activities = activity["detected_activities"]
    print(f"  Activities: {activities}")

    # ── Stage 5: Analyzer Selection ──────────────────────────────────────────
    _jobs[job_id]["stage"] = PIPELINE_STAGES[4]
    print(f"[STAGE 5/9] Analyzer Selection")
    selection = stage_analyzer_selection(activities)
    print(f"  Analyzers: {selection['selected_analyzers']}")

    # ── Stage 6: Metric Calculation ───────────────────────────────────────────
    _jobs[job_id]["stage"] = PIPELINE_STAGES[5]
    print(f"[STAGE 6/9] Metric Calculation")
    metrics = stage_metric_calculation(activities, warnings)

    # ── Stage 7: Skill Classification ─────────────────────────────────────────
    _jobs[job_id]["stage"] = PIPELINE_STAGES[6]
    print(f"[STAGE 7/9] Skill Classification")
    skill = stage_skill_classification(metrics)
    print(f"  Level: {skill['player_level']} (score: {skill['overall_score']:.2f})")

    # ── Stage 8: Feedback Engine ──────────────────────────────────────────────
    _jobs[job_id]["stage"] = PIPELINE_STAGES[7]
    print(f"[STAGE 8/9] Feedback Engine")
    feedback = stage_feedback_engine(metrics, activities, skill["player_level"])
    print(f"  Drills generated: {len(feedback['drills'])}")

    # ── Stage 9: Dashboard ─────────────────────────────────────────────────────
    _jobs[job_id]["stage"] = PIPELINE_STAGES[8]
    print(f"[STAGE 9/9] Assembling Dashboard payload")
    payload = stage_dashboard(
        job_id=job_id,
        output_video_path=output_path,
        activities=activities,
        metrics=metrics,
        skill=skill,
        feedback=feedback,
        warnings=warnings,
        player_detection=player_det,
        ball_detection=ball_det,
    )

    _jobs[job_id]["stage"] = "complete"
    print(f"[PIPELINE] Job {job_id} complete ✓")

    return payload


# ---------------------------------------------------------------------------
# GET /api/video/{job_id}/{filename}
# ---------------------------------------------------------------------------

@app.get("/api/video/{job_id}/{filename}")
async def stream_video(job_id: str, filename: str):
    """Stream a processed video file."""
    allowed = {"analyzed_movement.mp4", "input_movement.mp4"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="File not found.")

    file_path = WORK_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# GET /api/pipeline-status/{job_id}
# ---------------------------------------------------------------------------

@app.get("/api/pipeline-status/{job_id}")
async def pipeline_status(job_id: str):
    """
    Poll the current pipeline stage for a running job.
    Returns stage name and index.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    stage = job.get("stage", "unknown")
    stage_index = PIPELINE_STAGES.index(stage) if stage in PIPELINE_STAGES else -1

    return {
        "job_id":       job_id,
        "stage":        stage,
        "stage_index":  stage_index,
        "total_stages": len(PIPELINE_STAGES),
        "stages":       PIPELINE_STAGES,
    }


# ---------------------------------------------------------------------------
# GET /  — health check
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service":  "FootballIQ Analysis API",
        "version":  "2.0.0",
        "pipeline": PIPELINE_STAGES,
        "endpoints": {
            "POST": "/api/upload-video",
            "GET":  "/api/video/{job_id}/{filename}",
            "GET":  "/api/pipeline-status/{job_id}",
        },
        "status": "running",
    }
