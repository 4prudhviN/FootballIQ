#!/usr/bin/env python3
"""
Constants
=========
Hard-coded values that never change at runtime.
No environment variables here — these are fixed physical/domain facts.

Import from here — never hard-code magic numbers in other modules.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

DEFAULT_FPS             = 25.0      # assumed FPS when metadata is unavailable
DEFAULT_WIDTH           = 1920      # default processing resolution (px)
DEFAULT_HEIGHT          = 1080
DEFAULT_FRAME_STRIDE    = 3         # process every 3rd frame by default
MAX_VIDEO_DURATION_S    = 300.0     # 5 minutes — hard cap on input length
MAX_FILE_SIZE_MB        = 500.0     # maximum accepted upload size

SUPPORTED_EXTENSIONS    = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"
})


# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------

POSE_MODEL_COMPLEXITY       = 1       # 0=lite 1=full 2=heavy
POSE_DETECTION_CONFIDENCE   = 0.5
POSE_TRACKING_CONFIDENCE    = 0.5
POSE_LANDMARK_COUNT         = 33      # MediaPipe full-body landmark count
MIN_VISIBLE_LANDMARKS       = 8       # minimum for a usable pose reading


# ---------------------------------------------------------------------------
# Ball detection
# ---------------------------------------------------------------------------

HOUGH_DP                = 1.2
HOUGH_MIN_DIST          = 30        # px — minimum distance between circles
HOUGH_PARAM1            = 50        # Canny high threshold
HOUGH_PARAM2            = 30        # accumulator threshold
BALL_MIN_RADIUS_PX      = 5         # pixels
BALL_MAX_RADIUS_PX      = 60        # pixels
BALL_HISTORY_FRAMES     = 10        # frames kept for trajectory tracking


# ---------------------------------------------------------------------------
# Player detection
# ---------------------------------------------------------------------------

MOTION_PIXEL_RATIO          = 0.02   # fraction of frame pixels in motion
MOG2_HISTORY                = 200
MOG2_VAR_THRESHOLD          = 50
PLAYER_DETECTION_THRESHOLD  = 0.10   # min fraction of frames with player


# ---------------------------------------------------------------------------
# Activity detection
# ---------------------------------------------------------------------------

MIN_ACTIVITY_CONFIDENCE     = 0.15   # drop activities below this confidence
MAX_DETECTED_ACTIVITIES     = 4      # max simultaneous activities per session


# ---------------------------------------------------------------------------
# Skill classification score boundaries
# ---------------------------------------------------------------------------

ADVANCED_SCORE_THRESHOLD     = 0.70  # overall score ≥ this → Advanced
BEGINNER_SCORE_THRESHOLD     = 0.35  # overall score < this → Beginner


# ---------------------------------------------------------------------------
# Pipeline stage names (order must match execution order)
# ---------------------------------------------------------------------------

PIPELINE_STAGES = (
    "video_load",
    "frame_extract",
    "player_detect",
    "ball_detect",
    "pose_estimate",
    "activity_detect",
    "metric_calc",
    "skill_classify",
    "feedback_engine",
    "complete",
)


# ---------------------------------------------------------------------------
# AI / LLM
# ---------------------------------------------------------------------------

LLM_MAX_TOKENS      = 1024
LLM_TEMPERATURE     = 0.4
LLM_MAX_RETRIES     = 2
LLM_TIMEOUT_S       = 30.0
LLM_DEFAULT_PROVIDER = "gemini"


# ---------------------------------------------------------------------------
# Pixel / real-world calibration defaults
# ---------------------------------------------------------------------------

DEFAULT_PX_PER_METER    = 100.0     # rough default; override per video
PITCH_LENGTH_M          = 105.0     # standard FIFA pitch length
PITCH_WIDTH_M           = 68.0      # standard FIFA pitch width


# ---------------------------------------------------------------------------
# Feedback engine
# ---------------------------------------------------------------------------

MAX_DRILLS_PER_SESSION   = 5        # cap on drills returned per report
PRIORITY_DRILL_COUNT     = 1        # number of "start here" drills highlighted
