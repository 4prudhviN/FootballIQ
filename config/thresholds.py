#!/usr/bin/env python3
"""
Thresholds
==========
All numeric decision boundaries used across the analysis pipeline.
Changing a value here changes it everywhere — no hunting through modules.

Categories:
  - Biomechanical thresholds     (pose analysis)
  - Ball / action thresholds     (ball speed, shot power, pass confidence)
  - Activity detection           (minimum confidence per action)
  - Skill classification         (per-metric advanced / beginner boundaries)
  - Player movement              (sprint, stationary, gait asymmetry)
  - Quality gates                (minimum data required to proceed)
"""

from __future__ import annotations


# ============================================================
# BIOMECHANICAL THRESHOLDS
# ============================================================

# Torso lean — degrees from vertical.
# At or below ADVANCED → good form.
# At or above BEGINNER → flag as poor posture.
TORSO_LEAN_ADVANCED_DEG   = 8.0
TORSO_LEAN_BEGINNER_DEG   = 20.0
TORSO_LEAN_WARNING_DEG    = 15.0   # threshold used for real-time warning banner

# Knee deviation — fraction of thigh length.
KNEE_DEV_ADVANCED         = 0.15
KNEE_DEV_BEGINNER         = 0.30
KNEE_DEV_WARNING          = 0.25   # threshold for KNEE ALIGNMENT RISK warning
KNEE_DEV_SEVERE           = 0.40   # threshold for injury-risk flag

# Gait asymmetry — fraction (0 = symmetric, 1 = fully asymmetric).
GAIT_ASYMMETRY_ADVANCED   = 0.08
GAIT_ASYMMETRY_BEGINNER   = 0.20
GAIT_ASYMMETRY_WARNING    = 0.15   # threshold for ASYMMETRIC GAIT warning

# Minimum frames between two foot-plant events (debounce).
MIN_STRIDE_FRAMES         = 8
ASYMMETRY_WINDOW_FRAMES   = 6     # rolling window for gait calculation


# ============================================================
# BALL / ACTION THRESHOLDS
# ============================================================

# Pass confidence — minimum quality score for a pass to be counted.
PASS_CONFIDENCE           = 0.60   # 0.0–1.0

# Ball speed thresholds (pixels/frame in normalised [0,1] space).
BALL_SPEED_STATIONARY     = 0.002  # below this = ball not moving
BALL_SPEED_PASS           = 0.010  # typical pass speed (min)
BALL_SPEED_SHOT_MIN       = 0.025  # minimum speed to classify as a shot
BALL_SPEED_SHOT_MAX       = 0.120  # above this = very powerful shot

# Shot power limit (km/h) — cap for display and skill scoring.
SHOT_POWER_LIMIT_KMH      = 120.0  # elite max ~130 km/h

# Launch angle thresholds (degrees from horizontal).
LAUNCH_ANGLE_IDEAL_MIN    = 8.0    # below this = too flat (likely blocked)
LAUNCH_ANGLE_IDEAL_MAX    = 20.0   # above this = too high (wasteful)
LAUNCH_ANGLE_WARNING      = 25.0   # flag as poor technique

# Pass distance thresholds (normalised pitch units).
SHORT_PASS_THRESHOLD      = 0.15   # < 15% of pitch width
LONG_PASS_THRESHOLD       = 0.35   # > 35% of pitch width

# Touch tightness — proximity of ball to player body (normalised).
TOUCH_TIGHT_ADVANCED      = 0.04   # very close control
TOUCH_TIGHT_BEGINNER      = 0.12   # loose control


# ============================================================
# ACTIVITY DETECTION THRESHOLDS
# ============================================================

# Minimum confidence score for each action type to be included.
SHOOTING_MIN_CONFIDENCE    = 0.20
PASSING_MIN_CONFIDENCE     = 0.20
DRIBBLING_MIN_CONFIDENCE   = 0.15
DEFENDING_MIN_CONFIDENCE   = 0.15
GOALKEEPING_MIN_CONFIDENCE = 0.20
MOVEMENT_MIN_CONFIDENCE    = 0.15


# ============================================================
# SKILL CLASSIFICATION — PER METRIC THRESHOLDS
# ============================================================
# Format: (advanced_threshold, beginner_threshold, higher_is_better)

SKILL_THRESHOLDS: dict[str, tuple[float, float, bool]] = {
    "torso_lean":           (TORSO_LEAN_ADVANCED_DEG, TORSO_LEAN_BEGINNER_DEG, False),
    "knee_dev":             (KNEE_DEV_ADVANCED,        KNEE_DEV_BEGINNER,       False),
    "gait_asymmetry":       (GAIT_ASYMMETRY_ADVANCED,  GAIT_ASYMMETRY_BEGINNER, False),
    "leg_speed":            (50.0,  20.0,  True),   # pixels/frame
    "movement_consistency": (5.0,   15.0,  False),  # std-dev of torso lean (deg)
    "pass_accuracy":        (85.0,  60.0,  True),   # %
    "shot_accuracy":        (70.0,  40.0,  True),   # %
    "dribble_success_rate": (75.0,  45.0,  True),   # %
    "tackle_success_rate":  (70.0,  40.0,  True),   # %
}

# Metric weights for the weighted overall score (must sum to 1.0).
SKILL_WEIGHTS: dict[str, float] = {
    "torso_lean":           0.25,
    "knee_dev":             0.20,
    "gait_asymmetry":       0.15,
    "leg_speed":            0.15,
    "movement_consistency": 0.10,
    "pass_accuracy":        0.05,
    "shot_accuracy":        0.05,
    "dribble_success_rate": 0.03,
    "tackle_success_rate":  0.02,
}


# ============================================================
# PLAYER MOVEMENT THRESHOLDS
# ============================================================

SPRINT_THRESHOLD_RATIO    = 0.70   # fraction of max speed = sprint
SPRINT_SPEED_PX_F         = 0.020  # normalised units/frame (≈ fast run)
STATIONARY_SPEED_PX_F     = 0.002  # below this = player not moving
ACCEL_THRESHOLD           = 0.005  # normalised units/frame² = acceleration burst
DECEL_THRESHOLD           = -0.005 # deceleration event


# ============================================================
# QUALITY GATES
# ============================================================

# Minimum data required before each pipeline stage proceeds.
MIN_FRAMES_FOR_GAIT        = 10    # minimum frames for gait analysis
MIN_FRAMES_FOR_METRICS     = 5     # minimum detected pose frames
MIN_PASSES_FOR_ACCURACY    = 3     # minimum pass events for accuracy calculation
MIN_SHOTS_FOR_ACCURACY     = 2     # minimum shot events
MIN_STRIDES_FOR_ASYMMETRY  = 4     # minimum strides each side

# Goalkeeper
GK_MIN_SAVE_EVENTS         = 1     # minimum saves to compute GK metrics

# Confidence gates
PLAYER_DETECTION_MIN       = 0.10  # minimum player detection confidence to proceed
POSE_DETECTION_RATE_MIN    = 0.20  # minimum fraction of frames with valid pose
