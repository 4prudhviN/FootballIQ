#!/usr/bin/env python3
"""
Analysis Schema
===============
Typed data contracts for the full video analysis pipeline output.

This is the central schema of the application.
Every pipeline stage and server endpoint must return data
that conforms to these types — never raw dicts.

Maps directly to the FootballSession interface in types.ts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from schemas.player_schema   import PlayerProfile, PlayerSkillProfile, SkillLevel
from schemas.activity_schema import DetectedActivity, ActionMetrics, FootballAction


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SessionStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class PipelineStage(str, Enum):
    VIDEO_LOAD       = "video_load"
    FRAME_EXTRACT    = "frame_extract"
    PLAYER_DETECT    = "player_detect"
    BALL_DETECT      = "ball_detect"
    POSE_ESTIMATE    = "pose_estimate"
    ACTIVITY_DETECT  = "activity_detect"
    METRIC_CALC      = "metric_calc"
    SKILL_CLASSIFY   = "skill_classify"
    FEEDBACK_ENGINE  = "feedback_engine"
    COMPLETE         = "complete"


# ---------------------------------------------------------------------------
# Biomechanical metrics schema
# ---------------------------------------------------------------------------

@dataclass
class BiomechanicalMetrics:
    """
    Core biomechanical scalar metrics from the pose estimator.
    These are universal — measured regardless of the detected activity.
    """
    torso_lean:     float                             # degrees (neg = lean back)
    knee_stability: float                             # score 0–100
    gait_symmetry:  float                             # score 0–100
    warnings:       List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "torsoLean":     round(self.torso_lean, 1),
            "kneeStability": round(self.knee_stability, 1),
            "gaitSymmetry":  round(self.gait_symmetry, 1),
            "warnings":      self.warnings,
        }


# ---------------------------------------------------------------------------
# Per-session metrics schema
# ---------------------------------------------------------------------------

@dataclass
class SessionMetrics:
    """
    All metrics for one analysis session — biomechanical scalars
    plus per-action display metrics.
    """
    biomechanical:  BiomechanicalMetrics
    by_action:      List[ActionMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            **self.biomechanical.to_dict(),
            "byAction": {
                am.action.value: am.to_display_dict()
                for am in self.by_action
            },
        }


# ---------------------------------------------------------------------------
# Pipeline diagnostics schema
# ---------------------------------------------------------------------------

@dataclass
class PipelineDiagnostics:
    """Internal diagnostic data from each pipeline stage."""
    player_detection_confidence: float = 0.0
    ball_detection_confidence:   float = 0.0
    ball_detected:                bool = False
    pose_detected_frames:        int   = 0
    pose_total_frames:           int   = 0
    stage_timings_s:             Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "playerDetectionConfidence": round(self.player_detection_confidence, 3),
            "ballDetectionConfidence":   round(self.ball_detection_confidence, 3),
            "ballDetected":              self.ball_detected,
            "poseDetectedFrames":        self.pose_detected_frames,
            "poseTotalFrames":           self.pose_total_frames,
            "stageTimingsS":             {k: round(v, 3) for k, v in self.stage_timings_s.items()},
        }


# ---------------------------------------------------------------------------
# Full analysis session schema  (maps to FootballSession in types.ts)
# ---------------------------------------------------------------------------

@dataclass
class AnalysisSession:
    """
    The master output object for one complete analysis run.
    Returned by pipeline_manager.py and serialised by server.py.

    Maps directly to FootballSession in src/types.ts.
    """

    # Identity
    session_id:           str
    file_name:            str
    created_at:           datetime               = field(default_factory=datetime.utcnow)
    status:               SessionStatus          = SessionStatus.PENDING
    current_stage:        PipelineStage          = PipelineStage.VIDEO_LOAD
    error:                Optional[str]          = None

    # Player
    player:               Optional[PlayerProfile]      = None
    skill_profile:        Optional[PlayerSkillProfile] = None

    # Activities
    detected_activities:  List[DetectedActivity]       = field(default_factory=list)

    # Metrics
    metrics:              Optional[SessionMetrics]     = None

    # AI report (populated by ExplanationEngine)
    ai_summary:           str                    = ""
    ai_strengths:         List[str]              = field(default_factory=list)
    ai_weaknesses:        List[str]              = field(default_factory=list)
    ai_coaching_tips:     List[str]              = field(default_factory=list)
    ai_motivational_tip:  str                    = ""
    ai_drills:            List["DrillSchema"]    = field(default_factory=list)

    # Output video
    video_url:            Optional[str]          = None

    # Internal diagnostics
    diagnostics:          Optional[PipelineDiagnostics] = None

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def player_level(self) -> str:
        if self.skill_profile:
            return self.skill_profile.level.value
        return SkillLevel.BEGINNER.value

    @property
    def activity_names(self) -> List[str]:
        return [a.action.value for a in self.detected_activities]

    @property
    def primary_activity(self) -> Optional[str]:
        if self.detected_activities:
            best = max(self.detected_activities, key=lambda a: a.confidence)
            return best.action.value
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise to a JSON-compatible dict.
        Shape mirrors the FootballSession interface in types.ts.
        """
        return {
            # Status
            "status":    self.status.value,
            "session_id": self.session_id,
            "file_name":  self.file_name,
            "error":      self.error,

            # FootballSession fields
            "detectedActivities": self.activity_names,
            "playerLevel":        self.player_level,

            "metrics": self.metrics.to_dict() if self.metrics else {
                "byAction": {}, "torsoLean": 0,
                "kneeStability": 0, "gaitSymmetry": 0, "warnings": [],
            },

            "aiFeedback": {
                "summary":         self.ai_summary,
                "strengths":       self.ai_strengths,
                "weaknesses":      self.ai_weaknesses,
                "coachingTips":    self.ai_coaching_tips,
                "motivationalTip": self.ai_motivational_tip,
            },

            "drills": [d.to_dict() for d in self.ai_drills],

            "video_url": self.video_url,

            "_diagnostics": self.diagnostics.to_dict() if self.diagnostics else {},
        }


# ---------------------------------------------------------------------------
# Drill schema
# ---------------------------------------------------------------------------

@dataclass
class DrillSchema:
    """
    A single training drill assigned to the player.
    Produced by feedback_engine.py / ai/report_generator.py.
    """
    name:          str
    target_metric: str                 # e.g. "torso_lean"
    instructions:  str
    coach_tip:     str
    duration:      str                 # e.g. "10 min"
    difficulty:    SkillLevel          = SkillLevel.BEGINNER

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "targetMetric": self.target_metric,
            "instructions": self.instructions,
            "coachTip":     self.coach_tip,
            "duration":     self.duration,
            "difficulty":   self.difficulty.value,
        }
