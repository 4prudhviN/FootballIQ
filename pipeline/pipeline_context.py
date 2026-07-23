#!/usr/bin/env python3
"""
Pipeline Context
================
The heart of the FootballIQ project.

Every pipeline stage reads from and writes to this single object.
No module passes raw dicts between stages — everything goes through PipelineContext.

Structure:
  PipelineContext
    ├── Video Information   (video_path, fps, duration, resolution)
    ├── Frames              (original_frames, processed_frames)
    ├── Detection Results   (player_tracks, ball_tracks, pose_landmarks)
    ├── Activity Understanding (detected_activities, confidence_scores, timeline)
    ├── Analysis            (metrics, strengths, weaknesses)
    ├── Coaching            (player_level, recommendations, drills, coach_tips)
    └── Final Report        (report, overlays, output_json)

Usage::

    ctx = PipelineContext(video_path="clip.mp4")

    # Each stage writes its output directly to the context.
    ctx.fps           = 25.0
    ctx.player_tracks = [...]
    ctx.detected_activities = ["shooting", "passing"]

    # Any stage can read any previous stage's output.
    print(ctx.player_level)
    print(ctx.focus_this_week)

    # Serialise everything for the API response.
    payload = ctx.to_api_response()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Sub-section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    """Video file metadata."""
    video_path:    str    = ""
    fps:           float  = 25.0
    duration_s:    float  = 0.0
    width:         int    = 0
    height:        int    = 0
    frame_count:   int    = 0
    file_size_mb:  float  = 0.0

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def resolution_label(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class FrameStore:
    """Holds raw and processed frame references."""
    original_frames:   List[Any] = field(default_factory=list)   # List[ExtractedFrame]
    processed_frames:  List[Any] = field(default_factory=list)   # annotated BGR arrays
    key_frame_indices: List[int] = field(default_factory=list)   # frames selected for overlay

    @property
    def frame_count(self) -> int:
        return len(self.original_frames)


@dataclass
class PlayerTrack:
    """Tracking data for a single player across frames."""
    player_id:    str
    frame_index:  int
    bbox:         Tuple[float, float, float, float]   # x, y, w, h (normalised)
    confidence:   float


@dataclass
class BallTrack:
    """Ball position at a single frame."""
    frame_index:  int
    timestamp_s:  float
    center_x:     float
    center_y:     float
    radius:       float
    confidence:   float


@dataclass
class PoseLandmarkFrame:
    """Pose landmarks for one frame."""
    frame_index:  int
    timestamp_s:  float
    detected:     bool
    landmarks:    Dict[str, Any] = field(default_factory=dict)   # name → Landmark
    torso_lean:   Optional[float] = None
    knee_dev_l:   Optional[float] = None
    knee_dev_r:   Optional[float] = None


@dataclass
class DetectionResults:
    """Outputs from Stages 2b–2d: player, ball, pose."""
    player_tracks:     List[PlayerTrack]      = field(default_factory=list)
    ball_tracks:       List[BallTrack]        = field(default_factory=list)
    pose_landmarks:    List[PoseLandmarkFrame] = field(default_factory=list)
    warnings:          List[str]              = field(default_factory=list)
    avg_torso_lean:    Optional[float]        = None
    avg_knee_dev:      Optional[float]        = None
    gait_asymmetry:    Optional[float]        = None
    player_confidence: float                  = 0.0
    ball_confidence:   float                  = 0.0
    pose_detected_frames: int                 = 0
    pose_total_frames:    int                 = 0


@dataclass
class ActivitySegmentCtx:
    """One activity segment in the timeline."""
    action:       str
    start_time_s: float
    end_time_s:   float
    duration_s:   float
    confidence:   float
    label:        str     # "00:00–00:25  Passing"


@dataclass
class ActivityUnderstanding:
    """Outputs from Stage 3: activity detection and timeline."""
    detected_activities: List[str]              = field(default_factory=list)
    confidence_scores:   Dict[str, float]        = field(default_factory=dict)
    timeline:            List[ActivitySegmentCtx] = field(default_factory=list)
    primary_activity:    Optional[str]           = None
    raw_detection_count: int                     = 0


@dataclass
class AnalysisResults:
    """Outputs from Stages 4–5: per-activity metrics."""
    metrics:     Dict[str, Any]   = field(default_factory=dict)
    # {
    #   "byAction": {"shooting": {"Shot Velocity": "88 km/h"}, ...},
    #   "torsoLean": float,
    #   "kneeStability": float,
    #   "gaitSymmetry": float,
    # }
    strengths:   List[str]        = field(default_factory=list)
    weaknesses:  List[str]        = field(default_factory=list)
    metric_scores: Dict[str, float] = field(default_factory=dict)   # 0.0–1.0 per metric
    overall_score: float          = 0.0


@dataclass
class CoachingOutput:
    """Outputs from Stages 6–7: coach engine + recommendation engine."""
    player_level:      str                    = "Beginner"
    recommendations:   List[str]              = field(default_factory=list)
    drills:            List[Dict[str, Any]]   = field(default_factory=list)
    coach_tips:        List[str]              = field(default_factory=list)
    focus_this_week:   List[str]              = field(default_factory=list)
    training_plan:     Dict[str, Any]         = field(default_factory=dict)
    weekly_plan:       Dict[str, Any]         = field(default_factory=dict)
    recovery_advice:   Dict[str, Any]         = field(default_factory=dict)
    motivational_tip:  str                    = ""


@dataclass
class FinalReport:
    """Outputs from Stage 9: persisted report data."""
    report:       Dict[str, Any]  = field(default_factory=dict)    # full ai_feedback dict
    overlays:     Dict[str, str]  = field(default_factory=dict)    # {type: file_path}
    charts:       Dict[str, str]  = field(default_factory=dict)    # {type: file_path}
    output_json:  Dict[str, Any]  = field(default_factory=dict)    # full API response payload
    session_id:   str             = ""
    video_url:    Optional[str]   = None
    created_at:   str             = ""


# ---------------------------------------------------------------------------
# Master PipelineContext
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """
    The central data object for the FootballIQ pipeline.

    Every stage reads from and writes to this object.
    No stage passes raw dicts between modules — use this instead.

    Parameters
    ----------
    video_path : str — path to the input video file

    Example::

        ctx = PipelineContext(video_path="clip.mp4")

        # Stage 2 writes:
        ctx.video.fps         = 25.0
        ctx.video.frame_count = 750
        ctx.detections.warnings = ["POOR POSTURE / LEANING BACK"]

        # Stage 3 writes:
        ctx.activity.detected_activities = ["shooting", "passing"]

        # Stage 6 writes:
        ctx.coaching.player_level = "Intermediate"
        ctx.coaching.focus_this_week = ["Passing accuracy", "First touch"]

        # Serialise for the API:
        payload = ctx.to_api_response()
    """

    # Required at construction time.
    video_path: str = ""

    # Sub-sections — populated by each pipeline stage.
    video:     VideoInfo             = field(default_factory=VideoInfo)
    frames:    FrameStore            = field(default_factory=FrameStore)
    detections: DetectionResults     = field(default_factory=DetectionResults)
    activity:  ActivityUnderstanding = field(default_factory=ActivityUnderstanding)
    analysis:  AnalysisResults       = field(default_factory=AnalysisResults)
    coaching:  CoachingOutput        = field(default_factory=CoachingOutput)
    report:    FinalReport           = field(default_factory=FinalReport)

    # Pipeline metadata.
    session_id:   str            = ""
    status:       str            = "pending"   # pending | running | complete | failed
    error:        Optional[str]  = None
    stage_log:    List[str]      = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.video_path and not self.video.video_path:
            self.video.video_path = self.video_path

    # ------------------------------------------------------------------
    # Stage logging
    # ------------------------------------------------------------------

    def log_stage(self, stage: str, message: str) -> None:
        """Record a stage completion message."""
        self.stage_log.append(f"[{stage}] {message}")

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self.video.fps

    @property
    def duration_s(self) -> float:
        return self.video.duration_s

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.video.resolution

    @property
    def detected_activities(self) -> List[str]:
        return self.activity.detected_activities

    @property
    def player_level(self) -> str:
        return self.coaching.player_level

    @property
    def warnings(self) -> List[str]:
        return self.detections.warnings

    @property
    def focus_this_week(self) -> List[str]:
        return self.coaching.focus_this_week

    @property
    def primary_activity(self) -> Optional[str]:
        return self.activity.primary_activity

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_api_response(self) -> Dict[str, Any]:
        """
        Serialise the full context into the API response shape.
        This is the single output of the entire pipeline.
        """
        return {
            "status":    self.status,
            "session_id": self.session_id,
            "error":     self.error,

            # Video
            "video_url": self.report.video_url,
            "resolution": self.video.resolution_label,
            "duration_s": self.video.duration_s,

            # Activity understanding
            "detectedActivities": self.activity.detected_activities,
            "timeline":           [
                {
                    "label":      seg.label,
                    "action":     seg.action,
                    "startTime":  seg.start_time_s,
                    "endTime":    seg.end_time_s,
                    "duration":   seg.duration_s,
                    "confidence": seg.confidence,
                }
                for seg in self.activity.timeline
            ],

            # Metrics
            "playerLevel": self.coaching.player_level,
            "metrics":     self.analysis.metrics,

            # AI feedback (LLM rewrite of structured coaching output)
            "aiFeedback": self.report.report,

            # Drills
            "drills": self.coaching.drills,

            # Recommendation engine output
            "focusThisWeek":  self.coaching.focus_this_week,
            "trainingPlan":   self.coaching.training_plan,
            "weeklyPlan":     self.coaching.weekly_plan,
            "recoveryAdvice": self.coaching.recovery_advice,

            # Skill profile
            "skillProfile": {
                "level":        self.coaching.player_level,
                "overallScore": self.analysis.overall_score,
                "strengths":    self.analysis.strengths,
                "weaknesses":   self.analysis.weaknesses,
                "metricScores": self.analysis.metric_scores,
            },

            # Visual assets
            "overlays": self.report.overlays,
            "charts":   self.report.charts,

            # Internal diagnostics
            "_stageLog": self.stage_log,
        }

    def to_session_dict(self) -> Dict[str, Any]:
        """Serialise for session persistence (reports/json/)."""
        d = self.to_api_response()
        d["created_at"] = self.report.created_at or datetime.utcnow().isoformat() + "Z"
        return d

    def is_complete(self) -> bool:
        return self.status == "complete"

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error  = error
