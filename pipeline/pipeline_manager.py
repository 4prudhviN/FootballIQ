#!/usr/bin/env python3
"""
Pipeline Manager
================
Orchestrates the full FootballIQ analysis pipeline in the correct order,
passes data between stages, and returns a single PipelineOutput object
that server.py can serialise directly to JSON.

Pipeline order:
  0. VideoLoader          — validate and open video
  1. FrameExtractor       — sample frames
  2. PlayerDetector       — confirm player is visible
  3. BallDetector         — detect ball presence
  4. PoseEstimator        — MediaPipe landmark extraction
  5. ActivityDetector     — classify football actions
  6. AnalyzerSelector     — route to activity-specific analyzers
  7. MetricCalculator     — compute per-action metrics
  8. SkillClassifier      — Beginner / Intermediate / Advanced
  9. FeedbackEngine       — drills + coach tips

Usage::

    manager = PipelineManager()
    output  = manager.run("path/to/clip.mp4")
    print(output.player_level)
    print(output.detected_activities)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.video_loader      import VideoLoader, VideoContext
from pipeline.frame_extractor   import FrameExtractor, ExtractedFrame
from pipeline.player_detector   import PlayerDetector, PlayerDetectionResult
from pipeline.ball_detector     import BallDetector, BallDetectionResult
from pipeline.pose_estimator    import PoseEstimator, PoseEstimationResult
from pipeline.activity_detector import ActivityDetector, ActivityDetectionResult

# Downstream modules (root-level).
from skill_classifier import PlayerMetrics, classify_skill
from feedback_engine  import FeedbackEngine, FeedbackRequest


# ---------------------------------------------------------------------------
# Pipeline Output
# ---------------------------------------------------------------------------

@dataclass
class StageTimings:
    """Wall-clock time (seconds) spent in each pipeline stage."""
    video_load:       float = 0.0
    frame_extract:    float = 0.0
    player_detect:    float = 0.0
    ball_detect:      float = 0.0
    pose_estimate:    float = 0.0
    activity_detect:  float = 0.0
    metric_calc:      float = 0.0
    skill_classify:   float = 0.0
    feedback_engine:  float = 0.0
    total:            float = 0.0


@dataclass
class PipelineOutput:
    """
    Single output object returned by PipelineManager.run().
    Maps directly to the FootballSession interface in types.ts.
    """
    # Status
    success:              bool
    error:                Optional[str]

    # Session fields
    detected_activities:  List[str]
    player_level:         str

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "byAction":      { "shooting": { "Shot Velocity": "88 km/h", ... }, ... },
    #   "torsoLean":     float,
    #   "kneeStability": float,
    #   "gaitSymmetry":  float,
    #   "warnings":      [str, ...]
    # }

    # AI feedback
    ai_feedback: Dict[str, Any] = field(default_factory=dict)
    # { "summary", "strengths", "weaknesses", "coachingTips", "motivationalTip" }

    # Drills
    drills: List[Dict[str, Any]] = field(default_factory=list)

    # Internal diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    timings:     StageTimings   = field(default_factory=StageTimings)


# ---------------------------------------------------------------------------
# Default per-action metric lookup (replaced by real analyzers in future)
# ---------------------------------------------------------------------------

_ACTIVITY_METRICS: Dict[str, Dict[str, str]] = {
    "passing":     {"Ball Control": "92%", "First Touch": "0.36 m/s²",
                    "Pass Accuracy": "87%", "Weight of Pass": "Medium"},
    "dribbling":   {"Close Control": "88%", "Change of Direction": "5.8 m/s",
                    "Touch Tightness": "±2.4 cm", "Speed with Ball": "24 km/h"},
    "shooting":    {"Shot Velocity": "88 km/h", "Launch Angle": "14°",
                    "Target Accuracy": "81%", "Torso Alignment": "12°"},
    "goalkeeping": {"Reaction Time": "0.28s", "Diving Range": "2.4m",
                    "Distribution": "74%", "Positioning": "Good"},
    "defending":   {"Tackle Timing": "Good", "Positioning": "88%",
                    "Interception": "3", "Aerial Duels": "67%"},
    "movement":    {"Gait Symmetry": "92%", "Stride Length": "1.24m",
                    "Sprint Speed": "31.2 km/h", "Agility": "4.2s"},
}


# ---------------------------------------------------------------------------
# Pipeline Manager
# ---------------------------------------------------------------------------

class PipelineManager:
    """
    Orchestrates the full FootballIQ analysis pipeline.

    Parameters
    ----------
    frame_stride : int
        Sample every Nth frame (default 3 — balances speed vs accuracy).
    player_threshold : float
        Minimum player detection confidence to proceed.
    pose_model_complexity : int
        MediaPipe model complexity (0=lite, 1=full, 2=heavy).
    """

    def __init__(
        self,
        frame_stride:          int   = 3,
        player_threshold:      float = 0.10,
        pose_model_complexity: int   = 1,
    ) -> None:
        self.frame_stride          = frame_stride
        self.player_threshold      = player_threshold
        self.pose_model_complexity = pose_model_complexity

    def run(self, video_path: str) -> PipelineOutput:
        """
        Run the complete pipeline for a video file.

        Parameters
        ----------
        video_path : str
            Path to the input MP4/MOV/AVI file.

        Returns
        -------
        PipelineOutput
            Always returns — errors are captured in output.success / output.error.
        """
        t0      = time.perf_counter()
        timings = StageTimings()
        context: Optional[VideoContext] = None
        estimator: Optional[PoseEstimator] = None

        try:
            # ── Stage 0: Video Load ───────────────────────────────────────────
            t = time.perf_counter()
            loader  = VideoLoader()
            context = loader.load(video_path)
            timings.video_load = time.perf_counter() - t
            print(f"[Stage 0] Video loaded  {context.width}x{context.height} "
                  f"@ {context.fps:.1f}fps  {context.frame_count} frames")

            # ── Stage 1: Frame Extraction ─────────────────────────────────────
            t = time.perf_counter()
            extractor = FrameExtractor(stride=self.frame_stride)
            frames: List[ExtractedFrame] = extractor.extract_all(context)
            timings.frame_extract = time.perf_counter() - t
            print(f"[Stage 1] Extracted {len(frames)} frames (stride={self.frame_stride})")

            # ── Stage 2: Player Detection ─────────────────────────────────────
            t = time.perf_counter()
            player_det_result: PlayerDetectionResult = (
                PlayerDetector(threshold=self.player_threshold).detect(frames)
            )
            timings.player_detect = time.perf_counter() - t
            print(f"[Stage 2] Player detection: conf={player_det_result.confidence:.1%} "
                  f"passed={player_det_result.passed}")

            if not player_det_result.passed:
                return PipelineOutput(
                    success=False,
                    error=(
                        f"No player detected in video "
                        f"(confidence {player_det_result.confidence:.1%} < "
                        f"{self.player_threshold:.0%} threshold). "
                        "Ensure the player is clearly visible in the frame."
                    ),
                    detected_activities=[],
                    player_level="Beginner",
                )

            # ── Stage 3: Ball Detection ───────────────────────────────────────
            t = time.perf_counter()
            ball_det_result: BallDetectionResult = BallDetector().detect(frames)
            timings.ball_detect = time.perf_counter() - t
            print(f"[Stage 3] Ball detection: conf={ball_det_result.confidence:.1%} "
                  f"found={ball_det_result.ball_detected}")

            # ── Stage 4: Pose Estimation ──────────────────────────────────────
            t = time.perf_counter()
            estimator = PoseEstimator(model_complexity=self.pose_model_complexity)
            pose_result: PoseEstimationResult = estimator.estimate(frames)
            timings.pose_estimate = time.perf_counter() - t
            print(f"[Stage 4] Pose: {pose_result.detected_frames}/{pose_result.total_frames} "
                  f"frames  warnings={pose_result.warnings}")

            # ── Stage 5: Activity Detection ───────────────────────────────────
            t = time.perf_counter()
            activity_result: ActivityDetectionResult = (
                ActivityDetector().detect(pose_result, ball_det_result)
            )
            timings.activity_detect = time.perf_counter() - t
            activities = activity_result.names
            print(f"[Stage 5] Activities: {activities}")

            # ── Stage 6 & 7: Analyzer Selection + Metric Calculation ──────────
            t = time.perf_counter()
            torso_lean     = pose_result.avg_torso_lean or 8.0
            knee_stability = max(0.0, 100.0 - (pose_result.avg_knee_dev or 0.0) * 100)
            gait_symmetry  = max(0.0, 100.0 - (pose_result.gait_asymmetry or 0.0) * 100)
            by_action      = {a: _ACTIVITY_METRICS.get(a, {}) for a in activities}
            timings.metric_calc = time.perf_counter() - t
            print(f"[Stage 6-7] Metrics: torso={torso_lean:.1f}° "
                  f"knee={knee_stability:.0f}% gait={gait_symmetry:.0f}%")

            # ── Stage 8: Skill Classification ─────────────────────────────────
            t = time.perf_counter()
            pm = PlayerMetrics(
                torso_lean      = torso_lean,
                knee_dev        = 1.0 - knee_stability / 100.0,
                gait_asymmetry  = 1.0 - gait_symmetry  / 100.0,
            )
            skill_report = classify_skill(pm)
            player_level = skill_report.level.value
            timings.skill_classify = time.perf_counter() - t
            print(f"[Stage 8] Skill level: {player_level} "
                  f"(score={skill_report.overall_score:.2f})")

            # ── Stage 9: Feedback Engine ───────────────────────────────────────
            t = time.perf_counter()
            engine  = FeedbackEngine()
            primary = activity_result.primary or "general"
            fb_req  = FeedbackRequest(
                metrics  = {
                    "torso_lean":     torso_lean,
                    "knee_dev":       1.0 - knee_stability / 100.0,
                    "gait_asymmetry": 1.0 - gait_symmetry  / 100.0,
                },
                activity = primary,
                level    = player_level,
            )
            fb_report = engine.generate(fb_req)
            timings.feedback_engine = time.perf_counter() - t
            print(f"[Stage 9] Feedback: {len(fb_report.items)} issues, "
                  f"{len(fb_report.items)} drills")

            # ── Assemble output ────────────────────────────────────────────────
            timings.total = time.perf_counter() - t0

            drills = [
                {
                    "name":         item.drill.split(":")[0].strip(),
                    "targetMetric": item.metric,
                    "instructions": item.drill,
                    "coachTip":     item.coach_tip,
                    "duration":     "10-15 min",
                    "difficulty":   player_level,
                }
                for item in fb_report.items
            ]

            return PipelineOutput(
                success              = True,
                error                = None,
                detected_activities  = activities,
                player_level         = player_level,
                metrics              = {
                    "byAction":      by_action,
                    "torsoLean":     round(torso_lean, 1),
                    "kneeStability": round(knee_stability, 1),
                    "gaitSymmetry":  round(gait_symmetry, 1),
                    "warnings":      pose_result.warnings,
                },
                ai_feedback          = {
                    "summary":          fb_report.summary,
                    "strengths":        fb_report.positive,
                    "weaknesses":       [i.metric.replace("_", " ").title()
                                         for i in fb_report.items],
                    "coachingTips":     [i.coach_tip for i in fb_report.items],
                    "motivationalTip":  fb_report.motivational_tip,
                },
                drills               = drills,
                diagnostics          = {
                    "player_detection": {
                        "confidence": player_det_result.confidence,
                        "passed":     player_det_result.passed,
                    },
                    "ball_detection": {
                        "confidence":   ball_det_result.confidence,
                        "ball_detected": ball_det_result.ball_detected,
                    },
                    "pose": {
                        "detected_frames": pose_result.detected_frames,
                        "total_frames":    pose_result.total_frames,
                        "avg_torso_lean":  pose_result.avg_torso_lean,
                        "avg_knee_dev":    pose_result.avg_knee_dev,
                        "gait_asymmetry":  pose_result.gait_asymmetry,
                    },
                    "skill": {
                        "overall_score":  skill_report.overall_score,
                        "metric_scores":  skill_report.metric_scores,
                        "strengths":      skill_report.strengths,
                        "weaknesses":     skill_report.weaknesses,
                    },
                    "activity": {
                        "activities": [
                            {"name": a.name, "confidence": a.confidence,
                             "evidence": a.evidence}
                            for a in activity_result.activities
                        ]
                    },
                },
                timings = timings,
            )

        except Exception as exc:
            timings.total = time.perf_counter() - t0
            return PipelineOutput(
                success=False,
                error=str(exc),
                detected_activities=[],
                player_level="Beginner",
                timings=timings,
            )

        finally:
            if estimator:
                estimator.close()
            if context:
                context.release()
