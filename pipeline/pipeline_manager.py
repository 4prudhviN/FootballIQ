#!/usr/bin/env python3
"""
Pipeline Manager  (v2)
======================
Orchestrates the full FootballIQ analysis pipeline using all
properly structured modules.

Pipeline order:
  0. VideoLoader              — validate and open video
  1. FrameExtractor           — sample frames
  2. PlayerDetector           — confirm player is visible
  3. BallDetector             — detect ball presence
  4. PoseEstimator            — MediaPipe landmark extraction
  5. ActivityDetector (AU)    — per-frame action scoring
  6. ConfidenceFilter         — clean and deduplicate
  7. ActivityClassifier       — video-level activity ranking
  8. SequenceAnalyzer         — timeline segmentation
  9. AnalyzerRegistry         — run activity-specific metric calculators
 10. CoachSkillClassifier     — Beginner / Intermediate / Advanced
 11. CoachFeedbackEngine      — grounded observations + adapted drills
 12. DrillRecommender         — prioritised drill list
 13. ExplanationEngine (AI)   — optional LLM natural-language wrapping
 14. ReportWriter             — persist JSON report

Usage::

    manager = PipelineManager()
    output  = manager.run("path/to/clip.mp4")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Pipeline stages ─────────────────────────────────────────────────────────
from pipeline.video_loader      import VideoLoader, VideoContext
from pipeline.frame_extractor   import FrameExtractor, ExtractedFrame
from pipeline.player_detector   import PlayerDetector, PlayerDetectionResult
from pipeline.ball_detector     import BallDetector, BallDetectionResult
from pipeline.pose_estimator    import PoseEstimator, PoseEstimationResult

# ── Activity understanding ───────────────────────────────────────────────────
from activity_understanding import (
    ActivityDetector  as AUActivityDetector,
    ActivityClassifier,
    SequenceAnalyzer,
    ConfidenceFilter,
    RawActivityDetection,
    ClassifiedActivity,
    ActivitySegment,
)

# ── Analyzer registry ────────────────────────────────────────────────────────
from analyzers.analyzer_registry import get_registry

# ── Coach engine ─────────────────────────────────────────────────────────────
from coach_engine import (
    CoachSkillClassifier,
    CoachFeedbackEngine,
    DrillRecommender,
)

# ── AI layer (optional) ──────────────────────────────────────────────────────
from ai.explanation_engine import ExplanationEngine

# ── Config ───────────────────────────────────────────────────────────────────
from config.settings   import settings
from config.constants  import DEFAULT_FPS, PLAYER_DETECTION_THRESHOLD

# ── Utils ────────────────────────────────────────────────────────────────────
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Output
# ---------------------------------------------------------------------------

@dataclass
class StageTimings:
    video_load:        float = 0.0
    frame_extract:     float = 0.0
    player_detect:     float = 0.0
    ball_detect:       float = 0.0
    pose_estimate:     float = 0.0
    activity_detect:   float = 0.0
    metric_calc:       float = 0.0
    skill_classify:    float = 0.0
    feedback_engine:   float = 0.0
    ai_explanation:    float = 0.0
    total:             float = 0.0


@dataclass
class PipelineOutput:
    """
    Full analysis output — maps to FootballSession in types.ts.
    """
    success:             bool
    error:               Optional[str]
    detected_activities: List[str]
    player_level:        str
    metrics:             Dict[str, Any]  = field(default_factory=dict)
    ai_feedback:         Dict[str, Any]  = field(default_factory=dict)
    drills:              List[Dict[str, Any]] = field(default_factory=list)
    timeline:            List[Dict[str, Any]] = field(default_factory=list)
    diagnostics:         Dict[str, Any]  = field(default_factory=dict)
    timings:             StageTimings    = field(default_factory=StageTimings)


# ---------------------------------------------------------------------------
# Pipeline Manager
# ---------------------------------------------------------------------------

class PipelineManager:
    """
    Orchestrates all pipeline stages and returns a PipelineOutput.

    Parameters
    ----------
    frame_stride          : int   — sample every Nth frame
    player_threshold      : float — min player detection confidence
    pose_model_complexity : int   — 0=lite 1=full 2=heavy
    use_ai                : bool  — whether to call the LLM layer
    """

    def __init__(
        self,
        frame_stride:          int   = 3,
        player_threshold:      float = PLAYER_DETECTION_THRESHOLD,
        pose_model_complexity: int   = 1,
        use_ai:                bool  = True,
    ) -> None:
        self.frame_stride          = frame_stride
        self.player_threshold      = player_threshold
        self.pose_model_complexity = pose_model_complexity
        self.use_ai                = use_ai

        # Singletons initialised once.
        self._registry  = get_registry()
        self._cf        = ConfidenceFilter()
        self._ai_engine = ExplanationEngine() if use_ai else None

    def run(self, video_path: str) -> PipelineOutput:
        """Run the complete pipeline. Always returns — errors are captured."""
        t0      = time.perf_counter()
        timings = StageTimings()
        context: Optional[VideoContext]  = None
        estimator: Optional[PoseEstimator] = None

        try:
            # ── 0. Video load ────────────────────────────────────────────────
            t = time.perf_counter()
            context = VideoLoader().load(video_path)
            timings.video_load = time.perf_counter() - t
            log.pipeline("video_load", "%dx%d @ %.1f fps  %d frames",
                         context.width, context.height,
                         context.fps, context.frame_count)

            # ── 1. Frame extraction ──────────────────────────────────────────
            t = time.perf_counter()
            frames: List[ExtractedFrame] = FrameExtractor(
                stride=self.frame_stride
            ).extract_all(context)
            timings.frame_extract = time.perf_counter() - t
            log.pipeline("frame_extract", "%d frames (stride=%d)",
                         len(frames), self.frame_stride)

            fps = context.fps or DEFAULT_FPS

            # ── 2. Player detection ──────────────────────────────────────────
            t = time.perf_counter()
            player_result: PlayerDetectionResult = PlayerDetector(
                threshold=self.player_threshold
            ).detect(frames)
            timings.player_detect = time.perf_counter() - t
            log.pipeline("player_detect", "conf=%.1f%%  passed=%s",
                         player_result.confidence * 100, player_result.passed)

            if not player_result.passed:
                return PipelineOutput(
                    success=False,
                    error=(
                        f"No player detected (confidence {player_result.confidence:.1%} "
                        f"< {self.player_threshold:.0%}). "
                        "Ensure the player is fully visible in the frame."
                    ),
                    detected_activities=[],
                    player_level="Beginner",
                )

            # ── 3. Ball detection ────────────────────────────────────────────
            t = time.perf_counter()
            ball_result: BallDetectionResult = BallDetector().detect(frames)
            timings.ball_detect = time.perf_counter() - t
            log.pipeline("ball_detect", "conf=%.1f%%  found=%s",
                         ball_result.confidence * 100, ball_result.ball_detected)

            # ── 4. Pose estimation ───────────────────────────────────────────
            t = time.perf_counter()
            estimator = PoseEstimator(
                model_complexity=self.pose_model_complexity
            )
            pose_result: PoseEstimationResult = estimator.estimate(frames)
            timings.pose_estimate = time.perf_counter() - t
            log.pipeline("pose_estimate", "%d/%d frames  warnings=%s",
                         pose_result.detected_frames,
                         pose_result.total_frames,
                         pose_result.warnings)

            # ── 5–8. Activity understanding ───────────────────────────────────
            t = time.perf_counter()

            # 5. Per-frame detection
            raw_dets: List[RawActivityDetection] = AUActivityDetector().detect(
                pose_result, ball_result
            )

            # 6. Confidence filtering + deduplication
            raw_dets = self._cf.filter_raw(raw_dets)
            raw_dets = self._cf.deduplicate_frame(raw_dets)

            # 7. Video-level classification
            classified: List[ClassifiedActivity] = ActivityClassifier.classify(raw_dets)
            classified = self._cf.filter_classified(classified)
            classified = self._cf.normalise(classified)

            # 8. Timeline segmentation
            timeline: List[ActivitySegment] = SequenceAnalyzer.analyze(raw_dets, fps=fps)

            activities = [c.action for c in classified] if classified else ["passing"]
            if not activities:
                activities = ["passing"]

            timings.activity_detect = time.perf_counter() - t
            log.pipeline("activity_detect", "activities=%s  segments=%d",
                         activities, len(timeline))

            # ── 9. Metric calculation (analyzer registry) ────────────────────
            t = time.perf_counter()
            action_metrics = self._registry.run_for_activities(
                activities, frames, pose_result, ball_result
            )
            by_action = {
                action: am.to_display_dict()
                for action, am in action_metrics.items()
            }

            # Core biomechanical scalars from pose.
            torso_lean     = pose_result.avg_torso_lean  or 8.0
            knee_stability = max(0.0, 100.0 - (pose_result.avg_knee_dev  or 0.0) * 100)
            gait_symmetry  = max(0.0, 100.0 - (pose_result.gait_asymmetry or 0.0) * 100)

            timings.metric_calc = time.perf_counter() - t
            log.pipeline("metric_calc", "torso=%.1f°  knee=%.0f  gait=%.0f",
                         torso_lean, knee_stability, gait_symmetry)

            # ── 10. Skill classification ──────────────────────────────────────
            t = time.perf_counter()
            raw_metric_dict = {
                "torso_lean":     abs(torso_lean),
                "knee_dev":       1.0 - knee_stability / 100.0,
                "gait_asymmetry": 1.0 - gait_symmetry  / 100.0,
            }
            skill_profile = CoachSkillClassifier().classify(raw_metric_dict)
            player_level  = skill_profile.level
            timings.skill_classify = time.perf_counter() - t
            log.pipeline("skill_classify", "level=%s  score=%.3f",
                         player_level, skill_profile.overall_score)

            # ── 11. Feedback engine ───────────────────────────────────────────
            t = time.perf_counter()
            primary_activity = activities[0] if activities else "general"
            feedback_report  = CoachFeedbackEngine().generate(
                skill_profile,
                activity=primary_activity,
                metrics=raw_metric_dict,
            )
            timings.feedback_engine = time.perf_counter() - t
            log.pipeline("feedback_engine", "%d issues  level=%s",
                         len(feedback_report.items), player_level)

            # ── 12. Drill recommendations ─────────────────────────────────────
            drills_list = DrillRecommender().recommend(skill_profile, activity=primary_activity)

            drill_dicts = [
                {
                    "name":         d.name,
                    "targetMetric": d.target_metric,
                    "instructions": d.instructions,
                    "coachTip":     d.coach_tip,
                    "duration":     d.duration,
                    "difficulty":   d.difficulty,
                    "priority":     d.priority,
                }
                for d in drills_list
            ]

            # ── 13. AI explanation (optional) ─────────────────────────────────
            ai_feedback: Dict[str, Any] = {
                "summary":         feedback_report.summary,
                "strengths":       feedback_report.positive,
                "weaknesses":      [i.metric.replace("_", " ").title()
                                    for i in feedback_report.items],
                "coachingTips":    [i.adapted_coach_tip for i in feedback_report.items],
                "motivationalTip": feedback_report.motivational_tip,
            }

            if self._ai_engine and self.use_ai:
                t = time.perf_counter()
                try:
                    ai_report = self._ai_engine.explain(
                        detected_activities = activities,
                        player_level        = player_level,
                        torso_lean          = abs(torso_lean),
                        knee_stability      = knee_stability,
                        gait_symmetry       = gait_symmetry,
                        warnings            = pose_result.warnings,
                        by_action           = by_action,
                        video_duration_s    = context.duration_s,
                    )
                    # Merge AI summary into the deterministic feedback.
                    ai_feedback["summary"]         = ai_report.summary or ai_feedback["summary"]
                    ai_feedback["coachingTips"]    = ai_report.coaching_tips or ai_feedback["coachingTips"]
                    ai_feedback["motivationalTip"] = ai_report.coach_tip   or ai_feedback["motivationalTip"]

                    if not drill_dicts and ai_report.drills:
                        drill_dicts = [
                            {"name": d.name, "instructions": d.instructions,
                             "duration": d.duration, "targetMetric": "", "difficulty": player_level}
                            for d in ai_report.drills
                        ]
                    timings.ai_explanation = time.perf_counter() - t
                    log.pipeline("ai_explanation", "provider=%s  latency=%.2fs",
                                 ai_report.provider, ai_report.latency_s)
                except Exception as ai_exc:
                    log.warning("AI explanation failed (non-fatal): %s", ai_exc)
                    timings.ai_explanation = time.perf_counter() - t

            # ── 14. Assemble output ───────────────────────────────────────────
            timings.total = time.perf_counter() - t0
            log.timing("total", timings.total)

            return PipelineOutput(
                success              = True,
                error                = None,
                detected_activities  = activities,
                player_level         = player_level,
                metrics              = {
                    "byAction":      by_action,
                    "torsoLean":     round(abs(torso_lean), 1),
                    "kneeStability": round(knee_stability, 1),
                    "gaitSymmetry":  round(gait_symmetry, 1),
                    "warnings":      pose_result.warnings,
                },
                ai_feedback          = ai_feedback,
                drills               = drill_dicts,
                timeline             = [
                    {
                        "label":      seg.label,
                        "action":     seg.action,
                        "startTime":  seg.start_time_s,
                        "endTime":    seg.end_time_s,
                        "duration":   seg.duration_s,
                        "confidence": seg.confidence,
                    }
                    for seg in timeline
                ],
                diagnostics          = {
                    "player_detection": {
                        "confidence": player_result.confidence,
                        "passed":     player_result.passed,
                    },
                    "ball_detection": {
                        "confidence":   ball_result.confidence,
                        "ball_detected": ball_result.ball_detected,
                    },
                    "pose": {
                        "detected_frames": pose_result.detected_frames,
                        "total_frames":    pose_result.total_frames,
                    },
                    "skill": {
                        "overall_score": skill_profile.overall_score,
                        "top_gap":       skill_profile.top_gap,
                        "top_strength":  skill_profile.top_strength,
                    },
                    "timings": {
                        "total_s":          round(timings.total, 2),
                        "pose_s":           round(timings.pose_estimate, 2),
                        "activity_s":       round(timings.activity_detect, 2),
                        "feedback_s":       round(timings.feedback_engine, 2),
                        "ai_s":             round(timings.ai_explanation, 2),
                    },
                },
                timings = timings,
            )

        except Exception as exc:
            timings.total = time.perf_counter() - t0
            log.error("Pipeline failed: %s", exc, exc_info=True)
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
