#!/usr/bin/env python3
"""
Pipeline Manager  (v3 — correct architecture)
==============================================

The LLM is the COMMUNICATOR, not the decision-maker.
All football decisions come from measurable data and the coaching knowledge base.

Correct pipeline order:
  1.  Video
  2.  Pipeline (player detect → ball detect → pose estimate)
  3.  Activity Understanding (per-frame detect → classify → timeline → filter)
  4.  Analyzer Selection     (registry routes to correct activity analyzers)
  5.  Metrics                (pure number calculation per activity)
  6.  Coach Engine           (skill classify → feedback → drill recommend)
  7.  Recommendation Engine  (priority select → training plan → recovery advice)
  8.  LLM                    (rewrites structured data into natural language ONLY)
  9.  Report                 (JSON + overlays + charts persisted)

The LLM receives a fully structured coaching report and rewrites it into
plain, motivating English. It adds NO new football decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── 1–2. Core pipeline stages ───────────────────────────────────────────────
from pipeline.video_loader    import VideoLoader, VideoContext
from pipeline.frame_extractor import FrameExtractor, ExtractedFrame
from pipeline.player_detector import PlayerDetector, PlayerDetectionResult
from pipeline.ball_detector   import BallDetector, BallDetectionResult
from pipeline.pose_estimator  import PoseEstimator, PoseEstimationResult
from pipeline.pipeline_context import (
    PipelineContext, VideoInfo, FrameStore, DetectionResults,
    ActivityUnderstanding, ActivitySegmentCtx, AnalysisResults,
    CoachingOutput, FinalReport, BallTrack, PoseLandmarkFrame,
)

# ── 3. Activity understanding ────────────────────────────────────────────────
from activity_understanding import (
    ActivityDetector  as AUActivityDetector,
    ActivityClassifier,
    SequenceAnalyzer,
    ConfidenceFilter,
    RawActivityDetection,
    ClassifiedActivity,
    ActivitySegment,
)

# ── 4. Analyzer selection ────────────────────────────────────────────────────
from analyzers.analyzer_registry import get_registry

# ── 5. (Metrics are calculated inside the registry analyzers) ────────────────

# ── 6. Coach engine ──────────────────────────────────────────────────────────
from coach_engine import (
    CoachSkillClassifier,
    CoachFeedbackEngine,
    DrillRecommender,
    TerminologyAdapter,
)

# ── 7. Recommendation engine ─────────────────────────────────────────────────
from recommendation_engine.priority_selector import PrioritySelector
from recommendation_engine.training_plan     import TrainingPlanGenerator
from recommendation_engine.weekly_plan       import WeeklyPlanGenerator
from recommendation_engine.recovery_advice   import RecoveryAdvisor

# ── 8. LLM — communicator only ───────────────────────────────────────────────
from ai.explanation_engine import ExplanationEngine
from ai.json_validator     import JSONValidator

# ── Config & utils ───────────────────────────────────────────────────────────
from config.settings   import settings
from config.constants  import DEFAULT_FPS, PLAYER_DETECTION_THRESHOLD
from utils.logger      import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Timings
# ---------------------------------------------------------------------------

@dataclass
class StageTimings:
    video_load:         float = 0.0
    frame_extract:      float = 0.0
    player_detect:      float = 0.0
    ball_detect:        float = 0.0
    pose_estimate:      float = 0.0
    activity_understand: float = 0.0
    analyzer_metrics:   float = 0.0
    coach_engine:       float = 0.0
    recommendation:     float = 0.0
    llm_rewrite:        float = 0.0
    total:              float = 0.0


# ---------------------------------------------------------------------------
# Pipeline Output
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    """
    Full analysis output — maps to FootballSession in types.ts.
    The LLM-rewritten natural language is in ai_feedback.
    All football decisions are in the structured fields.
    """
    success:              bool
    error:                Optional[str]

    # ── 3. Activity Understanding ────────────────────────────────────────────
    detected_activities:  List[str]
    timeline:             List[Dict[str, Any]]   # ActivitySegment dicts

    # ── 4–5. Metrics ─────────────────────────────────────────────────────────
    player_level:         str
    metrics:              Dict[str, Any]

    # ── 6. Coach Engine output ────────────────────────────────────────────────
    skill_profile:        Dict[str, Any]         # CoachSkillProfile.to_dict()
    coaching_feedback:    Dict[str, Any]         # CoachFeedbackReport.to_dict()
    drills:               List[Dict[str, Any]]

    # ── 7. Recommendation Engine output ───────────────────────────────────────
    focus_areas:          List[str]              # "Focus this week" labels
    training_plan:        Dict[str, Any]
    weekly_plan:          Dict[str, Any]
    recovery_advice:      Dict[str, Any]

    # ── 8. LLM rewrite (natural language ONLY, no new decisions) ─────────────
    ai_feedback:          Dict[str, Any]

    # ── Meta ──────────────────────────────────────────────────────────────────
    diagnostics:          Dict[str, Any]  = field(default_factory=dict)
    timings:              StageTimings    = field(default_factory=StageTimings)


# ---------------------------------------------------------------------------
# Pipeline Manager
# ---------------------------------------------------------------------------

class PipelineManager:
    """
    Orchestrates the full 9-stage FootballIQ pipeline.

    Architecture principle:
      - Stages 1–7 are deterministic (no LLM)
      - Stage 8 (LLM) only rewrites structured data into natural language
      - Stage 9 (Report) persists the output

    Parameters
    ----------
    frame_stride          : int
    player_threshold      : float
    pose_model_complexity : int
    use_ai                : bool — enable/disable LLM rewrite stage
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

        # Singletons — initialised once.
        self._registry    = get_registry()
        self._cf          = ConfidenceFilter()
        self._terminology = TerminologyAdapter()
        self._validator   = JSONValidator()
        self._ai_engine   = ExplanationEngine() if use_ai else None

    def run(self, video_path: str) -> PipelineOutput:
        """Run the complete pipeline. Always returns — errors are captured."""
        t0      = time.perf_counter()
        timings = StageTimings()
        context:   Optional[VideoContext]   = None
        estimator: Optional[PoseEstimator]  = None

        try:
            # ── STAGE 1: Video load ──────────────────────────────────────────
            t = time.perf_counter()
            context = VideoLoader().load(video_path)
            timings.video_load = time.perf_counter() - t
            fps = context.fps or DEFAULT_FPS
            log.pipeline("video_load", "%dx%d @ %.1f fps  %d frames",
                         context.width, context.height, fps, context.frame_count)

            # ── STAGE 2a: Frame extraction ───────────────────────────────────
            t = time.perf_counter()
            frames: List[ExtractedFrame] = FrameExtractor(
                stride=self.frame_stride
            ).extract_all(context)
            timings.frame_extract = time.perf_counter() - t
            log.pipeline("frame_extract", "%d frames", len(frames))

            # ── STAGE 2b: Player detection ───────────────────────────────────
            t = time.perf_counter()
            player_result: PlayerDetectionResult = PlayerDetector(
                threshold=self.player_threshold
            ).detect(frames)
            timings.player_detect = time.perf_counter() - t
            log.pipeline("player_detect", "conf=%.1f%%  passed=%s",
                         player_result.confidence * 100, player_result.passed)

            if not player_result.passed:
                return self._fail(
                    f"No player detected (confidence {player_result.confidence:.1%}). "
                    "Ensure the player is fully visible in the frame.",
                    timings, t0,
                )

            # ── STAGE 2c: Ball detection ─────────────────────────────────────
            t = time.perf_counter()
            ball_result: BallDetectionResult = BallDetector().detect(frames)
            timings.ball_detect = time.perf_counter() - t
            log.pipeline("ball_detect", "conf=%.1f%%  found=%s",
                         ball_result.confidence * 100, ball_result.ball_detected)

            # ── STAGE 2d: Pose estimation ────────────────────────────────────
            t = time.perf_counter()
            estimator    = PoseEstimator(model_complexity=self.pose_model_complexity)
            pose_result: PoseEstimationResult = estimator.estimate(frames)
            timings.pose_estimate = time.perf_counter() - t
            log.pipeline("pose_estimate", "%d/%d frames  warnings=%s",
                         pose_result.detected_frames, pose_result.total_frames,
                         pose_result.warnings)

            # ── STAGE 3: Activity Understanding ─────────────────────────────
            t = time.perf_counter()

            raw_dets: List[RawActivityDetection] = AUActivityDetector().detect(
                pose_result, ball_result
            )
            raw_dets = self._cf.filter_raw(raw_dets)
            raw_dets = self._cf.deduplicate_frame(raw_dets)

            classified: List[ClassifiedActivity] = ActivityClassifier.classify(raw_dets)
            classified = self._cf.filter_classified(classified)
            classified = self._cf.normalise(classified)

            timeline: List[ActivitySegment] = SequenceAnalyzer.analyze(raw_dets, fps=fps)

            activities = [c.action for c in classified] if classified else ["passing"]

            timings.activity_understand = time.perf_counter() - t
            log.pipeline("activity_understand", "activities=%s  segments=%d",
                         activities, len(timeline))

            # ── STAGE 4–5: Analyzer Selection + Metrics ──────────────────────
            t = time.perf_counter()
            action_metrics = self._registry.run_for_activities(
                activities, frames, pose_result, ball_result
            )
            by_action = {
                action: am.to_display_dict()
                for action, am in action_metrics.items()
            }

            torso_lean      = pose_result.avg_torso_lean  or 8.0
            knee_stability  = max(0.0, 100.0 - (pose_result.avg_knee_dev  or 0.0) * 100)
            gait_symmetry   = max(0.0, 100.0 - (pose_result.gait_asymmetry or 0.0) * 100)

            raw_metrics = {
                "torso_lean":     abs(torso_lean),
                "knee_dev":       1.0 - knee_stability / 100.0,
                "gait_asymmetry": 1.0 - gait_symmetry  / 100.0,
            }

            timings.analyzer_metrics = time.perf_counter() - t
            log.pipeline("metrics", "torso=%.1f°  knee=%.0f  gait=%.0f",
                         torso_lean, knee_stability, gait_symmetry)

            # ── STAGE 6: Coach Engine ────────────────────────────────────────
            t = time.perf_counter()

            skill_profile   = CoachSkillClassifier().classify(raw_metrics)
            player_level    = skill_profile.level

            primary_activity = activities[0] if activities else "general"

            feedback_report = CoachFeedbackEngine(
                adapter=self._terminology
            ).generate(skill_profile, activity=primary_activity, metrics=raw_metrics)

            drills_list = DrillRecommender(
                adapter=self._terminology
            ).recommend(skill_profile, activity=primary_activity)

            timings.coach_engine = time.perf_counter() - t
            log.pipeline("coach_engine", "level=%s  issues=%d  drills=%d",
                         player_level, len(feedback_report.items), len(drills_list))

            # ── STAGE 7: Recommendation Engine ──────────────────────────────
            t = time.perf_counter()

            focus_areas = PrioritySelector().select(skill_profile)
            training_plan = TrainingPlanGenerator().generate(
                player_level, focus_areas, drills_list
            )
            weekly_plan = WeeklyPlanGenerator().generate(training_plan)
            recovery    = RecoveryAdvisor().advise(
                {
                    "torso_lean":     abs(torso_lean),
                    "knee_stability": knee_stability,
                    "gait_symmetry":  gait_symmetry,
                },
                player_level=player_level,
            )

            timings.recommendation = time.perf_counter() - t
            log.pipeline("recommendation", "focus=%s  rest=%s",
                         [a.label for a in focus_areas[:2]], recovery.rest_day_recommended)

            # ── STAGE 8: LLM rewrite (communicator only) ─────────────────────
            # The LLM receives structured coaching data and rewrites it into
            # natural language. It makes NO new football decisions.
            t = time.perf_counter()

            # Build structured data for the LLM to rewrite.
            structured_coaching = self._build_structured_coaching(
                feedback_report, focus_areas, drills_list, recovery, player_level
            )

            ai_feedback = structured_coaching   # start with deterministic output

            if self._ai_engine and self.use_ai:
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
                    # Validate the LLM rewrite.
                    validation = self._validator.validate_feedback_response(
                        str(ai_report.to_ai_feedback_dict())
                    )

                    # Only use LLM output to REWRITE language — keep all
                    # structured coaching decisions from Stage 6–7.
                    if ai_report.summary:
                        ai_feedback["summary"] = ai_report.summary
                    if ai_report.coach_tip:
                        ai_feedback["motivationalTip"] = ai_report.coach_tip
                    if ai_report.coaching_tips:
                        ai_feedback["coachingTips"] = ai_report.coaching_tips

                    log.pipeline("llm_rewrite", "provider=%s  repaired=%s  latency=%.2fs",
                                 ai_report.provider, ai_report.was_repaired, ai_report.latency_s)
                except Exception as ai_exc:
                    log.warning("LLM rewrite failed (non-fatal, using deterministic output): %s",
                                ai_exc)

            timings.llm_rewrite = time.perf_counter() - t
            timings.total       = time.perf_counter() - t0
            log.timing("total", timings.total)

            # ── STAGE 9: Assemble output ─────────────────────────────────────
            return PipelineOutput(
                success             = True,
                error               = None,
                detected_activities = activities,
                timeline            = [
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
                player_level        = player_level,
                metrics             = {
                    "byAction":      by_action,
                    "torsoLean":     round(abs(torso_lean), 1),
                    "kneeStability": round(knee_stability, 1),
                    "gaitSymmetry":  round(gait_symmetry, 1),
                    "warnings":      pose_result.warnings,
                },
                skill_profile       = skill_profile.to_dict(),
                coaching_feedback   = feedback_report.to_dict(),
                drills              = [
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
                ],
                focus_areas         = [a.label for a in focus_areas],
                training_plan       = training_plan.to_dict(),
                weekly_plan         = weekly_plan.to_dict(),
                recovery_advice     = recovery.to_dict(),
                ai_feedback         = ai_feedback,
                diagnostics         = {
                    "player_detection": {
                        "confidence": player_result.confidence,
                        "passed":     player_result.passed,
                    },
                    "ball_detection": {
                        "confidence":    ball_result.confidence,
                        "ball_detected": ball_result.ball_detected,
                    },
                    "pose": {
                        "detected_frames": pose_result.detected_frames,
                        "total_frames":    pose_result.total_frames,
                    },
                    "timings": {
                        "total_s":           round(timings.total, 2),
                        "pose_s":            round(timings.pose_estimate, 2),
                        "activity_s":        round(timings.activity_understand, 2),
                        "coach_s":           round(timings.coach_engine, 2),
                        "recommendation_s":  round(timings.recommendation, 2),
                        "llm_s":             round(timings.llm_rewrite, 2),
                    },
                },
                timings = timings,
            )

        except Exception as exc:
            timings.total = time.perf_counter() - t0
            log.error("Pipeline failed: %s", exc, exc_info=True)
            return self._fail(str(exc), timings, t0)

        finally:
            if estimator:
                estimator.close()
            if context:
                context.release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_structured_coaching(
        feedback_report: Any,
        focus_areas:     list,
        drills:          list,
        recovery:        Any,
        player_level:    str,
    ) -> Dict[str, Any]:
        """
        Build the ai_feedback dict from deterministic coach engine output.
        This is the baseline — the LLM only rewrites the language, not the content.
        """
        return {
            "summary":         feedback_report.summary,
            "strengths":       feedback_report.positive,
            "weaknesses":      [i.metric.replace("_", " ").title()
                                for i in feedback_report.items],
            "coachingTips":    [i.adapted_coach_tip for i in feedback_report.items],
            "motivationalTip": feedback_report.motivational_tip,
            "focusThisWeek":   [a.label for a in focus_areas],
            "priorityDrill":   drills[0].name if drills else None,
            "recoveryAdvice":  [item.advice for item in recovery.items[:2]],
            "restDayRecommended": recovery.rest_day_recommended,
        }

    @staticmethod
    def _fail(error: str, timings: StageTimings, t0: float) -> PipelineOutput:
        timings.total = time.perf_counter() - t0
        return PipelineOutput(
            success=False, error=error,
            detected_activities=[], timeline=[],
            player_level="Beginner",
            metrics={}, skill_profile={}, coaching_feedback={},
            drills=[], focus_areas=[], training_plan={},
            weekly_plan={}, recovery_advice={}, ai_feedback={},
            timings=timings,
        )

    def run_with_context(self, video_path: str, session_id: str = "") -> PipelineContext:
        """
        Run the full pipeline and return a populated PipelineContext.
        Every stage writes its output to ctx — nothing is passed as raw dicts.

        Parameters
        ----------
        video_path : str
        session_id : str — optional, set on the context

        Returns
        -------
        PipelineContext — always returns, errors captured in ctx.error
        """
        import uuid
        ctx            = PipelineContext(video_path=video_path)
        ctx.session_id = session_id or uuid.uuid4().hex[:12]
        ctx.status     = "running"

        video_ctx: Optional[VideoContext]  = None
        estimator: Optional[PoseEstimator] = None

        try:
            # ── Stage 1: Video load ──────────────────────────────────────────
            video_ctx = VideoLoader().load(video_path)
            ctx.video = VideoInfo(
                video_path  = video_path,
                fps         = video_ctx.fps or DEFAULT_FPS,
                duration_s  = video_ctx.duration_s,
                width       = video_ctx.width,
                height      = video_ctx.height,
                frame_count = video_ctx.frame_count,
            )
            ctx.log_stage("video_load", f"{ctx.video.resolution_label} @ {ctx.video.fps:.1f}fps")

            # ── Stage 2a: Frame extraction ───────────────────────────────────
            frames = FrameExtractor(stride=self.frame_stride).extract_all(video_ctx)
            ctx.frames.original_frames = frames
            ctx.log_stage("frame_extract", f"{len(frames)} frames")

            # ── Stage 2b: Player detection ───────────────────────────────────
            player_result = PlayerDetector(threshold=self.player_threshold).detect(frames)
            ctx.detections.player_confidence = player_result.confidence
            ctx.log_stage("player_detect", f"conf={player_result.confidence:.1%} passed={player_result.passed}")

            if not player_result.passed:
                ctx.mark_failed(
                    f"No player detected (confidence {player_result.confidence:.1%}). "
                    "Ensure the player is fully visible in the frame."
                )
                return ctx

            # ── Stage 2c: Ball detection ─────────────────────────────────────
            ball_result = BallDetector().detect(frames)
            ctx.detections.ball_confidence = ball_result.confidence
            ctx.detections.ball_tracks = [
                BallTrack(
                    frame_index = d.frame_index,
                    timestamp_s = d.timestamp_s,
                    center_x    = d.center_x,
                    center_y    = d.center_y,
                    radius      = d.radius,
                    confidence  = d.confidence,
                )
                for d in ball_result.detections
            ]
            ctx.log_stage("ball_detect", f"conf={ball_result.confidence:.1%} found={ball_result.ball_detected}")

            # ── Stage 2d: Pose estimation ────────────────────────────────────
            estimator   = PoseEstimator(model_complexity=self.pose_model_complexity)
            pose_result = estimator.estimate(frames)

            ctx.detections.pose_landmarks = [
                PoseLandmarkFrame(
                    frame_index = fp.frame_index,
                    timestamp_s = fp.timestamp_s,
                    detected    = fp.detected,
                    landmarks   = {k: v for k, v in (fp.landmarks or {}).items()},
                    torso_lean  = fp.torso_lean_deg if hasattr(fp, "torso_lean_deg") else fp.torso_lean,
                )
                for fp in pose_result.frame_poses
            ]
            ctx.detections.warnings            = pose_result.warnings
            ctx.detections.avg_torso_lean      = pose_result.avg_torso_lean
            ctx.detections.avg_knee_dev        = pose_result.avg_knee_dev
            ctx.detections.gait_asymmetry      = pose_result.gait_asymmetry
            ctx.detections.pose_detected_frames = pose_result.detected_frames
            ctx.detections.pose_total_frames   = pose_result.total_frames
            ctx.log_stage("pose_estimate", f"{pose_result.detected_frames}/{pose_result.total_frames} frames  warnings={pose_result.warnings}")

            # ── Stage 3: Activity understanding ──────────────────────────────
            raw_dets = AUActivityDetector().detect(pose_result, ball_result)
            raw_dets = self._cf.filter_raw(raw_dets)
            raw_dets = self._cf.deduplicate_frame(raw_dets)
            classified = ActivityClassifier.classify(raw_dets)
            classified = self._cf.filter_classified(classified)
            classified = self._cf.normalise(classified)
            timeline   = SequenceAnalyzer.analyze(raw_dets, fps=ctx.video.fps)

            activities = [c.action for c in classified] if classified else ["passing"]
            ctx.activity.detected_activities = activities
            ctx.activity.confidence_scores   = {c.action: c.combined_score for c in classified}
            ctx.activity.primary_activity    = activities[0] if activities else None
            ctx.activity.raw_detection_count = len(raw_dets)
            ctx.activity.timeline = [
                ActivitySegmentCtx(
                    action       = seg.action,
                    start_time_s = seg.start_time_s,
                    end_time_s   = seg.end_time_s,
                    duration_s   = seg.duration_s,
                    confidence   = seg.confidence,
                    label        = seg.label,
                )
                for seg in timeline
            ]
            ctx.log_stage("activity_understand", f"activities={activities} segments={len(timeline)}")

            # ── Stage 4–5: Analyzer selection + metrics ───────────────────────
            action_metrics = self._registry.run_for_activities(
                activities, frames, pose_result, ball_result
            )
            torso_lean     = pose_result.avg_torso_lean  or 8.0
            knee_stability = max(0.0, 100.0 - (pose_result.avg_knee_dev  or 0.0) * 100)
            gait_symmetry  = max(0.0, 100.0 - (pose_result.gait_asymmetry or 0.0) * 100)

            ctx.analysis.metrics = {
                "byAction":      {a: am.to_display_dict() for a, am in action_metrics.items()},
                "torsoLean":     round(abs(torso_lean), 1),
                "kneeStability": round(knee_stability, 1),
                "gaitSymmetry":  round(gait_symmetry, 1),
                "warnings":      pose_result.warnings,
            }
            ctx.log_stage("metrics", f"torso={torso_lean:.1f}° knee={knee_stability:.0f} gait={gait_symmetry:.0f}")

            # ── Stage 6: Coach engine ─────────────────────────────────────────
            raw_metrics = {
                "torso_lean":     abs(torso_lean),
                "knee_dev":       1.0 - knee_stability / 100.0,
                "gait_asymmetry": 1.0 - gait_symmetry  / 100.0,
            }
            skill_profile   = CoachSkillClassifier().classify(raw_metrics)
            primary         = activities[0] if activities else "general"
            feedback_report = CoachFeedbackEngine(adapter=self._terminology).generate(
                skill_profile, activity=primary, metrics=raw_metrics
            )
            drills_list = DrillRecommender(adapter=self._terminology).recommend(
                skill_profile, activity=primary
            )

            ctx.coaching.player_level  = skill_profile.level
            ctx.analysis.strengths     = skill_profile.strengths
            ctx.analysis.weaknesses    = skill_profile.weaknesses
            ctx.analysis.metric_scores = {ms.metric: ms.score for ms in skill_profile.metric_scores}
            ctx.analysis.overall_score = skill_profile.overall_score
            ctx.coaching.coach_tips    = [i.adapted_coach_tip for i in feedback_report.items]
            ctx.coaching.recommendations = [i.plain_observation for i in feedback_report.items]
            ctx.coaching.drills = [
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
            ctx.log_stage("coach_engine", f"level={skill_profile.level} issues={len(feedback_report.items)}")

            # ── Stage 7: Recommendation engine ────────────────────────────────
            focus_areas   = PrioritySelector().select(skill_profile)
            training_plan = TrainingPlanGenerator().generate(skill_profile.level, focus_areas, drills_list)
            weekly_plan   = WeeklyPlanGenerator().generate(training_plan)
            recovery      = RecoveryAdvisor().advise(raw_metrics, player_level=skill_profile.level)

            ctx.coaching.focus_this_week = [a.label for a in focus_areas]
            ctx.coaching.training_plan   = training_plan.to_dict()
            ctx.coaching.weekly_plan     = weekly_plan.to_dict()
            ctx.coaching.recovery_advice = recovery.to_dict()
            ctx.log_stage("recommendation", f"focus={ctx.coaching.focus_this_week[:2]} rest={recovery.rest_day_recommended}")

            # ── Stage 8: LLM rewrite ───────────────────────────────────────────
            structured = self._build_structured_coaching(
                feedback_report, focus_areas, drills_list, recovery, skill_profile.level
            )
            ctx.report.report = structured

            if self._ai_engine and self.use_ai:
                try:
                    ai_rep = self._ai_engine.explain(
                        detected_activities = activities,
                        player_level        = skill_profile.level,
                        torso_lean          = abs(torso_lean),
                        knee_stability      = knee_stability,
                        gait_symmetry       = gait_symmetry,
                        warnings            = pose_result.warnings,
                        by_action           = {a: am.to_display_dict() for a, am in action_metrics.items()},
                        video_duration_s    = ctx.video.duration_s,
                    )
                    if ai_rep.summary:
                        ctx.report.report["summary"] = ai_rep.summary
                    if ai_rep.coach_tip:
                        ctx.report.report["motivationalTip"] = ai_rep.coach_tip
                    ctx.log_stage("llm_rewrite", f"provider={ai_rep.provider}")
                except Exception as ai_exc:
                    log.warning("LLM rewrite failed (non-fatal): %s", ai_exc)
                    ctx.log_stage("llm_rewrite", "skipped — using deterministic output")

            # ── Stage 9: Finalise ─────────────────────────────────────────────
            ctx.report.session_id  = ctx.session_id
            ctx.report.video_url   = f"/api/video/{ctx.session_id}/analyzed.mp4"
            ctx.report.created_at  = __import__("datetime").datetime.utcnow().isoformat() + "Z"
            ctx.report.output_json = ctx.to_api_response()
            ctx.status             = "complete"
            ctx.log_stage("complete", f"total_stages=9 level={skill_profile.level}")

        except Exception as exc:
            log.error("Pipeline (context) failed: %s", exc, exc_info=True)
            ctx.mark_failed(str(exc))
        finally:
            if estimator:
                estimator.close()
            if video_ctx:
                video_ctx.release()

        return ctx
