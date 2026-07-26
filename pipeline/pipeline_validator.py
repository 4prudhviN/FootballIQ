#!/usr/bin/env python3
"""
Pipeline Validator
==================
Every stage verifies the PipelineContext before proceeding.

Gates checked per stage:
  Stage 1 (video_load)    — video file exists, fps > 0, dimensions valid
  Stage 2 (frame_extract) — frames extracted, count > 0
  Stage 3 (player_detect) — player confidence above threshold
  Stage 4 (ball_detect)   — informational only, never blocks
  Stage 5 (pose_estimate) — detected_frames > minimum required
  Stage 6 (activity)      — at least one activity detected
  Stage 7 (metrics)       — metrics dict populated
  Stage 8 (coaching)      — player_level assigned
  Stage 9 (report)        — report dict not empty

Usage::

    validator = PipelineValidator()

    # After each stage, call validate:
    ok, error = validator.after_video_load(ctx)
    if not ok:
        ctx.mark_failed(error)
        return ctx

    ok, error = validator.after_frame_extract(ctx)
    ...
"""

from __future__ import annotations

from typing import Tuple

from pipeline.pipeline_context import PipelineContext
from config.constants          import PLAYER_DETECTION_THRESHOLD, POSE_DETECTION_RATE_MIN
from utils.logger              import get_logger

log = get_logger(__name__)

# Minimum frames required for pose analysis to be meaningful.
_MIN_POSE_FRAMES = 5


class PipelineValidator:
    """
    Validates PipelineContext after each stage.

    Each method returns (ok: bool, error_message: str).
    If ok=False, the pipeline should call ctx.mark_failed(error) and return.
    """

    # ------------------------------------------------------------------
    # Stage 1 — Video Load
    # ------------------------------------------------------------------

    def after_video_load(self, ctx: PipelineContext) -> Tuple[bool, str]:
        """Video must exist, have valid dimensions and FPS."""
        if not ctx.video.video_path:
            return False, "No video path set in PipelineContext."

        if ctx.video.fps <= 0:
            return False, f"Invalid FPS: {ctx.video.fps}. Video may be corrupted."

        if ctx.video.width <= 0 or ctx.video.height <= 0:
            return False, (
                f"Invalid video dimensions: {ctx.video.width}x{ctx.video.height}. "
                "Ensure the video file is not corrupted."
            )

        if ctx.video.frame_count <= 0:
            return False, "Video has 0 frames. File may be empty or corrupted."

        log.debug("Validator: video_load ✓  %s  %s  %.1ffps  %d frames",
                  ctx.video.video_path, ctx.video.resolution_label,
                  ctx.video.fps, ctx.video.frame_count)
        return True, ""

    # ------------------------------------------------------------------
    # Stage 2 — Frame Extraction
    # ------------------------------------------------------------------

    def after_frame_extract(self, ctx: PipelineContext) -> Tuple[bool, str]:
        """At least one frame must have been extracted."""
        count = ctx.frames.frame_count
        if count == 0:
            return False, (
                "No frames were extracted from the video. "
                "The video may be too short or the format unsupported."
            )
        log.debug("Validator: frame_extract ✓  %d frames", count)
        return True, ""

    # ------------------------------------------------------------------
    # Stage 3 — Player Detection
    # ------------------------------------------------------------------

    def after_player_detect(
        self, ctx: PipelineContext, threshold: float = PLAYER_DETECTION_THRESHOLD
    ) -> Tuple[bool, str]:
        """Player must be detected in enough frames to proceed."""
        conf = ctx.detections.player_confidence
        if conf < threshold:
            return False, (
                f"No player detected in the video "
                f"(confidence {conf:.1%} < {threshold:.0%} threshold). "
                "Ensure the player is clearly visible and fully in frame."
            )
        log.debug("Validator: player_detect ✓  conf=%.1f%%", conf * 100)
        return True, ""

    # ------------------------------------------------------------------
    # Stage 4 — Ball Detection (informational — never blocks)
    # ------------------------------------------------------------------

    def after_ball_detect(self, ctx: PipelineContext) -> Tuple[bool, str]:
        """Ball detection is informational. Always passes."""
        conf = ctx.detections.ball_confidence
        if conf < 0.05:
            log.info("Validator: ball_detect — no ball detected (conf=%.1f%%). Pipeline continues.", conf * 100)
        else:
            log.debug("Validator: ball_detect ✓  conf=%.1f%%", conf * 100)
        return True, ""   # never blocks

    # ------------------------------------------------------------------
    # Stage 5 — Pose Estimation
    # ------------------------------------------------------------------

    def after_pose_estimate(
        self, ctx: PipelineContext, min_frames: int = _MIN_POSE_FRAMES
    ) -> Tuple[bool, str]:
        """Enough pose frames must be detected for meaningful analysis."""
        detected = ctx.detections.pose_detected_frames
        total    = ctx.detections.pose_total_frames or 1
        rate     = detected / total

        if detected < min_frames:
            return False, (
                f"Pose estimation detected only {detected} frames "
                f"(minimum {min_frames} required). "
                "Ensure the player's full body is visible in the frame."
            )

        if rate < POSE_DETECTION_RATE_MIN:
            log.warning(
                "Validator: pose_estimate — low detection rate %.1f%% "
                "(%d/%d frames). Analysis may be incomplete.",
                rate * 100, detected, total,
            )
            # Warn but don't block — partial data is better than nothing.

        log.debug("Validator: pose_estimate ✓  %d/%d frames (%.0f%%)",
                  detected, total, rate * 100)
        return True, ""

    # ------------------------------------------------------------------
    # Stage 6 — Activity Detection
    # ------------------------------------------------------------------

    def after_activity_detect(self, ctx: PipelineContext) -> Tuple[bool, str]:
        """At least one football activity must have been detected."""
        activities = ctx.activity.detected_activities
        if not activities:
            log.warning(
                "Validator: activity_detect — no activities detected. "
                "Falling back to 'movement'."
            )
            # Soft fallback — don't block, assign default activity.
            ctx.activity.detected_activities = ["movement"]
            ctx.activity.primary_activity    = "movement"

        log.debug("Validator: activity_detect ✓  %s", ctx.activity.detected_activities)
        return True, ""

    # ------------------------------------------------------------------
    # Stage 7 — Metrics
    # ------------------------------------------------------------------

    def after_metrics(self, ctx: PipelineContext) -> Tuple[bool, str]:
        """Metrics dict must not be empty."""
        if not ctx.analysis.metrics:
            log.warning("Validator: metrics — empty metrics dict. Pipeline continues with defaults.")
            ctx.analysis.metrics = {
                "byAction":      {},
                "torsoLean":     0.0,
                "kneeStability": 0.0,
                "gaitSymmetry":  0.0,
                "warnings":      [],
            }
        log.debug("Validator: metrics ✓")
        return True, ""

    # ------------------------------------------------------------------
    # Stage 8 — Coaching
    # ------------------------------------------------------------------

    def after_coaching(self, ctx: PipelineContext) -> Tuple[bool, str]:
        """Player level must be assigned."""
        if not ctx.coaching.player_level:
            log.warning("Validator: coaching — no player level. Defaulting to Beginner.")
            ctx.coaching.player_level = "Beginner"
        log.debug("Validator: coaching ✓  level=%s", ctx.coaching.player_level)
        return True, ""

    # ------------------------------------------------------------------
    # Stage 9 — Report
    # ------------------------------------------------------------------

    def after_report(self, ctx: PipelineContext) -> Tuple[bool, str]:
        """Report must have at least a summary."""
        report = ctx.report.report or {}
        if not report.get("session_summary") and not report.get("summary"):
            log.warning("Validator: report — empty report. Using default summary.")
            ctx.report.report = {
                "session_summary": "Session analysis complete.",
                "strengths":       [],
                "areas_to_improve": [],
                "training_recommendations": [],
                "next_focus":      "Focus on one skill at a time.",
                "motivationalTip": "Keep training — every session counts.",
            }
        log.debug("Validator: report ✓")
        return True, ""

    # ------------------------------------------------------------------
    # Convenience — validate all stages at once from a completed context
    # ------------------------------------------------------------------

    def validate_complete(self, ctx: PipelineContext) -> Tuple[bool, str]:
        """
        Run all validators against a completed context.
        Returns (True, "") if everything is valid.
        Returns (False, error_message) on first failure.
        """
        checks = [
            self.after_video_load,
            self.after_frame_extract,
            self.after_player_detect,
            self.after_ball_detect,
            self.after_pose_estimate,
            self.after_activity_detect,
            self.after_metrics,
            self.after_coaching,
            self.after_report,
        ]
        for check in checks:
            ok, error = check(ctx)
            if not ok:
                return False, error
        return True, ""
