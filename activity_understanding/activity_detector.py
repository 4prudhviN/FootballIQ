#!/usr/bin/env python3
"""
Activity Detector  ⭐
=====================
The heart of FootballIQ.

Answers ONE question: "What football actions did the player perform?"

Output:
{
    "activities": [
        {"name": "Passing",      "confidence": 0.96},
        {"name": "Receiving",    "confidence": 0.92},
        {"name": "Ball Control", "confidence": 0.89}
    ]
}

Rules:
  ❌ No coaching
  ❌ No metrics
  ✅ Pure understanding — what happened

Input: PoseEstimationResult + BallDetectionResult + PlayerDetectionResult
Output: ActivityDetectionOutput

Writes to PipelineContext:
  ctx.activity.detected_activities  — list of action names
  ctx.activity.confidence_scores    — {name: confidence}
  ctx.activity.primary_activity     — highest confidence action
  ctx.activity.timeline             — time-segmented action list
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pipeline.pose_estimator   import PoseEstimationResult, FramePose
from pipeline.ball_detector    import BallDetectionResult, BallDetection
from pipeline.player_detector  import PlayerDetectionResult
from pipeline.pipeline_context import PipelineContext, ActivitySegmentCtx
from utils.logger              import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# All detectable football actions
# ---------------------------------------------------------------------------

class Action:
    PASSING      = "Passing"
    RECEIVING    = "Receiving"
    BALL_CONTROL = "Ball Control"
    DRIBBLING    = "Dribbling"
    SHOOTING     = "Shooting"
    GOALKEEPING  = "Goalkeeping"
    DEFENDING    = "Defending"
    MOVEMENT     = "Movement"
    FREE_KICK    = "Free Kick"
    HEADER       = "Header"
    PENALTY      = "Penalty"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class DetectedActivity:
    """A single detected football action with confidence."""
    name:       str    # e.g. "Passing"
    confidence: float  # 0.0–1.0


@dataclass
class ActivityDetectionOutput:
    """
    The complete output of the Activity Detector.
    Answers: "What football actions did the player perform?"
    Nothing else.
    """
    activities: List[DetectedActivity]

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical output format."""
        return {
            "activities": [
                {
                    "name":       a.name,
                    "confidence": round(a.confidence, 2),
                }
                for a in self.activities
            ]
        }

    @property
    def names(self) -> List[str]:
        return [a.name for a in self.activities]

    @property
    def primary(self) -> Optional[str]:
        return self.activities[0].name if self.activities else None

    @property
    def confidence_map(self) -> Dict[str, float]:
        return {a.name: a.confidence for a in self.activities}


# ---------------------------------------------------------------------------
# Evidence accumulators
# ---------------------------------------------------------------------------
# Each signal contributes evidence for one or more actions.
# Confidence is accumulated from multiple independent signals.
# Final confidence is normalised to [0, 1].

_MAX_CONFIDENCE = 1.0

def _clamp(v: float) -> float:
    return max(0.0, min(_MAX_CONFIDENCE, v))


class _EvidenceAccumulator:
    """Accumulates confidence evidence for each action across all frames."""

    def __init__(self) -> None:
        self._scores: Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def add(self, action: str, score: float) -> None:
        """Add evidence for an action in one frame."""
        self._scores[action] = self._scores.get(action, 0.0) + score
        self._counts[action] = self._counts.get(action, 0)   + 1

    def normalise(self, total_frames: int) -> Dict[str, float]:
        """
        Normalise accumulated scores to [0, 1].
        Returns dict of {action: confidence}.
        """
        if total_frames <= 0:
            return {}
        result: Dict[str, float] = {}
        for action, total_score in self._scores.items():
            # Confidence = fraction of frames where action was evidenced,
            # weighted by the strength of evidence per frame.
            avg_score    = total_score / self._counts[action]
            frame_cover  = self._counts[action] / total_frames
            confidence   = _clamp(avg_score * 0.6 + frame_cover * 0.4)
            result[action] = round(confidence, 3)
        return result


# ---------------------------------------------------------------------------
# Per-frame signal extractors
# ---------------------------------------------------------------------------

class _SignalExtractor:
    """
    Extracts observable football signals from one frame.
    Each signal is a (action, strength) pair.
    Strength is in [0, 1].
    """

    def extract(
        self,
        pose:      Optional[FramePose],
        ball:      Optional[BallDetection],
        prev_ball: Optional[BallDetection],
        frame_idx: int,
        fps:       float,
    ) -> List[tuple[str, float]]:
        """Return list of (action_name, strength) for this frame."""
        signals: List[tuple[str, float]] = []

        # ── Ball movement signals ────────────────────────────────────────────

        if ball and prev_ball:
            dx    = ball.center_x - prev_ball.center_x
            dy    = ball.center_y - prev_ball.center_y
            speed = math.hypot(dx, dy)

            if speed > 50:
                # Fast ball departure — shooting or driven pass.
                signals.append((Action.SHOOTING,  0.75))
                signals.append((Action.PASSING,   0.50))

            elif speed > 20:
                # Moderate ball movement — pass or first touch.
                signals.append((Action.PASSING,      0.65))
                signals.append((Action.BALL_CONTROL, 0.40))

            elif speed > 5:
                # Slow ball movement — receiving or controlling.
                signals.append((Action.RECEIVING,    0.60))
                signals.append((Action.BALL_CONTROL, 0.55))

            elif speed < 3:
                # Ball nearly stationary.
                if pose and self._ball_at_feet(ball, pose):
                    signals.append((Action.BALL_CONTROL, 0.70))
                    signals.append((Action.DRIBBLING,    0.45))

        elif ball and not prev_ball:
            # Ball just appeared — receiving.
            signals.append((Action.RECEIVING, 0.55))

        # ── Ball proximity signals ───────────────────────────────────────────

        if ball and pose:
            if self._ball_at_feet(ball, pose):
                signals.append((Action.BALL_CONTROL, 0.60))
                signals.append((Action.DRIBBLING,    0.35))

            if self._ball_at_head(ball, pose):
                signals.append((Action.HEADER, 0.80))

        # ── Pose-based signals ───────────────────────────────────────────────

        if pose and pose.detected:

            if self._kicking_leg_raised(pose):
                signals.append((Action.SHOOTING,  0.65))
                signals.append((Action.PASSING,   0.45))

            if self._arms_wide(pose):
                signals.append((Action.GOALKEEPING, 0.75))

            if self._defensive_stance(pose):
                signals.append((Action.DEFENDING, 0.65))

            if self._running_arms(pose):
                signals.append((Action.MOVEMENT, 0.50))

            if self._leaning_forward(pose):
                # Forward lean at contact = shooting/passing posture.
                signals.append((Action.PASSING,  0.40))
                signals.append((Action.SHOOTING, 0.35))

        # ── Default: player is always moving ────────────────────────────────
        if pose and pose.detected:
            signals.append((Action.MOVEMENT, 0.25))

        return signals

    # ------------------------------------------------------------------
    # Observable pose cues — each returns True/False only
    # ------------------------------------------------------------------

    @staticmethod
    def _ball_at_feet(ball: BallDetection, pose: FramePose) -> bool:
        for name in ("left_ankle", "right_ankle"):
            ankle = pose.get(name)
            if ankle is None:
                continue
            dist = math.hypot(
                ankle.x * 1920 - ball.center_x,
                ankle.y * 1080 - ball.center_y,
            )
            if dist < 90:
                return True
        return False

    @staticmethod
    def _ball_at_head(ball: BallDetection, pose: FramePose) -> bool:
        nose = pose.get("nose")
        if nose is None:
            return False
        dist = math.hypot(
            nose.x * 1920 - ball.center_x,
            nose.y * 1080 - ball.center_y,
        )
        return dist < 70

    @staticmethod
    def _kicking_leg_raised(pose: FramePose) -> bool:
        lk = pose.get("left_knee")
        rk = pose.get("right_knee")
        if None in (lk, rk):
            return False
        return abs(lk.y - rk.y) > 0.12

    @staticmethod
    def _arms_wide(pose: FramePose) -> bool:
        lw = pose.get("left_wrist")
        rw = pose.get("right_wrist")
        ls = pose.get("left_shoulder")
        rs = pose.get("right_shoulder")
        if None in (lw, rw, ls, rs):
            return False
        shoulder_w = abs(ls.x - rs.x)
        wrist_w    = abs(lw.x - rw.x)
        return wrist_w > shoulder_w * 1.8

    @staticmethod
    def _defensive_stance(pose: FramePose) -> bool:
        lk = pose.get("left_knee")
        rk = pose.get("right_knee")
        lh = pose.get("left_hip")
        rh = pose.get("right_hip")
        if None in (lk, rk, lh, rh):
            return False
        return (lk.y > lh.y + 0.05) and (rk.y > rh.y + 0.05)

    @staticmethod
    def _running_arms(pose: FramePose) -> bool:
        lw = pose.get("left_wrist")
        rw = pose.get("right_wrist")
        if None in (lw, rw):
            return False
        return abs(lw.y - rw.y) > 0.08

    @staticmethod
    def _leaning_forward(pose: FramePose) -> bool:
        ls = pose.get("left_shoulder")
        rs = pose.get("right_shoulder")
        lh = pose.get("left_hip")
        rh = pose.get("right_hip")
        if None in (ls, rs, lh, rh):
            return False
        sh_y = (ls.y + rs.y) / 2
        hi_y = (lh.y + rh.y) / 2
        # Shoulders higher than hips (in y coords) = forward lean
        return (hi_y - sh_y) > 0.15


# ---------------------------------------------------------------------------
# Activity Detector
# ---------------------------------------------------------------------------

class ActivityDetector:
    """
    Detects which football actions the player performed.

    Input:  pose + ball + player results from pipeline stages 2b–2d
    Output: ActivityDetectionOutput

    ONE question: "What football actions did the player perform?"
    No coaching. No metrics. Pure understanding.

    Usage::

        detector = ActivityDetector()
        output   = detector.detect(pose_result, ball_result)
        print(output.to_dict())
        # {"activities": [{"name": "Passing", "confidence": 0.96}, ...]}

        detector.write_to_context(output, ctx)
    """

    # Minimum confidence to include in output.
    MIN_CONFIDENCE = 0.15

    def __init__(self) -> None:
        self._extractor = _SignalExtractor()

    def detect(
        self,
        pose_result:    PoseEstimationResult,
        ball_result:    BallDetectionResult,
        player_result:  Optional[PlayerDetectionResult] = None,
        fps:            float = 25.0,
    ) -> ActivityDetectionOutput:
        """
        Detect football actions from pose and ball data.

        Parameters
        ----------
        pose_result   : PoseEstimationResult
        ball_result   : BallDetectionResult
        player_result : PlayerDetectionResult (optional)
        fps           : float

        Returns
        -------
        ActivityDetectionOutput — {activities: [{name, confidence}, ...]}
        """
        pose_map: Dict[int, FramePose] = {
            fp.frame_index: fp
            for fp in pose_result.frame_poses
            if fp.detected
        }
        ball_map: Dict[int, BallDetection] = {
            bd.frame_index: bd
            for bd in ball_result.detections
        }

        accumulator   = _EvidenceAccumulator()
        sorted_frames = sorted(pose_map.keys())
        prev_ball: Optional[BallDetection] = None

        for fi in sorted_frames:
            pose = pose_map.get(fi)
            ball = ball_map.get(fi)

            signals = self._extractor.extract(
                pose, ball, prev_ball, fi, fps
            )
            for action, strength in signals:
                accumulator.add(action, strength)

            if ball is not None:
                prev_ball = ball

        total_frames = len(sorted_frames) or 1
        scores       = accumulator.normalise(total_frames)

        # Build output — filter below threshold, sort by confidence.
        activities = [
            DetectedActivity(name=name, confidence=conf)
            for name, conf in scores.items()
            if conf >= self.MIN_CONFIDENCE
        ]
        activities.sort(key=lambda a: a.confidence, reverse=True)

        # Always return at least one activity.
        if not activities:
            activities = [DetectedActivity(name=Action.MOVEMENT, confidence=0.50)]

        log.debug(
            "ActivityDetector: %d activities detected  primary=%s",
            len(activities), activities[0].name,
        )

        return ActivityDetectionOutput(activities=activities)

    @staticmethod
    def write_to_context(
        output: ActivityDetectionOutput,
        ctx:    PipelineContext,
    ) -> None:
        """Write detection results to PipelineContext."""
        ctx.activity.detected_activities = output.names
        ctx.activity.confidence_scores   = output.confidence_map
        ctx.activity.primary_activity    = output.primary

        # Build timeline segments from activity list.
        ctx.activity.timeline = [
            ActivitySegmentCtx(
                action       = a.name.lower().replace(" ", "_"),
                start_time_s = 0.0,
                end_time_s   = 0.0,
                duration_s   = 0.0,
                confidence   = a.confidence,
                label        = a.name,
            )
            for a in output.activities
        ]

        ctx.log_stage(
            "activity_detect",
            f"activities={output.names}  primary={output.primary}",
        )
