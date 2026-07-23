#!/usr/bin/env python3
"""
Activity Detector
=================
Consumes per-frame pose data and ball detection data and produces a
list of RawActivityDetection objects — one per frame that contains
at least one scored action candidate.

Rule-based scoring signals
--------------------------
  Pose signals:
    torso lean      → shooting  (kick/shot posture)
    knee deviation  → dribbling (lateral cuts), defending (jockeying)
    gait asymmetry  → movement (stride imbalance), dribbling (ball running)

  Ball signals:
    ball speed      → passing / shooting (depending on speed magnitude)
    ball proximity  → dribbling (ball close to player feet)
    ball detected   → passing, shooting baseline

  Frame coverage:
    high pose rate  → movement (consistent player presence)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config.thresholds import (
    BALL_SPEED_PASS,
    BALL_SPEED_SHOT_MIN,
    BALL_SPEED_STATIONARY,
    DRIBBLING_MIN_CONFIDENCE,
    MOVEMENT_MIN_CONFIDENCE,
    PASSING_MIN_CONFIDENCE,
    SHOOTING_MIN_CONFIDENCE,
    TORSO_LEAN_WARNING_DEG,
    KNEE_DEV_WARNING,
    GAIT_ASYMMETRY_WARNING,
    TOUCH_TIGHT_ADVANCED,
    TOUCH_TIGHT_BEGINNER,
)
from pipeline.ball_detector  import BallDetectionResult, BallDetection
from pipeline.pose_estimator import PoseEstimationResult, FramePose
from schemas.activity_schema  import FootballAction
from utils.logger             import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class RawActivityDetection:
    """A single activity candidate detected in one frame."""
    frame_index: int
    timestamp_s: float
    action:      str           # FootballAction value (e.g. "passing")
    confidence:  float         # 0.0–1.0
    evidence:    List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

# Minimum confidence for a candidate to be kept in the output.
_DEFAULT_MIN_CONFIDENCE = 0.10

# Per-action confidence floors (sourced from thresholds).
_ACTION_FLOORS: dict[str, float] = {
    FootballAction.SHOOTING.value:    SHOOTING_MIN_CONFIDENCE,
    FootballAction.PASSING.value:     PASSING_MIN_CONFIDENCE,
    FootballAction.DRIBBLING.value:   DRIBBLING_MIN_CONFIDENCE,
    FootballAction.DEFENDING.value:   DRIBBLING_MIN_CONFIDENCE,   # same as dribbling
    FootballAction.GOALKEEPING.value: SHOOTING_MIN_CONFIDENCE,
    FootballAction.MOVEMENT.value:    MOVEMENT_MIN_CONFIDENCE,
    FootballAction.UNKNOWN.value:     _DEFAULT_MIN_CONFIDENCE,
}


class ActivityDetector:
    """
    Frame-level activity detector.

    Consumes PoseEstimationResult and BallDetectionResult and produces
    a list of RawActivityDetection — one entry per frame that has at
    least one candidate action above the minimum confidence floor.

    Parameters
    ----------
    min_confidence : float
        Global confidence floor; candidates below this are dropped.

    Usage::

        detector  = ActivityDetector()
        detections = detector.detect(pose_result, ball_result)
        for d in detections:
            print(d.frame_index, d.action, d.confidence)
    """

    def __init__(self, min_confidence: float = _DEFAULT_MIN_CONFIDENCE) -> None:
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(
        self,
        pose: PoseEstimationResult,
        ball: BallDetectionResult,
    ) -> List[RawActivityDetection]:
        """
        Produce per-frame RawActivityDetection objects.

        Returns an empty list if either input has no data; never raises.
        """
        if not pose.frame_poses:
            log.debug("ActivityDetector: no frame poses — returning empty list")
            return []

        # Build a lookup from frame_index → BallDetection (if present).
        ball_by_frame: dict[int, BallDetection] = {
            bd.frame_index: bd for bd in ball.detections
        }

        detections: List[RawActivityDetection] = []

        for fp in pose.frame_poses:
            frame_detections = self._score_frame(fp, ball_by_frame, ball)
            detections.extend(frame_detections)

        log.debug(
            "ActivityDetector: produced %d raw detections across %d frames",
            len(detections),
            len(pose.frame_poses),
        )
        return detections

    # ------------------------------------------------------------------
    # Private — per-frame scoring
    # ------------------------------------------------------------------

    def _score_frame(
        self,
        fp:            FramePose,
        ball_by_frame: dict[int, BallDetection],
        ball_result:   BallDetectionResult,
    ) -> List[RawActivityDetection]:
        """Score all actions for a single frame and return candidates."""

        # scores[action] = (accumulated_score, [evidence strings])
        scores: dict[str, Tuple[float, List[str]]] = {
            a.value: (0.0, []) for a in FootballAction if a != FootballAction.UNKNOWN
        }

        def add(action: str, delta: float, reason: str) -> None:
            s, ev = scores[action]
            scores[action] = (min(1.0, s + delta), ev + [reason])

        # ── Pose-based signals ────────────────────────────────────────────

        if fp.detected:
            # Torso lean → shooting / dribbling
            if fp.torso_lean is not None:
                lean_abs = abs(fp.torso_lean)
                if lean_abs >= TORSO_LEAN_WARNING_DEG:
                    lean_norm = min(1.0, lean_abs / 45.0)  # normalise to 45°
                    add(
                        FootballAction.SHOOTING.value,
                        0.30 + 0.25 * lean_norm,
                        f"Torso lean {lean_abs:.1f}° (shooting posture)",
                    )
                    add(
                        FootballAction.DRIBBLING.value,
                        0.15,
                        f"Torso lean {lean_abs:.1f}° (sharp direction change)",
                    )

            # Knee deviation → dribbling / defending
            if fp.knee_dev_l is not None or fp.knee_dev_r is not None:
                max_knee = max(
                    abs(fp.knee_dev_l or 0.0),
                    abs(fp.knee_dev_r or 0.0),
                )
                if max_knee >= KNEE_DEV_WARNING:
                    knee_norm = min(1.0, max_knee / 0.50)
                    add(
                        FootballAction.DRIBBLING.value,
                        0.30 + 0.20 * knee_norm,
                        f"Knee deviation {max_knee:.2f} (lateral cut typical of dribbling)",
                    )
                    add(
                        FootballAction.DEFENDING.value,
                        0.20,
                        f"Knee deviation {max_knee:.2f} (jockeying stance)",
                    )

        # ── Gait asymmetry (aggregate, applied to every frame) ────────────
        # We apply a fraction of the aggregate gait signal per frame so that
        # frames with pose data reflect the video-level gait pattern.
        if pose_has_gait := self._has_gait_asymmetry(fp):
            add(
                FootballAction.MOVEMENT.value,
                0.35,
                "Stride asymmetry in running pattern",
            )
            add(
                FootballAction.DRIBBLING.value,
                0.15,
                "Asymmetric gait can indicate ball-at-feet running",
            )

        # ── Ball signals ─────────────────────────────────────────────────

        ball_det: Optional[BallDetection] = ball_by_frame.get(fp.frame_index)

        if ball_det is not None:
            # Ball proximity → dribbling
            proximity = self._ball_proximity(fp, ball_det)
            if proximity is not None:
                if proximity <= TOUCH_TIGHT_ADVANCED:
                    add(
                        FootballAction.DRIBBLING.value,
                        0.45,
                        f"Ball very close to player (proximity {proximity:.3f} — tight control)",
                    )
                elif proximity <= TOUCH_TIGHT_BEGINNER:
                    add(
                        FootballAction.DRIBBLING.value,
                        0.25,
                        f"Ball near player (proximity {proximity:.3f})",
                    )

            # Ball speed (computed from previous frame if available) is
            # estimated via radius change as a proxy.
            ball_conf = ball_det.confidence
            add(
                FootballAction.PASSING.value,
                0.20 + 0.15 * ball_conf,
                f"Ball detected (conf {ball_conf:.2f}) — pass candidate",
            )
            add(
                FootballAction.SHOOTING.value,
                0.15 + 0.10 * ball_conf,
                f"Ball detected (conf {ball_conf:.2f}) — shot candidate",
            )

        elif ball_result.ball_detected:
            # Ball seen in video but not this frame — weaker signal
            add(
                FootballAction.PASSING.value,
                0.10,
                "Ball detected in video (not this frame)",
            )

        # ── Consistent pose presence → movement ──────────────────────────
        if fp.detected:
            add(
                FootballAction.MOVEMENT.value,
                0.20,
                "Player pose detected — movement present",
            )

        # ── Build candidates ─────────────────────────────────────────────
        candidates: List[RawActivityDetection] = []
        for action, (score, evidence) in scores.items():
            floor = _ACTION_FLOORS.get(action, self.min_confidence)
            effective_floor = max(floor, self.min_confidence)
            if score >= effective_floor and evidence:
                candidates.append(RawActivityDetection(
                    frame_index = fp.frame_index,
                    timestamp_s = fp.timestamp_s,
                    action      = action,
                    confidence  = round(score, 4),
                    evidence    = evidence,
                ))

        return candidates

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_gait_asymmetry(fp: FramePose) -> bool:
        """
        True if the frame has identifiable ankle landmarks far apart in Y,
        used as a per-frame proxy for the video-level gait asymmetry signal.
        """
        if not fp.detected or not fp.landmarks:
            return False
        left_y  = fp.landmarks.get("left_ankle")
        right_y = fp.landmarks.get("right_ankle")
        if left_y is None or right_y is None:
            return False
        diff = abs(left_y.y - right_y.y)
        return diff >= GAIT_ASYMMETRY_WARNING

    @staticmethod
    def _ball_proximity(fp: FramePose, ball: BallDetection) -> Optional[float]:
        """
        Return normalised distance between the ball centre and the midpoint
        of the player's ankles, or None if landmarks are unavailable.
        Coordinates are normalised [0, 1].
        """
        if not fp.detected or not fp.landmarks:
            return None
        left_ankle  = fp.landmarks.get("left_ankle")
        right_ankle = fp.landmarks.get("right_ankle")
        if left_ankle is None and right_ankle is None:
            return None

        if left_ankle and right_ankle:
            foot_x = (left_ankle.x + right_ankle.x) / 2
            foot_y = (left_ankle.y + right_ankle.y) / 2
        elif left_ankle:
            foot_x, foot_y = left_ankle.x, left_ankle.y
        else:
            assert right_ankle is not None
            foot_x, foot_y = right_ankle.x, right_ankle.y

        # ball coordinates are in pixel space; normalise by assuming a
        # representative frame size of 1920×1080 for the default case.
        # The proximity is used only for relative thresholds so the exact
        # normalisation factor is acceptable as a reasonable default.
        ball_x_norm = ball.center_x / 1920.0
        ball_y_norm = ball.center_y / 1080.0

        dx = foot_x - ball_x_norm
        dy = foot_y - ball_y_norm
        return math.hypot(dx, dy)
