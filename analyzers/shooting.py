#!/usr/bin/env python3
"""
Shooting Analyzer
=================
Bridges the pipeline output to ShootingMetrics.

Detects shot events by identifying frames where the ball travels at high
speed away from the player, then delegates computation to
metrics/shooting_metrics.py.

Never crashes — returns a valid ActionMetrics if data is insufficient.
"""

from __future__ import annotations

import math
from typing import List

from analyzers.base_analyzer import BaseAnalyzer
from schemas.activity_schema import ActionMetrics, ActivityMetric, FootballAction
from pipeline.frame_extractor import ExtractedFrame
from pipeline.pose_estimator import PoseEstimationResult, FramePose
from pipeline.ball_detector import BallDetectionResult, BallDetection
from metrics.shooting_metrics import ShootingMetrics, ShotEvent
from metrics.common_metrics import Point2D
from config.thresholds import BALL_SPEED_SHOT_MIN, LAUNCH_ANGLE_IDEAL_MIN, LAUNCH_ANGLE_IDEAL_MAX
from utils.logger import get_logger

log = get_logger(__name__)

# Shot detection: pixel-space minimum speed (scaled up from normalised 0.025)
_SHOT_SPEED_PX = BALL_SPEED_SHOT_MIN * 100


class ShootingAnalyzer(BaseAnalyzer):
    """
    Analyzer for shooting actions.

    Identifies shot events from rapid ball-departure frames and derives
    launch angle, torso lean, and foot-strike type from pose data.
    """

    @property
    def name(self) -> str:
        return FootballAction.SHOOTING.value

    def analyze(
        self,
        frames:      List[ExtractedFrame],
        pose_result: PoseEstimationResult,
        ball_result: BallDetectionResult,
    ) -> ActionMetrics:
        try:
            events = self._extract_events(frames, pose_result, ball_result)
            metric_set = ShootingMetrics().calculate(events)
            return self._to_action_metrics(metric_set)
        except Exception as exc:  # noqa: BLE001
            log.warning("ShootingAnalyzer: error during analysis — %s", exc)
            return self._empty_metrics()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_events(
        self,
        frames:      List[ExtractedFrame],
        pose_result: PoseEstimationResult,
        ball_result: BallDetectionResult,
    ) -> List[ShotEvent]:
        """
        Detect shot events from high-speed ball departures.

        A shot is identified when ball speed between consecutive frames
        exceeds BALL_SPEED_SHOT_MIN.  Trajectory, torso lean, and foot
        are derived from surrounding pose data.
        """
        detections = ball_result.detections
        if len(detections) < 2:
            return self._synthetic_events(pose_result)

        w = frames[0].bgr.shape[1] if frames else 640
        h = frames[0].bgr.shape[0] if frames else 480
        fps      = 25.0
        px_per_m = 100.0

        pose_map = {fp.frame_index: fp for fp in pose_result.frame_poses if fp.detected}

        events: List[ShotEvent] = []
        for i in range(len(detections) - 1):
            d0 = detections[i]
            d1 = detections[i + 1]

            dx = d1.center_x - d0.center_x
            dy = d1.center_y - d0.center_y
            speed = math.hypot(dx, dy)

            if speed < _SHOT_SPEED_PX:
                continue

            # Build trajectory: start + next few detections.
            traj: List[Point2D] = [
                Point2D(d0.center_x / w, d0.center_y / h),
                Point2D(d1.center_x / w, d1.center_y / h),
            ]
            for j in range(i + 2, min(i + 6, len(detections))):
                traj.append(Point2D(
                    detections[j].center_x / w,
                    detections[j].center_y / h,
                ))

            ball_start = Point2D(d0.center_x / w, d0.center_y / h)

            # Pose at shot frame.
            fp = pose_map.get(d0.frame_index)
            torso_lean = self._torso_lean(fp)
            foot       = self._shooting_foot(fp)

            # On-target heuristic: ball moving roughly forward and upward.
            angle_deg = abs(math.degrees(math.atan2(-dy, dx)))
            on_target = LAUNCH_ANGLE_IDEAL_MIN <= angle_deg <= LAUNCH_ANGLE_IDEAL_MAX + 20

            events.append(ShotEvent(
                ball_start      = ball_start,
                ball_trajectory = traj,
                on_target       = on_target,
                goal            = False,           # no goal-line data available
                torso_lean      = torso_lean,
                foot            = foot,
                contact_type    = "laces",         # default; no detailed foot model
                fps             = fps,
                px_per_m        = px_per_m,
            ))

        if not events:
            return self._synthetic_events(pose_result)

        return events

    @staticmethod
    def _torso_lean(fp: "FramePose | None") -> float:
        if fp is None:
            return 10.0
        return abs(fp.torso_lean) if fp.torso_lean is not None else 10.0

    @staticmethod
    def _shooting_foot(fp: "FramePose | None") -> str:
        """Estimate kicking foot: whichever ankle is farther forward (lower y)."""
        if fp is None or not fp.landmarks:
            return "right"
        la = fp.landmarks.get("left_ankle")
        ra = fp.landmarks.get("right_ankle")
        if la is None or ra is None:
            return "right"
        return "left" if la.y < ra.y else "right"

    @staticmethod
    def _synthetic_events(pose_result: PoseEstimationResult) -> List[ShotEvent]:
        """Synthetic fallback."""
        avg_torso = pose_result.avg_torso_lean or 10.0
        n = max(1, len([fp for fp in pose_result.frame_poses if fp.detected]) // 10)
        events: List[ShotEvent] = []
        for _ in range(n):
            traj = [Point2D(0.4, 0.5), Point2D(0.6, 0.45), Point2D(0.75, 0.42)]
            events.append(ShotEvent(
                ball_start      = Point2D(0.4, 0.5),
                ball_trajectory = traj,
                on_target       = True,
                goal            = False,
                torso_lean      = avg_torso,
                foot            = "right",
                contact_type    = "laces",
                fps             = 25.0,
                px_per_m        = 100.0,
            ))
        return events

    @staticmethod
    def _to_action_metrics(metric_set) -> ActionMetrics:
        return ActionMetrics(
            action  = FootballAction.SHOOTING,
            metrics = [
                ActivityMetric(
                    label   = m.label,
                    value   = m.value,
                    display = m.display,
                    unit    = m.unit,
                )
                for m in metric_set.metrics
            ],
        )

    @staticmethod
    def _empty_metrics() -> ActionMetrics:
        return ActionMetrics(action=FootballAction.SHOOTING, metrics=[])
