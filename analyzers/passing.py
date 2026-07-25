#!/usr/bin/env python3
"""
Passing Analyzer
================
Calculates passing metrics from pose landmarks and ball detections.

Output:
{
    "successful_passes": 26,
    "failed_passes":      4,
    "accuracy":          86,
    "average_speed":     28,
    "body_alignment":    "Good"
}

No AI. Just numbers.
"""

from __future__ import annotations

import math
from typing import List

from analyzers.base_analyzer  import BaseAnalyzer
from schemas.activity_schema  import ActionMetrics, ActivityMetric, FootballAction
from pipeline.frame_extractor import ExtractedFrame
from pipeline.pose_estimator  import PoseEstimationResult
from pipeline.ball_detector   import BallDetectionResult
from utils.logger             import get_logger

log = get_logger(__name__)

# Minimum ball speed (pixels/frame) to count as a pass attempt.
_MIN_PASS_SPEED = 10.0

# Body alignment score → label mapping.
def _alignment_label(score: float) -> str:
    if score >= 80: return "Good"
    if score >= 60: return "Fair"
    return "Poor"


class PassingAnalyzer(BaseAnalyzer):
    """
    Analyzes passing actions from ball detections and pose data.
    Returns pure numbers — no coaching, no AI.
    """

    @property
    def name(self) -> str:
        return FootballAction.PASSING.value

    def analyze(
        self,
        frames:      List[ExtractedFrame],
        pose_result: PoseEstimationResult,
        ball_result: BallDetectionResult,
    ) -> ActionMetrics:
        try:
            return self._run(pose_result, ball_result)
        except Exception as exc:
            log.warning("PassingAnalyzer error: %s", exc)
            return self._empty()

    # ------------------------------------------------------------------

    def _run(
        self,
        pose_result: PoseEstimationResult,
        ball_result: BallDetectionResult,
    ) -> ActionMetrics:
        dets     = ball_result.detections
        pose_map = {fp.frame_index: fp for fp in pose_result.frame_poses if fp.detected}

        successful_passes = 0
        failed_passes     = 0
        speeds:           List[float] = []
        body_scores:      List[float] = []

        for i in range(1, len(dets)):
            d0, d1 = dets[i - 1], dets[i]
            dx     = d1.center_x - d0.center_x
            dy     = d1.center_y - d0.center_y
            speed  = math.hypot(dx, dy)

            if speed < _MIN_PASS_SPEED:
                continue   # not a pass — ball not moving enough

            speeds.append(speed)

            # Completion heuristic: ball maintains direction next frame.
            completed = False
            if i + 1 < len(dets):
                d2  = dets[i + 1]
                dx2 = d2.center_x - d1.center_x
                dy2 = d2.center_y - d1.center_y
                dot = dx * dx2 + dy * dy2
                completed = dot > 0
            else:
                completed = True   # last detected — assume completed

            if completed:
                successful_passes += 1
            else:
                failed_passes += 1

            # Body alignment: torso lean at moment of pass.
            fp = pose_map.get(d0.frame_index)
            if fp:
                lean = abs(getattr(fp, "torso_lean_deg", None) or fp.torso_lean or 0)
                score = max(0.0, 100.0 - lean * 2.5)
                body_scores.append(score)

        total_passes  = successful_passes + failed_passes
        accuracy      = round((successful_passes / total_passes * 100) if total_passes > 0 else 0)
        average_speed = round(sum(speeds) / len(speeds) if speeds else 0)
        body_score    = round(sum(body_scores) / len(body_scores) if body_scores else 75.0)
        alignment     = _alignment_label(body_score)

        return ActionMetrics(
            action  = FootballAction.PASSING,
            metrics = [
                ActivityMetric("successful_passes", float(successful_passes), str(successful_passes), ""),
                ActivityMetric("failed_passes",      float(failed_passes),     str(failed_passes),     ""),
                ActivityMetric("accuracy",           float(accuracy),          str(accuracy),          "%"),
                ActivityMetric("average_speed",      float(average_speed),     str(average_speed),     "px/f"),
                ActivityMetric("body_alignment",     float(body_score),        alignment,              ""),
            ],
        )

    @staticmethod
    def _empty() -> ActionMetrics:
        return ActionMetrics(
            action  = FootballAction.PASSING,
            metrics = [
                ActivityMetric("successful_passes", 0.0, "0",   ""),
                ActivityMetric("failed_passes",      0.0, "0",   ""),
                ActivityMetric("accuracy",           0.0, "0",   "%"),
                ActivityMetric("average_speed",      0.0, "0",   "px/f"),
                ActivityMetric("body_alignment",     0.0, "N/A", ""),
            ],
        )
