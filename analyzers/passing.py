#!/usr/bin/env python3
"""
Passing Analyzer
================
Calculates passing metrics from pose landmarks and ball detections.

Output metrics:
  - Successful Passes
  - Missed Passes
  - Accuracy (%)
  - Body Position Score (0–100)
  - Passing Consistency (0–100)

All merged into PipelineContext.analysis.metrics["byAction"]["passing"].
"""

from __future__ import annotations

import math
from typing import List

from analyzers.base_analyzer   import BaseAnalyzer
from schemas.activity_schema   import ActionMetrics, ActivityMetric, FootballAction
from pipeline.frame_extractor  import ExtractedFrame
from pipeline.pose_estimator   import PoseEstimationResult
from pipeline.ball_detector    import BallDetectionResult
from utils.logger              import get_logger

log = get_logger(__name__)


class PassingAnalyzer(BaseAnalyzer):

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
            return self._run(frames, pose_result, ball_result)
        except Exception as exc:
            log.warning("PassingAnalyzer error: %s", exc)
            return self._empty()

    def _run(self, frames, pose_result, ball_result) -> ActionMetrics:
        dets       = ball_result.detections
        pose_map   = {fp.frame_index: fp for fp in pose_result.frame_poses if fp.detected}
        total_passes, successful, missed = 0, 0, 0
        body_scores:   List[float] = []
        speed_samples: List[float] = []

        for i in range(1, len(dets)):
            d0, d1 = dets[i - 1], dets[i]
            dx     = d1.center_x - d0.center_x
            dy     = d1.center_y - d0.center_y
            speed  = math.hypot(dx, dy)

            if speed < 10:
                continue   # ball not moving — not a pass

            total_passes += 1
            speed_samples.append(speed)

            # Completion heuristic: ball keeps moving in same direction.
            if i + 1 < len(dets):
                d2 = dets[i + 1]
                dx2 = d2.center_x - d1.center_x
                dy2 = d2.center_y - d1.center_y
                dot = dx * dx2 + dy * dy2
                successful += 1 if dot > 0 else 0
                missed     += 0 if dot > 0 else 1
            else:
                successful += 1   # assume last pass completed

            # Body position score: penalise torso lean at pass moment.
            fp = pose_map.get(d0.frame_index)
            if fp:
                lean = abs(fp.torso_lean_deg or 0) if hasattr(fp, "torso_lean_deg") else 0
                score = max(0.0, 100.0 - lean * 2)
                body_scores.append(score)

        accuracy    = round((successful / total_passes * 100) if total_passes > 0 else 0.0, 1)
        body_score  = round(sum(body_scores) / len(body_scores) if body_scores else 75.0, 1)

        # Consistency: lower std-dev of pass speed = more consistent.
        if len(speed_samples) >= 2:
            import statistics
            std = statistics.stdev(speed_samples)
            consistency = round(max(0.0, 100.0 - std * 0.8), 1)
        else:
            consistency = 75.0

        return ActionMetrics(
            action  = FootballAction.PASSING,
            metrics = [
                ActivityMetric("Successful Passes",   float(successful), str(successful),          ""),
                ActivityMetric("Missed Passes",        float(missed),     str(missed),              ""),
                ActivityMetric("Accuracy",             accuracy,          f"{accuracy}%",           "%"),
                ActivityMetric("Body Position Score",  body_score,        f"{body_score}/100",      ""),
                ActivityMetric("Passing Consistency",  consistency,       f"{consistency}/100",     ""),
                ActivityMetric("Total Passes",         float(total_passes), str(total_passes),      ""),
            ],
        )

    @staticmethod
    def _empty() -> ActionMetrics:
        return ActionMetrics(action=FootballAction.PASSING, metrics=[
            ActivityMetric("Successful Passes",  0.0, "—"),
            ActivityMetric("Missed Passes",       0.0, "—"),
            ActivityMetric("Accuracy",            0.0, "—"),
            ActivityMetric("Body Position Score", 0.0, "—"),
            ActivityMetric("Passing Consistency", 0.0, "—"),
        ])
