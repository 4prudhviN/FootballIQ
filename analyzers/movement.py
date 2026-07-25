#!/usr/bin/env python3
"""
Movement Analyzer
=================
Calculates movement metrics from pose landmarks.

Output metrics:
  - Balance            (0–100)
  - Speed              (km/h estimate)
  - Foot Placement     (0–100)
  - Gait Symmetry      (%)
  - Stride Consistency (0–100)

All merged into PipelineContext.analysis.metrics["byAction"]["movement"].
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

_FPS     = 25.0
_PX_PER_M = 100.0


class MovementAnalyzer(BaseAnalyzer):

    @property
    def name(self) -> str:
        return FootballAction.MOVEMENT.value

    def analyze(
        self,
        frames:      List[ExtractedFrame],
        pose_result: PoseEstimationResult,
        ball_result: BallDetectionResult,
    ) -> ActionMetrics:
        try:
            return self._run(pose_result)
        except Exception as exc:
            log.warning("MovementAnalyzer error: %s", exc)
            return self._empty()

    def _run(self, pose_result: PoseEstimationResult) -> ActionMetrics:
        detected = [fp for fp in pose_result.frame_poses if fp.detected]
        if not detected:
            return self._empty()

        # ── Balance: penalise lateral centre-of-gravity shift ────────────────
        hip_x_vals = []
        for fp in detected:
            lh = fp.get("left_hip")
            rh = fp.get("right_hip")
            if lh and rh:
                hip_x_vals.append((lh.x + rh.x) / 2)

        if len(hip_x_vals) >= 2:
            import statistics
            hip_std  = statistics.stdev(hip_x_vals)
            balance  = round(max(0.0, 100.0 - hip_std * 300), 1)
        else:
            balance = 75.0

        # ── Speed: hip midpoint displacement per frame → km/h ────────────────
        speeds = []
        for i in range(1, len(detected)):
            fp0 = detected[i - 1]
            fp1 = detected[i]
            lh0, rh0 = fp0.get("left_hip"), fp0.get("right_hip")
            lh1, rh1 = fp1.get("left_hip"), fp1.get("right_hip")
            if None in (lh0, rh0, lh1, rh1):
                continue
            cx0 = (lh0.x + rh0.x) / 2
            cy0 = (lh0.y + rh0.y) / 2
            cx1 = (lh1.x + rh1.x) / 2
            cy1 = (lh1.y + rh1.y) / 2
            dist_norm = math.hypot(cx1 - cx0, cy1 - cy0)
            speeds.append(dist_norm)

        avg_speed_norm = sum(speeds) / len(speeds) if speeds else 0.0
        speed_kmh      = round(avg_speed_norm * _FPS / _PX_PER_M * 3.6 * 1000, 1)

        # ── Foot placement: how close ankles stay to hip line ────────────────
        foot_scores = []
        for fp in detected:
            lh = fp.get("left_hip")
            rh = fp.get("right_hip")
            la = fp.get("left_ankle")
            ra = fp.get("right_ankle")
            if None in (lh, rh, la, ra):
                continue
            hip_cx = (lh.x + rh.x) / 2
            # Good foot placement: ankles close to hip centre line.
            l_dev = abs(la.x - hip_cx)
            r_dev = abs(ra.x - hip_cx)
            score = max(0.0, 100.0 - (l_dev + r_dev) * 200)
            foot_scores.append(score)

        foot_placement = round(
            sum(foot_scores) / len(foot_scores) if foot_scores else 70.0, 1
        )

        # ── Gait symmetry: left vs right ankle Y variance ────────────────────
        left_y  = [fp.get("left_ankle").y  for fp in detected if fp.get("left_ankle")]
        right_y = [fp.get("right_ankle").y for fp in detected if fp.get("right_ankle")]

        if left_y and right_y:
            avg_l = sum(left_y)  / len(left_y)
            avg_r = sum(right_y) / len(right_y)
            denom = max(avg_l, avg_r)
            asym  = abs(avg_l - avg_r) / denom if denom > 0 else 0.0
            gait_symmetry = round((1.0 - asym) * 100, 1)
        else:
            gait_symmetry = 80.0

        # ── Stride consistency ────────────────────────────────────────────────
        stride_consistency = round(
            max(0.0, 100.0 - (abs(100 - gait_symmetry) * 1.5)), 1
        )

        return ActionMetrics(
            action  = FootballAction.MOVEMENT,
            metrics = [
                ActivityMetric("Balance",            balance,           f"{balance}/100",       ""),
                ActivityMetric("Speed",              speed_kmh,         f"{speed_kmh} km/h",    "km/h"),
                ActivityMetric("Foot Placement",     foot_placement,    f"{foot_placement}/100",""),
                ActivityMetric("Gait Symmetry",      gait_symmetry,     f"{gait_symmetry}%",    "%"),
                ActivityMetric("Stride Consistency", stride_consistency,f"{stride_consistency}/100",""),
            ],
        )

    @staticmethod
    def _empty() -> ActionMetrics:
        return ActionMetrics(action=FootballAction.MOVEMENT, metrics=[
            ActivityMetric("Balance",            0.0, "—"),
            ActivityMetric("Speed",              0.0, "—"),
            ActivityMetric("Foot Placement",     0.0, "—"),
            ActivityMetric("Gait Symmetry",      0.0, "—"),
            ActivityMetric("Stride Consistency", 0.0, "—"),
        ])
