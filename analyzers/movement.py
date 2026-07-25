#!/usr/bin/env python3
"""
Movement Analyzer
=================
Calculates movement metrics from pose landmarks.
These metrics support all other analyzers.

Output:
{
    "balance":        88,
    "stride_length":  1.2,
    "body_alignment": "Good",
    "acceleration":   3.4,
    "foot_placement": 76
}

No AI. Just numbers.
"""

from __future__ import annotations

import math
import statistics
from typing import List, Optional, Tuple

from analyzers.base_analyzer  import BaseAnalyzer
from schemas.activity_schema  import ActionMetrics, ActivityMetric, FootballAction
from pipeline.frame_extractor import ExtractedFrame
from pipeline.pose_estimator  import PoseEstimationResult, FramePose
from pipeline.ball_detector   import BallDetectionResult
from utils.logger             import get_logger

log = get_logger(__name__)

_FPS      = 25.0
_PX_PER_M = 100.0   # pixels per metre (default calibration)


def _alignment_label(score: float) -> str:
    if score >= 80: return "Good"
    if score >= 60: return "Fair"
    return "Poor"


class MovementAnalyzer(BaseAnalyzer):
    """
    Analyzes player movement from pose landmarks.
    Output supports all other activity analyzers.
    No AI — pure numbers.
    """

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

    # ------------------------------------------------------------------

    def _run(self, pose_result: PoseEstimationResult) -> ActionMetrics:
        detected = [fp for fp in pose_result.frame_poses if fp.detected]
        if not detected:
            return self._empty()

        # ── Balance ───────────────────────────────────────────────────────
        # Measures lateral hip stability (low std-dev = good balance).
        hip_x_vals = []
        for fp in detected:
            lh = fp.get("left_hip")
            rh = fp.get("right_hip")
            if lh and rh:
                hip_x_vals.append((lh.x + rh.x) / 2)

        if len(hip_x_vals) >= 2:
            hip_std = statistics.stdev(hip_x_vals)
            balance = round(max(0.0, 100.0 - hip_std * 400))
        else:
            balance = 75

        # ── Stride Length ─────────────────────────────────────────────────
        # Estimate from ankle-to-ankle horizontal distance during stance.
        stride_lengths: List[float] = []
        for fp in detected:
            la = fp.get("left_ankle")
            ra = fp.get("right_ankle")
            if la and ra:
                dist_norm = abs(la.x - ra.x)
                dist_m    = dist_norm / _PX_PER_M * 1000   # rough metres
                if 0.1 < dist_m < 3.0:   # sanity filter
                    stride_lengths.append(dist_m)

        avg_stride = round(
            sum(stride_lengths) / len(stride_lengths) if stride_lengths else 0.0, 2
        )

        # ── Body Alignment ────────────────────────────────────────────────
        # Average absolute torso lean across all frames.
        leans: List[float] = []
        for fp in detected:
            lean = self._torso_lean(fp)
            if lean is not None:
                leans.append(abs(lean))

        avg_lean      = sum(leans) / len(leans) if leans else 0.0
        align_score   = round(max(0.0, 100.0 - avg_lean * 2.5))
        body_alignment = _alignment_label(align_score)

        # ── Acceleration ──────────────────────────────────────────────────
        # Frame-to-frame hip midpoint speed change (pixels/frame²).
        speeds: List[float] = []
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
            speeds.append(math.hypot(cx1 - cx0, cy1 - cy0))

        if len(speeds) >= 2:
            accels = [abs(speeds[i] - speeds[i - 1]) for i in range(1, len(speeds))]
            avg_accel = round(sum(accels) / len(accels) * _FPS * 10, 1)  # scaled
        else:
            avg_accel = 0.0

        # ── Foot Placement ────────────────────────────────────────────────
        # How well ankles align under the hip centre line.
        foot_scores: List[float] = []
        for fp in detected:
            lh = fp.get("left_hip")
            rh = fp.get("right_hip")
            la = fp.get("left_ankle")
            ra = fp.get("right_ankle")
            if None in (lh, rh, la, ra):
                continue
            hip_cx = (lh.x + rh.x) / 2
            l_dev  = abs(la.x - hip_cx)
            r_dev  = abs(ra.x - hip_cx)
            score  = max(0.0, 100.0 - (l_dev + r_dev) * 250)
            foot_scores.append(score)

        foot_placement = round(
            sum(foot_scores) / len(foot_scores) if foot_scores else 70.0
        )

        return ActionMetrics(
            action  = FootballAction.MOVEMENT,
            metrics = [
                ActivityMetric("balance",        float(balance),        str(balance),        ""),
                ActivityMetric("stride_length",  avg_stride,            str(avg_stride),     "m"),
                ActivityMetric("body_alignment", float(align_score),    body_alignment,      ""),
                ActivityMetric("acceleration",   avg_accel,             str(avg_accel),      "m/s²"),
                ActivityMetric("foot_placement", float(foot_placement), str(foot_placement), ""),
            ],
        )

    @staticmethod
    def _torso_lean(fp: FramePose) -> Optional[float]:
        ls = fp.get("left_shoulder")
        rs = fp.get("right_shoulder")
        lh = fp.get("left_hip")
        rh = fp.get("right_hip")
        if None in (ls, rs, lh, rh):
            return None
        sh_x = (ls.x + rs.x) / 2
        sh_y = (ls.y + rs.y) / 2
        hi_x = (lh.x + rh.x) / 2
        hi_y = (lh.y + rh.y) / 2
        return math.degrees(math.atan2(-(sh_x - hi_x), -(sh_y - hi_y)))

    @staticmethod
    def _empty() -> ActionMetrics:
        return ActionMetrics(
            action  = FootballAction.MOVEMENT,
            metrics = [
                ActivityMetric("balance",        0.0, "0",   ""),
                ActivityMetric("stride_length",  0.0, "0",   "m"),
                ActivityMetric("body_alignment", 0.0, "N/A", ""),
                ActivityMetric("acceleration",   0.0, "0",   "m/s²"),
                ActivityMetric("foot_placement", 0.0, "0",   ""),
            ],
        )
