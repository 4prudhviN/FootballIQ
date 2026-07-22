#!/usr/bin/env python3
"""
Movement Metrics
================
Calculates pure numerical metrics for player movement and locomotion.
No AI, no feedback — numbers only.

Metrics produced:
  - Total Distance Covered   (m)
  - Max Sprint Speed         (km/h)
  - Average Speed            (km/h)
  - Sprint Count             (count)
  - Sprint Distance          (m)
  - Gait Asymmetry Index     (%)
  - Stride Length (Left)     (m)
  - Stride Length (Right)    (m)
  - Stride Frequency         (strides/min)
  - Acceleration Bursts      (count)
  - Deceleration Events      (count)
  - High-Intensity Runs      (count — above 70% max speed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from metrics.common_metrics import (
    MetricSet, MetricValue, Point2D,
    speed_px_per_frame, symmetry_ratio,
    fmt_pct, fmt_speed_kmh, fmt_distance, fmt_time,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class MovementFrame:
    """Player position and foot contact data for a single frame."""
    position:         Point2D
    left_ankle:       Point2D
    right_ankle:      Point2D
    timestamp_s:      float


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SPRINT_THRESHOLD_RATIO  = 0.70    # fraction of max speed = sprint
ACCEL_THRESHOLD         = 0.005   # normalised-units/frame² = acceleration burst
DECEL_THRESHOLD         = -0.005


class MovementMetrics:
    """
    Calculate movement metrics from a list of MovementFrame objects.

    Parameters
    ----------
    fps : float
    px_per_m : float

    Usage::

        calculator = MovementMetrics(fps=25.0, px_per_m=100.0)
        metric_set = calculator.calculate(frames)
        print(metric_set.to_dict())
    """

    def __init__(self, fps: float = 25.0, px_per_m: float = 100.0) -> None:
        self.fps      = fps
        self.px_per_m = px_per_m

    def calculate(self, frames: List[MovementFrame]) -> MetricSet:
        if len(frames) < 2:
            return self._empty()

        positions = [f.position for f in frames]
        speeds    = speed_px_per_frame(positions)   # normalised-units / frame

        if not speeds:
            return self._empty()

        fps   = self.fps
        px_m  = self.px_per_m

        max_speed = max(speeds)
        avg_speed = sum(speeds) / len(speeds)

        # Total distance.
        total_dist_n = sum(speeds)
        total_dist_m = total_dist_n / px_m if px_m > 0 else total_dist_n

        # Sprint detection.
        sprint_threshold = max_speed * SPRINT_THRESHOLD_RATIO
        sprinting         = [s >= sprint_threshold for s in speeds]
        sprint_count      = sum(
            1 for i in range(1, len(sprinting)) if sprinting[i] and not sprinting[i - 1]
        )
        sprint_dist_n = sum(s for s, sp in zip(speeds, sprinting) if sp)
        sprint_dist_m = sprint_dist_n / px_m if px_m > 0 else sprint_dist_n

        # High-intensity runs (> 70% max speed, not necessarily max sprint).
        hi_runs = sum(1 for s in speeds if s >= sprint_threshold)

        # Acceleration / deceleration events.
        accels  = 0
        decels  = 0
        for i in range(1, len(speeds)):
            delta = speeds[i] - speeds[i - 1]
            if delta >= ACCEL_THRESHOLD:
                accels += 1
            elif delta <= DECEL_THRESHOLD:
                decels += 1

        # Gait asymmetry — left vs right ankle Y-position variance.
        left_y  = [f.left_ankle.y  for f in frames]
        right_y = [f.right_ankle.y for f in frames]

        import numpy as np
        avg_l = float(np.mean(left_y))
        avg_r = float(np.mean(right_y))
        denom = max(avg_l, avg_r)
        gait_asym = abs(avg_l - avg_r) / denom * 100 if denom > 0 else 0.0

        # Stride length estimation from ankle Y-minima (foot plants).
        def stride_lengths(ankle_y: List[float]) -> List[float]:
            """Find local minima (foot plants) and measure spacing."""
            plants = []
            for i in range(1, len(ankle_y) - 1):
                if ankle_y[i] < ankle_y[i - 1] and ankle_y[i] < ankle_y[i + 1]:
                    plants.append(i)
            if len(plants) < 2:
                return []
            x_pos = [frames[p].position.x for p in plants]
            strides = [abs(x_pos[i] - x_pos[i - 1]) for i in range(1, len(x_pos))]
            return [s / px_m for s in strides] if px_m > 0 else strides

        sl_left  = stride_lengths(left_y)
        sl_right = stride_lengths(right_y)

        avg_sl_left  = sum(sl_left)  / len(sl_left)  if sl_left  else 0.0
        avg_sl_right = sum(sl_right) / len(sl_right) if sl_right else 0.0

        # Stride frequency (strides per minute).
        total_strides  = len(sl_left) + len(sl_right)
        duration_s     = frames[-1].timestamp_s - frames[0].timestamp_s if frames else 1.0
        stride_freq    = (total_strides / duration_s * 60) if duration_s > 0 else 0.0

        ms = MetricSet(activity="movement")
        ms.metrics = [
            MetricValue("Total Distance Covered",  total_dist_m, fmt_distance(total_dist_m),              "m"),
            MetricValue("Max Sprint Speed",        max_speed,    fmt_speed_kmh(max_speed, fps, px_m),     "km/h"),
            MetricValue("Average Speed",           avg_speed,    fmt_speed_kmh(avg_speed, fps, px_m),     "km/h"),
            MetricValue("Sprint Count",            sprint_count, str(sprint_count),                       ""),
            MetricValue("Sprint Distance",         sprint_dist_m,fmt_distance(sprint_dist_m),             "m"),
            MetricValue("Gait Asymmetry Index",    gait_asym,    f"{gait_asym:.1f}%",                     "%"),
            MetricValue("Stride Length (Left)",    avg_sl_left,  fmt_distance(avg_sl_left),               "m"),
            MetricValue("Stride Length (Right)",   avg_sl_right, fmt_distance(avg_sl_right),              "m"),
            MetricValue("Stride Frequency",        stride_freq,  f"{stride_freq:.0f} spm",               "spm"),
            MetricValue("Acceleration Bursts",     accels,       str(accels),                             ""),
            MetricValue("Deceleration Events",     decels,       str(decels),                             ""),
            MetricValue("High-Intensity Runs",     hi_runs,      str(hi_runs),                            ""),
        ]
        return ms

    def _empty(self) -> MetricSet:
        ms = MetricSet(activity="movement")
        for label in [
            "Total Distance Covered", "Max Sprint Speed", "Average Speed",
            "Sprint Count", "Sprint Distance", "Gait Asymmetry Index",
            "Stride Length (Left)", "Stride Length (Right)", "Stride Frequency",
            "Acceleration Bursts", "Deceleration Events", "High-Intensity Runs",
        ]:
            ms.metrics.append(MetricValue(label, 0.0, "—"))
        return ms
