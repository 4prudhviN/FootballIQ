#!/usr/bin/env python3
"""
Dribbling Metrics
=================
Calculates pure numerical metrics for dribbling actions.
No AI, no feedback — numbers only.

Metrics produced:
  - Successful Dribbles      (count)
  - Failed Dribbles          (count)
  - Dribble Success Rate     (%)
  - Average Speed with Ball  (km/h)
  - Max Speed with Ball      (km/h)
  - Touch Tightness Index    (cm variance from ideal contact)
  - Change of Direction Count (count)
  - Average Direction Change Angle (degrees)
  - Close Control Score      (0–100)
  - Ball Retention Rate      (%)
  - Distance Covered         (m)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from metrics.common_metrics import (
    MetricSet, MetricValue, Point2D,
    percentage, speed_px_per_frame, fmt_pct, fmt_speed_kmh,
    fmt_angle, fmt_distance,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class DribbleEvent:
    """Data for a single dribbling sequence."""
    ball_positions:  List[Point2D]    # ball centre per frame
    body_positions:  List[Point2D]    # player hip midpoint per frame
    completed:       bool             # True if player retained ball past defender
    fps:             float = 25.0
    px_per_m:        float = 100.0


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

COD_ANGLE_THRESHOLD = 30.0    # degrees — change of direction threshold


class DribblingMetrics:
    """
    Calculate dribbling metrics from a list of DribbleEvent objects.

    Usage::

        calculator = DribblingMetrics()
        metric_set = calculator.calculate(dribble_events)
        print(metric_set.to_dict())
    """

    def calculate(self, events: List[DribbleEvent]) -> MetricSet:
        if not events:
            return self._empty()

        total       = len(events)
        successful  = sum(1 for e in events if e.completed)
        failed      = total - successful
        success_rate = percentage(successful, total)

        # Aggregate metrics across all dribble events.
        all_speeds:       List[float] = []
        all_cod_angles:   List[float] = []
        all_cod_counts:   List[int]   = []
        all_touch_diffs:  List[float] = []
        total_distance    = 0.0

        for ev in events:
            if len(ev.ball_positions) < 2:
                continue

            # Ball speeds (pixels/frame).
            sp = speed_px_per_frame(ev.ball_positions)
            all_speeds.extend(sp)

            # Total distance (normalised units → metres).
            for i in range(1, len(ev.ball_positions)):
                total_distance += ev.ball_positions[i].distance_to(ev.ball_positions[i - 1])

            # Changes of direction.
            cod_count = 0
            for i in range(1, len(ev.ball_positions) - 1):
                a  = ev.ball_positions[i - 1]
                v  = ev.ball_positions[i]
                b  = ev.ball_positions[i + 1]
                dx1, dy1 = v.x - a.x, v.y - a.y
                dx2, dy2 = b.x - v.x, b.y - v.y
                if (dx1 == 0 and dy1 == 0) or (dx2 == 0 and dy2 == 0):
                    continue
                import math
                a1 = math.atan2(dy1, dx1)
                a2 = math.atan2(dy2, dx2)
                angle_change = abs(math.degrees(a2 - a1))
                if angle_change > 180:
                    angle_change = 360 - angle_change
                if angle_change >= COD_ANGLE_THRESHOLD:
                    cod_count += 1
                    all_cod_angles.append(angle_change)
            all_cod_counts.append(cod_count)

            # Touch tightness — proximity of ball to body.
            if len(ev.body_positions) == len(ev.ball_positions):
                for bp, pp in zip(ev.ball_positions, ev.body_positions):
                    all_touch_diffs.append(bp.distance_to(pp))

        fps     = events[0].fps
        px_m    = events[0].px_per_m

        avg_sp   = sum(all_speeds) / len(all_speeds)   if all_speeds   else 0.0
        max_sp   = max(all_speeds)                      if all_speeds   else 0.0
        avg_cod  = sum(all_cod_angles) / len(all_cod_angles) if all_cod_angles else 0.0
        total_cod = sum(all_cod_counts)

        # Touch tightness in normalised units → cm.
        avg_touch_n = sum(all_touch_diffs) / len(all_touch_diffs) if all_touch_diffs else 0.0
        avg_touch_cm = avg_touch_n * px_m * 0.01 * 100   # rough cm conversion

        # Close control score: lower touch variance = higher score.
        close_control = max(0.0, 100.0 - avg_touch_cm * 2)

        # Ball retention rate = success rate (same concept different name).
        ball_retention = success_rate

        dist_m = total_distance * (1.0 / px_m) if px_m > 0 else total_distance

        ms = MetricSet(activity="dribbling")
        ms.metrics = [
            MetricValue("Successful Dribbles",        successful,    str(successful),                    ""),
            MetricValue("Failed Dribbles",             failed,        str(failed),                        ""),
            MetricValue("Dribble Success Rate",        success_rate,  fmt_pct(success_rate),              "%"),
            MetricValue("Average Speed with Ball",     avg_sp,        fmt_speed_kmh(avg_sp, fps, px_m),   "km/h"),
            MetricValue("Max Speed with Ball",         max_sp,        fmt_speed_kmh(max_sp, fps, px_m),   "km/h"),
            MetricValue("Touch Tightness Index",       avg_touch_cm,  f"±{avg_touch_cm:.1f} cm",          "cm"),
            MetricValue("Change of Direction Count",   total_cod,     str(total_cod),                     ""),
            MetricValue("Avg Direction Change Angle",  avg_cod,       fmt_angle(avg_cod),                 "°"),
            MetricValue("Close Control Score",         close_control, fmt_pct(close_control),             ""),
            MetricValue("Ball Retention Rate",         ball_retention,fmt_pct(ball_retention),            "%"),
            MetricValue("Distance Covered",            dist_m,        fmt_distance(dist_m),               "m"),
        ]
        return ms

    def _empty(self) -> MetricSet:
        ms = MetricSet(activity="dribbling")
        for label in [
            "Successful Dribbles", "Failed Dribbles", "Dribble Success Rate",
            "Average Speed with Ball", "Max Speed with Ball", "Touch Tightness Index",
            "Change of Direction Count", "Avg Direction Change Angle",
            "Close Control Score", "Ball Retention Rate", "Distance Covered",
        ]:
            ms.metrics.append(MetricValue(label, 0.0, "—"))
        return ms
