#!/usr/bin/env python3
"""
Passing Metrics
===============
Calculates pure numerical metrics for passing actions.
No AI, no feedback — numbers only.

Metrics produced:
  - Passing Accuracy         (%)
  - Successful Passes        (count)
  - Failed Passes            (count)
  - Total Passes             (count)
  - Average Pass Speed       (km/h)
  - Average Pass Distance    (m)
  - Short Pass Ratio         (%)
  - Long Pass Ratio          (%)
  - First Touch Control      (score 0–100)
  - Ball Control Index       (score 0–100)
  - Weak Foot Ratio          (%)
  - Pass Completion Trend    (early% vs late% — fatigue indicator)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from metrics.common_metrics import (
    MetricSet, MetricValue, Point2D,
    percentage, fmt_pct, fmt_speed_kmh, fmt_distance, fmt_time,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class PassEvent:
    """Data for a single pass attempt."""
    start:       Point2D           # ball position at release
    end:         Point2D           # ball position at reception / out-of-bounds
    completed:   bool              # True if pass reached a teammate
    speed_px_f:  float             # ball speed in pixels/frame at release
    foot:        str               # "left" or "right"
    frame_index: int               # frame number in original video
    fps:         float = 25.0
    px_per_m:    float = 100.0     # pixels per metre (calibration)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

# Distance thresholds in normalised [0,1] space.
SHORT_PASS_THRESHOLD = 0.15   # < 15% of pitch width
LONG_PASS_THRESHOLD  = 0.35   # > 35% of pitch width


class PassingMetrics:
    """
    Calculate passing metrics from a list of PassEvent objects.

    Usage::

        calculator = PassingMetrics()
        metric_set = calculator.calculate(pass_events)
        print(metric_set.to_dict())
    """

    def calculate(self, events: List[PassEvent]) -> MetricSet:
        ms = MetricSet(activity="passing")

        if not events:
            return self._empty()

        total       = len(events)
        successful  = sum(1 for e in events if e.completed)
        failed      = total - successful
        accuracy    = percentage(successful, total)

        distances   = [e.start.distance_to(e.end) for e in events]
        avg_dist    = sum(distances) / len(distances) if distances else 0.0

        speeds      = [e.speed_px_f for e in events if e.speed_px_f > 0]
        avg_speed_pf = sum(speeds) / len(speeds) if speeds else 0.0

        short = sum(1 for d in distances if d < SHORT_PASS_THRESHOLD)
        long_ = sum(1 for d in distances if d > LONG_PASS_THRESHOLD)
        short_ratio = percentage(short, total)
        long_ratio  = percentage(long_, total)

        left_passes  = sum(1 for e in events if e.foot == "left")
        weak_foot_r  = percentage(left_passes, total)   # assumes right is dominant

        # Completion trend: compare first half vs second half.
        mid   = total // 2
        early = events[:mid]
        late  = events[mid:]
        early_acc = percentage(sum(1 for e in early if e.completed), len(early)) if early else 0.0
        late_acc  = percentage(sum(1 for e in late  if e.completed), len(late))  if late  else 0.0
        trend     = round(late_acc - early_acc, 1)

        # First touch control — proxy: lower average ball speed at reception = better control.
        first_touch_score = max(0.0, 100.0 - avg_speed_pf * 5)

        # Ball control index — composite of accuracy + first touch.
        ball_control = round((accuracy * 0.6 + first_touch_score * 0.4), 1)

        # Average pass speed display.
        avg_speed_display = fmt_speed_kmh(
            avg_speed_pf,
            events[0].fps,
            events[0].px_per_m,
        ) if events else "—"

        ms.metrics = [
            MetricValue("Passing Accuracy",      accuracy,          fmt_pct(accuracy),       "%"),
            MetricValue("Successful Passes",      successful,        str(successful),         ""),
            MetricValue("Failed Passes",          failed,            str(failed),             ""),
            MetricValue("Total Passes",           total,             str(total),              ""),
            MetricValue("Average Pass Speed",     avg_speed_pf,      avg_speed_display,       "km/h"),
            MetricValue("Average Pass Distance",  avg_dist,          fmt_distance(avg_dist * 10), "m"),
            MetricValue("Short Pass Ratio",       short_ratio,       fmt_pct(short_ratio),    "%"),
            MetricValue("Long Pass Ratio",        long_ratio,        fmt_pct(long_ratio),     "%"),
            MetricValue("First Touch Control",    first_touch_score, fmt_pct(first_touch_score), ""),
            MetricValue("Ball Control Index",     ball_control,      fmt_pct(ball_control),   ""),
            MetricValue("Weak Foot Ratio",        weak_foot_r,       fmt_pct(weak_foot_r),    "%"),
            MetricValue("Completion Trend",       trend,             f"{trend:+.1f}%",        "%"),
        ]
        return ms

    def _empty(self) -> MetricSet:
        ms = MetricSet(activity="passing")
        for label in [
            "Passing Accuracy", "Successful Passes", "Failed Passes",
            "Total Passes", "Average Pass Speed", "Average Pass Distance",
            "Short Pass Ratio", "Long Pass Ratio", "First Touch Control",
            "Ball Control Index", "Weak Foot Ratio", "Completion Trend",
        ]:
            ms.metrics.append(MetricValue(label, 0.0, "—"))
        return ms
