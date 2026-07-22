#!/usr/bin/env python3
"""
Shooting Metrics
================
Calculates pure numerical metrics for shooting actions.
No AI, no feedback — numbers only.

Metrics produced:
  - Total Shots              (count)
  - Shots on Target          (count)
  - Shots off Target         (count)
  - Shot Accuracy            (%)
  - Average Shot Velocity    (km/h)
  - Max Shot Velocity        (km/h)
  - Average Launch Angle     (degrees)
  - Torso Lean at Contact    (degrees)
  - Foot Strike Contact Type (categorical: Instep / Laces / Inside / Outside)
  - Weak Foot Shot Ratio     (%)
  - Average Shot Distance    (m)
  - Goals Scored             (count)  — if outcome data is available
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from metrics.common_metrics import (
    MetricSet, MetricValue, Point2D,
    percentage, fmt_pct, fmt_speed_kmh, fmt_angle, fmt_distance,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class ShotEvent:
    """Data for a single shot attempt."""
    ball_start:       Point2D       # ball position at contact
    ball_trajectory:  List[Point2D] # subsequent ball positions
    on_target:        bool          # True if the shot was on target
    goal:             bool          # True if it resulted in a goal
    torso_lean:       float         # degrees (neg = leaning back)
    foot:             str           # "left" or "right"
    contact_type:     str           # "instep", "laces", "inside", "outside"
    fps:              float = 25.0
    px_per_m:         float = 100.0


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class ShootingMetrics:
    """
    Calculate shooting metrics from a list of ShotEvent objects.

    Usage::

        calculator = ShootingMetrics()
        metric_set = calculator.calculate(shot_events)
        print(metric_set.to_dict())
    """

    def calculate(self, events: List[ShotEvent]) -> MetricSet:
        if not events:
            return self._empty()

        total       = len(events)
        on_target   = sum(1 for e in events if e.on_target)
        off_target  = total - on_target
        goals       = sum(1 for e in events if e.goal)
        accuracy    = percentage(on_target, total)

        # Shot velocity: speed of ball in first two frames after contact.
        velocities: List[float] = []
        for ev in events:
            if len(ev.ball_trajectory) >= 2:
                dx = ev.ball_trajectory[1].x - ev.ball_trajectory[0].x
                dy = ev.ball_trajectory[1].y - ev.ball_trajectory[0].y
                velocities.append(math.hypot(dx, dy))

        avg_vel = sum(velocities) / len(velocities) if velocities else 0.0
        max_vel = max(velocities) if velocities else 0.0

        fps   = events[0].fps
        px_m  = events[0].px_per_m

        # Launch angle: angle of ball trajectory from horizontal.
        launch_angles: List[float] = []
        for ev in events:
            if len(ev.ball_trajectory) >= 2:
                dx = ev.ball_trajectory[1].x - ev.ball_trajectory[0].x
                dy = ev.ball_trajectory[1].y - ev.ball_trajectory[0].y
                angle = abs(math.degrees(math.atan2(dy, dx)))
                launch_angles.append(angle)

        avg_launch = sum(launch_angles) / len(launch_angles) if launch_angles else 0.0

        # Torso lean at contact.
        torso_leans = [abs(e.torso_lean) for e in events]
        avg_torso   = sum(torso_leans) / len(torso_leans) if torso_leans else 0.0

        # Foot usage.
        left_shots    = sum(1 for e in events if e.foot == "left")
        weak_foot_ratio = percentage(left_shots, total)

        # Shot distance.
        distances = []
        for ev in events:
            if ev.ball_trajectory:
                d = ev.ball_start.distance_to(ev.ball_trajectory[-1])
                distances.append(d / px_m if px_m > 0 else d)
        avg_dist = sum(distances) / len(distances) if distances else 0.0

        # Most common contact type.
        contact_counts: dict[str, int] = {}
        for ev in events:
            contact_counts[ev.contact_type] = contact_counts.get(ev.contact_type, 0) + 1
        dominant_contact = max(contact_counts, key=contact_counts.get) if contact_counts else "—"

        ms = MetricSet(activity="shooting")
        ms.metrics = [
            MetricValue("Total Shots",             total,       str(total),                             ""),
            MetricValue("Shots on Target",         on_target,   str(on_target),                         ""),
            MetricValue("Shots off Target",        off_target,  str(off_target),                        ""),
            MetricValue("Shot Accuracy",           accuracy,    fmt_pct(accuracy),                      "%"),
            MetricValue("Goals Scored",            goals,       str(goals),                             ""),
            MetricValue("Average Shot Velocity",   avg_vel,     fmt_speed_kmh(avg_vel, fps, px_m),      "km/h"),
            MetricValue("Max Shot Velocity",       max_vel,     fmt_speed_kmh(max_vel, fps, px_m),      "km/h"),
            MetricValue("Average Launch Angle",    avg_launch,  fmt_angle(avg_launch),                  "°"),
            MetricValue("Torso Lean at Contact",   avg_torso,   fmt_angle(avg_torso),                   "°"),
            MetricValue("Foot Strike Contact",     0.0,         dominant_contact.capitalize(),           ""),
            MetricValue("Weak Foot Shot Ratio",    weak_foot_ratio, fmt_pct(weak_foot_ratio),           "%"),
            MetricValue("Average Shot Distance",   avg_dist,    fmt_distance(avg_dist),                 "m"),
        ]
        return ms

    def _empty(self) -> MetricSet:
        ms = MetricSet(activity="shooting")
        for label in [
            "Total Shots", "Shots on Target", "Shots off Target",
            "Shot Accuracy", "Goals Scored", "Average Shot Velocity",
            "Max Shot Velocity", "Average Launch Angle", "Torso Lean at Contact",
            "Foot Strike Contact", "Weak Foot Shot Ratio", "Average Shot Distance",
        ]:
            ms.metrics.append(MetricValue(label, 0.0, "—"))
        return ms
