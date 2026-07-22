#!/usr/bin/env python3
"""
Defending Metrics
=================
Calculates pure numerical metrics for defending actions.
No AI, no feedback — numbers only.

Metrics produced:
  - Total Defensive Actions  (count)
  - Successful Tackles       (count)
  - Failed Tackles           (count)
  - Tackle Success Rate      (%)
  - Interceptions            (count)
  - Clearances               (count)
  - Aerial Duels Won         (count)
  - Aerial Duel Win Rate     (%)
  - Positioning Score        (0–100)
  - Recovery Speed           (km/h)
  - Distance Covered         (m)
  - Fouls Committed          (count)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from metrics.common_metrics import (
    MetricSet, MetricValue, Point2D,
    percentage, speed_px_per_frame, fmt_pct, fmt_speed_kmh, fmt_distance,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class TackleEvent:
    """Data for a single tackle attempt."""
    successful:   bool
    foul:         bool = False


@dataclass
class AerialDuelEvent:
    """Data for a single aerial duel."""
    won: bool


@dataclass
class DefendingEvent:
    """Aggregate data for a defending sequence."""
    tackles:       List[TackleEvent]
    interceptions: int
    clearances:    int
    aerial_duels:  List[AerialDuelEvent]
    body_positions: List[Point2D]       # player position over time
    ideal_positions: List[Point2D]      # expected defensive position
    fps:           float = 25.0
    px_per_m:      float = 100.0


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class DefendingMetrics:
    """
    Calculate defending metrics from a list of DefendingEvent objects.

    Usage::

        calculator = DefendingMetrics()
        metric_set = calculator.calculate(defending_events)
        print(metric_set.to_dict())
    """

    def calculate(self, events: List[DefendingEvent]) -> MetricSet:
        if not events:
            return self._empty()

        total_tackles      = sum(len(e.tackles)       for e in events)
        successful_tackles = sum(sum(1 for t in e.tackles if t.successful) for e in events)
        failed_tackles     = total_tackles - successful_tackles
        fouls              = sum(sum(1 for t in e.tackles if t.foul) for e in events)
        tackle_rate        = percentage(successful_tackles, total_tackles)

        interceptions      = sum(e.interceptions for e in events)
        clearances         = sum(e.clearances     for e in events)

        total_aerial       = sum(len(e.aerial_duels) for e in events)
        won_aerial         = sum(sum(1 for a in e.aerial_duels if a.won) for e in events)
        aerial_rate        = percentage(won_aerial, total_aerial)

        total_actions      = total_tackles + interceptions + clearances + total_aerial

        # Positioning score: how close the player stayed to ideal defensive position.
        pos_errors: List[float] = []
        for ev in events:
            n = min(len(ev.body_positions), len(ev.ideal_positions))
            for i in range(n):
                pos_errors.append(ev.body_positions[i].distance_to(ev.ideal_positions[i]))

        avg_pos_error    = sum(pos_errors) / len(pos_errors) if pos_errors else 0.0
        positioning_score = max(0.0, 100.0 - avg_pos_error * 500)

        # Recovery speed (average player movement speed).
        all_speeds: List[float] = []
        for ev in events:
            if len(ev.body_positions) >= 2:
                all_speeds.extend(speed_px_per_frame(ev.body_positions))

        avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0.0
        fps  = events[0].fps
        px_m = events[0].px_per_m

        # Total distance covered.
        total_dist = 0.0
        for ev in events:
            for i in range(1, len(ev.body_positions)):
                total_dist += ev.body_positions[i].distance_to(ev.body_positions[i - 1])
        dist_m = total_dist / px_m if px_m > 0 else total_dist

        ms = MetricSet(activity="defending")
        ms.metrics = [
            MetricValue("Total Defensive Actions", total_actions,      str(total_actions),                   ""),
            MetricValue("Successful Tackles",      successful_tackles, str(successful_tackles),              ""),
            MetricValue("Failed Tackles",          failed_tackles,     str(failed_tackles),                  ""),
            MetricValue("Tackle Success Rate",     tackle_rate,        fmt_pct(tackle_rate),                 "%"),
            MetricValue("Interceptions",           interceptions,      str(interceptions),                   ""),
            MetricValue("Clearances",              clearances,         str(clearances),                      ""),
            MetricValue("Aerial Duels Won",        won_aerial,         str(won_aerial),                      ""),
            MetricValue("Aerial Duel Win Rate",    aerial_rate,        fmt_pct(aerial_rate),                 "%"),
            MetricValue("Positioning Score",       positioning_score,  fmt_pct(positioning_score),           ""),
            MetricValue("Recovery Speed",          avg_speed,          fmt_speed_kmh(avg_speed, fps, px_m),  "km/h"),
            MetricValue("Distance Covered",        dist_m,             fmt_distance(dist_m),                 "m"),
            MetricValue("Fouls Committed",         fouls,              str(fouls),                           ""),
        ]
        return ms

    def _empty(self) -> MetricSet:
        ms = MetricSet(activity="defending")
        for label in [
            "Total Defensive Actions", "Successful Tackles", "Failed Tackles",
            "Tackle Success Rate", "Interceptions", "Clearances",
            "Aerial Duels Won", "Aerial Duel Win Rate", "Positioning Score",
            "Recovery Speed", "Distance Covered", "Fouls Committed",
        ]:
            ms.metrics.append(MetricValue(label, 0.0, "—"))
        return ms
