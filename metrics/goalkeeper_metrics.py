#!/usr/bin/env python3
"""
Goalkeeper Metrics
==================
Calculates pure numerical metrics for goalkeeping actions.
No AI, no feedback — numbers only.

Metrics produced:
  - Total Saves              (count)
  - Goals Conceded           (count)
  - Save Percentage          (%)
  - Reaction Time            (seconds)
  - Average Diving Range     (m)
  - Max Diving Range         (m)
  - Distribution Accuracy    (%)
  - Successful Punches       (count)
  - Claiming Success Rate    (%)
  - Positioning Score        (0–100)
  - Sweeper Actions          (count)
  - High Claim Success Rate  (%)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from metrics.common_metrics import (
    MetricSet, MetricValue, Point2D,
    percentage, fmt_pct, fmt_distance, fmt_time,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class SaveEvent:
    """Data for a single save attempt."""
    saved:          bool            # True if the shot was stopped
    reaction_time_s: float          # seconds from shot to GK movement
    gk_start:       Point2D        # GK position when shot was taken
    gk_end:         Point2D        # GK final position at save attempt
    px_per_m:       float = 100.0


@dataclass
class DistributionEvent:
    """Data for a single distribution (throw/kick)."""
    successful: bool               # True if reached a teammate


@dataclass
class ClaimEvent:
    """Data for an attempt to claim a cross or aerial ball."""
    successful: bool
    punch:      bool               # True if punched rather than caught


@dataclass
class GoalkeeperEvent:
    """Aggregate data for a goalkeeping session."""
    saves:             List[SaveEvent]
    goals_conceded:    int
    distributions:     List[DistributionEvent]
    claims:            List[ClaimEvent]
    sweeper_actions:   int                        # times GK came off line
    body_positions:    List[Point2D]
    ideal_positions:   List[Point2D]
    fps:               float = 25.0
    px_per_m:          float = 100.0


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class GoalkeeperMetrics:
    """
    Calculate goalkeeper metrics from a list of GoalkeeperEvent objects.

    Usage::

        calculator = GoalkeeperMetrics()
        metric_set = calculator.calculate(gk_events)
        print(metric_set.to_dict())
    """

    def calculate(self, events: List[GoalkeeperEvent]) -> MetricSet:
        if not events:
            return self._empty()

        total_saves     = sum(len(e.saves) for e in events)
        successful_saves = sum(sum(1 for s in e.saves if s.saved) for e in events)
        goals_conceded  = sum(e.goals_conceded for e in events)
        save_pct        = percentage(successful_saves, total_saves + goals_conceded)

        reaction_times  = [s.reaction_time_s for e in events for s in e.saves if s.reaction_time_s > 0]
        avg_reaction    = sum(reaction_times) / len(reaction_times) if reaction_times else 0.0

        px_m = events[0].px_per_m

        diving_ranges   = [
            s.gk_start.distance_to(s.gk_end) / px_m
            for e in events for s in e.saves
        ]
        avg_range = sum(diving_ranges) / len(diving_ranges) if diving_ranges else 0.0
        max_range = max(diving_ranges) if diving_ranges else 0.0

        total_dist      = sum(len(e.distributions) for e in events)
        successful_dist = sum(sum(1 for d in e.distributions if d.successful) for e in events)
        dist_accuracy   = percentage(successful_dist, total_dist)

        total_claims    = sum(len(e.claims) for e in events)
        successful_claims = sum(sum(1 for c in e.claims if c.successful) for e in events)
        claim_rate      = percentage(successful_claims, total_claims)

        punches         = sum(sum(1 for c in e.claims if c.punch) for e in events)

        sweeper         = sum(e.sweeper_actions for e in events)

        # Positioning score.
        pos_errors: List[float] = []
        for ev in events:
            n = min(len(ev.body_positions), len(ev.ideal_positions))
            for i in range(n):
                pos_errors.append(ev.body_positions[i].distance_to(ev.ideal_positions[i]))
        avg_pos_error     = sum(pos_errors) / len(pos_errors) if pos_errors else 0.0
        positioning_score = max(0.0, 100.0 - avg_pos_error * 500)

        # High claim success rate (claims that were not punches).
        high_claims     = sum(sum(1 for c in e.claims if c.successful and not c.punch) for e in events)
        high_claim_rate = percentage(high_claims, total_claims)

        ms = MetricSet(activity="goalkeeping")
        ms.metrics = [
            MetricValue("Total Saves",            successful_saves, str(successful_saves),           ""),
            MetricValue("Goals Conceded",         goals_conceded,   str(goals_conceded),             ""),
            MetricValue("Save Percentage",        save_pct,         fmt_pct(save_pct),               "%"),
            MetricValue("Reaction Time",          avg_reaction,     fmt_time(avg_reaction),          "s"),
            MetricValue("Average Diving Range",   avg_range,        fmt_distance(avg_range),         "m"),
            MetricValue("Max Diving Range",       max_range,        fmt_distance(max_range),         "m"),
            MetricValue("Distribution Accuracy",  dist_accuracy,    fmt_pct(dist_accuracy),          "%"),
            MetricValue("Successful Punches",     punches,          str(punches),                    ""),
            MetricValue("Claiming Success Rate",  claim_rate,       fmt_pct(claim_rate),             "%"),
            MetricValue("Positioning Score",      positioning_score,fmt_pct(positioning_score),      ""),
            MetricValue("Sweeper Actions",        sweeper,          str(sweeper),                    ""),
            MetricValue("High Claim Success Rate",high_claim_rate,  fmt_pct(high_claim_rate),        "%"),
        ]
        return ms

    def _empty(self) -> MetricSet:
        ms = MetricSet(activity="goalkeeping")
        for label in [
            "Total Saves", "Goals Conceded", "Save Percentage",
            "Reaction Time", "Average Diving Range", "Max Diving Range",
            "Distribution Accuracy", "Successful Punches", "Claiming Success Rate",
            "Positioning Score", "Sweeper Actions", "High Claim Success Rate",
        ]:
            ms.metrics.append(MetricValue(label, 0.0, "—"))
        return ms
