#!/usr/bin/env python3
"""
Priority Selector
=================
Selects the highest-impact focus areas for a player this week.

Example output:
  Focus this week:
    1. Passing accuracy
    2. First touch
    3. Balance while shooting

Responsibilities:
  - Rank gaps by impact weight × severity
  - Cap focus areas to avoid overwhelming the player
  - Apply level-specific rules (beginners get 1–2 areas max)
  - Return labelled FocusArea objects — never raw dicts

Usage::

    selector = PrioritySelector()
    areas    = selector.select(skill_profile, max_areas=3)
    for area in areas:
        print(area.label)   # "Passing Accuracy"
        print(area.reason)  # "Your passing accuracy dropped to 61%..."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from coach_engine.skill_classifier import CoachSkillProfile, MetricScore
from config.thresholds             import SKILL_WEIGHTS
from utils.logger                  import get_logger

log = get_logger(__name__)

# Maximum focus areas by level.
_MAX_AREAS: Dict[str, int] = {
    "Beginner":     2,
    "Intermediate": 3,
    "Advanced":     4,
}

# Human-readable labels and plain-English reasons per metric.
_METRIC_LABELS: Dict[str, str] = {
    "torso_lean":           "Balance while shooting",
    "knee_dev":             "Knee stability on plant foot",
    "gait_asymmetry":       "Even stride on both legs",
    "leg_speed":            "Explosive leg speed",
    "movement_consistency": "Consistent movement pattern",
    "pass_accuracy":        "Passing accuracy",
    "shot_accuracy":        "Shot accuracy",
    "dribble_success_rate": "Dribble success rate",
    "tackle_success_rate":  "Tackle timing",
}

_METRIC_REASONS: Dict[str, str] = {
    "torso_lean":           "Your upper body is leaning back at contact — this costs accuracy and power.",
    "knee_dev":             "Your knee collapses inward when you plant your foot — a key power-loss point.",
    "gait_asymmetry":       "Your left and right strides are uneven — one leg is doing more work.",
    "leg_speed":            "Your leg speed is below the target range — limits shot power and acceleration.",
    "movement_consistency": "Your movement pattern varies significantly across the session.",
    "pass_accuracy":        "Your passing accuracy is below the target for your level.",
    "shot_accuracy":        "Your shot accuracy is below the target for your level.",
    "dribble_success_rate": "You are losing the ball too frequently during dribbling.",
    "tackle_success_rate":  "Your tackle attempts are not converting at the expected rate.",
}


@dataclass
class FocusArea:
    """A single recommended focus area for the player."""
    rank:         int
    metric:       str
    label:        str          # e.g. "Passing Accuracy"
    reason:       str          # plain English explanation
    score:        float        # current score 0.0–1.0
    impact_score: float        # priority score (weight × gap)
    level:        str          # player level this recommendation is for


class PrioritySelector:
    """
    Selects highest-impact focus areas from a skill profile.

    Parameters
    ----------
    gap_threshold : float
        Only metrics with score ≤ this are considered gaps. Default 0.65.
    """

    def __init__(self, gap_threshold: float = 0.65) -> None:
        self.gap_threshold = gap_threshold

    def select(
        self,
        profile:   CoachSkillProfile,
        max_areas: Optional[int] = None,
    ) -> List[FocusArea]:
        """
        Return a prioritised list of focus areas for the player.

        Parameters
        ----------
        profile   : CoachSkillProfile — from CoachSkillClassifier
        max_areas : int | None — override the level-based cap

        Returns
        -------
        List[FocusArea] sorted by impact_score descending
        """
        level = profile.level
        cap   = max_areas or _MAX_AREAS.get(level, 3)

        # Score each metric by: weight × (1 - current_score) = impact of fixing it.
        candidates: List[tuple[float, MetricScore]] = []
        for ms in profile.metric_scores:
            if ms.score > self.gap_threshold:
                continue
            weight       = SKILL_WEIGHTS.get(ms.metric, 0.05)
            impact_score = weight * (1.0 - ms.score)
            candidates.append((impact_score, ms))

        # Sort highest impact first.
        candidates.sort(key=lambda x: x[0], reverse=True)

        areas: List[FocusArea] = []
        for rank, (impact, ms) in enumerate(candidates[:cap], start=1):
            label  = _METRIC_LABELS.get(ms.metric, ms.metric.replace("_", " ").title())
            reason = _METRIC_REASONS.get(ms.metric, f"Your {label.lower()} score is {ms.score:.0%}.")

            areas.append(FocusArea(
                rank         = rank,
                metric       = ms.metric,
                label        = label,
                reason       = reason,
                score        = round(ms.score, 3),
                impact_score = round(impact, 4),
                level        = level,
            ))

        log.debug("PrioritySelector: %d focus areas for %s player", len(areas), level)
        return areas

    def focus_summary(self, profile: CoachSkillProfile) -> Dict[str, List[str]]:
        """
        Return a simple {title: [label, label, ...]} dict for the dashboard
        "Focus this week" section.
        """
        areas  = self.select(profile)
        labels = [a.label for a in areas]
        return {"Focus this week": labels}
