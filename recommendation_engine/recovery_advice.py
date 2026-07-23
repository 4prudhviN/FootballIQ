#!/usr/bin/env python3
"""
Recovery Advisor
================
Provides targeted recovery recommendations based on session metrics
and detected physical stress patterns.

Recovery is the missing link between training and performance.
This module ensures the recommendation engine addresses it explicitly.

Example output:
  Your knee deviation is elevated — prioritise hip mobility today.
  High ground contact time detected — avoid high-impact training for 24 hours.
  Your gait asymmetry suggests right-leg fatigue — focus left-leg activation.

Usage::

    advisor = RecoveryAdvisor()
    advice  = advisor.advise(session_metrics, player_level="Intermediate")
    for item in advice.items:
        print(item.advice)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Recovery rule database
# ---------------------------------------------------------------------------

@dataclass
class RecoveryRule:
    """A single recovery recommendation rule."""
    metric:      str
    condition:   str    # "above_threshold" | "below_threshold"
    threshold:   float
    advice:      str
    priority:    int    # 1=highest
    duration_h:  int    # recommended recovery window in hours


_RECOVERY_RULES: List[RecoveryRule] = [
    RecoveryRule(
        metric     = "knee_stability",
        condition  = "below_threshold",
        threshold  = 70.0,
        advice     = "Knee stability score is low — perform 10 min of hip mobility before any training today.",
        priority   = 1,
        duration_h = 24,
    ),
    RecoveryRule(
        metric     = "gait_symmetry",
        condition  = "below_threshold",
        threshold  = 75.0,
        advice     = "Stride asymmetry detected — avoid sprint sessions for 24 hours. Focus on single-leg balance work.",
        priority   = 2,
        duration_h = 24,
    ),
    RecoveryRule(
        metric     = "torso_lean",
        condition  = "above_threshold",
        threshold  = 20.0,
        advice     = "High torso lean indicates core fatigue — include 10 min of core activation before your next session.",
        priority   = 3,
        duration_h = 12,
    ),
]

# Level-specific general recovery advice.
_GENERAL_RECOVERY: Dict[str, List[str]] = {
    "Beginner": [
        "Sleep 8–9 hours per night — most skill consolidation happens during sleep.",
        "Drink at least 2 litres of water on training days.",
        "Stretch your hip flexors and hamstrings for 5 minutes after every session.",
    ],
    "Intermediate": [
        "Use contrast showers (30 sec cold, 30 sec warm, repeat 3 times) after intense sessions.",
        "Consume 20–30 g of protein within 30 minutes of finishing training.",
        "Foam roll your quads and calves for 5–10 min on rest days.",
    ],
    "Advanced": [
        "Track heart rate variability (HRV) on waking — train hard only when HRV is above your 7-day average.",
        "Prioritise sleep consistency over duration — same bedtime every night.",
        "Include 15 min of active recovery (light jog or swim) the day after a hard session.",
    ],
}


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class RecoveryItem:
    """A single recovery recommendation."""
    metric:     str
    advice:     str
    priority:   int
    duration_h: int        # recovery window


@dataclass
class RecoveryAdvice:
    """Full recovery advice for a session."""
    player_level:  str
    items:         List[RecoveryItem]      = field(default_factory=list)
    general:       List[str]               = field(default_factory=list)
    rest_day_recommended: bool             = False
    rest_reason:   Optional[str]           = None

    def to_dict(self) -> dict:
        return {
            "playerLevel":         self.player_level,
            "restDayRecommended":  self.rest_day_recommended,
            "restReason":          self.rest_reason,
            "items": [
                {
                    "metric":    i.metric,
                    "advice":    i.advice,
                    "priority":  i.priority,
                    "durationH": i.duration_h,
                }
                for i in self.items
            ],
            "general": self.general,
        }

    def summary(self) -> str:
        lines = [f"Recovery Advice — {self.player_level}", "-" * 40]
        if self.rest_day_recommended:
            lines.append(f"  ⚠  Rest day recommended: {self.rest_reason}")
        for item in self.items:
            lines.append(f"  • {item.advice}")
        if self.general:
            lines.append("\n  General:")
            for g in self.general:
                lines.append(f"  · {g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------

class RecoveryAdvisor:
    """
    Generates recovery recommendations from session metrics.

    Parameters
    ----------
    high_priority_threshold : int
        Number of high-priority issues that triggers a rest-day recommendation.
    """

    def __init__(self, high_priority_threshold: int = 2) -> None:
        self.high_priority_threshold = high_priority_threshold

    def advise(
        self,
        metrics:      Dict[str, float],   # raw metric values (torso_lean, knee_stability, etc.)
        player_level: str = "Intermediate",
    ) -> RecoveryAdvice:
        """
        Generate recovery advice from session metrics.

        Parameters
        ----------
        metrics      : dict of {metric_name: float_value}
        player_level : str

        Returns
        -------
        RecoveryAdvice
        """
        triggered: List[RecoveryItem] = []

        for rule in sorted(_RECOVERY_RULES, key=lambda r: r.priority):
            value = metrics.get(rule.metric)
            if value is None:
                continue

            fired = (
                (rule.condition == "above_threshold" and value > rule.threshold) or
                (rule.condition == "below_threshold" and value < rule.threshold)
            )

            if fired:
                triggered.append(RecoveryItem(
                    metric     = rule.metric,
                    advice     = rule.advice,
                    priority   = rule.priority,
                    duration_h = rule.duration_h,
                ))

        # Recommend a rest day if multiple high-priority issues found.
        high_priority = [i for i in triggered if i.priority <= 2]
        rest_day = len(high_priority) >= self.high_priority_threshold
        rest_reason = (
            "Multiple biomechanical stress indicators detected. "
            "A rest day will allow your body to consolidate improvements."
        ) if rest_day else None

        general = _GENERAL_RECOVERY.get(player_level, _GENERAL_RECOVERY["Beginner"])

        advice = RecoveryAdvice(
            player_level          = player_level,
            items                 = triggered,
            general               = general,
            rest_day_recommended  = rest_day,
            rest_reason           = rest_reason,
        )

        log.debug("RecoveryAdvisor: %d recovery items  rest_day=%s", len(triggered), rest_day)
        return advice
