#!/usr/bin/env python3
"""
Session Summary Generator
=========================
Generates a structured session summary BEFORE the LLM runs.
FootballIQ already knows everything — this captures it cleanly.

Output:
  Detected Activities
    Passing
    Ball Control
  --------------------------------
  Strengths
    Accurate short passes
    Good balance
  --------------------------------
  Weaknesses
    Body posture
    Power control
  --------------------------------
  Player Level
    Developing ⭐⭐

Usage::

    generator = SessionSummaryGenerator()
    summary   = generator.generate(
        activities   = ["Passing", "Ball Control"],
        metrics      = {"accuracy": 86, "balance": 88, "body_alignment": "Poor"},
        skill_report = skill_report,
        mistakes     = mistake_result,
        level        = "Developing",
    )
    print(summary.to_text())
    print(summary.to_dict())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

_DIVIDER = "-" * 32

_LEVEL_STARS = {
    "Beginner":     "⭐",
    "Developing":   "⭐⭐",
    "Intermediate": "⭐⭐⭐",
    "Advanced":     "⭐⭐⭐⭐",
    "Elite":        "⭐⭐⭐⭐⭐",
}

# Metric → human-readable strength label when value is good
_STRENGTH_LABELS: Dict[str, str] = {
    "accuracy":             "Accurate passing",
    "balance":              "Good balance",
    "body_alignment":       "Good body position",
    "foot_placement":       "Good foot placement",
    "dribble_success_rate": "Effective dribbling",
    "shot_accuracy":        "Accurate shooting",
    "passing_consistency":  "Consistent passing",
    "movement_consistency": "Consistent movement",
    "knee_stability":       "Strong knee alignment",
    "touch_tightness":      "Tight ball control",
    "ball_control":         "Good ball control",
    "stride_length":        "Good stride length",
}

# Metric → human-readable weakness label when value is poor
_WEAKNESS_LABELS: Dict[str, str] = {
    "accuracy":             "Pass accuracy needs work",
    "balance":              "Balance needs improvement",
    "body_alignment":       "Body posture",
    "foot_placement":       "Foot placement",
    "dribble_success_rate": "Dribble success rate",
    "shot_accuracy":        "Shot accuracy",
    "passing_consistency":  "Passing consistency",
    "movement_consistency": "Movement consistency",
    "knee_stability":       "Knee stability",
    "touch_tightness":      "Ball too far from feet",
    "average_speed":        "Power control",
    "torso_lean":           "Body posture",
    "ball_control":         "Ball control",
}

# Thresholds for strength vs weakness classification
_GOOD_THRESHOLDS: Dict[str, tuple] = {
    # (good_value, higher_is_better)
    "accuracy":             (78.0, True),
    "balance":              (78.0, True),
    "body_alignment":       (75.0, True),
    "foot_placement":       (75.0, True),
    "dribble_success_rate": (72.0, True),
    "shot_accuracy":        (68.0, True),
    "passing_consistency":  (72.0, True),
    "movement_consistency": (72.0, True),
    "knee_stability":       (75.0, True),
    "touch_tightness":      (0.06, False),   # lower is better
    "average_speed":        (30.0, True),
    "torso_lean":           (12.0, False),   # lower is better
    "ball_control":         (75.0, True),
}

_POOR_THRESHOLDS: Dict[str, tuple] = {
    "accuracy":             (62.0, True),
    "balance":              (60.0, True),
    "body_alignment":       (58.0, True),   # "Poor" maps to ~35
    "foot_placement":       (58.0, True),
    "dribble_success_rate": (52.0, True),
    "shot_accuracy":        (48.0, True),
    "passing_consistency":  (55.0, True),
    "movement_consistency": (55.0, True),
    "knee_stability":       (60.0, True),
    "touch_tightness":      (0.12, False),
    "average_speed":        (45.0, True),   # too fast = poor control
    "torso_lean":           (20.0, False),
    "ball_control":         (58.0, True),
}

_ALIGN_SCORES = {"Good": 85.0, "Fair": 60.0, "Poor": 35.0}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    """Structured session summary — everything FootballIQ knows before the LLM."""
    activities:   List[str]
    strengths:    List[str]
    weaknesses:   List[str]
    level:        str
    stars:        str
    top_strength: Optional[str]
    top_weakness: Optional[str]
    mistake_causes: List[str]          = field(default_factory=list)
    top_drill:      Optional[str]      = None
    top_coach_tip:  Optional[str]      = None

    def to_text(self) -> str:
        """Plain text version — exactly as shown in the spec."""
        lines = []

        lines.append("Detected Activities")
        for a in self.activities:
            lines.append(f"  {a}")
        lines.append(_DIVIDER)

        if self.strengths:
            lines.append("Strengths")
            for s in self.strengths:
                lines.append(f"  {s}")
            lines.append(_DIVIDER)

        if self.weaknesses:
            lines.append("Weaknesses")
            for w in self.weaknesses:
                lines.append(f"  {w}")
            lines.append(_DIVIDER)

        if self.mistake_causes:
            lines.append("Possible Causes")
            for c in self.mistake_causes:
                lines.append(f"  ✓ {c}")
            lines.append(_DIVIDER)

        lines.append("Player Level")
        lines.append(f"  {self.level} {self.stars}")

        if self.top_drill:
            lines.append(_DIVIDER)
            lines.append("Priority Drill")
            lines.append(f"  {self.top_drill}")

        if self.top_coach_tip:
            lines.append(_DIVIDER)
            lines.append("Coach Tip")
            lines.append(f"  {self.top_coach_tip}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activities":    self.activities,
            "strengths":     self.strengths,
            "weaknesses":    self.weaknesses,
            "level":         self.level,
            "stars":         self.stars,
            "top_strength":  self.top_strength,
            "top_weakness":  self.top_weakness,
            "mistake_causes": self.mistake_causes,
            "top_drill":     self.top_drill,
            "top_coach_tip": self.top_coach_tip,
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SessionSummaryGenerator:
    """
    Generates a structured session summary from all pipeline outputs.
    Runs before the LLM — captures everything FootballIQ already knows.
    """

    def generate(
        self,
        activities:     List[str],
        metrics:        Dict[str, Any],
        level:          str,
        mistake_causes: Optional[List[str]] = None,
        top_drill:      Optional[str]       = None,
        top_coach_tip:  Optional[str]       = None,
        skill_strengths: Optional[List[str]] = None,
        skill_weaknesses: Optional[List[str]] = None,
    ) -> SessionSummary:
        """
        Generate a session summary.

        Parameters
        ----------
        activities     : list of detected action names
        metrics        : dict of metric_name → value
        level          : player skill level string
        mistake_causes : list of cause strings from MistakeDetector
        top_drill      : name of the priority drill
        top_coach_tip  : the most important coach tip
        skill_strengths : pre-computed strengths from SkillClassifier
        skill_weaknesses: pre-computed weaknesses from SkillClassifier

        Returns
        -------
        SessionSummary
        """
        # Use pre-computed strengths/weaknesses if available, else derive.
        strengths  = skill_strengths  or self._derive_strengths(metrics)
        weaknesses = skill_weaknesses or self._derive_weaknesses(metrics)

        stars       = _LEVEL_STARS.get(level, "⭐")
        top_str     = strengths[0]  if strengths  else None
        top_weak    = weaknesses[0] if weaknesses else None
        causes      = mistake_causes or []

        summary = SessionSummary(
            activities      = activities,
            strengths       = strengths,
            weaknesses      = weaknesses,
            level           = level,
            stars           = stars,
            top_strength    = top_str,
            top_weakness    = top_weak,
            mistake_causes  = causes,
            top_drill       = top_drill,
            top_coach_tip   = top_coach_tip,
        )

        log.debug(
            "SessionSummaryGenerator: level=%s  %d activities  %d strengths  %d weaknesses",
            level, len(activities), len(strengths), len(weaknesses),
        )

        return summary

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _derive_strengths(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify metrics that are performing well."""
        strengths = []
        for metric, value in metrics.items():
            if value is None:
                continue
            val = self._coerce(metric, value)
            thresh = _GOOD_THRESHOLDS.get(metric)
            if thresh is None:
                continue
            good_val, higher_is_better = thresh
            is_good = (val >= good_val) if higher_is_better else (val <= good_val)
            if is_good:
                label = _STRENGTH_LABELS.get(metric)
                if label:
                    strengths.append(label)
        return strengths

    def _derive_weaknesses(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify metrics that need improvement."""
        weaknesses = []
        for metric, value in metrics.items():
            if value is None:
                continue
            val = self._coerce(metric, value)
            thresh = _POOR_THRESHOLDS.get(metric)
            if thresh is None:
                continue
            poor_val, higher_is_better = thresh
            is_poor = (val <= poor_val) if higher_is_better else (val >= poor_val)
            if is_poor:
                label = _WEAKNESS_LABELS.get(metric)
                if label:
                    weaknesses.append(label)
        return weaknesses

    @staticmethod
    def _coerce(metric: str, value: Any) -> float:
        """Convert metric value to float, handling string labels."""
        if isinstance(value, str):
            return _ALIGN_SCORES.get(value, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
