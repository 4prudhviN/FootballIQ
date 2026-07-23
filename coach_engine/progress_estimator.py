#!/usr/bin/env python3
"""
Progress Estimator  (coach_engine)
====================================
Compares two sessions to estimate player improvement.
All calculations are deterministic — no LLM.

Responsibilities:
  - Accept two CoachSkillProfile objects (previous and current)
  - Calculate metric-level deltas
  - Determine whether the player improved, regressed, or stayed stable
  - Estimate time-to-next-level based on current improvement rate
  - Return a ProgressReport — never raw dicts

Usage::

    estimator = ProgressEstimator()
    report    = estimator.compare(previous_profile, current_profile)
    print(report.overall_trend)   # "improving" | "stable" | "regressing"
    print(report.summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from coach_engine.skill_classifier    import CoachSkillProfile
from coach_engine.terminology_adapter import TerminologyAdapter
from utils.logger                      import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMPROVEMENT_THRESHOLD = 0.03    # delta ≥ this → "improved"
_REGRESSION_THRESHOLD  = -0.03   # delta ≤ this → "regressed"

# Rough sessions-to-level based on average improvement rate per session.
# Assumes 2 sessions per week, improvement rate varies by level.
_SESSIONS_PER_LEVEL_UP: Dict[str, Dict[str, int]] = {
    "Beginner":     {"fast": 12, "normal": 20, "slow": 35},
    "Intermediate": {"fast": 18, "normal": 30, "slow": 50},
    "Advanced":     {"fast": 30, "normal": 50, "slow": 80},
}


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class MetricDelta:
    """Change in a single metric between two sessions."""
    metric:       str
    prev_score:   float
    curr_score:   float
    delta:        float          # positive = improved
    trend:        str            # "improved" | "regressed" | "stable"
    display:      str            # e.g. "+0.12 (improved)" or "-0.05 (regressed)"


@dataclass
class ProgressReport:
    """Full progress report comparing two sessions."""
    overall_trend:         str                  # "improving" | "stable" | "regressing"
    overall_delta:         float                # current_score - prev_score
    prev_level:            str
    curr_level:            str
    level_changed:         bool
    metric_deltas:         List[MetricDelta]    = field(default_factory=list)
    most_improved:         Optional[str]        = None
    most_regressed:        Optional[str]        = None
    sessions_to_next_level: Optional[int]       = None
    summary:               str                  = ""
    encouragement:         str                  = ""

    def to_dict(self) -> dict:
        return {
            "overall_trend":          self.overall_trend,
            "overall_delta":          round(self.overall_delta, 3),
            "prev_level":             self.prev_level,
            "curr_level":             self.curr_level,
            "level_changed":          self.level_changed,
            "most_improved":          self.most_improved,
            "most_regressed":         self.most_regressed,
            "sessions_to_next_level": self.sessions_to_next_level,
            "summary":                self.summary,
            "encouragement":          self.encouragement,
            "metric_deltas": [
                {
                    "metric":    d.metric,
                    "prev":      round(d.prev_score, 3),
                    "curr":      round(d.curr_score, 3),
                    "delta":     round(d.delta, 3),
                    "trend":     d.trend,
                }
                for d in self.metric_deltas
            ],
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class ProgressEstimator:
    """
    Compares two skill profiles and estimates progress.

    Parameters
    ----------
    adapter : TerminologyAdapter | None
    """

    def __init__(self, adapter: Optional[TerminologyAdapter] = None) -> None:
        self._adapter = adapter or TerminologyAdapter()

    def compare(
        self,
        previous: CoachSkillProfile,
        current:  CoachSkillProfile,
    ) -> ProgressReport:
        """
        Compare two sessions and return a ProgressReport.

        Parameters
        ----------
        previous : CoachSkillProfile — older session
        current  : CoachSkillProfile — newer session

        Returns
        -------
        ProgressReport
        """
        # Build lookup dicts by metric name.
        prev_scores = {ms.metric: ms.score for ms in previous.metric_scores}
        curr_scores = {ms.metric: ms.score for ms in current.metric_scores}

        all_metrics = sorted(set(prev_scores) | set(curr_scores))

        deltas: List[MetricDelta] = []

        for metric in all_metrics:
            prev_s = prev_scores.get(metric, 0.0)
            curr_s = curr_scores.get(metric, 0.0)
            delta  = curr_s - prev_s

            if delta >= _IMPROVEMENT_THRESHOLD:
                trend = "improved"
            elif delta <= _REGRESSION_THRESHOLD:
                trend = "regressed"
            else:
                trend = "stable"

            sign    = "+" if delta >= 0 else ""
            display = f"{sign}{delta:.2f} ({trend})"

            deltas.append(MetricDelta(
                metric     = metric,
                prev_score = round(prev_s, 3),
                curr_score = round(curr_s, 3),
                delta      = round(delta, 3),
                trend      = trend,
                display    = display,
            ))

        overall_delta = current.overall_score - previous.overall_score
        if overall_delta >= _IMPROVEMENT_THRESHOLD:
            overall_trend = "improving"
        elif overall_delta <= _REGRESSION_THRESHOLD:
            overall_trend = "regressing"
        else:
            overall_trend = "stable"

        level_changed = current.level != previous.level

        # Most improved / most regressed metrics.
        improved_deltas  = [d for d in deltas if d.trend == "improved"]
        regressed_deltas = [d for d in deltas if d.trend == "regressed"]

        most_improved = (
            max(improved_deltas, key=lambda d: d.delta).metric.replace("_", " ").title()
            if improved_deltas else None
        )
        most_regressed = (
            min(regressed_deltas, key=lambda d: d.delta).metric.replace("_", " ").title()
            if regressed_deltas else None
        )

        # Estimate sessions to next level.
        sessions_estimate = self._estimate_sessions(
            current.level,
            current.overall_score,
            overall_delta,
        )

        summary      = self._build_summary(overall_trend, overall_delta, level_changed, current.level)
        encouragement = self._encouragement(overall_trend, current.level, level_changed)

        report = ProgressReport(
            overall_trend          = overall_trend,
            overall_delta          = round(overall_delta, 3),
            prev_level             = previous.level,
            curr_level             = current.level,
            level_changed          = level_changed,
            metric_deltas          = deltas,
            most_improved          = most_improved,
            most_regressed         = most_regressed,
            sessions_to_next_level = sessions_estimate,
            summary                = summary,
            encouragement          = encouragement,
        )

        log.debug(
            "ProgressEstimator: %s → %s  delta=%.3f  trend=%s",
            previous.level, current.level, overall_delta, overall_trend,
        )
        return report

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_sessions(
        level:   str,
        score:   float,
        delta:   float,
    ) -> Optional[int]:
        """
        Rough estimate of sessions needed to reach the next level.
        Returns None for Advanced players (no higher level).
        """
        if level == "Advanced":
            return None

        # Next-level thresholds.
        next_threshold = 0.70 if level == "Intermediate" else 0.35
        gap = max(0.0, next_threshold - score)

        if gap <= 0:
            return 0

        if delta <= 0:
            return _SESSIONS_PER_LEVEL_UP.get(level, {}).get("slow", 50)

        sessions_at_rate = int(gap / delta)
        benchmarks = _SESSIONS_PER_LEVEL_UP.get(level, {"fast": 20, "normal": 30, "slow": 50})

        if sessions_at_rate <= benchmarks["fast"]:
            return sessions_at_rate
        if sessions_at_rate <= benchmarks["normal"]:
            return sessions_at_rate
        return min(sessions_at_rate, benchmarks["slow"])

    @staticmethod
    def _build_summary(
        trend:         str,
        delta:         float,
        level_changed: bool,
        level:         str,
    ) -> str:
        sign = "+" if delta >= 0 else ""
        if level_changed:
            return f"Outstanding progress — you have reached {level} level!"
        if trend == "improving":
            return f"Clear improvement this session ({sign}{delta:.2f} overall score)."
        if trend == "regressing":
            return (f"Performance dipped slightly ({delta:.2f}). "
                    "This is normal — focus on the priority drill.")
        return "Stable performance this session — consistent execution."

    @staticmethod
    def _encouragement(trend: str, level: str, level_changed: bool) -> str:
        if level_changed:
            return (f"You just reached {level}! "
                    "Your consistent effort is paying off — keep pushing.")
        messages = {
            "improving":  "You're on the right track. Keep showing up and the gains will compound.",
            "stable":     "Consistency is the foundation of skill. Stay focused and trust the process.",
            "regressing": "One tough session doesn't define your progress. "
                          "Rest, reset, and come back stronger.",
        }
        return messages.get(trend, "Keep working — improvement takes time.")
