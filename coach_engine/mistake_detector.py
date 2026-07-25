#!/usr/bin/env python3
"""
Mistake Detector
================
Identifies CAUSES of poor performance, not just statistics.

Instead of:
  "Passing Accuracy = 61%"

It outputs:
  Possible Problems:
    ✓ Leaning backwards at contact
    ✓ Plant foot too far from the ball
    ✓ Ball struck too hard — losing direction
    ✓ Incorrect foot surface used

This module reads metrics + pose data and traces poor statistics
back to their root causes using the football_knowledge/mistakes/ database.

No AI. Pure rule-based cause analysis.

Usage::

    detector = MistakeDetector()
    result   = detector.detect(
        activity = "passing",
        metrics  = {"accuracy": 61, "body_alignment": "Poor", "average_speed": 45},
        level    = "Intermediate",
    )
    for problem in result.problems:
        print("✓", problem.cause)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

_KB_DIR = Path(__file__).parent.parent / "football_knowledge" / "mistakes"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class DetectedProblem:
    """A single identified cause of poor performance."""
    cause:        str          # e.g. "Leaning backwards at contact"
    evidence:     str          # what metric triggered this
    metric:       str          # metric name
    metric_value: Any          # actual value
    severity:     str          # "high" | "medium" | "low"
    correction:   str          # what to do to fix it
    drill_ref:    Optional[str] = None  # drill ID from knowledge base


@dataclass
class MistakeDetectionResult:
    """Full output of the MistakeDetector."""
    activity:  str
    level:     str
    problems:  List[DetectedProblem]

    def to_dict(self) -> dict:
        return {
            "activity": self.activity,
            "level":    self.level,
            "problems": [
                {
                    "cause":       p.cause,
                    "evidence":    p.evidence,
                    "metric":      p.metric,
                    "severity":    p.severity,
                    "correction":  p.correction,
                    "drill_ref":   p.drill_ref,
                }
                for p in self.problems
            ],
        }

    def summary_lines(self) -> List[str]:
        """Return formatted lines for display."""
        return [f"  ✓ {p.cause}" for p in self.problems]


# ---------------------------------------------------------------------------
# Built-in cause rules
# ---------------------------------------------------------------------------
# Each rule maps a metric + threshold to one or more possible causes.
# Rules are checked in order — first matching rules are reported.

@dataclass
class CauseRule:
    metric:       str
    condition:    str    # "below" | "above" | "equals"
    threshold:    Any
    cause:        str
    evidence_tpl: str    # e.g. "{metric} was {value}% — below the {threshold}% target"
    severity:     str
    correction:   str
    drill_ref:    Optional[str] = None


_CAUSE_RULES: List[CauseRule] = [

    # ── Passing ─────────────────────────────────────────────────────────────

    CauseRule(
        metric     = "accuracy",
        condition  = "below",
        threshold  = 65.0,
        cause      = "Leaning backwards at contact",
        evidence_tpl = "Passing accuracy {value}% — below 65% target suggests body lean issue",
        severity   = "high",
        correction = "Plant foot beside the ball, not behind it. Drive chest forward at contact.",
        drill_ref  = "beg_passing_wall",
    ),
    CauseRule(
        metric     = "accuracy",
        condition  = "below",
        threshold  = 65.0,
        cause      = "Plant foot too far from the ball",
        evidence_tpl = "Low accuracy ({value}%) — plant foot positioning likely off",
        severity   = "high",
        correction = "Place your plant foot directly beside the ball — not behind or in front.",
        drill_ref  = "beg_passing_wall",
    ),
    CauseRule(
        metric     = "average_speed",
        condition  = "above",
        threshold  = 40.0,
        cause      = "Ball struck too hard — losing direction",
        evidence_tpl = "Average pass speed {value} px/f — excessive force reducing accuracy",
        severity   = "medium",
        correction = "Reduce power on short and medium passes. Match speed to distance.",
        drill_ref  = "int_passing_triangle",
    ),
    CauseRule(
        metric     = "body_alignment",
        condition  = "equals",
        threshold  = "Poor",
        cause      = "Leaning backwards at contact",
        evidence_tpl = "Body alignment rated Poor — torso behind hips at ball contact",
        severity   = "high",
        correction = "Chin down, chest over the ball. Plant foot beside the ball.",
        drill_ref  = "beg_passing_wall",
    ),
    CauseRule(
        metric     = "body_alignment",
        condition  = "equals",
        threshold  = "Poor",
        cause      = "Incorrect foot surface used",
        evidence_tpl = "Poor body alignment often accompanies incorrect contact surface",
        severity   = "medium",
        correction = "Use inside of foot for accuracy. Laces only for power over distance.",
        drill_ref  = "beg_passing_wall",
    ),
    CauseRule(
        metric     = "passing_consistency",
        condition  = "below",
        threshold  = 60.0,
        cause      = "No pre-scan before receiving",
        evidence_tpl = "Low consistency ({value}/100) — decisions being made too late",
        severity   = "medium",
        correction = "Look up twice before the ball arrives. Decide your pass before contact.",
        drill_ref  = "adv_scanning_rondo",
    ),
    CauseRule(
        metric     = "failed_passes",
        condition  = "above",
        threshold  = 8.0,
        cause      = "Poor weight of pass",
        evidence_tpl = "{value} failed passes — consistent mis-weighting of passes",
        severity   = "medium",
        correction = "Match your pass speed to the distance and movement of the receiver.",
        drill_ref  = "int_passing_triangle",
    ),

    # ── Shooting ─────────────────────────────────────────────────────────────

    CauseRule(
        metric     = "shot_accuracy",
        condition  = "below",
        threshold  = 50.0,
        cause      = "Leaning backwards at contact",
        evidence_tpl = "Shot accuracy {value}% — backward lean sends ball over target",
        severity   = "high",
        correction = "Plant foot beside ball. Drive chest forward at contact.",
        drill_ref  = "beg_shooting_stationary",
    ),
    CauseRule(
        metric     = "shot_accuracy",
        condition  = "below",
        threshold  = 50.0,
        cause      = "Looking up before contact",
        evidence_tpl = "Low shot accuracy ({value}%) — head likely rising before contact",
        severity   = "high",
        correction = "Keep eyes on the ball until after your foot makes contact.",
        drill_ref  = "adv_shooting_pressure",
    ),
    CauseRule(
        metric     = "shot_accuracy",
        condition  = "below",
        threshold  = 60.0,
        cause      = "Incorrect foot surface used",
        evidence_tpl = "Shot accuracy {value}% — toe-poke or wrong surface suspected",
        severity   = "medium",
        correction = "Use laces for power, inside foot for placement. Never toe-poke.",
        drill_ref  = "beg_shooting_stationary",
    ),
    CauseRule(
        metric     = "average_speed",
        condition  = "below",
        threshold  = 15.0,
        cause      = "No follow-through after contact",
        evidence_tpl = "Low shot speed ({value} px/f) — leg stopping at contact",
        severity   = "medium",
        correction = "Drive through the ball. Foot should finish pointing at target.",
        drill_ref  = "int_shooting_moving",
    ),

    # ── Dribbling ─────────────────────────────────────────────────────────────

    CauseRule(
        metric     = "dribble_success_rate",
        condition  = "below",
        threshold  = 55.0,
        cause      = "Ball too far from feet",
        evidence_tpl = "Dribble success {value}% — ball likely pushed too far ahead",
        severity   = "high",
        correction = "Keep ball within one foot-length at all times. Smaller touches.",
        drill_ref  = "beg_dribbling_cones",
    ),
    CauseRule(
        metric     = "dribble_success_rate",
        condition  = "below",
        threshold  = 55.0,
        cause      = "Head down — no awareness of defenders",
        evidence_tpl = "Low dribble success ({value}%) — likely not reading defensive positions",
        severity   = "high",
        correction = "Force yourself to look up every 2–3 touches. Feel the ball, don't watch it.",
        drill_ref  = "int_dribbling_1v1",
    ),
    CauseRule(
        metric     = "touch_tightness",
        condition  = "above",
        threshold  = 0.12,
        cause      = "Ball too far from feet",
        evidence_tpl = "Touch tightness {value} — ball straying beyond safe control range",
        severity   = "high",
        correction = "Deliberately take shorter touches. Quality over speed.",
        drill_ref  = "beg_dribbling_cones",
    ),

    # ── Movement / Balance ────────────────────────────────────────────────────

    CauseRule(
        metric     = "balance",
        condition  = "below",
        threshold  = 60.0,
        cause      = "Unstable centre of gravity during movement",
        evidence_tpl = "Balance score {value}/100 — excessive lateral sway detected",
        severity   = "medium",
        correction = "Core stability work before every session. Stay on balls of feet.",
        drill_ref  = "beg_balance_single_leg",
    ),
    CauseRule(
        metric     = "foot_placement",
        condition  = "below",
        threshold  = 60.0,
        cause      = "Plant foot consistently in wrong position",
        evidence_tpl = "Foot placement {value}/100 — ankles misaligned under hip centre",
        severity   = "medium",
        correction = "Plant foot beside the ball. Mark the exact position during practice.",
        drill_ref  = "beg_shooting_stationary",
    ),
    CauseRule(
        metric     = "torso_lean",
        condition  = "above",
        threshold  = 20.0,
        cause      = "Leaning backwards at contact",
        evidence_tpl = "Torso lean {value}° — excessive backward lean at ball contact",
        severity   = "high",
        correction = "Drive chest forward and down over the ball at contact.",
        drill_ref  = "beg_passing_wall",
    ),
    CauseRule(
        metric     = "knee_stability",
        condition  = "below",
        threshold  = 65.0,
        cause      = "Knee collapsing inward on plant foot",
        evidence_tpl = "Knee stability {value}/100 — valgus collapse reducing power transfer",
        severity   = "high",
        correction = "Strengthen hip abductors. Push knee over middle toe when planting.",
        drill_ref  = "int_movement_change_dir",
    ),
]


# ---------------------------------------------------------------------------
# Mistake Detector
# ---------------------------------------------------------------------------

class MistakeDetector:
    """
    Identifies root causes of poor performance from metrics.

    Reads the football_knowledge/mistakes/ database for additional
    activity-specific causes, then applies built-in cause rules.

    No AI. Pure cause analysis.

    Usage::

        detector = MistakeDetector()
        result   = detector.detect(
            activity = "passing",
            metrics  = {"accuracy": 61, "body_alignment": "Poor"},
            level    = "Intermediate",
        )
        for line in result.summary_lines():
            print(line)
    """

    def detect(
        self,
        activity: str,
        metrics:  Dict[str, Any],
        level:    str = "Beginner",
    ) -> MistakeDetectionResult:
        """
        Detect root causes from metrics.

        Parameters
        ----------
        activity : str — football action (e.g. "passing", "shooting")
        metrics  : dict — metric name → value
        level    : str — player skill level

        Returns
        -------
        MistakeDetectionResult
        """
        problems: List[DetectedProblem] = []
        seen_causes: set[str] = set()

        # ── Apply built-in cause rules ────────────────────────────────────────
        for rule in _CAUSE_RULES:
            value = metrics.get(rule.metric)
            if value is None:
                continue

            triggered = self._check(value, rule.condition, rule.threshold)
            if not triggered:
                continue

            # Avoid duplicate causes.
            if rule.cause in seen_causes:
                continue
            seen_causes.add(rule.cause)

            evidence = rule.evidence_tpl.replace("{value}", str(value)).replace(
                "{threshold}", str(rule.threshold)
            )
            problems.append(DetectedProblem(
                cause        = rule.cause,
                evidence     = evidence,
                metric       = rule.metric,
                metric_value = value,
                severity     = rule.severity,
                correction   = rule.correction,
                drill_ref    = rule.drill_ref,
            ))

        # ── Load activity-specific mistakes from knowledge base ───────────────
        kb_problems = self._load_kb_problems(activity, metrics)
        for p in kb_problems:
            if p.cause not in seen_causes:
                seen_causes.add(p.cause)
                problems.append(p)

        # Sort: high severity first, then medium, then low.
        severity_order = {"high": 0, "medium": 1, "low": 2}
        problems.sort(key=lambda p: severity_order.get(p.severity, 1))

        log.debug(
            "MistakeDetector: %d problems for %s (%s)",
            len(problems), activity, level,
        )

        return MistakeDetectionResult(
            activity = activity,
            level    = level,
            problems = problems,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _check(value: Any, condition: str, threshold: Any) -> bool:
        """Check if a metric value triggers a rule."""
        try:
            if condition == "below":
                return float(value) < float(threshold)
            if condition == "above":
                return float(value) > float(threshold)
            if condition == "equals":
                return str(value).strip().lower() == str(threshold).strip().lower()
        except (TypeError, ValueError):
            return str(value).strip().lower() == str(threshold).strip().lower()
        return False

    def _load_kb_problems(
        self, activity: str, metrics: Dict[str, Any]
    ) -> List[DetectedProblem]:
        """Load additional causes from football_knowledge/mistakes/{activity}.json."""
        path = _KB_DIR / f"{activity}.json"
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []

        problems: List[DetectedProblem] = []
        for mistake in data.get("mistakes", []):
            metric = mistake.get("metric", "")
            value  = metrics.get(metric)
            if value is None:
                continue

            threshold_str = mistake.get("threshold", "")
            triggered     = self._eval_kb_threshold(value, threshold_str)
            if not triggered:
                continue

            problems.append(DetectedProblem(
                cause        = mistake.get("name", "Unknown issue"),
                evidence     = mistake.get("plain_english", ""),
                metric       = metric,
                metric_value = value,
                severity     = mistake.get("severity", "medium"),
                correction   = mistake.get("correction", ""),
                drill_ref    = mistake.get("drill_ref"),
            ))

        return problems

    @staticmethod
    def _eval_kb_threshold(value: Any, threshold_str: str) -> bool:
        """
        Evaluate KB threshold strings like:
          "accuracy < 65%"  |  "body_alignment = Poor"  |  "touch_tightness > 0.12"
        """
        if not threshold_str:
            return False
        parts = threshold_str.replace("%", "").split()
        if len(parts) < 3:
            return False
        _, op, tval = parts[0], parts[1], " ".join(parts[2:])
        try:
            if op == "<":
                return float(str(value).replace("%", "")) < float(tval)
            if op == ">":
                return float(str(value).replace("%", "")) > float(tval)
            if op == "=":
                return str(value).strip().lower() == tval.strip().lower()
        except (ValueError, TypeError):
            return str(value).strip().lower() == tval.strip().lower()
        return False
