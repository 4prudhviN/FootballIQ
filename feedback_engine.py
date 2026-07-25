#!/usr/bin/env python3
"""
Feedback Engine
===============
This is NOT AI. It looks up the coaching knowledge base.

Flow:
  Metric (e.g. "Passing Accuracy = 62%")
    ↓
  Rule lookup (which coaching rule applies?)
    ↓
  Root cause (e.g. "Body Leaning Back")
    ↓
  Correction (plain English fix)
    ↓
  Drill (specific exercise from knowledge base)

FootballIQ already knows the coaching points.
The LLM only rewrites the language afterward.

Usage::

    engine = FeedbackEngine()
    report = engine.generate(FeedbackRequest(
        metrics  = {"torso_lean": 22.0, "pass_accuracy": 62.0},
        activity = "passing",
        level    = "Developing",
    ))
    for item in report.items:
        print(item.metric, "→", item.root_cause, "→", item.drill)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

_KB_DIR = Path(__file__).parent / "football_knowledge"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRequest:
    metrics:  Dict[str, Any]  = field(default_factory=dict)
    activity: str             = "general"
    level:    str             = "Beginner"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class FeedbackItem:
    """One coaching point derived from a metric value."""
    metric:      str        # e.g. "pass_accuracy"
    value:       Any        # e.g. 62.0
    severity:    str        # "poor" | "fair" | "good"
    root_cause:  str        # e.g. "Body leaning back at contact"
    observation: str        # plain English — what was observed
    correction:  str        # what to do to fix it
    drill:       str        # specific drill from knowledge base
    coach_tip:   str        # one-line cue


@dataclass
class FeedbackReport:
    activity:         str
    level:            str
    summary:          str
    positive:         List[str]
    items:            List[FeedbackItem]
    priority_drill:   Optional[str]
    motivational_tip: str

    def to_dict(self) -> dict:
        return {
            "activity":        self.activity,
            "level":           self.level,
            "summary":         self.summary,
            "positive":        self.positive,
            "motivationalTip": self.motivational_tip,
            "priorityDrill":   self.priority_drill,
            "issues": [
                {
                    "metric":      i.metric,
                    "value":       i.value,
                    "severity":    i.severity,
                    "rootCause":   i.root_cause,
                    "observation": i.observation,
                    "correction":  i.correction,
                    "drill":       i.drill,
                    "coachTip":    i.coach_tip,
                }
                for i in self.items
            ],
        }


# ---------------------------------------------------------------------------
# Knowledge base — rule tables
# ---------------------------------------------------------------------------
# Each rule:  metric → severity → (root_cause, observation, correction, drill, coach_tip)
# Severity determined by threshold comparison.

_RULES: Dict[str, Dict[str, tuple]] = {

    "torso_lean": {
        "poor": (
            "Body leaning back at contact",
            "Your upper body is leaning behind your hips when you strike the ball.",
            "Move your plant foot closer to the ball and drive your chest forward.",
            "Wall Lean Drill: stand 30 cm from a wall, drive knee up without back touching. 3×15 reps.",
            "Chin down, chest over the ball at contact.",
        ),
        "fair": (
            "Slight backward lean at contact",
            "Your torso angle at contact is slightly off — costing you accuracy.",
            "Record yourself at 0.5× speed and pause at contact to check your angle.",
            "Mirror drill: self-record and pause at contact frame. Adjust and repeat 20 times.",
            "Plant foot beside the ball, not behind it.",
        ),
    },

    "pass_accuracy": {
        "poor": (
            "Inaccurate pass direction",
            "Your passing accuracy is low — the ball is not reaching its target consistently.",
            "Focus on your plant foot position and follow through toward the target.",
            "Wall rebounder: pass at a 50cm target on a wall from 5m. 3×20 each foot.",
            "Point your plant foot at the target before striking.",
        ),
        "fair": (
            "Inconsistent pass direction",
            "Your passing accuracy drops under pressure — technique is breaking down.",
            "Practise under light pressure — add a defender after achieving 80% accuracy alone.",
            "Triangle passing circuit: 3 cones, 8m apart, one-touch rhythm. 5 min each direction.",
            "Decide where to pass before the ball arrives.",
        ),
    },

    "knee_dev": {
        "poor": (
            "Knee collapsing inward on plant foot",
            "Your knee falls inward when you plant your foot — reducing power and risking injury.",
            "Strengthen hip abductors. Push knee outward over your middle toe on every step.",
            "Lateral band walk: resistance band above knees, 20 steps each direction. 3 sets daily.",
            "Knee over second toe at all times when weight-bearing.",
        ),
        "fair": (
            "Mild medial knee drift",
            "Slight inward knee movement detected — not dangerous yet but worth correcting now.",
            "Single-leg squats: 3×8 each leg, focusing on knee tracking over middle toe.",
            "Single-leg squat progression: 3×8 each side, track knee position throughout.",
            "Activate glutes with clamshells before every session.",
        ),
    },

    "gait_asymmetry": {
        "poor": (
            "Significant stride imbalance",
            "Your left and right strides are noticeably uneven — one leg is doing more work.",
            "Unilateral strength work to equalise both sides.",
            "Single-leg bounds: hop 20m on one leg then switch. 5 sets each side.",
            "Count your steps — aim for equal rhythm on both sides.",
        ),
        "fair": (
            "Slight stride asymmetry",
            "Small imbalance in stride length detected — adds up over 90 minutes.",
            "Single-leg hop and hold: 3×10 each side, compare steadiness between sides.",
            "High-knee running drill: 2×20m focusing on equal knee height each side.",
            "Equal arm swing produces equal stride length.",
        ),
    },

    "balance": {
        "poor": (
            "Poor dynamic balance",
            "Your centre of gravity shifts significantly during movement — unstable base.",
            "Core stability and single-leg balance work before every session.",
            "Single-leg standing: eyes closed, 30 sec each leg. Progress to unstable surface.",
            "Tight core = stable base. Brace your abs during all movements.",
        ),
        "fair": (
            "Moderate balance inconsistency",
            "Balance wavers under dynamic conditions — technique breaks down at speed.",
            "Add balance challenges to existing drills — receive a pass on one leg.",
            "Balance board: 3×1 min each leg while performing light ball control.",
            "Stay on the balls of your feet — never flat-footed.",
        ),
    },

    "shot_accuracy": {
        "poor": (
            "Poor shot placement",
            "Shots are not hitting the target consistently — technique at contact is off.",
            "Slow down. Practise placement at 50% pace before adding power.",
            "Corner target drill: 30 shots at cone targets in goal corners. Count hits.",
            "Inside foot for placement — strike through the side of the ball.",
        ),
        "fair": (
            "Inconsistent shot placement",
            "Shot accuracy drops when adding pace — technique breaks under pressure.",
            "Practise shooting after receiving a pass — first-time technique.",
            "One-two and shoot: receive pass, first-time shot at target. 15 reps each foot.",
            "Lock your ankle at contact for consistent direction.",
        ),
    },

    "dribble_success_rate": {
        "poor": (
            "Ball lost too frequently while dribbling",
            "You are losing the ball easily — control is too loose or head is down.",
            "Keep the ball within one touch of your feet at all times. Look up regularly.",
            "Cone slalom: 8 cones 1m apart, dribble through alternate feet. 10 runs.",
            "Small touches at pace. Never push the ball more than 1m ahead.",
        ),
        "fair": (
            "Occasional ball loss in tight situations",
            "Dribble success drops when pressed — need sharper change of direction.",
            "Practise feints before accelerating — commit to your move.",
            "1v1 box drill: 5m×5m, attacker must beat defender to far cone. 2-min rounds.",
            "Use your body — feint before accelerating.",
        ),
    },

    "movement_consistency": {
        "poor": (
            "Highly variable movement pattern",
            "Your movement mechanics change significantly across the session — inconsistent base.",
            "Slow repetition: practise at 50% speed until automatic, then increase pace.",
            "Repetition block: same skill 50 times at 70% intensity. Identical mechanics each rep.",
            "Slow is smooth, smooth is fast.",
        ),
        "fair": (
            "Movement consistency drops under fatigue",
            "Technique holds up early but degrades later in the session.",
            "End-of-session skill work — practise the target skill when already tired.",
            "Fatigue-state training: perform target skill in final 10 min of hard session.",
            "Video yourself at rep 1 and rep 30 — the difference shows where consistency breaks.",
        ),
    },

    "foot_placement": {
        "poor": (
            "Plant foot consistently in wrong position",
            "Your supporting foot is behind or too far from the ball — limiting power and accuracy.",
            "Practise slow-motion approach focusing on plant foot landing beside the ball.",
            "Plant foot marker drill: place a cone where plant foot should land. Approach and strike 20 times.",
            "Plant foot beside the ball — never behind it.",
        ),
        "fair": (
            "Plant foot position inconsistent",
            "Foot placement varies between attempts — causing unpredictable results.",
            "Mark the exact foot position before each attempt until it becomes automatic.",
            "Spot kicking: mark plant foot position with tape. 30 strikes checking position each time.",
            "Look at your plant foot position before every strike during practice.",
        ),
    },
}

# Severity thresholds per metric: (poor_threshold, fair_threshold, higher_is_worse)
_SEVERITY_THRESHOLDS: Dict[str, tuple] = {
    "torso_lean":           (20.0, 8.0,   True),
    "pass_accuracy":        (60.0, 75.0,  False),
    "knee_dev":             (0.30, 0.15,  True),
    "gait_asymmetry":       (0.20, 0.08,  True),
    "balance":              (60.0, 75.0,  False),
    "shot_accuracy":        (45.0, 65.0,  False),
    "dribble_success_rate": (50.0, 70.0,  False),
    "movement_consistency": (15.0, 5.0,   True),
    "foot_placement":       (55.0, 72.0,  False),
}

# Positive observations when metric is good
_POSITIVE: Dict[str, str] = {
    "torso_lean":           "Good body position at contact — torso well balanced.",
    "pass_accuracy":        "Excellent passing accuracy.",
    "knee_dev":             "Strong knee alignment throughout movement.",
    "gait_asymmetry":       "Symmetric stride — both legs contributing equally.",
    "balance":              "Good dynamic balance maintained.",
    "shot_accuracy":        "Accurate shot placement.",
    "dribble_success_rate": "High ball retention rate while dribbling.",
    "movement_consistency": "Consistent movement pattern throughout session.",
    "foot_placement":       "Good plant foot positioning.",
}

_MOTIVATIONAL: Dict[str, str] = {
    "Beginner":     "Every elite player started exactly here. One drill at a time — small gains compound fast.",
    "Developing":   "You are building the right foundations. Stay consistent and the level-up will come.",
    "Intermediate": "Solid base. The gap to Advanced is consistency — bring this focus every session.",
    "Advanced":     "Marginal gains are your edge. These drills sharpen the details that separate good from great.",
    "Elite":        "At this level, 1% improvements matter. Use this session data to find your edge.",
}


# ---------------------------------------------------------------------------
# Knowledge base loader
# ---------------------------------------------------------------------------

def _load_kb(activity: str) -> dict:
    """Load activity-specific knowledge base overrides."""
    path = _KB_DIR / "activities" / f"{activity}.json"
    if not path.exists():
        path = _KB_DIR / f"{activity}.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_mistakes_kb() -> dict:
    """Load common mistakes knowledge base."""
    path = _KB_DIR / "mistakes" / "common_mistakes.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {m["id"]: m for m in data.get("mistakes", [])}
    except Exception:
        return {}


def _load_drill(drill_ref: str, activity: str) -> Optional[str]:
    """Look up a drill from the drills knowledge base."""
    path = _KB_DIR / "drills" / f"{activity}_drills.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("drills", []):
            if d.get("id") == drill_ref:
                return d.get("instructions", "")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Feedback Engine
# ---------------------------------------------------------------------------

class FeedbackEngine:
    """
    Pure knowledge-base feedback engine. No AI, no LLM.

    Flow per metric:
      metric value → severity check → rule lookup → correction + drill
    """

    def generate(self, request: FeedbackRequest) -> FeedbackReport:
        level    = request.level
        activity = request.activity
        metrics  = request.metrics

        kb       = _load_kb(activity)
        mistakes = _load_mistakes_kb()

        items:    List[FeedbackItem] = []
        positive: List[str]          = []

        for metric, value in metrics.items():
            if value is None:
                continue

            severity = self._severity(metric, float(value))

            if severity == "good":
                obs = _POSITIVE.get(metric)
                if obs:
                    positive.append(obs)
                continue

            rule = _RULES.get(metric, {}).get(severity)
            if not rule:
                continue

            root_cause, observation, correction, drill, coach_tip = rule

            # Override with knowledge base if available.
            kb_entry = kb.get(metric, {}).get(level, {})
            if kb_entry:
                observation = kb_entry.get("observation", observation)
                drill       = kb_entry.get("drill",       drill)
                coach_tip   = kb_entry.get("coach_tip",   coach_tip)

            items.append(FeedbackItem(
                metric      = metric,
                value       = value,
                severity    = severity,
                root_cause  = root_cause,
                observation = observation,
                correction  = correction,
                drill       = drill,
                coach_tip   = coach_tip,
            ))

        summary          = self._summary(activity, level, items, positive)
        priority_drill   = items[0].drill if items else None
        motivational_tip = _MOTIVATIONAL.get(level, _MOTIVATIONAL["Beginner"])

        return FeedbackReport(
            activity         = activity,
            level            = level,
            summary          = summary,
            positive         = positive,
            items            = items,
            priority_drill   = priority_drill,
            motivational_tip = motivational_tip,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _severity(metric: str, value: float) -> str:
        thresh = _SEVERITY_THRESHOLDS.get(metric)
        if thresh is None:
            return "good"
        poor_t, fair_t, higher_is_worse = thresh
        if higher_is_worse:
            if value >= poor_t: return "poor"
            if value >= fair_t: return "fair"
        else:
            if value <= poor_t: return "poor"
            if value <= fair_t: return "fair"
        return "good"

    @staticmethod
    def _summary(activity: str, level: str, items: List[FeedbackItem], positive: List[str]) -> str:
        label = activity.capitalize()
        n     = len(items)
        g     = len(positive)
        if n == 0:
            return f"{label} session complete — all metrics within {level} range."
        if n == 1:
            return (f"{label} session complete for {level} player. "
                    f"One area needs attention: {items[0].root_cause}.")
        return (f"{label} session complete for {level} player. "
                f"{n} coaching points identified; {g} metric(s) performing well.")
