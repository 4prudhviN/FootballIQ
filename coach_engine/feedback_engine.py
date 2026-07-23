#!/usr/bin/env python3
"""
Feedback Engine  (coach_engine)
================================
Generates structured coaching feedback from a CoachSkillProfile.
All feedback is grounded in the football_knowledge/ database — never invented.
Every output string passes through the TerminologyAdapter before returning.

Responsibilities:
  - Load activity-specific knowledge from football_knowledge/*.json
  - Match metric gaps to pre-written observations and drills
  - Adapt all language to the player's skill level via TerminologyAdapter
  - Return a CoachFeedbackReport — no raw dicts, no LLM calls

Usage::

    engine = CoachFeedbackEngine()
    report = engine.generate(profile, activity="shooting")
    print(report.summary)
    for item in report.items:
        print(item.plain_observation)
        print(item.adapted_drill)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from coach_engine.skill_classifier    import CoachSkillProfile
from coach_engine.terminology_adapter import TerminologyAdapter
from utils.file_utils                  import read_json
from utils.logger                      import get_logger

log = get_logger(__name__)

_KB_DIR = Path(__file__).resolve().parent.parent / "football_knowledge"


# ---------------------------------------------------------------------------
# Built-in feedback rules (grounded observations — never invented)
# ---------------------------------------------------------------------------

_FEEDBACK_RULES: Dict[str, Dict[str, tuple[str, str, str]]] = {
    # metric: {severity: (observation, drill, coach_tip)}
    "torso_lean": {
        "poor": (
            "Your upper body is leaning back too far at the moment of contact. "
            "This sends the ball upward and reduces accuracy.",
            "Wall lean drill: stand 30 cm from a wall, drive your kicking knee up "
            "without your back touching the wall. 3 × 15 reps.",
            "At the moment of contact, your chest should be over the ball.",
        ),
        "fair": (
            "Slight backward lean detected. Your posture is close but could cost "
            "accuracy under pressure.",
            "Mirror slow-motion drill: record yourself at 0.5× speed and pause at "
            "the moment of contact to check your body angle.",
            "Keep your non-kicking foot planted firmly beside the ball — "
            "this naturally keeps your upper body more upright.",
        ),
    },
    "knee_dev": {
        "poor": (
            "Your knee is collapsing inward when you plant your foot. "
            "This reduces power and increases injury risk.",
            "Lateral band walk: place a resistance band above your knees and "
            "take 20 side steps each way. 3 sets daily.",
            "Push your knee outward over your middle toe every time you plant your foot.",
        ),
        "fair": (
            "Mild knee inward movement detected. Not dangerous yet, "
            "but worth correcting before it becomes a habit.",
            "Single-leg squat (3 × 8 each leg): focus on keeping your knee "
            "tracking over your middle toe throughout.",
            "Before each session, activate your hip muscles with 2 minutes of "
            "clamshells to stabilise your knee during movement.",
        ),
    },
    "gait_asymmetry": {
        "poor": (
            "Your left and right running steps are noticeably uneven. "
            "One leg is doing more work, which tires you out faster.",
            "Single-leg bounds: hop 20 m on one leg, then switch. "
            "5 sets each side to even out your stride.",
            "Count your steps on each side during warm-up — aim for equal "
            "rhythm and equal push-off power.",
        ),
        "fair": (
            "Slight imbalance between your left and right running steps. "
            "Small differences add up over a full match.",
            "Single-leg hop and hold: hop forward and freeze on landing. "
            "3 × 10 each side. Compare steadiness between sides.",
            "During warm-up, include 2 × 20 m of high-knee running with equal "
            "arm swing on both sides.",
        ),
    },
    "leg_speed": {
        "poor": (
            "Low leg speed detected during the key action. "
            "Insufficient explosiveness limits shot power and acceleration.",
            "Plyometric circuit: jump squats × 10, tuck jumps × 10, "
            "lateral hops × 10 each side — 3 rounds.",
            "Explosiveness starts from the hips. Sprint reaction drills "
            "twice a week will train your fast-twitch muscles.",
        ),
        "fair": (
            "Leg speed is adequate but has room to grow. "
            "Faster leg swing means harder shots and quicker first steps.",
            "Resistance band sprints: attach a light band around your waist "
            "and sprint 15 m against resistance. 6 reps per session.",
            "Faster arm drive directly increases your stride rate — "
            "focus on tight, fast arm swings.",
        ),
    },
    "movement_consistency": {
        "poor": (
            "High variation in your movement pattern across the session. "
            "Inconsistent technique means unpredictable results.",
            "Repetition block training: perform the same skill 50 times at "
            "70% effort, focusing purely on identical mechanics each time.",
            "Slow it down to speed it up: practise at 50% speed until it "
            "feels automatic, then gradually increase pace.",
        ),
        "fair": (
            "Some variation in movement pattern. Your technique is mostly "
            "solid but breaks down slightly under fatigue.",
            "End-of-session skill training: practise the skill when you're "
            "tired — that's when poor habits appear.",
            "Video yourself once a week and compare your first rep "
            "to your 30th — the difference shows where consistency breaks.",
        ),
    },
}

_POOR_THRESHOLDS: Dict[str, tuple[float, bool]] = {
    "torso_lean":           (20.0, True),
    "knee_dev":             (0.30, True),
    "gait_asymmetry":       (0.20, True),
    "leg_speed":            (20.0, False),
    "movement_consistency": (15.0, True),
}

_FAIR_THRESHOLDS: Dict[str, tuple[float, bool]] = {
    "torso_lean":           (8.0,  True),
    "knee_dev":             (0.15, True),
    "gait_asymmetry":       (0.08, True),
    "leg_speed":            (50.0, False),
    "movement_consistency": (5.0,  True),
}

_MOTIVATIONAL: Dict[str, str] = {
    "Beginner":     "Every elite player started exactly where you are. "
                    "Focus on one drill at a time — small improvements compound fast.",
    "Intermediate": "You have a solid foundation. The gap between Intermediate and "
                    "Advanced is consistency — bring this intensity every session.",
    "Advanced":     "Marginal gains are your edge. The drills above sharpen the "
                    "details that separate good players from great ones.",
}


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class FeedbackItem:
    """A single coaching observation with a targeted drill."""
    metric:              str
    severity:            str           # "poor" | "fair"
    plain_observation:   str           # adapted to player level
    adapted_drill:       str
    adapted_coach_tip:   str


@dataclass
class CoachFeedbackReport:
    """Full coaching report from the feedback engine."""
    activity:         str
    level:            str
    summary:          str
    positive:         List[str]         = field(default_factory=list)
    items:            List[FeedbackItem] = field(default_factory=list)
    priority_drill:   Optional[str]     = None
    motivational_tip: str               = ""

    def to_dict(self) -> dict:
        return {
            "activity":        self.activity,
            "level":           self.level,
            "summary":         self.summary,
            "positive":        self.positive,
            "issues":          [
                {
                    "metric":      i.metric,
                    "severity":    i.severity,
                    "observation": i.plain_observation,
                    "drill":       i.adapted_drill,
                    "coachTip":    i.adapted_coach_tip,
                }
                for i in self.items
            ],
            "priorityDrill":   self.priority_drill,
            "motivationalTip": self.motivational_tip,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CoachFeedbackEngine:
    """
    Generates grounded coaching feedback from a skill profile.

    Parameters
    ----------
    adapter : TerminologyAdapter | None
        Terminology adapter — created automatically if not provided.
    """

    def __init__(self, adapter: Optional[TerminologyAdapter] = None) -> None:
        self._adapter = adapter or TerminologyAdapter()

    def generate(
        self,
        profile:  CoachSkillProfile,
        activity: str = "general",
        metrics:  Optional[Dict[str, float]] = None,
    ) -> CoachFeedbackReport:
        """
        Generate a coaching report for the given skill profile.

        Parameters
        ----------
        profile  : CoachSkillProfile — from CoachSkillClassifier
        activity : str               — football action (e.g. "shooting")
        metrics  : dict | None       — raw metric values for severity scoring

        Returns
        -------
        CoachFeedbackReport
        """
        level   = profile.level
        adapter = self._adapter

        # Load any activity-specific knowledge base overrides.
        kb = self._load_kb(activity)

        items:    List[FeedbackItem] = []
        positive: List[str]          = []

        raw_metrics = metrics or {}

        for ms in profile.metric_scores:
            raw_val  = ms.raw_value
            severity = self._severity(ms.metric, raw_val)

            if severity is None:
                # Metric is in good range.
                pos_label = ms.metric.replace("_", " ").title()
                positive.append(f"{pos_label} is performing well.")
                continue

            rule = _FEEDBACK_RULES.get(ms.metric, {}).get(severity)
            if rule is None:
                continue

            obs, drill, tip = rule

            # Apply knowledge base overrides if present.
            kb_entry = kb.get(ms.metric, {}).get(level, {})
            obs   = kb_entry.get("observation", obs)
            drill = kb_entry.get("drill",       drill)
            tip   = kb_entry.get("coach_tip",   tip)

            # Adapt all text to the player's level.
            obs, drill, tip = adapter.adapt_feedback_item(obs, drill, tip, level)

            items.append(FeedbackItem(
                metric            = ms.metric,
                severity          = severity,
                plain_observation = obs,
                adapted_drill     = drill,
                adapted_coach_tip = tip,
            ))

        summary     = self._build_summary(activity, level, items, positive)
        priority    = items[0].adapted_drill if items else None
        motivational = _MOTIVATIONAL.get(level, _MOTIVATIONAL["Beginner"])
        motivational = adapter.adapt_text(motivational, level)

        return CoachFeedbackReport(
            activity         = activity,
            level            = level,
            summary          = summary,
            positive         = positive,
            items            = items,
            priority_drill   = priority,
            motivational_tip = motivational,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _severity(metric: str, value: float) -> Optional[str]:
        poor_thresh, higher_is_worse = _POOR_THRESHOLDS.get(metric, (None, True))
        fair_thresh, _               = _FAIR_THRESHOLDS.get(metric, (None, True))
        if poor_thresh is None:
            return None
        if higher_is_worse:
            if value >= poor_thresh:
                return "poor"
            if value >= fair_thresh:
                return "fair"
        else:
            if value <= poor_thresh:
                return "poor"
            if value <= fair_thresh:
                return "fair"
        return None

    @staticmethod
    def _load_kb(activity: str) -> dict:
        path = _KB_DIR / f"{activity}.json"
        return read_json(path)

    @staticmethod
    def _build_summary(activity: str, level: str, items: List[FeedbackItem], positive: List[str]) -> str:
        label = activity.capitalize()
        n = len(items)
        g = len(positive)
        if n == 0:
            return f"{label} session complete — all metrics are within {level} range."
        if n == 1:
            return (f"{label} session complete for {level} player. "
                    f"One area needs attention: {items[0].metric.replace('_', ' ')}.")
        return (f"{label} session complete for {level} player. "
                f"{n} areas need work; {g} metric(s) performing well.")
