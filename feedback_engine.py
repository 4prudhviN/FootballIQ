#!/usr/bin/env python3
"""
Feedback Engine
===============

Converts raw performance metrics into plain-English coaching feedback,
targeted training drills, and coach tips — grounded in the football
knowledge base (football_knowledge/*.json).

Pipeline:
    Metrics
      ↓
    Knowledge Base  (football_knowledge/*.json)
      ↓
    Simple English  (what the player did wrong / right)
      ↓
    Training Drill  (what to practise)
      ↓
    Coach Tip       (tactical / biomechanical cue)

Usage (standalone):
    python feedback_engine.py --metrics '{"torso_lean": -22.0, "knee_dev": 0.28, "gait_asymmetry": 0.18, "leg_speed": 35.0}' --activity shooting --level Intermediate
    python feedback_engine.py --help

Usage (as a module):
    from feedback_engine import FeedbackEngine, FeedbackRequest
    engine = FeedbackEngine()
    result = engine.generate(FeedbackRequest(metrics=..., activity="shooting", level="Intermediate"))
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "football_knowledge"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRequest:
    """
    Input to the feedback engine.

    Parameters
    ----------
    metrics : dict
        Key-value pairs from the analysis pipeline.
        Expected keys (all optional): torso_lean, knee_dev,
        gait_asymmetry, leg_speed, movement_consistency.
    activity : str
        The detected football activity: shooting, passing, dribbling,
        defending, goalkeeping.  Defaults to "general".
    level : str
        Player skill level: Beginner, Intermediate, Advanced.
        Produced by skill_classifier.py.
    """
    metrics:  dict              = field(default_factory=dict)
    activity: str               = "general"
    level:    str               = "Beginner"


@dataclass
class FeedbackItem:
    """A single piece of feedback for one metric."""
    metric:       str           # e.g. "torso_lean"
    observation:  str           # plain English — what was observed
    drill:        str           # specific training drill to fix it
    coach_tip:    str           # short tactical / biomechanical cue


@dataclass
class FeedbackReport:
    """Full output of the feedback engine."""
    activity:         str
    level:            str
    summary:          str           # one-sentence overall summary
    positive:         list[str]     # things the player did well
    items:            list[FeedbackItem]   # one item per weak metric
    priority_drill:   Optional[str] # the single most important drill
    motivational_tip: str           # closing encouragement


# ---------------------------------------------------------------------------
# Knowledge base loader
# ---------------------------------------------------------------------------

def load_knowledge(activity: str) -> dict:
    """
    Load the football knowledge JSON for the given activity.
    Falls back to an empty dict if the file is missing or malformed.
    """
    filename = KNOWLEDGE_BASE_DIR / f"{activity}.json"
    if not filename.exists():
        return {}
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Built-in feedback rules
# ---------------------------------------------------------------------------
# Structure:
#   FEEDBACK_RULES[metric][severity] = (observation, drill, coach_tip)
#   severity: "poor" | "fair"  (fair = approaching threshold, poor = clearly bad)

FEEDBACK_RULES: dict[str, dict[str, tuple[str, str, str]]] = {
    "torso_lean": {
        "poor": (
            "Your upper body is leaning back too far when striking the ball. "
            "This sends shots high and reduces passing accuracy.",
            "Wall lean drill: stand 30 cm from a wall and practise driving "
            "your knee up without your back touching the wall — 3 × 15 reps.",
            "At the moment of contact, your chest should be over the ball. "
            "Think 'chin down, chest forward'.",
        ),
        "fair": (
            "Slight backward lean detected during ball contact. "
            "Your posture is close but could cost accuracy under pressure.",
            "Mirror slow-motion kick drill: record yourself at 0.5× speed "
            "and pause at the moment of contact to check torso angle.",
            "Keep your non-kicking foot planted firmly beside the ball — "
            "this naturally keeps your torso more upright.",
        ),
    },
    "knee_dev": {
        "poor": (
            "Your knee is collapsing inward (valgus) during weight-bearing. "
            "This increases injury risk and reduces power transfer.",
            "Lateral band walk: place a resistance band above your knees and "
            "take 20 side steps each direction — 3 sets daily.",
            "Focus on pushing your knee outward over your second toe when "
            "planting or landing.  Strong glutes = stable knees.",
        ),
        "fair": (
            "Mild knee deviation detected. "
            "Not yet dangerous, but worth addressing before it becomes a habit.",
            "Single-leg squat (pistol squat progressions): 3 × 8 reps each leg, "
            "focusing on keeping the knee tracking over the middle toe.",
            "Before each training session, activate your glutes with 2 minutes "
            "of clamshells or monster walks to 'wake up' knee-stabilising muscles.",
        ),
    },
    "gait_asymmetry": {
        "poor": (
            "Significant left-right stride imbalance detected. "
            "One leg is doing more work, which fatigues you faster and "
            "makes your movement predictable to opponents.",
            "Unilateral running drills: run 20 m on one leg (bounds), then "
            "switch. Do 5 sets each leg to equalise strength and stride length.",
            "Check your weaker side — most players have a dominant leg that "
            "over-strides. Consciously shorten that stride and lengthen the other.",
        ),
        "fair": (
            "Slight asymmetry in your running stride. "
            "Small imbalances add up over 90 minutes.",
            "Single-leg hop and stick: hop forward on one foot and freeze on "
            "landing — 3 × 10 each side. Compare stability between sides.",
            "During warm-up, include 2 × 20 m of high-knee running focusing on "
            "equal arm swing and equal knee height each side.",
        ),
    },
    "leg_speed": {
        "poor": (
            "Low leg speed detected during the key action. "
            "Insufficient explosiveness will limit shot power and "
            "your ability to accelerate past defenders.",
            "Plyometric circuit: jump squats × 10, tuck jumps × 10, "
            "lateral hops × 10 each side — 3 rounds, 90 sec rest.",
            "Explosiveness starts from the hip flexors. "
            "Sprint reaction drills (on a clap signal) twice a week will "
            "train your fast-twitch fibres.",
        ),
        "fair": (
            "Leg speed is adequate but has room to improve. "
            "Faster leg swing means harder shots and quicker first steps.",
            "Resistance band sprints: attach a light band around your waist "
            "and sprint 15 m against resistance — 6 reps per session.",
            "Focus on your arm drive — arms pump legs. "
            "A faster, tighter arm swing directly increases stride frequency.",
        ),
    },
    "movement_consistency": {
        "poor": (
            "High variability in your movement pattern across frames. "
            "Inconsistent technique means unpredictable results under "
            "match pressure.",
            "Repetition block training: perform the same skill 50 times "
            "in a row at 70% intensity, focusing purely on identical mechanics "
            "each rep.",
            "Slow it down to speed it up. "
            "Practise the movement at 50% speed until it feels automatic, "
            "then gradually increase to full pace over 2–3 weeks.",
        ),
        "fair": (
            "Some variation in movement pattern detected. "
            "Your technique is mostly solid but breaks down slightly "
            "under fatigue or speed.",
            "Fatigue-state training: perform the skill at the end of a "
            "hard session when tired — this is when poor habits emerge.",
            "Video yourself once a week. "
            "Compare your first rep to your 30th rep — the difference "
            "tells you exactly where consistency breaks down.",
        ),
    },
}

# Thresholds for classifying severity (mirrors skill_classifier.py logic).
POOR_THRESHOLDS: dict[str, tuple[float, bool]] = {
    # (threshold, higher_is_worse)
    "torso_lean":           (20.0, True),
    "knee_dev":             (0.30, True),
    "gait_asymmetry":       (0.20, True),
    "leg_speed":            (20.0, False),   # low value is worse
    "movement_consistency": (15.0, True),
}

FAIR_THRESHOLDS: dict[str, tuple[float, bool]] = {
    "torso_lean":           (8.0,  True),
    "knee_dev":             (0.15, True),
    "gait_asymmetry":       (0.08, True),
    "leg_speed":            (50.0, False),
    "movement_consistency": (5.0,  True),
}

# Positive observations — shown when a metric is good.
POSITIVE_OBSERVATIONS: dict[str, str] = {
    "torso_lean":           "Great body posture — torso well balanced over the ball.",
    "knee_dev":             "Excellent knee alignment throughout the movement.",
    "gait_asymmetry":       "Symmetric stride pattern — both legs contributing equally.",
    "leg_speed":            "Strong leg speed and explosive movement detected.",
    "movement_consistency": "Consistent and repeatable movement mechanics.",
}

# Level-specific motivational closers.
MOTIVATIONAL_TIPS: dict[str, str] = {
    "Beginner":     "Every elite player started exactly where you are. "
                    "Focus on one drill at a time — small improvements compound fast.",
    "Intermediate": "You have a solid foundation. "
                    "The gap between Intermediate and Advanced is consistency — "
                    "bring this intensity to every session.",
    "Advanced":     "Marginal gains are your competitive edge. "
                    "The drills above will sharpen the details that separate "
                    "good players from great ones.",
}


# ---------------------------------------------------------------------------
# Feedback Engine
# ---------------------------------------------------------------------------

class FeedbackEngine:
    """
    Generates plain-English feedback, drills, and coach tips from
    player metrics and the football knowledge base.
    """

    def __init__(self, knowledge_base_dir: Optional[Path] = None):
        self._kb_dir = knowledge_base_dir or KNOWLEDGE_BASE_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, request: FeedbackRequest) -> FeedbackReport:
        """
        Generate a full feedback report for the given request.

        Parameters
        ----------
        request : FeedbackRequest

        Returns
        -------
        FeedbackReport
        """
        activity = request.activity.lower().strip()
        level    = request.level.strip()
        metrics  = request.metrics

        # Load activity-specific knowledge (may be empty dict).
        knowledge = load_knowledge(activity)

        # Classify each metric and build feedback items.
        feedback_items: list[FeedbackItem] = []
        positive_obs:   list[str]          = []

        for metric_name in FEEDBACK_RULES:
            raw_value = metrics.get(metric_name)
            if raw_value is None:
                continue

            # Use absolute value for lean angle comparison.
            value = abs(raw_value) if metric_name == "torso_lean" else raw_value

            severity = self._classify_severity(metric_name, value)

            if severity is None:
                # Metric is in good range — record as positive.
                pos = POSITIVE_OBSERVATIONS.get(metric_name)
                if pos:
                    positive_obs.append(pos)
                continue

            # Retrieve base rule.
            obs, drill, tip = FEEDBACK_RULES[metric_name][severity]

            # Optionally enrich drill / tip from knowledge base JSON.
            obs, drill, tip = self._enrich_from_knowledge(
                knowledge, activity, metric_name, level, obs, drill, tip
            )

            feedback_items.append(FeedbackItem(
                metric=metric_name,
                observation=obs,
                drill=drill,
                coach_tip=tip,
            ))

        # Build summary sentence.
        summary = self._build_summary(activity, level, feedback_items, positive_obs)

        # Pick the single most critical drill.
        priority_drill = feedback_items[0].drill if feedback_items else None

        # Motivational closer (level-aware).
        motivational = MOTIVATIONAL_TIPS.get(level, MOTIVATIONAL_TIPS["Beginner"])

        return FeedbackReport(
            activity=activity,
            level=level,
            summary=summary,
            positive=positive_obs,
            items=feedback_items,
            priority_drill=priority_drill,
            motivational_tip=motivational,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_severity(metric_name: str, value: float) -> Optional[str]:
        """
        Return "poor", "fair", or None (good) for the given metric value.
        """
        poor_thresh, higher_is_worse = POOR_THRESHOLDS.get(metric_name, (None, True))
        fair_thresh, _               = FAIR_THRESHOLDS.get(metric_name, (None, True))

        if poor_thresh is None or fair_thresh is None:
            return None

        if higher_is_worse:
            if value >= poor_thresh:
                return "poor"
            if value >= fair_thresh:
                return "fair"
        else:
            # Lower is worse (e.g. leg_speed).
            if value <= poor_thresh:
                return "poor"
            if value <= fair_thresh:
                return "fair"

        return None  # metric is in the good range

    @staticmethod
    def _enrich_from_knowledge(
        knowledge: dict,
        activity: str,
        metric: str,
        level: str,
        obs: str,
        drill: str,
        tip: str,
    ) -> tuple[str, str, str]:
        """
        If the knowledge base JSON contains activity- or level-specific
        overrides for this metric, use them; otherwise return originals.

        Expected JSON structure (example in football_knowledge/shooting.json):
        {
          "torso_lean": {
            "Beginner": {
              "drill": "...",
              "coach_tip": "..."
            }
          }
        }
        """
        if not knowledge:
            return obs, drill, tip

        metric_kb = knowledge.get(metric, {})
        if not metric_kb:
            return obs, drill, tip

        level_kb = metric_kb.get(level, metric_kb.get("all", {}))
        if not level_kb:
            return obs, drill, tip

        drill = level_kb.get("drill", drill)
        tip   = level_kb.get("coach_tip", tip)
        obs   = level_kb.get("observation", obs)

        return obs, drill, tip

    @staticmethod
    def _build_summary(
        activity: str,
        level: str,
        items: list[FeedbackItem],
        positive: list[str],
    ) -> str:
        activity_label = activity.capitalize() if activity != "general" else "Movement"
        n_issues = len(items)
        n_good   = len(positive)

        if n_issues == 0:
            return (
                f"{activity_label} analysis complete. "
                f"All metrics are within the {level} performance range — "
                f"strong session overall."
            )
        if n_issues == 1:
            return (
                f"{activity_label} analysis complete for {level} player. "
                f"One area needs attention: {items[0].metric.replace('_', ' ')}."
            )
        return (
            f"{activity_label} analysis complete for {level} player. "
            f"{n_issues} areas need work; {n_good} metric(s) are performing well. "
            f"See drills below."
        )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(report: FeedbackReport) -> None:
    divider = "─" * 56
    level_icons = {"Beginner": "🟡", "Intermediate": "🔵", "Advanced": "🟢"}
    icon = level_icons.get(report.level, "⚽")

    print(f"\n{divider}")
    print(f"  FootballIQ  —  Coaching Feedback Report")
    print(divider)
    print(f"  Activity : {report.activity.capitalize()}")
    print(f"  Level    : {icon}  {report.level}")
    print(f"\n  {report.summary}")

    if report.positive:
        print(f"\n{divider}")
        print("  ✅  What you're doing well:")
        for p in report.positive:
            print(f"      •  {p}")

    if report.items:
        print(f"\n{divider}")
        print("  ⚠️   Areas to improve:")
        for item in report.items:
            label = item.metric.replace("_", " ").title()
            print(f"\n  [{label}]")
            print(f"  Observation : {item.observation}")
            print(f"  Drill       : {item.drill}")
            print(f"  Coach tip   : {item.coach_tip}")

    if report.priority_drill:
        print(f"\n{divider}")
        print("  🎯  Priority drill (start here):")
        print(f"      {report.priority_drill}")

    print(f"\n{divider}")
    print(f"  💬  {report.motivational_tip}")
    print(divider)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "FootballIQ Feedback Engine — turn metrics into coaching feedback."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python feedback_engine.py \\
    --metrics '{"torso_lean": -22.0, "knee_dev": 0.28, "leg_speed": 35.0}' \\
    --activity shooting --level Intermediate

  python feedback_engine.py \\
    --metrics '{"gait_asymmetry": 0.25, "movement_consistency": 18.0}' \\
    --activity dribbling --level Beginner
        """,
    )
    parser.add_argument(
        "--metrics", type=str, required=True,
        help="JSON string of metric key-value pairs.",
    )
    parser.add_argument(
        "--activity", type=str, default="general",
        choices=["shooting", "passing", "dribbling", "defending", "goalkeeping", "general"],
        help="Football activity being analysed (default: general).",
    )
    parser.add_argument(
        "--level", type=str, default="Beginner",
        choices=["Beginner", "Intermediate", "Advanced"],
        help="Player skill level from skill_classifier.py (default: Beginner).",
    )
    args = parser.parse_args()

    try:
        metrics = json.loads(args.metrics)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in --metrics: {exc}", file=sys.stderr)
        return 1

    engine  = FeedbackEngine()
    request = FeedbackRequest(
        metrics=metrics,
        activity=args.activity,
        level=args.level,
    )
    report = engine.generate(request)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
