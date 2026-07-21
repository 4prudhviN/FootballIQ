#!/usr/bin/env python3
"""
Skill Classifier
================

Classifies a player's skill level (Beginner / Intermediate / Advanced)
based on performance metrics extracted from video analysis.

Pipeline:
    Video
      ↓
    Metrics  (produced by analyze_movement.py / activity_detector.py)
      ↓
    Player Level
      ↓
    Beginner | Intermediate | Advanced

The classifier is deterministic — no ML model required.  It scores
each metric against sport-science thresholds and derives an overall
level from the weighted score.

Usage (standalone):
    python skill_classifier.py --metrics '{"torso_lean": -8.5, "knee_dev": 0.18, "gait_asymmetry": 0.09, "leg_speed": 42.3}'
    python skill_classifier.py --help

Usage (as a module):
    from skill_classifier import classify_skill, SkillLevel, PlayerMetrics
    level, report = classify_skill(metrics)
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Skill levels
# ---------------------------------------------------------------------------

class SkillLevel(str, Enum):
    BEGINNER     = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED     = "Advanced"


# ---------------------------------------------------------------------------
# Input metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlayerMetrics:
    """
    Performance metrics produced by the video analysis pipeline.
    Pass None for any metric that could not be measured (e.g., the player
    was off-frame during that phase).

    Attributes
    ----------
    torso_lean : float | None
        Signed torso lean angle (degrees) at the key movement event.
        Negative = leaning back (bad for most actions).
        Range typically –90 to +90.

    knee_dev : float | None
        Absolute knee deviation ratio (fraction of thigh length).
        0 = neutral alignment; higher = more valgus/varus.

    gait_asymmetry : float | None
        Stride asymmetry ratio [0, 1].
        0 = perfectly symmetric; 1 = complete imbalance.

    leg_speed : float | None
        Peak leg speed in pixels/frame during the key action.
        Higher = faster, more explosive movement.

    movement_consistency : float | None
        Standard deviation of torso lean across frames (degrees).
        Lower = more consistent / controlled movement.
    """
    torso_lean:            Optional[float] = None
    knee_dev:              Optional[float] = None
    gait_asymmetry:        Optional[float] = None
    leg_speed:             Optional[float] = None
    movement_consistency:  Optional[float] = None


# ---------------------------------------------------------------------------
# Threshold tables
# ---------------------------------------------------------------------------
# Each metric has two thresholds:
#   - advanced_threshold : value at or below (or above) which the player is
#                          considered Advanced.
#   - beginner_threshold : value at or below (or above) which the player is
#                          considered Beginner.
# Everything between the two thresholds is Intermediate.
#
# Format: (advanced_threshold, beginner_threshold, higher_is_better)
#   higher_is_better=True  → high values are Advanced
#   higher_is_better=False → low values are Advanced

THRESHOLDS: dict[str, tuple[float, float, bool]] = {
    # Torso lean: we use absolute lean (ignore sign for level classification).
    # Advanced players maintain near-upright posture (lean < 8°).
    # Beginners often lean back past 20°.
    "torso_lean": (8.0, 20.0, False),

    # Knee deviation: Advanced < 0.15, Beginner > 0.30.
    "knee_dev": (0.15, 0.30, False),

    # Gait asymmetry: Advanced < 0.08, Beginner > 0.20.
    "gait_asymmetry": (0.08, 0.20, False),

    # Leg speed: Advanced > 50 px/frame, Beginner < 20 px/frame.
    "leg_speed": (50.0, 20.0, True),

    # Movement consistency (std dev of torso lean):
    # Advanced < 5°, Beginner > 15°.
    "movement_consistency": (5.0, 15.0, False),
}

# Weights for the weighted scoring (must sum to 1.0).
WEIGHTS: dict[str, float] = {
    "torso_lean":           0.30,
    "knee_dev":             0.25,
    "gait_asymmetry":       0.20,
    "leg_speed":            0.15,
    "movement_consistency": 0.10,
}

# Score boundaries for the final level assignment.
# score ∈ [0, 1]: closer to 1 = more Advanced.
ADVANCED_SCORE_THRESHOLD:     float = 0.70
BEGINNER_SCORE_THRESHOLD:     float = 0.35


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_metric(
    value: float,
    advanced_thresh: float,
    beginner_thresh: float,
    higher_is_better: bool,
) -> float:
    """
    Map a single metric value to a normalised score in [0.0, 1.0].

    1.0 = Advanced performance.
    0.0 = Beginner performance.

    Linear interpolation between the two thresholds for Intermediate values.
    """
    if higher_is_better:
        # Larger value → better (e.g. leg_speed).
        if value >= advanced_thresh:
            return 1.0
        if value <= beginner_thresh:
            return 0.0
        # Linear scale: beginner_thresh → 0.0, advanced_thresh → 1.0
        return (value - beginner_thresh) / (advanced_thresh - beginner_thresh)
    else:
        # Smaller value → better (e.g. torso_lean, knee_dev).
        if value <= advanced_thresh:
            return 1.0
        if value >= beginner_thresh:
            return 0.0
        # Linear scale: advanced_thresh → 1.0, beginner_thresh → 0.0
        return 1.0 - (value - advanced_thresh) / (beginner_thresh - advanced_thresh)


def weighted_score(metrics: PlayerMetrics) -> tuple[float, dict[str, float]]:
    """
    Compute the overall weighted skill score and per-metric breakdown.

    Returns
    -------
    overall : float
        Weighted average score in [0.0, 1.0].
    breakdown : dict
        Per-metric score (or None if metric was unavailable).
    """
    total_weight = 0.0
    total_score  = 0.0
    breakdown: dict[str, float] = {}

    metric_values = {
        "torso_lean":           abs(metrics.torso_lean)          if metrics.torso_lean          is not None else None,
        "knee_dev":             metrics.knee_dev                  if metrics.knee_dev             is not None else None,
        "gait_asymmetry":       metrics.gait_asymmetry            if metrics.gait_asymmetry       is not None else None,
        "leg_speed":            metrics.leg_speed                 if metrics.leg_speed            is not None else None,
        "movement_consistency": metrics.movement_consistency      if metrics.movement_consistency is not None else None,
    }

    for metric_name, value in metric_values.items():
        if value is None:
            breakdown[metric_name] = None
            continue  # skip unavailable metrics; redistribute their weight

        adv, beg, higher = THRESHOLDS[metric_name]
        s = score_metric(value, adv, beg, higher)
        w = WEIGHTS[metric_name]

        breakdown[metric_name] = round(s, 3)
        total_score  += s * w
        total_weight += w

    # Normalise by actually-used weight so missing metrics don't drag score down.
    overall = (total_score / total_weight) if total_weight > 0 else 0.0
    return round(overall, 3), breakdown


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

@dataclass
class ClassificationReport:
    """Full output of a skill classification run."""
    level:           SkillLevel
    overall_score:   float                  # 0.0 – 1.0
    metric_scores:   dict[str, Optional[float]]
    strengths:       list[str]
    weaknesses:      list[str]
    recommendations: list[str]


def classify_skill(metrics: PlayerMetrics) -> ClassificationReport:
    """
    Classify a player's skill level and generate a coaching report.

    Parameters
    ----------
    metrics : PlayerMetrics
        Performance metrics from the video analysis pipeline.

    Returns
    -------
    ClassificationReport
        Level, score breakdown, strengths, weaknesses, recommendations.
    """
    overall, breakdown = weighted_score(metrics)

    # Assign level.
    if overall >= ADVANCED_SCORE_THRESHOLD:
        level = SkillLevel.ADVANCED
    elif overall >= BEGINNER_SCORE_THRESHOLD:
        level = SkillLevel.INTERMEDIATE
    else:
        level = SkillLevel.BEGINNER

    # Identify strengths and weaknesses from per-metric scores.
    strengths:   list[str] = []
    weaknesses:  list[str] = []

    metric_labels = {
        "torso_lean":           "Torso alignment",
        "knee_dev":             "Knee stability",
        "gait_asymmetry":       "Gait symmetry",
        "leg_speed":            "Leg speed / explosiveness",
        "movement_consistency": "Movement consistency",
    }

    for metric, score in breakdown.items():
        if score is None:
            continue
        label = metric_labels.get(metric, metric)
        if score >= 0.70:
            strengths.append(label)
        elif score <= 0.35:
            weaknesses.append(label)

    # Generate targeted recommendations.
    recommendations = _generate_recommendations(level, weaknesses, breakdown)

    return ClassificationReport(
        level=level,
        overall_score=overall,
        metric_scores=breakdown,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations,
    )


def _generate_recommendations(
    level: SkillLevel,
    weaknesses: list[str],
    breakdown: dict[str, Optional[float]],
) -> list[str]:
    """Return a list of actionable coaching tips based on level and weak areas."""
    tips: list[str] = []

    if "Torso alignment" in weaknesses:
        tips.append(
            "Core activation drills: practice upright posture during ball strikes. "
            "Video self-review at 0.5× speed to see lean angle."
        )

    if "Knee stability" in weaknesses:
        tips.append(
            "Single-leg balance and lateral band walks to improve knee tracking. "
            "Ensure knee stays over second toe during all weight-bearing phases."
        )

    if "Gait symmetry" in weaknesses:
        tips.append(
            "Unilateral strength work (single-leg squats, lunges) to address "
            "left/right imbalances. Monitor stride length in slow-motion replay."
        )

    if "Leg speed / explosiveness" in weaknesses:
        if level == SkillLevel.BEGINNER:
            tips.append(
                "Focus on basic plyometrics (box jumps, skipping) to build "
                "explosive leg power before progressing to football-specific drills."
            )
        else:
            tips.append(
                "Sprint resistance training and reactive agility drills to "
                "increase peak leg velocity during the swing phase."
            )

    if "Movement consistency" in weaknesses:
        tips.append(
            "Increase repetitions of the target skill under low-pressure "
            "conditions to build motor pattern stability (200–300 reps/session)."
        )

    # Level-specific general advice.
    if level == SkillLevel.BEGINNER:
        tips.append(
            "Priority: fundamental movement literacy — focus on posture, "
            "balance, and basic ball control before tactical development."
        )
    elif level == SkillLevel.INTERMEDIATE:
        tips.append(
            "Priority: efficiency and consistency — reduce energy waste "
            "and tighten technical execution under match-speed conditions."
        )
    else:
        tips.append(
            "Priority: marginal gains — fine-tune biomechanical details "
            "and increase performance under high-pressure scenarios."
        )

    return tips


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(report: ClassificationReport) -> None:
    """Print a human-readable classification report to stdout."""
    divider = "─" * 52
    level_icons = {
        SkillLevel.BEGINNER:     "🟡",
        SkillLevel.INTERMEDIATE: "🔵",
        SkillLevel.ADVANCED:     "🟢",
    }
    icon = level_icons.get(report.level, "⚽")

    print(f"\n{divider}")
    print(f"  FootballIQ  —  Skill Classification Report")
    print(divider)
    print(f"  Level         : {icon}  {report.level.value}")
    print(f"  Overall score : {report.overall_score:.2f} / 1.00")
    print(divider)
    print("  Metric Scores:")
    for metric, score in report.metric_scores.items():
        label = metric.replace("_", " ").title()
        value = f"{score:.3f}" if score is not None else "N/A"
        print(f"    {label:<28} {value}")
    print(divider)
    if report.strengths:
        print("  Strengths:")
        for s in report.strengths:
            print(f"    ✓  {s}")
    if report.weaknesses:
        print("  Areas to improve:")
        for w in report.weaknesses:
            print(f"    ✗  {w}")
    print(divider)
    if report.recommendations:
        print("  Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            # Wrap long lines at 60 chars.
            words = rec.split()
            line, lines = [], []
            for word in words:
                if sum(len(w) + 1 for w in line) + len(word) > 60:
                    lines.append(" ".join(line))
                    line = [word]
                else:
                    line.append(word)
            if line:
                lines.append(" ".join(line))
            print(f"    {i}. {lines[0]}")
            for continuation in lines[1:]:
                print(f"       {continuation}")
    print(divider)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "FootballIQ Skill Classifier — classify player level from "
            "performance metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python skill_classifier.py --metrics '{"torso_lean": -6.0, "knee_dev": 0.12, "gait_asymmetry": 0.07, "leg_speed": 55.0}'
  python skill_classifier.py --metrics '{"torso_lean": -25.0, "knee_dev": 0.35}'
  python skill_classifier.py --metrics-file player_metrics.json
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--metrics", type=str,
        help="JSON string of metric key-value pairs.",
    )
    group.add_argument(
        "--metrics-file", type=str,
        help="Path to a JSON file containing metric key-value pairs.",
    )
    args = parser.parse_args()

    # Load metrics.
    try:
        if args.metrics:
            raw = json.loads(args.metrics)
        else:
            with open(args.metrics_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    metrics = PlayerMetrics(
        torso_lean=raw.get("torso_lean"),
        knee_dev=raw.get("knee_dev"),
        gait_asymmetry=raw.get("gait_asymmetry"),
        leg_speed=raw.get("leg_speed"),
        movement_consistency=raw.get("movement_consistency"),
    )

    report = classify_skill(metrics)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
