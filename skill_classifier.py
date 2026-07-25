#!/usr/bin/env python3
"""
Skill Classifier
================
Classifies a player's skill level from performance metrics.

Five levels:
  Beginner     — just starting out
  Developing   — building fundamentals
  Intermediate — solid core skills
  Advanced     — high technical ability
  Elite        — professional / near-professional standard

Pipeline position:
  Metrics → Skill Classifier → Player Level → Coach Engine

Usage (standalone):
    python skill_classifier.py --metrics '{"torso_lean": 8.5, "knee_dev": 0.18}'

Usage (as a module):
    from skill_classifier import classify_skill, SkillLevel, PlayerMetrics
    report = classify_skill(metrics)
    print(report.level)   # "Intermediate"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Skill Levels  (5 levels)
# ---------------------------------------------------------------------------

class SkillLevel(str, Enum):
    BEGINNER     = "Beginner"
    DEVELOPING   = "Developing"
    INTERMEDIATE = "Intermediate"
    ADVANCED     = "Advanced"
    ELITE        = "Elite"


# Level score boundaries — score ∈ [0, 1]
_LEVEL_BOUNDARIES = [
    (0.85, SkillLevel.ELITE),
    (0.70, SkillLevel.ADVANCED),
    (0.50, SkillLevel.INTERMEDIATE),
    (0.30, SkillLevel.DEVELOPING),
    (0.00, SkillLevel.BEGINNER),
]


# ---------------------------------------------------------------------------
# Input metrics
# ---------------------------------------------------------------------------

@dataclass
class PlayerMetrics:
    """
    Performance metrics produced by the video analysis pipeline.
    All fields optional — missing metrics are excluded from scoring.
    """
    torso_lean:            Optional[float] = None   # degrees (abs)
    knee_dev:              Optional[float] = None   # deviation ratio
    gait_asymmetry:        Optional[float] = None   # asymmetry fraction
    leg_speed:             Optional[float] = None   # pixels/frame
    movement_consistency:  Optional[float] = None   # std-dev degrees
    pass_accuracy:         Optional[float] = None   # %
    shot_accuracy:         Optional[float] = None   # %
    dribble_success_rate:  Optional[float] = None   # %
    tackle_success_rate:   Optional[float] = None   # %
    balance:               Optional[float] = None   # 0–100
    foot_placement:        Optional[float] = None   # 0–100


# ---------------------------------------------------------------------------
# Thresholds — (elite_thresh, beginner_thresh, higher_is_better)
# ---------------------------------------------------------------------------

_THRESHOLDS: Dict[str, tuple[float, float, bool]] = {
    # Biomechanical — lower values are better
    "torso_lean":           (5.0,  25.0,  False),
    "knee_dev":             (0.10, 0.35,  False),
    "gait_asymmetry":       (0.05, 0.25,  False),
    "movement_consistency": (3.0,  18.0,  False),

    # Speed — higher is better
    "leg_speed":            (60.0, 15.0,  True),

    # Accuracy — higher is better
    "pass_accuracy":        (90.0, 50.0,  True),
    "shot_accuracy":        (80.0, 35.0,  True),
    "dribble_success_rate": (85.0, 40.0,  True),
    "tackle_success_rate":  (80.0, 35.0,  True),

    # Composites — higher is better
    "balance":              (92.0, 55.0,  True),
    "foot_placement":       (90.0, 50.0,  True),
}

# Metric weights — must sum to 1.0
_WEIGHTS: Dict[str, float] = {
    "torso_lean":           0.20,
    "knee_dev":             0.15,
    "gait_asymmetry":       0.12,
    "movement_consistency": 0.08,
    "leg_speed":            0.10,
    "pass_accuracy":        0.08,
    "shot_accuracy":        0.07,
    "dribble_success_rate": 0.06,
    "tackle_success_rate":  0.06,
    "balance":              0.05,
    "foot_placement":       0.03,
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    metric:  str
    value:   float
    score:   float   # 0.0–1.0
    label:   str     # "Elite" | "Advanced" | "Intermediate" | "Developing" | "Beginner"


@dataclass
class ClassificationReport:
    level:          SkillLevel
    overall_score:  float                   # 0.0–1.0
    metric_scores:  List[MetricScore]       = field(default_factory=list)
    strengths:      List[str]               = field(default_factory=list)
    weaknesses:     List[str]               = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level":         self.level.value,
            "overall_score": round(self.overall_score, 3),
            "strengths":     self.strengths,
            "weaknesses":    self.weaknesses,
            "metric_scores": {ms.metric: round(ms.score, 3) for ms in self.metric_scores},
        }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _score_metric(value: float, elite: float, beginner: float, higher_is_better: bool) -> float:
    """Map a metric value to [0, 1]. 1.0 = Elite performance."""
    if higher_is_better:
        if value >= elite:    return 1.0
        if value <= beginner: return 0.0
        return (value - beginner) / (elite - beginner)
    else:
        if value <= elite:    return 1.0
        if value >= beginner: return 0.0
        return 1.0 - (value - elite) / (beginner - elite)


def _level_from_score(score: float) -> SkillLevel:
    for threshold, level in _LEVEL_BOUNDARIES:
        if score >= threshold:
            return level
    return SkillLevel.BEGINNER


def _metric_level_label(score: float) -> str:
    return _level_from_score(score).value


def classify_skill(metrics: PlayerMetrics) -> ClassificationReport:
    """
    Classify player skill level from a PlayerMetrics object.

    Parameters
    ----------
    metrics : PlayerMetrics

    Returns
    -------
    ClassificationReport with 5-level classification
    """
    metric_values = {
        "torso_lean":           metrics.torso_lean,
        "knee_dev":             metrics.knee_dev,
        "gait_asymmetry":       metrics.gait_asymmetry,
        "movement_consistency": metrics.movement_consistency,
        "leg_speed":            metrics.leg_speed,
        "pass_accuracy":        metrics.pass_accuracy,
        "shot_accuracy":        metrics.shot_accuracy,
        "dribble_success_rate": metrics.dribble_success_rate,
        "tackle_success_rate":  metrics.tackle_success_rate,
        "balance":              metrics.balance,
        "foot_placement":       metrics.foot_placement,
    }

    metric_scores: List[MetricScore] = []
    total_weight   = 0.0
    weighted_score = 0.0

    for name, value in metric_values.items():
        if value is None:
            continue
        elite, beg, higher = _THRESHOLDS[name]
        score  = _score_metric(value, elite, beg, higher)
        weight = _WEIGHTS.get(name, 0.05)
        label  = _metric_level_label(score)

        metric_scores.append(MetricScore(
            metric = name,
            value  = round(value, 3),
            score  = round(score, 3),
            label  = label,
        ))
        weighted_score += score * weight
        total_weight   += weight

    overall = round(weighted_score / total_weight, 3) if total_weight > 0 else 0.0
    level   = _level_from_score(overall)

    strengths  = [ms.metric.replace("_", " ").title()
                  for ms in metric_scores if ms.score >= 0.70]
    weaknesses = [ms.metric.replace("_", " ").title()
                  for ms in metric_scores if ms.score <= 0.35]

    return ClassificationReport(
        level         = level,
        overall_score = overall,
        metric_scores = metric_scores,
        strengths     = strengths,
        weaknesses     = weaknesses,
    )


def classify_from_dict(metrics_dict: Dict[str, float]) -> ClassificationReport:
    """Convenience: classify from a plain dict."""
    return classify_skill(PlayerMetrics(
        torso_lean           = metrics_dict.get("torso_lean"),
        knee_dev             = metrics_dict.get("knee_dev"),
        gait_asymmetry       = metrics_dict.get("gait_asymmetry"),
        movement_consistency = metrics_dict.get("movement_consistency"),
        leg_speed            = metrics_dict.get("leg_speed"),
        pass_accuracy        = metrics_dict.get("pass_accuracy"),
        shot_accuracy        = metrics_dict.get("shot_accuracy"),
        dribble_success_rate = metrics_dict.get("dribble_success_rate"),
        tackle_success_rate  = metrics_dict.get("tackle_success_rate"),
        balance              = metrics_dict.get("balance"),
        foot_placement       = metrics_dict.get("foot_placement"),
    ))


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(report: ClassificationReport) -> None:
    icons = {
        SkillLevel.ELITE:        "🏆",
        SkillLevel.ADVANCED:     "🟢",
        SkillLevel.INTERMEDIATE: "🔵",
        SkillLevel.DEVELOPING:   "🟡",
        SkillLevel.BEGINNER:     "⚪",
    }
    icon = icons.get(report.level, "⚽")
    div  = "─" * 52
    print(f"\n{div}")
    print(f"  FootballIQ — Skill Classification")
    print(div)
    print(f"  Level         : {icon}  {report.level.value}")
    print(f"  Overall Score : {report.overall_score:.2f} / 1.00")
    print(div)
    print("  Metric Scores:")
    for ms in report.metric_scores:
        label = ms.metric.replace("_", " ").title()
        print(f"    {label:<28} {ms.score:.3f}  ({ms.label})")
    print(div)
    if report.strengths:
        print("  Strengths :")
        for s in report.strengths:
            print(f"    ✓  {s}")
    if report.weaknesses:
        print("  Weaknesses:")
        for w in report.weaknesses:
            print(f"    ✗  {w}")
    print(div)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="FootballIQ Skill Classifier (5 levels)")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metrics",      type=str, help="JSON string of metrics")
    group.add_argument("--metrics-file", type=str, help="Path to JSON file")
    args   = parser.parse_args()

    try:
        if args.metrics:
            raw = json.loads(args.metrics)
        else:
            with open(args.metrics_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    report = classify_from_dict(raw)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
