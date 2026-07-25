#!/usr/bin/env python3
"""
Skill Classifier
================
Classifies a player's skill level from multiple performance metrics.

Output:
  ⭐           Beginner
  ⭐⭐         Developing
  ⭐⭐⭐       Intermediate
  ⭐⭐⭐⭐     Advanced
  ⭐⭐⭐⭐⭐   Elite

Not based on one metric. Considers:
  - Passing Accuracy
  - Ball Control
  - Balance
  - Movement
  - Consistency
  - Technique (body alignment, foot placement)
  - Reaction Time (goalkeeper — future)
  - Decision Making (future)

Deterministic — no ML, no AI.

Usage (standalone):
    python skill_classifier.py --metrics '{"accuracy": 86, "balance": 88}'

Usage (as a module):
    from skill_classifier import classify_skill, PlayerMetrics
    report = classify_skill(metrics_dict)
    print(report.stars, report.level)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Skill Levels — 5 stars
# ---------------------------------------------------------------------------

class SkillLevel(str, Enum):
    BEGINNER     = "Beginner"
    DEVELOPING   = "Developing"
    INTERMEDIATE = "Intermediate"
    ADVANCED     = "Advanced"
    ELITE        = "Elite"


_LEVEL_STARS: Dict[SkillLevel, str] = {
    SkillLevel.BEGINNER:     "⭐",
    SkillLevel.DEVELOPING:   "⭐⭐",
    SkillLevel.INTERMEDIATE: "⭐⭐⭐",
    SkillLevel.ADVANCED:     "⭐⭐⭐⭐",
    SkillLevel.ELITE:        "⭐⭐⭐⭐⭐",
}

# Score → Level (score ∈ [0, 1])
_BOUNDARIES = [
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
    All metrics considered for skill classification.
    Pass None for metrics that are unavailable — they are excluded from scoring.
    """
    # Passing
    accuracy:            Optional[float] = None   # % (0–100)
    successful_passes:   Optional[float] = None   # count
    passing_consistency: Optional[float] = None   # 0–100

    # Ball control
    ball_control:        Optional[float] = None   # 0–100
    touch_tightness:     Optional[float] = None   # lower = better (ratio)
    dribble_success_rate: Optional[float] = None  # % (0–100)

    # Balance & movement
    balance:             Optional[float] = None   # 0–100
    stride_length:       Optional[float] = None   # metres
    acceleration:        Optional[float] = None   # m/s²
    foot_placement:      Optional[float] = None   # 0–100

    # Technique (body alignment)
    body_alignment:      Optional[float] = None   # 0–100 (converted from label)
    torso_lean:          Optional[float] = None   # degrees abs (lower = better)
    knee_stability:      Optional[float] = None   # 0–100

    # Consistency
    movement_consistency: Optional[float] = None  # 0–100

    # Shooting
    shot_accuracy:       Optional[float] = None   # % (0–100)
    shot_speed:          Optional[float] = None   # pixels/frame

    # Goalkeeper (future)
    reaction_time:       Optional[float] = None   # seconds (lower = better)
    save_percentage:     Optional[float] = None   # % (0–100)


# ---------------------------------------------------------------------------
# Scoring thresholds
# (elite_value, beginner_value, higher_is_better)
# ---------------------------------------------------------------------------

_THRESHOLDS: Dict[str, tuple] = {
    # Passing
    "accuracy":             (90.0, 50.0,  True),
    "passing_consistency":  (88.0, 50.0,  True),

    # Ball control
    "ball_control":         (90.0, 50.0,  True),
    "touch_tightness":      (0.03, 0.15,  False),  # lower is better
    "dribble_success_rate": (85.0, 40.0,  True),

    # Balance & movement
    "balance":              (92.0, 50.0,  True),
    "acceleration":         (6.0,  1.0,   True),
    "foot_placement":       (90.0, 50.0,  True),

    # Technique
    "body_alignment":       (90.0, 50.0,  True),   # converted from label
    "torso_lean":           (5.0,  25.0,  False),  # lower is better
    "knee_stability":       (90.0, 55.0,  True),

    # Consistency
    "movement_consistency": (88.0, 50.0,  True),

    # Shooting
    "shot_accuracy":        (80.0, 35.0,  True),
    "shot_speed":           (50.0, 15.0,  True),

    # Goalkeeper
    "reaction_time":        (0.20, 0.50,  False),  # lower is better
    "save_percentage":      (80.0, 40.0,  True),
}

# Metric weights (must sum to 1.0 across all possible metrics)
_WEIGHTS: Dict[str, float] = {
    "accuracy":             0.14,
    "passing_consistency":  0.06,
    "ball_control":         0.10,
    "touch_tightness":      0.06,
    "dribble_success_rate": 0.07,
    "balance":              0.08,
    "acceleration":         0.05,
    "foot_placement":       0.06,
    "body_alignment":       0.10,
    "torso_lean":           0.05,
    "knee_stability":       0.06,
    "movement_consistency": 0.07,
    "shot_accuracy":        0.06,
    "shot_speed":           0.04,
    "reaction_time":        0.02,
    "save_percentage":      0.02,
}

# Label → numeric score for body_alignment.
_ALIGNMENT_SCORES = {"Good": 85.0, "Fair": 60.0, "Poor": 35.0}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    metric:  str
    value:   float
    score:   float   # 0.0–1.0
    level:   str     # which level this metric individually corresponds to


@dataclass
class ClassificationReport:
    """Full skill classification output."""
    level:         SkillLevel
    stars:         str           # e.g. "⭐⭐⭐"
    star_count:    int           # 1–5
    overall_score: float         # 0.0–1.0
    metric_scores: List[MetricScore] = field(default_factory=list)
    strengths:     List[str]         = field(default_factory=list)
    weaknesses:    List[str]         = field(default_factory=list)
    top_weakness:  Optional[str]     = None
    top_strength:  Optional[str]     = None

    def to_dict(self) -> dict:
        return {
            "level":        self.level.value,
            "stars":        self.stars,
            "star_count":   self.star_count,
            "overall_score": round(self.overall_score, 3),
            "strengths":    self.strengths,
            "weaknesses":   self.weaknesses,
            "top_weakness": self.top_weakness,
            "top_strength": self.top_strength,
            "metric_scores": {
                ms.metric: {"score": round(ms.score, 3), "level": ms.level}
                for ms in self.metric_scores
            },
        }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _score_one(value: float, elite: float, beginner: float, higher_is_better: bool) -> float:
    if higher_is_better:
        if value >= elite:    return 1.0
        if value <= beginner: return 0.0
        return (value - beginner) / (elite - beginner)
    else:
        if value <= elite:    return 1.0
        if value >= beginner: return 0.0
        return 1.0 - (value - elite) / (beginner - elite)


def _level_for_score(score: float) -> SkillLevel:
    for threshold, level in _BOUNDARIES:
        if score >= threshold:
            return level
    return SkillLevel.BEGINNER


def classify_skill(metrics: "PlayerMetrics | Dict[str, float]") -> ClassificationReport:
    """
    Classify player skill level from metrics.

    Parameters
    ----------
    metrics : PlayerMetrics or dict

    Returns
    -------
    ClassificationReport with star rating
    """
    # Accept either PlayerMetrics dataclass or plain dict.
    if isinstance(metrics, dict):
        # Coerce alignment label if present.
        if "body_alignment" in metrics and isinstance(metrics["body_alignment"], str):
            metrics = dict(metrics)
            metrics["body_alignment"] = _ALIGNMENT_SCORES.get(
                metrics["body_alignment"], 60.0
            )
        pm = PlayerMetrics(**{
            k: v for k, v in metrics.items()
            if k in PlayerMetrics.__dataclass_fields__
        })
    else:
        pm = metrics
        if pm.body_alignment is None:
            pass   # skip

    # Coerce string body_alignment.
    if isinstance(pm.body_alignment, str):
        pm.body_alignment = _ALIGNMENT_SCORES.get(pm.body_alignment, 60.0)

    # Build per-metric scores.
    metric_values = {
        "accuracy":             pm.accuracy,
        "passing_consistency":  pm.passing_consistency,
        "ball_control":         pm.ball_control,
        "touch_tightness":      pm.touch_tightness,
        "dribble_success_rate": pm.dribble_success_rate,
        "balance":              pm.balance,
        "acceleration":         pm.acceleration,
        "foot_placement":       pm.foot_placement,
        "body_alignment":       pm.body_alignment,
        "torso_lean":           pm.torso_lean,
        "knee_stability":       pm.knee_stability,
        "movement_consistency": pm.movement_consistency,
        "shot_accuracy":        pm.shot_accuracy,
        "shot_speed":           pm.shot_speed,
        "reaction_time":        pm.reaction_time,
        "save_percentage":      pm.save_percentage,
    }

    metric_scores: List[MetricScore] = []
    total_weight   = 0.0
    weighted_score = 0.0

    for name, value in metric_values.items():
        if value is None:
            continue
        if name not in _THRESHOLDS:
            continue
        elite, beg, higher = _THRESHOLDS[name]
        score  = _score_one(float(value), elite, beg, higher)
        weight = _WEIGHTS.get(name, 0.05)
        label  = _level_for_score(score).value

        metric_scores.append(MetricScore(
            metric = name,
            value  = round(float(value), 3),
            score  = round(score, 3),
            level  = label,
        ))
        weighted_score += score * weight
        total_weight   += weight

    overall = round(weighted_score / total_weight, 3) if total_weight > 0 else 0.0
    level   = _level_for_score(overall)
    stars   = _LEVEL_STARS[level]
    star_n  = list(_LEVEL_STARS.values()).index(stars) + 1

    strengths  = [ms.metric.replace("_", " ").title()
                  for ms in metric_scores if ms.score >= 0.70]
    weaknesses = [ms.metric.replace("_", " ").title()
                  for ms in metric_scores if ms.score <= 0.35]

    top_weak = (
        min(metric_scores, key=lambda ms: ms.score).metric.replace("_", " ").title()
        if metric_scores else None
    )
    top_str = (
        max(metric_scores, key=lambda ms: ms.score).metric.replace("_", " ").title()
        if metric_scores else None
    )

    return ClassificationReport(
        level         = level,
        stars         = stars,
        star_count    = star_n,
        overall_score = overall,
        metric_scores = metric_scores,
        strengths     = strengths,
        weaknesses    = weaknesses,
        top_weakness  = top_weak,
        top_strength  = top_str,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_report(report: ClassificationReport) -> None:
    div = "─" * 52
    print(f"\n{div}")
    print(f"  FootballIQ — Skill Classification")
    print(div)
    print(f"  {report.stars}  {report.level.value}  (score: {report.overall_score:.2f})")
    print(div)
    print("  Metric Scores:")
    for ms in report.metric_scores:
        label = ms.metric.replace("_", " ").title()
        bar   = "█" * int(ms.score * 10) + "░" * (10 - int(ms.score * 10))
        print(f"    {label:<26} [{bar}]  {ms.score:.2f}  ({ms.level})")
    print(div)
    if report.strengths:
        print("  Strengths :", ", ".join(report.strengths[:3]))
    if report.weaknesses:
        print("  Focus on  :", ", ".join(report.weaknesses[:3]))
    if report.top_weakness:
        print(f"  Priority  : {report.top_weakness}")
    print(div)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FootballIQ Skill Classifier — 5-star rating system"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metrics",      type=str)
    group.add_argument("--metrics-file", type=str)
    args = parser.parse_args()

    try:
        if args.metrics:
            raw = json.loads(args.metrics)
        else:
            with open(args.metrics_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    report = classify_skill(raw)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
