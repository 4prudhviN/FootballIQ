#!/usr/bin/env python3
"""
Skill Classifier  (coach_engine)
=================================
Classifies a player's skill level from session metrics.
Wraps and extends the root-level skill_classifier.py with
coach-engine-specific output types.

Responsibilities:
  - Accept session metrics from the analysis pipeline
  - Return a CoachSkillProfile with level + per-metric scores
  - Identify the player's top strengths and biggest gaps
  - Never use the LLM — purely deterministic scoring

Usage::

    from coach_engine.skill_classifier import CoachSkillClassifier
    profile = CoachSkillClassifier().classify(metrics_dict)
    print(profile.level)          # "Intermediate"
    print(profile.top_gap)        # "Knee stability"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.thresholds import SKILL_THRESHOLDS, SKILL_WEIGHTS, ADVANCED_SCORE_THRESHOLD, BEGINNER_SCORE_THRESHOLD
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    """Score for a single performance metric."""
    metric:       str
    raw_value:    float
    score:        float      # 0.0–1.0 (1.0 = Advanced)
    level_label:  str        # "Advanced", "Intermediate", "Beginner"
    weight:       float


@dataclass
class CoachSkillProfile:
    """Full skill profile produced by CoachSkillClassifier."""
    level:          str                     # "Beginner" | "Intermediate" | "Advanced"
    overall_score:  float                   # 0.0–1.0
    metric_scores:  List[MetricScore]       = field(default_factory=list)
    strengths:      List[str]               = field(default_factory=list)
    gaps:           List[str]               = field(default_factory=list)
    top_gap:        Optional[str]           = None   # single most critical area
    top_strength:   Optional[str]           = None

    def to_dict(self) -> dict:
        return {
            "level":         self.level,
            "overall_score": round(self.overall_score, 3),
            "strengths":     self.strengths,
            "gaps":          self.gaps,
            "top_gap":       self.top_gap,
            "top_strength":  self.top_strength,
            "metric_scores": {
                ms.metric: round(ms.score, 3) for ms in self.metric_scores
            },
        }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class CoachSkillClassifier:
    """
    Deterministic skill classifier for the coaching engine.

    Parameters
    ----------
    strength_threshold : float
        Score at or above which a metric is considered a strength. Default 0.70.
    gap_threshold : float
        Score at or below which a metric is flagged as a gap. Default 0.35.
    """

    def __init__(
        self,
        strength_threshold: float = 0.70,
        gap_threshold:      float = 0.35,
    ) -> None:
        self.strength_threshold = strength_threshold
        self.gap_threshold      = gap_threshold

    def classify(self, metrics: Dict[str, float]) -> CoachSkillProfile:
        """
        Classify player skill level from a metric dictionary.

        Parameters
        ----------
        metrics : dict[str, float]
            Keys match SKILL_THRESHOLDS (e.g. "torso_lean", "knee_dev").
            Missing keys are skipped; score is normalised by used weight.

        Returns
        -------
        CoachSkillProfile
        """
        if not metrics:
            return CoachSkillProfile(
                level="Beginner", overall_score=0.0,
                top_gap="Unknown", top_strength=None,
            )

        metric_scores: List[MetricScore] = []
        total_weight   = 0.0
        weighted_score = 0.0

        for metric, (adv_thresh, beg_thresh, higher_is_better) in SKILL_THRESHOLDS.items():
            raw = metrics.get(metric)
            if raw is None:
                continue

            score = self._score_metric(raw, adv_thresh, beg_thresh, higher_is_better)
            weight = SKILL_WEIGHTS.get(metric, 0.05)
            label = (
                "Advanced"     if score >= 0.70 else
                "Intermediate" if score >= 0.35 else
                "Beginner"
            )

            metric_scores.append(MetricScore(
                metric      = metric,
                raw_value   = raw,
                score       = round(score, 3),
                level_label = label,
                weight      = weight,
            ))
            weighted_score += score * weight
            total_weight   += weight

        overall = (weighted_score / total_weight) if total_weight > 0 else 0.0
        overall = round(overall, 3)

        level = (
            "Advanced"     if overall >= ADVANCED_SCORE_THRESHOLD else
            "Intermediate" if overall >= BEGINNER_SCORE_THRESHOLD else
            "Beginner"
        )

        strengths = [
            ms.metric.replace("_", " ").title()
            for ms in metric_scores
            if ms.score >= self.strength_threshold
        ]
        gaps = [
            ms.metric.replace("_", " ").title()
            for ms in metric_scores
            if ms.score <= self.gap_threshold
        ]

        # Top gap = lowest-scoring metric weighted by importance.
        bottom = sorted(
            [ms for ms in metric_scores if ms.score <= self.gap_threshold],
            key=lambda ms: ms.score * ms.weight,
        )
        top_gap = bottom[0].metric.replace("_", " ").title() if bottom else None

        # Top strength = highest-scoring metric.
        top_strength_ms = max(metric_scores, key=lambda ms: ms.score, default=None)
        top_strength = top_strength_ms.metric.replace("_", " ").title() if top_strength_ms else None

        profile = CoachSkillProfile(
            level         = level,
            overall_score = overall,
            metric_scores = metric_scores,
            strengths     = strengths,
            gaps          = gaps,
            top_gap       = top_gap,
            top_strength  = top_strength,
        )

        log.debug("CoachSkillClassifier: level=%s score=%.3f", level, overall)
        return profile

    @staticmethod
    def _score_metric(
        value:            float,
        adv_thresh:       float,
        beg_thresh:       float,
        higher_is_better: bool,
    ) -> float:
        """Map a metric value to a [0, 1] score."""
        if higher_is_better:
            if value >= adv_thresh:
                return 1.0
            if value <= beg_thresh:
                return 0.0
            return (value - beg_thresh) / (adv_thresh - beg_thresh)
        else:
            if value <= adv_thresh:
                return 1.0
            if value >= beg_thresh:
                return 0.0
            return 1.0 - (value - adv_thresh) / (beg_thresh - adv_thresh)
