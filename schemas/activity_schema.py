#!/usr/bin/env python3
"""
Activity Schema
===============
Typed data contracts for detected football activities and
their associated per-action metrics.

Every module that classifies or measures football actions
must use these types — never raw dicts or bare strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class FootballAction(str, Enum):
    PASSING     = "passing"
    DRIBBLING   = "dribbling"
    SHOOTING    = "shooting"
    GOALKEEPING = "goalkeeping"
    DEFENDING   = "defending"
    MOVEMENT    = "movement"
    UNKNOWN     = "unknown"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

@dataclass
class DetectedActivity:
    """
    A single football activity identified by the activity detector.
    """
    action:     FootballAction
    confidence: float                        # 0.0 – 1.0
    evidence:   List[str] = field(default_factory=list)  # reasons for classification

    def to_dict(self) -> dict:
        return {
            "action":     self.action.value,
            "confidence": round(self.confidence, 3),
            "evidence":   self.evidence,
        }


@dataclass
class ActivityMetric:
    """
    A single numeric metric for one football action.
    """
    label:   str    # e.g. "Shot Velocity"
    value:   float  # raw numeric value
    display: str    # formatted display string, e.g. "88 km/h"
    unit:    str    # e.g. "km/h", "%", "°"


@dataclass
class ActionMetrics:
    """
    All metrics calculated for a single football action.
    Produced by the metrics/ calculators.
    """
    action:  FootballAction
    metrics: List[ActivityMetric] = field(default_factory=list)

    def to_display_dict(self) -> Dict[str, str]:
        """Return {label: display} — consumed by the frontend."""
        return {m.label: m.display for m in self.metrics}

    def to_numeric_dict(self) -> Dict[str, float]:
        """Return {label: float} — consumed by skill_classifier.py."""
        return {m.label: m.value for m in self.metrics}

    def to_dict(self) -> dict:
        return {
            "action":  self.action.value,
            "metrics": self.to_display_dict(),
        }


@dataclass
class ActivityDetectionResult:
    """
    Full output of the activity detection stage.
    Returned by pipeline/activity_detector.py.
    """
    detected:    List[DetectedActivity]     = field(default_factory=list)
    primary:     Optional[FootballAction]   = None   # highest-confidence action
    action_names: List[str]                 = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "detected":     [a.to_dict() for a in self.detected],
            "primary":      self.primary.value if self.primary else None,
            "action_names": self.action_names,
        }
