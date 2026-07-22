#!/usr/bin/env python3
"""
Player Schema
=============
Typed data contracts describing a player and their skill profile.

Every module that deals with player identity or classification
must use these types — never raw dicts or bare strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SkillLevel(str, Enum):
    BEGINNER     = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED     = "Advanced"


class FootPreference(str, Enum):
    RIGHT    = "right"
    LEFT     = "left"
    BALANCED = "balanced"


class PlayerPosition(str, Enum):
    GOALKEEPER = "goalkeeper"
    DEFENDER   = "defender"
    MIDFIELDER = "midfielder"
    FORWARD    = "forward"
    UNKNOWN    = "unknown"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

@dataclass
class PlayerProfile:
    """
    Static player identity data.
    Populated at session start from user input or profile store.
    """
    player_id:       str
    name:            str
    age:             Optional[int]           = None
    position:        PlayerPosition          = PlayerPosition.UNKNOWN
    foot_preference: FootPreference          = FootPreference.RIGHT
    team:            Optional[str]           = None

    def to_dict(self) -> dict:
        return {
            "player_id":       self.player_id,
            "name":            self.name,
            "age":             self.age,
            "position":        self.position.value,
            "foot_preference": self.foot_preference.value,
            "team":            self.team,
        }


@dataclass
class SkillScore:
    """
    Per-metric skill score produced by skill_classifier.py.
    Value is in [0.0, 1.0] — closer to 1.0 = more advanced.
    """
    metric:  str
    score:   float            # 0.0 – 1.0
    display: str              # e.g. "0.82"


@dataclass
class PlayerSkillProfile:
    """
    Dynamic skill profile derived from video analysis.
    Produced by skill_classifier.py.
    """
    level:          SkillLevel
    overall_score:  float                    # 0.0 – 1.0
    metric_scores:  List[SkillScore]         = field(default_factory=list)
    strengths:      List[str]                = field(default_factory=list)
    weaknesses:     List[str]                = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level":          self.level.value,
            "overall_score":  round(self.overall_score, 3),
            "metric_scores":  {s.metric: s.score for s in self.metric_scores},
            "strengths":      self.strengths,
            "weaknesses":     self.weaknesses,
        }
