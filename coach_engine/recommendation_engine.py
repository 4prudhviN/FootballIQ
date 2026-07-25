#!/usr/bin/env python3
"""
Recommendation Engine
=====================
Maps detected mistakes to training drills and coach tips.

Input:  MistakeDetectionResult (from MistakeDetector)
Output: List[Recommendation]
          ├── priority         (1 = most important)
          ├── problem          (e.g. "Body leaning backwards")
          ├── drill_name       (e.g. "Wall Passing Drill")
          ├── drill_instructions
          ├── coach_tip        (e.g. "Keep your chest over the ball.")
          └── duration

Example:
  Problem   : Body leaning backwards
  Drill     : Wall Passing Drill
  Coach Tip : Keep your chest over the ball.

No AI. Looks up the knowledge base.

Usage::

    engine = RecommendationEngine()
    recs   = engine.recommend(mistake_result, level="Intermediate")
    for r in recs:
        print(f"Priority {r.priority}: {r.problem}")
        print(f"  Drill: {r.drill_name}")
        print(f"  Tip:   {r.coach_tip}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from coach_engine.mistake_detector import MistakeDetectionResult, DetectedProblem
from utils.logger import get_logger

log = get_logger(__name__)

_KB_DRILLS_DIR = Path(__file__).parent.parent / "football_knowledge" / "drills"


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """A single drill recommendation addressing one detected problem."""
    priority:           int
    problem:            str
    drill_name:         str
    drill_instructions: str
    coach_tip:          str
    duration:           str
    severity:           str
    drill_ref:          Optional[str] = None


@dataclass
class RecommendationReport:
    """Full output of the RecommendationEngine."""
    activity:      str
    level:         str
    recommendations: List[Recommendation]
    top_priority:  Optional[Recommendation]

    def to_dict(self) -> dict:
        return {
            "activity": self.activity,
            "level":    self.level,
            "top_priority": {
                "problem":   self.top_priority.problem,
                "drill":     self.top_priority.drill_name,
                "coach_tip": self.top_priority.coach_tip,
            } if self.top_priority else None,
            "recommendations": [
                {
                    "priority":     r.priority,
                    "problem":      r.problem,
                    "drill":        r.drill_name,
                    "instructions": r.drill_instructions,
                    "coach_tip":    r.coach_tip,
                    "duration":     r.duration,
                    "severity":     r.severity,
                }
                for r in self.recommendations
            ],
        }

    def summary_lines(self) -> List[str]:
        lines = []
        for r in self.recommendations:
            lines.append(f"  [{r.priority}] {r.problem}")
            lines.append(f"      Drill: {r.drill_name}  ({r.duration})")
            lines.append(f"      Tip:   {r.coach_tip}")
        return lines


# ---------------------------------------------------------------------------
# Built-in drill lookup
# ---------------------------------------------------------------------------
# Maps drill_ref → {name, instructions, coach_tip, duration}

_BUILT_IN_DRILLS: Dict[str, Dict[str, str]] = {

    "beg_passing_wall": {
        "name":         "Wall Passing Drill",
        "instructions": "Stand 5m from a wall. Pass at a 50cm target. Receive rebound and repeat. 3×20 each foot.",
        "coach_tip":    "Keep your chest over the ball. Plant foot beside it, not behind.",
        "duration":     "10 min",
    },
    "int_passing_triangle": {
        "name":         "Triangle Passing Circuit",
        "instructions": "3 cones 8m apart. Pass and move. One-touch rhythm. 5 min each direction.",
        "coach_tip":    "Decide your pass before the ball arrives. Open your body on receive.",
        "duration":     "15 min",
    },
    "adv_scanning_rondo": {
        "name":         "Pre-Scan Rondo",
        "instructions": "4v1 in 6m box. Rule: scan twice before each receive. Rotate defender every 90 sec.",
        "coach_tip":    "The best players make decisions before the ball arrives.",
        "duration":     "20 min",
    },
    "int_weak_foot_passing": {
        "name":         "Weak Foot Isolation",
        "instructions": "Weaker foot only: pass against wall from 6m. 100 passes minimum.",
        "coach_tip":    "Your weaker foot only improves through dedicated volume. No shortcuts.",
        "duration":     "10 min",
    },
    "beg_shooting_stationary": {
        "name":         "Stationary Shot Technique",
        "instructions": "Ball on ground. 3-step approach, laces contact. 20 strikes each foot.",
        "coach_tip":    "Chin down, chest over the ball at contact. Don't look up early.",
        "duration":     "10 min",
    },
    "int_shooting_moving": {
        "name":         "Shoot After Receive",
        "instructions": "Receive pass, control, shoot in 2 touches. 15 attempts each foot.",
        "coach_tip":    "Set your body before the ball arrives — don't adjust after.",
        "duration":     "15 min",
    },
    "adv_shooting_pressure": {
        "name":         "Shooting Under Pressure",
        "instructions": "Receive with defender closing from behind. First-time or 2-touch shot. 20 attempts.",
        "coach_tip":    "Choose your corner early. Placement beats power.",
        "duration":     "20 min",
    },
    "beg_dribbling_cones": {
        "name":         "Cone Slalom Dribble",
        "instructions": "6 cones 1.5m apart. Dribble through, alternating feet. 5 runs each foot.",
        "coach_tip":    "Ball within one touch of your foot at all times. Small, controlled touches.",
        "duration":     "10 min",
    },
    "int_dribbling_1v1": {
        "name":         "1v1 Dribbling Box",
        "instructions": "5m x 5m box. Dribble to touch far cone past defender. 2-min rounds.",
        "coach_tip":    "Use a feint before accelerating. Commit to your move.",
        "duration":     "15 min",
    },
    "adv_dribbling_overspeed": {
        "name":         "Overspeed Dribble Sprint",
        "instructions": "Light resistance band. Dribble 20m at full speed. Remove band and repeat. 6 reps.",
        "coach_tip":    "Push ball further than feels comfortable. Trust your pace to recover it.",
        "duration":     "15 min",
    },
    "beg_balance_single_leg": {
        "name":         "Single-Leg Balance Hold",
        "instructions": "Stand on one leg 30 sec. Switch. Eyes closed when comfortable. 3 sets each.",
        "coach_tip":    "Stay on ball of foot. Tight core. Don't let your hip drop.",
        "duration":     "8 min",
    },
    "beg_receiving_chest": {
        "name":         "Chest Control from Toss",
        "instructions": "Partner tosses from 3m. Control with chest, let drop to feet. 3×15.",
        "coach_tip":    "Cushion the ball — let it come to you, don't push into it.",
        "duration":     "10 min",
    },
    "int_movement_change_dir": {
        "name":         "Change of Direction Speed Circuit",
        "instructions": "10m x 5m rectangle. Sprint long side, shuffle short side. 8 reps at 80%.",
        "coach_tip":    "Drive your arms — faster arms means faster feet.",
        "duration":     "12 min",
    },

    # Fallback for unknown drill refs
    "_default": {
        "name":         "Fundamental Repetition Drill",
        "instructions": "Perform the target skill 50 times at 70% intensity. Focus on identical mechanics each rep.",
        "coach_tip":    "Slow is smooth, smooth is fast. Master the basics first.",
        "duration":     "15 min",
    },
}

# Problem → direct coach tip (used when drill lookup fails)
_PROBLEM_TIPS: Dict[str, str] = {
    "leaning backwards at contact":         "Keep your chest over the ball. Drive it forward at contact.",
    "plant foot too far from the ball":      "Plant your foot beside the ball — not behind or in front of it.",
    "ball struck too hard — losing direction": "Match your pass power to the distance. Shorter passes need less force.",
    "incorrect foot surface used":           "Inside foot for accuracy. Laces for power. Never toe-poke.",
    "no pre-scan before receiving":          "Look up twice before the ball arrives. Decide early.",
    "no follow-through after contact":       "Drive through the ball. Foot must finish pointing at the target.",
    "ball too far from feet":                "Smaller touches. Keep ball within arm's reach at all times.",
    "head down — no awareness of defenders": "Every 2–3 touches, look up. Feel the ball, don't watch it.",
    "knee collapsing inward on plant foot":  "Push knee over middle toe when you plant. Activate your glutes.",
    "unstable centre of gravity during movement": "Stay on balls of feet. Brace your core during all movement.",
}


# ---------------------------------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """
    Maps detected mistakes to training drills and coach tips.

    Reads the football_knowledge/drills/ database for level-specific drills.
    Falls back to built-in drill table when knowledge base entry not found.

    No AI. Pure lookup.
    """

    def __init__(self, max_recommendations: int = 5) -> None:
        self.max_recs = max_recommendations

    def recommend(
        self,
        mistakes: MistakeDetectionResult,
        level:    str = "Beginner",
    ) -> RecommendationReport:
        """
        Generate drill recommendations from detected mistakes.

        Parameters
        ----------
        mistakes : MistakeDetectionResult — from MistakeDetector
        level    : str — player skill level

        Returns
        -------
        RecommendationReport
        """
        # Load level-specific drills from knowledge base.
        level_drills = self._load_level_drills(level)

        recommendations: List[Recommendation] = []
        priority = 1

        for problem in mistakes.problems[:self.max_recs]:
            drill_data = self._lookup_drill(problem.drill_ref, level_drills)
            coach_tip  = self._get_tip(problem)

            recommendations.append(Recommendation(
                priority           = priority,
                problem            = problem.cause,
                drill_name         = drill_data["name"],
                drill_instructions = drill_data["instructions"],
                coach_tip          = coach_tip,
                duration           = drill_data["duration"],
                severity           = problem.severity,
                drill_ref          = problem.drill_ref,
            ))
            priority += 1

        top = recommendations[0] if recommendations else None

        log.debug(
            "RecommendationEngine: %d recommendations for %s/%s",
            len(recommendations), mistakes.activity, level,
        )

        return RecommendationReport(
            activity         = mistakes.activity,
            level            = level,
            recommendations  = recommendations,
            top_priority     = top,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _load_level_drills(level: str) -> Dict[str, Any]:
        """Load level-specific drills from knowledge base."""
        filename = {
            "Beginner":     "beginner.json",
            "Developing":   "beginner.json",
            "Intermediate": "intermediate.json",
            "Advanced":     "advanced.json",
            "Elite":        "advanced.json",
        }.get(level, "beginner.json")

        path = _KB_DRILLS_DIR / filename
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {d["id"]: d for d in data.get("drills", [])}
        except Exception:
            return {}

    def _lookup_drill(
        self,
        drill_ref:    Optional[str],
        level_drills: Dict[str, Any],
    ) -> Dict[str, str]:
        """Find drill data — KB first, then built-in table, then default."""
        if drill_ref:
            # Try knowledge base first.
            kb_drill = level_drills.get(drill_ref)
            if kb_drill:
                return {
                    "name":         kb_drill.get("name", "Drill"),
                    "instructions": kb_drill.get("instructions", ""),
                    "coach_tip":    kb_drill.get("coach_tip", ""),
                    "duration":     kb_drill.get("duration", "10 min"),
                }
            # Try built-in table.
            built_in = _BUILT_IN_DRILLS.get(drill_ref)
            if built_in:
                return built_in

        return _BUILT_IN_DRILLS["_default"]

    @staticmethod
    def _get_tip(problem: DetectedProblem) -> str:
        """Get coach tip — from problem.correction or tip table."""
        if problem.correction:
            return problem.correction
        key = problem.cause.lower().strip()
        return _PROBLEM_TIPS.get(key, "Focus on technique first, then add speed.")
