#!/usr/bin/env python3
"""
Drill Recommender  (coach_engine)
==================================
Selects and prioritises training drills from the knowledge base.
Never invents drills — all recommendations are grounded in the
football_knowledge/ database and built-in drill catalogue.

Responsibilities:
  - Match player gaps to targeted drills
  - Filter by skill level (beginner drills ≠ advanced drills)
  - Rank drills by impact (highest-weight gap first)
  - Cap the number of drills per session (avoid overwhelming the player)
  - Return DrillRecommendation objects — never raw dicts

Usage::

    recommender = DrillRecommender()
    drills = recommender.recommend(profile, activity="shooting", max_drills=3)
    for d in drills:
        print(d.name, "—", d.instructions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from coach_engine.skill_classifier    import CoachSkillProfile, MetricScore
from coach_engine.terminology_adapter import TerminologyAdapter
from config.constants                  import MAX_DRILLS_PER_SESSION
from utils.logger                      import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Drill catalogue
# ---------------------------------------------------------------------------
# Structure: metric → level → List[drill_dicts]
# Each drill_dict: {name, instructions, duration, difficulty, coach_tip}

_DRILL_CATALOGUE: Dict[str, Dict[str, List[dict]]] = {

    "torso_lean": {
        "Beginner": [
            {
                "name":         "Wall Lean Drill",
                "instructions": "Stand 30 cm from a wall. Drive your kicking knee up "
                                "without letting your back touch the wall. "
                                "3 × 15 reps each leg.",
                "duration":     "10 min",
                "coach_tip":    "Your chest should be over the ball at contact. "
                                "Think: chin down, chest forward.",
            },
        ],
        "Intermediate": [
            {
                "name":         "Slow-Motion Contact Drill",
                "instructions": "Record yourself at 0.5× speed, pause at contact. "
                                "Check your torso angle against a vertical reference. "
                                "Adjust and repeat 20 times.",
                "duration":     "15 min",
                "coach_tip":    "Your non-kicking foot placement directly controls "
                                "your lean — position it beside, not behind, the ball.",
            },
        ],
        "Advanced": [
            {
                "name":         "Loaded Contact Precision Drill",
                "instructions": "Strike 30 balls at a target (< 2 m²) from 18 m. "
                                "After each miss, review lean angle on replay.",
                "duration":     "20 min",
                "coach_tip":    "At elite level, even 3° of extra lean costs 8–12% "
                                "of target accuracy over a full match.",
            },
        ],
    },

    "knee_dev": {
        "Beginner": [
            {
                "name":         "Lateral Band Walk",
                "instructions": "Place a resistance band above your knees. "
                                "Take 20 side steps each direction. 3 sets daily.",
                "duration":     "8 min",
                "coach_tip":    "Push your knee outward over your middle toe on every step.",
            },
        ],
        "Intermediate": [
            {
                "name":         "Single-Leg Squat Progression",
                "instructions": "3 × 8 single-leg squats each leg. "
                                "Focus on knee tracking over middle toe throughout.",
                "duration":     "10 min",
                "coach_tip":    "Activate your hip muscles with clamshells before "
                                "every session to pre-load knee stability.",
            },
        ],
        "Advanced": [
            {
                "name":         "Bulgarian Split Squat with Lateral Hold",
                "instructions": "Rear foot elevated, hold a 5–10 kg plate at chest. "
                                "3 × 10 each side. Record side-on to check knee alignment.",
                "duration":     "15 min",
                "coach_tip":    "The goal is < 0.15 knee deviation ratio under load — "
                                "this directly transfers to planting stability.",
            },
        ],
    },

    "gait_asymmetry": {
        "Beginner": [
            {
                "name":         "Single-Leg Bounds",
                "instructions": "Bound 20 m on one leg, then switch. "
                                "5 sets each side.",
                "duration":     "12 min",
                "coach_tip":    "Count your steps on each side — aim for equal "
                                "rhythm and equal push-off power.",
            },
        ],
        "Intermediate": [
            {
                "name":         "Single-Leg Hop and Hold",
                "instructions": "Hop forward on one foot and freeze on landing. "
                                "Hold for 2 seconds. 3 × 10 each side.",
                "duration":     "10 min",
                "coach_tip":    "Compare steadiness between your left and right side — "
                                "the weaker side needs more volume.",
            },
        ],
        "Advanced": [
            {
                "name":         "Asymmetric Plyometric Protocol",
                "instructions": "Alternating single-leg depth jumps from a 40 cm box. "
                                "Record contact time on each foot with a stopwatch. "
                                "Target: < 10% difference between sides.",
                "duration":     "15 min",
                "coach_tip":    "Stride asymmetry > 8% measurably increases "
                                "fatigue rate. This drill directly addresses it.",
            },
        ],
    },

    "leg_speed": {
        "Beginner": [
            {
                "name":         "Basic Plyometric Circuit",
                "instructions": "Jump squats × 10, tuck jumps × 10, "
                                "lateral hops × 10 each side — 3 rounds. 90 sec rest.",
                "duration":     "15 min",
                "coach_tip":    "Explosiveness starts from the hips. "
                                "Focus on quick, powerful push-offs.",
            },
        ],
        "Intermediate": [
            {
                "name":         "Resistance Band Sprints",
                "instructions": "Attach a light resistance band around your waist. "
                                "Sprint 15 m against resistance. 6 reps.",
                "duration":     "12 min",
                "coach_tip":    "Faster arm drive directly increases your stride rate — "
                                "keep your arms tight and fast.",
            },
        ],
        "Advanced": [
            {
                "name":         "Overspeed Sprint Protocol",
                "instructions": "Partner assists with bungee cord. Sprint 20 m at "
                                "105% of your max speed. 4 reps with full recovery.",
                "duration":     "20 min",
                "coach_tip":    "Overspeed training re-programs your nervous system "
                                "to accept faster leg turnover as normal.",
            },
        ],
    },

    "movement_consistency": {
        "Beginner": [
            {
                "name":         "Repetition Block Training",
                "instructions": "Perform the same skill 50 times at 70% effort. "
                                "Focus on identical mechanics every rep.",
                "duration":     "20 min",
                "coach_tip":    "Slow it down to speed it up — practise at 50% speed "
                                "until it feels automatic.",
            },
        ],
        "Intermediate": [
            {
                "name":         "Fatigue-State Training",
                "instructions": "Practise the target skill at the very end of a hard "
                                "session — that is when inconsistency appears.",
                "duration":     "10 min",
                "coach_tip":    "Video yourself once a week and compare your first "
                                "rep to your 30th — the difference shows where "
                                "consistency breaks down.",
            },
        ],
        "Advanced": [
            {
                "name":         "Pressure Consistency Protocol",
                "instructions": "Perform the skill with a defender applying light "
                                "pressure, then increase intensity each set. "
                                "Target: < 10% drop in quality under pressure.",
                "duration":     "20 min",
                "coach_tip":    "Elite consistency means mechanics survive both "
                                "fatigue AND defensive pressure simultaneously.",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class DrillRecommendation:
    """A single recommended training drill."""
    name:          str
    target_metric: str
    instructions:  str
    coach_tip:     str
    duration:      str
    difficulty:    str     # player's level
    priority:      int     # 1 = highest priority


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class DrillRecommender:
    """
    Selects targeted drills from the catalogue based on a skill profile.

    Parameters
    ----------
    adapter : TerminologyAdapter | None
        Terminology adapter — created automatically if not provided.
    max_drills : int
        Global cap on drills returned per session.
    """

    def __init__(
        self,
        adapter:    Optional[TerminologyAdapter] = None,
        max_drills: int = MAX_DRILLS_PER_SESSION,
    ) -> None:
        self._adapter    = adapter or TerminologyAdapter()
        self._max_drills = max_drills

    def recommend(
        self,
        profile:    CoachSkillProfile,
        activity:   str = "general",
        max_drills: Optional[int] = None,
    ) -> List[DrillRecommendation]:
        """
        Return a prioritised list of drill recommendations.

        Parameters
        ----------
        profile    : CoachSkillProfile
        activity   : str  — football action context (informational)
        max_drills : int | None — overrides default cap

        Returns
        -------
        List[DrillRecommendation] sorted by priority (1 = most important)
        """
        cap   = max_drills if max_drills is not None else self._max_drills
        level = profile.level

        # Sort gaps by score ascending (worst first), then by weight descending.
        gap_metrics = sorted(
            [ms for ms in profile.metric_scores if ms.score <= 0.50],
            key=lambda ms: (ms.score, -ms.weight),
        )

        recommendations: List[DrillRecommendation] = []
        priority = 1

        for ms in gap_metrics:
            if priority > cap:
                break

            drills = _DRILL_CATALOGUE.get(ms.metric, {}).get(level, [])
            if not drills:
                # Fall back to Intermediate if level-specific drills missing.
                drills = _DRILL_CATALOGUE.get(ms.metric, {}).get("Intermediate", [])
            if not drills:
                continue

            for drill_dict in drills[:1]:   # one drill per metric by default
                instr = self._adapter.adapt_text(drill_dict["instructions"], level)
                tip   = self._adapter.adapt_text(drill_dict["coach_tip"],   level)

                recommendations.append(DrillRecommendation(
                    name          = drill_dict["name"],
                    target_metric = ms.metric,
                    instructions  = instr,
                    coach_tip     = tip,
                    duration      = drill_dict["duration"],
                    difficulty    = level,
                    priority      = priority,
                ))
                priority += 1
                if priority > cap:
                    break

        log.debug(
            "DrillRecommender: %d drills for %s / %s",
            len(recommendations), level, activity,
        )
        return recommendations
