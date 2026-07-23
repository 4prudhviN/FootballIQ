#!/usr/bin/env python3
"""
Training Plan Generator
=======================
Builds a structured multi-session training plan from focus areas and drills.

Example output:
  Session 1 (Tuesday): Passing accuracy + First touch  — 45 min
  Session 2 (Thursday): Balance while shooting          — 30 min
  Session 3 (Saturday): Full review + weak side work   — 60 min

Usage::

    gen  = TrainingPlanGenerator()
    plan = gen.generate(profile, focus_areas, drills)
    for session in plan.sessions:
        print(session.label, session.duration_min, "min")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from recommendation_engine.priority_selector import FocusArea
from coach_engine.drill_recommender         import DrillRecommendation
from utils.logger                            import get_logger

log = get_logger(__name__)

_SESSIONS_PER_WEEK: Dict[str, int] = {
    "Beginner":     2,
    "Intermediate": 3,
    "Advanced":     4,
}

_SESSION_DURATION: Dict[str, int] = {
    "Beginner":     30,   # minutes
    "Intermediate": 45,
    "Advanced":     60,
}


@dataclass
class TrainingSession:
    """One training session within a plan."""
    session_number: int
    label:          str                    # e.g. "Session 1 — Passing & Touch"
    focus_areas:    List[str]              # metric labels
    drills:         List[DrillRecommendation]
    duration_min:   int
    notes:          str = ""


@dataclass
class TrainingPlan:
    """A complete training plan for one week."""
    player_level:    str
    sessions:        List[TrainingSession] = field(default_factory=list)
    total_sessions:  int = 0
    total_duration_min: int = 0
    focus_summary:   List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "playerLevel":      self.player_level,
            "totalSessions":    self.total_sessions,
            "totalDurationMin": self.total_duration_min,
            "focusSummary":     self.focus_summary,
            "sessions": [
                {
                    "session":    s.session_number,
                    "label":      s.label,
                    "focusAreas": s.focus_areas,
                    "drills":     [d.name for d in s.drills],
                    "durationMin": s.duration_min,
                    "notes":      s.notes,
                }
                for s in self.sessions
            ],
        }


class TrainingPlanGenerator:
    """
    Generates a structured weekly training plan.

    Distributes focus areas and drills across multiple sessions,
    scaled to the player's level and available time.
    """

    def generate(
        self,
        player_level: str,
        focus_areas:  List[FocusArea],
        drills:       List[DrillRecommendation],
        sessions_override: Optional[int] = None,
    ) -> TrainingPlan:
        """
        Build a training plan.

        Parameters
        ----------
        player_level : str
        focus_areas  : from PrioritySelector.select()
        drills       : from DrillRecommender.recommend()
        sessions_override : override the default sessions-per-week

        Returns
        -------
        TrainingPlan
        """
        n_sessions   = sessions_override or _SESSIONS_PER_WEEK.get(player_level, 3)
        duration_min = _SESSION_DURATION.get(player_level, 45)

        sessions: List[TrainingSession] = []
        total_duration = 0

        # Distribute focus areas across sessions.
        for i in range(n_sessions):
            # Pick focus areas for this session (rotate through the list).
            session_areas = [
                focus_areas[j].label
                for j in range(len(focus_areas))
                if j % n_sessions == i
            ]
            if not session_areas and focus_areas:
                session_areas = [focus_areas[0].label]

            # Pick drills for this session (rotate through the drill list).
            session_drills = [
                drills[j]
                for j in range(len(drills))
                if j % n_sessions == i
            ]

            label = f"Session {i + 1} — {' & '.join(session_areas[:2])}"
            notes = self._session_notes(i + 1, n_sessions, player_level)

            sessions.append(TrainingSession(
                session_number = i + 1,
                label          = label,
                focus_areas    = session_areas,
                drills         = session_drills,
                duration_min   = duration_min,
                notes          = notes,
            ))
            total_duration += duration_min

        focus_summary = [a.label for a in focus_areas]

        plan = TrainingPlan(
            player_level       = player_level,
            sessions           = sessions,
            total_sessions     = n_sessions,
            total_duration_min = total_duration,
            focus_summary      = focus_summary,
        )

        log.debug("TrainingPlanGenerator: %d sessions for %s player", n_sessions, player_level)
        return plan

    @staticmethod
    def _session_notes(session_num: int, total: int, level: str) -> str:
        if session_num == 1:
            return "Start at 70% intensity. Focus on quality of movement, not speed."
        if session_num == total:
            return "Final session of the week. Include a full review at match pace."
        return "Build on session 1. Introduce slight pressure in drills."
