#!/usr/bin/env python3
"""
Weekly Plan Generator
=====================
Assigns training sessions to specific days of the week with
rest and recovery scheduling built in.

Example output:
  Monday    — Rest / light recovery
  Tuesday   — Session 1: Passing accuracy + First touch  (45 min)
  Wednesday — Rest
  Thursday  — Session 2: Balance while shooting           (45 min)
  Friday    — Light stretching / mobility
  Saturday  — Session 3: Full review + weak side work     (60 min)
  Sunday    — Rest

Usage::

    gen  = WeeklyPlanGenerator()
    plan = gen.generate(training_plan, level="Intermediate")
    for day in plan.days:
        print(day.day_name, "—", day.activity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from recommendation_engine.training_plan import TrainingPlan, TrainingSession
from utils.logger                         import get_logger

log = get_logger(__name__)

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Day templates per number of sessions.
_SCHEDULE_TEMPLATES = {
    2: [None, 0, None, 1, None, None, None],       # Tue, Thu
    3: [None, 0, None, 1, None, 2, None],           # Tue, Thu, Sat
    4: [None, 0, None, 1, 2, None, 3],              # Tue, Thu, Fri, Sun
}

_REST_ACTIVITIES = {
    "full_rest":  "Rest — full recovery day.",
    "light":      "Light stretching or 20-min walk. No football.",
    "mobility":   "Mobility work — hip flexors, hamstrings, calves (15 min).",
}


@dataclass
class WeekDay:
    """One day in the weekly plan."""
    day_name:    str
    day_index:   int        # 0=Monday
    is_training: bool
    activity:    str        # human-readable description
    session:     Optional[TrainingSession] = None
    duration_min: int = 0


@dataclass
class WeeklyPlan:
    """A full 7-day plan with training and recovery days."""
    player_level: str
    days:         List[WeekDay] = field(default_factory=list)
    training_days: int = 0
    rest_days:     int = 0

    def to_dict(self) -> dict:
        return {
            "playerLevel":  self.player_level,
            "trainingDays": self.training_days,
            "restDays":     self.rest_days,
            "days": [
                {
                    "day":         d.day_name,
                    "isTraining":  d.is_training,
                    "activity":    d.activity,
                    "durationMin": d.duration_min,
                    "session":     d.session.label if d.session else None,
                }
                for d in self.days
            ],
        }

    def summary(self) -> str:
        lines = ["Weekly Training Plan", "=" * 40]
        for d in self.days:
            icon = "⚽" if d.is_training else "💤"
            lines.append(f"  {icon}  {d.day_name:<12} — {d.activity}")
        return "\n".join(lines)


class WeeklyPlanGenerator:
    """Assigns training sessions to days of the week."""

    def generate(
        self,
        training_plan: TrainingPlan,
        start_day:     int = 0,   # 0=Monday
    ) -> WeeklyPlan:
        """
        Build a 7-day schedule from a TrainingPlan.

        Parameters
        ----------
        training_plan : TrainingPlan
        start_day     : int — 0=Monday (week start)

        Returns
        -------
        WeeklyPlan
        """
        n         = training_plan.total_sessions
        template  = _SCHEDULE_TEMPLATES.get(n, _SCHEDULE_TEMPLATES[3])
        sessions  = training_plan.sessions
        level     = training_plan.player_level

        days: List[WeekDay] = []
        training_count = 0
        rest_count     = 0

        for i, slot in enumerate(template):
            day_idx  = (start_day + i) % 7
            day_name = _DAYS[day_idx]

            if slot is not None and slot < len(sessions):
                session  = sessions[slot]
                activity = f"{session.label}  ({session.duration_min} min)"
                days.append(WeekDay(
                    day_name    = day_name,
                    day_index   = day_idx,
                    is_training = True,
                    activity    = activity,
                    session     = session,
                    duration_min = session.duration_min,
                ))
                training_count += 1
            else:
                # Assign rest type based on position in week.
                rest = self._rest_activity(i, template, level)
                days.append(WeekDay(
                    day_name    = day_name,
                    day_index   = day_idx,
                    is_training = False,
                    activity    = rest,
                ))
                rest_count += 1

        plan = WeeklyPlan(
            player_level  = level,
            days          = days,
            training_days = training_count,
            rest_days     = rest_count,
        )

        log.debug("WeeklyPlanGenerator: %d training / %d rest days", training_count, rest_count)
        return plan

    @staticmethod
    def _rest_activity(day_pos: int, template: list, level: str) -> str:
        """Choose an appropriate rest activity for the day position."""
        # Day after training → mobility work.
        if day_pos > 0 and template[day_pos - 1] is not None:
            return _REST_ACTIVITIES["mobility"]
        # Day before training → light stretching.
        if day_pos < len(template) - 1 and template[day_pos + 1] is not None:
            return _REST_ACTIVITIES["light"]
        return _REST_ACTIVITIES["full_rest"]
