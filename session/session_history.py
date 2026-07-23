#!/usr/bin/env python3
"""
Session History
===============
Loads and compares multiple sessions to track player progress over time.

Responsibilities:
  - Load all sessions for a player (sorted newest first)
  - Compare any two sessions (delta in metrics, level changes)
  - Generate a progress timeline: list of (date, level, score) tuples
  - Identify the player's most improved and most consistent metrics

Usage::

    history = SessionHistory()
    sessions = history.all_sessions()

    if len(sessions) >= 2:
        comparison = history.compare(sessions[-1], sessions[0])
        print(comparison["overall_trend"])  # "improving"

    timeline = history.progress_timeline(sessions)
    for point in timeline:
        print(point["date"], point["level"], point["score"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from session.session_manager import Session, SessionManager
from utils.logger            import get_logger

log = get_logger(__name__)

_IMPROVE_THRESHOLD  = 0.03
_REGRESS_THRESHOLD  = -0.03


@dataclass
class ProgressPoint:
    """A single point on the player's progress timeline."""
    session_id:    str
    date:          str
    player_level:  str
    primary_activity: Optional[str]
    torso_lean:    float
    knee_stability: float
    gait_symmetry: float
    warning_count: int


class SessionHistory:
    """
    Loads and analyses the full history of sessions.

    Parameters
    ----------
    manager : SessionManager | None
    """

    def __init__(self, manager: Optional[SessionManager] = None) -> None:
        self._manager = manager or SessionManager()

    def all_sessions(self) -> List[Session]:
        """
        Load all saved sessions, newest first.
        Skips any sessions that fail to load.
        """
        ids      = self._manager.list_sessions()
        sessions: List[Session] = []
        for sid in ids:
            s = self._manager.load(sid)
            if s:
                sessions.append(s)
        log.debug("SessionHistory: loaded %d sessions", len(sessions))
        return sessions

    def latest(self) -> Optional[Session]:
        """Return the most recent session, or None."""
        sessions = self.all_sessions()
        return sessions[0] if sessions else None

    def compare(
        self,
        older: Session,
        newer: Session,
    ) -> Dict[str, Any]:
        """
        Compare two sessions and return a progress dict.

        Returns
        -------
        dict with keys: overall_trend, level_changed, metric_deltas,
        most_improved, most_regressed, summary
        """
        metrics = {
            "torso_lean":     (older.torso_lean,     newer.torso_lean),
            "knee_stability": (older.knee_stability,  newer.knee_stability),
            "gait_symmetry":  (older.gait_symmetry,   newer.gait_symmetry),
        }

        deltas: Dict[str, float] = {}
        for name, (old_val, new_val) in metrics.items():
            deltas[name] = round(new_val - old_val, 2)

        # For torso_lean: lower is better, so negate for trend.
        # For knee_stability and gait_symmetry: higher is better.
        trend_score = (
            -deltas["torso_lean"]
            + deltas["knee_stability"]
            + deltas["gait_symmetry"]
        ) / 3.0

        if trend_score >= _IMPROVE_THRESHOLD:
            overall_trend = "improving"
        elif trend_score <= _REGRESS_THRESHOLD:
            overall_trend = "regressing"
        else:
            overall_trend = "stable"

        level_changed = older.player_level != newer.player_level

        # Most improved: metric with largest positive delta (good direction).
        signed = {
            "torso_lean":     -deltas["torso_lean"],
            "knee_stability":  deltas["knee_stability"],
            "gait_symmetry":   deltas["gait_symmetry"],
        }
        most_improved  = max(signed, key=signed.get) if any(v > 0 for v in signed.values()) else None
        most_regressed = min(signed, key=signed.get) if any(v < 0 for v in signed.values()) else None

        summary = self._build_summary(overall_trend, level_changed, newer.player_level)

        return {
            "older_id":        older.id,
            "newer_id":        newer.id,
            "overall_trend":   overall_trend,
            "level_changed":   level_changed,
            "prev_level":      older.player_level,
            "curr_level":      newer.player_level,
            "metric_deltas":   {k: {"delta": v, "trend": "up" if signed[k] > 0 else "down" if signed[k] < 0 else "stable"} for k, v in deltas.items()},
            "most_improved":   most_improved.replace("_", " ").title() if most_improved else None,
            "most_regressed":  most_regressed.replace("_", " ").title() if most_regressed else None,
            "summary":         summary,
        }

    def progress_timeline(self, sessions: Optional[List[Session]] = None) -> List[ProgressPoint]:
        """
        Build a chronological progress timeline from all sessions.

        Returns list sorted oldest → newest.
        """
        if sessions is None:
            sessions = self.all_sessions()

        # Sort oldest first for timeline.
        sorted_sessions = sorted(sessions, key=lambda s: s.created_at)

        timeline: List[ProgressPoint] = []
        for s in sorted_sessions:
            timeline.append(ProgressPoint(
                session_id       = s.id,
                date             = s.created_at,
                player_level     = s.player_level,
                primary_activity = s.primary_activity,
                torso_lean       = s.torso_lean,
                knee_stability   = s.knee_stability,
                gait_symmetry    = s.gait_symmetry,
                warning_count    = len(s.warnings),
            ))

        return timeline

    def most_common_activity(self, sessions: Optional[List[Session]] = None) -> Optional[str]:
        """Return the most frequently detected primary activity across sessions."""
        if sessions is None:
            sessions = self.all_sessions()
        counts: Dict[str, int] = {}
        for s in sessions:
            if s.primary_activity:
                counts[s.primary_activity] = counts.get(s.primary_activity, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def persistent_weaknesses(self, sessions: Optional[List[Session]] = None) -> List[str]:
        """
        Return metrics that appear in ai_feedback weaknesses in > 50% of sessions.
        These are the player's recurring problem areas.
        """
        if sessions is None:
            sessions = self.all_sessions()
        if not sessions:
            return []

        counts: Dict[str, int] = {}
        for s in sessions:
            for weakness in s.ai_feedback.get("weaknesses", []):
                counts[weakness] = counts.get(weakness, 0) + 1

        threshold = len(sessions) * 0.5
        return [w for w, c in counts.items() if c >= threshold]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(trend: str, level_changed: bool, level: str) -> str:
        if level_changed:
            return f"Excellent progress — you reached {level} level!"
        return {
            "improving":  "Clear improvement across sessions.",
            "stable":     "Consistent performance — keep building.",
            "regressing": "Some metrics dipped — focus on the priority drill.",
        }.get(trend, "Progress tracked.")
