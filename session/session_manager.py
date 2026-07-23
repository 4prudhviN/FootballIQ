#!/usr/bin/env python3
"""
Session Manager
===============
Every uploaded video becomes a Session object.
SessionManager creates, persists, and retrieves sessions.

Responsibilities:
  - Create a new Session from a pipeline PipelineOutput
  - Persist session as JSON via ReportWriter
  - Load existing sessions by ID
  - Provide the latest session for a player

Usage::

    manager = SessionManager()
    session = manager.create(pipeline_output, file_name="clip.mp4")
    print(session.id, session.player_level)

    loaded = manager.load(session.id)
    print(loaded.ai_feedback["summary"])
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from reports.report_writer import ReportWriter
from utils.file_utils      import ensure_dir
from utils.logger          import get_logger
from config.settings       import settings

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """
    Represents one complete analysis session.
    Created from a PipelineOutput and persisted as JSON.
    """
    id:                   str
    file_name:            str
    created_at:           str                    # ISO 8601
    status:               str                    # "complete" | "failed"
    player_level:         str
    detected_activities:  List[str]
    metrics:              Dict[str, Any]
    ai_feedback:          Dict[str, Any]
    drills:               List[Dict[str, Any]]
    timeline:             List[Dict[str, Any]]
    video_url:            Optional[str]           = None
    error:                Optional[str]           = None
    diagnostics:          Dict[str, Any]          = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def warnings(self) -> List[str]:
        return self.metrics.get("warnings", [])

    @property
    def torso_lean(self) -> float:
        return float(self.metrics.get("torsoLean", 0.0))

    @property
    def knee_stability(self) -> float:
        return float(self.metrics.get("kneeStability", 0.0))

    @property
    def gait_symmetry(self) -> float:
        return float(self.metrics.get("gaitSymmetry", 0.0))

    @property
    def summary(self) -> str:
        return self.ai_feedback.get("summary", "")

    @property
    def primary_activity(self) -> Optional[str]:
        return self.detected_activities[0] if self.detected_activities else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":                   self.id,
            "file_name":            self.file_name,
            "created_at":           self.created_at,
            "status":               self.status,
            "player_level":         self.player_level,
            "detected_activities":  self.detected_activities,
            "metrics":              self.metrics,
            "ai_feedback":          self.ai_feedback,
            "drills":               self.drills,
            "timeline":             self.timeline,
            "video_url":            self.video_url,
            "error":                self.error,
            "diagnostics":          self.diagnostics,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Session":
        return Session(
            id                   = data.get("id", ""),
            file_name            = data.get("file_name", ""),
            created_at           = data.get("created_at", ""),
            status               = data.get("status", "complete"),
            player_level         = data.get("player_level", "Beginner"),
            detected_activities  = data.get("detected_activities", []) or data.get("detectedActivities", []),
            metrics              = data.get("metrics", {}),
            ai_feedback          = data.get("ai_feedback", {}) or data.get("aiFeedback", {}),
            drills               = data.get("drills", []),
            timeline             = data.get("timeline", []),
            video_url            = data.get("video_url"),
            error                = data.get("error"),
            diagnostics          = data.get("diagnostics", {}),
        )


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Creates, persists, and loads analysis sessions.

    Parameters
    ----------
    writer : ReportWriter | None
        JSON persistence layer — created automatically if not provided.
    """

    def __init__(self, writer: Optional[ReportWriter] = None) -> None:
        self._writer = writer or ReportWriter()

    def create(
        self,
        pipeline_output: Any,         # PipelineOutput from pipeline_manager
        file_name:       str = "",
        player_id:       Optional[str] = None,
    ) -> Session:
        """
        Create a new Session from a PipelineOutput and persist it.

        Parameters
        ----------
        pipeline_output : PipelineOutput
        file_name       : str — original uploaded filename
        player_id       : str | None — optional player identifier

        Returns
        -------
        Session
        """
        session_id = uuid.uuid4().hex[:12]
        now        = datetime.utcnow().isoformat() + "Z"

        video_url = (
            f"/api/video/{session_id}/analyzed.mp4"
            if pipeline_output.success else None
        )

        session = Session(
            id                   = session_id,
            file_name            = file_name,
            created_at           = now,
            status               = "complete" if pipeline_output.success else "failed",
            player_level         = getattr(pipeline_output, "player_level", "Beginner"),
            detected_activities  = getattr(pipeline_output, "detected_activities", []),
            metrics              = getattr(pipeline_output, "metrics", {}),
            ai_feedback          = getattr(pipeline_output, "ai_feedback", {}),
            drills               = getattr(pipeline_output, "drills", []),
            timeline             = getattr(pipeline_output, "timeline", []),
            video_url            = video_url,
            error                = getattr(pipeline_output, "error", None),
            diagnostics          = getattr(pipeline_output, "diagnostics", {}),
        )

        # Persist to JSON.
        data = session.to_dict()
        if player_id:
            data["player_id"] = player_id
        self._writer.save_json(data, session_id=session_id)

        log.info("Session created: %s  level=%s  activities=%s",
                 session_id, session.player_level, session.detected_activities)
        return session

    def load(self, session_id: str) -> Optional[Session]:
        """
        Load a session by ID from persistent storage.

        Returns None if not found.
        """
        data = self._writer.load_json(session_id)
        if not data:
            log.warning("SessionManager.load: session not found — %s", session_id)
            return None
        return Session.from_dict(data)

    def list_sessions(self) -> List[str]:
        """Return all saved session IDs, newest first."""
        return self._writer.list_sessions()

    def delete(self, session_id: str) -> bool:
        """Delete a session JSON. Returns True if successful."""
        from utils.file_utils import safe_remove
        path = self._writer._json_dir / f"{session_id}.json"
        ok   = safe_remove(path)
        if ok:
            log.info("Session deleted: %s", session_id)
        return ok
