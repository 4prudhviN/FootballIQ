#!/usr/bin/env python3
"""
Session Exporter
================
Exports one or multiple sessions to different formats.

Supported exports:
  - JSON  (single session or batch)
  - CSV   (metric summary for spreadsheet analysis)
  - Text  (plain-text coaching report)
  - PDF   (placeholder — structure ready for future pdf library)

Usage::

    exporter = SessionExporter()

    # Export one session to JSON
    path = exporter.to_json(session)

    # Export to CSV for spreadsheet
    path = exporter.to_csv([session1, session2, session3])

    # Export plain-text report
    text = exporter.to_text(session)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from session.session_manager import Session
from utils.file_utils        import ensure_dir
from utils.logger            import get_logger

log = get_logger(__name__)

_EXPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


class SessionExporter:
    """
    Exports session data to multiple formats.

    Parameters
    ----------
    export_dir : Path | None
        Base export directory. Defaults to reports/.
    """

    def __init__(self, export_dir: Optional[Path] = None) -> None:
        self._base = export_dir or _EXPORT_DIR
        ensure_dir(self._base / "json")
        ensure_dir(self._base / "csv")
        ensure_dir(self._base / "text")
        ensure_dir(self._base / "pdf")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, session: Session) -> Path:
        """Export a single session to JSON. Returns the file path."""
        path = self._base / "json" / f"{session.id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        log.info("SessionExporter: JSON → %s", path)
        return path

    def batch_to_json(self, sessions: List[Session]) -> Path:
        """Export a list of sessions as a JSON array. Returns the file path."""
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self._base / "json" / f"batch_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in sessions], f, indent=2, ensure_ascii=False)
        log.info("SessionExporter: batch JSON (%d sessions) → %s", len(sessions), path)
        return path

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def to_csv(self, sessions: List[Session]) -> Path:
        """
        Export metric summaries for multiple sessions to a CSV file.
        Each row = one session.
        """
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self._base / "csv" / f"sessions_{ts}.csv"

        fieldnames = [
            "session_id", "date", "file_name",
            "player_level", "primary_activity",
            "torso_lean", "knee_stability", "gait_symmetry",
            "warning_count", "drill_count", "status",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in sessions:
                writer.writerow({
                    "session_id":       s.id,
                    "date":             s.created_at,
                    "file_name":        s.file_name,
                    "player_level":     s.player_level,
                    "primary_activity": s.primary_activity or "",
                    "torso_lean":       round(s.torso_lean, 2),
                    "knee_stability":   round(s.knee_stability, 2),
                    "gait_symmetry":    round(s.gait_symmetry, 2),
                    "warning_count":    len(s.warnings),
                    "drill_count":      len(s.drills),
                    "status":           s.status,
                })

        log.info("SessionExporter: CSV (%d rows) → %s", len(sessions), path)
        return path

    # ------------------------------------------------------------------
    # Plain text coaching report
    # ------------------------------------------------------------------

    def to_text(self, session: Session) -> str:
        """
        Generate a plain-text coaching report from a session.
        Returns the text string and also saves it to reports/text/.
        """
        lines = [
            "=" * 56,
            "  FootballIQ — Session Coaching Report",
            "=" * 56,
            f"  Session   : {session.id}",
            f"  File      : {session.file_name}",
            f"  Date      : {session.created_at[:10]}",
            f"  Level     : {session.player_level}",
            f"  Activities: {', '.join(session.detected_activities) or 'None detected'}",
            "=" * 56,
            "",
            "  BIOMECHANICAL METRICS",
            f"  Torso Lean      : {session.torso_lean:.1f}°",
            f"  Knee Stability  : {session.knee_stability:.0f}/100",
            f"  Gait Symmetry   : {session.gait_symmetry:.0f}/100",
            "",
        ]

        feedback = session.ai_feedback
        if feedback.get("summary"):
            lines += ["  SUMMARY", f"  {feedback['summary']}", ""]

        if feedback.get("strengths"):
            lines.append("  STRENGTHS")
            for s in feedback["strengths"]:
                lines.append(f"  ✓  {s}")
            lines.append("")

        if feedback.get("weaknesses"):
            lines.append("  AREAS TO IMPROVE")
            for w in feedback["weaknesses"]:
                lines.append(f"  ✗  {w}")
            lines.append("")

        if session.drills:
            lines.append("  TRAINING DRILLS")
            for i, d in enumerate(session.drills, 1):
                lines.append(f"  {i}. {d.get('name', 'Drill')}")
                lines.append(f"     {d.get('instructions', '')}")
                lines.append(f"     Duration: {d.get('duration', '10 min')}")
            lines.append("")

        if feedback.get("motivationalTip"):
            lines += ["  COACH TIP", f"  {feedback['motivationalTip']}", ""]

        if session.timeline:
            lines.append("  ACTIVITY TIMELINE")
            for seg in session.timeline:
                lines.append(f"  {seg.get('label', '')}")
            lines.append("")

        lines.append("=" * 56)
        text = "\n".join(lines)

        # Save to file.
        path = self._base / "text" / f"{session.id}.txt"
        path.write_text(text, encoding="utf-8")
        log.info("SessionExporter: text report → %s", path)

        return text

    # ------------------------------------------------------------------
    # PDF (placeholder)
    # ------------------------------------------------------------------

    def to_pdf(self, session: Session) -> Path:
        """
        PDF export placeholder.
        Currently writes a plain-text file with .pdf extension.
        Replace body with reportlab / weasyprint when ready.
        """
        text = self.to_text(session)
        path = self._base / "pdf" / f"{session.id}.pdf"
        path.write_text(text, encoding="utf-8")
        log.info("SessionExporter: PDF placeholder → %s", path)
        return path
