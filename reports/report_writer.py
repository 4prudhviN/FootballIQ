#!/usr/bin/env python3
"""
Report Writer
=============
Saves every analysis session as structured output files.

Every analysis generates:
  reports/json/<session_id>.json   — full machine-readable session data
  reports/images/<session_id>/     — annotated frame snapshots (future)
  reports/pdf/<session_id>.pdf     — printable coaching report (future)

Usage::

    from reports.report_writer import ReportWriter
    writer = ReportWriter()
    writer.save_json(session_dict, session_id="abc123")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.file_utils import ensure_dir, write_json
from utils.logger     import get_logger

log = get_logger(__name__)

_BASE = Path(__file__).resolve().parent


class ReportWriter:
    """
    Writes analysis output to the reports/ directory tree.

    Parameters
    ----------
    base_dir : Path | None
        Override the default reports/ directory (useful for testing).
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base      = base_dir or _BASE
        self._json_dir  = ensure_dir(self._base / "json")
        self._img_dir   = ensure_dir(self._base / "images")
        self._pdf_dir   = ensure_dir(self._base / "pdf")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(self, data: dict[str, Any], session_id: str) -> Path:
        """
        Serialise the full session dict to reports/json/<session_id>.json.

        Parameters
        ----------
        data       : dict — output of AnalysisSession.to_dict()
        session_id : str

        Returns
        -------
        Path to the written file.
        """
        path = self._json_dir / f"{session_id}.json"
        # Inject a write timestamp.
        data["_saved_at"] = datetime.utcnow().isoformat() + "Z"
        write_json(path, data, indent=2)
        log.info("JSON report saved: %s", path)
        return path

    def load_json(self, session_id: str) -> dict[str, Any]:
        """Load a previously saved JSON report. Returns {} if not found."""
        path = self._json_dir / f"{session_id}.json"
        if not path.exists():
            log.warning("JSON report not found: %s", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_sessions(self) -> list[str]:
        """Return all saved session IDs (sorted newest first by mtime)."""
        files = sorted(
            self._json_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [p.stem for p in files]

    # ------------------------------------------------------------------
    # Images (placeholder — implemented when frame export is added)
    # ------------------------------------------------------------------

    def image_dir(self, session_id: str) -> Path:
        """Return (and create) the image output directory for a session."""
        return ensure_dir(self._img_dir / session_id)

    # ------------------------------------------------------------------
    # PDF (placeholder — implemented when PDF export is added)
    # ------------------------------------------------------------------

    def pdf_path(self, session_id: str) -> Path:
        """Return the expected PDF output path for a session."""
        return self._pdf_dir / f"{session_id}.pdf"

    def pdf_exists(self, session_id: str) -> bool:
        return self.pdf_path(session_id).exists()
