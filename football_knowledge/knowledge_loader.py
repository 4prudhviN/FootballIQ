#!/usr/bin/env python3
"""
Knowledge Loader
================
Single source of truth for all football_knowledge/ JSON files.

Every module calls:
    from football_knowledge.knowledge_loader import KnowledgeLoader
    loader = KnowledgeLoader()
    data = loader.load("passing")          # football_knowledge/activities/passing.json
    data = loader.load("drills/beginner")  # football_knowledge/drills/beginner.json
    data = loader.load("mistakes/passing") # football_knowledge/mistakes/passing.json
    data = loader.terminology()            # football_knowledge/terminology.json

Features:
  - Single source of truth — no more raw json.load() in 15 files
  - In-memory cache — each file loaded once per process
  - Graceful fallback — returns {} on missing files, never raises
  - Category helpers: activities(), drills(), mistakes(), coaching_rules()

Usage::

    from football_knowledge.knowledge_loader import KnowledgeLoader
    kb = KnowledgeLoader()
    passing = kb.load("passing")          # activities/passing.json
    beginner = kb.load("drills/beginner") # drills/beginner.json
    term = kb.terminology()               # terminology.json
    term_text = kb.translate("support foot", level="Beginner")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

_BASE = Path(__file__).resolve().parent


class KnowledgeLoader:
    """
    Loads and caches football_knowledge JSON files.

    Parameters
    ----------
    base_dir : Path | None — override the default football_knowledge/ directory
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base  = base_dir or _BASE
        self._cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def load(self, key: str) -> Dict[str, Any]:
        """
        Load a knowledge file by key.

        Key resolution order:
          1. Exact path: {key}.json
          2. activities/{key}.json
          3. drills/{key}.json
          4. mistakes/{key}.json
          5. coaching_rules/{key}.json

        Parameters
        ----------
        key : str
            e.g. "passing", "drills/beginner", "mistakes/shooting"

        Returns
        -------
        dict — parsed JSON, or {} if file not found
        """
        if key in self._cache:
            return self._cache[key]

        data = self._try_load(key)
        self._cache[key] = data
        return data

    def terminology(self) -> Dict[str, Any]:
        """Load terminology.json — the football term dictionary."""
        return self.load("terminology")

    def translate(self, term: str, level: str = "Beginner") -> str:
        """
        Translate a football term into level-appropriate plain English.

        Returns the original term if not found in the terminology database.
        """
        terms = self.terminology()
        key   = term.strip().lower().replace(" ", "_").replace("-", "_")
        entry = terms.get(key) or terms.get(term.strip().lower())
        if not entry:
            return term
        if isinstance(entry, dict):
            return entry.get(level, entry.get("Beginner", term))
        return str(entry)

    def activities(self, activity: str) -> Dict[str, Any]:
        """Load activities/{activity}.json."""
        return self.load(f"activities/{activity}")

    def drills(self, level: str) -> Dict[str, Any]:
        """Load drills/{level}.json (beginner / intermediate / advanced)."""
        return self.load(f"drills/{level.lower()}")

    def mistakes(self, activity: str) -> Dict[str, Any]:
        """Load mistakes/{activity}.json."""
        return self.load(f"mistakes/{activity}")

    def coaching_rules(self, level: str) -> Dict[str, Any]:
        """Load coaching_rules/{level}_rules.json."""
        return self.load(f"coaching_rules/{level.lower()}_rules")

    def all_drills_for_level(self, level: str) -> list[dict]:
        """Return flat list of all drills for a level."""
        data = self.drills(level)
        return data.get("drills", [])

    def drill_by_id(self, drill_id: str, level: str) -> Optional[Dict[str, Any]]:
        """Find a drill by ID within a level's drill file."""
        for drill in self.all_drills_for_level(level):
            if drill.get("id") == drill_id:
                return drill
        return None

    def mistakes_for_activity(self, activity: str) -> list[dict]:
        """Return flat list of mistakes for an activity."""
        data = self.mistakes(activity)
        return data.get("mistakes", [])

    def clear_cache(self) -> None:
        """Clear the in-memory cache (useful for testing)."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _try_load(self, key: str) -> Dict[str, Any]:
        """Try multiple path resolutions for a key."""
        candidates = [
            self._base / f"{key}.json",
            self._base / "activities" / f"{key}.json",
            self._base / "drills"     / f"{key}.json",
            self._base / "mistakes"   / f"{key}.json",
            self._base / "coaching_rules" / f"{key}.json",
            self._base / "coaching_rules" / f"{key}_rules.json",
        ]

        for path in candidates:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    return {}

        return {}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_loader: Optional[KnowledgeLoader] = None


def get_knowledge_loader() -> KnowledgeLoader:
    """Return the shared KnowledgeLoader singleton."""
    global _loader
    if _loader is None:
        _loader = KnowledgeLoader()
    return _loader
