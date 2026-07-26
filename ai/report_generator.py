#!/usr/bin/env python3
"""
Report Generator
================
Parses the LLM response into a structured FootballReport with all 7 sections.

Sections:
  1. Session Summary
  2. Activity Analysis
  3. Strengths
  4. Areas to Improve
  5. Coach Explanation
  6. Training Recommendations
  7. Next Focus

Every LLM response is validated by JSONValidator before use.
Falls back gracefully — never crashes the pipeline.

Pipeline position:
  LLM text → JSONValidator → ReportGenerator → FootballReport → Dashboard
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.llm_provider    import LLMResponse
from ai.json_validator  import JSONValidator
from utils.logger       import get_logger

log = get_logger(__name__)

_validator = JSONValidator()

# Expected JSON schema with safe defaults.
_REPORT_SCHEMA: Dict[str, Any] = {
    "session_summary":          "",
    "activity_analysis":        "",
    "strengths":                [],
    "areas_to_improve":         [],
    "coach_explanation":        "",
    "training_recommendations": [],
    "next_focus":               "",
    "motivationalTip":          "",
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class DrillReport:
    """A single training drill from the LLM response."""
    name:         str
    instructions: str
    duration:     str = "10 min"


@dataclass
class FootballReport:
    """
    Structured coaching report with all 7 sections.
    Maps to the full aiFeedback payload in FootballSession.
    """
    # 7 sections
    session_summary:          str
    activity_analysis:        str
    strengths:                List[str]
    areas_to_improve:         List[str]
    coach_explanation:        str
    training_recommendations: List[str]
    next_focus:               str
    motivational_tip:         str

    # Metadata
    provider:      str   = "unknown"
    model:         str   = "unknown"
    latency_s:     float = 0.0
    from_fallback: bool  = False
    was_repaired:  bool  = False
    raw_text:      str   = ""

    # ------------------------------------------------------------------

    @property
    def summary(self) -> str:
        """Alias for session_summary — backwards compatibility."""
        return self.session_summary

    @property
    def coach_tip(self) -> str:
        """Single coach tip — next_focus or motivational_tip."""
        return self.next_focus or self.motivational_tip

    @property
    def coaching_tips(self) -> List[str]:
        return self.training_recommendations

    def to_ai_feedback_dict(self) -> Dict[str, Any]:
        """Return the aiFeedback shape expected by the frontend (FootballSession)."""
        return {
            "summary":                 self.session_summary,
            "activityAnalysis":        self.activity_analysis,
            "strengths":               self.strengths,
            "weaknesses":              self.areas_to_improve,
            "coachExplanation":        self.coach_explanation,
            "coachingTips":            self.training_recommendations,
            "nextFocus":               self.next_focus,
            "motivationalTip":         self.motivational_tip,
        }

    def to_drills_list(self) -> List[Dict[str, str]]:
        return [
            {
                "name":         rec.split(":")[0].strip() if ":" in rec else "Drill",
                "instructions": rec,
                "duration":     "10 min",
                "targetMetric": "",
                "difficulty":   "",
            }
            for rec in self.training_recommendations
        ]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Parses the LLM response into a structured FootballReport.

    Strategy:
      1. Try JSON parse → validate against 7-section schema
      2. Repair missing fields with safe defaults
      3. Type-coerce all fields

    Always returns — never raises.
    """

    def parse(self, response: LLMResponse) -> FootballReport:
        """Parse and validate LLM response into a FootballReport."""
        text         = response.text or ""
        was_repaired = False

        # ── Validate + repair ──────────────────────────────────────────────
        try:
            result = _validator.validate(text)
            data   = result.best or {}

            repaired, repair_warnings = _validator.repair(data, _REPORT_SCHEMA)
            was_repaired = result.was_repaired or bool(repair_warnings)

            if repair_warnings:
                log.debug("ReportGenerator: %d repairs applied", len(repair_warnings))

            # Type coerce.
            repaired = self._coerce(repaired)

        except Exception as exc:
            log.warning("ReportGenerator: parse error (%s) — using defaults", exc)
            repaired     = dict(_REPORT_SCHEMA)
            was_repaired = True

        # ── Apply final safety defaults ────────────────────────────────────
        if not repaired.get("session_summary"):
            repaired["session_summary"] = "Session analysis complete. Review the findings below."
            was_repaired = True
        if not repaired.get("next_focus"):
            repaired["next_focus"] = "Focus on consistent technique in your next session."
            was_repaired = True
        if not repaired.get("motivationalTip"):
            repaired["motivationalTip"] = "Keep training — every session makes you better."
            was_repaired = True

        return FootballReport(
            session_summary          = repaired.get("session_summary", ""),
            activity_analysis        = repaired.get("activity_analysis", ""),
            strengths                = repaired.get("strengths", []),
            areas_to_improve         = repaired.get("areas_to_improve", []),
            coach_explanation        = repaired.get("coach_explanation", ""),
            training_recommendations = repaired.get("training_recommendations", []),
            next_focus               = repaired.get("next_focus", ""),
            motivational_tip         = repaired.get("motivationalTip", ""),
            provider                 = response.provider,
            model                    = response.model,
            latency_s                = response.latency_s,
            from_fallback            = response.from_fallback,
            was_repaired             = was_repaired,
            raw_text                 = text,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce(data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure correct types for all fields."""
        list_fields   = ["strengths", "areas_to_improve", "training_recommendations"]
        string_fields = [
            "session_summary", "activity_analysis", "coach_explanation",
            "next_focus", "motivationalTip",
        ]

        for f in list_fields:
            val = data.get(f)
            if isinstance(val, str):
                data[f] = [val] if val else []
            elif not isinstance(val, list):
                data[f] = []
            else:
                # Ensure all items are strings.
                data[f] = [str(item) for item in val if item]

        for f in string_fields:
            val = data.get(f)
            if not isinstance(val, str):
                data[f] = str(val) if val is not None else ""

        return data
