#!/usr/bin/env python3
"""
Report Generator
================
Parses the raw LLM text output into a structured FootballReport object
that the rest of the application (server.py, dashboard) can consume.

Responsibilities:
  - Receive raw LLM response text
  - Parse the five mandatory sections (Summary, Strengths, Areas to Improve,
    Training Drills, Coach Tip)
  - Return a typed FootballReport dataclass
  - Handle malformed LLM output gracefully — never crash the pipeline

Pipeline position:
  LLM raw text → ReportGenerator → FootballReport → Dashboard
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.llm_provider import LLMResponse


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class DrillReport:
    """A single training drill extracted from the LLM response."""
    name:         str
    instructions: str
    duration:     str


@dataclass
class FootballReport:
    """
    Structured coaching report parsed from the LLM response.
    Maps directly to the aiFeedback + drills fields in FootballSession.
    """
    summary:          str
    strengths:        List[str]
    areas_to_improve: List[str]
    drills:           List[DrillReport]
    coach_tip:        str

    # Metadata
    provider:         str    = "unknown"
    model:            str    = "unknown"
    latency_s:        float  = 0.0
    from_fallback:    bool   = False
    raw_text:         str    = ""

    def to_ai_feedback_dict(self) -> Dict[str, Any]:
        """Return the aiFeedback shape expected by types.ts / FootballSession."""
        return {
            "summary":         self.summary,
            "strengths":       self.strengths,
            "weaknesses":      self.areas_to_improve,
            "coachingTips":    [d.instructions for d in self.drills],
            "motivationalTip": self.coach_tip,
        }

    def to_drills_list(self) -> List[Dict[str, str]]:
        """Return the drills shape expected by types.ts / FootballSession."""
        return [
            {
                "name":         d.name,
                "instructions": d.instructions,
                "duration":     d.duration,
                "targetMetric": "",
                "difficulty":   "",
            }
            for d in self.drills
        ]


# ---------------------------------------------------------------------------
# Section header patterns
# ---------------------------------------------------------------------------

_SECTION_PATTERNS = {
    "summary":          re.compile(r"^\s*1[.):\s]*SUMMARY", re.I | re.M),
    "strengths":        re.compile(r"^\s*2[.):\s]*STRENGTHS", re.I | re.M),
    "areas_to_improve": re.compile(r"^\s*3[.):\s]*(AREAS?\s*(TO\s*IMPROVE|FOR\s*IMPROVEMENT)|WEAKNESSES)", re.I | re.M),
    "drills":           re.compile(r"^\s*4[.):\s]*TRAINING\s*DRILLS?", re.I | re.M),
    "coach_tip":        re.compile(r"^\s*5[.):\s]*COACH\s*TIP", re.I | re.M),
}

# Bullet list markers.
_BULLET = re.compile(r"^\s*[•\-\*\u2022]\s+", re.M)

# Drill line format:  Drill Name | Instructions | Duration
_DRILL_SEP = re.compile(r"\|")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Parse LLM response text into a structured FootballReport.

    Usage::

        generator = ReportGenerator()
        report    = generator.parse(llm_response)
        print(report.summary)
    """

    def parse(self, response: LLMResponse) -> FootballReport:
        """
        Parse the LLM response into a FootballReport.

        Parameters
        ----------
        response : LLMResponse

        Returns
        -------
        FootballReport — always returns, never raises.
        """
        text = response.text or ""

        try:
            sections = self._split_sections(text)
            summary         = self._clean_block(sections.get("summary", ""))
            strengths       = self._parse_bullets(sections.get("strengths", ""))
            areas           = self._parse_bullets(sections.get("areas_to_improve", ""))
            drills          = self._parse_drills(sections.get("drills", ""))
            coach_tip       = self._clean_block(sections.get("coach_tip", "")).split("\n")[0].strip()
        except Exception as exc:
            print(f"[ReportGenerator] Parse error: {exc} — using fallback values.")
            summary   = "Session analysis complete."
            strengths = []
            areas     = []
            drills    = []
            coach_tip = "Keep training consistently and focus on one improvement at a time."

        # Ensure minimum content.
        if not summary:
            summary = "Session analysis complete. Review the metrics above for details."
        if not coach_tip:
            coach_tip = "Consistency is key — repeat your best reps every session."

        return FootballReport(
            summary          = summary,
            strengths        = strengths,
            areas_to_improve = areas,
            drills           = drills,
            coach_tip        = coach_tip,
            provider         = response.provider,
            model            = response.model,
            latency_s        = response.latency_s,
            from_fallback    = response.from_fallback,
            raw_text         = text,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_sections(self, text: str) -> Dict[str, str]:
        """Split LLM text into named sections by header pattern."""
        # Find all section start positions.
        hits: List[tuple[str, int]] = []
        for name, pattern in _SECTION_PATTERNS.items():
            m = pattern.search(text)
            if m:
                hits.append((name, m.end()))

        # Sort by position.
        hits.sort(key=lambda x: x[1])

        sections: Dict[str, str] = {}
        for i, (name, start) in enumerate(hits):
            end = hits[i + 1][1] - len(hits[i + 1][0]) if i + 1 < len(hits) else len(text)
            # Back up to the start of the next header line.
            next_start = hits[i + 1][1] if i + 1 < len(hits) else len(text)
            # Find the next section header start.
            if i + 1 < len(hits):
                next_name, next_pos = hits[i + 1]
                # Find the actual line start of the next header.
                line_start = text.rfind("\n", 0, next_pos - 1)
                next_start = line_start if line_start != -1 else next_pos
            sections[name] = text[start:next_start].strip()

        return sections

    @staticmethod
    def _clean_block(text: str) -> str:
        """Remove bullets and leading/trailing whitespace from a text block."""
        lines = [_BULLET.sub("", line).strip() for line in text.splitlines()]
        lines = [l for l in lines if l]
        return " ".join(lines).strip()

    @staticmethod
    def _parse_bullets(text: str) -> List[str]:
        """Extract bullet points as a list of strings."""
        items: List[str] = []
        for line in text.splitlines():
            stripped = _BULLET.sub("", line).strip()
            if stripped:
                items.append(stripped)
        return items if items else [text.strip()] if text.strip() else []

    @staticmethod
    def _parse_drills(text: str) -> List[DrillReport]:
        """
        Parse drill entries.
        Expected format per line:  Drill Name | Instructions | Duration
        Falls back to treating each non-empty line as a single instruction.
        """
        drills: List[DrillReport] = []
        for line in text.splitlines():
            line = _BULLET.sub("", line).strip()
            if not line:
                continue
            parts = [p.strip() for p in _DRILL_SEP.split(line)]
            if len(parts) >= 3:
                drills.append(DrillReport(
                    name         = parts[0],
                    instructions = parts[1],
                    duration     = parts[2],
                ))
            elif len(parts) == 2:
                drills.append(DrillReport(
                    name         = parts[0],
                    instructions = parts[1],
                    duration     = "10 min",
                ))
            elif len(parts) == 1 and parts[0]:
                drills.append(DrillReport(
                    name         = "Drill",
                    instructions = parts[0],
                    duration     = "10 min",
                ))
        return drills
