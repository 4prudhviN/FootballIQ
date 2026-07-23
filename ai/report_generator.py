#!/usr/bin/env python3
"""
Report Generator
================
Parses the raw LLM text output into a structured FootballReport object.

Every LLM response is validated by JSONValidator before use.
Malformed, partial, or empty responses are repaired automatically.
The frontend always receives a correctly-shaped object — never a crash.

Parse strategy (in order):
  1. Try to parse response as JSON → validate against feedback schema
  2. If JSON fails, fall back to section-based text parser
  3. Fill any missing fields with safe defaults via JSONValidator.repair()
  4. Type-coerce all fields so the frontend never gets unexpected types

Pipeline position:
  LLM raw text → JSONValidator → ReportGenerator → FootballReport → Dashboard
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.llm_provider   import LLMResponse
from ai.json_validator import JSONValidator, FEEDBACK_SCHEMA
from utils.logger      import get_logger

log = get_logger(__name__)

_validator = JSONValidator()


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
    provider:      str   = "unknown"
    model:         str   = "unknown"
    latency_s:     float = 0.0
    from_fallback: bool  = False
    was_repaired:  bool  = False
    raw_text:      str   = ""

    @property
    def coaching_tips(self) -> List[str]:
        return [d.instructions for d in self.drills]

    def to_ai_feedback_dict(self) -> Dict[str, Any]:
        """Return the aiFeedback shape expected by types.ts / FootballSession."""
        return {
            "summary":         self.summary,
            "strengths":       self.strengths,
            "weaknesses":      self.areas_to_improve,
            "coachingTips":    self.coaching_tips,
            "motivationalTip": self.coach_tip,
        }

    def to_drills_list(self) -> List[Dict[str, str]]:
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
# Section header patterns (text-fallback parser)
# ---------------------------------------------------------------------------

_SECTION_PATTERNS = {
    "summary":          re.compile(r"^\s*1[.):\s]*SUMMARY", re.I | re.M),
    "strengths":        re.compile(r"^\s*2[.):\s]*STRENGTHS", re.I | re.M),
    "areas_to_improve": re.compile(r"^\s*3[.):\s]*(AREAS?\s*(TO\s*IMPROVE|FOR\s*IMPROVEMENT)|WEAKNESSES)", re.I | re.M),
    "drills":           re.compile(r"^\s*4[.):\s]*TRAINING\s*DRILLS?", re.I | re.M),
    "coach_tip":        re.compile(r"^\s*5[.):\s]*COACH\s*TIP", re.I | re.M),
}

_BULLET    = re.compile(r"^\s*[•\-\*\u2022]\s+", re.M)
_DRILL_SEP = re.compile(r"\|")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Parse LLM response text into a structured FootballReport.

    Tries JSON parsing first (validated + repaired), then falls back to
    section-based text parsing. Always returns a valid FootballReport.

    Usage::

        generator = ReportGenerator()
        report    = generator.parse(llm_response)
        print(report.summary)
        print(report.was_repaired)   # True if the response needed fixing
    """

    def parse(self, response: LLMResponse) -> FootballReport:
        """
        Parse and validate an LLM response into a FootballReport.

        Strategy:
          1. Attempt JSON extraction + validation via JSONValidator
          2. If JSON found → build report from validated/repaired dict
          3. If no JSON → fall back to section-based text parser
          4. Apply final safety defaults

        Always returns — never raises.
        """
        text         = response.text or ""
        was_repaired = False

        # ── Strategy 1: JSON path ──────────────────────────────────────────
        try:
            validation = _validator.validate_feedback_response(text)

            if validation.best:
                data         = validation.best
                was_repaired = validation.was_repaired

                if validation.warnings:
                    log.debug("ReportGenerator (JSON): %d repair(s) applied: %s",
                              len(validation.warnings), validation.warnings[:2])

                summary   = str(data.get("summary", "")).strip()
                strengths = self._ensure_list(data.get("strengths", []))
                weaknesses = self._ensure_list(data.get("weaknesses", []))
                tips      = self._ensure_list(data.get("coachingTips", []))
                mot_tip   = str(data.get("motivationalTip", "")).strip()

                # Convert coachingTips to DrillReport objects.
                drills = [
                    DrillReport(name="Drill", instructions=tip, duration="10 min")
                    for tip in tips
                    if tip
                ]

                coach_tip = mot_tip or (tips[-1] if tips else "")

                if summary:
                    return self._make_report(
                        summary, strengths, weaknesses, drills, coach_tip,
                        response, was_repaired, text,
                    )

        except Exception as exc:
            log.debug("ReportGenerator: JSON path failed (%s) — trying text parser", exc)

        # ── Strategy 2: Section-based text parser ──────────────────────────
        was_repaired = True   # text parsing always considered a repair
        try:
            sections   = self._split_sections(text)
            summary    = self._clean_block(sections.get("summary", ""))
            strengths  = self._parse_bullets(sections.get("strengths", ""))
            areas      = self._parse_bullets(sections.get("areas_to_improve", ""))
            drills     = self._parse_drills(sections.get("drills", ""))
            coach_tip  = self._clean_block(sections.get("coach_tip", "")).split("\n")[0].strip()
        except Exception as exc:
            log.warning("ReportGenerator: text parser failed (%s) — using empty defaults", exc)
            summary, strengths, areas, drills, coach_tip = "", [], [], [], ""

        return self._make_report(
            summary, strengths, areas, drills, coach_tip,
            response, was_repaired, text,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_report(
        summary:    str,
        strengths:  List[str],
        areas:      List[str],
        drills:     List[DrillReport],
        coach_tip:  str,
        response:   LLMResponse,
        repaired:   bool,
        raw_text:   str,
    ) -> FootballReport:
        """Apply final safety defaults and build the FootballReport."""
        if not summary:
            summary   = "Session analysis complete. Review the metrics above for details."
            repaired  = True
        if not coach_tip:
            coach_tip = "Consistency is key — repeat your best reps every session."
            repaired  = True

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
            was_repaired     = repaired,
            raw_text         = raw_text,
        )

    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        """Coerce a value to List[str] safely."""
        if isinstance(value, list):
            return [str(v) for v in value if v]
        if isinstance(value, str) and value:
            return [value]
        return []

    def _split_sections(self, text: str) -> Dict[str, str]:
        """Split LLM text into named sections by header pattern."""
        hits: List[tuple[str, int]] = []
        for name, pattern in _SECTION_PATTERNS.items():
            m = pattern.search(text)
            if m:
                hits.append((name, m.end()))
        hits.sort(key=lambda x: x[1])

        sections: Dict[str, str] = {}
        for i, (name, start) in enumerate(hits):
            if i + 1 < len(hits):
                next_pos   = hits[i + 1][1]
                line_start = text.rfind("\n", 0, next_pos - 1)
                next_start = line_start if line_start != -1 else next_pos
            else:
                next_start = len(text)
            sections[name] = text[start:next_start].strip()
        return sections

    @staticmethod
    def _clean_block(text: str) -> str:
        lines = [_BULLET.sub("", line).strip() for line in text.splitlines()]
        return " ".join(l for l in lines if l).strip()

    @staticmethod
    def _parse_bullets(text: str) -> List[str]:
        items = [_BULLET.sub("", line).strip() for line in text.splitlines()]
        items = [i for i in items if i]
        return items if items else ([text.strip()] if text.strip() else [])

    @staticmethod
    def _parse_drills(text: str) -> List[DrillReport]:
        drills: List[DrillReport] = []
        for line in text.splitlines():
            line = _BULLET.sub("", line).strip()
            if not line:
                continue
            parts = [p.strip() for p in _DRILL_SEP.split(line)]
            if len(parts) >= 3:
                drills.append(DrillReport(parts[0], parts[1], parts[2]))
            elif len(parts) == 2:
                drills.append(DrillReport(parts[0], parts[1], "10 min"))
            elif parts[0]:
                drills.append(DrillReport("Drill", parts[0], "10 min"))
        return drills
