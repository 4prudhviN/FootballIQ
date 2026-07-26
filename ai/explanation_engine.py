#!/usr/bin/env python3
"""
Explanation Engine
==================
Orchestrates Prompt → LLM → Validate → Regenerate → Report.

Never trust the LLM response.
Validates all 7 required sections. Regenerates if something is missing.

Required sections:
  session_summary, strengths, areas_to_improve,
  training_recommendations, next_focus

If any are missing → regenerate (up to max_retries times).
If still missing after retries → fill with safe defaults.

Pipeline position:
  Structured coaching data → ExplanationEngine → validated FootballReport
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ai.prompt_builder               import PromptBuilder, PromptContext
from ai.llm_provider                 import LLMProvider
from ai.report_generator             import ReportGenerator, FootballReport
from ai.json_validator               import JSONValidator
from ai.prompt_templates.template_loader import TemplateLoader
from utils.logger                    import get_logger

log = get_logger(__name__)

_REQUIRED_SECTIONS = [
    "session_summary",
    "strengths",
    "areas_to_improve",
    "training_recommendations",
    "next_focus",
]

_REGENERATION_INSTRUCTION = """\

IMPORTANT: Your previous response was missing required sections.
You MUST include ALL of these in your JSON response:
  - session_summary (non-empty string)
  - strengths (non-empty list)
  - areas_to_improve (non-empty list)
  - training_recommendations (non-empty list)
  - next_focus (non-empty string)

Respond with valid JSON only. No text outside the JSON object.\
"""


class ExplanationEngine:
    """
    Validates LLM responses and regenerates if required sections are missing.

    Parameters
    ----------
    provider    : str | None — LLM provider override
    max_retries : int — regeneration attempts if validation fails (default 2)
    """

    def __init__(
        self,
        provider:    Optional[str] = None,
        max_retries: int           = 2,
    ) -> None:
        self._builder     = PromptBuilder()
        self._llm         = LLMProvider(provider=provider)
        self._parser      = ReportGenerator()
        self._validator   = JSONValidator()
        self._templates   = TemplateLoader()
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def explain(
        self,
        detected_activities: List[str],
        player_level:        str,
        torso_lean:          float,
        knee_stability:      float,
        gait_symmetry:       float,
        warnings:            List[str],
        by_action:           Optional[Dict[str, Dict[str, str]]] = None,
        coaching_issues:     Optional[List[Dict]] = None,
        positive_findings:   Optional[List[str]]  = None,
        session_id:          Optional[str]  = None,
        video_duration_s:    Optional[float] = None,
    ) -> FootballReport:
        """
        Rewrite structured coaching data into natural language.
        Validates all 7 sections. Regenerates if anything is missing.

        Returns
        -------
        FootballReport — always valid, all sections present
        """
        ctx = PromptContext(
            player_level         = player_level,
            detected_activities  = detected_activities,
            torso_lean           = torso_lean,
            knee_stability       = knee_stability,
            gait_symmetry        = gait_symmetry,
            warnings             = warnings,
            by_action            = by_action or {},
            coaching_issues      = coaching_issues or [],
            positive_findings    = positive_findings or [],
            session_id           = session_id,
            video_duration_s     = video_duration_s,
        )

        prompt   = self._builder.build(ctx)
        response = None

        for attempt in range(1, self._max_retries + 2):
            response = self._llm.call(prompt)

            log.debug(
                "ExplanationEngine: attempt %d/%d  provider=%s  chars=%d",
                attempt, self._max_retries + 1,
                response.provider, len(response.text),
            )

            # ── Validate all required sections ────────────────────────────
            missing = self._find_missing_sections(response.text)

            if not missing:
                log.debug("ExplanationEngine: all sections present — attempt %d", attempt)
                break

            log.warning(
                "ExplanationEngine: attempt %d — missing sections: %s",
                attempt, missing,
            )

            if attempt <= self._max_retries:
                # Regenerate: append explicit instruction listing what's missing.
                missing_str = ", ".join(missing)
                prompt = (
                    self._builder.build(ctx)
                    + f"\n\nMISSING FROM LAST RESPONSE: {missing_str}"
                    + _REGENERATION_INSTRUCTION
                )
            # else: use whatever we have, ReportGenerator fills defaults

        report = self._parser.parse(response)

        if report.was_repaired:
            log.info("ExplanationEngine: report required repair/defaults")

        return report

    def quick_tip(
        self,
        detected_activities: List[str],
        player_level:        str,
        warnings:            List[str],
    ) -> str:
        """Return a single validated coaching tip."""
        ctx = PromptContext(
            player_level        = player_level,
            detected_activities = detected_activities,
            torso_lean          = 0.0,
            knee_stability      = 0.0,
            gait_symmetry       = 0.0,
            warnings            = warnings,
            by_action           = {},
        )
        prompt   = self._builder.build_quick_tip(ctx)
        response = self._llm.call(prompt)

        text      = (response.text or "").strip()
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        result    = sentences[0] + "." if sentences else text

        if not result:
            result = "Keep training consistently and focus on one drill at a time."

        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _find_missing_sections(self, text: str) -> List[str]:
        """
        Parse the LLM response and return a list of missing required sections.
        Empty list = all sections present.
        """
        try:
            validation = self._validator.validate(text)
            data       = validation.best or {}

            missing: List[str] = []
            for section in _REQUIRED_SECTIONS:
                val = data.get(section)
                if val is None or val == "" or val == []:
                    missing.append(section)
            return missing
        except Exception:
            return list(_REQUIRED_SECTIONS)   # assume all missing if parse fails
