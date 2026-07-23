#!/usr/bin/env python3
"""
Explanation Engine
==================
The top-level orchestrator for the AI layer.

Ties together PromptBuilder → LLMProvider → JSONValidator → ReportGenerator
into a single clean call:  metrics_in → validated FootballReport_out.

Every LLM response is validated by JSONValidator before leaving this module.
The frontend is guaranteed to receive a correctly-shaped object.

Pipeline position:
  Metrics → ExplanationEngine → JSONValidator → FootballReport → Dashboard

Usage::

    engine = ExplanationEngine()
    report = engine.explain(
        detected_activities = ["shooting"],
        player_level        = "Intermediate",
        torso_lean          = 22.0,
        knee_stability      = 72.0,
        gait_symmetry       = 92.0,
        warnings            = ["POOR POSTURE / LEANING BACK"],
    )
    print(report.summary)
    print(report.was_repaired)   # True if LLM response needed fixing
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


class ExplanationEngine:
    """
    Orchestrates the full Metrics → Prompt → LLM → Validate → Report pipeline.

    Parameters
    ----------
    provider : str | None
        LLM provider override ("gemini", "openai", "fireworks", "offline").
        Defaults to the LLM_PROVIDER environment variable.
    """

    def __init__(self, provider: Optional[str] = None) -> None:
        self._builder   = PromptBuilder()
        self._llm       = LLMProvider(provider=provider)
        self._parser    = ReportGenerator()
        self._validator = JSONValidator()
        self._templates = TemplateLoader()

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
        session_id:          Optional[str]  = None,
        video_duration_s:    Optional[float] = None,
    ) -> FootballReport:
        """
        Run the full AI explanation pipeline and return a validated FootballReport.

        Parameters
        ----------
        detected_activities : list[str]
        player_level        : str   — "Beginner" | "Intermediate" | "Advanced"
        torso_lean          : float — degrees
        knee_stability      : float — 0–100
        gait_symmetry       : float — 0–100
        warnings            : list[str]
        by_action           : dict | None
        session_id          : str | None
        video_duration_s    : float | None

        Returns
        -------
        FootballReport — always valid, always has all required fields
        """
        primary = detected_activities[0] if detected_activities else "general"

        # Build prompt using activity-specific template.
        ctx = PromptContext(
            player_level        = player_level,
            detected_activities = detected_activities,
            torso_lean          = torso_lean,
            knee_stability      = knee_stability,
            gait_symmetry       = gait_symmetry,
            warnings            = warnings,
            by_action           = by_action or {},
            session_id          = session_id,
            video_duration_s    = video_duration_s,
        )

        prompt   = self._builder.build(ctx)
        response = self._llm.call(prompt)

        log.debug(
            "ExplanationEngine: LLM response %d chars  provider=%s  fallback=%s",
            len(response.text), response.provider, response.from_fallback,
        )

        # ── Validate before parsing ────────────────────────────────────────
        # If the LLM returned structured JSON, validate and repair it first.
        # ReportGenerator will also validate internally, but this gives us
        # an early log of any issues.
        validation = self._validator.validate_feedback_response(response.text)
        if not validation.valid:
            log.warning(
                "ExplanationEngine: LLM response failed validation — errors: %s",
                validation.errors,
            )
        if validation.was_repaired:
            log.debug(
                "ExplanationEngine: LLM response repaired — warnings: %s",
                validation.warnings[:3],
            )

        # Parse into FootballReport (ReportGenerator also validates internally).
        report = self._parser.parse(response)

        log.debug(
            "ExplanationEngine: report ready — repaired=%s  summary_len=%d",
            report.was_repaired, len(report.summary),
        )

        return report

    def quick_tip(
        self,
        detected_activities: List[str],
        player_level:        str,
        warnings:            List[str],
    ) -> str:
        """
        Return a single coaching tip without a full report.
        Faster/cheaper than explain().

        Returns
        -------
        str — one or two sentences, validated to be non-empty.
        """
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

        # Safety: never return empty string.
        if not result:
            result = "Keep training consistently and focus on one drill at a time."

        return result
