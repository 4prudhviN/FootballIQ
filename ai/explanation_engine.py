#!/usr/bin/env python3
"""
Explanation Engine
==================
The top-level orchestrator for the AI layer.

Ties together PromptBuilder → LLMProvider → ReportGenerator into a single
clean call:  metrics_in → FootballReport_out.

Responsibilities:
  - Accept a MetricSet (or raw dict) + session context
  - Build the prompt via PromptBuilder
  - Send to LLM via LLMProvider
  - Parse the response via ReportGenerator
  - Return a FootballReport

This is the ONLY module that server.py and pipeline_manager.py should
import from the ai/ package.

Pipeline position:
  Metrics → ExplanationEngine → FootballReport → Dashboard

Usage::

    engine = ExplanationEngine()
    report = engine.explain(
        metrics          = {"Shot Velocity": "88 km/h", ...},
        detected_activities = ["shooting"],
        player_level     = "Intermediate",
        torso_lean       = 22.0,
        knee_stability   = 72.0,
        gait_symmetry    = 92.0,
        warnings         = ["POOR POSTURE / LEANING BACK"],
    )
    print(report.summary)
    print(report.coach_tip)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai.prompt_builder   import PromptBuilder, PromptContext
from ai.llm_provider     import LLMProvider
from ai.report_generator import ReportGenerator, FootballReport


class ExplanationEngine:
    """
    Orchestrates the full Metrics → Prompt → LLM → Report pipeline.

    Parameters
    ----------
    provider : str | None
        LLM provider override ("gemini", "openai", "fireworks", "offline").
        Defaults to the LLM_PROVIDER environment variable.

    Usage::

        engine = ExplanationEngine()
        report = engine.explain(...)
    """

    def __init__(self, provider: Optional[str] = None) -> None:
        self._builder   = PromptBuilder()
        self._llm       = LLMProvider(provider=provider)
        self._parser    = ReportGenerator()

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
        Run the full AI explanation pipeline and return a FootballReport.

        Parameters
        ----------
        detected_activities : list[str]
            Activities detected by the pipeline (e.g. ["shooting", "movement"]).
        player_level : str
            "Beginner", "Intermediate", or "Advanced".
        torso_lean : float
            Torso lean angle in degrees (from pose estimator).
        knee_stability : float
            Knee stability score 0–100 (from pose estimator).
        gait_symmetry : float
            Gait symmetry score 0–100 (from pose estimator).
        warnings : list[str]
            Raw warning flags (e.g. "POOR POSTURE / LEANING BACK").
        by_action : dict | None
            Per-action display metrics (e.g. {"shooting": {"Shot Velocity": "88 km/h"}}).
        session_id : str | None
            Optional session identifier for logging.
        video_duration_s : float | None
            Video duration in seconds.

        Returns
        -------
        FootballReport
        """
        ctx = PromptContext(
            player_level         = player_level,
            detected_activities  = detected_activities,
            torso_lean           = torso_lean,
            knee_stability       = knee_stability,
            gait_symmetry        = gait_symmetry,
            warnings             = warnings,
            by_action            = by_action or {},
            session_id           = session_id,
            video_duration_s     = video_duration_s,
        )

        prompt   = self._builder.build(ctx)
        response = self._llm.call(prompt)
        report   = self._parser.parse(response)

        return report

    def quick_tip(
        self,
        detected_activities: List[str],
        player_level:        str,
        warnings:            List[str],
    ) -> str:
        """
        Return a single coaching tip string without a full report.
        Faster and cheaper than explain() — uses a shorter prompt.

        Returns
        -------
        str — one or two sentences.
        """
        ctx = PromptContext(
            player_level         = player_level,
            detected_activities  = detected_activities,
            torso_lean           = 0.0,
            knee_stability       = 0.0,
            gait_symmetry        = 0.0,
            warnings             = warnings,
            by_action            = {},
        )
        prompt   = self._builder.build_quick_tip(ctx)
        response = self._llm.call(prompt)
        # Return just the first meaningful sentence.
        text = response.text.strip()
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return sentences[0] + "." if sentences else text
