#!/usr/bin/env python3
"""
Prompt Builder
==============
Converts structured metric data into a well-formed LLM prompt.

Responsibilities:
  - Accept metrics dict, detected activities, player level
  - Build a structured, context-rich prompt
  - Return a plain string ready for the LLM provider

Pipeline position:
  Metrics → Prompt → LLM → Football Report

No AI calls here — this module only builds text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.prompt_templates.template_loader import TemplateLoader


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class PromptContext:
    """All data needed to build a coaching prompt."""

    # Player context
    player_level:        str                       # "Beginner" | "Intermediate" | "Advanced"
    detected_activities: List[str]                 # e.g. ["shooting", "movement"]

    # Core biomechanical scalars
    torso_lean:          float                     # degrees
    knee_stability:      float                     # 0–100
    gait_symmetry:       float                     # 0–100
    warnings:            List[str]                 # raw warning flags

    # Per-action display metrics
    by_action:           Dict[str, Dict[str, str]] # {"shooting": {"Shot Velocity": "88 km/h"}}

    # Optional session metadata
    session_id:          Optional[str] = None
    video_duration_s:    Optional[float] = None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PERSONA = """\
You are FootballIQ Coach — an elite AI football performance analyst who combines \
the precision of sports science with the clarity of a top-level coach. \
You speak directly to the player in plain, motivating English. \
No jargon, no waffle. Short sentences. Specific feedback only.\
"""

_METRIC_ROW = "  • {label}: {value}"


class PromptBuilder:
    """
    Build structured coaching prompts from metric data.

    Usage::

        builder = PromptBuilder()
        prompt  = builder.build(context)
        # pass prompt to LLMProvider
    """

    def build(self, ctx: PromptContext) -> str:
        """
        Build and return the full LLM prompt string.

        Parameters
        ----------
        ctx : PromptContext

        Returns
        -------
        str — ready-to-send prompt
        """
        sections: List[str] = []

        # ── System persona ────────────────────────────────────────────────
        sections.append(_SYSTEM_PERSONA)
        sections.append("")

        # ── Session overview ──────────────────────────────────────────────
        sections.append("═" * 56)
        sections.append("PLAYER SESSION REPORT")
        sections.append("═" * 56)
        sections.append(f"Skill Level : {ctx.player_level}")
        sections.append(f"Activities  : {', '.join(ctx.detected_activities) or 'General movement'}")
        if ctx.video_duration_s:
            sections.append(f"Duration    : {ctx.video_duration_s:.0f}s")
        sections.append("")

        # ── Core biomechanical metrics ────────────────────────────────────
        sections.append("── Core Biomechanics ──")
        sections.append(_METRIC_ROW.format(label="Torso Lean at Contact",
                                           value=f"{ctx.torso_lean:.1f}°"))
        sections.append(_METRIC_ROW.format(label="Knee Stability Score",
                                           value=f"{ctx.knee_stability:.0f}/100"))
        sections.append(_METRIC_ROW.format(label="Gait Symmetry Score",
                                           value=f"{ctx.gait_symmetry:.0f}/100"))
        sections.append("")

        # ── Active warnings ───────────────────────────────────────────────
        if ctx.warnings:
            sections.append("── Detected Issues ──")
            for w in ctx.warnings:
                sections.append(f"  ⚠ {w}")
            sections.append("")

        # ── Per-activity metrics ──────────────────────────────────────────
        for activity, metrics in ctx.by_action.items():
            if not metrics:
                continue
            sections.append(f"── {activity.capitalize()} Metrics ──")
            for label, value in metrics.items():
                sections.append(_METRIC_ROW.format(label=label, value=value))
            sections.append("")

        # ── Coaching instructions ─────────────────────────────────────────
        sections.append("═" * 56)
        sections.append("YOUR TASK")
        sections.append("═" * 56)
        sections.append(
            "Based on the session data above, produce a coaching report with "
            "exactly these five sections. Use the section headers exactly as shown."
        )
        sections.append("")
        sections.append("1. SUMMARY")
        sections.append(
            "   One paragraph (3–4 sentences). What happened in this session overall? "
            "State the player level and primary activity. Be specific, not generic."
        )
        sections.append("")
        sections.append("2. STRENGTHS")
        sections.append(
            "   Bullet list. What did the player do well? "
            "Reference actual metric values. Minimum 2 points."
        )
        sections.append("")
        sections.append("3. AREAS TO IMPROVE")
        sections.append(
            "   Bullet list. What must the player fix? "
            "Each point must reference a specific metric or warning flag. "
            "Minimum 2 points. Be direct — say exactly what is wrong."
        )
        sections.append("")
        sections.append("4. TRAINING DRILLS")
        sections.append(
            "   For each area to improve, give ONE specific drill. "
            "Format: Drill Name | Instructions | Duration. "
            "No generic advice — every drill must target a specific metric."
        )
        sections.append("")
        sections.append("5. COACH TIP")
        sections.append(
            "   One sentence. The single most important thing the player should "
            f"focus on in their next session as a {ctx.player_level} player."
        )
        sections.append("")
        sections.append(
            "Write for the player directly. Use 'you' and 'your'. "
            "Keep each section concise. No filler words."
        )

        return "\n".join(sections)

    def build_quick_tip(self, ctx: PromptContext) -> str:
        """
        Build a shorter prompt for a single quick coaching tip.
        Used when a full report is not needed.
        """
        warnings_str = ", ".join(ctx.warnings) if ctx.warnings else "none"
        activities_str = ", ".join(ctx.detected_activities) or "general movement"

        return (
            f"{_SYSTEM_PERSONA}\n\n"
            f"Player: {ctx.player_level} | Activities: {activities_str} | "
            f"Warnings: {warnings_str}\n\n"
            f"Give ONE specific, actionable coaching tip in 2 sentences maximum. "
            f"Reference a real metric or warning. No generic advice."
        )
