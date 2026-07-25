#!/usr/bin/env python3
"""
Prompt Builder
==============
Builds the LLM prompt from structured coaching data.

The LLM's ONLY job is to rewrite structured findings into natural language.
It must NOT invent new football advice — all coaching points come from the
FeedbackEngine and knowledge base.

Instead of sending:
  "Passing Accuracy: 61%"

We send:
  "The player completed 19 out of 31 passes successfully.
   Root cause: Body leaning backward before striking.
   Correction: Keep chest facing the target.
   Drill: Wall rebounder drill."

The LLM rewrites this into:
  "You completed 19 out of 31 passes successfully. Most missed passes
   happened because your body leaned backward before striking the ball.
   Keeping your chest facing the target will improve accuracy."

Pipeline position:
  FeedbackReport + Metrics → PromptBuilder → LLM → natural language
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.prompt_templates.template_loader import TemplateLoader


# ---------------------------------------------------------------------------
# Input context
# ---------------------------------------------------------------------------

@dataclass
class PromptContext:
    """All structured data needed to build the rewrite prompt."""
    player_level:         str
    detected_activities:  List[str]
    torso_lean:           float
    knee_stability:       float
    gait_symmetry:        float
    warnings:             List[str]
    by_action:            Dict[str, Dict[str, str]]

    # Structured coaching findings from FeedbackEngine (not raw metrics)
    coaching_issues:      List[Dict[str, Any]] = field(default_factory=list)
    # Each: {metric, value, root_cause, observation, correction, drill, coach_tip}

    positive_findings:    List[str]            = field(default_factory=list)
    session_id:           Optional[str]        = None
    video_duration_s:     Optional[float]      = None


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are a football performance analyst rewriting structured coaching data into natural language.

RULES — you must follow these exactly:
1. You are a TRANSLATOR only. Every coaching point you write must come from the data provided.
2. Do NOT invent new coaching advice, drills, or observations.
3. Do NOT use technical jargon — write in plain English the player can understand.
4. Reference actual numbers from the data (e.g. "19 out of 31 passes").
5. Be direct and specific. No filler phrases.
6. Tone: supportive but honest. Like a coach who respects the player's time.\
"""

_OUTPUT_FORMAT = """\
Respond with valid JSON only:
{
  "summary":         "2-3 sentences. What happened in this session overall, referencing actual numbers.",
  "strengths":       ["one sentence per strength, referencing a specific metric value"],
  "weaknesses":      ["one sentence per issue, naming the root cause and its effect on performance"],
  "coachingTips":    ["one actionable tip per issue, written as a direct instruction"],
  "motivationalTip": "one sentence closing message appropriate for the player's level"
}
Do not include any text outside the JSON object.\
"""


class PromptBuilder:
    """
    Builds LLM prompts from structured coaching data.

    The prompt tells the LLM exactly what to rewrite and in what format.
    The LLM adds no new football knowledge — only natural language.
    """

    def __init__(self) -> None:
        self._templates = TemplateLoader()

    def build(self, ctx: PromptContext) -> str:
        """
        Build the full rewrite prompt from structured coaching context.

        Parameters
        ----------
        ctx : PromptContext

        Returns
        -------
        str — complete prompt ready to send to LLM
        """
        sections: List[str] = []

        sections.append(_SYSTEM_INSTRUCTION)
        sections.append("")

        # ── Session context ────────────────────────────────────────────────
        sections.append("═" * 56)
        sections.append("SESSION DATA TO REWRITE")
        sections.append("═" * 56)
        sections.append(f"Player Level : {ctx.player_level}")
        sections.append(f"Activities   : {', '.join(ctx.detected_activities) or 'General movement'}")
        if ctx.video_duration_s:
            sections.append(f"Duration     : {ctx.video_duration_s:.0f} seconds")
        sections.append("")

        # ── Per-action metrics (actual numbers) ────────────────────────────
        if ctx.by_action:
            sections.append("── Measured Metrics ──")
            for activity, metrics in ctx.by_action.items():
                sections.append(f"  {activity.capitalize()}:")
                for label, value in metrics.items():
                    sections.append(f"    • {label}: {value}")
            sections.append("")

        # ── Biomechanical readings ─────────────────────────────────────────
        sections.append("── Biomechanical Readings ──")
        sections.append(f"  • Torso lean at contact : {ctx.torso_lean:.1f}°  "
                        f"({'poor — lean back' if ctx.torso_lean > 15 else 'good'})")
        sections.append(f"  • Knee stability score  : {ctx.knee_stability:.0f}/100")
        sections.append(f"  • Gait symmetry score   : {ctx.gait_symmetry:.0f}/100")
        sections.append("")

        # ── Structured coaching issues (from FeedbackEngine) ──────────────
        if ctx.coaching_issues:
            sections.append("── Coaching Issues Found (rewrite each into natural language) ──")
            for i, issue in enumerate(ctx.coaching_issues, 1):
                sections.append(f"  Issue {i}: {issue.get('metric', '').replace('_', ' ').title()}")
                sections.append(f"    Value      : {issue.get('value', '')}")
                sections.append(f"    Root cause : {issue.get('root_cause', '')}")
                sections.append(f"    Observation: {issue.get('observation', '')}")
                sections.append(f"    Correction : {issue.get('correction', '')}")
                sections.append(f"    Drill      : {issue.get('drill', '')}")
                sections.append(f"    Coach tip  : {issue.get('coach_tip', '')}")
                sections.append("")

        # ── Positive findings ──────────────────────────────────────────────
        if ctx.positive_findings:
            sections.append("── What the Player Did Well ──")
            for p in ctx.positive_findings:
                sections.append(f"  ✓ {p}")
            sections.append("")

        # ── Rewrite instructions ───────────────────────────────────────────
        sections.append("═" * 56)
        sections.append("YOUR TASK: REWRITE THE ABOVE INTO NATURAL LANGUAGE")
        sections.append("═" * 56)
        sections.append(
            f"Rewrite the session data above into clear, plain English for a "
            f"{ctx.player_level} football player. "
            "Use the actual numbers. Reference the root causes. "
            "Do not add any coaching advice that is not in the data above."
        )
        sections.append("")
        sections.append(_OUTPUT_FORMAT)

        return "\n".join(sections)

    def build_quick_tip(self, ctx: PromptContext) -> str:
        """
        Build a short rewrite prompt for a single coaching tip.
        Used for quick feedback without a full report.
        """
        if not ctx.coaching_issues:
            return (
                f"{_SYSTEM_INSTRUCTION}\n\n"
                f"Player: {ctx.player_level} | "
                f"Activities: {', '.join(ctx.detected_activities)}\n\n"
                f"Rewrite this into one plain English sentence: "
                f"'Good session overall with no major issues detected.'"
            )

        issue = ctx.coaching_issues[0]
        return (
            f"{_SYSTEM_INSTRUCTION}\n\n"
            f"Player: {ctx.player_level}\n"
            f"Issue: {issue.get('metric')} = {issue.get('value')}\n"
            f"Root cause: {issue.get('root_cause')}\n"
            f"Correction: {issue.get('correction')}\n\n"
            f"Rewrite the above into ONE plain English sentence (max 30 words). "
            f"Reference the actual value. Use 'you' and 'your'. No jargon."
        )
