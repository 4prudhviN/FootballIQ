#!/usr/bin/env python3
"""
Prompt Builder
==============
Builds the LLM prompt from structured football knowledge.

Instead of sending raw metrics, sends structured coaching data:

  Player Level      : Developing
  Detected Activities: Passing
  Metrics           : Passing Accuracy 82%
  Detected Mistakes : Body leaning, Plant foot position
  Recommended Drill : Wall Passing
  Tone              : Explain like a coach to a beginner.

The LLM receives structured knowledge and rewrites it into
natural language. It does NOT invent football advice.

Pipeline position:
  SessionSummary + Recommendations → PromptBuilder → LLM
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
    """All structured data the LLM needs to rewrite."""
    player_level:         str
    detected_activities:  List[str]
    torso_lean:           float
    knee_stability:       float
    gait_symmetry:        float
    warnings:             List[str]
    by_action:            Dict[str, Dict[str, str]]

    # Structured coaching findings (from FeedbackEngine / MistakeDetector)
    coaching_issues:     List[Dict[str, Any]] = field(default_factory=list)
    # Each: {metric, value, root_cause, correction, drill, coach_tip}

    positive_findings:   List[str]            = field(default_factory=list)
    mistake_causes:      List[str]            = field(default_factory=list)
    top_drill:           Optional[str]        = None
    top_coach_tip:       Optional[str]        = None
    session_id:          Optional[str]        = None
    video_duration_s:    Optional[float]      = None


# ---------------------------------------------------------------------------
# Tone instructions per level
# ---------------------------------------------------------------------------

_TONE: Dict[str, str] = {
    "Beginner":     "Explain like a coach talking to a beginner. Use plain everyday language. No jargon. Short sentences. Be encouraging.",
    "Developing":   "Explain like a coach to a developing player. Simple language with some football terms explained. Be direct and positive.",
    "Intermediate": "Speak as a coach to an intermediate player. Use standard football terminology. Be clear and specific.",
    "Advanced":     "Speak as a technical analyst to an advanced player. Use correct football terminology. Be precise and concise.",
    "Elite":        "Speak as a performance analyst to an elite player. Use technical language. Focus on marginal gains. Be data-driven.",
}

_SYSTEM = """\
You are FootballIQ Coach — rewriting structured football analysis into natural language.

STRICT RULES:
1. Every coaching point you write MUST come from the structured data provided below.
2. Do NOT invent new football advice, drills, or observations.
3. Reference actual numbers where given (e.g. "82% passing accuracy").
4. Write in the tone specified — match it exactly.
5. Respond with valid JSON only — no text outside the JSON.\
"""

_OUTPUT_FORMAT = """\
Respond with valid JSON only:
{
  "summary":         "2-3 sentences. Reference actual numbers. What happened this session.",
  "strengths":       ["one sentence per strength, referencing a specific metric or observation"],
  "weaknesses":      ["one sentence per issue, naming the root cause and its effect on play"],
  "coachingTips":    ["one actionable instruction per issue, written directly to the player"],
  "motivationalTip": "one closing sentence, level-appropriate, encouraging"
}\
"""


class PromptBuilder:
    """
    Builds the LLM prompt from structured football knowledge.
    The prompt tells the LLM exactly what to rewrite and in what tone.
    """

    def __init__(self) -> None:
        self._templates = TemplateLoader()

    def build(self, ctx: PromptContext) -> str:
        """Build the full structured prompt."""
        level = ctx.player_level
        tone  = _TONE.get(level, _TONE["Beginner"])
        sections: List[str] = []

        sections.append(_SYSTEM)
        sections.append("")

        # ── Structured coaching data ───────────────────────────────────────
        sections.append("═" * 56)
        sections.append("STRUCTURED FOOTBALL KNOWLEDGE TO REWRITE")
        sections.append("═" * 56)

        sections.append(f"Player Level       : {level}")
        sections.append(f"Detected Activities: {', '.join(ctx.detected_activities) or 'General movement'}")
        if ctx.video_duration_s:
            sections.append(f"Session Duration   : {ctx.video_duration_s:.0f} seconds")
        sections.append("")

        # ── Metrics (actual numbers) ───────────────────────────────────────
        if ctx.by_action:
            sections.append("── Metrics ──")
            for activity, metrics in ctx.by_action.items():
                sections.append(f"  {activity.capitalize()}:")
                for label, value in metrics.items():
                    sections.append(f"    {label}: {value}")
            sections.append("")

        sections.append("── Biomechanical Readings ──")
        sections.append(f"  Torso lean at contact : {ctx.torso_lean:.1f}°")
        sections.append(f"  Knee stability        : {ctx.knee_stability:.0f}/100")
        sections.append(f"  Gait symmetry         : {ctx.gait_symmetry:.0f}/100")
        sections.append("")

        # ── Detected mistakes ──────────────────────────────────────────────
        if ctx.mistake_causes:
            sections.append("── Detected Mistakes ──")
            for cause in ctx.mistake_causes:
                sections.append(f"  • {cause}")
            sections.append("")

        # ── Coaching issues with corrections ──────────────────────────────
        if ctx.coaching_issues:
            sections.append("── Coaching Issues (rewrite each into natural language) ──")
            for i, issue in enumerate(ctx.coaching_issues, 1):
                sections.append(f"  Issue {i}: {issue.get('metric', '').replace('_', ' ').title()}")
                sections.append(f"    Value     : {issue.get('value', '')}")
                sections.append(f"    Root cause: {issue.get('root_cause', '')}")
                sections.append(f"    Correction: {issue.get('correction', '')}")
                if issue.get("drill"):
                    sections.append(f"    Drill     : {issue.get('drill', '')}")
                if issue.get("coach_tip"):
                    sections.append(f"    Coach tip : {issue.get('coach_tip', '')}")
                sections.append("")

        # ── Recommended drill ──────────────────────────────────────────────
        if ctx.top_drill:
            sections.append("── Recommended Drill ──")
            sections.append(f"  {ctx.top_drill}")
            sections.append("")

        # ── Top coach tip ──────────────────────────────────────────────────
        if ctx.top_coach_tip:
            sections.append("── Key Coach Tip ──")
            sections.append(f"  {ctx.top_coach_tip}")
            sections.append("")

        # ── What the player did well ───────────────────────────────────────
        if ctx.positive_findings:
            sections.append("── What the Player Did Well ──")
            for p in ctx.positive_findings:
                sections.append(f"  ✓ {p}")
            sections.append("")

        # ── Tone instruction ───────────────────────────────────────────────
        sections.append("═" * 56)
        sections.append("TONE INSTRUCTION")
        sections.append("═" * 56)
        sections.append(tone)
        sections.append("")

        # ── Output format ──────────────────────────────────────────────────
        sections.append("═" * 56)
        sections.append("OUTPUT FORMAT")
        sections.append("═" * 56)
        sections.append(_OUTPUT_FORMAT)

        return "\n".join(sections)

    def build_quick_tip(self, ctx: PromptContext) -> str:
        """Build a short prompt for a single coaching tip."""
        tone  = _TONE.get(ctx.player_level, _TONE["Beginner"])
        issue = ctx.coaching_issues[0] if ctx.coaching_issues else None

        if not issue:
            return (
                f"{_SYSTEM}\n\n"
                f"Player: {ctx.player_level}\n"
                f"Activities: {', '.join(ctx.detected_activities)}\n\n"
                f"Tone: {tone}\n\n"
                f"Rewrite this into ONE plain sentence: 'Good session overall — keep training consistently.'"
            )

        return (
            f"{_SYSTEM}\n\n"
            f"Player Level      : {ctx.player_level}\n"
            f"Detected Activity : {ctx.detected_activities[0] if ctx.detected_activities else 'general'}\n"
            f"Issue             : {issue.get('metric', '').replace('_', ' ').title()} = {issue.get('value', '')}\n"
            f"Root Cause        : {issue.get('root_cause', '')}\n"
            f"Correction        : {issue.get('correction', '')}\n"
            f"Recommended Drill : {ctx.top_drill or issue.get('drill', 'N/A')}\n\n"
            f"Tone: {tone}\n\n"
            f"Rewrite the above into ONE plain sentence (max 30 words). "
            f"Reference the actual value. Speak directly to the player."
        )
