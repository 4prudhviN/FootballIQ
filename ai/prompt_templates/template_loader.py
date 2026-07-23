#!/usr/bin/env python3
"""
Template Loader
===============
Loads the correct activity-specific prompt template and merges it
with the base template for a given activity and skill level.

Usage::

    loader  = TemplateLoader()
    context = loader.get_context("shooting", level="Beginner")
    focus   = loader.get_coaching_focus("shooting", level="Beginner")
"""

from __future__ import annotations

from typing import Optional

from ai.prompt_templates.base_template     import SYSTEM_PERSONA, OUTPUT_FORMAT_INSTRUCTION
from ai.prompt_templates.shooting_template import SHOOTING_CONTEXT, SHOOTING_COACHING_FOCUS
from ai.prompt_templates.passing_template  import PASSING_CONTEXT, PASSING_COACHING_FOCUS
from ai.prompt_templates.movement_template import MOVEMENT_CONTEXT, MOVEMENT_COACHING_FOCUS
from utils.logger import get_logger

log = get_logger(__name__)

_CONTEXT_MAP = {
    "shooting":    SHOOTING_CONTEXT,
    "passing":     PASSING_CONTEXT,
    "movement":    MOVEMENT_CONTEXT,
    "dribbling":   "ACTIVITY: Dribbling\nFocus areas: close control, change of direction, ball speed.",
    "defending":   "ACTIVITY: Defending\nFocus areas: tackle timing, positioning, interceptions.",
    "goalkeeping": "ACTIVITY: Goalkeeping\nFocus areas: reaction time, diving range, distribution.",
    "general":     "ACTIVITY: General movement analysis.",
}

_FOCUS_MAP = {
    "shooting":    SHOOTING_COACHING_FOCUS,
    "passing":     PASSING_COACHING_FOCUS,
    "movement":    MOVEMENT_COACHING_FOCUS,
}

_DEFAULT_FOCUS = {
    "Beginner":     "Building fundamental technique.",
    "Intermediate": "Improving consistency and decision-making.",
    "Advanced":     "Marginal gains and performance under pressure.",
}


class TemplateLoader:
    """Loads and assembles prompt templates."""

    @staticmethod
    def get_system_persona() -> str:
        return SYSTEM_PERSONA

    @staticmethod
    def get_output_format() -> str:
        return OUTPUT_FORMAT_INSTRUCTION

    @staticmethod
    def get_context(activity: str, level: Optional[str] = None) -> str:
        """Return the activity-specific context block."""
        return _CONTEXT_MAP.get(activity.lower(), _CONTEXT_MAP["general"])

    @staticmethod
    def get_coaching_focus(activity: str, level: str = "Beginner") -> str:
        """Return the level-specific coaching focus for this activity."""
        focus_map = _FOCUS_MAP.get(activity.lower(), _DEFAULT_FOCUS)
        return focus_map.get(level, focus_map.get("Beginner", "Focus on fundamentals."))

    @staticmethod
    def assemble(
        activity:  str,
        level:     str,
        metrics_block: str,
        warnings_block: str,
    ) -> str:
        """
        Assemble a complete prompt from all template components.

        Parameters
        ----------
        activity       : str — football action (e.g. "shooting")
        level          : str — player level
        metrics_block  : str — pre-formatted metric section
        warnings_block : str — pre-formatted warnings section

        Returns
        -------
        str — complete prompt ready to send to LLM
        """
        loader  = TemplateLoader
        context = loader.get_context(activity, level)
        focus   = loader.get_coaching_focus(activity, level)

        return "\n\n".join([
            loader.get_system_persona(),
            context,
            f"PLAYER LEVEL: {level}",
            f"COACHING FOCUS: {focus}",
            metrics_block,
            warnings_block,
            loader.get_output_format(),
        ])
