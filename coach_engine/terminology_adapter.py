#!/usr/bin/env python3
"""
Terminology Adapter  (Language Adapter)
========================================
Rewrites football terminology depending on player level.

Don't say: "Support foot"
Instead:   "The foot you place beside the ball before kicking."

Don't say: "Hip rotation"
Instead:   "Turn your body towards the target as you kick."

Every technical term has a plain-English version for Beginners,
a balanced version for Intermediate, and the correct technical
term for Advanced/Elite.

Usage::

    adapter = TerminologyAdapter()

    # Single term
    plain = adapter.translate("support foot", level="Beginner")
    # → "The foot you place beside the ball before kicking."

    # Entire sentence
    text = adapter.adapt_text(
        "Ensure hip rotation is maximised and your support foot is beside the ball.",
        level="Beginner"
    )
    # → "Turn your body towards the target as you kick and the foot
    #    you place beside the ball before kicking is beside the ball."
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Term dictionary
# ---------------------------------------------------------------------------
# Structure: term → {level: plain_english}
# Keys are lowercase, stripped. Longest terms matched first.

_TERMS: Dict[str, Dict[str, str]] = {

    # ── Foot and contact ─────────────────────────────────────────────────────

    "support foot": {
        "Beginner":     "the foot you place beside the ball before kicking",
        "Developing":   "your plant foot — the one beside the ball",
        "Intermediate": "your plant foot",
        "Advanced":     "plant foot",
        "Elite":        "plant foot",
    },
    "plant foot": {
        "Beginner":     "the foot you put on the ground next to the ball",
        "Developing":   "your non-kicking foot placed beside the ball",
        "Intermediate": "your plant foot",
        "Advanced":     "plant foot",
        "Elite":        "stance foot",
    },
    "kicking foot": {
        "Beginner":     "the foot you use to kick the ball",
        "Developing":   "your striking foot",
        "Intermediate": "your kicking foot",
        "Advanced":     "striking foot",
        "Elite":        "striking foot",
    },
    "weak foot": {
        "Beginner":     "your less-practised foot",
        "Developing":   "your weaker foot — the one you don't usually kick with",
        "Intermediate": "your non-dominant foot",
        "Advanced":     "non-dominant foot",
        "Elite":        "non-dominant foot",
    },
    "dominant foot": {
        "Beginner":     "your stronger kicking foot",
        "Developing":   "your preferred kicking foot",
        "Intermediate": "your dominant foot",
        "Advanced":     "dominant foot",
        "Elite":        "dominant foot",
    },
    "instep": {
        "Beginner":     "the laces area on top of your foot",
        "Developing":   "the top of your foot — the laces area",
        "Intermediate": "the instep — top of your foot",
        "Advanced":     "instep",
        "Elite":        "instep (dorsal surface)",
    },
    "inside of the foot": {
        "Beginner":     "the flat inner side of your foot",
        "Developing":   "the inside of your foot",
        "Intermediate": "inside of the foot",
        "Advanced":     "medial foot surface",
        "Elite":        "medial foot contact",
    },
    "outside of the foot": {
        "Beginner":     "the outer edge of your foot",
        "Developing":   "the outside of your foot",
        "Intermediate": "outside of the foot",
        "Advanced":     "lateral foot surface",
        "Elite":        "lateral foot contact",
    },
    "laces": {
        "Beginner":     "the laces part of your boot — used for powerful kicks",
        "Developing":   "the laces of your boot",
        "Intermediate": "laces contact",
        "Advanced":     "laces (instep drive)",
        "Elite":        "instep drive",
    },

    # ── Body movement ─────────────────────────────────────────────────────────

    "hip rotation": {
        "Beginner":     "turning your body towards the target as you kick",
        "Developing":   "rotating your hips through the ball",
        "Intermediate": "hip rotation through contact",
        "Advanced":     "hip rotation",
        "Elite":        "hip-to-shoulder kinetic chain",
    },
    "follow through": {
        "Beginner":     "continuing to swing your leg after you kick the ball",
        "Developing":   "following through with your leg after contact",
        "Intermediate": "completing your follow-through",
        "Advanced":     "follow-through",
        "Elite":        "post-contact swing arc",
    },
    "torso lean": {
        "Beginner":     "how far your upper body leans forward or backward",
        "Developing":   "your body angle at the moment of contact",
        "Intermediate": "torso lean at contact",
        "Advanced":     "trunk lean angle",
        "Elite":        "trunk lean relative to vertical",
    },
    "body position": {
        "Beginner":     "the way your body is set up before you kick",
        "Developing":   "your body shape before contact",
        "Intermediate": "your body alignment",
        "Advanced":     "body positioning",
        "Elite":        "biomechanical positioning",
    },
    "centre of gravity": {
        "Beginner":     "the balance point of your body",
        "Developing":   "your body's balance point",
        "Intermediate": "your centre of gravity",
        "Advanced":     "centre of mass",
        "Elite":        "COM displacement",
    },
    "weight transfer": {
        "Beginner":     "shifting your body weight as you kick",
        "Developing":   "moving your weight into the ball",
        "Intermediate": "weight transfer through contact",
        "Advanced":     "weight transfer",
        "Elite":        "force transfer mechanics",
    },

    # ── Knee and stability ────────────────────────────────────────────────────

    "valgus collapse": {
        "Beginner":     "your knee falling inward when you kick or land",
        "Developing":   "your knee collapsing inward",
        "Intermediate": "medial knee collapse",
        "Advanced":     "valgus collapse",
        "Elite":        "dynamic valgus",
    },
    "knee stability": {
        "Beginner":     "how steady your knee is when you move",
        "Developing":   "keeping your knee aligned when you move",
        "Intermediate": "knee stability",
        "Advanced":     "knee alignment under load",
        "Elite":        "patellofemoral stability",
    },
    "hip flexor": {
        "Beginner":     "the muscle at the top of your thigh that lifts your leg",
        "Developing":   "the muscle in your upper thigh that drives your knee up",
        "Intermediate": "hip flexor",
        "Advanced":     "hip flexor",
        "Elite":        "hip flexor (iliopsoas)",
    },
    "hip abductor": {
        "Beginner":     "the muscle on the outside of your hip that stabilises your knee",
        "Developing":   "the hip muscle that keeps your knee from collapsing inward",
        "Intermediate": "hip abductor",
        "Advanced":     "hip abductor",
        "Elite":        "gluteus medius / hip abductor complex",
    },

    # ── Running and movement ──────────────────────────────────────────────────

    "gait asymmetry": {
        "Beginner":     "an imbalance between your left and right running steps",
        "Developing":   "uneven strides on your left and right sides",
        "Intermediate": "gait asymmetry",
        "Advanced":     "bilateral gait asymmetry",
        "Elite":        "stride length differential",
    },
    "ground contact time": {
        "Beginner":     "how long your foot stays on the ground each step",
        "Developing":   "the time your foot is on the ground per stride",
        "Intermediate": "ground contact time",
        "Advanced":     "ground contact time (GCT)",
        "Elite":        "GCT — target < 200ms",
    },
    "stride frequency": {
        "Beginner":     "how many steps you take per second",
        "Developing":   "your step rate",
        "Intermediate": "stride frequency",
        "Advanced":     "cadence / stride frequency",
        "Elite":        "stride frequency (spm)",
    },
    "acceleration": {
        "Beginner":     "how quickly you speed up",
        "Developing":   "building speed quickly",
        "Intermediate": "explosive acceleration",
        "Advanced":     "acceleration",
        "Elite":        "first-step acceleration",
    },

    # ── Ball technique ────────────────────────────────────────────────────────

    "first touch": {
        "Beginner":     "what you do with the ball the moment it arrives to you",
        "Developing":   "your first contact with the ball when receiving",
        "Intermediate": "first touch",
        "Advanced":     "first touch quality",
        "Elite":        "first-touch control",
    },
    "touch tightness": {
        "Beginner":     "how close you keep the ball to your feet",
        "Developing":   "keeping the ball tight to your feet while running",
        "Intermediate": "touch tightness",
        "Advanced":     "ball-to-foot proximity",
        "Elite":        "touch tightness index",
    },
    "back-spin": {
        "Beginner":     "spin that makes the ball stop quickly after landing",
        "Developing":   "backspin — the ball decelerates on landing",
        "Intermediate": "backspin",
        "Advanced":     "backspin",
        "Elite":        "retrograde spin (backspin)",
    },
    "top-spin": {
        "Beginner":     "spin that makes the ball dip and bounce forward",
        "Developing":   "topspin — the ball dips and bounces aggressively",
        "Intermediate": "topspin",
        "Advanced":     "topspin",
        "Elite":        "topspin (Magnus effect)",
    },
    "curl": {
        "Beginner":     "making the ball bend in the air after you kick it",
        "Developing":   "swerving the ball using spin",
        "Intermediate": "curl / swerve",
        "Advanced":     "lateral spin curl",
        "Elite":        "Magnus-effect trajectory",
    },

    # ── Tactics ───────────────────────────────────────────────────────────────

    "pressing": {
        "Beginner":     "running at the player with the ball to win it back",
        "Developing":   "pressuring the player with the ball",
        "Intermediate": "pressing",
        "Advanced":     "high press",
        "Elite":        "coordinated press trigger",
    },
    "scanning": {
        "Beginner":     "looking around before the ball comes to you",
        "Developing":   "checking your surroundings before receiving",
        "Intermediate": "pre-scan",
        "Advanced":     "scanning before receiving",
        "Elite":        "pre-reception scanning",
    },
    "transition": {
        "Beginner":     "switching from attacking to defending (or the other way around)",
        "Developing":   "quickly switching between attack and defence",
        "Intermediate": "transition",
        "Advanced":     "transitional phase",
        "Elite":        "in/out of possession transition",
    },

    # ── Training ──────────────────────────────────────────────────────────────

    "plyometrics": {
        "Beginner":     "explosive jumping and hopping exercises",
        "Developing":   "explosive jump training",
        "Intermediate": "plyometrics",
        "Advanced":     "plyometric training",
        "Elite":        "stretch-shortening cycle training",
    },
    "eccentric": {
        "Beginner":     "the part where your muscle lengthens under load (lowering phase)",
        "Developing":   "the lowering/lengthening phase of the exercise",
        "Intermediate": "eccentric phase",
        "Advanced":     "eccentric loading",
        "Elite":        "eccentric contraction",
    },
    "proprioception": {
        "Beginner":     "your body's ability to sense where your limbs are without looking",
        "Developing":   "body awareness during movement",
        "Intermediate": "proprioception",
        "Advanced":     "proprioceptive feedback",
        "Elite":        "neuromuscular proprioception",
    },
}

# Alias map — alternate spellings → canonical key
_ALIASES: Dict[str, str] = {
    "plant foot":      "support foot",
    "kicking leg":     "kicking foot",
    "striking foot":   "kicking foot",
    "weak side":       "weak foot",
    "back spin":       "back-spin",
    "top spin":        "top-spin",
    "instep drive":    "laces",
    "first-touch":     "first touch",
    "touch tight":     "touch tightness",
    "hip turn":        "hip rotation",
    "body turn":       "hip rotation",
    "scan":            "scanning",
    "pre scan":        "scanning",
    "pre-scan":        "scanning",
    "valgus":          "valgus collapse",
    "knee inward":     "valgus collapse",
    "follow-through":  "follow through",
}

_VALID_LEVELS = {"Beginner", "Developing", "Intermediate", "Advanced", "Elite"}


class TerminologyAdapter:
    """
    Rewrites football terminology for the player's skill level.

    Parameters
    ----------
    default_level : str — used when no level is provided
    """

    def __init__(self, default_level: str = "Beginner") -> None:
        self.default_level = default_level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, term: str, level: Optional[str] = None) -> str:
        """
        Translate a single football term into level-appropriate language.

        Parameters
        ----------
        term  : str — the technical term
        level : str — player level

        Returns
        -------
        str — plain English translation, or original if not in dictionary
        """
        level = self._norm_level(level)
        key   = self._norm_key(term)

        # Resolve alias.
        key = _ALIASES.get(key, key)

        entry = _TERMS.get(key)
        if entry is None:
            log.debug("TerminologyAdapter: '%s' not in dictionary", term)
            return term

        return entry.get(level, entry.get("Beginner", term))

    def adapt_text(self, text: str, level: Optional[str] = None) -> str:
        """
        Replace all known technical terms in a string with level-appropriate language.
        Case-insensitive. Longest match first to avoid partial replacements.

        Parameters
        ----------
        text  : str
        level : str

        Returns
        -------
        str — adapted text
        """
        if not text:
            return text

        level = self._norm_level(level)

        # Build sorted key list: longest first, then aliases.
        all_keys = sorted(
            list(_TERMS.keys()) + list(_ALIASES.keys()),
            key=len,
            reverse=True,
        )

        result = text
        for term_key in all_keys:
            canon = _ALIASES.get(term_key, term_key)
            entry = _TERMS.get(canon)
            if entry is None:
                continue
            replacement = entry.get(level, entry.get("Beginner", term_key))
            pattern     = re.compile(re.escape(term_key), re.IGNORECASE)
            result      = pattern.sub(replacement, result)

        return result

    def adapt_recommendation(
        self,
        problem:      str,
        drill_name:   str,
        instructions: str,
        coach_tip:    str,
        level:        Optional[str] = None,
    ) -> Tuple[str, str, str, str]:
        """
        Adapt all parts of a recommendation for the player's level.

        Returns (problem, drill_name, instructions, coach_tip)
        """
        return (
            self.adapt_text(problem,      level),
            self.adapt_text(drill_name,   level),
            self.adapt_text(instructions, level),
            self.adapt_text(coach_tip,    level),
        )

    def list_terms(self) -> list[str]:
        """Return all known term keys."""
        return sorted(_TERMS.keys())

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _norm_level(self, level: Optional[str]) -> str:
        if level is None:
            return self.default_level
        l = level.strip().capitalize()
        return l if l in _VALID_LEVELS else self.default_level

    @staticmethod
    def _norm_key(term: str) -> str:
        return term.strip().lower()
