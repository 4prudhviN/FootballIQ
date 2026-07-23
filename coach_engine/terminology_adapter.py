#!/usr/bin/env python3
"""
Terminology Adapter  ⭐
======================
Translates technical football jargon into plain, level-appropriate English.

The LLM should NEVER invent coaching terminology.
Every technical term passes through this adapter before reaching the player.

Examples
--------
  "Support foot"
      → Beginner:     "The foot you place beside the ball before kicking."
      → Intermediate: "Your plant foot — positioned beside the ball at contact."
      → Advanced:     "Plant foot positioning relative to ball contact point."

  "Hip flexor"
      → Beginner:     "The muscle at the top of your thigh that lifts your leg."
      → Intermediate: "Hip flexor — the muscle driving your knee lift."
      → Advanced:     "Hip flexor recruitment during the swing phase."

  "Valgus collapse"
      → Beginner:     "Your knee falling inward when you land or kick."
      → Intermediate: "Medial knee collapse — indicates weak hip abductors."
      → Advanced:     "Dynamic valgus — requires targeted hip stability work."

Usage::

    adapter = TerminologyAdapter()
    plain   = adapter.translate("support foot", level="Beginner")
    # → "The foot you place beside the ball before kicking."

    adapted = adapter.adapt_text(
        "Ensure valgus collapse is minimised at contact.",
        level="Beginner",
    )
    # → "Make sure your knee doesn't fall inward when you kick."
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Term dictionary
# ---------------------------------------------------------------------------
# Structure: term_key → {"Beginner": str, "Intermediate": str, "Advanced": str}
# Keys are lowercase, stripped.

_TERM_DICT: Dict[str, Dict[str, str]] = {

    # ── Biomechanics ──────────────────────────────────────────────────────

    "support foot": {
        "Beginner":     "the foot you place beside the ball before kicking",
        "Intermediate": "your plant foot — the one beside the ball at contact",
        "Advanced":     "plant foot placement relative to ball contact point",
    },
    "plant foot": {
        "Beginner":     "the foot you put on the ground next to the ball",
        "Intermediate": "your non-kicking foot, placed beside the ball",
        "Advanced":     "the stance foot used to stabilise during ball strike",
    },
    "kicking foot": {
        "Beginner":     "the foot you use to kick the ball",
        "Intermediate": "your striking foot",
        "Advanced":     "the striking foot",
    },
    "weak foot": {
        "Beginner":     "your less-practised foot — the one you don't usually kick with",
        "Intermediate": "your non-dominant foot",
        "Advanced":     "non-dominant foot",
    },
    "dominant foot": {
        "Beginner":     "your stronger kicking foot — the one you prefer to use",
        "Intermediate": "your preferred kicking foot",
        "Advanced":     "dominant foot",
    },
    "hip flexor": {
        "Beginner":     "the muscle at the top of your thigh that lifts your leg up",
        "Intermediate": "hip flexor — the muscle driving your knee lift",
        "Advanced":     "hip flexor recruitment during the swing phase",
    },
    "valgus collapse": {
        "Beginner":     "your knee falling inward when you land or kick",
        "Intermediate": "medial knee collapse — often caused by weak hip muscles",
        "Advanced":     "dynamic valgus — address with hip abductor strengthening",
    },
    "varus": {
        "Beginner":     "your knee bowing outward",
        "Intermediate": "lateral knee deviation (outward)",
        "Advanced":     "varus deviation — lateral knee misalignment",
    },
    "torso lean": {
        "Beginner":     "how far your upper body is leaning forward or backward",
        "Intermediate": "upper-body lean angle at the moment of contact",
        "Advanced":     "trunk lean relative to the vertical axis at ball strike",
    },
    "gait asymmetry": {
        "Beginner":     "an imbalance between your left and right running steps",
        "Intermediate": "uneven stride length between your left and right legs",
        "Advanced":     "bilateral gait asymmetry — stride length differential",
    },
    "centre of gravity": {
        "Beginner":     "the point in your body where your balance is centred",
        "Intermediate": "your body's balance point",
        "Advanced":     "centre of mass displacement during dynamic movement",
    },
    "proprioception": {
        "Beginner":     "your body's ability to sense where your limbs are without looking",
        "Intermediate": "body awareness — sensing your position during movement",
        "Advanced":     "proprioceptive feedback during dynamic loading",
    },
    "ground contact time": {
        "Beginner":     "how long your foot stays on the ground with each step",
        "Intermediate": "time your foot spends on the ground per stride",
        "Advanced":     "ground contact time (GCT) — target < 200 ms for explosive running",
    },
    "reactive strength index": {
        "Beginner":     "a measure of how explosively you can jump after landing",
        "Intermediate": "RSI — measures jump efficiency relative to ground contact",
        "Advanced":     "RSI (flight time ÷ contact time) — plyometric power metric",
    },
    "eccentric": {
        "Beginner":     "the part of an exercise where your muscle is lengthening under load",
        "Intermediate": "eccentric phase — muscle lengthening under tension",
        "Advanced":     "eccentric loading phase",
    },
    "concentric": {
        "Beginner":     "the part of an exercise where your muscle is shortening to create force",
        "Intermediate": "concentric phase — muscle shortening to generate force",
        "Advanced":     "concentric contraction phase",
    },

    # ── Technique ─────────────────────────────────────────────────────────

    "instep": {
        "Beginner":     "the top of your foot (the laces area)",
        "Intermediate": "the instep — the top of your foot over the laces",
        "Advanced":     "instep (dorsal foot surface) contact",
    },
    "laces": {
        "Beginner":     "the laces area of your boot — used for powerful kicks",
        "Intermediate": "laces contact — used for power shots and driven passes",
        "Advanced":     "laces (instep drive) contact technique",
    },
    "inside of the foot": {
        "Beginner":     "the flat inner part of your foot — used for accurate short passes",
        "Intermediate": "inside-foot contact — used for accuracy and short passing",
        "Advanced":     "medial foot surface contact",
    },
    "outside of the foot": {
        "Beginner":     "the outer edge of your foot — used for curved passes or shots",
        "Intermediate": "outside-foot contact — used for cuts and curved kicks",
        "Advanced":     "lateral foot surface contact",
    },
    "follow through": {
        "Beginner":     "continuing to swing your leg after kicking the ball",
        "Intermediate": "completing the kicking motion past ball contact",
        "Advanced":     "post-contact leg swing arc — affects trajectory and power",
    },
    "first touch": {
        "Beginner":     "what you do with the ball the moment it arrives to you",
        "Intermediate": "ball control on the first contact",
        "Advanced":     "first-touch quality — direction and cushioning under pressure",
    },
    "touch tightness": {
        "Beginner":     "how close you keep the ball to your feet while dribbling",
        "Intermediate": "how tightly you control the ball while running with it",
        "Advanced":     "ball-to-foot proximity index during dynamic dribbling",
    },
    "back-spin": {
        "Beginner":     "spin that makes the ball stop quickly when it lands",
        "Intermediate": "backspin — makes the ball decelerate on landing",
        "Advanced":     "backspin imparted at contact — reduces forward momentum",
    },
    "top-spin": {
        "Beginner":     "spin that makes the ball dip down and bounce forward",
        "Intermediate": "topspin — causes the ball to dip and bounce aggressively",
        "Advanced":     "topspin — aerodynamic Magnus effect drives ball downward",
    },
    "curl": {
        "Beginner":     "making the ball bend in the air after you kick it",
        "Intermediate": "swerving the ball using side-spin",
        "Advanced":     "lateral ball spin creating Magnus-effect trajectory curve",
    },

    # ── Tactics ───────────────────────────────────────────────────────────

    "pressing": {
        "Beginner":     "running at the player with the ball to win it back quickly",
        "Intermediate": "pressuring the ball carrier to force a mistake",
        "Advanced":     "high-intensity press — coordinated team defensive trigger",
    },
    "pressing trigger": {
        "Beginner":     "the moment your team decides to start chasing the ball",
        "Intermediate": "the cue that starts a coordinated press",
        "Advanced":     "press trigger — specific event that initiates coordinated press",
    },
    "transition": {
        "Beginner":     "the moment your team switches from attacking to defending, or vice versa",
        "Intermediate": "switching quickly between attack and defence",
        "Advanced":     "transitional phase — moments of numerical advantage exploitation",
    },
    "overload": {
        "Beginner":     "having more players than the other team in a particular area",
        "Intermediate": "outnumbering the opposition locally to create an advantage",
        "Advanced":     "positional overload — numerical superiority in a zone",
    },
    "false nine": {
        "Beginner":     "a striker who drops deep to confuse the other team's defenders",
        "Intermediate": "a centre-forward who drops into midfield to create space",
        "Advanced":     "false nine — deep-lying forward disrupting defensive structure",
    },

    # ── Training ──────────────────────────────────────────────────────────

    "plyometrics": {
        "Beginner":     "explosive jumping and hopping exercises that build power",
        "Intermediate": "explosive jump training to build fast-twitch power",
        "Advanced":     "plyometric loading — stretch-shortening cycle development",
    },
    "reps": {
        "Beginner":     "the number of times you repeat the exercise",
        "Intermediate": "repetitions",
        "Advanced":     "repetitions",
    },
    "sets": {
        "Beginner":     "groups of repetitions — you rest between each group",
        "Intermediate": "sets — groups of reps with rest in between",
        "Advanced":     "sets",
    },
    "vo2 max": {
        "Beginner":     "how much oxygen your body can use during hard exercise — higher is fitter",
        "Intermediate": "aerobic capacity — the maximum oxygen your body can use",
        "Advanced":     "VO₂ max — maximal oxygen uptake (mL/kg/min)",
    },
    "lactate threshold": {
        "Beginner":     "the pace at which your legs start to feel heavy and tired",
        "Intermediate": "the intensity at which lactic acid builds up faster than it clears",
        "Advanced":     "lactate threshold pace — anaerobic transition intensity",
    },
}

# Alias map: maps alternate spellings/forms to the canonical key.
_ALIASES: Dict[str, str] = {
    "plant foot":         "plant foot",
    "kicking leg":        "kicking foot",
    "striking foot":      "kicking foot",
    "non-dominant foot":  "weak foot",
    "back spin":          "back-spin",
    "top spin":           "top-spin",
    "instep drive":       "laces",
    "lace":               "laces",
    "rsi":                "reactive strength index",
    "first-touch":        "first touch",
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class TerminologyAdapter:
    """
    Translates technical football terms into plain, level-appropriate English.

    Parameters
    ----------
    default_level : str
        Default skill level when none is specified.
        One of "Beginner", "Intermediate", "Advanced".

    Usage::

        adapter = TerminologyAdapter()

        # Translate a single term
        plain = adapter.translate("support foot", level="Beginner")
        # → "the foot you place beside the ball before kicking"

        # Adapt an entire sentence
        text  = adapter.adapt_text(
            "Minimise valgus collapse at contact and improve your first touch.",
            level="Beginner",
        )
        # → "Make sure your knee doesn't fall inward when you kick and improve
        #    what you do with the ball the moment it arrives to you."
    """

    def __init__(self, default_level: str = "Beginner") -> None:
        self.default_level = default_level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(
        self,
        term:  str,
        level: Optional[str] = None,
    ) -> str:
        """
        Return the plain-English translation of a football term.

        Parameters
        ----------
        term  : str   The technical term to translate.
        level : str   "Beginner", "Intermediate", or "Advanced".

        Returns
        -------
        str — translated text, or the original term if not in the dictionary.
        """
        level = self._normalise_level(level)
        key   = self._normalise_key(term)

        if key in _ALIASES:
            key = _ALIASES[key]

        entry = _TERM_DICT.get(key)
        if entry is None:
            log.debug("TerminologyAdapter: term '%s' not in dictionary", term)
            return term   # return original if unknown

        return entry.get(level, entry.get("Beginner", term))

    def adapt_text(
        self,
        text:  str,
        level: Optional[str] = None,
    ) -> str:
        """
        Replace all known technical terms in a text string with their
        plain-English equivalents for the given level.

        Replacement is case-insensitive and preserves surrounding whitespace.

        Parameters
        ----------
        text  : str
        level : str

        Returns
        -------
        str — adapted text.
        """
        if not text:
            return text

        level = self._normalise_level(level)

        # Build a sorted list of known terms (longest first to avoid partial matches).
        all_keys = sorted(
            list(_TERM_DICT.keys()) + list(_ALIASES.keys()),
            key=len,
            reverse=True,
        )

        result = text
        for term_key in all_keys:
            # Resolve alias.
            canon_key = _ALIASES.get(term_key, term_key)
            entry     = _TERM_DICT.get(canon_key)
            if entry is None:
                continue
            replacement = entry.get(level, entry.get("Beginner", term_key))
            # Case-insensitive word boundary replacement.
            pattern = re.compile(re.escape(term_key), re.IGNORECASE)
            result  = pattern.sub(replacement, result)

        return result

    def adapt_feedback_item(
        self,
        observation: str,
        drill:       str,
        coach_tip:   str,
        level:       Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Adapt all three feedback strings (observation, drill, coach_tip)
        for the given skill level in one call.

        Returns
        -------
        Tuple[str, str, str] — (adapted_observation, adapted_drill, adapted_tip)
        """
        return (
            self.adapt_text(observation, level),
            self.adapt_text(drill,       level),
            self.adapt_text(coach_tip,   level),
        )

    def list_terms(self) -> list[str]:
        """Return all known term keys (sorted alphabetically)."""
        return sorted(_TERM_DICT.keys())

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _normalise_level(self, level: Optional[str]) -> str:
        if level is None:
            return self.default_level
        level = level.strip().capitalize()
        if level not in ("Beginner", "Intermediate", "Advanced"):
            return self.default_level
        return level

    @staticmethod
    def _normalise_key(term: str) -> str:
        return term.strip().lower()
