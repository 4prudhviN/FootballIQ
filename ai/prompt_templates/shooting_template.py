#!/usr/bin/env python3
"""Shooting-specific prompt template."""

from __future__ import annotations

SHOOTING_CONTEXT = """\
ACTIVITY: Shooting
Focus areas: torso lean at contact, launch angle, shot velocity,
foot strike contact type, and target accuracy.
A good shot has: chest over ball, lean < 15°, launch angle 8–20°.\
"""

SHOOTING_COACHING_FOCUS = {
    "Beginner":     "Basic body position and making clean contact with the ball.",
    "Intermediate": "Consistency of technique and accuracy under light pressure.",
    "Advanced":     "Marginal gains in power, spin, and placement under match pressure.",
}
