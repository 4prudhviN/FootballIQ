#!/usr/bin/env python3
"""Passing-specific prompt template."""

from __future__ import annotations

PASSING_CONTEXT = """\
ACTIVITY: Passing
Focus areas: pass accuracy, first touch, ball control,
weight of pass, and weak foot usage.
A good pass has: clean contact, appropriate weight, correct foot selection.\
"""

PASSING_COACHING_FOCUS = {
    "Beginner":     "Making clean contact and getting the ball to a teammate.",
    "Intermediate": "Weight of pass, timing, and using both feet.",
    "Advanced":     "Disguise, through-balls, and execution under defensive pressure.",
}
