#!/usr/bin/env python3
"""Movement-specific prompt template."""

from __future__ import annotations

MOVEMENT_CONTEXT = """\
ACTIVITY: Movement
Focus areas: gait symmetry, sprint speed, stride length,
acceleration, and distance covered.
Good movement has: symmetric stride, efficient ground contact, explosive acceleration.\
"""

MOVEMENT_COACHING_FOCUS = {
    "Beginner":     "Building a natural, balanced running pattern.",
    "Intermediate": "Improving stride efficiency and reducing gait asymmetry.",
    "Advanced":     "Optimising sprint mechanics and reactive acceleration.",
}
