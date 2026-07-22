#!/usr/bin/env python3
"""
Common Metrics
==============
Shared data structures and utility functions used by all activity-specific
metric calculators.

No AI, no feedback — pure mathematics only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

@dataclass
class Point2D:
    """A 2-D coordinate in normalised [0, 1] space."""
    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Vector2D:
    """A 2-D direction vector."""
    dx: float
    dy: float

    @property
    def magnitude(self) -> float:
        return math.hypot(self.dx, self.dy)

    def angle_deg(self) -> float:
        """Angle from positive x-axis, degrees."""
        return math.degrees(math.atan2(self.dy, self.dx))


@dataclass
class MetricValue:
    """A single calculated metric with label, numeric value, and display string."""
    label:   str
    value:   float
    display: str         # e.g. "87%", "4.2s", "88 km/h"
    unit:    str = ""


@dataclass
class MetricSet:
    """Collection of MetricValues for one activity instance."""
    activity:  str
    metrics:   List[MetricValue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, str]:
        """Return {label: display} dict — consumed by the frontend."""
        return {m.label: m.display for m in self.metrics}

    def to_numeric_dict(self) -> Dict[str, float]:
        """Return {label: float} — consumed by skill_classifier.py."""
        return {m.label: m.value for m in self.metrics}


# ---------------------------------------------------------------------------
# Shared geometric helpers
# ---------------------------------------------------------------------------

def angle_between_points(
    a: Point2D,
    vertex: Point2D,
    b: Point2D,
) -> float:
    """
    Interior angle at `vertex` formed by rays vertex→a and vertex→b.
    Returns degrees in [0, 180].
    """
    v1 = np.array([a.x - vertex.x, a.y - vertex.y])
    v2 = np.array([b.x - vertex.x, b.y - vertex.y])
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / mag, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def perpendicular_deviation(
    hip: Point2D,
    knee: Point2D,
    ankle: Point2D,
) -> float:
    """
    Perpendicular deviation of `knee` from the hip-ankle line,
    expressed as a ratio of thigh length (hip→knee distance).
    Positive = medial (valgus), negative = lateral (varus).
    """
    A = np.array([hip.x,   hip.y])
    B = np.array([knee.x,  knee.y])
    C = np.array([ankle.x, ankle.y])
    thigh = float(np.linalg.norm(B - A))
    if thigh < 1e-9:
        return 0.0
    cross = float(np.cross(C - A, B - A))
    span  = float(np.linalg.norm(C - A))
    if span < 1e-9:
        return 0.0
    return (cross / span) / thigh


def speed_px_per_frame(
    positions: Sequence[Point2D],
) -> List[float]:
    """
    Frame-to-frame speed in normalised units/frame.
    Returns an empty list if fewer than 2 positions are given.
    """
    speeds = []
    for i in range(1, len(positions)):
        speeds.append(positions[i].distance_to(positions[i - 1]))
    return speeds


def symmetry_ratio(left: float, right: float) -> float:
    """
    Symmetry ratio between two values.
    Returns 1.0 for perfect symmetry, 0.0 for complete imbalance.
    """
    denom = max(abs(left), abs(right))
    if denom < 1e-9:
        return 1.0
    return 1.0 - abs(left - right) / denom


def percentage(numerator: float, denominator: float) -> float:
    """Safe percentage calculation."""
    if denominator <= 0:
        return 0.0
    return round(min(100.0, (numerator / denominator) * 100.0), 1)


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def fmt_speed_kmh(px_per_frame: float, fps: float, px_per_meter: float) -> str:
    """Convert pixels/frame to km/h display string."""
    if px_per_meter <= 0 or fps <= 0:
        return "—"
    ms  = px_per_frame * fps / px_per_meter
    kmh = ms * 3.6
    return f"{kmh:.1f} km/h"


def fmt_angle(deg: float) -> str:
    return f"{deg:.1f}°"


def fmt_time(seconds: float) -> str:
    return f"{seconds:.2f}s"


def fmt_distance(meters: float) -> str:
    return f"{meters:.2f}m"
