#!/usr/bin/env python3
"""
Geometry Utilities
==================
Pure 2-D geometric functions used across the entire codebase.
All functions are stateless and side-effect free.
Import from here — never re-implement in another module.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Point / Vector helpers
# ---------------------------------------------------------------------------

def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(bx - ax, by - ay)


def midpoint(ax: float, ay: float, bx: float, by: float) -> Tuple[float, float]:
    """Midpoint of two 2-D points."""
    return (ax + bx) / 2.0, (ay + by) / 2.0


def vector(ax: float, ay: float, bx: float, by: float) -> Tuple[float, float]:
    """Direction vector from A to B."""
    return bx - ax, by - ay


def magnitude(dx: float, dy: float) -> float:
    """Magnitude of a 2-D vector."""
    return math.hypot(dx, dy)


def normalise(dx: float, dy: float) -> Tuple[float, float]:
    """Return a unit vector in the direction (dx, dy). Returns (0,0) for zero vector."""
    mag = magnitude(dx, dy)
    if mag < 1e-9:
        return 0.0, 0.0
    return dx / mag, dy / mag


def dot(ax: float, ay: float, bx: float, by: float) -> float:
    """Dot product of two 2-D vectors."""
    return ax * bx + ay * by


def cross_2d(ax: float, ay: float, bx: float, by: float) -> float:
    """Z-component of the cross product of two 2-D vectors."""
    return ax * by - ay * bx


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def angle_between_vectors(
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """
    Unsigned angle (degrees) between two 2-D vectors.
    Returns 0 for degenerate (zero-length) inputs.
    """
    mag_a = magnitude(ax, ay)
    mag_b = magnitude(bx, by)
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    cos_a = np.clip(dot(ax, ay, bx, by) / (mag_a * mag_b), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_a)))


def angle_at_vertex(
    ax: float, ay: float,
    vx: float, vy: float,
    bx: float, by: float,
) -> float:
    """
    Interior angle (degrees) at vertex V formed by rays V→A and V→B.
    Range [0, 180].
    """
    return angle_between_vectors(ax - vx, ay - vy, bx - vx, by - vy)


def signed_angle_from_vertical(dx: float, dy: float) -> float:
    """
    Signed angle (degrees) of vector (dx, dy) from the upward vertical (0, -1).
    Positive = rightward lean, negative = leftward lean.
    Used for torso lean calculation.
    """
    return float(math.degrees(math.atan2(-dx, -dy)))


def direction_deg(dx: float, dy: float) -> float:
    """Angle (degrees) of vector (dx, dy) from positive x-axis."""
    return float(math.degrees(math.atan2(dy, dx)))


# ---------------------------------------------------------------------------
# Point-to-line helpers
# ---------------------------------------------------------------------------

def perpendicular_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """
    Signed perpendicular distance of point P from line A→B.
    Positive = P is to the left of A→B (counter-clockwise side).
    """
    abx, aby = bx - ax, by - ay
    ab_len = magnitude(abx, aby)
    if ab_len < 1e-9:
        return distance(px, py, ax, ay)
    return cross_2d(abx, aby, px - ax, py - ay) / ab_len


def knee_deviation_ratio(
    hip_x: float, hip_y: float,
    knee_x: float, knee_y: float,
    ankle_x: float, ankle_y: float,
) -> Optional[float]:
    """
    Perpendicular deviation of the knee from the hip-ankle axis,
    expressed as a ratio of thigh length (hip→knee distance).

    Returns None if landmarks are degenerate.
    Positive = valgus (inward collapse), negative = varus (outward).
    """
    thigh = distance(hip_x, hip_y, knee_x, knee_y)
    if thigh < 1e-9:
        return None
    dev = perpendicular_distance(knee_x, knee_y, hip_x, hip_y, ankle_x, ankle_y)
    return dev / thigh


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

def bbox_centre(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    """Centre of a bounding box (x, y, w, h)."""
    return x + w / 2.0, y + h / 2.0


def bbox_area(w: float, h: float) -> float:
    return w * h


def bbox_iou(
    ax: float, ay: float, aw: float, ah: float,
    bx: float, by: float, bw: float, bh: float,
) -> float:
    """Intersection-over-Union of two axis-aligned bounding boxes."""
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix) * max(0.0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
