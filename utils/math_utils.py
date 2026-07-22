#!/usr/bin/env python3
"""
Math Utilities
==============
General-purpose numeric helpers used across the codebase.
All functions are stateless and side-effect free.
Import from here — never re-implement in another module.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

def safe_mean(values: Sequence[float]) -> Optional[float]:
    """Mean of a sequence, or None if empty."""
    if not values:
        return None
    return float(np.mean(values))


def safe_max(values: Sequence[float]) -> Optional[float]:
    """Max of a sequence, or None if empty."""
    return float(max(values)) if values else None


def safe_min(values: Sequence[float]) -> Optional[float]:
    """Min of a sequence, or None if empty."""
    return float(min(values)) if values else None


def safe_std(values: Sequence[float]) -> Optional[float]:
    """Standard deviation, or None if fewer than 2 values."""
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1))


def safe_percentage(numerator: float, denominator: float) -> float:
    """percentage(n, d) → 0–100. Returns 0.0 for zero denominator."""
    if denominator <= 0:
        return 0.0
    return round(min(100.0, (numerator / denominator) * 100.0), 2)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def normalise_0_1(value: float, min_val: float, max_val: float) -> float:
    """
    Normalise value to [0, 1] given known min/max.
    Returns 0.5 for degenerate ranges.
    """
    span = max_val - min_val
    if abs(span) < 1e-9:
        return 0.5
    return clamp((value - min_val) / span, 0.0, 1.0)


def symmetry_ratio(a: float, b: float) -> float:
    """
    Symmetry ratio between two values [0, 1].
    1.0 = perfectly symmetric, 0.0 = completely asymmetric.
    """
    denom = max(abs(a), abs(b))
    if denom < 1e-9:
        return 1.0
    return 1.0 - abs(a - b) / denom


# ---------------------------------------------------------------------------
# Rolling / windowed operations
# ---------------------------------------------------------------------------

def rolling_mean(values: List[float], window: int) -> List[float]:
    """Compute rolling mean with the given window size."""
    if window < 1 or not values:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start: i + 1])))
    return result


def frame_to_frame_delta(values: List[float]) -> List[float]:
    """Compute frame-to-frame differences (derivative)."""
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def local_minima(values: List[float]) -> List[int]:
    """Return indices of local minima (values[i] < neighbours)."""
    indices = []
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            indices.append(i)
    return indices


def local_maxima(values: List[float]) -> List[int]:
    """Return indices of local maxima (values[i] > neighbours)."""
    indices = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            indices.append(i)
    return indices


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def px_per_frame_to_kmh(
    speed_px_f: float,
    fps:        float,
    px_per_m:   float,
) -> float:
    """Convert pixels/frame to km/h."""
    if fps <= 0 or px_per_m <= 0:
        return 0.0
    m_per_s = speed_px_f * fps / px_per_m
    return m_per_s * 3.6


def normalised_to_pixels(
    normalised: float,
    dimension:  int,
) -> float:
    """Convert normalised [0,1] coordinate to pixel value."""
    return normalised * dimension


def pixels_to_normalised(
    pixels:    float,
    dimension: int,
) -> float:
    """Convert pixel coordinate to normalised [0,1]."""
    if dimension <= 0:
        return 0.0
    return pixels / dimension


# ---------------------------------------------------------------------------
# Formatting helpers (numeric → display string)
# ---------------------------------------------------------------------------

def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{round(value, decimals)}%"


def fmt_angle(deg: float, decimals: int = 1) -> str:
    return f"{round(deg, decimals)}°"


def fmt_speed_kmh(kmh: float, decimals: int = 1) -> str:
    return f"{round(kmh, decimals)} km/h"


def fmt_distance_m(metres: float, decimals: int = 2) -> str:
    return f"{round(metres, decimals)}m"


def fmt_time_s(seconds: float, decimals: int = 2) -> str:
    return f"{round(seconds, decimals)}s"


def fmt_score(value: float, out_of: int = 100) -> str:
    return f"{round(value, 1)}/{out_of}"
