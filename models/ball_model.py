#!/usr/bin/env python3
"""
Ball Model
==========
Reusable object for football detection and tracking across frames.

Load once — reuse across every request.

Responsibilities:
  - Hold the OpenCV Hough circle detector configuration
  - Expose a detect_frame() method that returns a typed BallState
  - Track ball trajectory across frames (last N positions)
  - Estimate ball velocity and direction from recent positions
  - Support context-manager usage
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BallPosition:
    """Ball position detected in a single frame."""
    frame_index:  int
    timestamp_s:  float
    cx:           float          # centre x (pixels)
    cy:           float          # centre y (pixels)
    radius:       float          # radius (pixels)
    confidence:   float          # Hough accumulator normalised [0, 1]

    def as_tuple(self) -> Tuple[float, float]:
        return (self.cx, self.cy)


@dataclass
class BallState:
    """
    Full ball state for a single frame, including tracking history.
    Returned by BallModel.detect_frame().
    """
    frame_index:    int
    timestamp_s:    float
    detected:       bool
    position:       Optional[BallPosition] = None

    # Derived from trajectory history
    velocity_px_f:  Optional[float] = None    # pixels/frame
    direction_deg:  Optional[float] = None    # degrees from +x axis
    is_moving:      bool            = False


# ---------------------------------------------------------------------------
# Ball Model
# ---------------------------------------------------------------------------

# Hough circle defaults — tuned for a standard football at typical distances.
_HOUGH_DP         = 1.2
_HOUGH_MIN_DIST   = 30
_HOUGH_PARAM1     = 50
_HOUGH_PARAM2     = 30
_HOUGH_MIN_RADIUS = 5
_HOUGH_MAX_RADIUS = 60
_MOTION_THRESHOLD = 2.0    # pixels/frame — below this the ball is considered static
_HISTORY_SIZE     = 10     # number of recent positions to keep for velocity calc


class BallModel:
    """
    Reusable ball detector and tracker.

    Parameters
    ----------
    min_radius : int    Minimum ball radius in pixels.
    max_radius : int    Maximum ball radius in pixels.
    history    : int    Number of recent frames to keep for velocity calculation.

    Usage (single use)::

        model = BallModel()
        state = model.detect_frame(bgr_image, frame_index=0, timestamp_s=0.0)

    Usage (context manager)::

        with BallModel() as model:
            state = model.detect_frame(bgr_image)
    """

    def __init__(
        self,
        min_radius: int = _HOUGH_MIN_RADIUS,
        max_radius: int = _HOUGH_MAX_RADIUS,
        history:    int = _HISTORY_SIZE,
    ) -> None:
        self.min_radius = min_radius
        self.max_radius = max_radius
        self._history:  Deque[BallPosition] = deque(maxlen=history)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_frame(
        self,
        bgr:         np.ndarray,
        frame_index: int   = 0,
        timestamp_s: float = 0.0,
    ) -> BallState:
        """
        Detect the ball in one BGR frame and update tracking state.

        Parameters
        ----------
        bgr : np.ndarray    BGR image from OpenCV.
        frame_index : int
        timestamp_s : float

        Returns
        -------
        BallState
        """
        position = self._detect(bgr, frame_index, timestamp_s)

        if position is not None:
            self._history.append(position)

        velocity, direction = self._estimate_velocity()
        is_moving = velocity is not None and velocity > _MOTION_THRESHOLD

        return BallState(
            frame_index   = frame_index,
            timestamp_s   = timestamp_s,
            detected      = position is not None,
            position      = position,
            velocity_px_f = round(velocity, 2) if velocity is not None else None,
            direction_deg = round(direction, 1) if direction is not None else None,
            is_moving     = is_moving,
        )

    def trajectory(self) -> List[BallPosition]:
        """Return the tracked trajectory as a list of recent positions."""
        return list(self._history)

    def reset(self) -> None:
        """Clear tracking history."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BallModel":
        return self

    def __exit__(self, *_) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _detect(
        self,
        bgr:         np.ndarray,
        frame_index: int,
        timestamp_s: float,
    ) -> Optional[BallPosition]:
        """Run Hough circle detection and return the best candidate."""
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp        = _HOUGH_DP,
            minDist   = _HOUGH_MIN_DIST,
            param1    = _HOUGH_PARAM1,
            param2    = _HOUGH_PARAM2,
            minRadius = self.min_radius,
            maxRadius = self.max_radius,
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(int)
        cx, cy, r = circles[0]
        conf = min(1.0, float(r) / self.max_radius)

        return BallPosition(
            frame_index = frame_index,
            timestamp_s = timestamp_s,
            cx          = float(cx),
            cy          = float(cy),
            radius      = float(r),
            confidence  = round(conf, 3),
        )

    def _estimate_velocity(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate ball speed (pixels/frame) and direction (degrees)
        from the last two tracked positions.
        """
        if len(self._history) < 2:
            return None, None

        prev = self._history[-2]
        curr = self._history[-1]

        dt = max(1, curr.frame_index - prev.frame_index)
        dx = (curr.cx - prev.cx) / dt
        dy = (curr.cy - prev.cy) / dt

        speed     = math.hypot(dx, dy)
        direction = math.degrees(math.atan2(dy, dx))

        return speed, direction
