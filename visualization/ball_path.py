#!/usr/bin/env python3
"""
Ball Path Visualizer
====================
Draws the ball's trajectory arc and speed indicators on video frames.
Produces the "ball path" visual shown on the dashboard.

Usage::

    viz   = BallPathVisualizer()
    frame = viz.draw(frame_bgr, ball_detections)
    frame = viz.draw_trajectory(frame_bgr, trajectory_points)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.drawing_utils import YELLOW, WHITE, GREY, _ipt
from utils.math_utils    import px_per_frame_to_kmh
from utils.logger        import get_logger

log = get_logger(__name__)


@dataclass
class BallPoint:
    """A single point in the ball trajectory."""
    x:          float    # pixel x
    y:          float    # pixel y
    frame_index: int
    confidence: float = 1.0


class BallPathVisualizer:
    """
    Draws ball trajectory and speed overlays on video frames.

    Parameters
    ----------
    trail_length : int    — how many past positions to draw as trail
    fps          : float  — video FPS for speed calculation
    px_per_m     : float  — pixels per metre for speed display
    """

    def __init__(
        self,
        trail_length: int   = 15,
        fps:          float = 25.0,
        px_per_m:     float = 100.0,
    ) -> None:
        self.trail_length = trail_length
        self.fps          = fps
        self.px_per_m     = px_per_m

    def draw(
        self,
        frame:       np.ndarray,
        detections:  list,       # List[BallDetection] from BallDetector
    ) -> np.ndarray:
        """
        Draw ball circles and trajectory trail on a copy of the frame.

        Parameters
        ----------
        frame      : BGR image
        detections : list of BallDetection objects

        Returns
        -------
        Annotated BGR image.
        """
        img = frame.copy()
        if not detections:
            return img

        pts = [
            BallPoint(
                x           = getattr(d, "center_x", 0),
                y           = getattr(d, "center_y", 0),
                frame_index = getattr(d, "frame_index", 0),
                confidence  = getattr(d, "confidence", 1.0),
            )
            for d in detections
        ]

        return self.draw_trajectory(img, pts)

    def draw_trajectory(
        self,
        frame: np.ndarray,
        points: List[BallPoint],
    ) -> np.ndarray:
        """
        Draw trajectory from a list of BallPoints.
        Fades older positions — newest is brightest.
        """
        img   = frame.copy()
        recent = points[-self.trail_length:]
        n      = len(recent)

        for i in range(n - 1):
            p0  = _ipt((recent[i].x,     recent[i].y))
            p1  = _ipt((recent[i + 1].x, recent[i + 1].y))
            # Fade: older segments are more transparent (darker).
            alpha = (i + 1) / n
            color = (int(YELLOW[0] * alpha), int(YELLOW[1] * alpha), int(YELLOW[2] * alpha))
            thickness = max(1, int(3 * alpha))
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)

        # Draw latest ball circle.
        if recent:
            last = recent[-1]
            cv2.circle(img, _ipt((last.x, last.y)), 8, YELLOW, 2, cv2.LINE_AA)
            cv2.circle(img, _ipt((last.x, last.y)), 3, YELLOW, -1, cv2.LINE_AA)

            # Speed label (if enough points).
            if n >= 2:
                dx = recent[-1].x - recent[-2].x
                dy = recent[-1].y - recent[-2].y
                speed_px_f = float(np.hypot(dx, dy))
                kmh        = px_per_frame_to_kmh(speed_px_f, self.fps, self.px_per_m)
                cv2.putText(img, f"{kmh:.0f} km/h",
                            (_ipt((last.x, last.y))[0] + 10, _ipt((last.x, last.y))[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        return img

    def draw_arc(
        self,
        frame:  np.ndarray,
        points: List[BallPoint],
    ) -> np.ndarray:
        """
        Draw a smooth spline-like arc through ball positions using
        a sequence of Bezier-approximated polylines.
        """
        img = frame.copy()
        if len(points) < 2:
            return img

        pts_array = np.array(
            [(int(p.x), int(p.y)) for p in points],
            dtype=np.int32,
        )
        cv2.polylines(img, [pts_array], False, YELLOW, 2, cv2.LINE_AA)

        # Mark start and end.
        cv2.circle(img, (pts_array[0][0],  pts_array[0][1]),  6, (0, 200, 255), 2)
        cv2.circle(img, (pts_array[-1][0], pts_array[-1][1]), 6, YELLOW, -1)

        return img
