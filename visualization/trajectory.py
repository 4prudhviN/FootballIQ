#!/usr/bin/env python3
"""
Trajectory Visualizer
=====================
Draws player movement trajectories and activity timelines
on a 2-D pitch diagram for the dashboard.

Produces:
  - Player run trajectory on a pitch bird's-eye view
  - Per-activity colour-coded movement segments
  - Direction arrows showing movement intent

Usage::

    viz = TrajectoryVisualizer()

    # Draw player movement path on pitch
    pitch = viz.draw_player_path(movement_frames)

    # Colour-coded by activity segment
    pitch = viz.draw_segmented_path(movement_frames, timeline_segments)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

# Activity → BGR colour for trajectory segments.
_ACTIVITY_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "passing":     (50,  200, 50),
    "shooting":    (50,  50,  220),
    "dribbling":   (220, 180, 50),
    "defending":   (50,  150, 220),
    "goalkeeping": (180, 50,  220),
    "movement":    (200, 200, 200),
    "unknown":     (100, 100, 100),
}

_DEFAULT_W = 700
_DEFAULT_H = 460
_PITCH_GREEN = (22, 80, 22)
_LINE_WHITE  = (200, 200, 200)


class TrajectoryVisualizer:
    """
    Draws movement trajectories on a 2-D pitch diagram.

    Parameters
    ----------
    width, height : int — output pitch image size
    """

    def __init__(self, width: int = _DEFAULT_W, height: int = _DEFAULT_H) -> None:
        self.width  = width
        self.height = height

    def draw_player_path(
        self,
        positions: List[Tuple[float, float]],   # normalised (x, y) in [0,1]
        color:     Tuple[int, int, int] = (50, 220, 50),
        trail_fade: bool = True,
    ) -> np.ndarray:
        """
        Draw the player's movement path on a pitch background.

        Parameters
        ----------
        positions  : list of normalised (x, y) tuples
        color      : BGR line colour
        trail_fade : if True, older positions are drawn dimmer

        Returns
        -------
        BGR pitch image with trajectory drawn.
        """
        pitch = self._draw_pitch()
        if len(positions) < 2:
            return pitch

        n = len(positions)
        for i in range(n - 1):
            p0  = self._to_px(positions[i])
            p1  = self._to_px(positions[i + 1])
            alpha = (i + 1) / n if trail_fade else 1.0
            c   = tuple(int(ch * alpha) for ch in color)
            cv2.line(pitch, p0, p1, c, 2, cv2.LINE_AA)

        # Draw direction arrow at last segment.
        if n >= 2:
            self._draw_arrow(pitch, positions[-2], positions[-1], color)

        # Mark start/end.
        cv2.circle(pitch, self._to_px(positions[0]),  5, (50, 200, 50), -1)
        cv2.circle(pitch, self._to_px(positions[-1]), 5, (50,  50, 220), -1)

        return pitch

    def draw_segmented_path(
        self,
        positions: List[Tuple[float, float]],
        timestamps: List[float],
        segments:  List[object],              # List[ActivitySegment]
    ) -> np.ndarray:
        """
        Draw path with each activity segment in a different colour.

        Parameters
        ----------
        positions  : normalised (x, y)
        timestamps : timestamp_s per position
        segments   : ActivitySegment list from SequenceAnalyzer
        """
        pitch = self._draw_pitch()
        if not positions or not timestamps:
            return pitch

        # Build a timestamp → activity lookup from segments.
        def activity_at(t: float) -> str:
            for seg in segments:
                s = getattr(seg, "start_time_s", 0)
                e = getattr(seg, "end_time_s", 0)
                if s <= t <= e:
                    return getattr(seg, "action", "unknown")
            return "movement"

        for i in range(len(positions) - 1):
            p0 = self._to_px(positions[i])
            p1 = self._to_px(positions[i + 1])
            t  = timestamps[i] if i < len(timestamps) else 0.0
            act = activity_at(t)
            color = _ACTIVITY_COLOURS.get(act, _ACTIVITY_COLOURS["unknown"])
            cv2.line(pitch, p0, p1, color, 2, cv2.LINE_AA)

        # Legend.
        self._draw_legend(pitch, segments)
        return pitch

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_pitch(self) -> np.ndarray:
        """Draw a simple 2-D football pitch bird's-eye view."""
        img = np.full((self.height, self.width, 3), _PITCH_GREEN, dtype=np.uint8)
        W, H = self.width, self.height
        pad  = 20

        # Outer boundary.
        cv2.rectangle(img, (pad, pad), (W - pad, H - pad), _LINE_WHITE, 2)
        # Centre line.
        cv2.line(img, (W // 2, pad), (W // 2, H - pad), _LINE_WHITE, 1)
        # Centre circle.
        cv2.circle(img, (W // 2, H // 2), int(H * 0.12), _LINE_WHITE, 1)
        cv2.circle(img, (W // 2, H // 2), 3, _LINE_WHITE, -1)
        # Penalty boxes.
        pb_w = int(W * 0.12)
        pb_h = int(H * 0.40)
        pb_y = (H - pb_h) // 2
        cv2.rectangle(img, (pad, pb_y), (pad + pb_w, pb_y + pb_h), _LINE_WHITE, 1)
        cv2.rectangle(img, (W - pad - pb_w, pb_y), (W - pad, pb_y + pb_h), _LINE_WHITE, 1)

        return img

    def _to_px(self, pt: Tuple[float, float]) -> Tuple[int, int]:
        """Convert normalised [0,1] to pixel coordinate on the pitch."""
        pad = 20
        x = int(np.clip(pt[0], 0, 1) * (self.width  - 2 * pad) + pad)
        y = int(np.clip(pt[1], 0, 1) * (self.height - 2 * pad) + pad)
        return x, y

    def _draw_arrow(
        self,
        img:  np.ndarray,
        p0:   Tuple[float, float],
        p1:   Tuple[float, float],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a small directional arrow from p0 toward p1."""
        src = self._to_px(p0)
        dst = self._to_px(p1)
        cv2.arrowedLine(img, src, dst, color, 2, tipLength=0.4)

    def _draw_legend(self, img: np.ndarray, segments: List[object]) -> None:
        """Draw a small colour legend for activity types."""
        seen_activities: List[str] = []
        for seg in segments:
            act = getattr(seg, "action", "unknown")
            if act not in seen_activities:
                seen_activities.append(act)

        x, y = 10, self.height - 20 - len(seen_activities) * 18
        for act in seen_activities:
            color = _ACTIVITY_COLOURS.get(act, _ACTIVITY_COLOURS["unknown"])
            cv2.rectangle(img, (x, y), (x + 12, y + 12), color, -1)
            cv2.putText(img, act.capitalize(), (x + 16, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (200, 200, 200), 1, cv2.LINE_AA)
            y += 18
