#!/usr/bin/env python3
"""
Heatmap Generator
=================
Generates position and activity density heatmaps for the dashboard.

Produces:
  - Player position heatmap (where the player spent most time)
  - Ball contact heatmap (where the ball was touched most)
  - Shot target heatmap (goal zone impact distribution)

Usage::

    gen  = HeatmapGenerator(width=500, height=400)
    heat = gen.player_heatmap(position_list, frame_bgr)
    shot = gen.shot_heatmap(shot_points, goal_frame)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


class HeatmapGenerator:
    """
    Generates Gaussian-blurred heatmaps from point data.

    Parameters
    ----------
    width, height : int  — output image dimensions
    blur_radius   : int  — Gaussian blur kernel size (odd number)
    colormap      : int  — OpenCV colormap (default COLORMAP_JET)
    alpha         : float — overlay transparency when blending
    """

    def __init__(
        self,
        width:       int   = 500,
        height:      int   = 400,
        blur_radius: int   = 31,
        colormap:    int   = cv2.COLORMAP_JET,
        alpha:       float = 0.6,
    ) -> None:
        self.width       = width
        self.height      = height
        self.blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        self.colormap    = colormap
        self.alpha       = alpha

    # ------------------------------------------------------------------
    # Player position heatmap
    # ------------------------------------------------------------------

    def player_heatmap(
        self,
        positions:    List[Tuple[float, float]],   # normalised (x, y) in [0,1]
        background:   Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate a heatmap of where the player spent most time.

        Parameters
        ----------
        positions  : list of (x, y) in normalised [0,1] coordinates
        background : optional BGR background image to blend over

        Returns
        -------
        BGR heatmap image
        """
        heat = self._accumulate(positions)
        return self._render(heat, background, title="Player Position Heatmap")

    # ------------------------------------------------------------------
    # Ball contact heatmap
    # ------------------------------------------------------------------

    def ball_heatmap(
        self,
        ball_positions: List[Tuple[float, float]],   # normalised (x, y)
        background:     Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate a heatmap of where the ball was most active.
        """
        heat = self._accumulate(ball_positions)
        return self._render(heat, background, title="Ball Activity Heatmap")

    # ------------------------------------------------------------------
    # Shot target heatmap (goal zone)
    # ------------------------------------------------------------------

    def shot_heatmap(
        self,
        impact_points: List[Tuple[float, float]],   # normalised (x, y)
        goal_frame:    Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate a shot impact heatmap overlaid on a goal frame.
        Normalised coordinates refer to the goal rectangle [0,1]×[0,1].
        """
        # Build goal background if not provided.
        if goal_frame is None:
            goal_frame = self._draw_goal_background()

        bg = cv2.resize(goal_frame, (self.width, self.height))
        heat = self._accumulate(impact_points)
        return self._render(heat, bg, title="Shot Impact Heatmap")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _accumulate(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """Build a float32 density map from normalised point list."""
        heat = np.zeros((self.height, self.width), dtype=np.float32)
        for x_norm, y_norm in points:
            px = int(np.clip(x_norm, 0, 1) * (self.width  - 1))
            py = int(np.clip(y_norm, 0, 1) * (self.height - 1))
            heat[py, px] += 1.0
        return heat

    def _render(
        self,
        heat:       np.ndarray,
        background: Optional[np.ndarray],
        title:      str = "",
    ) -> np.ndarray:
        """Blur, normalise, colorize and blend the heatmap."""
        blurred = cv2.GaussianBlur(heat, (self.blur_radius, self.blur_radius), 0)

        # Normalise to [0, 255].
        max_val = blurred.max()
        if max_val > 0:
            blurred = (blurred / max_val * 255).astype(np.uint8)
        else:
            blurred = blurred.astype(np.uint8)

        colored = cv2.applyColorMap(blurred, self.colormap)

        if background is not None:
            bg   = cv2.resize(background, (self.width, self.height))
            out  = cv2.addWeighted(colored, self.alpha, bg, 1 - self.alpha, 0)
        else:
            out = colored

        if title:
            cv2.putText(out, title, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def _draw_goal_background(self) -> np.ndarray:
        """Draw a minimal goal diagram as background for shot heatmap."""
        bg = np.full((self.height, self.width, 3), (15, 15, 15), dtype=np.uint8)
        m  = 0.12   # margin
        x1 = int(m * self.width)
        x2 = int((1 - m) * self.width)
        y1 = int(m * self.height)
        y2 = int((1 - m) * self.height)
        cv2.rectangle(bg, (x1, y1), (x2, y2), (80, 140, 80), 2)
        # Centre line.
        cx = self.width // 2
        cv2.line(bg, (cx, y1), (cx, y2), (50, 80, 50), 1)
        # Third lines.
        th = (y2 - y1) // 3
        for k in range(1, 3):
            yt = y1 + k * th
            cv2.line(bg, (x1, yt), (x2, yt), (50, 80, 50), 1)
        return bg
