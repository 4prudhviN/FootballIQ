#!/usr/bin/env python3
"""
Annotation Renderer
===================
Composites all visual layers (pose, ball path, warnings, metrics)
into a single annotated frame ready for the dashboard.

This is the top-level visual composition module — call this instead
of calling individual visualizers separately.

Usage::

    renderer = AnnotationRenderer()
    annotated = renderer.render(
        frame        = raw_bgr,
        frame_pose   = pose_frame,
        ball_detections = ball_dets,
        warnings     = ["POOR POSTURE / LEANING BACK"],
        metrics      = {"Torso Lean": "22°", "Knee Stability": "72"},
        config       = RenderConfig(draw_skeleton=True, draw_ball=True),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from visualization.pose_overlay import PoseOverlay, OverlayConfig
from visualization.ball_path    import BallPathVisualizer, BallPoint
from utils.drawing_utils        import draw_warning_banner, draw_metric_overlay, draw_progress_bar
from utils.logger               import get_logger

log = get_logger(__name__)


@dataclass
class RenderConfig:
    """Controls which annotation layers are rendered."""
    draw_skeleton:   bool  = True
    draw_joints:     bool  = True
    draw_angles:     bool  = False
    draw_ball:       bool  = True
    draw_warnings:   bool  = True
    draw_metrics:    bool  = True
    draw_progress:   bool  = False
    progress_value:  float = 0.0      # 0.0–1.0 for progress bar
    player_level:    str   = ""


class AnnotationRenderer:
    """
    Composites all visual annotation layers onto a video frame.

    Parameters
    ----------
    fps      : float — video FPS (used for speed labels)
    px_per_m : float — pixel/metre ratio for speed labels
    """

    def __init__(self, fps: float = 25.0, px_per_m: float = 100.0) -> None:
        self._pose_overlay = PoseOverlay(
            OverlayConfig(draw_bones=True, draw_joints=True)
        )
        self._ball_viz     = BallPathVisualizer(fps=fps, px_per_m=px_per_m)

    def render(
        self,
        frame:           np.ndarray,
        frame_pose:      Optional[object]    = None,
        ball_detections: Optional[list]      = None,
        warnings:        Optional[List[str]] = None,
        metrics:         Optional[Dict[str, str]] = None,
        config:          Optional[RenderConfig]   = None,
    ) -> np.ndarray:
        """
        Composite all layers onto the frame and return the result.

        Parameters
        ----------
        frame           : raw BGR video frame
        frame_pose      : FramePose from PoseEstimator
        ball_detections : list of BallDetection from BallDetector
        warnings        : list of warning strings
        metrics         : {label: display_value} to show as overlay text
        config          : RenderConfig controlling which layers to draw

        Returns
        -------
        Annotated BGR image (copy of input).
        """
        cfg = config or RenderConfig()
        img = frame.copy()

        # Layer 1: Skeleton pose.
        if cfg.draw_skeleton and frame_pose is not None:
            self._pose_overlay.cfg.draw_bones   = cfg.draw_skeleton
            self._pose_overlay.cfg.draw_joints  = cfg.draw_joints
            self._pose_overlay.cfg.draw_angles  = cfg.draw_angles
            img = self._pose_overlay.draw(img, frame_pose)

        # Layer 2: Ball path trail.
        if cfg.draw_ball and ball_detections:
            img = self._ball_viz.draw(img, ball_detections)

        # Layer 3: Warning banner (top of frame).
        if cfg.draw_warnings and warnings:
            draw_warning_banner(img, warnings)

        # Layer 4: Metric text overlay (bottom-left).
        if cfg.draw_metrics and metrics:
            draw_metric_overlay(img, metrics)

        # Layer 5: Progress bar (bottom of frame).
        if cfg.draw_progress:
            draw_progress_bar(img, cfg.progress_value)

        # Layer 6: Player level badge (top-right).
        if cfg.player_level:
            self._draw_level_badge(img, cfg.player_level)

        return img

    def render_batch(
        self,
        frames:          List[np.ndarray],
        frame_poses:     List[object],
        ball_detections: list,
        warnings:        List[str],
        metrics:         Dict[str, str],
        config:          Optional[RenderConfig] = None,
    ) -> List[np.ndarray]:
        """
        Render annotations on a batch of frames.
        Ball detections and warnings are applied to every frame.
        """
        results = []
        for frame, pose in zip(frames, frame_poses):
            annotated = self.render(
                frame           = frame,
                frame_pose      = pose,
                ball_detections = ball_detections,
                warnings        = warnings,
                metrics         = metrics,
                config          = config,
            )
            results.append(annotated)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_level_badge(img: np.ndarray, level: str) -> None:
        """Draw a small skill-level badge in the top-right corner."""
        H, W = img.shape[:2]
        colours = {
            "Beginner":     (50,  200,  50),
            "Intermediate": (50,  180, 220),
            "Advanced":     (220, 180,  50),
        }
        color = colours.get(level, (150, 150, 150))
        text  = level

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        bx = W - tw - 20
        by = 12

        cv2.rectangle(img, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (0, 0, 0), -1)
        cv2.rectangle(img, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), color, 1)
        cv2.putText(img, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, color, 1, cv2.LINE_AA)
