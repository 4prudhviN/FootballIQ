#!/usr/bin/env python3
"""
Overlay Generator
=================
Produces annotated frame images that visually explain the analysis.
Saved to reports/overlays/<session_id>/.

Overlays produced:
  - pose_overlay.png      — skeleton landmarks drawn on the best pose frame
  - ball_path.png         — ball trajectory arc drawn on a video frame
  - contact_point.png     — foot-to-ball contact zone highlighted
  - warning_frame.png     — frame where a biomechanical warning was triggered

Usage::

    gen   = OverlayGenerator()
    paths = gen.generate(session_id, frames, pose_result, ball_result, warnings)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.drawing_utils import (
    draw_skeleton, draw_warning_banner, draw_text,
    draw_circle_detection, draw_bbox,
    NEON, RED, YELLOW, WHITE, GREY,
)
from utils.file_utils import ensure_dir
from utils.logger     import get_logger

log = get_logger(__name__)

_OVERLAYS_DIR = Path(__file__).resolve().parent / "overlays"


class OverlayGenerator:
    """
    Generates annotated frame images for the analysis report.

    Parameters
    ----------
    overlays_dir : Path | None — output base directory
    """

    def __init__(self, overlays_dir: Optional[Path] = None) -> None:
        self._base = overlays_dir or _OVERLAYS_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        session_id:  str,
        frames:      list,               # List[ExtractedFrame]
        pose_result: object,             # PoseEstimationResult
        ball_result: object,             # BallDetectionResult
        warnings:    List[str],
    ) -> Dict[str, Path]:
        """
        Generate all overlays for a session. Returns {overlay_type: path}.
        """
        out_dir = ensure_dir(self._base / session_id)
        paths:  Dict[str, Path] = {}

        if not frames:
            log.debug("OverlayGenerator: no frames — skipping")
            return paths

        try:
            p = self._pose_overlay(frames, pose_result, out_dir)
            if p:
                paths["pose"] = p
        except Exception as exc:
            log.warning("OverlayGenerator: pose overlay failed — %s", exc)

        try:
            p = self._ball_path_overlay(frames, ball_result, out_dir)
            if p:
                paths["ball_path"] = p
        except Exception as exc:
            log.warning("OverlayGenerator: ball path overlay failed — %s", exc)

        try:
            p = self._warning_overlay(frames, pose_result, warnings, out_dir)
            if p:
                paths["warning"] = p
        except Exception as exc:
            log.warning("OverlayGenerator: warning overlay failed — %s", exc)

        log.info("OverlayGenerator: %d overlays for session %s", len(paths), session_id)
        return paths

    # ------------------------------------------------------------------
    # Pose overlay
    # ------------------------------------------------------------------

    def _pose_overlay(self, frames: list, pose_result: object, out_dir: Path) -> Optional[Path]:
        """Draw the skeleton on the frame with the highest pose detection confidence."""
        frame_poses = getattr(pose_result, "frame_poses", [])
        detected    = [fp for fp in frame_poses if getattr(fp, "detected", False)]
        if not detected:
            return None

        # Pick the middle detected frame.
        best_fp = detected[len(detected) // 2]

        # Find the corresponding raw frame.
        raw_frame = self._get_frame(frames, best_fp.frame_index)
        if raw_frame is None:
            return None

        img = raw_frame.copy()
        H, W = img.shape[:2]

        landmarks = getattr(best_fp, "landmarks", {})
        if landmarks:
            # Convert normalised → pixel coordinates.
            pixel_lm = {
                name: (lm.x * W, lm.y * H)
                for name, lm in landmarks.items()
            }
            draw_skeleton(img, pixel_lm)

        # Annotate torso lean.
        torso_lean = getattr(best_fp, "torso_lean_deg", None)
        if torso_lean is not None:
            draw_text(img, f"Torso lean: {torso_lean:.1f}°", 12, H - 12,
                      color=WHITE, scale=0.55)

        draw_text(img, "Pose Overlay", 12, 24, color=NEON, scale=0.65)

        path = out_dir / "pose_overlay.png"
        cv2.imwrite(str(path), img)
        return path

    # ------------------------------------------------------------------
    # Ball path overlay
    # ------------------------------------------------------------------

    def _ball_path_overlay(self, frames: list, ball_result: object, out_dir: Path) -> Optional[Path]:
        """Draw ball trajectory arcs on a frame."""
        detections = getattr(ball_result, "detections", [])
        if len(detections) < 2:
            return None

        # Use the frame at the first detection.
        raw_frame = self._get_frame(frames, detections[0].frame_index)
        if raw_frame is None:
            return None

        img = raw_frame.copy()
        H, W = img.shape[:2]

        # Draw trajectory line.
        for i in range(len(detections) - 1):
            d0 = detections[i]
            d1 = detections[i + 1]
            p0 = (int(d0.center_x), int(d0.center_y))
            p1 = (int(d1.center_x), int(d1.center_y))
            cv2.line(img, p0, p1, YELLOW, 2, cv2.LINE_AA)

        # Draw ball circles.
        for d in detections:
            draw_circle_detection(img, int(d.center_x), int(d.center_y),
                                  int(d.radius), color=YELLOW)

        draw_text(img, "Ball Path Overlay", 12, 24, color=YELLOW, scale=0.65)
        path = out_dir / "ball_path.png"
        cv2.imwrite(str(path), img)
        return path

    # ------------------------------------------------------------------
    # Warning overlay
    # ------------------------------------------------------------------

    def _warning_overlay(
        self,
        frames:      list,
        pose_result: object,
        warnings:    List[str],
        out_dir:     Path,
    ) -> Optional[Path]:
        """Draw warning banners on the frame where issues were detected."""
        if not warnings:
            return None

        frame_poses = getattr(pose_result, "frame_poses", [])
        detected    = [fp for fp in frame_poses if getattr(fp, "detected", False)]
        if not detected:
            return None

        # Use the first detected frame.
        raw_frame = self._get_frame(frames, detected[0].frame_index)
        if raw_frame is None:
            return None

        img = raw_frame.copy()
        H, W = img.shape[:2]

        # Draw skeleton.
        landmarks = getattr(detected[0], "landmarks", {})
        if landmarks:
            pixel_lm = {
                name: (lm.x * W, lm.y * H)
                for name, lm in landmarks.items()
            }
            draw_skeleton(img, pixel_lm)

        draw_warning_banner(img, warnings)
        draw_text(img, "Diagnostic Frame", 12, H - 12, color=WHITE, scale=0.50)

        path = out_dir / "warning_frame.png"
        cv2.imwrite(str(path), img)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_frame(frames: list, frame_index: int) -> Optional[np.ndarray]:
        """Return the BGR array for a given frame index, or None."""
        for f in frames:
            if getattr(f, "index", -1) == frame_index:
                bgr = getattr(f, "bgr", None)
                if bgr is not None:
                    return bgr.copy()
        # Fallback: return the middle frame.
        if frames:
            mid = frames[len(frames) // 2]
            bgr = getattr(mid, "bgr", None)
            if bgr is not None:
                return bgr.copy()
        return None
