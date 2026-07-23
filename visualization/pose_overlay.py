#!/usr/bin/env python3
"""
Pose Overlay
============
Draws the MediaPipe skeleton (bones, joints, angles) onto video frames.
Produces the "skeleton tracking" visual shown on the dashboard.

Usage::

    overlay = PoseOverlay()
    annotated_frame = overlay.draw(frame_bgr, frame_pose)
    annotated_frame = overlay.draw_with_angles(frame_bgr, frame_pose)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.drawing_utils import (
    SKELETON_CONNECTIONS, NEON, WHITE, YELLOW, GREY, _ipt,
)
from utils.geometry import angle_at_vertex
from utils.logger   import get_logger

log = get_logger(__name__)

# Key joint angles to display on the dashboard.
_ANGLE_TRIPLETS: List[Tuple[str, str, str, str]] = [
    # (joint_a, vertex, joint_b, label)
    ("left_hip",   "left_knee",  "left_ankle",  "L Knee"),
    ("right_hip",  "right_knee", "right_ankle", "R Knee"),
    ("left_shoulder", "left_hip", "left_knee",  "L Hip"),
    ("right_shoulder","right_hip","right_knee", "R Hip"),
]


@dataclass
class OverlayConfig:
    """Visual configuration for the pose overlay."""
    draw_bones:    bool  = True
    draw_joints:   bool  = True
    draw_angles:   bool  = False
    draw_labels:   bool  = False
    bone_color:    Tuple[int, int, int] = NEON
    joint_color:   Tuple[int, int, int] = NEON
    angle_color:   Tuple[int, int, int] = YELLOW
    bone_thickness: int  = 3
    joint_radius:   int  = 6
    font_scale:    float = 0.45


class PoseOverlay:
    """
    Draws skeleton pose overlays on video frames.

    Parameters
    ----------
    config : OverlayConfig | None — visual options
    """

    def __init__(self, config: Optional[OverlayConfig] = None) -> None:
        self.cfg = config or OverlayConfig()

    def draw(
        self,
        frame:       np.ndarray,
        frame_pose:  object,        # FramePose from pose_estimator
    ) -> np.ndarray:
        """
        Draw skeleton on a copy of the frame.

        Parameters
        ----------
        frame      : BGR image
        frame_pose : FramePose — landmarks in normalised [0,1] coordinates

        Returns
        -------
        Annotated BGR image (copy — original untouched).
        """
        img = frame.copy()
        if not getattr(frame_pose, "detected", False):
            return img

        H, W  = img.shape[:2]
        lm    = getattr(frame_pose, "landmarks", {})
        if not lm:
            return img

        px: Dict[str, Tuple[float, float]] = {
            name: (lm_obj.x * W, lm_obj.y * H)
            for name, lm_obj in lm.items()
        }

        if self.cfg.draw_bones:
            self._draw_bones(img, px)

        if self.cfg.draw_joints:
            self._draw_joints(img, px)

        if self.cfg.draw_angles:
            self._draw_angles(img, px)

        if self.cfg.draw_labels:
            self._draw_labels(img, px)

        return img

    def draw_with_angles(
        self,
        frame:      np.ndarray,
        frame_pose: object,
    ) -> np.ndarray:
        """Convenience: draw skeleton + joint angles."""
        old = self.cfg.draw_angles
        self.cfg.draw_angles = True
        result = self.draw(frame, frame_pose)
        self.cfg.draw_angles = old
        return result

    def draw_batch(
        self,
        frames:      List[np.ndarray],
        frame_poses: List[object],
    ) -> List[np.ndarray]:
        """Draw overlays on a list of frames."""
        return [
            self.draw(f, fp)
            for f, fp in zip(frames, frame_poses)
        ]

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_bones(self, img: np.ndarray, px: Dict[str, Tuple[float, float]]) -> None:
        for a, b in SKELETON_CONNECTIONS:
            if a in px and b in px:
                cv2.line(img, _ipt(px[a]), _ipt(px[b]),
                         self.cfg.bone_color, self.cfg.bone_thickness, cv2.LINE_AA)

    def _draw_joints(self, img: np.ndarray, px: Dict[str, Tuple[float, float]]) -> None:
        for pt in px.values():
            cv2.circle(img, _ipt(pt), self.cfg.joint_radius,
                       self.cfg.joint_color, -1, cv2.LINE_AA)
            cv2.circle(img, _ipt(pt), self.cfg.joint_radius,
                       (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_angles(self, img: np.ndarray, px: Dict[str, Tuple[float, float]]) -> None:
        for a_name, v_name, b_name, label in _ANGLE_TRIPLETS:
            if a_name not in px or v_name not in px or b_name not in px:
                continue
            a, v, b = px[a_name], px[v_name], px[b_name]
            ang = angle_at_vertex(a[0], a[1], v[0], v[1], b[0], b[1])
            vp  = _ipt(v)
            cv2.putText(img, f"{ang:.0f}°",
                        (vp[0] + 8, vp[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, self.cfg.font_scale,
                        self.cfg.angle_color, 1, cv2.LINE_AA)

    def _draw_labels(self, img: np.ndarray, px: Dict[str, Tuple[float, float]]) -> None:
        key_joints = ["left_knee", "right_knee", "left_ankle", "right_ankle",
                      "left_hip", "right_hip"]
        for name in key_joints:
            if name not in px:
                continue
            pt = _ipt(px[name])
            short = name.replace("_", " ").title()
            cv2.putText(img, short, (pt[0] + 8, pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREY, 1, cv2.LINE_AA)
