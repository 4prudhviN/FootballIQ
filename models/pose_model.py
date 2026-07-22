#!/usr/bin/env python3
"""
Pose Model
==========
Reusable, singleton-style wrapper around MediaPipe Pose.

Load once at server startup — reuse across every request.
Handles lifecycle (init / process / close) in one place.

Responsibilities:
  - Hold the MediaPipe Pose instance
  - Expose a process_frame() method that returns typed landmarks
  - Support context-manager usage (with PoseModel() as pm: ...)
  - Cache configuration so it never needs to be re-read per frame
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# Landmark data types
# ---------------------------------------------------------------------------

@dataclass
class Landmark:
    """A single body landmark in normalised [0, 1] coordinates."""
    name:       str
    x:          float
    y:          float
    z:          float
    visibility: float

    @property
    def is_visible(self) -> bool:
        return self.visibility >= 0.5


@dataclass
class PoseFrame:
    """All landmarks detected in a single frame."""
    frame_index:  int
    timestamp_s:  float
    detected:     bool
    landmarks:    Dict[str, Landmark] = field(default_factory=dict)

    # Computed biomechanical scalars (filled by PoseModel.process_frame)
    torso_lean_deg:   Optional[float] = None   # degrees from vertical
    left_knee_dev:    Optional[float] = None   # deviation ratio
    right_knee_dev:   Optional[float] = None   # deviation ratio

    def get(self, name: str) -> Optional[Landmark]:
        return self.landmarks.get(name)


# ---------------------------------------------------------------------------
# Landmark name constants
# ---------------------------------------------------------------------------

_LM = mp.solutions.pose.PoseLandmark

LANDMARK_NAMES: Dict[str, int] = {
    "nose":            _LM.NOSE.value,
    "left_shoulder":   _LM.LEFT_SHOULDER.value,
    "right_shoulder":  _LM.RIGHT_SHOULDER.value,
    "left_elbow":      _LM.LEFT_ELBOW.value,
    "right_elbow":     _LM.RIGHT_ELBOW.value,
    "left_wrist":      _LM.LEFT_WRIST.value,
    "right_wrist":     _LM.RIGHT_WRIST.value,
    "left_hip":        _LM.LEFT_HIP.value,
    "right_hip":       _LM.RIGHT_HIP.value,
    "left_knee":       _LM.LEFT_KNEE.value,
    "right_knee":      _LM.RIGHT_KNEE.value,
    "left_ankle":      _LM.LEFT_ANKLE.value,
    "right_ankle":     _LM.RIGHT_ANKLE.value,
    "left_heel":       _LM.LEFT_HEEL.value,
    "right_heel":      _LM.RIGHT_HEEL.value,
    "left_foot_index": _LM.LEFT_FOOT_INDEX.value,
    "right_foot_index":_LM.RIGHT_FOOT_INDEX.value,
}


# ---------------------------------------------------------------------------
# Pose Model
# ---------------------------------------------------------------------------

class PoseModel:
    """
    Reusable MediaPipe Pose wrapper.

    Parameters
    ----------
    model_complexity : int
        0 = lite (fast), 1 = full (balanced), 2 = heavy (accurate).
    min_detection_confidence : float
    min_tracking_confidence  : float

    Usage (single use)::

        model = PoseModel()
        frame_pose = model.process_frame(bgr_image, frame_index=0, timestamp_s=0.0)
        model.close()

    Usage (context manager)::

        with PoseModel() as model:
            frame_pose = model.process_frame(bgr_image)
    """

    def __init__(
        self,
        model_complexity:         int   = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
    ) -> None:
        self._config = dict(
            static_image_mode         = False,
            model_complexity          = model_complexity,
            enable_segmentation       = False,
            min_detection_confidence  = min_detection_confidence,
            min_tracking_confidence   = min_tracking_confidence,
        )
        self._pose: Optional[mp.solutions.pose.Pose] = None
        self._lock = threading.Lock()
        self._init()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        bgr: np.ndarray,
        frame_index: int   = 0,
        timestamp_s: float = 0.0,
    ) -> PoseFrame:
        """
        Run pose estimation on one BGR frame.

        Parameters
        ----------
        bgr : np.ndarray — BGR image from OpenCV
        frame_index : int
        timestamp_s : float

        Returns
        -------
        PoseFrame with detected=True if a pose was found.
        """
        with self._lock:
            if self._pose is None:
                self._init()

            rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return PoseFrame(
                frame_index = frame_index,
                timestamp_s = timestamp_s,
                detected    = False,
            )

        raw_lm = results.pose_landmarks.landmark
        landmarks: Dict[str, Landmark] = {
            name: Landmark(
                name       = name,
                x          = raw_lm[idx].x,
                y          = raw_lm[idx].y,
                z          = raw_lm[idx].z,
                visibility = raw_lm[idx].visibility,
            )
            for name, idx in LANDMARK_NAMES.items()
        }

        frame_pose = PoseFrame(
            frame_index = frame_index,
            timestamp_s = timestamp_s,
            detected    = True,
            landmarks   = landmarks,
        )

        # Compute biomechanical scalars in-place.
        frame_pose.torso_lean_deg = self._torso_lean(landmarks)
        frame_pose.left_knee_dev  = self._knee_dev(landmarks, "left")
        frame_pose.right_knee_dev = self._knee_dev(landmarks, "right")

        return frame_pose

    def close(self) -> None:
        """Release MediaPipe resources."""
        with self._lock:
            if self._pose is not None:
                self._pose.close()
                self._pose = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PoseModel":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _init(self) -> None:
        self._pose = mp.solutions.pose.Pose(**self._config)

    @staticmethod
    def _torso_lean(lm: Dict[str, Landmark]) -> Optional[float]:
        """Signed torso lean angle from vertical (degrees)."""
        import math
        try:
            sh_x = (lm["left_shoulder"].x + lm["right_shoulder"].x) / 2
            sh_y = (lm["left_shoulder"].y + lm["right_shoulder"].y) / 2
            hi_x = (lm["left_hip"].x      + lm["right_hip"].x)      / 2
            hi_y = (lm["left_hip"].y      + lm["right_hip"].y)      / 2
            return float(np.degrees(math.atan2(-(sh_x - hi_x), -(sh_y - hi_y))))
        except KeyError:
            return None

    @staticmethod
    def _knee_dev(lm: Dict[str, Landmark], side: str) -> Optional[float]:
        """Perpendicular knee deviation ratio relative to thigh length."""
        try:
            A = np.array([lm[f"{side}_hip"].x,   lm[f"{side}_hip"].y])
            B = np.array([lm[f"{side}_knee"].x,  lm[f"{side}_knee"].y])
            C = np.array([lm[f"{side}_ankle"].x, lm[f"{side}_ankle"].y])
            thigh = float(np.linalg.norm(B - A))
            if thigh < 1e-9:
                return None
            span = float(np.linalg.norm(C - A))
            if span < 1e-9:
                return None
            cross = float(np.cross(C - A, B - A))
            return (cross / span) / thigh
        except KeyError:
            return None
