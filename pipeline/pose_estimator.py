#!/usr/bin/env python3
"""
Stage 4 — Pose Estimator
=========================
Runs MediaPipe Pose on each frame and returns per-frame landmark
data and aggregated biomechanical warnings.

Responsibilities:
  - Run MediaPipe full-body pose estimation on every frame
  - Return normalised landmarks per frame
  - Compute torso lean, knee deviation, and gait asymmetry
  - Emit warning flags consumed by the Activity Detector and Analyzers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from pipeline.frame_extractor import ExtractedFrame


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class PoseLandmark:
    """A single body landmark in normalised coordinates [0, 1]."""
    x:          float
    y:          float
    z:          float
    visibility: float


@dataclass
class FramePose:
    """Pose estimation result for one frame."""
    frame_index:  int
    timestamp_s:  float
    detected:     bool
    landmarks:    Dict[str, PoseLandmark] = field(default_factory=dict)
    torso_lean:   Optional[float] = None   # degrees, negative = leaning back
    knee_dev_l:   Optional[float] = None   # left knee deviation ratio
    knee_dev_r:   Optional[float] = None   # right knee deviation ratio


@dataclass
class PoseEstimationResult:
    """Aggregated pose estimation result for the whole video."""
    frame_poses:         List[FramePose]
    detected_frames:     int
    total_frames:        int
    warnings:            List[str]
    avg_torso_lean:      Optional[float]
    avg_knee_dev:        Optional[float]
    gait_asymmetry:      Optional[float]


# ---------------------------------------------------------------------------
# Warning thresholds (mirrors analyze_movement.py)
# ---------------------------------------------------------------------------

TORSO_LEAN_THRESHOLD   = 15.0   # degrees
KNEE_DEV_THRESHOLD     = 0.25   # ratio of thigh length
GAIT_ASYMMETRY_THRESH  = 0.15   # fraction


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class PoseEstimator:
    """
    Stage 4: Run MediaPipe Pose on extracted frames.

    Parameters
    ----------
    model_complexity : int
        0 = lite (fast), 1 = full (balanced), 2 = heavy (accurate).
    min_detection_confidence : float
    min_tracking_confidence  : float

    Usage::

        estimator = PoseEstimator()
        result    = estimator.estimate(frames)
        print(result.warnings)
    """

    def __init__(
        self,
        model_complexity:           int   = 1,
        min_detection_confidence:   float = 0.5,
        min_tracking_confidence:    float = 0.5,
    ) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode           = False,
            model_complexity            = model_complexity,
            enable_segmentation         = False,
            min_detection_confidence    = min_detection_confidence,
            min_tracking_confidence     = min_tracking_confidence,
        )

    def estimate(self, frames: List[ExtractedFrame]) -> PoseEstimationResult:
        """
        Run pose estimation on all frames and return aggregated results.
        """
        frame_poses: List[FramePose] = []

        for ef in frames:
            fp = self._process_frame(ef)
            frame_poses.append(fp)

        return self._aggregate(frame_poses)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _process_frame(self, ef: ExtractedFrame) -> FramePose:
        rgb     = cv2.cvtColor(ef.bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return FramePose(
                frame_index = ef.index,
                timestamp_s = ef.timestamp_s,
                detected    = False,
            )

        lm_list = results.pose_landmarks.landmark
        lm_names = {
            "left_shoulder":  mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
            "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
            "left_hip":       mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
            "right_hip":      mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
            "left_knee":      mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
            "right_knee":     mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value,
            "left_ankle":     mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value,
            "right_ankle":    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value,
        }

        landmarks: Dict[str, PoseLandmark] = {}
        for name, idx in lm_names.items():
            lm = lm_list[idx]
            landmarks[name] = PoseLandmark(
                x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
            )

        torso_lean = self._torso_lean(landmarks)
        knee_l     = self._knee_dev(landmarks, "left")
        knee_r     = self._knee_dev(landmarks, "right")

        return FramePose(
            frame_index = ef.index,
            timestamp_s = ef.timestamp_s,
            detected    = True,
            landmarks   = landmarks,
            torso_lean  = torso_lean,
            knee_dev_l  = knee_l,
            knee_dev_r  = knee_r,
        )

    def _aggregate(self, frame_poses: List[FramePose]) -> PoseEstimationResult:
        detected = [fp for fp in frame_poses if fp.detected]

        torso_leans = [abs(fp.torso_lean) for fp in detected if fp.torso_lean is not None]
        knee_devs   = [
            max(abs(fp.knee_dev_l or 0), abs(fp.knee_dev_r or 0))
            for fp in detected
        ]

        avg_torso = float(np.mean(torso_leans)) if torso_leans else None
        avg_knee  = float(np.mean(knee_devs))   if knee_devs   else None
        gait_asym = self._gait_asymmetry(detected)

        warnings: List[str] = []
        if avg_torso is not None and avg_torso > TORSO_LEAN_THRESHOLD:
            warnings.append("POOR POSTURE / LEANING BACK")
        if avg_knee is not None and avg_knee > KNEE_DEV_THRESHOLD:
            warnings.append("KNEE ALIGNMENT RISK")
        if gait_asym is not None and gait_asym > GAIT_ASYMMETRY_THRESH:
            warnings.append("ASYMMETRIC GAIT DETECTED")

        return PoseEstimationResult(
            frame_poses     = frame_poses,
            detected_frames = len(detected),
            total_frames    = len(frame_poses),
            warnings        = warnings,
            avg_torso_lean  = round(avg_torso, 2) if avg_torso is not None else None,
            avg_knee_dev    = round(avg_knee,  3) if avg_knee  is not None else None,
            gait_asymmetry  = round(gait_asym, 3) if gait_asym is not None else None,
        )

    @staticmethod
    def _torso_lean(lm: Dict[str, PoseLandmark]) -> Optional[float]:
        """Signed torso lean angle from vertical (degrees)."""
        try:
            sh_x = (lm["left_shoulder"].x + lm["right_shoulder"].x) / 2
            sh_y = (lm["left_shoulder"].y + lm["right_shoulder"].y) / 2
            hi_x = (lm["left_hip"].x      + lm["right_hip"].x)      / 2
            hi_y = (lm["left_hip"].y      + lm["right_hip"].y)      / 2
            dx, dy = sh_x - hi_x, sh_y - hi_y
            return float(np.degrees(math.atan2(-dx, -dy)))
        except (KeyError, ZeroDivisionError):
            return None

    @staticmethod
    def _knee_dev(lm: Dict[str, PoseLandmark], side: str) -> Optional[float]:
        """Perpendicular knee deviation ratio relative to thigh length."""
        try:
            A = np.array([lm[f"{side}_hip"].x,   lm[f"{side}_hip"].y])
            B = np.array([lm[f"{side}_knee"].x,  lm[f"{side}_knee"].y])
            C = np.array([lm[f"{side}_ankle"].x, lm[f"{side}_ankle"].y])
            thigh = float(np.linalg.norm(B - A))
            if thigh < 1e-6:
                return None
            cross = float(np.cross(C - A, B - A))
            span  = float(np.linalg.norm(C - A))
            if span < 1e-6:
                return None
            return (cross / span) / thigh
        except KeyError:
            return None

    @staticmethod
    def _gait_asymmetry(detected: List[FramePose]) -> Optional[float]:
        """Rough gait asymmetry from ankle Y-position variance between sides."""
        left_y  = [fp.landmarks["left_ankle"].y  for fp in detected if "left_ankle"  in fp.landmarks]
        right_y = [fp.landmarks["right_ankle"].y for fp in detected if "right_ankle" in fp.landmarks]
        if len(left_y) < 3 or len(right_y) < 3:
            return None
        avg_l = float(np.mean(left_y))
        avg_r = float(np.mean(right_y))
        denom = max(avg_l, avg_r)
        return abs(avg_l - avg_r) / denom if denom > 0 else 0.0
