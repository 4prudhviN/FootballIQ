#!/usr/bin/env python3
"""
Pose Estimator  (MediaPipe)
============================
Runs MediaPipe Pose on every frame and stores all 33 body landmarks.

Input:  List[ExtractedFrame]
Output: PoseEstimationResult
          └── frame_poses  — List[FramePose]
              ├── frame_index
              ├── timestamp_s
              ├── detected     — True if MediaPipe found a pose
              └── landmarks    — dict of all 33 named landmarks
                  └── each: x, y, z (normalised [0,1]), visibility

Nothing is calculated here.
No angles, no warnings, no metrics.
Raw landmarks only — downstream modules do the analysis.

Writes to PipelineContext:
  ctx.detections.pose_landmarks  — List[PoseLandmarkFrame]
  ctx.detections.pose_detected_frames
  ctx.detections.pose_total_frames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from pipeline.frame_extractor  import ExtractedFrame
from pipeline.pipeline_context import PipelineContext, PoseLandmarkFrame
from config.constants import (
    POSE_MODEL_COMPLEXITY,
    POSE_DETECTION_CONFIDENCE,
    POSE_TRACKING_CONFIDENCE,
    POSE_LANDMARK_COUNT,
)
from utils.logger import get_logger

log = get_logger(__name__)

_mp_pose = mp.solutions.pose


# ---------------------------------------------------------------------------
# All 33 MediaPipe landmark names (in index order)
# ---------------------------------------------------------------------------

LANDMARK_NAMES: List[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

assert len(LANDMARK_NAMES) == POSE_LANDMARK_COUNT, (
    f"Expected {POSE_LANDMARK_COUNT} landmarks, got {len(LANDMARK_NAMES)}"
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PoseLandmark:
    """A single body landmark in normalised [0,1] coordinates."""
    name:       str
    x:          float     # normalised horizontal position [0,1]
    y:          float     # normalised vertical position [0,1]
    z:          float     # depth (relative to hips, normalised)
    visibility: float     # confidence that landmark is visible [0,1]

    @property
    def is_visible(self) -> bool:
        return self.visibility >= 0.5

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.z, self.visibility)


@dataclass
class FramePose:
    """
    All 33 pose landmarks for a single frame.
    No calculated values — raw storage only.
    """
    frame_index:  int
    timestamp_s:  float
    detected:     bool
    landmarks:    Dict[str, PoseLandmark] = field(default_factory=dict)

    def get(self, name: str) -> Optional[PoseLandmark]:
        """Return a landmark by name, or None if not detected."""
        return self.landmarks.get(name)

    @property
    def visible_count(self) -> int:
        """Number of landmarks with visibility ≥ 0.5."""
        return sum(1 for lm in self.landmarks.values() if lm.is_visible)

    @property
    def landmark_count(self) -> int:
        return len(self.landmarks)


@dataclass
class PoseEstimationResult:
    """Full pose estimation output for one video."""
    frame_poses:      List[FramePose]
    detected_frames:  int
    total_frames:     int

    # Aggregate placeholders — filled by downstream analysis modules, not here.
    warnings:         List[str]          = field(default_factory=list)
    avg_torso_lean:   Optional[float]    = None
    avg_knee_dev:     Optional[float]    = None
    gait_asymmetry:   Optional[float]    = None


# ---------------------------------------------------------------------------
# Pose Estimator
# ---------------------------------------------------------------------------

class PoseEstimator:
    """
    Runs MediaPipe Pose on extracted frames and stores all 33 landmarks.

    Parameters
    ----------
    model_complexity          : int   — 0=lite 1=full 2=heavy
    min_detection_confidence  : float
    min_tracking_confidence   : float

    Usage::

        estimator = PoseEstimator()
        result    = estimator.estimate(frames)
        estimator.write_to_context(result, ctx)
        estimator.close()

    Or as a context manager::

        with PoseEstimator() as estimator:
            result = estimator.estimate(frames)
    """

    def __init__(
        self,
        model_complexity:         int   = POSE_MODEL_COMPLEXITY,
        min_detection_confidence: float = POSE_DETECTION_CONFIDENCE,
        min_tracking_confidence:  float = POSE_TRACKING_CONFIDENCE,
    ) -> None:
        self._pose = _mp_pose.Pose(
            static_image_mode          = False,
            model_complexity           = model_complexity,
            enable_segmentation        = False,
            min_detection_confidence   = min_detection_confidence,
            min_tracking_confidence    = min_tracking_confidence,
        )
        log.debug("PoseEstimator initialised (complexity=%d)", model_complexity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frames: List[ExtractedFrame]) -> PoseEstimationResult:
        """
        Run MediaPipe Pose on all frames. Store raw landmarks only.

        Parameters
        ----------
        frames : List[ExtractedFrame]

        Returns
        -------
        PoseEstimationResult with all 33 landmarks per detected frame.
        No calculations performed — landmarks are raw storage.
        """
        frame_poses: List[FramePose] = []

        for ef in frames:
            fp = self._process_frame(ef)
            frame_poses.append(fp)

        detected = sum(1 for fp in frame_poses if fp.detected)

        log.debug(
            "PoseEstimator: %d/%d frames detected",
            detected, len(frames),
        )

        return PoseEstimationResult(
            frame_poses     = frame_poses,
            detected_frames = detected,
            total_frames    = len(frames),
        )

    def estimate_single(self, frame: ExtractedFrame) -> FramePose:
        """Run pose estimation on a single frame."""
        return self._process_frame(frame)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write to PipelineContext
    # ------------------------------------------------------------------

    @staticmethod
    def write_to_context(
        result: PoseEstimationResult,
        ctx:    PipelineContext,
    ) -> None:
        """
        Write raw landmark data to PipelineContext.
        Does NOT write any calculated values (torso_lean, knee_dev, etc.)
        — those are written by downstream analysis modules.
        """
        ctx.detections.pose_landmarks = [
            PoseLandmarkFrame(
                frame_index = fp.frame_index,
                timestamp_s = fp.timestamp_s,
                detected    = fp.detected,
                landmarks   = {
                    name: lm
                    for name, lm in fp.landmarks.items()
                },
            )
            for fp in result.frame_poses
        ]
        ctx.detections.pose_detected_frames = result.detected_frames
        ctx.detections.pose_total_frames    = result.total_frames

        ctx.log_stage(
            "pose_estimate",
            f"{result.detected_frames}/{result.total_frames} frames detected  "
            f"landmarks=33 per frame",
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _process_frame(self, ef: ExtractedFrame) -> FramePose:
        """
        Run MediaPipe on one frame.
        Store all 33 raw landmarks. No calculations.
        """
        rgb     = cv2.cvtColor(ef.bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return FramePose(
                frame_index = ef.index,
                timestamp_s = ef.timestamp_s,
                detected    = False,
            )

        raw_lm = results.pose_landmarks.landmark
        landmarks: Dict[str, PoseLandmark] = {}

        for idx, name in enumerate(LANDMARK_NAMES):
            lm = raw_lm[idx]
            landmarks[name] = PoseLandmark(
                name       = name,
                x          = float(lm.x),
                y          = float(lm.y),
                z          = float(lm.z),
                visibility = float(lm.visibility),
            )

        return FramePose(
            frame_index = ef.index,
            timestamp_s = ef.timestamp_s,
            detected    = True,
            landmarks   = landmarks,
        )
