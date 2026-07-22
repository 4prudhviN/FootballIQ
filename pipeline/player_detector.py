#!/usr/bin/env python3
"""
Stage 2 — Player Detector
==========================
Determines whether a player is present in each frame and
returns a detection confidence score for the whole video.

Responsibilities:
  - Detect human-shaped motion / presence in frames
  - Return per-frame detection flags
  - Return overall confidence (fraction of frames with a player)
  - Raise an error if confidence is below the minimum threshold

Current implementation: MOG2 background subtraction (fast, no GPU required).
Upgrade path: replace _detect_frame() with a YOLOv8-person detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from pipeline.frame_extractor import ExtractedFrame


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class PlayerDetectionResult:
    """Output of the PlayerDetector for one video."""
    detected_frames:   int
    total_frames:      int
    confidence:        float          # fraction of frames with a player
    passed:            bool           # True if confidence >= threshold
    frame_flags:       List[bool] = field(default_factory=list)  # per-frame


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

# Minimum fraction of frames that must contain a player.
DEFAULT_THRESHOLD = 0.10

# Minimum fraction of pixels that must be in motion to count as "player present".
MOTION_PIXEL_RATIO = 0.02


class PlayerDetector:
    """
    Stage 2: Detect whether a player is visible in the video.

    Parameters
    ----------
    threshold : float
        Minimum detection confidence (0–1) required to pass.
        Default: 0.10 (player present in at least 10% of frames).

    Usage::

        detector = PlayerDetector()
        result   = detector.detect(frames)
        if not result.passed:
            raise ValueError("No player detected")
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold  = threshold
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=50,
            detectShadows=False,
        )

    def detect(self, frames: List[ExtractedFrame]) -> PlayerDetectionResult:
        """
        Run player detection over a list of frames.

        Parameters
        ----------
        frames : list[ExtractedFrame]

        Returns
        -------
        PlayerDetectionResult
        """
        if not frames:
            return PlayerDetectionResult(
                detected_frames=0, total_frames=0,
                confidence=0.0, passed=False,
            )

        frame_flags: List[bool] = []

        for ef in frames:
            flag = self._detect_frame(ef.bgr)
            frame_flags.append(flag)

        detected = sum(frame_flags)
        total    = len(frame_flags)
        conf     = detected / total if total > 0 else 0.0

        return PlayerDetectionResult(
            detected_frames = detected,
            total_frames    = total,
            confidence      = round(conf, 3),
            passed          = conf >= self.threshold,
            frame_flags     = frame_flags,
        )

    def _detect_frame(self, bgr: np.ndarray) -> bool:
        """
        Return True if a player-like moving region is detected in this frame.

        Uses MOG2 background subtraction — large motion blobs indicate a
        person is present.  Replace with YOLOv8 for production accuracy.
        """
        mask         = self._subtractor.apply(bgr)
        total_pixels = bgr.shape[0] * bgr.shape[1]
        motion_ratio = cv2.countNonZero(mask) / total_pixels
        return motion_ratio > MOTION_PIXEL_RATIO
