#!/usr/bin/env python3
"""
Stage 3 — Ball Detector
========================
Detects the football in each frame using circular-object detection
and returns per-frame detections and overall confidence.

Responsibilities:
  - Detect circular objects consistent with a football in each frame
  - Return bounding circles (center_x, center_y, radius) per frame
  - Return overall ball detection confidence for the video
  - Detection is informational — the pipeline continues even if no ball found

Current implementation: Hough Circle Transform (fast, no GPU required).
Upgrade path: replace _detect_frame() with YOLOv8-ball or TrackNet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pipeline.frame_extractor import ExtractedFrame


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class BallDetection:
    """A detected ball in a single frame."""
    frame_index:  int
    timestamp_s:  float
    center_x:     float
    center_y:     float
    radius:       float
    confidence:   float   # Hough accumulator score normalised to [0, 1]


@dataclass
class BallDetectionResult:
    """Output of the BallDetector for one video."""
    detected_frames:  int
    total_frames:     int
    confidence:       float                  # fraction of frames with a ball
    ball_detected:    bool
    detections:       List[BallDetection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.05   # ball present in at least 5% of frames

# Hough circle parameters.
HOUGH_DP         = 1.2
HOUGH_MIN_DIST   = 30      # px between circle centres
HOUGH_PARAM1     = 50      # Canny high threshold
HOUGH_PARAM2     = 30      # accumulator threshold
HOUGH_MIN_RADIUS = 5       # px
HOUGH_MAX_RADIUS = 60      # px


class BallDetector:
    """
    Stage 3: Detect the football in video frames.

    Parameters
    ----------
    threshold : float
        Minimum detection confidence to set ball_detected=True.
        Default: 0.05.

    Usage::

        detector = BallDetector()
        result   = detector.detect(frames)
        print(result.ball_detected, result.confidence)
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def detect(self, frames: List[ExtractedFrame]) -> BallDetectionResult:
        """
        Run ball detection over a list of frames.

        Parameters
        ----------
        frames : list[ExtractedFrame]

        Returns
        -------
        BallDetectionResult
        """
        if not frames:
            return BallDetectionResult(
                detected_frames=0, total_frames=0,
                confidence=0.0, ball_detected=False,
            )

        detections: List[BallDetection] = []
        total = len(frames)

        for ef in frames:
            det = self._detect_frame(ef)
            if det is not None:
                detections.append(det)

        detected = len(detections)
        conf     = detected / total if total > 0 else 0.0

        return BallDetectionResult(
            detected_frames = detected,
            total_frames    = total,
            confidence      = round(conf, 3),
            ball_detected   = conf >= self.threshold,
            detections      = detections,
        )

    def _detect_frame(self, ef: ExtractedFrame) -> Optional[BallDetection]:
        """
        Detect the strongest ball candidate in a single frame.
        Returns None if no circle is found.
        """
        gray    = cv2.cvtColor(ef.bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp        = HOUGH_DP,
            minDist   = HOUGH_MIN_DIST,
            param1    = HOUGH_PARAM1,
            param2    = HOUGH_PARAM2,
            minRadius = HOUGH_MIN_RADIUS,
            maxRadius = HOUGH_MAX_RADIUS,
        )

        if circles is None:
            return None

        # Take the circle with the highest accumulator vote (first in array).
        circles = np.round(circles[0, :]).astype(int)
        cx, cy, r = circles[0]

        # Normalise radius to [0,1] as a rough confidence proxy.
        conf = min(1.0, r / HOUGH_MAX_RADIUS)

        return BallDetection(
            frame_index = ef.index,
            timestamp_s = ef.timestamp_s,
            center_x    = float(cx),
            center_y    = float(cy),
            radius      = float(r),
            confidence  = round(conf, 3),
        )
