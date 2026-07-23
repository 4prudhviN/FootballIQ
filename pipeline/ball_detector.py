#!/usr/bin/env python3
"""
Ball Detector  (YOLO-backed)
=============================
Detects the football in every frame.

Input:  List[ExtractedFrame]
Output: BallDetectionResult
          ├── detections       — List[BallDetection]
          │   ├── frame_index  — frame number
          │   ├── timestamp_s  — time in video
          │   ├── center_x     — ball centre x (pixels)
          │   ├── center_y     — ball centre y (pixels)
          │   ├── radius       — estimated ball radius (pixels)
          │   └── confidence   — detection confidence 0.0–1.0
          ├── detected_frames
          ├── total_frames
          ├── confidence       — fraction of frames with ball
          └── ball_detected    — True if above threshold

Detection backends (automatic fallback):
  1. YOLOv8  (ultralytics)       — best accuracy for sports balls
  2. YOLOv5  (torch hub)         — fallback
  3. Hough circle transform      — no-GPU fallback, always available

Writes to PipelineContext:
  ctx.detections.ball_tracks     — List[BallTrack]
  ctx.detections.ball_confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pipeline.frame_extractor  import ExtractedFrame
from pipeline.pipeline_context import PipelineContext, BallTrack
from config.constants import BALL_MIN_RADIUS_PX, BALL_MAX_RADIUS_PX
from config.settings  import settings
from utils.logger     import get_logger

log = get_logger(__name__)

_DEFAULT_THRESHOLD = 0.05   # ball present in ≥ 5% of frames


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BallDetection:
    """Ball detected in a single frame."""
    frame_index: int
    timestamp_s: float
    center_x:    float      # pixels
    center_y:    float      # pixels
    radius:      float      # pixels
    confidence:  float      # 0.0–1.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)

    @property
    def center_norm(self) -> Tuple[float, float]:
        """Normalised centre — requires frame dimensions. Use raw pixels elsewhere."""
        return (self.center_x, self.center_y)


@dataclass
class BallDetectionResult:
    """Full output of BallDetector for one video."""
    detections:     List[BallDetection]
    detected_frames: int
    total_frames:   int
    confidence:     float       # fraction of frames with ball
    ball_detected:  bool
    backend:        str         # "yolov8" | "hough"


# ---------------------------------------------------------------------------
# Detection backends
# ---------------------------------------------------------------------------

class _YOLOv8BallBackend:
    """YOLOv8 detection for sports ball (COCO class 32)."""

    _model = None

    @classmethod
    def load(cls, model_path: str = "") -> bool:
        try:
            from ultralytics import YOLO
            path = model_path or settings.MODEL_PATH or "yolov8n.pt"
            cls._model = YOLO(path)
            log.info("YOLOv8 ball backend loaded: %s", path)
            return True
        except Exception as exc:
            log.debug("YOLOv8 ball backend not available: %s", exc)
            return False

    @classmethod
    def detect(cls, bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Detect the ball. Returns (cx, cy, radius, confidence) in pixels or None.
        COCO class 32 = sports ball.
        """
        if cls._model is None:
            return None
        try:
            results = cls._model(bgr, classes=[32], verbose=False)
            best_conf = 0.0
            best_det  = None
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        radius = min(x2 - x1, y2 - y1) / 2
                        best_conf = conf
                        best_det  = (cx, cy, radius, conf)
            return best_det
        except Exception as exc:
            log.debug("YOLOv8 ball detect error: %s", exc)
            return None


class _HoughBallBackend:
    """Hough circle transform fallback (no GPU required)."""

    def detect(self, bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Returns (cx, cy, radius, confidence) or None."""
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp        = 1.2,
            minDist   = 30,
            param1    = 50,
            param2    = 30,
            minRadius = BALL_MIN_RADIUS_PX,
            maxRadius = BALL_MAX_RADIUS_PX,
        )
        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(int)
        cx, cy, r = circles[0]
        conf = min(1.0, float(r) / BALL_MAX_RADIUS_PX)
        return (float(cx), float(cy), float(r), conf)


# ---------------------------------------------------------------------------
# Ball Detector
# ---------------------------------------------------------------------------

class BallDetector:
    """
    Detects the football in video frames using YOLO with Hough fallback.

    Parameters
    ----------
    threshold      : float — min fraction of frames with ball to set ball_detected=True
    model_path     : str   — custom YOLO weights path
    min_confidence : float — minimum detection confidence

    Usage::

        detector = BallDetector()
        result   = detector.detect(frames)
        detector.write_to_context(result, ctx)
    """

    def __init__(
        self,
        threshold:      float = _DEFAULT_THRESHOLD,
        model_path:     str   = "",
        min_confidence: float = 0.20,
    ) -> None:
        self.threshold      = threshold
        self.min_confidence = min_confidence

        if _YOLOv8BallBackend.load(model_path):
            self._backend      = _YOLOv8BallBackend()
            self._backend_name = "yolov8"
        else:
            self._backend      = _HoughBallBackend()
            self._backend_name = "hough"
            log.info("BallDetector: using Hough fallback (install ultralytics for YOLOv8)")

    def detect(self, frames: List[ExtractedFrame]) -> BallDetectionResult:
        """
        Detect the ball in every frame.

        Parameters
        ----------
        frames : List[ExtractedFrame]

        Returns
        -------
        BallDetectionResult
        """
        if not frames:
            return BallDetectionResult(
                detections=[], detected_frames=0, total_frames=0,
                confidence=0.0, ball_detected=False,
                backend=self._backend_name,
            )

        detections: List[BallDetection] = []

        for ef in frames:
            det = self._backend.detect(ef.bgr)
            if det is None:
                continue
            cx, cy, radius, conf = det
            if conf < self.min_confidence:
                continue
            detections.append(BallDetection(
                frame_index = ef.index,
                timestamp_s = ef.timestamp_s,
                center_x    = round(cx, 2),
                center_y    = round(cy, 2),
                radius      = round(radius, 2),
                confidence  = round(conf, 3),
            ))

        total      = len(frames)
        detected   = len(detections)
        confidence = detected / total if total > 0 else 0.0

        return BallDetectionResult(
            detections      = detections,
            detected_frames = detected,
            total_frames    = total,
            confidence      = round(confidence, 3),
            ball_detected   = confidence >= self.threshold,
            backend         = self._backend_name,
        )

    @staticmethod
    def write_to_context(
        result: BallDetectionResult,
        ctx:    PipelineContext,
    ) -> None:
        """Write detection results to PipelineContext.detections."""
        ctx.detections.ball_confidence = result.confidence
        ctx.detections.ball_tracks = [
            BallTrack(
                frame_index = d.frame_index,
                timestamp_s = d.timestamp_s,
                center_x    = d.center_x,
                center_y    = d.center_y,
                radius      = d.radius,
                confidence  = d.confidence,
            )
            for d in result.detections
        ]
        ctx.log_stage(
            "ball_detect",
            f"backend={result.backend}  "
            f"conf={result.confidence:.1%}  "
            f"found={result.ball_detected}  "
            f"frames={result.detected_frames}/{result.total_frames}",
        )
