#!/usr/bin/env python3
"""
Player Detector  (YOLO-backed)
===============================
Detects and tracks players in video frames.

Input:  List[ExtractedFrame]
Output: PlayerDetectionResult
          ├── per_frame_detections  — List[FramePlayerDetections]
          │   └── players           — List[DetectedPlayer]
          │       ├── player_id     — assigned tracking ID
          │       ├── bbox          — (x, y, w, h) normalised [0,1]
          │       ├── confidence    — YOLO detection confidence
          │       └── tracking_id   — persistent ID across frames
          ├── detected_frames       — how many frames had ≥1 player
          ├── total_frames
          └── passed                — True if above threshold

Detector backends (in priority order):
  1. YOLOv8 (ultralytics) — best accuracy, requires `pip install ultralytics`
  2. YOLOv5 (torch hub)   — fallback if ultralytics not installed
  3. MOG2 background subtraction — no-GPU fallback, always available

Multi-player support:
  - All players detected per frame are returned
  - Simple centroid-based tracking assigns consistent IDs across frames
  - Ready for full ByteTrack / BotSORT integration

Writes to PipelineContext:
  ctx.detections.player_tracks   — flat list of PlayerTrack per frame
  ctx.detections.player_confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.frame_extractor import ExtractedFrame
from pipeline.pipeline_context import PipelineContext, PlayerTrack
from config.constants import PLAYER_DETECTION_THRESHOLD, MOTION_PIXEL_RATIO
from config.settings  import settings
from utils.logger     import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DetectedPlayer:
    """One player detection in a single frame."""
    player_id:   str                            # e.g. "P1"
    tracking_id: int                            # persistent ID across frames
    bbox:        Tuple[float, float, float, float]  # (x, y, w, h) normalised [0,1]
    confidence:  float                          # 0.0–1.0
    frame_index: int
    timestamp_s: float

    @property
    def cx(self) -> float:
        """Bounding box centre x."""
        return self.bbox[0] + self.bbox[2] / 2

    @property
    def cy(self) -> float:
        """Bounding box centre y."""
        return self.bbox[1] + self.bbox[3] / 2


@dataclass
class FramePlayerDetections:
    """All player detections for one frame."""
    frame_index: int
    timestamp_s: float
    players:     List[DetectedPlayer] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.players)

    @property
    def primary(self) -> Optional[DetectedPlayer]:
        """Return the highest-confidence player."""
        return max(self.players, key=lambda p: p.confidence) if self.players else None


@dataclass
class PlayerDetectionResult:
    """Full output of the PlayerDetector for one video."""
    per_frame_detections: List[FramePlayerDetections]
    detected_frames:      int
    total_frames:         int
    confidence:           float    # fraction of frames with ≥1 player
    passed:               bool
    backend:              str      # "yolov8" | "yolov5" | "mog2"
    max_players_per_frame: int


# ---------------------------------------------------------------------------
# Centroid tracker (simple, no external dependency)
# ---------------------------------------------------------------------------

class _CentroidTracker:
    """
    Assigns consistent tracking IDs to detected players across frames
    by matching centroids (nearest-neighbour).

    Upgrade path: replace with ByteTrack or BotSORT for production.
    """

    def __init__(self, max_distance: float = 0.15) -> None:
        self.max_distance = max_distance
        self._next_id     = 1
        self._tracks:  Dict[int, Tuple[float, float]] = {}   # {track_id: (cx, cy)}
        self._missing: Dict[int, int]                 = {}   # {track_id: frames_missing}
        self._max_missing = 5

    def update(
        self,
        detections: List[Tuple[float, float]],   # list of (cx, cy)
    ) -> List[int]:
        """
        Match detections to existing tracks.
        Returns a list of track IDs, one per detection.
        """
        if not self._tracks:
            ids = []
            for cx, cy in detections:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = (cx, cy)
                ids.append(tid)
            return ids

        # Build cost matrix.
        track_ids  = list(self._tracks.keys())
        track_cxcy = [self._tracks[tid] for tid in track_ids]

        assigned_det = [-1] * len(detections)
        assigned_trk = [False] * len(track_ids)

        for di, (dcx, dcy) in enumerate(detections):
            best_dist = self.max_distance
            best_ti   = -1
            for ti, (tcx, tcy) in enumerate(track_cxcy):
                dist = float(np.hypot(dcx - tcx, dcy - tcy))
                if dist < best_dist:
                    best_dist = dist
                    best_ti   = ti
            if best_ti >= 0:
                assigned_det[di] = track_ids[best_ti]
                assigned_trk[best_ti] = True

        # Assign new IDs for unmatched detections.
        result_ids = []
        for di, (dcx, dcy) in enumerate(detections):
            if assigned_det[di] >= 0:
                tid = assigned_det[di]
                self._tracks[tid] = (dcx, dcy)
                self._missing.pop(tid, None)
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = (dcx, dcy)
            result_ids.append(tid)

        # Age out missing tracks.
        for ti, tid in enumerate(track_ids):
            if not assigned_trk[ti]:
                self._missing[tid] = self._missing.get(tid, 0) + 1
                if self._missing[tid] > self._max_missing:
                    del self._tracks[tid]
                    self._missing.pop(tid, None)

        return result_ids


# ---------------------------------------------------------------------------
# Detector backends
# ---------------------------------------------------------------------------

class _YOLOv8Backend:
    """YOLOv8 detection backend using ultralytics."""

    _model = None
    _model_path: str = ""

    @classmethod
    def load(cls, model_path: str = "") -> bool:
        try:
            from ultralytics import YOLO
            path = model_path or settings.MODEL_PATH or "yolov8n.pt"
            cls._model = YOLO(path)
            cls._model_path = path
            log.info("YOLOv8 loaded: %s", path)
            return True
        except Exception as exc:
            log.debug("YOLOv8 not available: %s", exc)
            return False

    @classmethod
    def detect(cls, bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Return list of (x_norm, y_norm, w_norm, h_norm, confidence) for class=person."""
        if cls._model is None:
            return []
        H, W = bgr.shape[:2]
        results = cls._model(bgr, classes=[0], verbose=False)   # class 0 = person
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append((
                    x1 / W, y1 / H,
                    (x2 - x1) / W, (y2 - y1) / H,
                    conf,
                ))
        return detections


class _MOG2Backend:
    """MOG2 background subtraction fallback (no GPU, no YOLO required)."""

    def __init__(self) -> None:
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )
        self._min_area = 0.01   # minimum fraction of frame area

    def detect(self, bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        H, W  = bgr.shape[:2]
        mask  = self._sub.apply(bgr)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        frame_area = H * W
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area / frame_area < self._min_area:
                continue
            conf = min(1.0, area / (frame_area * 0.15))
            detections.append((x / W, y / H, w / W, h / H, conf))
        return detections


# ---------------------------------------------------------------------------
# Player Detector
# ---------------------------------------------------------------------------

class PlayerDetector:
    """
    Detects and tracks players in video frames.

    Uses YOLOv8 if available, falls back to MOG2.
    All detected players per frame are returned — multi-player ready.
    Simple centroid tracker assigns consistent IDs across frames.

    Parameters
    ----------
    threshold     : float — min fraction of frames with player to pass
    model_path    : str   — path to custom YOLO weights (optional)
    min_confidence: float — minimum per-detection confidence

    Usage::

        detector = PlayerDetector()
        result   = detector.detect(frames)
        detector.write_to_context(result, ctx)
    """

    def __init__(
        self,
        threshold:      float = PLAYER_DETECTION_THRESHOLD,
        model_path:     str   = "",
        min_confidence: float = 0.30,
    ) -> None:
        self.threshold      = threshold
        self.min_confidence = min_confidence
        self._tracker       = _CentroidTracker()

        # Try YOLOv8 first.
        if _YOLOv8Backend.load(model_path):
            self._backend = _YOLOv8Backend()
            self._backend_name = "yolov8"
        else:
            self._backend = _MOG2Backend()
            self._backend_name = "mog2"
            log.info("PlayerDetector: using MOG2 fallback (install ultralytics for YOLOv8)")

    def detect(self, frames: List[ExtractedFrame]) -> PlayerDetectionResult:
        """
        Detect and track players across all frames.

        Parameters
        ----------
        frames : List[ExtractedFrame]

        Returns
        -------
        PlayerDetectionResult
        """
        if not frames:
            return PlayerDetectionResult(
                per_frame_detections=[], detected_frames=0,
                total_frames=0, confidence=0.0, passed=False,
                backend=self._backend_name, max_players_per_frame=0,
            )

        per_frame:  List[FramePlayerDetections] = []
        detected    = 0
        max_players = 0

        for ef in frames:
            raw = self._backend.detect(ef.bgr)
            H, W = ef.bgr.shape[:2]

            # Filter by confidence.
            raw = [d for d in raw if d[4] >= self.min_confidence]

            # Sort by confidence descending.
            raw.sort(key=lambda d: d[4], reverse=True)

            # Track centroids.
            centroids = [(d[0] + d[2] / 2, d[1] + d[3] / 2) for d in raw]
            track_ids = self._tracker.update(centroids) if centroids else []

            players: List[DetectedPlayer] = []
            for i, (x, y, w, h, conf) in enumerate(raw):
                tid = track_ids[i] if i < len(track_ids) else i + 1
                players.append(DetectedPlayer(
                    player_id   = f"P{tid}",
                    tracking_id = tid,
                    bbox        = (round(x, 4), round(y, 4), round(w, 4), round(h, 4)),
                    confidence  = round(conf, 3),
                    frame_index = ef.index,
                    timestamp_s = ef.timestamp_s,
                ))

            fd = FramePlayerDetections(
                frame_index = ef.index,
                timestamp_s = ef.timestamp_s,
                players     = players,
            )
            per_frame.append(fd)

            if players:
                detected    += 1
                max_players  = max(max_players, len(players))

        total      = len(frames)
        confidence = detected / total if total > 0 else 0.0

        return PlayerDetectionResult(
            per_frame_detections  = per_frame,
            detected_frames       = detected,
            total_frames          = total,
            confidence            = round(confidence, 3),
            passed                = confidence >= self.threshold,
            backend               = self._backend_name,
            max_players_per_frame = max_players,
        )

    @staticmethod
    def write_to_context(
        result: PlayerDetectionResult,
        ctx:    PipelineContext,
    ) -> None:
        """Write detection results to PipelineContext.detections."""
        ctx.detections.player_confidence = result.confidence
        ctx.detections.player_tracks = [
            PlayerTrack(
                player_id  = p.player_id,
                frame_index = p.frame_index,
                bbox        = p.bbox,
                confidence  = p.confidence,
            )
            for fd in result.per_frame_detections
            for p in fd.players
        ]
        ctx.log_stage(
            "player_detect",
            f"backend={result.backend}  "
            f"conf={result.confidence:.1%}  "
            f"passed={result.passed}  "
            f"max_players={result.max_players_per_frame}",
        )
