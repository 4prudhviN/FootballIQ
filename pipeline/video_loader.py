#!/usr/bin/env python3
"""
Stage 0 — Video Loader
=======================
Validates and opens a video file, extracts metadata, and
returns a VideoContext object consumed by every downstream stage.

Responsibilities:
  - Validate file exists and is a supported format
  - Open the video capture handle (OpenCV)
  - Extract metadata: fps, resolution, frame count, duration
  - Pass the handle + metadata to the next stage
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class VideoContext:
    """Carries the open video handle and metadata through the pipeline."""

    path:        str
    capture:     Optional[cv2.VideoCapture] = field(default=None, repr=False)
    fps:         float = 0.0
    width:       int   = 0
    height:      int   = 0
    frame_count: int   = 0
    duration_s:  float = 0.0

    def release(self) -> None:
        """Release the OpenCV capture handle."""
        if self.capture and self.capture.isOpened():
            self.capture.release()

    @property
    def is_open(self) -> bool:
        return self.capture is not None and self.capture.isOpened()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class VideoLoader:
    """
    Stage 0: Load and validate a video file.

    Usage::

        loader  = VideoLoader()
        context = loader.load("path/to/clip.mp4")
    """

    def load(self, path: str) -> VideoContext:
        """
        Open the video file and return a populated VideoContext.

        Parameters
        ----------
        path : str
            Absolute or relative path to the video file.

        Returns
        -------
        VideoContext

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file extension is not supported or the file cannot be opened.
        """
        p = Path(path).resolve()

        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{p.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video file: {path}")

        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s  = frame_count / fps if fps > 0 else 0.0

        if width == 0 or height == 0:
            cap.release()
            raise ValueError(f"Video has invalid dimensions ({width}x{height}): {path}")

        return VideoContext(
            path        = str(p),
            capture     = cap,
            fps         = fps,
            width       = width,
            height      = height,
            frame_count = frame_count,
            duration_s  = duration_s,
        )
