#!/usr/bin/env python3
"""
Stage 1 — Frame Extractor
==========================
Reads frames from a VideoContext and yields them for downstream
processing.  Supports full extraction, stride-based sampling, and
time-range clipping.

Responsibilities:
  - Read frames from the OpenCV capture handle
  - Apply stride / sample-rate to reduce processing load
  - Optionally clip to a time range (start_s, end_s)
  - Yield (frame_index, timestamp_s, frame_bgr) tuples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Optional

import cv2
import numpy as np

from pipeline.video_loader import VideoContext


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ExtractedFrame:
    """A single frame produced by the FrameExtractor."""
    index:       int              # 0-based frame number in the original video
    timestamp_s: float            # time in seconds from start
    bgr:         np.ndarray       # raw BGR image (H, W, 3)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class FrameExtractor:
    """
    Stage 1: Extract frames from a VideoContext.

    Parameters
    ----------
    stride : int
        Process every Nth frame (1 = every frame, 5 = every 5th).
    start_s : float | None
        Start extraction at this timestamp (seconds).  None = beginning.
    end_s : float | None
        Stop extraction at this timestamp (seconds).  None = end.
    max_frames : int | None
        Hard cap on the number of frames yielded.

    Usage::

        extractor = FrameExtractor(stride=3)
        for frame in extractor.extract(context):
            process(frame.bgr)
    """

    def __init__(
        self,
        stride:     int            = 1,
        start_s:    Optional[float] = None,
        end_s:      Optional[float] = None,
        max_frames: Optional[int]   = None,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride     = stride
        self.start_s    = start_s
        self.end_s      = end_s
        self.max_frames = max_frames

    def extract(
        self,
        context: VideoContext,
    ) -> Generator[ExtractedFrame, None, None]:
        """
        Yield ExtractedFrame objects from the VideoContext.

        Parameters
        ----------
        context : VideoContext
            Must be open (context.is_open == True).

        Yields
        ------
        ExtractedFrame
        """
        if not context.is_open:
            raise RuntimeError("VideoContext capture handle is not open.")

        cap = context.capture
        fps = context.fps or 25.0

        # Seek to start position if requested.
        start_frame = 0
        if self.start_s is not None:
            start_frame = max(0, int(self.start_s * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        end_frame = context.frame_count
        if self.end_s is not None:
            end_frame = min(context.frame_count, int(self.end_s * fps))

        yielded  = 0
        idx      = start_frame

        while idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (idx - start_frame) % self.stride == 0:
                timestamp_s = idx / fps
                yield ExtractedFrame(
                    index       = idx,
                    timestamp_s = round(timestamp_s, 4),
                    bgr         = frame,
                )
                yielded += 1
                if self.max_frames is not None and yielded >= self.max_frames:
                    break

            idx += 1

    def extract_all(self, context: VideoContext) -> list[ExtractedFrame]:
        """Convenience wrapper — returns all frames as a list."""
        return list(self.extract(context))
