#!/usr/bin/env python3
"""
Video Utilities
===============
OpenCV helpers for reading, writing, and inspecting video files.
Import from here — never re-implement in another module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def get_video_info(path: str) -> dict:
    """
    Return basic metadata for a video file without opening a persistent handle.

    Returns
    -------
    dict with keys: fps, width, height, frame_count, duration_s, file_size_mb
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_s   = frame_count / fps if fps > 0 else 0.0
    file_size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0

    return {
        "fps":          round(fps, 2),
        "width":        width,
        "height":       height,
        "frame_count":  frame_count,
        "duration_s":   round(duration_s, 2),
        "file_size_mb": round(file_size_mb, 2),
    }


# ---------------------------------------------------------------------------
# Frame iteration
# ---------------------------------------------------------------------------

def iter_frames(
    path:       str,
    stride:     int            = 1,
    start_s:    Optional[float] = None,
    end_s:      Optional[float] = None,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """
    Iterate over frames of a video file.

    Yields
    ------
    (frame_index, timestamp_s, bgr_frame)

    Parameters
    ----------
    path    : str   Path to video file.
    stride  : int   Yield every Nth frame.
    start_s : float Start time in seconds (None = beginning).
    end_s   : float End time in seconds (None = end).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_s * fps) if start_s is not None else 0
    end_frame   = min(frame_count, int(end_s * fps)) if end_s is not None else frame_count

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = start_frame
    try:
        while idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (idx - start_frame) % stride == 0:
                yield idx, round(idx / fps, 4), frame
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def open_writer(
    output_path: str,
    fps:         float,
    width:       int,
    height:      int,
    codec:       str = "mp4v",
) -> cv2.VideoWriter:
    """
    Open and return a VideoWriter.

    Parameters
    ----------
    output_path : str   Destination file path (must end in .mp4).
    fps         : float
    width, height : int
    codec       : str   FourCC codec string (default 'mp4v').

    Returns
    -------
    cv2.VideoWriter — caller is responsible for calling .release().
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
    return writer


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames(
    path:       str,
    n:          int,
) -> list[np.ndarray]:
    """
    Return exactly N evenly-spaced frames from a video as BGR arrays.
    Useful for thumbnail generation or quick previews.
    """
    info   = get_video_info(path)
    total  = info["frame_count"]
    stride = max(1, total // n)

    frames = []
    for _, _, frame in iter_frames(path, stride=stride):
        frames.append(frame)
        if len(frames) >= n:
            break
    return frames


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def is_valid_video(path: str) -> bool:
    """Return True if the file exists and OpenCV can open it."""
    if not os.path.exists(path):
        return False
    cap = cv2.VideoCapture(path)
    ok  = cap.isOpened()
    cap.release()
    return ok


def extension_ok(filename: str) -> bool:
    """Return True if the file extension is a supported video format."""
    return Path(filename).suffix.lower() in {
        ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"
    }
