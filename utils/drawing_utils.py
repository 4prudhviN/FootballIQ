#!/usr/bin/env python3
"""
Drawing Utilities
=================
OpenCV drawing helpers for annotating video frames with
skeletons, bounding boxes, text banners, and heatmaps.
Import from here — never re-implement in another module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------

GREEN   = (50,  255,  50)
CYAN    = (255, 255,   0)
RED     = (0,    50,  255)
ORANGE  = (0,   165,  255)
YELLOW  = (0,   255,  255)
WHITE   = (255, 255,  255)
GREY    = (150, 150,  150)
BLACK   = (0,     0,    0)
NEON    = (0,   255,  128)


# ---------------------------------------------------------------------------
# Skeleton drawing
# ---------------------------------------------------------------------------

# Standard bone connections used across the codebase.
SKELETON_CONNECTIONS: List[Tuple[str, str]] = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("right_hip",      "right_knee"),
    ("left_knee",      "left_ankle"),
    ("right_knee",     "right_ankle"),
    ("left_shoulder",  "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_elbow",    "right_wrist"),
]


def draw_skeleton(
    frame:     np.ndarray,
    landmarks: Dict[str, Tuple[float, float]],   # name → (x_px, y_px)
    color:     Tuple[int, int, int] = NEON,
    thickness: int   = 3,
    radius:    int   = 6,
) -> None:
    """
    Draw skeleton bones and joint circles onto a BGR frame (in-place).

    Parameters
    ----------
    frame     : BGR image (mutated).
    landmarks : dict mapping landmark name → (x_pixel, y_pixel).
    color     : BGR colour for bones and joints.
    thickness : line thickness in pixels.
    radius    : joint circle radius in pixels.
    """
    # Draw bones.
    for a, b in SKELETON_CONNECTIONS:
        pa = landmarks.get(a)
        pb = landmarks.get(b)
        if pa and pb:
            cv2.line(frame, _ipt(pa), _ipt(pb), color, thickness, cv2.LINE_AA)

    # Draw joints.
    for pt in landmarks.values():
        cv2.circle(frame, _ipt(pt), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, _ipt(pt), radius, BLACK, 1,  cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Text overlays
# ---------------------------------------------------------------------------

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO  = cv2.FONT_HERSHEY_PLAIN


def draw_text(
    frame:     np.ndarray,
    text:      str,
    x:         int,
    y:         int,
    color:     Tuple[int, int, int] = WHITE,
    scale:     float = 0.65,
    thickness: int   = 2,
    bg:        bool  = True,
) -> None:
    """Draw text with an optional dark background for readability."""
    if bg:
        (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
        cv2.rectangle(
            frame,
            (x - 2, y - th - baseline - 2),
            (x + tw + 2, y + baseline + 2),
            (0, 0, 0), -1,
        )
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_warning_banner(
    frame:    np.ndarray,
    messages: List[str],
    alpha:    float = 0.70,
) -> None:
    """
    Draw a semi-transparent warning banner at the top of the frame.
    Each message appears on its own line in red.
    """
    if not messages:
        return
    h, w = frame.shape[:2]
    banner_h = 40 + 28 * len(messages)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), BLACK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, msg in enumerate(messages):
        y = 30 + i * 28
        (tw, _), _ = cv2.getTextSize(msg, FONT, 0.70, 2)
        x = (w - tw) // 2
        cv2.putText(frame, msg, (x, y), FONT, 0.70, RED, 2, cv2.LINE_AA)


def draw_metric_overlay(
    frame:   np.ndarray,
    metrics: Dict[str, str],
    x:       int = 12,
    start_y: int = -1,
) -> None:
    """
    Draw a list of label: value pairs in the bottom-left corner of the frame.
    """
    h = frame.shape[0]
    n = len(metrics)
    if start_y < 0:
        start_y = h - 12 - (n - 1) * 22

    for i, (label, value) in enumerate(metrics.items()):
        y = start_y + i * 22
        draw_text(frame, f"{label}: {value}", x, y, color=GREY, scale=0.55)


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

def draw_bbox(
    frame:  np.ndarray,
    x:      int,
    y:      int,
    w:      int,
    h:      int,
    color:  Tuple[int, int, int] = CYAN,
    label:  Optional[str]        = None,
    thickness: int = 2,
) -> None:
    """Draw an axis-aligned bounding box with an optional label."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
    if label:
        draw_text(frame, label, x, max(0, y - 6), color=color, scale=0.55, thickness=1)


def draw_circle_detection(
    frame:      np.ndarray,
    cx:         int,
    cy:         int,
    radius:     int,
    color:      Tuple[int, int, int] = YELLOW,
    thickness:  int = 2,
) -> None:
    """Draw a detected circle (e.g. ball) on the frame."""
    cv2.circle(frame, (cx, cy), radius, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Angle arc
# ---------------------------------------------------------------------------

def draw_angle_arc(
    frame:     np.ndarray,
    vertex_px: Tuple[int, int],
    angle_deg: float,
    color:     Tuple[int, int, int] = YELLOW,
    radius:    int = 20,
) -> None:
    """Draw a small arc at a joint to visualise the joint angle."""
    cv2.ellipse(
        frame,
        vertex_px,
        (radius, radius),
        0, -angle_deg / 2, angle_deg / 2,
        color, 2, cv2.LINE_AA,
    )
    text_x = vertex_px[0] + radius + 4
    text_y = vertex_px[1] - 4
    draw_text(frame, f"{angle_deg:.0f}°", text_x, text_y,
              color=color, scale=0.45, thickness=1, bg=False)


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def draw_progress_bar(
    frame:    np.ndarray,
    progress: float,            # 0.0 – 1.0
    x:        int = 0,
    y:        int = -1,
    width:    int = -1,
    height:   int = 6,
    color:    Tuple[int, int, int] = GREEN,
) -> None:
    """Draw a horizontal progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_y = y if y >= 0 else h - height - 2
    bar_w = width if width > 0 else w
    cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + height), GREY, -1)
    filled = int(bar_w * max(0.0, min(1.0, progress)))
    if filled > 0:
        cv2.rectangle(frame, (x, bar_y), (x + filled, bar_y + height), color, -1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ipt(pt: Tuple[float, float]) -> Tuple[int, int]:
    """Convert float tuple to integer pixel tuple for OpenCV."""
    return int(pt[0]), int(pt[1])
