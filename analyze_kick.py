#!/usr/bin/env python3
"""
Football Kick Analysis — MediaPipe Pose + OpenCV
=================================================

Analyzes a video of a football (soccer) kicking motion by:

  1. Loading a local video file.
  2. Running Google MediaPipe Pose (full-body landmark tracking) on every frame.
  3. Calculating the player's torso lean angle (shoulder‑midpoint → hip‑midpoint
     line relative to vertical).
  4. Detecting the kick impact moment by tracking the kicking leg's acceleration
     (frame‑to‑frame change in ankle / knee displacement).
  5. Flagging a "LEANING BACK TOO FAR" warning when the torso is leaned back
     beyond 15° at the moment of peak leg acceleration.
  6. Drawing a bright‑green skeleton overlay (shoulders, hips, knees, ankles)
     and any warnings directly onto every frame.
  7. Saving the annotated video as 'analyzed_kick.mp4'.

Usage:
    python analyze_kick.py                    # uses default 'input_kick.mp4'
    python analyze_kick.py --input my_clip.mp4  # custom input path
    python analyze_kick.py --input clip.mp4 --output result.mp4

Dependencies:
    pip install opencv-python mediapipe numpy
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Constants (tweak these to tune sensitivity)
# ---------------------------------------------------------------------------

# Angle threshold (degrees from vertical).  If the torso leans back past this
# threshold at the moment of peak kick acceleration, we show a warning.
LEAN_THRESHOLD_DEG: float = 15.0

# Minimum confidence values for MediaPipe — tune down for noisy / low‑res video.
DETECTION_CONFIDENCE: float = 0.5
TRACKING_CONFIDENCE: float = 0.5

# How many consecutive frames the kick speed must exceed the threshold before
# we consider it the "impact" moment (avoids false positives from jitter).
MIN_IMPACT_WINDOW: int = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Point2D:
    """A 2‑D pixel coordinate (x, y) together with visibility from MediaPipe."""
    x: float
    y: float
    visibility: float = 1.0


@dataclass
class Skeleton:
    """
    Lightweight container for the landmarks we care about.
    All coordinates are in *pixel* space.
    """
    left_shoulder:  Optional[Point2D] = None
    right_shoulder: Optional[Point2D] = None
    left_hip:       Optional[Point2D] = None
    right_hip:      Optional[Point2D] = None
    left_knee:      Optional[Point2D] = None
    right_knee:     Optional[Point2D] = None
    left_ankle:     Optional[Point2D] = None
    right_ankle:    Optional[Point2D] = None

    def is_valid(self) -> bool:
        """Return True only if every landmark we need is present."""
        return all(
            getattr(self, f) is not None
            for f in (
                "left_shoulder", "right_shoulder",
                "left_hip", "right_hip",
                "left_knee", "right_knee",
                "left_ankle", "right_ankle",
            )
        )


# ---------------------------------------------------------------------------
# MediaPipe → Skeleton converter
# ---------------------------------------------------------------------------

def landmarks_to_skeleton(
    landmarks,  # mp.solutions.pose.PoseLandmark list
    frame_w: int,
    frame_h: int,
) -> Skeleton:
    """
    Convert raw MediaPipe Pose landmarks (normalised 0‑1) into a `Skeleton`
    struct with pixel coordinates.  Returns an **empty** Skeleton if the pose
    was not detected (all fields stay None).
    """
    if landmarks is None:
        return Skeleton()

    def pt(idx: int) -> Optional[Point2D]:
        """Extract a single landmark, scaled to pixel space."""
        lm = landmarks[idx]
        # MediaPipe reports visibility in [0, 1]; we keep it for later use.
        return Point2D(
            x=lm.x * frame_w,
            y=lm.y * frame_h,
            visibility=lm.visibility,
        )

    skel = Skeleton()
    skel.left_shoulder  = pt(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value)
    skel.right_shoulder = pt(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value)
    skel.left_hip       = pt(mp.solutions.pose.PoseLandmark.LEFT_HIP.value)
    skel.right_hip      = pt(mp.solutions.pose.PoseLandmark.RIGHT_HIP.value)
    skel.left_knee      = pt(mp.solutions.pose.PoseLandmark.LEFT_KNEE.value)
    skel.right_knee     = pt(mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value)
    skel.left_ankle     = pt(mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value)
    skel.right_ankle    = pt(mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value)
    return skel


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def midpoint(a: Point2D, b: Point2D) -> Point2D:
    """Return the midpoint of two points."""
    return Point2D(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0)


def torso_lean_angle(skel: Skeleton) -> Optional[float]:
    """
    Compute the torso lean angle in degrees **from vertical**.

    The torso is represented by the line connecting:
        shoulder_midpoint → hip_midpoint

    Angle = 0  → perfectly upright.
    Angle > 0  → leaning forward.
    Angle < 0  → leaning backward (the dangerous case for a kick).

    Returns None if the skeleton is incomplete.
    """
    if not skel.is_valid():
        return None

    # Midpoints
    shoulder_mid = midpoint(skel.left_shoulder, skel.right_shoulder)
    hip_mid = midpoint(skel.left_hip, skel.right_hip)

    # Vector from hips → shoulders (torso axis, pointing up).
    dx = shoulder_mid.x - hip_mid.x
    dy = shoulder_mid.y - hip_mid.y

    # Angle between this vector and the upward vertical (0, -1).
    #   dot( (dx, dy), (0, -1) ) = -dy
    #   cross_z( (dx, dy), (0, -1) ) = -dx   (positive = rightward lean)
    #
    # atan2(cross, dot) gives signed angle in radians.
    angle_rad = np.arctan2(-dx, -dy)  # vertical reference (pointing up)
    angle_deg = np.degrees(angle_rad)

    # Clamp to [-90, 90] — we don't care about upside‑down poses.
    return float(np.clip(angle_deg, -90.0, 90.0))


def leg_speed(skel: Skeleton, prev_skel: Skeleton) -> Optional[float]:
    """
    Estimate the *speed* of the kicking leg as the average pixel displacement
    of the ankle and knee between two consecutive frames.

    A higher value means faster leg movement (swing phase of the kick).
    We track *both* legs and return the larger speed — MediaPipe sometimes
    swaps left/right labels when the player faces sideways.
    """
    if not skel.is_valid() or not prev_skel.is_valid():
        return None

    def displacement(a: Point2D, b: Point2D) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    # Left leg
    left_knee_disp = displacement(skel.left_knee, prev_skel.left_knee)
    left_ankle_disp = displacement(skel.left_ankle, prev_skel.left_ankle)
    left_speed = (left_knee_disp + left_ankle_disp) / 2.0

    # Right leg
    right_knee_disp = displacement(skel.right_knee, prev_skel.right_knee)
    right_ankle_disp = displacement(skel.right_ankle, prev_skel.right_ankle)
    right_speed = (right_knee_disp + right_ankle_disp) / 2.0

    return max(left_speed, right_speed)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

SKELETON_COLOR = (0, 255, 128)       # bright green (BGR for OpenCV)
SKELETON_THICKNESS = 3
JOINT_RADIUS = 6
WARNING_COLOR = (0, 50, 255)        # orange‑red (BGR)
WARNING_THICKNESS = 3
TEXT_COLOR = (255, 255, 255)        # white
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_skeleton(frame: np.ndarray, skel: Skeleton) -> None:
    """Draw bright‑green skeleton lines and joint dots on the frame (mutates)."""
    if not skel.is_valid():
        return

    # Define the bone connections we want to draw:
    #   (start_attr, end_attr) — each is a string matching the Skeleton field.
    bones = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    ]

    for attr_a, attr_b in bones:
        pa: Point2D = getattr(skel, attr_a)
        pb: Point2D = getattr(skel, attr_b)
        if pa is None or pb is None:
            continue
        # Convert to integer tuples for OpenCV.
        pt_a = (int(pa.x), int(pa.y))
        pt_b = (int(pb.x), int(pb.y))
        cv2.line(frame, pt_a, pt_b, SKELETON_COLOR, SKELETON_THICKNESS)

    # Draw joint circles.
    joint_attrs = [
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    for attr in joint_attrs:
        p: Optional[Point2D] = getattr(skel, attr)
        if p is None:
            continue
        center = (int(p.x), int(p.y))
        cv2.circle(frame, center, JOINT_RADIUS, SKELETON_COLOR, -1)  # filled


def draw_warning(frame: np.ndarray, message: str, angle: float) -> None:
    """
    Draw a banner‑style warning at the top of the frame.
    Displays the lean angle value alongside the message.
    """
    h, w, _ = frame.shape

    # Semi‑transparent overlay bar at the top.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Warning text.
    warning_text = f"⚠ {message}  (torso lean: {angle:.1f}°)"
    cv2.putText(
        frame, warning_text,
        (20, 45), FONT, 0.85, WARNING_COLOR, WARNING_THICKNESS, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_video(input_path: str, output_path: str = "analyzed_kick.mp4") -> int:
    """
    Open `input_path`, run pose estimation + kick analysis on every frame,
    save the annotated result to `output_path`.

    Returns 0 on success, 1 on error.
    """

    # ── 1. Validate input ──────────────────────────────────────────────────
    if not input_path:
        print("[ERROR] No input video path provided.", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {input_path}", file=sys.stderr)
        return 1

    # ── 2. Video metadata ──────────────────────────────────────────────────
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if w == 0 or h == 0:
        print("[ERROR] Invalid video dimensions.", file=sys.stderr)
        cap.release()
        return 1

    print(f"[INFO] Input:  {input_path}  ({w}x{h} @ {fps:.1f} fps, {total_frames} frames)")

    # ── 3. Video writer ────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # or 'avc1' depending on platform
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[ERROR] Cannot create output video: {output_path}", file=sys.stderr)
        cap.release()
        return 1

    # ── 4. MediaPipe Pose initialisation ───────────────────────────────────
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,                 # 0 = lite, 1 = full, 2 = heavy
        enable_segmentation=False,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    # ── 5. State variables ─────────────────────────────────────────────────
    prev_skeleton: Optional[Skeleton] = None
    kick_impact_frame: Optional[int] = None  # frame index where peak accel occurs
    max_speed_observed: float = 0.0
    speed_buffer: list[float] = []           # sliding window of leg speeds
    frame_idx: int = 0
    warning_flagged: bool = False
    lean_at_impact: Optional[float] = None

    print(f"[INFO] Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        frame_idx += 1

        # ── 5a. Convert BGR → RGB (MediaPipe expects RGB) ──────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # ── 5b. Convert landmarks → Skeleton ───────────────────────────────
        skeleton = landmarks_to_skeleton(results.pose_landmarks, w, h)

        if skeleton.is_valid():
            # ── 5c. Draw the skeleton overlay ──────────────────────────────
            draw_skeleton(frame, skeleton)

            # ── 5d. Calculate torso lean angle for this frame ──────────────
            lean = torso_lean_angle(skeleton)

            # ── 5e. Calculate leg speed (displacement from previous frame) ─
            if prev_skeleton is not None:
                speed = leg_speed(skeleton, prev_skeleton)
                if speed is not None:
                    speed_buffer.append(speed)

                    # Keep a sliding window of recent speed values.
                    if len(speed_buffer) > MIN_IMPACT_WINDOW:
                        speed_buffer.pop(0)

                    # Detect "peak leg acceleration" — the moment the leg is
                    # moving fastest.  We compare the average speed in the
                    # window against the all‑time maximum.
                    avg_speed = float(np.mean(speed_buffer))
                    if avg_speed > max_speed_observed:
                        max_speed_observed = avg_speed
                        kick_impact_frame = frame_idx
                        # At this candidate impact frame, check the lean angle.
                        if lean is not None:
                            lean_at_impact = lean
                            if lean < -LEAN_THRESHOLD_DEG:
                                warning_flagged = True

            # ── 5f. Display lean angle (always, for debugging) ─────────────
            if lean is not None:
                cv2.putText(
                    frame,
                    f"Torso lean: {lean:+.1f}°",
                    (20, h - 20),
                    FONT, 0.65, (200, 200, 200), 2, cv2.LINE_AA,
                )

            # ── 5g. Show impact frame marker ───────────────────────────────
            if kick_impact_frame is not None and frame_idx == kick_impact_frame:
                cv2.putText(
                    frame,
                    "KICK IMPACT",
                    (w - 220, 45),
                    FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
                )

        else:
            # Player is partially out of bounds or not detected.
            cv2.putText(
                frame,
                "Player not in frame",
                (20, h - 20),
                FONT, 0.65, (100, 100, 100), 2, cv2.LINE_AA,
            )

        # ── 5h. Warning banner (if condition was met at peak acceleration) ─
        if warning_flagged:
            draw_warning(
                frame,
                "LEANING BACK TOO FAR",
                angle=lean_at_impact if lean_at_impact is not None else 0.0,
            )

        # ── 6. Write annotated frame to output ─────────────────────────────
        writer.write(frame)

        # Keep previous skeleton for speed calculation.
        prev_skeleton = skeleton

        # Progress indicator (every 10 %)
        if total_frames > 0 and frame_idx % max(1, total_frames // 10) == 0:
            pct = int(frame_idx / total_frames * 100)
            print(f"   ... {pct}% ({frame_idx}/{total_frames})")

    # ── 7. Cleanup ─────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    pose.close()
    cv2.destroyAllWindows()

    print(f"[INFO] Output: {output_path}")
    if warning_flagged:
        print(f"[RESULT] ⚠  WARNING: Player was leaning back "
              f"({lean_at_impact:.1f}°) at kick impact (frame {kick_impact_frame}).")
    else:
        if lean_at_impact is not None:
            print(f"[RESULT] ✅ Torso lean at kick impact: {lean_at_impact:.1f}° "
                  f"(within {LEAN_THRESHOLD_DEG}° threshold — good form).")
        else:
            print("[RESULT] Could not determine torso lean at impact (pose not detected).")

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse football kicking technique using MediaPipe Pose.",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="input_kick.mp4",
        help="Path to input video file (default: input_kick.mp4).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="analyzed_kick.mp4",
        help="Path for the annotated output video (default: analyzed_kick.mp4).",
    )
    args = parser.parse_args()

    return analyze_video(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())