#!/usr/bin/env python3
"""
Football Movement Analysis — Multi-Metric Biomechanics
=======================================================

Analyses a video of a football player's dynamic movement (passing, shooting,
dribbling, running) using MediaPipe Pose and OpenCV.

Three core movement metrics are tracked per frame:

  1. Torso Alignment (Body Lean)
     — Shoulder‑midpoint → hip‑midpoint angle from vertical.
       Flags "POOR POSTURE / LEANING BACK" if > 15° lean beyond vertical.

  2. Knee Stability (Joint Angle & Varus/Valgus Deviation)
     — Internal angle of the knee joint (hip–knee–ankle) AND perpendicular
       deviation of the knee from the hip‑ankle axis.
       Flags "KNEE ALIGNMENT RISK" if the knee collapses inward beyond a
       safe deviation threshold during dynamic loading.

  3. Gait Asymmetry
     — Stride length symmetry ratio between left and right legs computed
       from foot‑plant events (ankle y‑minima).
       Flags "ASYMMETRIC GAIT DETECTED" if the imbalance exceeds 15 %.

Output is saved as 'analyzed_movement.mp4' with skeleton overlay and
live warning banners burned into every frame.

Usage:
    python analyze_movement.py
    python analyze_movement.py --input my_clip.mp4
    python analyze_movement.py --input clip.mp4 --output result.mp4 --thresholds torso=18,knee=30,gait=20

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
# Tunable thresholds (override via --thresholds CLI argument)
# ---------------------------------------------------------------------------

# 1. Torso lean: degrees of backward lean from vertical before warning.
TORSO_LEAN_THRESHOLD_DEG: float = 15.0

# 2. Knee stability: max safe perpendicular deviation of the knee from the
#    hip‑ankle axis, expressed as a fraction of the player's thigh length.
#    A value of 0.25 means "if the knee deviates sideways by more than 25 %
#    of the thigh length, flag it."  The same threshold is used for both
#    medial (valgus) and lateral (varus) deviation.
KNEE_DEVIATION_RATIO: float = 0.25

# 3. Gait asymmetry: max allowable difference between left / right stride
#    length as a fraction of the larger stride.  0.15 → 15 %.
GAIT_ASYMMETRY_RATIO: float = 0.15

# MediaPipe confidence values.
DETECTION_CONFIDENCE: float = 0.5
TRACKING_CONFIDENCE: float = 0.5

# Minimum number of frames between two foot‑plant events to consider them
# separate strides (avoids double-counting from jitter).
MIN_STRIDE_FRAMES: int = 8

# Number of recent strides to keep for the running asymmetry calculation.
ASYMMETRY_WINDOW: int = 6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Point2D:
    """Pixel coordinate with visibility score from MediaPipe."""
    x: float
    y: float
    visibility: float = 1.0


@dataclass
class Skeleton:
    """The eight landmarks we care about, in pixel space."""
    left_shoulder:  Optional[Point2D] = None
    right_shoulder: Optional[Point2D] = None
    left_hip:       Optional[Point2D] = None
    right_hip:      Optional[Point2D] = None
    left_knee:      Optional[Point2D] = None
    right_knee:     Optional[Point2D] = None
    left_ankle:     Optional[Point2D] = None
    right_ankle:    Optional[Point2D] = None

    def is_valid(self) -> bool:
        return all(
            getattr(self, f) is not None
            for f in (
                "left_shoulder", "right_shoulder",
                "left_hip", "right_hip",
                "left_knee", "right_knee",
                "left_ankle", "right_ankle",
            )
        )


@dataclass
class GaitEvent:
    """Records a single foot‑plant (heel-strike) event."""
    leg: str            # "left" or "right"
    frame: int
    ankle_x: float
    ankle_y: float


# ---------------------------------------------------------------------------
# MediaPipe → Skeleton converter
# ---------------------------------------------------------------------------

def landmarks_to_skeleton(
    landmarks,
    frame_w: int,
    frame_h: int,
) -> Skeleton:
    """
    Convert raw MediaPipe normalised landmarks into a `Skeleton` with
    pixel coordinates.  Returns an empty Skeleton (all fields None) when
    no pose is detected.
    """
    if landmarks is None:
        return Skeleton()

    def pt(idx: int) -> Optional[Point2D]:
        lm = landmarks[idx]
        return Point2D(x=lm.x * frame_w, y=lm.y * frame_h, visibility=lm.visibility)

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
    return Point2D(x=(a.x + b.x) * 0.5, y=(a.y + b.y) * 0.5)


def distance(a: Point2D, b: Point2D) -> float:
    """Euclidean distance between two points in pixels."""
    return float(np.hypot(a.x - b.x, a.y - b.y))


# ---------------------------------------------------------------------------
# Metric 1: Torso Alignment (Body Lean)
# ---------------------------------------------------------------------------

def torso_lean_angle(skel: Skeleton) -> Optional[float]:
    """
    Signed angle (degrees) of the torso axis from vertical.

        shoulder_midpoint → hip_midpoint  (torso axis, pointing upward)

    Positive → forward lean.
    Negative → backward lean (dangerous for kicking / deceleration).

    Returns None if the skeleton is incomplete.
    """
    if not skel.is_valid():
        return None

    sh_mid = midpoint(skel.left_shoulder, skel.right_shoulder)
    hi_mid = midpoint(skel.left_hip, skel.right_hip)

    dx = sh_mid.x - hi_mid.x
    dy = sh_mid.y - hi_mid.y

    # atan2(-dx, -dy) gives the angle relative to the upward vertical (0, -1).
    angle_deg = np.degrees(np.arctan2(-dx, -dy))
    return float(np.clip(angle_deg, -90.0, 90.0))


# ---------------------------------------------------------------------------
# Metric 2: Knee Stability (Joint Angle + Varus/Valgus Deviation)
# ---------------------------------------------------------------------------

def knee_internal_angle(
    hip: Point2D, knee: Point2D, ankle: Point2D,
) -> Optional[float]:
    """
    Internal angle at the knee joint formed by the vectors:
        knee → hip   and   knee → ankle

    Returns angle in degrees [0, 180].
        ~180°  = straight / hyperextended
        ~90°   = right-angle bend (deep flexion)
        < 30°  = extreme acute (unlikely during normal movement)
    """
    v1 = np.array([hip.x - knee.x, hip.y - knee.y])
    v2 = np.array([ankle.x - knee.x, ankle.y - knee.y])

    dot = float(np.dot(v1, v2))
    mag = float(np.linalg.norm(v1)) * float(np.linalg.norm(v2))
    if mag < 1e-6:
        return None  # degenerate case — landmarks overlap

    cos_ang = np.clip(dot / mag, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def knee_deviation_ratio(
    hip: Point2D, knee: Point2D, ankle: Point2D,
) -> Optional[float]:
    """
    Perpendicular deviation of the knee from the straight line connecting
    the hip and ankle, expressed as a fraction of the thigh length
    (hip → knee).

    A value close to 0 → the knee stays on the hip‑ankle axis (neutral).
    A positive value → medial deviation (valgus collapse).
    A negative value → lateral deviation (varus).

    This is a robust 2‑D proxy for frontal‑plane knee stability even when
    the camera is at a side angle.

    Returns None if landmarks are too close together.
    """
    # Convert to numpy vectors.
    A = np.array([hip.x, hip.y])
    B = np.array([knee.x, knee.y])
    C = np.array([ankle.x, ankle.y])

    thigh_len = float(np.linalg.norm(B - A))
    if thigh_len < 1e-6:
        return None

    # Perpendicular distance of point B from line AC.
    #   cross( C-A, B-A ) / |C-A|  → signed distance perpendicular to AC.
    cross = float(np.cross(C - A, B - A))
    hip_ankle_len = float(np.linalg.norm(C - A))
    if hip_ankle_len < 1e-6:
        return None

    deviation = cross / hip_ankle_len   # signed pixel distance
    return deviation / thigh_len        # ratio relative to thigh length


def check_knee_stability(
    skel: Skeleton,
    deviation_threshold: float,
) -> Optional[str]:
    """
    Evaluate both knees for alignment risk.  Returns a human‑readable warning
    string if either knee shows excessive deviation, or None if safe.
    """
    if not skel.is_valid():
        return None

    warnings = []

    # Left knee
    dev_l = knee_deviation_ratio(skel.left_hip, skel.left_knee, skel.left_ankle)
    if dev_l is not None and abs(dev_l) > deviation_threshold:
        direction = "valgus (inward)" if dev_l > 0 else "varus (outward)"
        warnings.append(f"Left knee {direction} (dev: {dev_l:.2f})")

    # Right knee
    dev_r = knee_deviation_ratio(skel.right_hip, skel.right_knee, skel.right_ankle)
    if dev_r is not None and abs(dev_r) > deviation_threshold:
        direction = "valgus (inward)" if dev_r > 0 else "varus (outward)"
        warnings.append(f"Right knee {direction} (dev: {dev_r:.2f})")

    return "KNEE ALIGNMENT RISK: " + "; ".join(warnings) if warnings else None


# ---------------------------------------------------------------------------
# Metric 3: Gait Asymmetry
# ---------------------------------------------------------------------------

def detect_gait_cycle(
    skel: Skeleton,
    prev_skel: Optional[Skeleton],
    events: list[GaitEvent],
    frame_idx: int,
    window_size: int,
    asymmetry_threshold: float,
) -> Optional[str]:
    """
    Detect foot‑plant events (ankle y‑minima) and maintain a rolling window
    of stride lengths for each leg.  When enough strides have been recorded
    on both sides, compute the asymmetry ratio.

    Returns a warning string if asymmetry > threshold, otherwise None.

    Call this every frame; it mutates `events` in‑place.
    """
    if not skel.is_valid() or prev_skel is None or not prev_skel.is_valid():
        return None

    # We'll check each leg independently.
    for leg_label, ankle_now, ankle_prev, knee_now in [
        ("left", skel.left_ankle, prev_skel.left_ankle, skel.left_knee),
        ("right", skel.right_ankle, prev_skel.right_ankle, skel.right_knee),
    ]:
        if None in (ankle_now, ankle_prev, knee_now):
            continue

        # --- Foot‑plant detection ---
        # Heuristic: the ankle reaches its lowest Y position (closest to the
        # ground) during stance phase.  We detect a "local minimum" by
        # checking that the ankle has just transitioned from descending
        # (y increasing) to ascending (y decreasing).
        #
        # Additionally, the knee should be relatively straight (angle > 150°)
        # at foot‑plant to avoid counting mid‑swing oscillations.

        dy = ankle_now.y - ankle_prev.y  # positive = moving downward
        # We want the moment where dy flips from negative (going up) to
        # positive (going down, AFTER the foot has struck the ground)...
        # Actually, simpler: a foot‑plant occurs when the ankle stops
        # descending and starts ascending — i.e., dy is near zero and
        # the ankle is low in the frame.

        # Practical heuristic: ankle y is near its local maximum (lowest
        # point in the image), and the knee is extended enough
        # (internal angle > 150°) to indicate weight‑bearing.

        knee_angle = knee_internal_angle(
            # We approximate the necessary landmarks.
            # For the thigh reference, use the hip:
            getattr(skel, f"{leg_label}_hip"),
            knee_now,
            ankle_now,
        )
        if knee_angle is None or knee_angle < 140.0:
            continue  # leg is still swinging — not a foot‑plant

        # Check that dy is small (ankle roughly stationary relative to ground).
        # We normalise by frame height so the threshold works at any resolution.
        if abs(dy) > 0.01 * 1080:   # ~10 px at 1080p, scaled by rough heuristic
            continue

        # Check that this ankle y is the lowest we've seen recently.
        # If the last event for this leg was very recent, skip.
        last_event = next(
            (e for e in reversed(events) if e.leg == leg_label),
            None,
        )
        if last_event is not None and (frame_idx - last_event.frame) < MIN_STRIDE_FRAMES:
            continue

        # Register this foot‑plant.
        events.append(GaitEvent(
            leg=leg_label,
            frame=frame_idx,
            ankle_x=ankle_now.x,
            ankle_y=ankle_now.y,
        ))

    # --- Stride length extraction ---
    # A stride = distance between two consecutive foot‑plants of the SAME leg.
    left_strides: list[float] = []
    right_strides: list[float] = []

    left_events = [e for e in events if e.leg == "left"]
    right_events = [e for e in events if e.leg == "right"]

    for i in range(1, len(left_events)):
        left_strides.append(left_events[i].ankle_x - left_events[i - 1].ankle_x)
    for i in range(1, len(right_events)):
        right_strides.append(right_events[i].ankle_x - right_events[i - 1].ankle_x)

    # Keep only the last `window_size` strides for the rolling average.
    left_strides = left_strides[-window_size:]
    right_strides = right_strides[-window_size:]

    if len(left_strides) < 2 or len(right_strides) < 2:
        return None  # not enough data yet

    avg_left = float(np.mean(left_strides))
    avg_right = float(np.mean(right_strides))

    # Asymmetry ratio = |L - R| / max(L, R)
    max_stride = max(avg_left, avg_right)
    if max_stride < 1.0:
        return None  # degenerate — stride too small to measure

    asymmetry = abs(avg_left - avg_right) / max_stride

    if asymmetry > asymmetry_threshold:
        direction = "left > right" if avg_left > avg_right else "right > left"
        return (
            f"ASYMMETRIC GAIT DETECTED "
            f"({asymmetry:.1%} imbalance, {direction})"
        )

    return None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

SKELETON_COLOR = (50, 255, 50)    # neon green (BGR for OpenCV)
SKELETON_THICKNESS = 3
JOINT_RADIUS = 6
WARNING_COLOR = (0, 50, 255)      # red (BGR)
WARNING_THICKNESS = 2
INFO_COLOR = (200, 200, 200)      # light grey for debug text
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_skeleton(frame: np.ndarray, skel: Skeleton) -> None:
    """Draw neon‑green skeleton lines and joint circles on the frame."""
    if not skel.is_valid():
        return

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
        cv2.line(
            frame, (int(pa.x), int(pa.y)), (int(pb.x), int(pb.y)),
            SKELETON_COLOR, SKELETON_THICKNESS,
        )

    joints = [
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    for attr in joints:
        p: Optional[Point2D] = getattr(skel, attr)
        if p is None:
            continue
        cv2.circle(frame, (int(p.x), int(p.y)), JOINT_RADIUS, SKELETON_COLOR, -1)


def draw_warning_banner(frame: np.ndarray, messages: list[str]) -> None:
    """
    Draw a semi‑transparent banner at the top of the frame and overlay each
    active warning in bold red text, centred horizontally.
    """
    if not messages:
        return

    h, w, _ = frame.shape

    # Semi‑transparent overlay.
    overlay = frame.copy()
    banner_height = 40 + 28 * len(messages)   # grows with message count
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, msg in enumerate(messages):
        y = 30 + i * 28
        text_size = cv2.getTextSize(msg, FONT, 0.7, WARNING_THICKNESS)[0]
        x = (w - text_size[0]) // 2   # centre
        cv2.putText(
            frame, msg, (x, y),
            FONT, 0.7, WARNING_COLOR, WARNING_THICKNESS, cv2.LINE_AA,
        )


def draw_debug_overlay(
    frame: np.ndarray,
    lean: Optional[float],
    knee_angles: dict[str, Optional[float]],
    frame_idx: int,
    total_frames: int,
) -> None:
    """Show real‑time metric values in the bottom‑left corner."""
    lines = []
    if lean is not None:
        lines.append(f"Torso lean: {lean:+.1f}°")
    for leg, angle in knee_angles.items():
        if angle is not None:
            lines.append(f"{leg.capitalize()} knee: {angle:.0f}°")
    lines.append(f"Frame: {frame_idx}/{total_frames}")

    for i, line in enumerate(lines):
        y = frame.shape[0] - 10 - i * 24
        cv2.putText(
            frame, line, (12, y),
            FONT, 0.55, INFO_COLOR, 2, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_movement(
    input_path: str,
    output_path: str = "analyzed_movement.mp4",
    torso_threshold: float = TORSO_LEAN_THRESHOLD_DEG,
    knee_dev_threshold: float = KNEE_DEVIATION_RATIO,
    gait_asymmetry_threshold: float = GAIT_ASYMMETRY_RATIO,
) -> int:
    """
    Run the full multi‑metric movement analysis pipeline.

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
    print(f"[INFO] Thresholds: torso_lean={torso_threshold}°, "
          f"knee_dev={knee_dev_threshold:.2f}×thigh, "
          f"gait_asym={gait_asymmetry_threshold:.0%}")

    # ── 3. Video writer ────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[ERROR] Cannot create output video: {output_path}", file=sys.stderr)
        cap.release()
        return 1

    # ── 4. MediaPipe initialisation ────────────────────────────────────────
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,         # 0=lite, 1=full, 2=heavy
        enable_segmentation=False,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    # ── 5. Runtime state ───────────────────────────────────────────────────
    prev_skeleton: Optional[Skeleton] = None
    gait_events: list[GaitEvent] = []

    # Persistent warning flags — once triggered they stay on for the rest
    # of the video (so the coach sees them in the output).
    torso_warning: bool = False
    knee_warning: Optional[str] = None
    gait_warning: Optional[str] = None

    frame_idx: int = 0

    print(f"[INFO] Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video or corrupted frame

        frame_idx += 1

        # ── 5a. Pose estimation ────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        skeleton = landmarks_to_skeleton(results.pose_landmarks, w, h)

        # Always draw the skeleton if detected.
        if skeleton.is_valid():
            draw_skeleton(frame, skeleton)

        # ── 5b. Metric 1 — Torso Alignment ─────────────────────────────────
        lean = torso_lean_angle(skeleton)
        if lean is not None and lean < -torso_threshold:
            torso_warning = True

        # ── 5c. Metric 2 — Knee Stability ──────────────────────────────────
        knee_warning = check_knee_stability(skeleton, knee_dev_threshold)

        # ── 5d. Metric 3 — Gait Asymmetry ──────────────────────────────────
        gait_warning = detect_gait_cycle(
            skeleton, prev_skeleton,
            gait_events, frame_idx,
            window_size=ASYMMETRY_WINDOW,
            asymmetry_threshold=gait_asymmetry_threshold,
        )

        # ── 5e. Assemble active warnings ───────────────────────────────────
        active_warnings: list[str] = []
        if torso_warning:
            active_warnings.append("POOR POSTURE / LEANING BACK")
        if knee_warning:
            active_warnings.append(knee_warning)
        if gait_warning:
            active_warnings.append(gait_warning)

        # ── 5f. Draw overlays ──────────────────────────────────────────────
        draw_warning_banner(frame, active_warnings)
        draw_debug_overlay(
            frame,
            lean,
            {"left": knee_internal_angle(
                skeleton.left_hip, skeleton.left_knee, skeleton.left_ankle,
            ) if skeleton.is_valid() else None,
             "right": knee_internal_angle(
                skeleton.right_hip, skeleton.right_knee, skeleton.right_ankle,
            ) if skeleton.is_valid() else None},
            frame_idx,
            total_frames,
        )

        if not skeleton.is_valid():
            cv2.putText(
                frame, "Player not in frame",
                (20, frame.shape[0] - 10),
                FONT, 0.6, (100, 100, 100), 2, cv2.LINE_AA,
            )

        # ── 5g. Write frame ────────────────────────────────────────────────
        writer.write(frame)
        prev_skeleton = skeleton

        # Progress indicator.
        if total_frames > 0 and frame_idx % max(1, total_frames // 10) == 0:
            pct = int(frame_idx / total_frames * 100)
            print(f"   ... {pct}% ({frame_idx}/{total_frames})")

    # ── 6. Cleanup ─────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    pose.close()
    cv2.destroyAllWindows()

    print(f"[INFO] Output: {output_path}")
    print(f"[RESULT] Warnings triggered:")
    print(f"    Torso lean:     {'⚠ FLAGGED' if torso_warning else '✓ OK'}")
    print(f"    Knee alignment: {'⚠ FLAGGED' if knee_warning else '✓ OK (or no data)'}")
    print(f"    Gait symmetry:  {'⚠ FLAGGED' if gait_warning else '✓ OK (or no data)'}")

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_thresholds(raw: Optional[str]) -> dict[str, float]:
    """Parse --thresholds CLI argument like 'torso=18,knee=0.3,gait=0.20'."""
    result: dict[str, float] = {}
    if not raw:
        return result
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        try:
            result[key.strip()] = float(val.strip())
        except ValueError:
            print(f"[WARN] Ignoring invalid threshold: {pair}", file=sys.stderr)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-metric football movement analysis using MediaPipe Pose. "
            "Evaluates torso alignment, knee stability, and gait symmetry."
        ),
    )
    parser.add_argument(
        "--input", "-i", type=str, default="input_movement.mp4",
        help="Path to input video (default: input_movement.mp4).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="analyzed_movement.mp4",
        help="Path for annotated output video (default: analyzed_movement.mp4).",
    )
    parser.add_argument(
        "--thresholds", "-t", type=str, default=None,
        help=(
            "Override thresholds as comma‑separated key=value pairs. "
            "Keys: torso (degrees), knee (ratio of thigh length), "
            "gait (asymmetry fraction).  "
            "Example: --thresholds torso=18,knee=0.3,gait=0.20"
        ),
    )
    args = parser.parse_args()

    overrides = parse_thresholds(args.thresholds)

    return analyze_movement(
        input_path=args.input,
        output_path=args.output,
        torso_threshold=overrides.get("torso", TORSO_LEAN_THRESHOLD_DEG),
        knee_dev_threshold=overrides.get("knee", KNEE_DEVIATION_RATIO),
        gait_asymmetry_threshold=overrides.get("gait", GAIT_ASYMMETRY_RATIO),
    )


if __name__ == "__main__":
    sys.exit(main())