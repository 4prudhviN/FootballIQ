#!/usr/bin/env python3
"""
Player Model
============
Reusable object representing a tracked player across video frames.

Holds identity, position history, and computed movement state.
Used by the pipeline to maintain per-player context without
recalculating state from scratch every frame.

Responsibilities:
  - Store player identity (id, position role, foot preference)
  - Track body position history across frames
  - Compute movement speed, direction, and acceleration
  - Maintain a rolling window of pose frames for analysis
  - Support multiple players (one PlayerModel instance per player)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

import numpy as np

from models.pose_model import PoseFrame, Landmark


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PlayerRole(str, Enum):
    GOALKEEPER = "goalkeeper"
    DEFENDER   = "defender"
    MIDFIELDER = "midfielder"
    FORWARD    = "forward"
    UNKNOWN    = "unknown"


class FootPreference(str, Enum):
    RIGHT    = "right"
    LEFT     = "left"
    BALANCED = "balanced"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PlayerPosition:
    """Player body position at a single point in time."""
    frame_index:  int
    timestamp_s:  float
    hip_x:        float          # normalised [0, 1]
    hip_y:        float          # normalised [0, 1]


@dataclass
class MovementState:
    """
    Current movement state derived from position history.
    Recomputed on every update.
    """
    speed_px_f:      float        # pixels/frame (normalised units/frame)
    direction_deg:   float        # degrees from +x axis
    acceleration:    float        # change in speed from previous frame
    is_sprinting:    bool         # True if speed > sprint threshold
    is_stationary:   bool         # True if speed < stationary threshold


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_SPRINT_THRESHOLD      = 0.020   # normalised units/frame (≈ fast run)
_STATIONARY_THRESHOLD  = 0.002   # normalised units/frame
_HISTORY_SIZE          = 30      # frames of position history


# ---------------------------------------------------------------------------
# Player Model
# ---------------------------------------------------------------------------

class PlayerModel:
    """
    Reusable object representing one tracked player.

    Create one instance per player detected in the video.
    Call update() each frame to maintain state.

    Parameters
    ----------
    player_id      : str  Unique identifier for this player.
    role           : PlayerRole
    foot_preference: FootPreference
    history_size   : int  Number of frames to keep in position history.

    Usage::

        player = PlayerModel(player_id="P1", role=PlayerRole.FORWARD)

        for frame_pose in pose_frames:
            player.update(frame_pose)

        print(player.movement_state.speed_px_f)
        print(player.avg_torso_lean())
    """

    def __init__(
        self,
        player_id:       str            = "P1",
        role:            PlayerRole     = PlayerRole.UNKNOWN,
        foot_preference: FootPreference = FootPreference.RIGHT,
        history_size:    int            = _HISTORY_SIZE,
    ) -> None:
        self.player_id       = player_id
        self.role            = role
        self.foot_preference = foot_preference

        self._pose_history:     Deque[PoseFrame]     = deque(maxlen=history_size)
        self._position_history: Deque[PlayerPosition] = deque(maxlen=history_size)
        self._movement_state:   Optional[MovementState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_pose: PoseFrame) -> None:
        """
        Ingest a new PoseFrame and update all internal state.

        Parameters
        ----------
        frame_pose : PoseFrame — from PoseModel.process_frame()
        """
        self._pose_history.append(frame_pose)

        if not frame_pose.detected:
            return

        # Extract hip midpoint as the canonical position.
        hip_x, hip_y = self._hip_midpoint(frame_pose)
        pos = PlayerPosition(
            frame_index = frame_pose.frame_index,
            timestamp_s = frame_pose.timestamp_s,
            hip_x       = hip_x,
            hip_y       = hip_y,
        )
        self._position_history.append(pos)

        # Recompute movement state.
        self._movement_state = self._compute_movement()

    @property
    def movement_state(self) -> Optional[MovementState]:
        """Current movement state (None if fewer than 2 frames recorded)."""
        return self._movement_state

    @property
    def pose_history(self) -> List[PoseFrame]:
        return list(self._pose_history)

    @property
    def position_history(self) -> List[PlayerPosition]:
        return list(self._position_history)

    @property
    def latest_pose(self) -> Optional[PoseFrame]:
        return self._pose_history[-1] if self._pose_history else None

    def avg_torso_lean(self) -> Optional[float]:
        """
        Average absolute torso lean angle across all recorded frames.
        Returns None if no pose data is available.
        """
        leans = [
            abs(fp.torso_lean_deg)
            for fp in self._pose_history
            if fp.detected and fp.torso_lean_deg is not None
        ]
        return float(np.mean(leans)) if leans else None

    def avg_knee_deviation(self) -> Optional[float]:
        """
        Average knee deviation ratio (max of left/right) across frames.
        """
        devs = [
            max(abs(fp.left_knee_dev or 0), abs(fp.right_knee_dev or 0))
            for fp in self._pose_history
            if fp.detected
        ]
        return float(np.mean(devs)) if devs else None

    def gait_asymmetry(self) -> Optional[float]:
        """
        Gait asymmetry ratio [0, 1] from ankle Y-position variance.
        0 = perfectly symmetric, 1 = complete imbalance.
        """
        left_y  = [
            fp.landmarks["left_ankle"].y
            for fp in self._pose_history
            if fp.detected and "left_ankle" in fp.landmarks
        ]
        right_y = [
            fp.landmarks["right_ankle"].y
            for fp in self._pose_history
            if fp.detected and "right_ankle" in fp.landmarks
        ]
        if len(left_y) < 3 or len(right_y) < 3:
            return None
        avg_l = float(np.mean(left_y))
        avg_r = float(np.mean(right_y))
        denom = max(avg_l, avg_r)
        return abs(avg_l - avg_r) / denom if denom > 0 else 0.0

    def total_distance(self) -> float:
        """
        Total distance covered in normalised units
        (sum of frame-to-frame hip displacements).
        """
        positions = list(self._position_history)
        total = 0.0
        for i in range(1, len(positions)):
            dx = positions[i].hip_x - positions[i - 1].hip_x
            dy = positions[i].hip_y - positions[i - 1].hip_y
            total += math.hypot(dx, dy)
        return total

    def reset(self) -> None:
        """Clear all history and state."""
        self._pose_history.clear()
        self._position_history.clear()
        self._movement_state = None

    def to_dict(self) -> dict:
        """Serialise current player state to a plain dict."""
        ms = self._movement_state
        return {
            "player_id":       self.player_id,
            "role":            self.role.value,
            "foot_preference": self.foot_preference.value,
            "frames_recorded": len(self._pose_history),
            "avg_torso_lean":  round(self.avg_torso_lean() or 0, 2),
            "avg_knee_dev":    round(self.avg_knee_deviation() or 0, 3),
            "gait_asymmetry":  round(self.gait_asymmetry() or 0, 3),
            "total_distance":  round(self.total_distance(), 4),
            "movement": {
                "speed_px_f":    round(ms.speed_px_f, 3)    if ms else 0,
                "direction_deg": round(ms.direction_deg, 1) if ms else 0,
                "acceleration":  round(ms.acceleration, 3)  if ms else 0,
                "is_sprinting":  ms.is_sprinting             if ms else False,
                "is_stationary": ms.is_stationary            if ms else True,
            },
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _hip_midpoint(fp: PoseFrame) -> Tuple[float, float]:
        """Return the hip midpoint (x, y) or (0.5, 0.5) if not available."""
        lh = fp.landmarks.get("left_hip")
        rh = fp.landmarks.get("right_hip")
        if lh and rh:
            return (lh.x + rh.x) / 2, (lh.y + rh.y) / 2
        if lh:
            return lh.x, lh.y
        if rh:
            return rh.x, rh.y
        return 0.5, 0.5

    def _compute_movement(self) -> Optional[MovementState]:
        """Compute movement state from the last two position samples."""
        positions = list(self._position_history)
        if len(positions) < 2:
            return None

        curr = positions[-1]
        prev = positions[-2]

        dt = max(1, curr.frame_index - prev.frame_index)
        dx = (curr.hip_x - prev.hip_x) / dt
        dy = (curr.hip_y - prev.hip_y) / dt

        speed     = math.hypot(dx, dy)
        direction = math.degrees(math.atan2(dy, dx))

        # Acceleration vs previous state.
        prev_speed = self._movement_state.speed_px_f if self._movement_state else speed
        accel      = speed - prev_speed

        return MovementState(
            speed_px_f    = speed,
            direction_deg = direction,
            acceleration  = accel,
            is_sprinting  = speed > _SPRINT_THRESHOLD,
            is_stationary = speed < _STATIONARY_THRESHOLD,
        )
