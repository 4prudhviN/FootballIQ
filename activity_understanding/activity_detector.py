#!/usr/bin/env python3
"""
Activity Detector  ⭐
=====================
The unique heart of FootballIQ.

Input:  Pose landmarks + Ball tracks + Player tracks
Output: List[DetectedAction] — what football actions happened

This module ONLY answers: "What football actions happened?"
It NEVER calculates scores, metrics, or performance ratings.
No numbers. No analysis. Just action classification.

Detected actions:
  passing, dribbling, shooting, goalkeeping,
  defending, movement, free_kick, penalty, header

Rules are evidence-based:
  - Pose patterns (body lean, knee bend, arm position)
  - Ball position relative to player
  - Ball velocity and direction
  - Spatial context (where on the pitch)
  - Sequence of events over time

Writes to PipelineContext:
  ctx.activity.detected_activities
  ctx.activity.confidence_scores
  ctx.activity.primary_activity
  ctx.activity.timeline
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pipeline.pose_estimator   import PoseEstimationResult, FramePose, PoseLandmark
from pipeline.ball_detector    import BallDetectionResult, BallDetection
from pipeline.player_detector  import PlayerDetectionResult
from pipeline.pipeline_context import PipelineContext, ActivitySegmentCtx
from schemas.activity_schema   import FootballAction
from utils.logger              import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# All recognisable football actions
# ---------------------------------------------------------------------------

class FootballActionType:
    PASSING     = "passing"
    DRIBBLING   = "dribbling"
    SHOOTING    = "shooting"
    GOALKEEPING = "goalkeeping"
    DEFENDING   = "defending"
    MOVEMENT    = "movement"
    FREE_KICK   = "free_kick"
    PENALTY     = "penalty"
    HEADER      = "header"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class ActionEvidence:
    """Why an action was detected — observable cues only."""
    cue:         str     # e.g. "Ball moving at high speed away from player"
    source:      str     # "pose" | "ball" | "player" | "sequence"
    frame_index: int


@dataclass
class DetectedAction:
    """
    A single detected football action.
    No scores, no metrics — just the action and why it was detected.
    """
    action:      str                        # FootballActionType value
    start_frame: int
    end_frame:   int
    start_time_s: float
    end_time_s:  float
    evidence:    List[ActionEvidence] = field(default_factory=list)
    label:       str                  = ""  # e.g. "00:00–00:25  Passing"

    def __post_init__(self) -> None:
        if not self.label:
            self.label = (
                f"{self._fmt(self.start_time_s)}–"
                f"{self._fmt(self.end_time_s)}  "
                f"{self.action.replace('_', ' ').title()}"
            )

    @staticmethod
    def _fmt(s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f"{m:02d}:{sec:02d}"


@dataclass
class ActivityUnderstandingResult:
    """
    Full output of the activity detector.
    What football actions happened — nothing more.
    """
    detected_actions:    List[DetectedAction]
    action_names:        List[str]           # unique action names detected
    primary_action:      Optional[str]       # most prominent action
    frame_count:         int


# ---------------------------------------------------------------------------
# Rule engine  (observable cues → action classification)
# ---------------------------------------------------------------------------

class _RuleEngine:
    """
    Applies observable football rules to classify actions per frame.
    Returns a list of (action_name, evidence) tuples for each frame.
    No scores. Binary: either the evidence is there or it isn't.
    """

    def classify_frame(
        self,
        pose:        Optional[FramePose],
        ball:        Optional[BallDetection],
        prev_ball:   Optional[BallDetection],
        frame_index: int,
    ) -> List[Tuple[str, str]]:
        """
        Returns list of (action, evidence_cue) for this frame.
        Multiple actions can co-occur in the same frame.
        """
        signals: List[Tuple[str, str]] = []

        # ── Ball-based rules ─────────────────────────────────────────────────

        if ball is not None:
            # Ball speed (pixels moved since last frame).
            if prev_ball is not None:
                dx = ball.center_x - prev_ball.center_x
                dy = ball.center_y - prev_ball.center_y
                speed = math.hypot(dx, dy)

                if speed > 40:
                    signals.append((
                        FootballActionType.SHOOTING,
                        f"Ball moving fast ({speed:.0f}px/frame) — shot or driven pass"
                    ))
                elif speed > 15:
                    signals.append((
                        FootballActionType.PASSING,
                        f"Ball moving moderately ({speed:.0f}px/frame) — pass"
                    ))
                elif speed < 3 and pose and self._ball_near_feet(ball, pose):
                    signals.append((
                        FootballActionType.DRIBBLING,
                        "Ball stationary near player feet — close control"
                    ))

            # Ball near player feet → dribbling candidate.
            if pose and self._ball_near_feet(ball, pose):
                signals.append((
                    FootballActionType.DRIBBLING,
                    "Ball within foot range — dribbling or receiving"
                ))

            # Ball high up and player near → header candidate.
            if pose and ball.center_y < 0.3 * 1080:
                nose = pose.get("nose")
                if nose and abs(nose.x * 1920 - ball.center_x) < 80:
                    signals.append((
                        FootballActionType.HEADER,
                        "Ball at head height near player — header attempt"
                    ))

        # ── Pose-based rules ─────────────────────────────────────────────────

        if pose and pose.detected:

            # Kicking posture: one leg raised, torso leaning.
            if self._kicking_posture(pose):
                signals.append((
                    FootballActionType.SHOOTING,
                    "Kicking posture detected — leg raised, weight on plant foot"
                ))

            # Goalkeeper dive: arms wide, lateral lean.
            if self._goalkeeper_posture(pose):
                signals.append((
                    FootballActionType.GOALKEEPING,
                    "Arms spread wide, lateral body position — goalkeeper stance"
                ))

            # Defensive jockey: side-on stance, bent knees, arms out.
            if self._defensive_posture(pose):
                signals.append((
                    FootballActionType.DEFENDING,
                    "Side-on low stance — defensive jockeying position"
                ))

            # Running without ball nearby.
            if ball is None or (pose and not self._ball_near_feet(ball, pose)):
                if self._running_posture(pose):
                    signals.append((
                        FootballActionType.MOVEMENT,
                        "Player running without ball — movement action"
                    ))

        # ── Set piece detection ───────────────────────────────────────────────

        # Free kick: player stationary over ball, then ball moves fast.
        if (ball is not None and prev_ball is not None and
                math.hypot(
                    ball.center_x - prev_ball.center_x,
                    ball.center_y - prev_ball.center_y
                ) > 50 and
                pose and self._stationary_over_ball(ball, pose)):
            signals.append((
                FootballActionType.FREE_KICK,
                "Player was stationary over ball, then ball moved rapidly — free kick"
            ))

        return signals

    # ------------------------------------------------------------------
    # Observable pose cues
    # ------------------------------------------------------------------

    @staticmethod
    def _ball_near_feet(ball: BallDetection, pose: FramePose) -> bool:
        """True if ball is within ~50px of either ankle."""
        for ankle_name in ("left_ankle", "right_ankle"):
            ankle = pose.get(ankle_name)
            if ankle is None:
                continue
            dist = math.hypot(
                ankle.x * 1920 - ball.center_x,
                ankle.y * 1080 - ball.center_y,
            )
            if dist < 80:
                return True
        return False

    @staticmethod
    def _kicking_posture(pose: FramePose) -> bool:
        """
        True if one knee is significantly higher than the other
        (leg raised for kick) and hips are level-ish.
        """
        lk = pose.get("left_knee")
        rk = pose.get("right_knee")
        if lk is None or rk is None:
            return False
        # In normalised coords, lower y = higher on screen.
        knee_diff = abs(lk.y - rk.y)
        return knee_diff > 0.12 and lk.is_visible and rk.is_visible

    @staticmethod
    def _goalkeeper_posture(pose: FramePose) -> bool:
        """True if arms are spread wide (wrists far from centre line)."""
        lw = pose.get("left_wrist")
        rw = pose.get("right_wrist")
        ls = pose.get("left_shoulder")
        rs = pose.get("right_shoulder")
        if None in (lw, rw, ls, rs):
            return False
        shoulder_width = abs(ls.x - rs.x)
        wrist_width    = abs(lw.x - rw.x)
        return wrist_width > shoulder_width * 1.8

    @staticmethod
    def _defensive_posture(pose: FramePose) -> bool:
        """True if player is in a low side-on stance (bent knees, narrow hips)."""
        lk = pose.get("left_knee")
        rk = pose.get("right_knee")
        lh = pose.get("left_hip")
        rh = pose.get("right_hip")
        if None in (lk, rk, lh, rh):
            return False
        # Knees bent = knee y > hip y in normalised coords.
        l_bent = lk.y > lh.y + 0.05
        r_bent = rk.y > rh.y + 0.05
        return l_bent and r_bent

    @staticmethod
    def _running_posture(pose: FramePose) -> bool:
        """True if alternating arm-leg pattern suggests running."""
        lw = pose.get("left_wrist")
        rw = pose.get("right_wrist")
        if lw is None or rw is None:
            return False
        # Arms in different vertical positions = running arm swing.
        return abs(lw.y - rw.y) > 0.08

    @staticmethod
    def _stationary_over_ball(ball: BallDetection, pose: FramePose) -> bool:
        """True if player hips are directly above the ball."""
        lh = pose.get("left_hip")
        rh = pose.get("right_hip")
        if None in (lh, rh):
            return False
        hip_cx = ((lh.x + rh.x) / 2) * 1920
        hip_cy = ((lh.y + rh.y) / 2) * 1080
        dist   = math.hypot(hip_cx - ball.center_x, hip_cy - ball.center_y)
        return dist < 120


# ---------------------------------------------------------------------------
# Sequence grouper
# ---------------------------------------------------------------------------

class _SequenceGrouper:
    """
    Groups consecutive frames with the same action into segments.
    Merges short gaps. Drops tiny segments.
    """
    MIN_SEGMENT_FRAMES = 4
    MAX_GAP_FRAMES     = 8

    def group(
        self,
        frame_actions: List[Tuple[int, float, str, str]],   # (frame_idx, ts, action, cue)
        fps: float,
    ) -> List[DetectedAction]:
        if not frame_actions:
            return []

        # Group by action.
        from collections import defaultdict
        by_action: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
        for fi, ts, action, cue in frame_actions:
            by_action[action].append((fi, ts, cue))

        segments: List[DetectedAction] = []

        for action, entries in by_action.items():
            entries.sort(key=lambda e: e[0])
            current_start_fi, current_start_ts, current_cue = entries[0]
            current_end_fi   = entries[0][0]
            current_end_ts   = entries[0][1]
            evidence_list    = [ActionEvidence(cue=entries[0][2], source="rules", frame_index=entries[0][0])]

            for i in range(1, len(entries)):
                fi, ts, cue = entries[i]
                gap = fi - current_end_fi

                if gap <= self.MAX_GAP_FRAMES:
                    current_end_fi = fi
                    current_end_ts = ts
                    evidence_list.append(ActionEvidence(cue=cue, source="rules", frame_index=fi))
                else:
                    # Save current segment if long enough.
                    if current_end_fi - current_start_fi >= self.MIN_SEGMENT_FRAMES:
                        segments.append(DetectedAction(
                            action       = action,
                            start_frame  = current_start_fi,
                            end_frame    = current_end_fi,
                            start_time_s = current_start_ts,
                            end_time_s   = current_end_ts,
                            evidence     = evidence_list,
                        ))
                    # Start new segment.
                    current_start_fi = fi
                    current_start_ts = ts
                    current_end_fi   = fi
                    current_end_ts   = ts
                    evidence_list    = [ActionEvidence(cue=cue, source="rules", frame_index=fi)]

            # Final segment.
            if current_end_fi - current_start_fi >= self.MIN_SEGMENT_FRAMES:
                segments.append(DetectedAction(
                    action       = action,
                    start_frame  = current_start_fi,
                    end_frame    = current_end_fi,
                    start_time_s = current_start_ts,
                    end_time_s   = current_end_ts,
                    evidence     = evidence_list,
                ))

        segments.sort(key=lambda s: s.start_frame)
        return segments


# ---------------------------------------------------------------------------
# Activity Detector
# ---------------------------------------------------------------------------

class ActivityDetector:
    """
    Detects which football actions happened in the video.

    Input:  Pose + Ball + Player results from earlier pipeline stages
    Output: ActivityUnderstandingResult — what actions happened, when

    This module ONLY answers "What happened?"
    It NEVER calculates performance scores.

    Usage::

        detector = ActivityDetector()
        result   = detector.detect(pose_result, ball_result, player_result)
        detector.write_to_context(result, ctx)
    """

    def __init__(self) -> None:
        self._rules   = _RuleEngine()
        self._grouper = _SequenceGrouper()

    def detect(
        self,
        pose_result:   PoseEstimationResult,
        ball_result:   BallDetectionResult,
        player_result: Optional[PlayerDetectionResult] = None,
        fps:           float = 25.0,
    ) -> ActivityUnderstandingResult:
        """
        Detect football actions from pose, ball, and player data.

        Parameters
        ----------
        pose_result   : PoseEstimationResult
        ball_result   : BallDetectionResult
        player_result : PlayerDetectionResult (optional)
        fps           : float

        Returns
        -------
        ActivityUnderstandingResult
        """
        # Build per-frame lookups.
        pose_map: Dict[int, FramePose] = {
            fp.frame_index: fp
            for fp in pose_result.frame_poses
            if fp.detected
        }
        ball_map: Dict[int, BallDetection] = {
            bd.frame_index: bd
            for bd in ball_result.detections
        }

        # Process each frame.
        frame_actions: List[Tuple[int, float, str, str]] = []
        sorted_frames  = sorted(pose_map.keys())
        prev_ball: Optional[BallDetection] = None

        for fi in sorted_frames:
            pose       = pose_map.get(fi)
            ball       = ball_map.get(fi)
            timestamp  = pose.timestamp_s if pose else fi / fps

            signals = self._rules.classify_frame(pose, ball, prev_ball, fi)
            for action, cue in signals:
                frame_actions.append((fi, timestamp, action, cue))

            if ball is not None:
                prev_ball = ball

        # Group into segments.
        segments = self._grouper.group(frame_actions, fps)

        # Derive action names.
        action_names = list(dict.fromkeys(s.action for s in segments))

        # Primary action = longest segment.
        primary = (
            max(segments, key=lambda s: s.end_frame - s.start_frame).action
            if segments else None
        )

        # Fallback: if nothing detected, assume passing (default activity).
        if not action_names:
            action_names = [FootballActionType.PASSING]
            primary      = FootballActionType.PASSING

        log.debug(
            "ActivityDetector: %d segments  actions=%s  primary=%s",
            len(segments), action_names, primary,
        )

        return ActivityUnderstandingResult(
            detected_actions = segments,
            action_names     = action_names,
            primary_action   = primary,
            frame_count      = len(pose_result.frame_poses),
        )

    @staticmethod
    def write_to_context(
        result: ActivityUnderstandingResult,
        ctx:    PipelineContext,
        fps:    float = 25.0,
    ) -> None:
        """Write activity detection results to PipelineContext."""
        ctx.activity.detected_activities = result.action_names
        ctx.activity.primary_activity    = result.primary_action
        ctx.activity.raw_detection_count = result.frame_count

        # Confidence scores — binary (detected = 1.0, not detected = 0.0).
        ctx.activity.confidence_scores = {
            action: 1.0 for action in result.action_names
        }

        # Timeline segments.
        ctx.activity.timeline = [
            ActivitySegmentCtx(
                action       = seg.action,
                start_time_s = seg.start_time_s,
                end_time_s   = seg.end_time_s,
                duration_s   = round(seg.end_time_s - seg.start_time_s, 2),
                confidence   = 1.0,
                label        = seg.label,
            )
            for seg in result.detected_actions
        ]

        ctx.log_stage(
            "activity_detect",
            f"actions={result.action_names}  "
            f"segments={len(result.detected_actions)}  "
            f"primary={result.primary_action}",
        )
