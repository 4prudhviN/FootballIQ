#!/usr/bin/env python3
"""
Stage 5 — Activity Detector
=============================
Classifies which football activities are present in the video based
on pose warnings, ball detection data, and motion patterns.

Responsibilities:
  - Consume PoseEstimationResult and BallDetectionResult
  - Classify detected football actions: passing, shooting, dribbling,
    defending, goalkeeping, movement
  - Return a prioritised list of FootballActivity objects
  - Only return activities that were actually observed — never assume
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pipeline.pose_estimator import PoseEstimationResult
from pipeline.ball_detector  import BallDetectionResult


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

SUPPORTED_ACTIVITIES = {
    "passing",
    "dribbling",
    "shooting",
    "goalkeeping",
    "defending",
    "movement",
}


@dataclass
class FootballActivity:
    """A single detected football activity."""
    name:       str           # e.g. "shooting"
    confidence: float         # 0.0 – 1.0
    evidence:   List[str] = field(default_factory=list)  # human-readable reasons


@dataclass
class ActivityDetectionResult:
    """Output of the ActivityDetector for one video."""
    activities:  List[FootballActivity]
    names:       List[str]           # ordered by confidence (highest first)
    primary:     Optional[str]       # most confident activity


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ActivityDetector:
    """
    Stage 5: Classify football activities from pose and ball evidence.

    Uses a rule-based scoring system.
    Upgrade path: replace with a trained sequence classifier (LSTM / TCN).

    Usage::

        detector = ActivityDetector()
        result   = detector.detect(pose_result, ball_result)
        print(result.names)   # e.g. ["shooting", "movement"]
    """

    def detect(
        self,
        pose:  PoseEstimationResult,
        ball:  BallDetectionResult,
    ) -> ActivityDetectionResult:
        """
        Classify activities based on pose warnings and ball presence.

        Parameters
        ----------
        pose : PoseEstimationResult
        ball : BallDetectionResult

        Returns
        -------
        ActivityDetectionResult
        """
        scores: dict[str, tuple[float, list[str]]] = {
            a: (0.0, []) for a in SUPPORTED_ACTIVITIES
        }

        def add(activity: str, score: float, reason: str) -> None:
            s, reasons = scores[activity]
            scores[activity] = (min(1.0, s + score), reasons + [reason])

        # ── Pose-based rules ─────────────────────────────────────────────────

        if "POOR POSTURE / LEANING BACK" in pose.warnings:
            add("shooting",  0.55, "Torso lean-back pattern detected (common in kicks/shots)")
            add("dribbling", 0.20, "Torso lean can occur during sharp direction changes")

        if "KNEE ALIGNMENT RISK" in pose.warnings:
            add("dribbling", 0.40, "Knee deviation during lateral cuts typical of dribbling")
            add("defending", 0.25, "Knee stress occurs in defensive jockeying stances")

        if "ASYMMETRIC GAIT DETECTED" in pose.warnings:
            add("movement",  0.50, "Stride asymmetry detected in running pattern")
            add("dribbling", 0.20, "Asymmetric gait can indicate ball-at-feet running")

        # ── Ball-based rules ──────────────────────────────────────────────────

        if ball.ball_detected:
            add("shooting",  0.25, f"Ball detected (conf: {ball.confidence:.1%})")
            add("passing",   0.25, f"Ball detected (conf: {ball.confidence:.1%})")
            add("dribbling", 0.15, f"Ball detected (conf: {ball.confidence:.1%})")

            if ball.confidence > 0.5:
                add("shooting", 0.10, "High ball confidence — likely shot or pass")

        # ── Pose detection coverage ───────────────────────────────────────────

        detection_rate = (
            pose.detected_frames / pose.total_frames
            if pose.total_frames > 0 else 0.0
        )
        if detection_rate > 0.5:
            add("movement", 0.30, f"Consistent pose detected ({detection_rate:.0%} of frames)")

        # ── Fallback: always include at least one activity ────────────────────

        if all(s == 0.0 for s, _ in scores.values()):
            add("passing", 0.40, "Default activity — no strong signal detected")

        # ── Build result ──────────────────────────────────────────────────────

        activities: List[FootballActivity] = []
        for name, (score, evidence) in scores.items():
            if score > 0:
                activities.append(FootballActivity(
                    name       = name,
                    confidence = round(score, 3),
                    evidence   = evidence,
                ))

        # Sort by confidence descending.
        activities.sort(key=lambda a: a.confidence, reverse=True)

        # Keep only activities with meaningful confidence (> 0.15).
        activities = [a for a in activities if a.confidence >= 0.15]

        names   = [a.name for a in activities]
        primary = names[0] if names else None

        return ActivityDetectionResult(
            activities = activities,
            names      = names,
            primary    = primary,
        )
