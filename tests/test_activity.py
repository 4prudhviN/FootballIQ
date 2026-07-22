#!/usr/bin/env python3
"""
Tests — Activity Detector
=========================
Verifies that the ActivityDetector correctly classifies football actions
from pose warnings and ball detection data.

Run:  pytest tests/test_activity.py -v
"""

import pytest
from unittest.mock import MagicMock

from pipeline.activity_detector import ActivityDetector
from pipeline.pose_estimator    import PoseEstimationResult
from pipeline.ball_detector     import BallDetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pose(warnings: list[str], detected: int = 20, total: int = 30) -> PoseEstimationResult:
    return PoseEstimationResult(
        frame_poses     = [],
        detected_frames = detected,
        total_frames    = total,
        warnings        = warnings,
        avg_torso_lean  = None,
        avg_knee_dev    = None,
        gait_asymmetry  = None,
    )


def make_ball(detected: bool, confidence: float = 0.0) -> BallDetectionResult:
    return BallDetectionResult(
        detected_frames = int(confidence * 10),
        total_frames    = 10,
        confidence      = confidence,
        ball_detected   = detected,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActivityDetector:
    def setup_method(self):
        self.detector = ActivityDetector()

    def test_shooting_detected_from_torso_lean_warning(self):
        pose   = make_pose(["POOR POSTURE / LEANING BACK"])
        ball   = make_ball(detected=False)
        result = self.detector.detect(pose, ball)
        assert "shooting" in result.names

    def test_dribbling_detected_from_knee_warning(self):
        pose   = make_pose(["KNEE ALIGNMENT RISK"])
        ball   = make_ball(detected=False)
        result = self.detector.detect(pose, ball)
        assert "dribbling" in result.names

    def test_movement_detected_from_gait_warning(self):
        pose   = make_pose(["ASYMMETRIC GAIT DETECTED"])
        ball   = make_ball(detected=False)
        result = self.detector.detect(pose, ball)
        assert "movement" in result.names

    def test_ball_presence_boosts_shooting_and_passing(self):
        pose   = make_pose([])
        ball   = make_ball(detected=True, confidence=0.60)
        result = self.detector.detect(pose, ball)
        assert "shooting" in result.names or "passing" in result.names

    def test_fallback_returns_passing_when_no_signals(self):
        pose   = make_pose([], detected=0, total=0)
        ball   = make_ball(detected=False)
        result = self.detector.detect(pose, ball)
        assert len(result.names) > 0   # always at least one activity

    def test_primary_is_highest_confidence(self):
        pose   = make_pose(["POOR POSTURE / LEANING BACK", "KNEE ALIGNMENT RISK"])
        ball   = make_ball(detected=True, confidence=0.50)
        result = self.detector.detect(pose, ball)
        assert result.primary is not None
        top = max(result.activities, key=lambda a: a.confidence)
        assert result.primary == top.name

    def test_activities_sorted_by_confidence_descending(self):
        pose   = make_pose(["POOR POSTURE / LEANING BACK"])
        ball   = make_ball(detected=True, confidence=0.40)
        result = self.detector.detect(pose, ball)
        confs  = [a.confidence for a in result.activities]
        assert confs == sorted(confs, reverse=True)

    def test_low_confidence_activities_filtered_out(self):
        pose   = make_pose([])
        ball   = make_ball(detected=False)
        result = self.detector.detect(pose, ball)
        for a in result.activities:
            assert a.confidence >= 0.15

    def test_all_warnings_produce_activities(self):
        pose = make_pose([
            "POOR POSTURE / LEANING BACK",
            "KNEE ALIGNMENT RISK",
            "ASYMMETRIC GAIT DETECTED",
        ])
        ball   = make_ball(detected=True, confidence=0.50)
        result = self.detector.detect(pose, ball)
        assert len(result.names) > 0

    def test_result_names_match_activity_list(self):
        pose   = make_pose(["POOR POSTURE / LEANING BACK"])
        ball   = make_ball(detected=False)
        result = self.detector.detect(pose, ball)
        assert result.names == [a.name for a in result.activities]
