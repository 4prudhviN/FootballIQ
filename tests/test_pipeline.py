#!/usr/bin/env python3
"""
Tests — Pipeline Validator
===========================
Verifies that PipelineValidator gates work correctly for each stage.

Run:  pytest tests/test_pipeline.py -v
"""

import pytest
from unittest.mock import MagicMock
from pipeline.pipeline_context  import (
    PipelineContext, VideoInfo, FrameStore, DetectionResults,
    ActivityUnderstanding, AnalysisResults, CoachingOutput, FinalReport,
)
from pipeline.pipeline_validator import PipelineValidator


def make_ctx(**kwargs) -> PipelineContext:
    ctx = PipelineContext(video_path="test.mp4")
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


class TestVideoLoad:
    def setup_method(self): self.v = PipelineValidator()

    def test_passes_valid_video(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.video = VideoInfo(video_path="test.mp4", fps=25.0, width=1920, height=1080, frame_count=750)
        ok, err = self.v.after_video_load(ctx)
        assert ok and not err

    def test_fails_zero_fps(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.video = VideoInfo(video_path="test.mp4", fps=0.0, width=1920, height=1080, frame_count=100)
        ok, err = self.v.after_video_load(ctx)
        assert not ok and "FPS" in err

    def test_fails_zero_dimensions(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.video = VideoInfo(video_path="test.mp4", fps=25.0, width=0, height=0, frame_count=100)
        ok, err = self.v.after_video_load(ctx)
        assert not ok

    def test_fails_zero_frames(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.video = VideoInfo(video_path="test.mp4", fps=25.0, width=1920, height=1080, frame_count=0)
        ok, err = self.v.after_video_load(ctx)
        assert not ok


class TestFrameExtract:
    def setup_method(self): self.v = PipelineValidator()

    def test_passes_with_frames(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.frames = FrameStore(original_frames=[MagicMock()] * 10)
        ok, err = self.v.after_frame_extract(ctx)
        assert ok

    def test_fails_empty_frames(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.frames = FrameStore(original_frames=[])
        ok, err = self.v.after_frame_extract(ctx)
        assert not ok


class TestPlayerDetect:
    def setup_method(self): self.v = PipelineValidator()

    def test_passes_above_threshold(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.detections.player_confidence = 0.85
        ok, _ = self.v.after_player_detect(ctx, threshold=0.10)
        assert ok

    def test_fails_below_threshold(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.detections.player_confidence = 0.05
        ok, err = self.v.after_player_detect(ctx, threshold=0.10)
        assert not ok and "player" in err.lower()


class TestBallDetect:
    def setup_method(self): self.v = PipelineValidator()

    def test_always_passes(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.detections.ball_confidence = 0.0
        ok, _ = self.v.after_ball_detect(ctx)
        assert ok   # informational only


class TestPoseEstimate:
    def setup_method(self): self.v = PipelineValidator()

    def test_passes_enough_frames(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.detections.pose_detected_frames = 10
        ctx.detections.pose_total_frames    = 20
        ok, _ = self.v.after_pose_estimate(ctx, min_frames=5)
        assert ok

    def test_fails_too_few_frames(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.detections.pose_detected_frames = 2
        ctx.detections.pose_total_frames    = 20
        ok, err = self.v.after_pose_estimate(ctx, min_frames=5)
        assert not ok


class TestActivityDetect:
    def setup_method(self): self.v = PipelineValidator()

    def test_passes_with_activities(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.activity.detected_activities = ["passing"]
        ok, _ = self.v.after_activity_detect(ctx)
        assert ok

    def test_falls_back_to_movement(self):
        ctx = PipelineContext(video_path="test.mp4")
        ctx.activity.detected_activities = []
        ok, _ = self.v.after_activity_detect(ctx)
        assert ok
        assert ctx.activity.detected_activities == ["movement"]
