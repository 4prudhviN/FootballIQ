#!/usr/bin/env python3
"""
Base Analyzer
=============
Abstract base class for all activity-specific analyzers.
Every concrete analyzer must inherit from BaseAnalyzer and
implement the ``analyze()`` method and ``name`` property.

Usage::

    class MyAnalyzer(BaseAnalyzer):
        @property
        def name(self) -> str:
            return FootballAction.PASSING.value

        def analyze(self, frames, pose_result, ball_result) -> ActionMetrics:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from schemas.activity_schema import ActionMetrics, FootballAction
from pipeline.frame_extractor import ExtractedFrame
from pipeline.pose_estimator import PoseEstimationResult
from pipeline.ball_detector import BallDetectionResult


class BaseAnalyzer(ABC):
    """
    Abstract base for all FootballIQ activity analyzers.

    Subclasses must implement:
      - ``name``     — the FootballAction string this analyzer handles
      - ``analyze()`` — runs analysis and returns ActionMetrics

    The ``analyze()`` method must *never* raise — if data is insufficient,
    return a valid ActionMetrics with empty / default metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the FootballAction value string, e.g. ``"passing"``."""

    @abstractmethod
    def analyze(
        self,
        frames:      List[ExtractedFrame],
        pose_result: PoseEstimationResult,
        ball_result: BallDetectionResult,
    ) -> ActionMetrics:
        """
        Run the activity-specific analysis and return computed metrics.

        Parameters
        ----------
        frames : list[ExtractedFrame]
            Raw video frames sampled by FrameExtractor.
        pose_result : PoseEstimationResult
            Per-frame pose landmarks and aggregate biomechanics.
        ball_result : BallDetectionResult
            Per-frame ball detections and overall confidence.

        Returns
        -------
        ActionMetrics
            Always a valid ActionMetrics instance — never None, never raises.
        """
