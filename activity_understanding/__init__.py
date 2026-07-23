#!/usr/bin/env python3
"""
activity_understanding
======================
Frame-level activity detection, video-level classification,
sequence timeline analysis, and confidence filtering.

Public API
----------
    from activity_understanding import (
        ActivityDetector, RawActivityDetection,
        ActivityClassifier, ClassifiedActivity,
        SequenceAnalyzer, ActivitySegment,
        ConfidenceFilter,
    )
"""

from __future__ import annotations

from activity_understanding.activity_detector   import ActivityDetector, RawActivityDetection
from activity_understanding.activity_classifier import ActivityClassifier, ClassifiedActivity
from activity_understanding.sequence_analyzer   import SequenceAnalyzer, ActivitySegment
from activity_understanding.confidence_filter   import ConfidenceFilter

__all__ = [
    "ActivityDetector",
    "RawActivityDetection",
    "ActivityClassifier",
    "ClassifiedActivity",
    "SequenceAnalyzer",
    "ActivitySegment",
    "ConfidenceFilter",
]
