#!/usr/bin/env python3
"""
Activity Classifier
===================
Takes a list of per-frame RawActivityDetection objects and classifies
the *video* into a ranked list of ClassifiedActivity objects.

Algorithm
---------
For each unique action seen across all frames:
  1. coverage        = frames_with_action / total_frames
  2. avg_confidence  = mean(confidence) for frames where this action appeared
  3. combined_score  = coverage × avg_confidence

The returned list is sorted descending by combined_score so the most
prominent activity in the video comes first.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from activity_understanding.activity_detector import RawActivityDetection
from config.thresholds import (
    SHOOTING_MIN_CONFIDENCE,
    PASSING_MIN_CONFIDENCE,
    DRIBBLING_MIN_CONFIDENCE,
    MOVEMENT_MIN_CONFIDENCE,
)
from schemas.activity_schema import FootballAction
from utils.logger            import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedActivity:
    """Video-level classification result for one football action."""
    action:           str
    combined_score:   float         # coverage × avg_confidence
    avg_confidence:   float
    coverage:         float         # fraction of total frames this action was detected in
    frame_count:      int
    evidence_summary: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ActivityClassifier:
    """
    Classifies a video's dominant activities from per-frame detections.

    Usage::

        classified = ActivityClassifier.classify(raw_detections)
        for ca in classified:
            print(ca.action, ca.combined_score)
    """

    @staticmethod
    def classify(
        raw_detections: List[RawActivityDetection],
    ) -> List[ClassifiedActivity]:
        """
        Group frame-level detections by action and compute video-level scores.

        Parameters
        ----------
        raw_detections : List[RawActivityDetection]
            All per-frame detections from ActivityDetector.detect().

        Returns
        -------
        List[ClassifiedActivity]
            Sorted descending by combined_score.  Empty list if no
            detections are provided.
        """
        if not raw_detections:
            log.debug("ActivityClassifier.classify: empty input — returning []")
            return []

        # Count unique frames (by frame_index) across all detections.
        total_frames = len({d.frame_index for d in raw_detections})
        if total_frames == 0:
            return []

        # Accumulate per action: frame indices seen + confidence values + evidence.
        frames_per_action:      dict[str, set[int]]   = defaultdict(set)
        confidences_per_action: dict[str, List[float]] = defaultdict(list)
        evidence_per_action:    dict[str, List[str]]   = defaultdict(list)

        for det in raw_detections:
            frames_per_action[det.action].add(det.frame_index)
            confidences_per_action[det.action].append(det.confidence)
            evidence_per_action[det.action].extend(det.evidence)

        results: List[ClassifiedActivity] = []

        for action, frame_set in frames_per_action.items():
            frame_count    = len(frame_set)
            coverage       = frame_count / total_frames
            confidences    = confidences_per_action[action]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            combined_score = coverage * avg_confidence

            # Deduplicate evidence strings (preserve order of first occurrence).
            seen_ev: set[str] = set()
            unique_evidence: List[str] = []
            for ev in evidence_per_action[action]:
                if ev not in seen_ev:
                    seen_ev.add(ev)
                    unique_evidence.append(ev)

            results.append(ClassifiedActivity(
                action          = action,
                combined_score  = round(combined_score,  4),
                avg_confidence  = round(avg_confidence,  4),
                coverage        = round(coverage,        4),
                frame_count     = frame_count,
                evidence_summary = unique_evidence,
            ))

        # Sort by combined_score descending.
        results.sort(key=lambda ca: ca.combined_score, reverse=True)

        log.debug(
            "ActivityClassifier: classified %d actions from %d detections / %d frames",
            len(results),
            len(raw_detections),
            total_frames,
        )

        return results
