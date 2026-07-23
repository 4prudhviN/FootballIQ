#!/usr/bin/env python3
"""
Confidence Filter
=================
Filters, deduplicates, and normalises activity detection results.

Responsibilities
----------------
  filter_raw        — drop RawActivityDetection below a confidence floor
  filter_classified — drop ClassifiedActivity below a combined-score floor
  deduplicate_frame — for each (frame, action) pair keep only the highest-
                      confidence RawActivityDetection
  normalise         — rescale ClassifiedActivity combined_scores to [0, 1]
                      so the best activity always scores 1.0
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

from activity_understanding.activity_detector   import RawActivityDetection
from activity_understanding.activity_classifier import ClassifiedActivity
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_RAW_MIN_CONFIDENCE  = 0.15
_DEFAULT_CLASSIFIED_MIN_SCORE = 0.10


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class ConfidenceFilter:
    """
    Post-processing utility for activity detection results.

    All methods are stateless and can be called on any instance or
    directly via ConfidenceFilter().method(...).  They never raise on
    empty input — they return empty lists gracefully.

    Usage::

        cf = ConfidenceFilter()
        raw        = cf.filter_raw(raw_detections, min_confidence=0.15)
        raw        = cf.deduplicate_frame(raw)
        classified = cf.filter_classified(classified_activities, min_score=0.10)
        classified = cf.normalise(classified)
    """

    # ------------------------------------------------------------------
    # Raw detections
    # ------------------------------------------------------------------

    def filter_raw(
        self,
        detections:     List[RawActivityDetection],
        min_confidence: float = _DEFAULT_RAW_MIN_CONFIDENCE,
    ) -> List[RawActivityDetection]:
        """
        Remove RawActivityDetection objects below min_confidence.

        Parameters
        ----------
        detections : List[RawActivityDetection]
        min_confidence : float
            Inclusive lower bound (detections at exactly this value are kept).

        Returns
        -------
        List[RawActivityDetection]
            Filtered list in the same order as the input.
        """
        if not detections:
            return []

        result = [d for d in detections if d.confidence >= min_confidence]

        log.debug(
            "ConfidenceFilter.filter_raw: %d → %d (min_confidence=%.3f)",
            len(detections),
            len(result),
            min_confidence,
        )
        return result

    def deduplicate_frame(
        self,
        detections: List[RawActivityDetection],
    ) -> List[RawActivityDetection]:
        """
        For each (frame_index, action) pair keep only the detection with the
        highest confidence.  Preserves relative ordering of the surviving items
        (first occurrence per pair wins on tie).

        Parameters
        ----------
        detections : List[RawActivityDetection]

        Returns
        -------
        List[RawActivityDetection]
        """
        if not detections:
            return []

        # best[(frame_index, action)] → RawActivityDetection
        best: dict[tuple[int, str], RawActivityDetection] = {}

        for det in detections:
            key = (det.frame_index, det.action)
            existing = best.get(key)
            if existing is None or det.confidence > existing.confidence:
                best[key] = det

        # Restore original relative order.
        seen: set[tuple[int, str]] = set()
        result: List[RawActivityDetection] = []
        for det in detections:
            key = (det.frame_index, det.action)
            if key in best and best[key] is det and key not in seen:
                result.append(det)
                seen.add(key)

        log.debug(
            "ConfidenceFilter.deduplicate_frame: %d → %d",
            len(detections),
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Classified activities
    # ------------------------------------------------------------------

    def filter_classified(
        self,
        activities: List[ClassifiedActivity],
        min_score:  float = _DEFAULT_CLASSIFIED_MIN_SCORE,
    ) -> List[ClassifiedActivity]:
        """
        Remove ClassifiedActivity objects whose combined_score is below min_score.

        Parameters
        ----------
        activities : List[ClassifiedActivity]
        min_score : float
            Inclusive lower bound.

        Returns
        -------
        List[ClassifiedActivity]
        """
        if not activities:
            return []

        result = [a for a in activities if a.combined_score >= min_score]

        log.debug(
            "ConfidenceFilter.filter_classified: %d → %d (min_score=%.3f)",
            len(activities),
            len(result),
            min_score,
        )
        return result

    def normalise(
        self,
        activities: List[ClassifiedActivity],
    ) -> List[ClassifiedActivity]:
        """
        Rescale combined_score (and avg_confidence) values so the highest
        combined_score maps to 1.0 and the rest are scaled proportionally.

        The original objects are mutated in-place and also returned so the
        method is usable both ways::

            activities = cf.normalise(activities)   # reassign
            cf.normalise(activities)                # or mutate in-place

        Parameters
        ----------
        activities : List[ClassifiedActivity]

        Returns
        -------
        List[ClassifiedActivity]
            Same list, with combined_score values rescaled to [0, 1].
        """
        if not activities:
            return []

        max_score = max(a.combined_score for a in activities)
        if max_score <= 0.0:
            log.debug("ConfidenceFilter.normalise: max_score=0 — nothing to normalise")
            return activities

        for a in activities:
            a.combined_score = round(a.combined_score / max_score, 4)

        log.debug(
            "ConfidenceFilter.normalise: normalised %d activities (max_score was %.4f)",
            len(activities),
            max_score,
        )
        return activities
