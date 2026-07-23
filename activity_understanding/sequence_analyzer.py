#!/usr/bin/env python3
"""
Sequence Analyzer
=================
Takes per-frame RawActivityDetection objects and groups CONSECUTIVE
frames of the same action into ActivitySegment objects, producing a
clean activity timeline for the video.

Algorithm
---------
1. For each action, collect the sorted list of frame indices where it
   was detected (taking the highest-confidence detection per frame).
2. Group consecutive frame indices into initial segments.
3. Merge adjacent segments whose gap is < min_gap_frames.
4. Drop segments shorter than min_segment_frames.
5. Build a timeline sorted by start frame.

Label format  →  "MM:SS–MM:SS  Action"
Example       →  "00:00–00:25  Passing"
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List

from activity_understanding.activity_detector import RawActivityDetection
from activity_understanding.activity_classifier import ClassifiedActivity
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _fmt_mmss(seconds: float) -> str:
    """Format seconds as MM:SS (e.g. 65.3 → '01:05')."""
    total_s = max(0, int(seconds))
    mm = total_s // 60
    ss = total_s % 60
    return f"{mm:02d}:{ss:02d}"


@dataclass
class ActivitySegment:
    """A contiguous time-range in the video dominated by one action."""
    action:       str
    start_frame:  int
    end_frame:    int
    start_time_s: float
    end_time_s:   float
    duration_s:   float
    confidence:   float
    label:        str     # e.g. "00:00–00:25  Passing"


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

_DEFAULT_MIN_GAP_FRAMES     = 8
_DEFAULT_MIN_SEGMENT_FRAMES = 5


class SequenceAnalyzer:
    """
    Groups per-frame detections into contiguous activity segments.

    Usage::

        segments = SequenceAnalyzer.analyze(raw_detections, fps=30)
        for seg in segments:
            print(seg.label, f"(confidence: {seg.confidence:.2f})")
    """

    @staticmethod
    def analyze(
        raw_detections:    List[RawActivityDetection],
        fps:               float,
        min_gap_frames:    int = _DEFAULT_MIN_GAP_FRAMES,
        min_segment_frames: int = _DEFAULT_MIN_SEGMENT_FRAMES,
    ) -> List[ActivitySegment]:
        """
        Build a sorted activity timeline from per-frame detections.

        Parameters
        ----------
        raw_detections : List[RawActivityDetection]
        fps : float
            Frames per second of the source video.
        min_gap_frames : int
            Gaps smaller than this between same-action detections are
            bridged (segments merged).
        min_segment_frames : int
            Segments shorter than this are dropped.

        Returns
        -------
        List[ActivitySegment]
            Sorted by start_frame.  Empty list on empty input.
        """
        if not raw_detections or fps <= 0:
            log.debug("SequenceAnalyzer.analyze: empty input or invalid fps")
            return []

        # Step 1: per action → {frame_index: best_confidence}
        best_conf_by_frame: dict[str, dict[int, float]] = defaultdict(dict)
        for det in raw_detections:
            existing = best_conf_by_frame[det.action].get(det.frame_index, 0.0)
            if det.confidence > existing:
                best_conf_by_frame[det.action][det.frame_index] = det.confidence

        segments: List[ActivitySegment] = []

        for action, frame_conf_map in best_conf_by_frame.items():
            sorted_frames = sorted(frame_conf_map.keys())
            if not sorted_frames:
                continue

            raw_segments = SequenceAnalyzer._group_consecutive(
                sorted_frames, frame_conf_map
            )
            merged = SequenceAnalyzer._merge_gaps(
                raw_segments, frame_conf_map, min_gap_frames
            )

            for seg_frames, seg_confs in merged:
                if len(seg_frames) < min_segment_frames:
                    continue

                start_frame = seg_frames[0]
                end_frame   = seg_frames[-1]
                start_time  = start_frame / fps
                end_time    = end_frame   / fps
                duration    = end_time - start_time
                confidence  = (
                    sum(seg_confs) / len(seg_confs) if seg_confs else 0.0
                )

                action_title = action.capitalize()
                label = (
                    f"{_fmt_mmss(start_time)}–{_fmt_mmss(end_time)}"
                    f"  {action_title}"
                )

                segments.append(ActivitySegment(
                    action       = action,
                    start_frame  = start_frame,
                    end_frame    = end_frame,
                    start_time_s = round(start_time, 3),
                    end_time_s   = round(end_time,   3),
                    duration_s   = round(duration,   3),
                    confidence   = round(confidence,  4),
                    label        = label,
                ))

        segments.sort(key=lambda s: s.start_frame)

        log.debug(
            "SequenceAnalyzer: produced %d segments from %d raw detections",
            len(segments),
            len(raw_detections),
        )

        return segments

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_consecutive(
        sorted_frames: List[int],
        frame_conf_map: dict[int, float],
    ) -> List[tuple[List[int], List[float]]]:
        """
        Split sorted_frames into runs of consecutive indices (gap == 1).
        Returns list of (frame_list, confidence_list) tuples.
        """
        if not sorted_frames:
            return []

        groups: List[tuple[List[int], List[float]]] = []
        current_frames: List[int]   = [sorted_frames[0]]
        current_confs:  List[float] = [frame_conf_map[sorted_frames[0]]]

        for i in range(1, len(sorted_frames)):
            f     = sorted_frames[i]
            f_prev = sorted_frames[i - 1]

            if f - f_prev == 1:
                current_frames.append(f)
                current_confs.append(frame_conf_map[f])
            else:
                groups.append((current_frames, current_confs))
                current_frames = [f]
                current_confs  = [frame_conf_map[f]]

        groups.append((current_frames, current_confs))
        return groups

    @staticmethod
    def _merge_gaps(
        segments:       List[tuple[List[int], List[float]]],
        frame_conf_map: dict[int, float],
        min_gap_frames: int,
    ) -> List[tuple[List[int], List[float]]]:
        """
        Merge adjacent segments whose gap is strictly less than min_gap_frames.
        Fills in the bridged frames with interpolated confidence values.
        """
        if len(segments) <= 1:
            return segments

        merged: List[tuple[List[int], List[float]]] = [segments[0]]

        for i in range(1, len(segments)):
            prev_frames, prev_confs = merged[-1]
            curr_frames, curr_confs = segments[i]

            gap = curr_frames[0] - prev_frames[-1] - 1

            if gap < min_gap_frames:
                # Bridge the gap: fill intermediate frames with the average
                # confidence of the two boundary frames.
                bridge_conf = (prev_confs[-1] + curr_confs[0]) / 2.0
                bridge_frames = list(range(prev_frames[-1] + 1, curr_frames[0]))
                bridge_confs  = [bridge_conf] * len(bridge_frames)

                merged[-1] = (
                    prev_frames  + bridge_frames + curr_frames,
                    prev_confs   + bridge_confs  + curr_confs,
                )
            else:
                merged.append((curr_frames, curr_confs))

        return merged
