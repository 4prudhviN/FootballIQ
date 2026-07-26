#!/usr/bin/env python3
"""
Session Timeline
================
Stores the activity timeline inside PipelineContext so the report reads like
a real coach watched the session.

Output:
  Passing     00:00-00:18
  Ball Control 00:18-00:45
  Shooting    00:45-01:06

Report becomes:
  "The session started with passing drills. After improving ball control,
   the player switched to shooting practice."

Also stores frame-level evidence per recommendation:
  Passing Accuracy 74%
  Evidence:
    Frame 182 — Body leaning backwards
    Frame 214 — Plant foot too far
    Frame 301 — Eyes looking down
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FrameEvidence:
    """Observed issue at a specific frame."""
    frame_index:  int
    timestamp_s:  float
    observation:  str     # e.g. "Body leaning backwards"
    metric:       str     # which metric this relates to


@dataclass
class TimelineSegment:
    """One activity segment in the session timeline."""
    action:        str
    start_time_s:  float
    end_time_s:    float
    duration_s:    float
    label:         str           # e.g. "00:00–00:18  Passing"
    evidence:      List[FrameEvidence] = field(default_factory=list)

    @staticmethod
    def fmt(s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f"{m:02d}:{sec:02d}"


@dataclass
class SessionTimeline:
    """
    Complete activity timeline for one session.
    Stored inside PipelineContext.activity.
    """
    segments:           List[TimelineSegment] = field(default_factory=list)
    narrative_summary:  str                   = ""

    def build_narrative(self) -> str:
        """
        Build a coach-style narrative from the timeline.
        e.g. "The session started with passing drills. After improving
              control, the player switched to shooting practice."
        """
        if not self.segments:
            return "Session analysis complete."

        parts = []
        for i, seg in enumerate(self.segments):
            action = seg.action.replace("_", " ").capitalize()
            t      = seg.label.split("  ")[0]   # "00:00–00:18"

            if i == 0:
                parts.append(f"The session started with {action} ({t}).")
            elif i == len(self.segments) - 1:
                parts.append(f"The session ended with {action} ({t}).")
            else:
                prev = self.segments[i - 1].action.replace("_", " ")
                parts.append(
                    f"After {prev}, the player switched to {action} ({t})."
                )

        self.narrative_summary = " ".join(parts)
        return self.narrative_summary

    def to_dict(self) -> dict:
        return {
            "segments": [
                {
                    "action":     s.action,
                    "startTime":  s.start_time_s,
                    "endTime":    s.end_time_s,
                    "duration":   s.duration_s,
                    "label":      s.label,
                    "evidence":   [
                        {
                            "frame":       e.frame_index,
                            "time":        e.timestamp_s,
                            "observation": e.observation,
                            "metric":      e.metric,
                        }
                        for e in s.evidence
                    ],
                }
                for s in self.segments
            ],
            "narrative": self.narrative_summary or self.build_narrative(),
        }


def build_timeline_from_context(ctx) -> SessionTimeline:
    """
    Build a SessionTimeline from PipelineContext activity segments.
    Attaches frame-level evidence from pose landmarks and mistake detection.
    """
    timeline = SessionTimeline()

    for seg_ctx in (ctx.activity.timeline or []):
        seg = TimelineSegment(
            action       = seg_ctx.action,
            start_time_s = seg_ctx.start_time_s,
            end_time_s   = seg_ctx.end_time_s,
            duration_s   = seg_ctx.duration_s,
            label        = seg_ctx.label,
        )

        # Attach frame evidence from pose landmarks in this time window.
        for pose_frame in (ctx.detections.pose_landmarks or []):
            if not (seg.start_time_s <= pose_frame.timestamp_s <= seg.end_time_s):
                continue

            torso = getattr(pose_frame, "torso_lean", None)
            if torso is not None and abs(torso) > 18:
                seg.evidence.append(FrameEvidence(
                    frame_index = pose_frame.frame_index,
                    timestamp_s = pose_frame.timestamp_s,
                    observation = f"Body leaning backwards ({abs(torso):.1f}°)",
                    metric      = "torso_lean",
                ))

        timeline.segments.append(seg)

    timeline.build_narrative()
    return timeline
