#!/usr/bin/env python3
"""Tests — Mistake Detector"""
import pytest
from coach_engine.mistake_detector import MistakeDetector

class TestMistakeDetector:
    def setup_method(self): self.d = MistakeDetector()

    def test_poor_accuracy_detects_lean_back(self):
        result = self.d.detect("passing", {"accuracy": 55.0}, "Beginner")
        causes = [p.cause for p in result.problems]
        assert any("lean" in c.lower() or "plant" in c.lower() for c in causes)

    def test_poor_alignment_detects_posture(self):
        result = self.d.detect("passing", {"body_alignment": "Poor"}, "Beginner")
        assert len(result.problems) > 0

    def test_no_problems_for_good_metrics(self):
        result = self.d.detect("passing", {"accuracy": 90.0, "body_alignment": "Good"}, "Advanced")
        assert len(result.problems) == 0

    def test_high_severity_first(self):
        result = self.d.detect("shooting", {"shot_accuracy": 40.0, "body_alignment": "Poor"}, "Beginner")
        if len(result.problems) >= 2:
            assert result.problems[0].severity in ("high", "medium")

    def test_empty_metrics_no_crash(self):
        result = self.d.detect("passing", {}, "Beginner")
        assert result is not None

    def test_unknown_activity_no_crash(self):
        result = self.d.detect("unknown_sport", {"accuracy": 50.0}, "Beginner")
        assert result is not None

    def test_summary_lines_format(self):
        result = self.d.detect("passing", {"accuracy": 55.0}, "Beginner")
        lines  = result.summary_lines()
        for line in lines:
            assert line.startswith("  ✓")

    def test_each_problem_has_correction(self):
        result = self.d.detect("passing", {"accuracy": 55.0, "body_alignment": "Poor"}, "Intermediate")
        for p in result.problems:
            assert p.correction

    def test_dribbling_ball_too_far(self):
        result = self.d.detect("dribbling", {"dribble_success_rate": 40.0}, "Beginner")
        causes = [p.cause.lower() for p in result.problems]
        assert any("ball" in c for c in causes)

    def test_result_has_activity(self):
        result = self.d.detect("shooting", {"shot_accuracy": 45.0}, "Developing")
        assert result.activity == "shooting"
