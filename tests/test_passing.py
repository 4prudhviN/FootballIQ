#!/usr/bin/env python3
"""
Tests — Passing Metrics
========================
Verifies that PassingMetrics produces correct numeric output
for a variety of pass event inputs.

Run:  pytest tests/test_passing.py -v
"""

import pytest
from metrics.passing_metrics import PassingMetrics, PassEvent
from metrics.common_metrics  import Point2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pass(
    completed: bool = True,
    foot:      str  = "right",
    sx: float = 0.1, sy: float = 0.5,
    ex: float = 0.4, ey: float = 0.5,
    speed: float = 5.0,
) -> PassEvent:
    return PassEvent(
        start      = Point2D(sx, sy),
        end        = Point2D(ex, ey),
        completed  = completed,
        speed_px_f = speed,
        foot       = foot,
        frame_index= 0,
        fps        = 25.0,
        px_per_m   = 100.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPassingMetrics:
    def setup_method(self):
        self.calc = PassingMetrics()

    def test_empty_events_returns_metric_set(self):
        ms = self.calc.calculate([])
        assert ms.activity == "passing"
        assert len(ms.metrics) > 0

    def test_all_completed_passes_gives_100_accuracy(self):
        events = [make_pass(completed=True) for _ in range(5)]
        ms     = self.calc.calculate(events)
        acc    = ms.to_numeric_dict()["Passing Accuracy"]
        assert acc == 100.0

    def test_zero_completed_passes_gives_0_accuracy(self):
        events = [make_pass(completed=False) for _ in range(5)]
        ms     = self.calc.calculate(events)
        acc    = ms.to_numeric_dict()["Passing Accuracy"]
        assert acc == 0.0

    def test_successful_and_failed_counts_sum_to_total(self):
        events = [make_pass(True), make_pass(False), make_pass(True)]
        ms     = self.calc.calculate(events)
        nd     = ms.to_numeric_dict()
        assert nd["Successful Passes"] + nd["Failed Passes"] == nd["Total Passes"]

    def test_total_passes_equals_input_length(self):
        events = [make_pass() for _ in range(7)]
        ms     = self.calc.calculate(events)
        assert ms.to_numeric_dict()["Total Passes"] == 7

    def test_short_pass_ratio_for_short_passes(self):
        # All passes are short (distance < 0.15)
        events = [make_pass(sx=0.1, ex=0.2) for _ in range(4)]
        ms     = self.calc.calculate(events)
        assert ms.to_numeric_dict()["Short Pass Ratio"] == 100.0

    def test_long_pass_ratio_for_long_passes(self):
        # All passes are long (distance > 0.35)
        events = [make_pass(sx=0.1, ex=0.6) for _ in range(4)]
        ms     = self.calc.calculate(events)
        assert ms.to_numeric_dict()["Long Pass Ratio"] == 100.0

    def test_weak_foot_ratio_all_left(self):
        events = [make_pass(foot="left") for _ in range(4)]
        ms     = self.calc.calculate(events)
        assert ms.to_numeric_dict()["Weak Foot Ratio"] == 100.0

    def test_weak_foot_ratio_all_right(self):
        events = [make_pass(foot="right") for _ in range(4)]
        ms     = self.calc.calculate(events)
        assert ms.to_numeric_dict()["Weak Foot Ratio"] == 0.0

    def test_to_dict_returns_string_values(self):
        ms = self.calc.calculate([make_pass()])
        d  = ms.to_dict()
        for v in d.values():
            assert isinstance(v, str)

    def test_ball_control_index_between_0_and_100(self):
        events = [make_pass(speed=3.0)]
        ms     = self.calc.calculate(events)
        bci    = ms.to_numeric_dict()["Ball Control Index"]
        assert 0.0 <= bci <= 100.0

    def test_mixed_accuracy(self):
        events = [make_pass(True), make_pass(True), make_pass(False)]
        ms     = self.calc.calculate(events)
        acc    = ms.to_numeric_dict()["Passing Accuracy"]
        assert abs(acc - 66.7) < 1.0
