#!/usr/bin/env python3
"""
Tests — Dribbling Metrics
==========================
Verifies that DribblingMetrics produces correct numeric output.

Run:  pytest tests/test_dribbling.py -v
"""

import pytest
from metrics.dribbling_metrics import DribblingMetrics, DribbleEvent
from metrics.common_metrics    import Point2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def straight_positions(n: int = 10, step: float = 0.05) -> list[Point2D]:
    """Generate n points moving straight to the right."""
    return [Point2D(i * step, 0.5) for i in range(n)]


def make_dribble(
    completed:  bool          = True,
    ball_pts:   list[Point2D] | None = None,
    body_pts:   list[Point2D] | None = None,
) -> DribbleEvent:
    pts = ball_pts or straight_positions()
    body = body_pts or [Point2D(p.x, p.y + 0.05) for p in pts]
    return DribbleEvent(
        ball_positions = pts,
        body_positions = body,
        completed      = completed,
        fps            = 25.0,
        px_per_m       = 100.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDribblingMetrics:
    def setup_method(self):
        self.calc = DribblingMetrics()

    def test_empty_events_returns_metric_set(self):
        ms = self.calc.calculate([])
        assert ms.activity == "dribbling"
        assert len(ms.metrics) > 0

    def test_all_successful_gives_100_success_rate(self):
        events = [make_dribble(completed=True) for _ in range(4)]
        ms     = self.calc.calculate(events)
        assert ms.to_numeric_dict()["Dribble Success Rate"] == 100.0

    def test_all_failed_gives_0_success_rate(self):
        events = [make_dribble(completed=False) for _ in range(4)]
        ms     = self.calc.calculate(events)
        assert ms.to_numeric_dict()["Dribble Success Rate"] == 0.0

    def test_successful_plus_failed_equals_total(self):
        events = [make_dribble(True), make_dribble(False), make_dribble(True)]
        ms     = self.calc.calculate(events)
        nd     = ms.to_numeric_dict()
        assert nd["Successful Dribbles"] + nd["Failed Dribbles"] == len(events)

    def test_distance_covered_positive(self):
        events = [make_dribble()]
        ms     = self.calc.calculate(events)
        assert ms.to_numeric_dict()["Distance Covered"] >= 0.0

    def test_close_control_score_between_0_and_100(self):
        events = [make_dribble()]
        ms     = self.calc.calculate(events)
        cc     = ms.to_numeric_dict()["Close Control Score"]
        assert 0.0 <= cc <= 100.0

    def test_ball_retention_equals_success_rate(self):
        events = [make_dribble(True), make_dribble(False)]
        ms     = self.calc.calculate(events)
        nd     = ms.to_numeric_dict()
        assert nd["Ball Retention Rate"] == nd["Dribble Success Rate"]

    def test_to_dict_returns_string_values(self):
        ms = self.calc.calculate([make_dribble()])
        for v in ms.to_dict().values():
            assert isinstance(v, str)

    def test_tight_control_better_score_than_loose(self):
        # Ball very close to body = tight control
        tight_pts = straight_positions()
        tight_body = [Point2D(p.x, p.y + 0.01) for p in tight_pts]

        # Ball far from body = loose control
        loose_pts = straight_positions()
        loose_body = [Point2D(p.x, p.y + 0.30) for p in loose_pts]

        tight_ms = self.calc.calculate([make_dribble(ball_pts=tight_pts, body_pts=tight_body)])
        loose_ms = self.calc.calculate([make_dribble(ball_pts=loose_pts, body_pts=loose_body)])

        tight_cc = tight_ms.to_numeric_dict()["Close Control Score"]
        loose_cc = loose_ms.to_numeric_dict()["Close Control Score"]
        assert tight_cc >= loose_cc
