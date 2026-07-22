#!/usr/bin/env python3
"""
Tests — Goalkeeper Metrics
===========================
Verifies that GoalkeeperMetrics produces correct numeric output.

Run:  pytest tests/test_goalkeeper.py -v
"""

import pytest
from metrics.goalkeeper_metrics import GoalkeeperMetrics, GoalkeeperEvent, SaveEvent, DistributionEvent, ClaimEvent
from metrics.common_metrics     import Point2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_save(saved: bool = True, reaction: float = 0.3) -> SaveEvent:
    return SaveEvent(
        saved            = saved,
        reaction_time_s  = reaction,
        gk_start         = Point2D(0.5, 0.5),
        gk_end           = Point2D(0.6, 0.4),
        px_per_m         = 100.0,
    )


def make_event(
    saves:          list[SaveEvent]         | None = None,
    goals_conceded: int                            = 0,
    distributions:  list[DistributionEvent] | None = None,
    claims:         list[ClaimEvent]        | None = None,
    sweeper:        int                            = 0,
) -> GoalkeeperEvent:
    return GoalkeeperEvent(
        saves           = saves or [],
        goals_conceded  = goals_conceded,
        distributions   = distributions or [],
        claims          = claims or [],
        aerial_duels    = [],
        sweeper_actions = sweeper,
        body_positions  = [Point2D(0.5, 0.5)],
        ideal_positions = [Point2D(0.5, 0.5)],
        fps             = 25.0,
        px_per_m        = 100.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGoalkeeperMetrics:
    def setup_method(self):
        self.calc = GoalkeeperMetrics()

    def test_empty_events_returns_metric_set(self):
        ms = self.calc.calculate([])
        assert ms.activity == "goalkeeping"
        assert len(ms.metrics) > 0

    def test_all_saves_gives_100_save_pct(self):
        ev = make_event(saves=[make_save(True), make_save(True)], goals_conceded=0)
        ms = self.calc.calculate([ev])
        assert ms.to_numeric_dict()["Save Percentage"] == 100.0

    def test_all_goals_gives_0_save_pct(self):
        ev = make_event(saves=[], goals_conceded=3)
        ms = self.calc.calculate([ev])
        assert ms.to_numeric_dict()["Save Percentage"] == 0.0

    def test_reaction_time_averaged_correctly(self):
        saves = [make_save(saved=True, reaction=0.2), make_save(saved=True, reaction=0.4)]
        ev    = make_event(saves=saves)
        ms    = self.calc.calculate([ev])
        assert abs(ms.to_numeric_dict()["Reaction Time"] - 0.3) < 0.01

    def test_distribution_accuracy_all_successful(self):
        dists = [DistributionEvent(successful=True)] * 5
        ev    = make_event(distributions=dists)
        ms    = self.calc.calculate([ev])
        assert ms.to_numeric_dict()["Distribution Accuracy"] == 100.0

    def test_distribution_accuracy_none_successful(self):
        dists = [DistributionEvent(successful=False)] * 5
        ev    = make_event(distributions=dists)
        ms    = self.calc.calculate([ev])
        assert ms.to_numeric_dict()["Distribution Accuracy"] == 0.0

    def test_claiming_rate_all_successful(self):
        claims = [ClaimEvent(successful=True, punch=False)] * 4
        ev     = make_event(claims=claims)
        ms     = self.calc.calculate([ev])
        assert ms.to_numeric_dict()["Claiming Success Rate"] == 100.0

    def test_sweeper_actions_counted(self):
        ev = make_event(sweeper=3)
        ms = self.calc.calculate([ev])
        assert ms.to_numeric_dict()["Sweeper Actions"] == 3

    def test_positioning_score_between_0_and_100(self):
        ev = make_event()
        ms = self.calc.calculate([ev])
        ps = ms.to_numeric_dict()["Positioning Score"]
        assert 0.0 <= ps <= 100.0

    def test_to_dict_returns_string_values(self):
        ms = self.calc.calculate([make_event()])
        for v in ms.to_dict().values():
            assert isinstance(v, str)
