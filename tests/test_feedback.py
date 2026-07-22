#!/usr/bin/env python3
"""
Tests — Feedback Engine
========================
Verifies that FeedbackEngine produces correct plain-English output,
drills, and coach tips for all severity levels and activities.

Run:  pytest tests/test_feedback.py -v
"""

import pytest
from feedback_engine import FeedbackEngine, FeedbackRequest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> FeedbackEngine:
    return FeedbackEngine()


def make_request(
    metrics:  dict | None = None,
    activity: str         = "shooting",
    level:    str         = "Intermediate",
) -> FeedbackRequest:
    return FeedbackRequest(
        metrics  = metrics or {"torso_lean": -22.0, "knee_dev": 0.28, "leg_speed": 35.0},
        activity = activity,
        level    = level,
    )


# ---------------------------------------------------------------------------
# Report structure tests
# ---------------------------------------------------------------------------

class TestFeedbackEngineStructure:
    def test_report_has_summary(self, engine):
        report = engine.generate(make_request())
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_report_has_items_list(self, engine):
        report = engine.generate(make_request())
        assert isinstance(report.items, list)

    def test_report_has_motivational_tip(self, engine):
        report = engine.generate(make_request())
        assert isinstance(report.motivational_tip, str)
        assert len(report.motivational_tip) > 0

    def test_report_has_positive_list(self, engine):
        report = engine.generate(make_request())
        assert isinstance(report.positive, list)

    def test_priority_drill_is_first_item_drill(self, engine):
        report = engine.generate(make_request())
        if report.items:
            assert report.priority_drill == report.items[0].drill


# ---------------------------------------------------------------------------
# Severity and metric tests
# ---------------------------------------------------------------------------

class TestFeedbackEngineMetrics:
    def test_poor_torso_lean_produces_feedback_item(self, engine):
        report = engine.generate(make_request(metrics={"torso_lean": -25.0}))
        metrics = [i.metric for i in report.items]
        assert "torso_lean" in metrics

    def test_poor_knee_dev_produces_feedback_item(self, engine):
        report = engine.generate(make_request(metrics={"knee_dev": 0.35}))
        metrics = [i.metric for i in report.items]
        assert "knee_dev" in metrics

    def test_good_metrics_produce_no_items(self, engine):
        report = engine.generate(make_request(metrics={
            "torso_lean": -5.0, "knee_dev": 0.10, "gait_asymmetry": 0.05
        }))
        assert len(report.items) == 0

    def test_good_metrics_produce_positive_observations(self, engine):
        report = engine.generate(make_request(metrics={
            "torso_lean": -5.0, "knee_dev": 0.10
        }))
        assert len(report.positive) > 0

    def test_each_item_has_observation_drill_tip(self, engine):
        report = engine.generate(make_request())
        for item in report.items:
            assert item.observation
            assert item.drill
            assert item.coach_tip


# ---------------------------------------------------------------------------
# Level-specific tests
# ---------------------------------------------------------------------------

class TestFeedbackEngineLevel:
    def test_beginner_tip_in_motivational(self, engine):
        report = engine.generate(make_request(level="Beginner"))
        assert len(report.motivational_tip) > 0

    def test_advanced_tip_differs_from_beginner(self, engine):
        beginner = engine.generate(make_request(level="Beginner"))
        advanced = engine.generate(make_request(level="Advanced"))
        assert beginner.motivational_tip != advanced.motivational_tip

    def test_all_activities_supported(self, engine):
        for activity in ["shooting", "passing", "dribbling", "defending", "goalkeeping", "general"]:
            report = engine.generate(make_request(activity=activity))
            assert report.summary  # no crash, has output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestFeedbackEngineEdgeCases:
    def test_empty_metrics_does_not_crash(self, engine):
        report = engine.generate(FeedbackRequest(metrics={}, activity="general", level="Beginner"))
        assert report.summary

    def test_none_metric_values_skipped(self, engine):
        report = engine.generate(make_request(metrics={"torso_lean": None}))
        assert report is not None

    def test_unknown_activity_falls_back_gracefully(self, engine):
        report = engine.generate(FeedbackRequest(
            metrics={"torso_lean": -20.0},
            activity="unknown_sport",
            level="Intermediate",
        ))
        assert report.summary
