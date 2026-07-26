#!/usr/bin/env python3
"""Tests — Feedback Engine (coach_engine)"""
import pytest
from coach_engine.feedback_engine import CoachFeedbackEngine, CoachFeedbackReport
from coach_engine.skill_classifier import CoachSkillClassifier

def make_profile(metrics):
    return CoachSkillClassifier().classify(metrics)

class TestCoachFeedbackEngine:
    def setup_method(self): self.engine = CoachFeedbackEngine()

    def test_generates_report(self):
        profile = make_profile({"accuracy": 60.0, "torso_lean": 22.0})
        report  = self.engine.generate(profile, activity="passing")
        assert isinstance(report, CoachFeedbackReport)

    def test_report_has_summary(self):
        profile = make_profile({"accuracy": 60.0})
        report  = self.engine.generate(profile, activity="passing")
        assert report.summary

    def test_poor_metrics_produce_items(self):
        profile = make_profile({"accuracy": 55.0, "torso_lean": 25.0})
        report  = self.engine.generate(profile, activity="passing", metrics={"accuracy": 55.0, "torso_lean": 25.0})
        assert len(report.items) > 0

    def test_good_metrics_produce_positive(self):
        profile = make_profile({"accuracy": 90.0, "balance": 92.0})
        report  = self.engine.generate(profile, activity="passing", metrics={"accuracy": 90.0, "balance": 92.0})
        assert len(report.positive) > 0

    def test_each_item_has_drill(self):
        profile = make_profile({"accuracy": 55.0})
        report  = self.engine.generate(profile, activity="passing", metrics={"accuracy": 55.0})
        for item in report.items:
            assert item.adapted_drill

    def test_motivational_tip_set(self):
        profile = make_profile({"accuracy": 70.0})
        report  = self.engine.generate(profile)
        assert report.motivational_tip

    def test_all_levels_work(self):
        for level in ["Beginner", "Developing", "Intermediate", "Advanced", "Elite"]:
            profile = make_profile({"accuracy": 70.0})
            profile.level = level
            report = self.engine.generate(profile, activity="passing")
            assert report.summary
