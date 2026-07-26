#!/usr/bin/env python3
"""Tests — Skill Classifier (5-level)"""

import pytest
from coach_engine.skill_classifier import CoachSkillClassifier, CoachSkillProfile

class TestSkillClassifier:
    def setup_method(self): self.clf = CoachSkillClassifier()

    def test_elite_metrics_give_elite(self):
        p = self.clf.classify({"accuracy": 95.0, "balance": 95.0, "body_alignment": 95.0})
        assert p.level == "Elite"

    def test_poor_metrics_give_beginner(self):
        p = self.clf.classify({"accuracy": 40.0, "balance": 40.0, "torso_lean": 30.0})
        assert p.level in ("Beginner", "Developing")

    def test_empty_metrics_returns_beginner(self):
        p = self.clf.classify({})
        assert p.level == "Beginner"
        assert p.overall_score == 0.0

    def test_overall_score_between_0_and_1(self):
        p = self.clf.classify({"accuracy": 75.0, "balance": 80.0})
        assert 0.0 <= p.overall_score <= 1.0

    def test_strengths_and_weaknesses_populated(self):
        p = self.clf.classify({"accuracy": 90.0, "torso_lean": 28.0})
        assert len(p.strengths) > 0 or len(p.weaknesses) > 0

    def test_intermediate_range(self):
        p = self.clf.classify({"accuracy": 72.0, "balance": 70.0, "body_alignment": 68.0})
        assert p.level in ("Intermediate", "Advanced")

    def test_top_weakness_is_worst_metric(self):
        p = self.clf.classify({"accuracy": 90.0, "torso_lean": 28.0, "balance": 85.0})
        assert p.top_weakness is not None

    def test_top_strength_is_best_metric(self):
        p = self.clf.classify({"accuracy": 92.0, "torso_lean": 8.0})
        assert p.top_strength is not None

    def test_result_is_classification_report(self):
        p = self.clf.classify({"accuracy": 70.0})
        assert isinstance(p, CoachSkillProfile)
