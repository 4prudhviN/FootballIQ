#!/usr/bin/env python3
"""Tests — Analyzer Registry"""
import pytest
from analyzers.analyzer_registry import get_registry, AnalyzerRegistry
from analyzers.base_analyzer     import BaseAnalyzer

class TestAnalyzerRegistry:
    def setup_method(self): self.reg = get_registry()

    def test_singleton_returns_same_instance(self):
        assert get_registry() is get_registry()

    def test_all_six_activities_registered(self):
        actions = self.reg.registered_actions()
        for a in ["passing", "dribbling", "shooting", "goalkeeping", "defending", "movement"]:
            assert a in actions

    def test_get_returns_analyzer(self):
        analyzer = self.reg.get("passing")
        assert analyzer is not None
        assert isinstance(analyzer, BaseAnalyzer)

    def test_get_unknown_returns_none(self):
        assert self.reg.get("unknown_sport") is None

    def test_run_for_empty_activities_returns_empty(self):
        result = self.reg.run_for_activities([], [], None, None)
        assert result == {}

    def test_run_for_unknown_activity_skips_gracefully(self):
        result = self.reg.run_for_activities(["unknown_sport"], [], None, None)
        assert result == {}

    def test_each_analyzer_has_name(self):
        for name in self.reg.registered_actions():
            analyzer = self.reg.get(name)
            assert analyzer.name == name

    def test_register_new_analyzer(self):
        from schemas.activity_schema import ActionMetrics, FootballAction

        class TestAnalyzer(BaseAnalyzer):
            @property
            def name(self): return "test_action"
            def analyze(self, frames, pose, ball): return ActionMetrics(action=FootballAction.PASSING, metrics=[])

        reg = AnalyzerRegistry()
        reg.register(TestAnalyzer())
        assert "test_action" in reg.registered_actions()
