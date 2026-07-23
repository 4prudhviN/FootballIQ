#!/usr/bin/env python3
"""
coach_engine
============
Deterministic coaching logic — no LLM invention.

All coaching output is grounded in the football_knowledge/ database
and the built-in rule sets. The LLM is only used for natural-language
wrapping AFTER the coach_engine has determined what to say.

Public API
----------
    from coach_engine import (
        CoachSkillClassifier, CoachSkillProfile,
        CoachFeedbackEngine, CoachFeedbackReport,
        DrillRecommender, DrillRecommendation,
        TerminologyAdapter,
        ProgressEstimator, ProgressReport,
    )
"""

from coach_engine.skill_classifier    import CoachSkillClassifier, CoachSkillProfile
from coach_engine.feedback_engine     import CoachFeedbackEngine, CoachFeedbackReport, FeedbackItem
from coach_engine.drill_recommender   import DrillRecommender, DrillRecommendation
from coach_engine.terminology_adapter import TerminologyAdapter
from coach_engine.progress_estimator  import ProgressEstimator, ProgressReport

__all__ = [
    "CoachSkillClassifier",
    "CoachSkillProfile",
    "CoachFeedbackEngine",
    "CoachFeedbackReport",
    "FeedbackItem",
    "DrillRecommender",
    "DrillRecommendation",
    "TerminologyAdapter",
    "ProgressEstimator",
    "ProgressReport",
]
