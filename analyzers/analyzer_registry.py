#!/usr/bin/env python3
"""
Analyzer Registry
=================
Central registry that maps FootballAction values to their concrete
BaseAnalyzer instances.

The module-level ``registry`` singleton is pre-populated with all six
concrete analyzers at import time.

Usage::

    from analyzers.analyzer_registry import get_registry

    registry = get_registry()
    results  = registry.run_for_activities(
        ["shooting", "passing"],
        frames,
        pose_result,
        ball_result,
    )
    # results → {"shooting": ActionMetrics(...), "passing": ActionMetrics(...)}
"""

from __future__ import annotations

from typing import Dict, List, Optional

from analyzers.base_analyzer import BaseAnalyzer
from schemas.activity_schema import ActionMetrics, FootballAction
from pipeline.frame_extractor import ExtractedFrame
from pipeline.pose_estimator import PoseEstimationResult
from pipeline.ball_detector import BallDetectionResult
from utils.logger import get_logger

log = get_logger(__name__)


class AnalyzerRegistry:
    """
    Registry that maps FootballAction strings to BaseAnalyzer instances.

    Methods
    -------
    register(analyzer)
        Add a concrete analyzer to the registry, keyed by its ``name``.
    get(action) -> BaseAnalyzer | None
        Retrieve the analyzer for a given FootballAction (string or enum).
    run_for_activities(activities, frames, pose_result, ball_result)
        Run all registered analyzers whose actions appear in ``activities``.
        Skips activities with no registered analyzer gracefully.
        Returns ``{activity_name: ActionMetrics}`` dict.
    """

    def __init__(self) -> None:
        self._analyzers: Dict[str, BaseAnalyzer] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, analyzer: BaseAnalyzer) -> None:
        """
        Register a concrete analyzer.

        Parameters
        ----------
        analyzer : BaseAnalyzer
            An instance whose ``name`` property returns the FootballAction
            value string (e.g. ``"passing"``).
        """
        key = analyzer.name
        if key in self._analyzers:
            log.warning("AnalyzerRegistry: overwriting existing analyzer for '%s'", key)
        self._analyzers[key] = analyzer
        log.debug("AnalyzerRegistry: registered analyzer for '%s'", key)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, action: "str | FootballAction") -> Optional[BaseAnalyzer]:
        """
        Return the analyzer registered for ``action``, or ``None``.

        Parameters
        ----------
        action : str | FootballAction
            A FootballAction enum member or its string value (e.g. ``"passing"``).
        """
        key = action.value if isinstance(action, FootballAction) else str(action)
        return self._analyzers.get(key)

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def run_for_activities(
        self,
        activities:  List[str],
        frames:      List[ExtractedFrame],
        pose_result: PoseEstimationResult,
        ball_result: BallDetectionResult,
    ) -> Dict[str, ActionMetrics]:
        """
        Run analyzers for each detected activity and return results.

        Parameters
        ----------
        activities : list[str]
            Activity name strings as produced by ActivityDetector
            (e.g. ``["shooting", "passing"]``).
        frames : list[ExtractedFrame]
            Sampled video frames.
        pose_result : PoseEstimationResult
        ball_result : BallDetectionResult

        Returns
        -------
        dict[str, ActionMetrics]
            Maps each activity name to its computed ActionMetrics.
            Activities with no registered analyzer are silently skipped.
        """
        results: Dict[str, ActionMetrics] = {}

        for activity in activities:
            analyzer = self.get(activity)
            if analyzer is None:
                log.debug(
                    "AnalyzerRegistry: no analyzer registered for '%s' — skipping",
                    activity,
                )
                continue

            try:
                log.debug("AnalyzerRegistry: running analyzer for '%s'", activity)
                metrics = analyzer.analyze(frames, pose_result, ball_result)
                results[activity] = metrics
                log.debug(
                    "AnalyzerRegistry: '%s' produced %d metrics",
                    activity,
                    len(metrics.metrics),
                )
            except Exception as exc:  # noqa: BLE001
                # Each concrete analyzer should never raise, but belt-and-braces.
                log.error(
                    "AnalyzerRegistry: unhandled error in '%s' analyzer — %s",
                    activity,
                    exc,
                )

        return results

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def registered_actions(self) -> List[str]:
        """Return the list of action strings that have a registered analyzer."""
        return list(self._analyzers.keys())

    def __repr__(self) -> str:
        return f"AnalyzerRegistry(registered={self.registered_actions()})"


# ---------------------------------------------------------------------------
# Module-level singleton — pre-populated with all concrete analyzers
# ---------------------------------------------------------------------------

# Import concrete analyzers here to avoid circular imports at the module level.
from analyzers.passing     import PassingAnalyzer       # noqa: E402
from analyzers.dribbling   import DribblingAnalyzer     # noqa: E402
from analyzers.shooting    import ShootingAnalyzer      # noqa: E402
from analyzers.goalkeeping import GoalkeepingAnalyzer   # noqa: E402
from analyzers.defending   import DefendingAnalyzer     # noqa: E402
from analyzers.movement    import MovementAnalyzer      # noqa: E402

registry = AnalyzerRegistry()
registry.register(PassingAnalyzer())
registry.register(DribblingAnalyzer())
registry.register(ShootingAnalyzer())
registry.register(GoalkeepingAnalyzer())
registry.register(DefendingAnalyzer())
registry.register(MovementAnalyzer())

log.debug("AnalyzerRegistry singleton initialised: %s", registry)


def get_registry() -> AnalyzerRegistry:
    """
    Return the shared AnalyzerRegistry singleton.

    Returns
    -------
    AnalyzerRegistry
        The pre-populated module-level registry instance.

    Example
    -------
    ::

        from analyzers.analyzer_registry import get_registry

        reg     = get_registry()
        results = reg.run_for_activities(
            ["shooting", "passing"],
            frames,
            pose_result,
            ball_result,
        )
    """
    return registry
