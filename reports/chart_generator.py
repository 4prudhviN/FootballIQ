#!/usr/bin/env python3
"""
Chart Generator
===============
Generates performance charts from session metric data.
Saves PNG images to reports/charts/<session_id>/.

Charts produced:
  - radar_chart.png     — skill-level radar across all metrics
  - metric_bars.png     — horizontal bar chart of metric scores
  - timeline_chart.png  — metric trend across multiple sessions (progress)

No frontend dependency — pure OpenCV + NumPy drawing.
Optional matplotlib support when available.

Usage::

    gen = ChartGenerator()
    paths = gen.generate_session_charts(session, session_id="abc123")
    print(paths)  # {"radar": Path(...), "bars": Path(...)}
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.file_utils import ensure_dir
from utils.logger     import get_logger

log = get_logger(__name__)

_CHARTS_DIR = Path(__file__).resolve().parent / "charts"

# Colour palette (BGR)
_GREEN  = (50,  220,  80)
_CYAN   = (200, 220,  50)
_RED    = (50,   50, 220)
_WHITE  = (240, 240, 240)
_GREY   = (80,   80,  80)
_BG     = (18,   18,  18)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX


class ChartGenerator:
    """Generates PNG charts from session metric data."""

    def __init__(self, charts_dir: Optional[Path] = None) -> None:
        self._base = charts_dir or _CHARTS_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_session_charts(
        self,
        session_id:    str,
        metric_scores: Dict[str, float],   # {metric_name: 0.0–1.0}
        player_level:  str = "Intermediate",
    ) -> Dict[str, Path]:
        """
        Generate all charts for a session. Returns {chart_type: path}.
        """
        out_dir = ensure_dir(self._base / session_id)
        paths:  Dict[str, Path] = {}

        try:
            radar = self._radar_chart(metric_scores, player_level, out_dir)
            paths["radar"] = radar
        except Exception as exc:
            log.warning("ChartGenerator: radar chart failed — %s", exc)

        try:
            bars = self._bar_chart(metric_scores, player_level, out_dir)
            paths["bars"] = bars
        except Exception as exc:
            log.warning("ChartGenerator: bar chart failed — %s", exc)

        log.info("ChartGenerator: %d charts for session %s", len(paths), session_id)
        return paths

    def generate_progress_chart(
        self,
        session_id:    str,
        history:       List[Dict[str, float]],   # [{metric: score}, ...]
        dates:         List[str],
        metric:        str = "overall",
    ) -> Optional[Path]:
        """Generate a line-chart showing metric progress over multiple sessions."""
        if len(history) < 2:
            return None
        out_dir = ensure_dir(self._base / session_id)
        try:
            return self._line_chart(history, dates, metric, out_dir)
        except Exception as exc:
            log.warning("ChartGenerator: progress chart failed — %s", exc)
            return None

    # ------------------------------------------------------------------
    # Radar chart
    # ------------------------------------------------------------------

    def _radar_chart(
        self,
        scores:       Dict[str, float],
        player_level: str,
        out_dir:      Path,
    ) -> Path:
        """Draw a radar/spider chart for the metric scores."""
        W, H = 500, 500
        img  = np.full((H, W, 3), _BG, dtype=np.uint8)

        labels = list(scores.keys())
        values = [max(0.0, min(1.0, scores[k])) for k in labels]
        n      = len(labels)
        if n < 3:
            # Not enough metrics for radar — fall back to bar chart.
            return self._bar_chart(scores, player_level, out_dir)

        cx, cy = W // 2, H // 2
        R      = 160

        # Draw grid rings.
        for ring in [0.25, 0.5, 0.75, 1.0]:
            pts = []
            for i in range(n):
                angle = math.pi / 2 + 2 * math.pi * i / n
                r     = ring * R
                pts.append((int(cx + r * math.cos(angle)), int(cy - r * math.sin(angle))))
            for i in range(n):
                cv2.line(img, pts[i], pts[(i + 1) % n], _GREY, 1)

        # Draw spokes.
        for i in range(n):
            angle = math.pi / 2 + 2 * math.pi * i / n
            end   = (int(cx + R * math.cos(angle)), int(cy - R * math.sin(angle)))
            cv2.line(img, (cx, cy), end, _GREY, 1)

        # Draw metric polygon.
        pts_data = []
        for i, v in enumerate(values):
            angle = math.pi / 2 + 2 * math.pi * i / n
            r     = v * R
            pts_data.append((int(cx + r * math.cos(angle)), int(cy - r * math.sin(angle))))

        poly = np.array(pts_data, dtype=np.int32)
        cv2.fillPoly(img, [poly], (*_GREEN[::-1], 60))   # semi-transparent fill
        cv2.polylines(img, [poly], True, _GREEN, 2)

        # Draw dots and labels.
        for i, (pt, label, val) in enumerate(zip(pts_data, labels, values)):
            cv2.circle(img, pt, 5, _GREEN, -1)
            angle  = math.pi / 2 + 2 * math.pi * i / n
            lx     = int(cx + (R + 30) * math.cos(angle))
            ly     = int(cy - (R + 30) * math.sin(angle))
            short  = label.replace("_", " ").title()[:12]
            cv2.putText(img, short, (lx - 40, ly + 5), _FONT, 0.38, _WHITE, 1, cv2.LINE_AA)

        # Title.
        title = f"Skill Radar — {player_level}"
        cv2.putText(img, title, (W // 2 - 90, 30), _FONT, 0.55, _CYAN, 1, cv2.LINE_AA)

        path = out_dir / "radar_chart.png"
        cv2.imwrite(str(path), img)
        return path

    # ------------------------------------------------------------------
    # Bar chart
    # ------------------------------------------------------------------

    def _bar_chart(
        self,
        scores:       Dict[str, float],
        player_level: str,
        out_dir:      Path,
    ) -> Path:
        """Draw a horizontal bar chart of metric scores."""
        n    = len(scores)
        W    = 500
        H    = max(200, 60 + n * 50)
        img  = np.full((H, W, 3), _BG, dtype=np.uint8)

        # Title.
        cv2.putText(img, f"Metric Scores — {player_level}", (14, 28),
                    _FONT, 0.55, _CYAN, 1, cv2.LINE_AA)

        bar_x     = 170
        bar_max_w = 280
        y_start   = 55

        for i, (metric, score) in enumerate(scores.items()):
            y     = y_start + i * 50
            label = metric.replace("_", " ").title()[:20]
            val   = max(0.0, min(1.0, score))

            # Label.
            cv2.putText(img, label, (10, y + 14), _FONT, 0.42, _WHITE, 1, cv2.LINE_AA)

            # Background bar.
            cv2.rectangle(img, (bar_x, y), (bar_x + bar_max_w, y + 24), _GREY, -1)

            # Value bar — colour based on score.
            bar_w = int(val * bar_max_w)
            colour = _GREEN if val >= 0.70 else _CYAN if val >= 0.35 else _RED
            if bar_w > 0:
                cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + 24), colour, -1)

            # Score text.
            cv2.putText(img, f"{val:.0%}", (bar_x + bar_max_w + 8, y + 17),
                        _FONT, 0.42, _WHITE, 1, cv2.LINE_AA)

        path = out_dir / "metric_bars.png"
        cv2.imwrite(str(path), img)
        return path

    # ------------------------------------------------------------------
    # Line chart (progress over sessions)
    # ------------------------------------------------------------------

    def _line_chart(
        self,
        history: List[Dict[str, float]],
        dates:   List[str],
        metric:  str,
        out_dir: Path,
    ) -> Path:
        """Draw a line chart of one metric across multiple sessions."""
        W, H   = 500, 300
        img    = np.full((H, W, 3), _BG, dtype=np.uint8)
        values = [h.get(metric, 0.0) for h in history]
        n      = len(values)

        pad_l, pad_r, pad_t, pad_b = 60, 30, 40, 50
        chart_w = W - pad_l - pad_r
        chart_h = H - pad_t - pad_b

        # Axes.
        cv2.line(img, (pad_l, pad_t), (pad_l, H - pad_b), _GREY, 1)
        cv2.line(img, (pad_l, H - pad_b), (W - pad_r, H - pad_b), _GREY, 1)

        # Y grid lines at 0.25, 0.5, 0.75, 1.0.
        for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = int(H - pad_b - tick * chart_h)
            cv2.line(img, (pad_l - 4, y), (W - pad_r, y), _GREY, 1)
            cv2.putText(img, f"{tick:.0%}", (4, y + 4), _FONT, 0.35, _GREY, 1)

        # Plot line.
        step = chart_w / max(1, n - 1)
        pts  = []
        for i, v in enumerate(values):
            x = int(pad_l + i * step)
            y = int(H - pad_b - max(0.0, min(1.0, v)) * chart_h)
            pts.append((x, y))

        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], _GREEN, 2)

        for i, pt in enumerate(pts):
            cv2.circle(img, pt, 5, _GREEN, -1)
            short_date = dates[i][:5] if i < len(dates) else str(i)
            cv2.putText(img, short_date, (pt[0] - 15, H - pad_b + 18),
                        _FONT, 0.32, _WHITE, 1)

        title = f"{metric.replace('_', ' ').title()} Progress"
        cv2.putText(img, title, (pad_l, 26), _FONT, 0.52, _CYAN, 1, cv2.LINE_AA)

        path = out_dir / f"progress_{metric}.png"
        cv2.imwrite(str(path), img)
        return path
