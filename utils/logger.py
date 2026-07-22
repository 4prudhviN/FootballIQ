#!/usr/bin/env python3
"""
Logger
======
Centralised logging for the FootballIQ backend.
Every module imports and uses this logger — no print() calls in production.

Usage::

    from utils.logger import get_logger
    log = get_logger(__name__)

    log.info("Pipeline started")
    log.warning("Ball not detected")
    log.error("Pose estimation failed: %s", exc)
    log.pipeline("player_detect", "Player detected — confidence 0.87")
    log.timing("pose_estimate", 1.24)
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration (from environment / defaults)
# ---------------------------------------------------------------------------

LOG_LEVEL   = os.getenv("LOG_LEVEL",   "INFO").upper()
LOG_FILE    = os.getenv("LOG_FILE",    "")           # empty = console only
LOG_MAX_MB  = int(os.getenv("LOG_MAX_MB",  "10"))
LOG_BACKUPS = int(os.getenv("LOG_BACKUPS", "3"))

# ANSI colour codes for console output.
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}

_USE_COLOUR = sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Custom log levels
# ---------------------------------------------------------------------------

PIPELINE_LEVEL = 25   # between DEBUG (10) and INFO (20) — but we want it above INFO
TIMING_LEVEL   = 15   # between DEBUG and INFO

logging.addLevelName(PIPELINE_LEVEL, "PIPELINE")
logging.addLevelName(TIMING_LEVEL,   "TIMING")


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class _ColourFormatter(logging.Formatter):
    """Console formatter with ANSI colour coding per level."""

    FMT = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
    DATEFMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if _USE_COLOUR:
            colour = _COLOURS.get(record.levelname, "")
            reset  = _COLOURS["RESET"]
            return f"{colour}{msg}{reset}"
        return msg


# ---------------------------------------------------------------------------
# FootballIQ Logger
# ---------------------------------------------------------------------------

class FootballIQLogger(logging.Logger):
    """
    Extended logger with pipeline-specific convenience methods.
    """

    def pipeline(self, stage: str, message: str, *args, **kwargs) -> None:
        """Log a pipeline stage transition."""
        if self.isEnabledFor(PIPELINE_LEVEL):
            self._log(PIPELINE_LEVEL, f"[{stage.upper()}] {message}", args, **kwargs)

    def timing(self, stage: str, seconds: float) -> None:
        """Log stage execution time."""
        if self.isEnabledFor(TIMING_LEVEL):
            self._log(TIMING_LEVEL, f"[TIMING] {stage} — {seconds:.3f}s", ())


# ---------------------------------------------------------------------------
# Root logger setup (called once at import time)
# ---------------------------------------------------------------------------

logging.setLoggerClass(FootballIQLogger)

_root = logging.getLogger("footballiq")
_root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Console handler.
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(_ColourFormatter(
    fmt     = _ColourFormatter.FMT,
    datefmt = _ColourFormatter.DATEFMT,
))
_root.addHandler(_console)

# File handler (optional).
if LOG_FILE:
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    _file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes    = LOG_MAX_MB * 1024 * 1024,
        backupCount = LOG_BACKUPS,
        encoding    = "utf-8",
    )
    _file_handler.setFormatter(logging.Formatter(
        fmt     = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    ))
    _root.addHandler(_file_handler)

# Suppress noisy third-party loggers.
for _noisy in ("urllib3", "httpx", "httpcore", "mediapipe", "absl"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> FootballIQLogger:
    """
    Return a FootballIQ logger for the given module name.

    Usage::

        from utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Server started on port 8000")
        log.pipeline("pose_estimate", "Processing 240 frames")
        log.timing("pose_estimate", 1.45)
    """
    return logging.getLogger(f"footballiq.{name}")  # type: ignore[return-value]
