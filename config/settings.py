#!/usr/bin/env python3
"""
Settings
========
Runtime configuration loaded from environment variables and .env file.
All env-dependent values live here — never call os.getenv() in other modules.

Usage::

    from config.settings import settings
    print(settings.LLM_PROVIDER)
    print(settings.MODEL_PATH)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file).
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """
    All runtime-configurable settings for the FootballIQ backend.
    Values are read from environment variables at import time.
    Defaults are safe for local development.
    """

    # ── Server ───────────────────────────────────────────────────────────
    HOST:              str   = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT:              int   = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    DEBUG:             bool  = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    WORK_DIR:          str   = field(default_factory=lambda: os.getenv("WORK_DIR", "backend_temp"))

    # ── Video processing ──────────────────────────────────────────────────
    FPS:               float = field(default_factory=lambda: float(os.getenv("FPS", "25.0")))
    FRAME_STRIDE:      int   = field(default_factory=lambda: int(os.getenv("FRAME_STRIDE", "3")))
    MAX_FILE_SIZE_MB:  float = field(default_factory=lambda: float(os.getenv("MAX_FILE_SIZE_MB", "500")))
    MAX_DURATION_S:    float = field(default_factory=lambda: float(os.getenv("MAX_DURATION_S", "300")))

    # ── Model paths ───────────────────────────────────────────────────────
    # Path to a custom YOLO weights file (optional — falls back to Hough).
    MODEL_PATH:        str   = field(default_factory=lambda: os.getenv("MODEL_PATH", ""))
    POSE_MODEL_COMPLEXITY: int = field(default_factory=lambda: int(os.getenv("POSE_MODEL_COMPLEXITY", "1")))

    # ── LLM / AI ──────────────────────────────────────────────────────────
    LLM_PROVIDER:      str   = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    GEMINI_API_KEY:    str   = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))
    OPENAI_API_KEY:    str   = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    FIREWORKS_API_KEY: str   = field(default_factory=lambda: os.getenv("FIREWORKS_API_KEY", ""))
    GEMINI_MODEL:      str   = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    OPENAI_MODEL:      str   = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    FIREWORKS_MODEL:   str   = field(default_factory=lambda: os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"))
    LLM_MAX_TOKENS:    int   = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024")))
    LLM_TEMPERATURE:   float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.4")))
    LLM_MAX_RETRIES:   int   = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")))
    LLM_TIMEOUT_S:     float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "30")))

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL:         str   = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    LOG_FILE:          str   = field(default_factory=lambda: os.getenv("LOG_FILE", ""))
    LOG_MAX_MB:        int   = field(default_factory=lambda: int(os.getenv("LOG_MAX_MB", "10")))
    LOG_BACKUPS:       int   = field(default_factory=lambda: int(os.getenv("LOG_BACKUPS", "3")))

    # ── CORS / Frontend ───────────────────────────────────────────────────
    FRONTEND_URL:      str   = field(default_factory=lambda: os.getenv("FRONTEND_URL", "http://localhost:5173"))
    ALLOW_ALL_ORIGINS: bool  = field(default_factory=lambda: os.getenv("ALLOW_ALL_ORIGINS", "true").lower() == "true")

    # ── Calibration ───────────────────────────────────────────────────────
    PX_PER_METER:      float = field(default_factory=lambda: float(os.getenv("PX_PER_METER", "100.0")))

    # ------------------------------------------------------------------
    # Derived / computed properties
    # ------------------------------------------------------------------

    @property
    def work_dir_path(self) -> Path:
        """Resolved absolute path to the working directory."""
        return Path(self.WORK_DIR).resolve()

    @property
    def model_path_or_none(self) -> Path | None:
        """Return model path as Path if set, else None."""
        return Path(self.MODEL_PATH) if self.MODEL_PATH else None

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.GEMINI_API_KEY)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.OPENAI_API_KEY)

    @property
    def has_fireworks_key(self) -> bool:
        return bool(self.FIREWORKS_API_KEY)

    @property
    def active_llm_key(self) -> str:
        """Return the API key for the currently configured LLM provider."""
        return {
            "gemini":    self.GEMINI_API_KEY,
            "openai":    self.OPENAI_API_KEY,
            "fireworks": self.FIREWORKS_API_KEY,
        }.get(self.LLM_PROVIDER, "")

    @property
    def active_llm_model(self) -> str:
        """Return the model name for the currently configured LLM provider."""
        return {
            "gemini":    self.GEMINI_MODEL,
            "openai":    self.OPENAI_MODEL,
            "fireworks": self.FIREWORKS_MODEL,
        }.get(self.LLM_PROVIDER, "")

    def summary(self) -> dict:
        """Return a safe (no secrets) summary dict for health-check endpoints."""
        return {
            "host":               self.HOST,
            "port":               self.PORT,
            "debug":              self.DEBUG,
            "fps":                self.FPS,
            "frame_stride":       self.FRAME_STRIDE,
            "llm_provider":       self.LLM_PROVIDER,
            "llm_model":          self.active_llm_model,
            "has_llm_key":        bool(self.active_llm_key),
            "pose_complexity":    self.POSE_MODEL_COMPLEXITY,
            "model_path":         self.MODEL_PATH or "(default)",
            "log_level":          self.LOG_LEVEL,
        }


# ---------------------------------------------------------------------------
# Singleton instance — import this everywhere
# ---------------------------------------------------------------------------

settings = Settings()
