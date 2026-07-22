#!/usr/bin/env python3
"""
File Utilities
==============
File system helpers for path management, safe I/O, and cleanup.
Import from here — never re-implement in another module.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_job_dir(base_dir: str | Path) -> tuple[str, Path]:
    """
    Create a unique job sub-directory inside base_dir.
    Returns (job_id, job_path).
    """
    job_id  = uuid.uuid4().hex[:12]
    job_dir = ensure_dir(Path(base_dir) / job_id)
    return job_id, job_dir


def safe_remove(path: str | Path) -> bool:
    """Delete a file silently. Returns True if deleted, False if missing."""
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError:
        return False


def safe_remove_dir(path: str | Path) -> bool:
    """Recursively delete a directory silently. Returns True on success."""
    try:
        shutil.rmtree(path, ignore_errors=True)
        return True
    except OSError:
        return False


def file_size_mb(path: str | Path) -> float:
    """Return file size in megabytes, or 0.0 if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024 * 1024)


def file_exists(path: str | Path) -> bool:
    return Path(path).exists()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def read_json(path: str | Path) -> Any:
    """
    Read and parse a JSON file.

    Returns
    -------
    Parsed object (dict, list, etc.), or an empty dict on error.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def write_json(path: str | Path, data: Any, indent: int = 2) -> bool:
    """
    Write data to a JSON file.

    Returns
    -------
    True on success, False on failure.
    """
    try:
        ensure_dir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Temp file helpers
# ---------------------------------------------------------------------------

def make_temp_path(suffix: str = ".mp4", prefix: str = "footballiq_") -> str:
    """Return a path for a temporary file (not yet created)."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    return path


def save_bytes(data: bytes, path: str | Path) -> Path:
    """Write raw bytes to a file, creating parent directories as needed."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

VALID_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def is_valid_video_file(path: str | Path) -> bool:
    """Return True if path exists and has a supported video extension."""
    p = Path(path)
    return p.exists() and p.suffix.lower() in VALID_VIDEO_EXTENSIONS


def assert_file_exists(path: str | Path) -> Path:
    """Raise FileNotFoundError if the file doesn't exist. Return Path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")
    return p
