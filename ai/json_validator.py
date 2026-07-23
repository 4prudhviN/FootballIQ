#!/usr/bin/env python3
"""
JSON Validator  ⭐
==================
Validates and repairs every LLM response before the frontend uses it.

The LLM can return:
  - Valid JSON                 → pass through
  - Valid JSON with wrong keys → repair missing fields with safe defaults
  - Malformed JSON             → attempt repair via regex extraction
  - Pure text (no JSON)        → extract structured data from text

Every response that reaches the frontend is guaranteed to have the
correct shape — never a crash, never a missing field.

Usage::

    validator = JSONValidator()

    # Validate a raw LLM response string
    result = validator.validate(raw_llm_text)
    if result.valid:
        print(result.data)           # clean dict
    else:
        print(result.repaired_data)  # best-effort repair
        print(result.errors)         # what was wrong

    # Validate against a known schema
    result = validator.validate_feedback_response(raw_llm_text)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Expected schemas (field → default value)
# ---------------------------------------------------------------------------

# The shape the frontend expects for aiFeedback.
FEEDBACK_SCHEMA: Dict[str, Any] = {
    "summary":         "",
    "strengths":       [],
    "weaknesses":      [],
    "coachingTips":    [],
    "motivationalTip": "",
}

# The shape the frontend expects for a drill.
DRILL_SCHEMA: Dict[str, Any] = {
    "name":         "Drill",
    "targetMetric": "",
    "instructions": "",
    "coachTip":     "",
    "duration":     "10 min",
    "difficulty":   "Beginner",
}

# The full report schema returned by /api/upload-video.
REPORT_SCHEMA: Dict[str, Any] = {
    "status":             "complete",
    "job_id":             "",
    "video_url":          "",
    "detectedActivities": [],
    "playerLevel":        "Beginner",
    "metrics": {
        "byAction":      {},
        "torsoLean":     0.0,
        "kneeStability": 0.0,
        "gaitSymmetry":  0.0,
        "warnings":      [],
    },
    "aiFeedback": FEEDBACK_SCHEMA.copy(),
    "drills":             [],
}


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a JSON validation + repair attempt."""
    valid:         bool
    data:          Optional[Dict[str, Any]] = None    # original parsed data
    repaired_data: Optional[Dict[str, Any]] = None    # repaired / defaults-filled
    errors:        List[str]                = field(default_factory=list)
    warnings:      List[str]               = field(default_factory=list)
    was_repaired:  bool                    = False

    @property
    def best(self) -> Optional[Dict[str, Any]]:
        """Return repaired_data if available, else data."""
        return self.repaired_data or self.data


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class JSONValidator:
    """
    Validates and repairs LLM JSON responses.

    Methods
    -------
    validate(text)
        Generic JSON extraction and validation from any text.
    validate_feedback_response(text)
        Validate against the aiFeedback schema.
    validate_report_response(text)
        Validate against the full report schema.
    repair(data, schema)
        Fill missing fields from a schema with safe defaults.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, text: str) -> ValidationResult:
        """
        Extract and validate JSON from any LLM response string.

        Attempts in order:
          1. Direct json.loads()
          2. Extract JSON object via regex
          3. Extract JSON array via regex
          4. Return empty dict with errors

        Parameters
        ----------
        text : str — raw LLM response

        Returns
        -------
        ValidationResult
        """
        if not text or not text.strip():
            return ValidationResult(
                valid=False,
                errors=["LLM response is empty."],
            )

        # Attempt 1: direct parse.
        try:
            data = json.loads(text.strip())
            return ValidationResult(valid=True, data=data)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract first {...} block.
        extracted = self._extract_json_object(text)
        if extracted is not None:
            log.debug("JSONValidator: repaired via JSON object extraction")
            return ValidationResult(
                valid=True,
                data=extracted,
                was_repaired=True,
                warnings=["JSON extracted from surrounding text."],
            )

        # Attempt 3: extract first [...] block.
        extracted_list = self._extract_json_array(text)
        if extracted_list is not None:
            log.debug("JSONValidator: repaired via JSON array extraction")
            return ValidationResult(
                valid=True,
                data={"items": extracted_list},
                was_repaired=True,
                warnings=["JSON array extracted and wrapped in {'items': ...}."],
            )

        # Attempt 4: extract key-value pairs from plain text.
        extracted_kv = self._extract_key_values(text)
        if extracted_kv:
            log.debug("JSONValidator: repaired via key-value extraction")
            return ValidationResult(
                valid=False,
                repaired_data=extracted_kv,
                was_repaired=True,
                errors=["No valid JSON found — key-value pairs extracted from text."],
                warnings=["Data quality may be degraded."],
            )

        return ValidationResult(
            valid=False,
            errors=["Could not extract any structured data from LLM response."],
        )

    def validate_feedback_response(self, text: str) -> ValidationResult:
        """
        Parse and validate an LLM response against the aiFeedback schema.
        Missing fields are filled with safe defaults.
        """
        result = self.validate(text)
        data   = result.best or {}

        repaired, repair_warnings = self.repair(data, FEEDBACK_SCHEMA)

        # Type coercion.
        repaired = self._coerce_feedback(repaired)

        result.repaired_data = repaired
        result.warnings.extend(repair_warnings)
        if repair_warnings:
            result.was_repaired = True

        return result

    def validate_report_response(self, text: str) -> ValidationResult:
        """
        Parse and validate an LLM response against the full report schema.
        """
        result = self.validate(text)
        data   = result.best or {}

        repaired, repair_warnings = self.repair(data, REPORT_SCHEMA)

        result.repaired_data = repaired
        result.warnings.extend(repair_warnings)
        if repair_warnings:
            result.was_repaired = True

        return result

    def validate_drills(self, drills_raw: Any) -> List[Dict[str, Any]]:
        """
        Validate and repair a list of drill dicts.
        Always returns a list — never crashes.
        """
        if not isinstance(drills_raw, list):
            log.warning("JSONValidator.validate_drills: expected list, got %s", type(drills_raw))
            return []

        cleaned: List[Dict[str, Any]] = []
        for i, item in enumerate(drills_raw):
            if not isinstance(item, dict):
                log.warning("JSONValidator.validate_drills: item %d is not a dict", i)
                continue
            repaired, _ = self.repair(item, DRILL_SCHEMA)
            cleaned.append(repaired)

        return cleaned

    @staticmethod
    def repair(
        data:   Dict[str, Any],
        schema: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Fill any missing or None fields in `data` with defaults from `schema`.

        Parameters
        ----------
        data   : dict — possibly incomplete data
        schema : dict — expected fields with default values

        Returns
        -------
        (repaired_dict, list_of_repair_warnings)
        """
        warnings: List[str] = []
        repaired  = dict(data)   # shallow copy

        for key, default in schema.items():
            if key not in repaired or repaired[key] is None:
                repaired[key] = default
                warnings.append(f"Field '{key}' was missing — set to default.")
            elif isinstance(default, dict) and isinstance(repaired[key], dict):
                # Recurse into nested dicts.
                repaired[key], nested_w = JSONValidator.repair(repaired[key], default)
                warnings.extend(nested_w)

        return repaired, warnings

    # ------------------------------------------------------------------
    # Private — extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Find and parse the first {...} block in text."""
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_json_array(text: str) -> Optional[List[Any]]:
        """Find and parse the first [...] block in text."""
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return None
        try:
            result = json.loads(match.group())
            return result if isinstance(result, list) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_key_values(text: str) -> Dict[str, Any]:
        """
        Extract key: value pairs from plain text as a last resort.
        Handles patterns like 'Summary: ...' or 'summary: ...'.
        """
        result: Dict[str, Any] = {}
        known_keys = [
            "summary", "strengths", "weaknesses", "coachingTips",
            "motivationalTip", "playerLevel", "detectedActivities",
        ]
        for key in known_keys:
            pattern = re.compile(
                rf'{key}\s*[:\-]\s*(.+?)(?=\n[A-Za-z]|\Z)',
                re.IGNORECASE | re.DOTALL,
            )
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                # Try to parse as JSON list.
                try:
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    result[key] = value
        return result

    @staticmethod
    def _coerce_feedback(data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure list fields are lists and string fields are strings."""
        list_fields   = ["strengths", "weaknesses", "coachingTips"]
        string_fields = ["summary", "motivationalTip"]

        for f in list_fields:
            val = data.get(f)
            if isinstance(val, str):
                data[f] = [val] if val else []
            elif not isinstance(val, list):
                data[f] = []

        for f in string_fields:
            val = data.get(f)
            if not isinstance(val, str):
                data[f] = str(val) if val is not None else ""

        return data
