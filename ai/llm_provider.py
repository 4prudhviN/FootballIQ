#!/usr/bin/env python3
"""
LLM Provider
============
Sends a prompt to the configured LLM and returns the raw text response.

Responsibilities:
  - Load API key from environment / .env
  - Call the LLM API (Google Gemini by default)
  - Handle errors, retries, and timeouts
  - Return raw response text to the caller
  - Fall back to a deterministic offline response if the API is unavailable

Supported providers (set LLM_PROVIDER in .env):
  gemini    — Google Gemini (default, uses google-generativeai)
  openai    — OpenAI ChatGPT  (requires openai package)
  fireworks — Fireworks AI    (requires requests)
  offline   — No API call, returns structured placeholder text

Pipeline position:
  Prompt → LLM → raw text → ReportGenerator
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (read from environment / .env)
# ---------------------------------------------------------------------------

LLM_PROVIDER    = os.getenv("LLM_PROVIDER",    "gemini")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY",  os.getenv("GOOGLE_API_KEY", ""))
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "")
FIREWORKS_KEY   = os.getenv("FIREWORKS_API_KEY", "")

GEMINI_MODEL    = os.getenv("GEMINI_MODEL",    "gemini-1.5-flash")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",    "gpt-4o-mini")
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct")

MAX_RETRIES     = int(os.getenv("LLM_MAX_RETRIES", "2"))
TIMEOUT_S       = float(os.getenv("LLM_TIMEOUT_S", "30"))
MAX_TOKENS      = int(os.getenv("LLM_MAX_TOKENS",  "1024"))
TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.4"))


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Wrapper around the raw LLM response."""
    text:         str
    provider:     str
    model:        str
    tokens_used:  Optional[int] = None
    latency_s:    float         = 0.0
    from_fallback: bool         = False


# ---------------------------------------------------------------------------
# Offline fallback
# ---------------------------------------------------------------------------

_OFFLINE_TEMPLATE = """\
1. SUMMARY
Your session showed {level}-level performance across {activities}. \
The biomechanical data highlights specific areas for improvement in posture and movement efficiency.

2. STRENGTHS
• Consistent engagement with the activity throughout the session.
• Movement patterns recorded across multiple frames indicating active participation.

3. AREAS TO IMPROVE
• Torso alignment at contact requires attention based on lean measurements.
• Lower-body stability metrics indicate room for improvement in knee tracking.

4. TRAINING DRILLS
Wall Lean Drill | Stand 30 cm from a wall, drive knee up without back touching wall | 10 min
Lateral Band Walk | Resistance band above knees, 20 side steps each direction | 8 min

5. COACH TIP
Focus on keeping your chest over the ball at the moment of contact — \
this single correction will improve both accuracy and power immediately.
"""


def _offline_response(prompt: str, level: str = "Intermediate",
                      activities: str = "general movement") -> LLMResponse:
    text = _OFFLINE_TEMPLATE.format(level=level, activities=activities)
    return LLMResponse(
        text=text, provider="offline", model="none",
        from_fallback=True, latency_s=0.0,
    )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str) -> LLMResponse:
    """Call Google Gemini via google-generativeai."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to your .env file."
        )

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=genai.types.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        ),
    )

    t0       = time.perf_counter()
    response = model.generate_content(prompt)
    latency  = time.perf_counter() - t0

    return LLMResponse(
        text      = response.text,
        provider  = "gemini",
        model     = GEMINI_MODEL,
        latency_s = round(latency, 3),
    )


def _call_openai(prompt: str) -> LLMResponse:
    """Call OpenAI ChatCompletion."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai not installed. Run: pip install openai")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model       = OPENAI_MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS,
    )
    latency = time.perf_counter() - t0

    return LLMResponse(
        text         = resp.choices[0].message.content or "",
        provider     = "openai",
        model        = OPENAI_MODEL,
        tokens_used  = resp.usage.total_tokens if resp.usage else None,
        latency_s    = round(latency, 3),
    )


def _call_fireworks(prompt: str) -> LLMResponse:
    """Call Fireworks AI via REST API."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests not installed. Run: pip install requests")

    if not FIREWORKS_KEY:
        raise ValueError("FIREWORKS_API_KEY not set.")

    url     = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {FIREWORKS_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       FIREWORKS_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
    }

    t0   = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_S)
    resp.raise_for_status()
    latency = time.perf_counter() - t0

    data = resp.json()
    text = data["choices"][0]["message"]["content"]

    return LLMResponse(
        text      = text,
        provider  = "fireworks",
        model     = FIREWORKS_MODEL,
        latency_s = round(latency, 3),
    )


# ---------------------------------------------------------------------------
# Main provider dispatcher
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "gemini":    _call_gemini,
    "openai":    _call_openai,
    "fireworks": _call_fireworks,
}


class LLMProvider:
    """
    Send a prompt to the configured LLM and return the response.

    Parameters
    ----------
    provider : str | None
        Override the LLM_PROVIDER env variable.
    max_retries : int
        Number of retry attempts on transient errors.

    Usage::

        provider = LLMProvider()
        response = provider.call(prompt)
        print(response.text)
    """

    def __init__(
        self,
        provider:    Optional[str] = None,
        max_retries: int            = MAX_RETRIES,
    ) -> None:
        self.provider    = (provider or LLM_PROVIDER).lower()
        self.max_retries = max_retries

    def call(self, prompt: str) -> LLMResponse:
        """
        Send the prompt to the LLM and return the response.
        Falls back to the offline template on any unrecoverable error.

        Parameters
        ----------
        prompt : str — the full prompt string from PromptBuilder

        Returns
        -------
        LLMResponse
        """
        if self.provider == "offline":
            return _offline_response(prompt)

        fn = _PROVIDER_MAP.get(self.provider)
        if fn is None:
            print(f"[LLMProvider] Unknown provider '{self.provider}' — using offline fallback.")
            return _offline_response(prompt)

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 2):
            try:
                response = fn(prompt)
                print(f"[LLMProvider] {self.provider}/{response.model} "
                      f"— {response.latency_s:.2f}s "
                      f"({len(response.text)} chars)")
                return response

            except Exception as exc:
                last_error = exc
                print(f"[LLMProvider] Attempt {attempt} failed: {exc}")
                if attempt <= self.max_retries:
                    time.sleep(1.5 * attempt)   # simple back-off

        # All retries exhausted — use offline fallback.
        print(f"[LLMProvider] All retries failed ({last_error}). Using offline fallback.")
        return _offline_response(prompt)
