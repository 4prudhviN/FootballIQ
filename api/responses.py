#!/usr/bin/env python3
"""
API Response Models
===================
Typed Pydantic response models for all API endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    status:               str
    job_id:               str
    video_url:            str
    detectedActivities:   List[str]
    playerLevel:          str
    metrics:              Dict[str, Any]
    aiFeedback:           Dict[str, Any]
    drills:               List[Dict[str, Any]]
    timeline:             List[Dict[str, Any]]
    focusThisWeek:        List[str]
    trainingPlan:         Dict[str, Any]
    weeklyPlan:           Dict[str, Any]
    recoveryAdvice:       Dict[str, Any]
    session_id:           Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: List[str]


class ProgressResponse(BaseModel):
    session_count:         int
    overall_trend:         str
    summary:               str
    comparison:            Dict[str, Any]
    metric_history:        Dict[str, Any]
    persistent_weaknesses: List[str]


class PipelineStatusResponse(BaseModel):
    job_id:  str
    stage:   str
    status:  str


class HealthResponse(BaseModel):
    service:      str
    version:      str
    status:       str
    entry_point:  str
    endpoints:    Dict[str, str]
    config:       Dict[str, Any]
