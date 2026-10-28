"""
src/api/schemas.py
==================
Pydantic schemas for the LeafMind FastAPI backend contract.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.orchestrator.state import FinalResponse


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class MetaResponse(BaseModel):
    supported_crops: List[str]
    single_class_crops: List[str]
    multi_class_crops: List[str]
    capture_tips: List[str]
    disclaimers: List[str]


class Top3Item(BaseModel):
    name: str
    probability: float


class AnalyzeResponse(FinalResponse):
    """
    Unified analyze response combining the full FinalResponse contract
    with farmer display-ready fields.
    """
    farmer_summary: str
    top3: List[Top3Item] = Field(default_factory=list)
    risk_label: str
    urgency: str
    steps: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    disclaimers: List[str] = Field(default_factory=list)
    escalation_banner: Optional[str] = None
