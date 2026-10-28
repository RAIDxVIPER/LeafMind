from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, field_validator
from src.orchestrator.state import FinalResponse, TriggerReason
from src.schemas.schemas import (
    RAGOutput,
    RecommendationOutput,
    RiskOutput,
    VisionOutput,
    WeatherOutput,
)


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


class SafeImageStub(BaseModel):
    """
    Safe image metadata stub replacing raw binary image bytes in API responses.
    Prevents UTF-8 encoding failures and response bloat while retaining metadata.
    """
    image_included: bool = Field(
        default=False,
        description="Whether raw image binary data is included in JSON response (always False).",
    )
    image_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 digest of original image bytes if available.",
    )
    size_bytes: int = Field(
        default=0,
        description="Original image byte length.",
    )


class SafeHumanEscalationPayload(BaseModel):
    """
    API-safe escalation payload schema. Replaces raw image bytes with SafeImageStub.
    """
    case_id: uuid.UUID
    farmer_id: str
    escalation_reason: List[str]
    vision_output: VisionOutput
    weather_output: WeatherOutput
    rag_output: RAGOutput
    risk_output: RiskOutput
    draft_recommendation: RecommendationOutput
    image_bytes: Optional[SafeImageStub] = Field(
        default_factory=SafeImageStub,
        description="Safe image metadata stub replacing raw binary image bytes in API responses.",
    )
    escalated_at: datetime
    priority: str

    @field_validator("image_bytes", mode="before")
    @classmethod
    def sanitize_image_bytes(cls, v: Any) -> Optional[SafeImageStub]:
        if isinstance(v, SafeImageStub):
            return v
        if isinstance(v, bytes):
            sha = hashlib.sha256(v).hexdigest() if v else None
            return SafeImageStub(
                image_included=False,
                image_sha256=sha,
                size_bytes=len(v),
            )
        if isinstance(v, dict):
            return SafeImageStub(**v)
        if isinstance(v, str):
            return SafeImageStub(
                image_included=False,
                image_sha256=hashlib.sha256(v.encode("utf-8")).hexdigest() if v else None,
                size_bytes=len(v),
            )
        return SafeImageStub()


class SafeEscalationSummary(BaseModel):
    """
    API-safe escalation summary schema.
    """
    escalate_to_human: bool = False
    priority: str = "medium"
    triggered_triggers: List[str] = Field(default_factory=list)
    reasons: List[TriggerReason | str | Dict[str, Any]] = Field(default_factory=list)
    payload: Optional[SafeHumanEscalationPayload] = None


class AnalyzeResponse(FinalResponse):
    """
    Unified analyze response combining the full FinalResponse contract
    with farmer display-ready fields. Replaces raw image bytes in escalation.payload
    with SafeImageStub.
    """
    escalation: SafeEscalationSummary
    farmer_summary: str
    top3: List[Top3Item] = Field(default_factory=list)
    risk_label: str
    urgency: str
    steps: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    disclaimers: List[str] = Field(default_factory=list)
    escalation_banner: Optional[str] = None

