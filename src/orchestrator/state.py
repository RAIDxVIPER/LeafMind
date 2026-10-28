"""
src/orchestrator/state.py
=========================
State schemas and data models for the LeafMind LangGraph orchestrator (Stage C7).

Enforces:
  - Spec 3c-ii: LeafMindState TypedDict
  - Concurrent branch safety: Annotated reducers for node_timings_ms, errors, flags
  - D2/R3: severity_pct = None with mandatory severity_note in FinalResponse and VisionSummary
  - D3: Node execution timings in ms measured per node
  - D10/R14: Separation of risk and uncertainty
  - Secret protection: API keys and internal stack traces sanitized from all errors
"""
from __future__ import annotations

from datetime import datetime
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import uuid

from pydantic import BaseModel, Field

from src.schemas.schemas import (
    CaseState,
    Citation,
    ClassProb,
    FarmerRequest,
    FarmerResponse,
    HumanEscalationPayload,
    ImageQualityFlags,
    MonitoringOutput,
    RAGOutput,
    RecommendationOutput,
    RiskOutput,
    VisionOutput,
    WeatherOutput,
)


# =============================================================================
# Secret & Error Sanitization Helper
# =============================================================================

def sanitize_error(message: Any) -> str:
    """
    Sanitize an error message or exception representation to ensure that
    no API keys, tokens, secrets, or raw internal paths/tracebacks leak to the user.
    """
    if message is None:
        return ""
    text = str(message)

    # Redact Google / Gemini API keys (AIza...)
    text = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "[REDACTED_API_KEY]", text)
    # Redact OpenAI-style keys (sk-...)
    text = re.sub(r"sk-[0-9A-Za-z\-_]{15,}", "[REDACTED_API_KEY]", text)
    # Redact Bearer tokens
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9\-\._~+/]+=*", "Bearer [REDACTED_TOKEN]", text)
    # Redact generic key=value patterns where key looks like api_key or secret
    text = re.sub(
        r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"]?[^\s'\",]+['\"]?",
        r"\1=[REDACTED]",
        text,
    )
    # Strip internal file paths and raw traceback lines while preserving error text
    lines = []
    in_tb = False
    for line in text.splitlines():
        if "Traceback (most recent call last):" in line:
            in_tb = True
            continue
        if in_tb:
            if line.strip().startswith("File ") or line.strip().startswith("line "):
                continue
            lines.append(line.strip())
            in_tb = False
        else:
            lines.append(line.strip())

    cleaned = " | ".join(l for l in lines if l)
    return cleaned.strip()


# =============================================================================
# LangGraph Channel Reducer Helpers
# =============================================================================

def merge_dicts(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reducer that merges dictionary updates from sequential or concurrent nodes."""
    res = dict(a or {})
    if b:
        res.update(b)
    return res


def merge_lists(a: Optional[List[Any]], b: Optional[List[Any]]) -> List[Any]:
    """Reducer that merges unique list entries from sequential or concurrent nodes."""
    res = list(a or [])
    if b:
        for x in b:
            if x not in res:
                res.append(x)
    return res


def merge_bool_or(a: Optional[bool], b: Optional[bool]) -> bool:
    """Reducer that combines boolean flags with logical OR."""
    return bool(a or b)


# =============================================================================
# Response Sub-Summaries
# =============================================================================

class VisionSummary(BaseModel):
    """Summarized vision result for the final farmer-facing response."""
    top_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_3: List[ClassProb] = Field(default_factory=list)
    can_diagnose: bool = True
    single_class_crop: bool = False
    ood_flag: bool = False
    uncertainty_flags: ImageQualityFlags = Field(default_factory=ImageQualityFlags)
    severity_pct: None = Field(
        None,
        description="Always null. Severity percentage is unsupported by current data [D2/R3]."
    )
    severity_note: str = Field(
        "Severity percentage is not computed. No per-leaf severity labels or lesion masks "
        "exist in the training data.",
        description="Mandatory explanation accompanying severity_pct=null [D2/R3]."
    )


class WeatherSummary(BaseModel):
    """Summarized weather context for the final farmer-facing response."""
    weather_available: bool
    current_temp_c: Optional[float] = None
    current_humidity_pct: Optional[float] = None
    triggered_rules: List[str] = Field(default_factory=list)
    weather_score: float = 0.0
    data_source: Optional[str] = None
    cached: bool = False
    location_name: Optional[str] = None


class RAGSummary(BaseModel):
    """Summarized retrieval context for the final farmer-facing response."""
    found: bool
    kb_gap: bool = False
    disease_name: Optional[str] = None
    sources: List[Citation] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    chemical_options_count: int = 0
    what_to_do_first: Optional[str] = None
    doses_included: bool = False


class TriggerReason(BaseModel):
    """Structured detail for an individual escalation trigger that fired."""
    trigger_id: str
    description: str
    priority: str = Field("medium", pattern=r"^(low|medium|high|urgent)$")
    value: Optional[Any] = None
    threshold: Optional[Any] = None


class EscalationSummary(BaseModel):
    """Summarized escalation outcome and agronomist payload."""
    escalate_to_human: bool
    priority: str = Field("low", pattern=r"^(low|medium|high|urgent)$")
    triggered_triggers: List[str] = Field(default_factory=list)
    reasons: List[TriggerReason] = Field(default_factory=list)
    payload: Optional[HumanEscalationPayload] = None


# =============================================================================
# Final Response Schema
# =============================================================================

class FinalResponse(BaseModel):
    """
    Unified final response produced by the Orchestrator (Stage C7).
    Contains full agent outputs, summaries, escalation info, and execution timing.
    """
    case_id: uuid.UUID
    farmer_id: str
    submitted_at: datetime
    crop: Optional[str] = None
    is_followup: bool = False

    # Core agent outputs
    vision: Optional[VisionSummary] = None
    weather: Optional[WeatherSummary] = None
    rag: Optional[RAGSummary] = None
    risk: Optional[RiskOutput] = None
    recommendation: Optional[RecommendationOutput] = None
    escalation: EscalationSummary
    monitoring: Optional[MonitoringOutput] = None

    # Full raw outputs for downstream consumers/APIs
    raw_vision: Optional[VisionOutput] = None
    raw_weather: Optional[WeatherOutput] = None
    raw_rag: Optional[RAGOutput] = None

    # Operational status & metadata
    partial_result: bool = False
    flags: List[str] = Field(default_factory=list)
    timing_ms: Dict[str, float] = Field(
        default_factory=dict,
        description="Execution measurement in milliseconds per node [D3]."
    )
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Sanitized non-fatal or fatal error messages keyed by node name."
    )
    fatal_error: Optional[str] = None

    # Severity invariant (D2/R3)
    severity_pct: None = Field(
        None,
        description="Always null. Severity percentage is unsupported by current data [D2/R3]."
    )
    severity_note: str = Field(
        "Severity percentage is not computed. No per-leaf severity labels or lesion masks "
        "exist in the training data.",
        description="Mandatory explanation accompanying severity_pct=null [D2/R3]."
    )


# =============================================================================
# LangGraph State Object (Spec 3c-ii)
# =============================================================================

class LeafMindState(TypedDict, total=False):
    """
    LangGraph state dict for the LeafMind orchestrator.
    All fields are populated incrementally across graph nodes.
    thread_id = str(case_id) is used by LangGraph checkpointing.
    """
    # ── Input ──────────────────────────────────────────────────────────
    farmer_request: Optional[FarmerRequest]
    is_followup: bool
    image_bytes: Optional[bytes]
    crop: Optional[str]
    location: Optional[Any]
    include_doses: bool

    # ── Case Identity ──────────────────────────────────────────────────
    case_id: Optional[str]               # str(UUID4)
    farmer_id: Optional[str]
    submitted_at: Optional[datetime]

    # ── Branch Outputs (Fan-out) ───────────────────────────────────────
    vision_output: Optional[VisionOutput]
    weather_output: Optional[WeatherOutput]
    rag_output: Optional[RAGOutput]

    # ── Synthesis ──────────────────────────────────────────────────────
    risk_output: Optional[RiskOutput]
    recommendation_output: Optional[RecommendationOutput]

    # ── Follow-up / Monitoring (C8 slot) ───────────────────────────────
    monitoring_output: Optional[MonitoringOutput]
    loaded_case_state: Optional[CaseState]

    # ── Escalation & Response ──────────────────────────────────────────
    escalate_to_human: bool
    escalation_summary: Optional[EscalationSummary]
    escalation_payload: Optional[HumanEscalationPayload]
    farmer_response: Optional[FarmerResponse]
    final_response: Optional[FinalResponse]
    partial_result: Annotated[bool, merge_bool_or]
    flags: Annotated[List[str], merge_lists]

    # ── Error & Timing Tracking ────────────────────────────────────────
    vision_error: Optional[str]
    weather_error: Optional[str]
    rag_error: Optional[str]
    risk_error: Optional[str]
    recommendation_error: Optional[str]
    fatal_error: Optional[str]
    errors: Annotated[Dict[str, str], merge_dicts]
    node_timings_ms: Annotated[Dict[str, float], merge_dicts]
