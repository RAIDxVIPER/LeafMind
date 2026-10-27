"""
LeafMind Pydantic Schemas — v1.0
All inter-agent message contracts. Field names and types are final.
This file is specification code, not application logic.

Source: docs/spec/part6_build_spec.md §6d-iii
Copied verbatim — do NOT modify field names or types without updating the spec.
"""
from __future__ import annotations
from datetime import datetime, date
from typing import Optional, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid


# ── Farmer Request ─────────────────────────────────────────────────────────────

class FarmerRequest(BaseModel):
    """Initial or follow-up submission from the farmer."""
    farmer_id: str = Field(..., description="Unique farmer identifier")
    image_bytes: bytes = Field(..., description="Raw image bytes (JPEG or PNG)")
    image_mime_type: str = Field(..., pattern=r"^image/(jpeg|png)$")
    gps_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    gps_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    place_name: Optional[str] = Field(None, description="Village, town, or PIN")
    crop_type: Optional[str] = Field(None, description="Optional farmer-supplied crop name")
    submitted_at: datetime
    is_followup: bool = Field(False)
    case_id: Optional[uuid.UUID] = Field(None, description="Required when is_followup=True")

    @field_validator("case_id")
    @classmethod
    def case_id_required_for_followup(cls, v, info):
        if info.data.get("is_followup") and v is None:
            raise ValueError("case_id is required when is_followup=True")
        return v

    @model_validator(mode="after")
    def sync_coords(self) -> "FarmerRequest":
        if self.latitude is None and self.gps_lat is not None:
            self.latitude = self.gps_lat
        elif self.gps_lat is None and self.latitude is not None:
            self.gps_lat = self.latitude
        if self.longitude is None and self.gps_lon is not None:
            self.longitude = self.gps_lon
        elif self.gps_lon is None and self.longitude is not None:
            self.gps_lon = self.longitude
        return self


# ── Vision Agent ───────────────────────────────────────────────────────────────

class ClassProb(BaseModel):
    label: str
    prob: float = Field(..., ge=0.0, le=1.0)

    def __iter__(self):
        """Allow tuple unpacking: (label, prob) = candidate."""
        return iter((self.label, self.prob))

    def __getitem__(self, item):
        """Allow indexing: candidate[0] == label, candidate[1] == prob."""
        return (self.label, self.prob)[item]

    @property
    def class_name(self) -> str:
        return self.label


class DetectedLeaf(BaseModel):
    class_id: int
    class_name: str
    bbox: List[float] = Field(..., min_length=4, max_length=4,
                              description="[cx, cy, w, h] normalised 0-1 (YOLO format)")
    det_conf: float = Field(..., ge=0.0, le=1.0)

class ImageQualityFlags(BaseModel):
    low_resolution: bool = False
    blurry: bool = False
    non_leaf_suspected: bool = False

class VisionOutput(BaseModel):
    """
    Output of the Vision Agent.
    severity_pct is always null: no per-leaf severity labels or masks exist in
    the training data. severity_note carries the mandatory explanation.
    """
    top_label: str
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Calibrated (temperature-scaled) softmax probability")
    top_k_probs: List[ClassProb] = Field(..., min_length=1, max_length=10)
    severity_pct: None = Field(
        None,
        description="Always null. Severity percentage is unsupported by current data."
    )
    severity_note: str = Field(
        "Severity percentage is not computed. No per-leaf severity labels or lesion masks "
        "exist in the training data. The PPT design target is acknowledged but cannot be "
        "implemented without expert-annotated mask data.",
        description="Mandatory explanation accompanying severity_pct=null"
    )
    image_quality_flags: ImageQualityFlags
    ood_flag: bool = Field(False, description="True when max confidence < 0.40 [ENG assumption]")
    leaf_detected: bool = Field(
        False, description="True if optional YOLOv8s detector found at least one leaf box"
    )
    detected_leaves: List[DetectedLeaf] = Field(default_factory=list)
    escalate_to_human: bool = Field(
        False,
        description="True when confidence < 0.60 OR (top1-top2) < 0.15 [ENG assumption]"
    )
    model_version: str
    temperature: float = Field(..., description="Temperature scalar applied to logits")

    # Crop-restricted diagnostic and prediction fields [C2 follow-up 2]
    # [ENG decision] crop is user-supplied input, not predicted
    crop: Optional[str] = Field(
        default=None,
        description="User-supplied crop name, if provided. [ENG decision: crop is user-supplied input, not predicted]"
    )
    top_3_candidates: Optional[List[ClassProb]] = Field(
        default=None,
        description="Top-3 candidate classes with probabilities from the crop-masked distribution"
    )
    single_class_crop: bool = Field(
        default=False,
        description="True if crop has only one class in taxonomy (cannot distinguish disease)"
    )
    can_diagnose: bool = Field(
        default=True,
        description="False if single_class_crop=True; model cannot provide a disease diagnosis"
    )

    @property
    def crop_candidates(self) -> Optional[List[ClassProb]]:
        """Alias for top_3_candidates."""
        return self.top_3_candidates

    @property
    def top3_candidates(self) -> Optional[List[ClassProb]]:
        """Alias for top_3_candidates."""
        return self.top_3_candidates

    @property
    def candidates(self) -> Optional[List[ClassProb]]:
        """Alias for top_3_candidates."""
        return self.top_3_candidates


# ── Weather Agent ──────────────────────────────────────────────────────────────

class WeatherInput(BaseModel):
    gps_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    gps_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    place_name: Optional[str] = Field(None, description="Village, town, or PIN")
    query_datetime: Optional[datetime] = None
    detected_disease: Optional[str] = Field(None, description="Optional diagnosed disease class for computing weather_score")

    @model_validator(mode="after")
    def sync_coords(self) -> "WeatherInput":
        if self.latitude is None and self.gps_lat is not None:
            self.latitude = self.gps_lat
        elif self.gps_lat is None and self.latitude is not None:
            self.gps_lat = self.latitude
        if self.longitude is None and self.gps_lon is not None:
            self.longitude = self.gps_lon
        elif self.gps_lon is None and self.longitude is not None:
            self.gps_lon = self.longitude
        return self

class CurrentWeather(BaseModel):
    temperature_c: float
    relative_humidity_pct: float = Field(..., ge=0.0, le=100.0)
    precipitation_mm: float = Field(..., ge=0.0)
    wind_speed_kmh: float = Field(..., ge=0.0)
    cloud_cover_pct: float = Field(..., ge=0.0, le=100.0)
    dew_point_c: float

class DailyForecast(BaseModel):
    forecast_date: date
    temp_max_c: float
    temp_min_c: float
    humidity_max_pct: float = Field(..., ge=0.0, le=100.0)
    precipitation_sum_mm: float = Field(..., ge=0.0)
    precipitation_hours: int = Field(..., ge=0)
    wind_speed_max_kmh: float = Field(..., ge=0.0)

class WeatherRiskFlag(BaseModel):
    condition_name: str
    triggered: bool
    current_value: str          # Human-readable description of observed value(s)
    threshold: str              # Human-readable threshold expression
    threshold_source: str       # Citation; "[UNVERIFIED — I14]" appended where applicable

class WeatherWindow(BaseModel):
    past_days: int = Field(7, description="Number of past days analyzed [ENG]")
    forecast_days: int = Field(3, description="Number of forecast days analyzed [ENG]")
    start_date: Optional[date] = None
    end_date: Optional[date] = None

class WeatherSummaryStats(BaseModel):
    mean_temperature_c: float
    max_temperature_c: float
    min_temperature_c: float
    mean_relative_humidity_pct: float
    max_relative_humidity_pct: float
    min_relative_humidity_pct: float
    total_precipitation_mm: float
    precipitation_hours: int
    hours_rh_above_threshold: int = Field(..., description="Hours with RH >= 90% [ENG proxy]")
    leaf_wetness_hours_proxy: int = Field(..., description="Hours with RH >= 90% and/or precip > 0.1mm [ENG proxy]")
    max_wind_speed_kmh: float

class RuleTriggerResult(BaseModel):
    rule_id: str
    condition_name: str
    triggered: bool
    measured_values: str
    threshold: str
    time_window: str
    threshold_source: str
    disease_targets: List[str] = Field(default_factory=list)
    tag: str = "[UNVERIFIED — I14]"
    omitted_from_scoring: bool = False

class WeatherOutput(BaseModel):
    weather_available: bool
    weather_partial: bool = False
    cache_hit: bool = False
    cache_stale: bool = False
    stale: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: Optional[str] = None
    location_admin: Optional[str] = None
    location_source: Optional[str] = Field(None, pattern=r"^(gps|geocoded|default)$")
    is_ambiguous_location: bool = False
    ambiguity_flag: bool = False
    weather_fetch_timestamp: Optional[datetime] = None
    fetched_at: Optional[datetime] = None
    data_window: Optional[WeatherWindow] = None
    summary_statistics: Optional[WeatherSummaryStats] = None
    current_weather: Optional[CurrentWeather] = None
    forecast_7day: Optional[List[DailyForecast]] = None
    disease_risk_flags: List[WeatherRiskFlag] = Field(default_factory=list)
    rules_triggered: List[RuleTriggerResult] = Field(default_factory=list)
    weather_score: float = Field(0.0, ge=0.0, le=1.0, description="Computed per Part 4 formula [ENG]")
    weather_data_source: Optional[str] = None   # "open-meteo" | "openweathermap" | null
    unavailable_reason: Optional[str] = None

    @model_validator(mode="after")
    def sync_aliases(self) -> "WeatherOutput":
        # Synchronize stale / cache_stale
        if self.cache_stale and not self.stale:
            self.stale = True
        elif self.stale and not self.cache_stale:
            self.cache_stale = True
        # Synchronize fetched_at / weather_fetch_timestamp
        if self.weather_fetch_timestamp and not self.fetched_at:
            self.fetched_at = self.weather_fetch_timestamp
        elif self.fetched_at and not self.weather_fetch_timestamp:
            self.weather_fetch_timestamp = self.fetched_at
        # Synchronize ambiguity_flag / is_ambiguous_location
        if self.is_ambiguous_location and not self.ambiguity_flag:
            self.ambiguity_flag = True
        elif self.ambiguity_flag and not self.is_ambiguous_location:
            self.is_ambiguous_location = True
        return self


# ── RAG Agent ──────────────────────────────────────────────────────────────────

class ChemicalOption(BaseModel):
    product: str
    verification_tag: str = Field(..., description="Verification tag (V1, V2, V3, US, or --)")
    dose: Optional[str] = Field(None, description="Formulation dose")
    water: Optional[str] = Field(None, description="Water volume")
    target: Optional[str] = Field(None, description="Target disease or crop stage")
    basis_phi: Optional[str] = Field(None, description="Regulatory basis, pre-harvest interval, or source note")


class Citation(BaseModel):
    tag: str = Field(..., description="Citation tag, e.g. T8, IH26, RED")
    title: str = Field(..., description="Source title or bibliographical description")
    url: Optional[str] = Field(None, description="Source URL if available")


class RAGInput(BaseModel):
    disease_label: str          # e.g. "Tomato___Early_blight"
    top_k_probs: List[ClassProb]  # Passed for context


class RAGOutput(BaseModel):
    """
    Output of the RAG Agent (per disease candidate or query).
    Extractive assembly only — no LLM generation.
    When found=False or kb_gap=True, treatment fields remain empty.
    Chemical doses are returned only when include_doses=True.
    """
    # Core candidate fields (Stage C3)
    crop: Optional[str] = Field(None, description="Crop name (e.g. Tomato, Apple, Corn)")
    model_class: Optional[str] = Field(None, description="Model class string from class_map_d2.json")
    disease_name: Optional[str] = Field(None, description="Human-readable disease name from KB chapter")
    found: bool = Field(False, description="True if disease chapter was found and extracted from KB")
    kb_gap: bool = Field(False, description="True if model class has no KB chapter (e.g. Potato/Pepper until added)")
    reason: Optional[str] = Field(None, description="Explanation when not found (e.g. 'healthy_class', 'no_kb_chapter')")
    what_it_is: Optional[str] = Field(None, description="Disease identification and symptoms from Chapter Sec 1 / At a glance")
    why_it_happens: Optional[str] = Field(None, description="Causal pathogen, spread and progression from Chapter Sec 2")
    what_to_do_first: Optional[str] = Field(None, description="Immediate cultural/sanitation/preventive action from Chapter Sec 4")
    favourable_conditions: Optional[str] = Field(None, description="Weather/environmental drivers from Chapter Sec 3 (feeds C5 risk agent)")
    look_alikes: Optional[str] = Field(None, description="Look-alikes and differential diagnosis from Chapter Sec 1")
    prevention: Optional[str] = Field(None, description="Full prevention and cultural management measures from Chapter Sec 4")
    chemical_options: List[ChemicalOption] = Field(
        default_factory=list,
        description="Verified chemical options with verification tags (empty unless include_doses=True)"
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Statutory caveats (confirm current label, CIB&RC compilation dates)"
    )
    citations: List[Union[Citation, str]] = Field(
        default_factory=list,
        description="Source citations with tags and URLs from Chapter Sec 6"
    )

    # Backward-compatible fields from spec v1.0
    rag_available: bool = Field(True, description="True if RAG lookup succeeded without system error")
    knowledge_gap_flag: bool = Field(False, description="Legacy alias for kb_gap")
    synthesis_failed: bool = False
    treatment_options: List[str] = Field(default_factory=list)
    active_ingredients: List[str] = Field(default_factory=list)
    application_timing: Optional[str] = None
    preventive_measures: List[str] = Field(default_factory=list)
    cost_estimate_note: None = Field(
        None,
        description="Always null until RAG knowledge base contains cost data [GAP]"
    )
    retrieval_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Mean cross-encoder reranker score of top-3 chunks"
    )
    raw_chunks: Optional[List[str]] = Field(
        None,
        description="Set when synthesis_failed=True; raw retrieved text as fallback"
    )

# ── Risk/Reason Agent ──────────────────────────────────────────────────────────

class RiskFactor(BaseModel):
    source: str                 # e.g. "vision_confidence", "weather_W03"
    value: str                  # Human-readable observed value
    contribution: float         # Signed contribution to risk_score [ENG assumption]
    tag: str                    # Provenance tag, e.g. "[ENG assumption]"

class RiskInput(BaseModel):
    vision: VisionOutput
    weather: WeatherOutput
    rag: RAGOutput

class RiskOutput(BaseModel):
    """
    Risk score is computed from disease threat + weather + history.
    Uncertainty (confidence, image quality, OOD) is kept SEPARATE and drives escalation only.
    A confident high-threat disease always scores high risk.
    A blurry low-confidence image always escalates, not inflates the risk score.
    """
    risk_level: str = Field(..., pattern=r"^(low|moderate|severe)$")
    risk_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Threat-driven composite score [ENG assumption]; "
                    "independent of classifier confidence"
    )
    factors: List[RiskFactor]
    escalate_to_human: bool
    uncertainty_level: str = Field(
        ...,
        pattern=r"^(high_confidence|moderate_confidence|low_confidence)$",
        description="Separate from risk_score; driven by U1-U6 uncertainty signals"
    )
    uncertainty_signals_triggered: List[str] = Field(
        default_factory=list,
        description="e.g. ['U1_low_confidence', 'U4_blurry_image']; does NOT affect risk_score"
    )
    weather_unavailable: bool = False
    rag_unavailable: bool = False
    progression_note: str = Field(
        ...,
        description="Text heuristic only. NOT a quantitative prediction. [ENG][GAP]"
    )
    risk_computed_at: datetime


# ── Recommendation Agent ───────────────────────────────────────────────────────

class RecommendationInput(BaseModel):
    risk: RiskOutput
    rag: RAGOutput
    vision: Optional[VisionOutput] = None
    weather: Optional[WeatherOutput] = None
    include_doses: bool = False

class RecommendationOutput(BaseModel):
    """
    Farmer-facing recommendation (Stage C6).
    All treatment content must be traceable to RAGOutput.citations or risk/weather fields.
    cost_estimate_note is null until knowledge base contains cost data [GAP].
    """
    # C6 specification fields
    summary: str = Field(default="", description="Advisory summary")
    urgency: str = Field(..., pattern=r"^(immediate|within_24h|within_48h|within_week|monitor|consult_agronomist)$")
    what_to_do_first: Optional[str] = Field(None, description="Immediate cultural/sanitation/treatment action")
    steps: List[str] = Field(default_factory=list, description="Actionable advisory steps")
    weather_note: Optional[str] = Field(None, description="Weather-related advisory context [UNVERIFIED — I14]")
    uncertainty_note: Optional[str] = Field(None, description="Uncertainty explanation, top-3 candidates, or healthy disclaimer")
    citations: List[str] = Field(default_factory=list, description="Citations passed through from RAG")
    flags: List[str] = Field(default_factory=list, description="Operational flags (e.g. PROVISIONAL_SEVERITY_GAP_ROW, WEATHER_UNVERIFIED_I14)")
    llm_used: bool = Field(default=False, description="True if LLM reworded advice; False if deterministic fallback used")
    fallback_reason: Optional[str] = Field(None, description="Reason deterministic fallback was used when llm_used=False")

    # Backward-compatible fields from spec v1.0 / test_schemas.py
    actions: List[str] = Field(default_factory=list, description="Alias for steps")
    risk_level: str = Field(default="moderate", description="Risk level from RiskOutput")
    confidence_justification: str = Field(default="", description="Plain-language explanation of confidence for farmer")
    caveats: List[str] = Field(default_factory=list, description="Any limitations or warnings")
    cost_estimate_note: None = Field(None, description="Always null [GAP — no cost data in RAG]")
    rag_grounded: bool = Field(default=False, description="False when RAG unavailable or empty citations")
    escalate_to_human: bool = Field(default=False, description="True if agronomist escalation required")
    knowledge_gap_caveated: bool = Field(default=False, description="True when knowledge_gap_flag was set")

    @model_validator(mode="after")
    def sync_legacy_fields(self) -> "RecommendationOutput":
        # Keep steps and actions synchronized
        if self.steps and not self.actions:
            self.actions = list(self.steps)
        elif self.actions and not self.steps:
            self.steps = list(self.actions)
        # Summary fallback
        if not self.summary:
            if self.what_to_do_first:
                self.summary = self.what_to_do_first
            elif self.actions:
                self.summary = self.actions[0]
            else:
                self.summary = "Advisory recommendation generated."
        # Actions fallback
        if not self.actions and self.what_to_do_first:
            self.actions = [self.what_to_do_first]
            self.steps = [self.what_to_do_first]
        return self


# ── Monitoring Agent ───────────────────────────────────────────────────────────

class MonitoringInput(BaseModel):
    case_id: uuid.UUID
    farmer_id: str
    followup_image_bytes: bytes
    followup_submitted_at: datetime
    visit_number: int = Field(..., ge=2, description="Visit 1 = initial; follow-ups start at 2")

class DetectionChange(BaseModel):
    baseline_detected: bool
    followup_detected: bool

class MonitoringOutput(BaseModel):
    """
    'disease_status' characterises change between visits; NOT a clinical efficacy judgement.
    Confidence change (confidence_raw_delta) is NOT an efficacy signal.
    severity_delta_pct is always null: no severity data exists.
    """
    case_id: uuid.UUID
    visit_number: int
    baseline_label: str
    baseline_confidence: float
    followup_label: str
    followup_confidence: float
    label_same: bool
    disease_still_detected: bool
    new_label_is_healthy: bool
    confidence_raw_delta: float  # followup_confidence - baseline_confidence; NOT efficacy signal
    disease_status: str = Field(
        ...,
        pattern=r"^(same_disease_still_detected|no_disease_detected_at_followup|"
                r"different_disease_detected_at_followup|not_applicable_healthy_baseline|unknown)$",
        description="Characterises change between visits; NOT a clinical efficacy judgement"
    )
    detection_change: Optional[DetectionChange] = None
    severity_delta_pct: None = Field(
        None,
        description="Always null. Numeric severity delta is not computable."
    )
    severity_note: str = Field(
        "Numeric severity delta is not computable. No per-leaf severity labels or masks exist in "
        "the training data. PPT claim of '% change in affected area' is unsupported by current data.",
        description="Mandatory explanation accompanying severity_delta_pct=null"
    )
    same_image_flag: bool = False
    poor_quality_flag: bool = False
    crop_mismatch_flag: bool = False
    too_soon_flag: bool = False
    escalate_to_human: bool
    followup_vision: VisionOutput


# ── Case State Object (persisted across visits) ────────────────────────────────

class FollowUpRecord(BaseModel):
    visit_number: int
    submitted_at: datetime
    monitoring_output: MonitoringOutput
    risk_output: RiskOutput
    recommendation_output: RecommendationOutput

class CaseState(BaseModel):
    """
    Persisted across days. Loaded by Orchestrator on follow-up path.
    Stored as JSON in SQLite (Phase 1) or a document store (Phase 2).
    """
    case_id: uuid.UUID
    farmer_id: str
    created_at: datetime
    crop_type: Optional[str] = None
    initial_request: FarmerRequest
    initial_vision: VisionOutput
    initial_weather: WeatherOutput
    initial_rag: RAGOutput
    initial_risk: RiskOutput
    initial_recommendation: RecommendationOutput
    follow_ups: List[FollowUpRecord] = Field(default_factory=list)
    status: str = Field("active", pattern=r"^(active|escalated|closed)$")


# ── Final Farmer Response ──────────────────────────────────────────────────────

class FarmerResponse(BaseModel):
    case_id: uuid.UUID
    submitted_at: datetime
    is_followup: bool
    vision: VisionOutput
    risk: RiskOutput
    recommendation: RecommendationOutput
    monitoring: Optional[MonitoringOutput] = None   # Present on follow-up path only
    escalate_to_human: bool
    partial_result: bool = False    # True when one or more parallel branches failed


# ── Human Escalation Payload ───────────────────────────────────────────────────

class HumanEscalationPayload(BaseModel):
    """
    Sent to agronomist review queue when escalate_to_human=True.
    Contains all evidence needed for manual review.
    """
    case_id: uuid.UUID
    farmer_id: str
    escalation_reason: List[str]    # Human-readable list of reasons
    vision_output: VisionOutput
    weather_output: WeatherOutput
    rag_output: RAGOutput
    risk_output: RiskOutput
    draft_recommendation: RecommendationOutput  # Agent's best-effort; agronomist can override
    image_bytes: bytes
    escalated_at: datetime
    priority: str = Field(..., pattern=r"^(low|medium|high|urgent)$")
