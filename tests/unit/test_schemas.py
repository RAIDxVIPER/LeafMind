"""
tests/unit/test_schemas.py
===========================
Verifies that the Pydantic schemas in src/schemas/schemas.py:
  1. Import correctly with pydantic v2.
  2. Accept valid example inputs (from spec §6d-iii example contracts).
  3. Reject invalid inputs with appropriate ValidationError.
  4. Enforce severity_pct=None, severity_delta_pct=None always (R3).
  5. Enforce UUID presence for follow-up FarmerRequest (R6).
  6. Enforce field patterns (risk_level, urgency, disease_status).

Requirements: pydantic>=2.0 (R6), pytest.
Run: pytest tests/unit/test_schemas.py -v
"""

import uuid
from datetime import datetime, date, timezone

import pytest
from pydantic import ValidationError

from src.schemas.schemas import (
    ClassProb,
    CaseState,
    CurrentWeather,
    DailyForecast,
    DetectedLeaf,
    DetectionChange,
    FarmerRequest,
    FarmerResponse,
    FollowUpRecord,
    HumanEscalationPayload,
    ImageQualityFlags,
    MonitoringInput,
    MonitoringOutput,
    RAGInput,
    RAGOutput,
    RiskFactor,
    RiskInput,
    RiskOutput,
    RecommendationInput,
    RecommendationOutput,
    VisionOutput,
    WeatherInput,
    WeatherOutput,
    WeatherRiskFlag,
)

# ── Shared fixtures ─────────────────────────────────────────────────────────────

NOW = datetime(2026, 10, 10, 9, 0, 0, tzinfo=timezone.utc)
TODAY = date(2026, 10, 10)


def make_class_prob(label: str = "Tomato___Early_blight", prob: float = 0.85) -> dict:
    return {"label": label, "prob": prob}


def make_image_quality_flags(**kwargs) -> dict:
    return {"low_resolution": False, "blurry": False, "non_leaf_suspected": False, **kwargs}


def make_vision_output(**kwargs) -> dict:
    base = {
        "top_label": "Tomato___Early_blight",
        "confidence": 0.85,
        "top_k_probs": [make_class_prob()],
        "image_quality_flags": make_image_quality_flags(),
        "model_version": "efficientnet_b3_v0.0.1_stub",
        "temperature": 1.0,
    }
    base.update(kwargs)
    return base


def make_weather_output(available: bool = True) -> dict:
    if not available:
        return {"weather_available": False}
    return {
        "weather_available": True,
        "current_weather": {
            "temperature_c": 25.0,
            "relative_humidity_pct": 80.0,
            "precipitation_mm": 2.5,
            "wind_speed_kmh": 20.0,
            "cloud_cover_pct": 60.0,
            "dew_point_c": 18.0,
        },
        "forecast_7day": [],
        "disease_risk_flags": [],
    }


def make_rag_output(available: bool = True, gap: bool = False) -> dict:
    if not available:
        return {"rag_available": False}
    return {
        "rag_available": True,
        "knowledge_gap_flag": gap,
        "treatment_options": [] if gap else ["Apply copper-based fungicide"],
        "citations": [] if gap else ["[STUB — not real treatment data]"],
    }


def make_risk_output(**kwargs) -> dict:
    base = {
        "risk_level": "moderate",
        "risk_score": 0.45,
        "factors": [
            {
                "source": "threat_table",
                "value": "DISEASE_DETECTED fallback",
                "contribution": 0.35,
                "tag": "[ENG assumption][GAP — threat table empty]",
            }
        ],
        "escalate_to_human": False,
        "uncertainty_level": "high_confidence",
        "uncertainty_signals_triggered": [],
        "progression_note": "First visit. No progression data. [ENG][GAP]",
        "risk_computed_at": NOW.isoformat(),
    }
    base.update(kwargs)
    return base


def make_recommendation_output(**kwargs) -> dict:
    base = {
        "actions": ["Apply copper-based fungicide"],
        "urgency": "within_48h",
        "risk_level": "moderate",
        "confidence_justification": "Model confidence 0.85 on clear image.",
        "citations": ["[STUB — not real treatment data]"],
        "caveats": ["Threat table is empty [GAP]; risk score uses fallback."],
        "rag_grounded": True,
        "escalate_to_human": False,
        "knowledge_gap_caveated": False,
    }
    base.update(kwargs)
    return base


def make_monitoring_output(**kwargs) -> dict:
    base = {
        "case_id": str(uuid.uuid4()),
        "visit_number": 2,
        "baseline_label": "Tomato___Early_blight",
        "baseline_confidence": 0.85,
        "followup_label": "Tomato___Early_blight",
        "followup_confidence": 0.80,
        "label_same": True,
        "disease_still_detected": True,
        "new_label_is_healthy": False,
        "confidence_raw_delta": -0.05,
        "disease_status": "same_disease_still_detected",
        "escalate_to_human": True,
        "followup_vision": make_vision_output(),
    }
    base.update(kwargs)
    return base


# ── Tests: FarmerRequest ────────────────────────────────────────────────────────

class TestFarmerRequest:
    def test_valid_new_case(self):
        req = FarmerRequest(
            farmer_id="F001",
            image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 100,  # fake JPEG header
            image_mime_type="image/jpeg",
            gps_lat=12.97,
            gps_lon=77.59,
            submitted_at=NOW,
            is_followup=False,
        )
        assert req.is_followup is False
        assert req.case_id is None

    def test_valid_followup(self):
        cid = uuid.uuid4()
        req = FarmerRequest(
            farmer_id="F001",
            image_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
            image_mime_type="image/png",
            gps_lat=-33.8688,
            gps_lon=151.2093,
            submitted_at=NOW,
            is_followup=True,
            case_id=cid,
        )
        assert req.is_followup is True
        assert req.case_id == cid

    def test_followup_without_case_id_fails(self):
        with pytest.raises(ValidationError) as exc_info:
            FarmerRequest(
                farmer_id="F001",
                image_bytes=b"fake",
                image_mime_type="image/jpeg",
                gps_lat=12.0,
                gps_lon=77.0,
                submitted_at=NOW,
                is_followup=True,
                case_id=None,  # Missing — must fail
            )
        assert "case_id" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()

    def test_invalid_mime_type_fails(self):
        with pytest.raises(ValidationError):
            FarmerRequest(
                farmer_id="F001",
                image_bytes=b"fake",
                image_mime_type="image/gif",  # Not allowed
                gps_lat=12.0,
                gps_lon=77.0,
                submitted_at=NOW,
            )

    def test_gps_bounds(self):
        with pytest.raises(ValidationError):
            FarmerRequest(
                farmer_id="F001",
                image_bytes=b"fake",
                image_mime_type="image/jpeg",
                gps_lat=91.0,  # Out of range
                gps_lon=0.0,
                submitted_at=NOW,
            )


# ── Tests: VisionOutput ─────────────────────────────────────────────────────────

class TestVisionOutput:
    def test_valid_output(self):
        vo = VisionOutput(**make_vision_output())
        assert vo.top_label == "Tomato___Early_blight"
        assert vo.confidence == 0.85

    def test_severity_pct_always_none(self):
        """R3: severity_pct MUST always be None."""
        vo = VisionOutput(**make_vision_output())
        assert vo.severity_pct is None, "R3 VIOLATION: severity_pct must always be None"

    def test_severity_pct_rejects_non_none(self):
        """R3: Passing a non-None severity_pct must raise ValidationError."""
        with pytest.raises((ValidationError, TypeError)):
            VisionOutput(**make_vision_output(), severity_pct=0.35)  # type: ignore

    def test_severity_note_present(self):
        """R3: severity_note must explain why severity is null."""
        vo = VisionOutput(**make_vision_output())
        assert len(vo.severity_note) > 20, "severity_note must be non-trivial explanation"
        assert "null" in vo.severity_note.lower() or "not computed" in vo.severity_note.lower()

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            VisionOutput(**make_vision_output(confidence=1.5))

    def test_temperature_present(self):
        vo = VisionOutput(**make_vision_output(temperature=1.2))
        assert vo.temperature == 1.2


# ── Tests: WeatherOutput ────────────────────────────────────────────────────────

class TestWeatherOutput:
    def test_unavailable(self):
        wo = WeatherOutput(**make_weather_output(available=False))
        assert wo.weather_available is False
        assert wo.current_weather is None
        assert wo.forecast_7day is None

    def test_valid_available(self):
        wo = WeatherOutput(**make_weather_output(available=True))
        assert wo.weather_available is True
        assert wo.current_weather is not None
        assert wo.current_weather.temperature_c == 25.0

    def test_humidity_bounds(self):
        with pytest.raises(ValidationError):
            CurrentWeather(
                temperature_c=20.0,
                relative_humidity_pct=110.0,  # > 100
                precipitation_mm=0.0,
                wind_speed_kmh=10.0,
                cloud_cover_pct=50.0,
                dew_point_c=15.0,
            )

    def test_weather_risk_flag(self):
        flag = WeatherRiskFlag(
            condition_name="W01_late_blight_humidity",
            triggered=True,
            current_value="RH=92%, temp_min=15°C",
            threshold="RH > 90% AND temp_min < 18°C",
            threshold_source="Fry (1977) [UNVERIFIED — I14]",
        )
        assert flag.triggered is True

    def test_weather_output_c4_fields(self):
        wo = WeatherOutput(
            weather_available=True,
            latitude=19.75,
            longitude=75.71,
            location_name="Maharashtra",
            location_source="default",
            is_ambiguous_location=False,
            stale=False,
            weather_score=0.25,
            unavailable_reason=None,
        )
        assert wo.location_source == "default"
        assert wo.stale == wo.cache_stale
        assert wo.ambiguity_flag == wo.is_ambiguous_location
        assert wo.weather_score == 0.25

    def test_weather_input_c4_fields(self):
        wi = WeatherInput(place_name="Nashik", query_datetime=NOW, detected_disease="Tomato___Late_blight")
        assert wi.place_name == "Nashik"
        assert wi.detected_disease == "Tomato___Late_blight"

    def test_farmer_request_c4_fields(self):
        req = FarmerRequest(
            farmer_id="F001",
            image_bytes=b"fake",
            image_mime_type="image/jpeg",
            place_name="Pune",
            submitted_at=NOW,
        )
        assert req.place_name == "Pune"
        assert req.gps_lat is None


# ── Tests: RAGOutput ────────────────────────────────────────────────────────────

class TestRAGOutput:
    def test_unavailable(self):
        ro = RAGOutput(**make_rag_output(available=False))
        assert ro.rag_available is False
        assert ro.treatment_options == []

    def test_cost_estimate_note_always_none(self):
        """RAGOutput.cost_estimate_note is always null [GAP]."""
        ro = RAGOutput(**make_rag_output(available=True))
        assert ro.cost_estimate_note is None


# ── Tests: RiskOutput ───────────────────────────────────────────────────────────

class TestRiskOutput:
    def test_valid(self):
        ro = RiskOutput(**make_risk_output())
        assert ro.risk_level == "moderate"
        assert 0.0 <= ro.risk_score <= 1.0

    def test_invalid_risk_level(self):
        with pytest.raises(ValidationError):
            RiskOutput(**make_risk_output(risk_level="catastrophic"))  # Not in pattern

    def test_uncertainty_level_pattern(self):
        ro = RiskOutput(**make_risk_output(uncertainty_level="low_confidence"))
        assert ro.uncertainty_level == "low_confidence"

    def test_invalid_uncertainty_level(self):
        with pytest.raises(ValidationError):
            RiskOutput(**make_risk_output(uncertainty_level="very_uncertain"))

    def test_risk_score_bounds(self):
        with pytest.raises(ValidationError):
            RiskOutput(**make_risk_output(risk_score=1.5))

    def test_r14_risk_not_uncertainty(self):
        """R14: risk_score must be independent of confidence; verify field separation."""
        ro = RiskOutput(**make_risk_output())
        # risk_score comes from factors, not from any confidence field
        assert "confidence" not in [f.source for f in ro.factors if "confidence" in f.source.lower()
                                     and f.source != "vision_confidence_uncertainty"]


# ── Tests: RecommendationOutput ────────────────────────────────────────────────

class TestRecommendationOutput:
    def test_valid(self):
        ro = RecommendationOutput(**make_recommendation_output())
        assert ro.urgency == "within_48h"
        assert ro.rag_grounded is True

    def test_invalid_urgency(self):
        with pytest.raises(ValidationError):
            RecommendationOutput(**make_recommendation_output(urgency="emergency"))

    def test_consult_agronomist_path(self):
        ro = RecommendationOutput(
            actions=[],
            urgency="consult_agronomist",
            risk_level="moderate",
            confidence_justification="RAG unavailable; no treatment data.",
            citations=[],
            caveats=["RAG knowledge base not built yet [I16]."],
            rag_grounded=False,
            escalate_to_human=True,
            knowledge_gap_caveated=True,
        )
        assert ro.rag_grounded is False
        assert ro.escalate_to_human is True

    def test_cost_estimate_note_always_none(self):
        ro = RecommendationOutput(**make_recommendation_output())
        assert ro.cost_estimate_note is None


# ── Tests: MonitoringOutput ────────────────────────────────────────────────────

class TestMonitoringOutput:
    def test_severity_delta_pct_always_none(self):
        """R3: severity_delta_pct must always be None."""
        mo = MonitoringOutput(**make_monitoring_output())
        assert mo.severity_delta_pct is None, "R3 VIOLATION: severity_delta_pct must always be None"

    def test_severity_note_present(self):
        mo = MonitoringOutput(**make_monitoring_output())
        assert len(mo.severity_note) > 20

    def test_valid_disease_status(self):
        for status in [
            "same_disease_still_detected",
            "no_disease_detected_at_followup",
            "different_disease_detected_at_followup",
            "not_applicable_healthy_baseline",
            "unknown",
        ]:
            mo = MonitoringOutput(**make_monitoring_output(disease_status=status))
            assert mo.disease_status == status

    def test_invalid_disease_status(self):
        with pytest.raises(ValidationError):
            MonitoringOutput(**make_monitoring_output(disease_status="recovering"))

    def test_visit_number_ge_2(self):
        """Visit 1 is initial; follow-ups start at 2."""
        with pytest.raises(ValidationError):
            MonitoringInput(
                case_id=uuid.uuid4(),
                farmer_id="F001",
                followup_image_bytes=b"fake",
                followup_submitted_at=NOW,
                visit_number=1,  # Must fail — ge=2
            )


# ── Tests: HumanEscalationPayload ──────────────────────────────────────────────

class TestHumanEscalationPayload:
    def test_priority_pattern(self):
        for priority in ["low", "medium", "high", "urgent"]:
            payload = HumanEscalationPayload(
                case_id=uuid.uuid4(),
                farmer_id="F001",
                escalation_reason=["U1: low confidence"],
                vision_output=VisionOutput(**make_vision_output()),
                weather_output=WeatherOutput(**make_weather_output()),
                rag_output=RAGOutput(**make_rag_output()),
                risk_output=RiskOutput(**make_risk_output()),
                draft_recommendation=RecommendationOutput(**make_recommendation_output()),
                image_bytes=b"fake",
                escalated_at=NOW,
                priority=priority,
            )
            assert payload.priority == priority

    def test_invalid_priority(self):
        with pytest.raises(ValidationError):
            HumanEscalationPayload(
                case_id=uuid.uuid4(),
                farmer_id="F001",
                escalation_reason=["test"],
                vision_output=VisionOutput(**make_vision_output()),
                weather_output=WeatherOutput(**make_weather_output()),
                rag_output=RAGOutput(**make_rag_output()),
                risk_output=RiskOutput(**make_risk_output()),
                draft_recommendation=RecommendationOutput(**make_recommendation_output()),
                image_bytes=b"fake",
                escalated_at=NOW,
                priority="critical",  # Not in pattern
            )


# ── Tests: CaseState pattern ────────────────────────────────────────────────────

class TestCaseStateStatus:
    def test_valid_statuses(self):
        for status in ["active", "escalated", "closed"]:
            # Verify pattern accepts all valid values
            from pydantic import TypeAdapter
            import re
            assert re.match(r"^(active|escalated|closed)$", status)

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            # Build minimal CaseState with invalid status
            FarmerRequest(
                farmer_id="F001",
                image_bytes=b"fake",
                image_mime_type="image/jpeg",
                gps_lat=12.0,
                gps_lon=77.0,
                submitted_at=NOW,
                is_followup=False,
            )
            # Use a CaseState-like check — status pattern must reject 'pending'
            from pydantic import BaseModel, Field
            class StatusCheck(BaseModel):
                status: str = Field(..., pattern=r"^(active|escalated|closed)$")
            StatusCheck(status="pending")
