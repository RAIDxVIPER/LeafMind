"""
tests/unit/test_orchestrator.py
===============================
Unit tests for LeafMind Orchestrator (Stage C7).

Requirements:
  - Zero network calls, zero real keys, zero GPU requirement
  - Mocks/stubs for external dependencies
  - Tests:
      * Happy path end-to-end for a diseased class
      * Healthy class path
      * Squash___Powdery_mildew path (RAG gap)
      * Single-class crop (can_diagnose=False) path
      * Graceful degradation on weather down / RAG not found / LLM down / vision error
      * Escalation triggers T1–T11 (reachable T1–T7 and synthetic T8–T11 with documentation)
      * Checkpoint persistence: run, reload by case_id, verify matching response
      * No API key or traceback leaks in final response or persisted state
      * Threshold changes through config take effect
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, List, Optional
import uuid

import pytest
import yaml

from src.orchestrator.escalation import evaluate_escalation
from src.orchestrator.pipeline import load_case_response, load_case_state, run_case
from src.orchestrator.state import (
    EscalationSummary,
    FinalResponse,
    sanitize_error,
)
from src.orchestrator.store import (
    get_checkpointer,
    load_case_record,
    load_final_response,
    save_case_record,
    save_final_response,
)
from src.schemas.schemas import (
    CaseState,
    Citation,
    ClassProb,
    CurrentWeather,
    DailyForecast,
    FarmerRequest,
    ImageQualityFlags,
    MonitoringOutput,
    RAGOutput,
    RecommendationOutput,
    RiskFactor,
    RiskOutput,
    VisionOutput,
    WeatherOutput,
    WeatherRiskFlag,
)

# 1x1 dummy JPEG bytes for tests
DUMMY_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08"
    b"\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13"
    b"\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05"
    b"\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07"
    b"\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
)


# =============================================================================
# Mock Agents
# =============================================================================

class MockVisionAgent:
    def __init__(
        self,
        top_label: str = "Tomato___Late_blight",
        confidence: float = 0.92,
        can_diagnose: bool = True,
        single_class_crop: bool = False,
        ood_flag: bool = False,
        escalate_to_human: bool = False,
        blurry: bool = False,
        low_res: bool = False,
        top_k: Optional[List[ClassProb]] = None,
        raise_error: Optional[Exception] = None,
    ):
        self.top_label = top_label
        self.confidence = confidence
        self.can_diagnose = can_diagnose
        self.single_class_crop = single_class_crop
        self.ood_flag = ood_flag
        self.escalate_to_human = escalate_to_human
        self.blurry = blurry
        self.low_res = low_res
        self.top_k = top_k or [
            ClassProb(label=top_label, prob=confidence),
            ClassProb(label="Tomato___Early_blight", prob=round(1.0 - confidence, 4)),
        ]
        self.raise_error = raise_error

    def predict(self, image_bytes: bytes, crop: Optional[str] = None) -> VisionOutput:
        if self.raise_error:
            raise self.raise_error

        # If single-class crop requested
        if crop and crop.lower() in ("squash", "blueberry", "cherry"):
            return VisionOutput(
                top_label=self.top_label,
                confidence=self.confidence,
                top_k_probs=self.top_k,
                can_diagnose=False,
                single_class_crop=True,
                crop=crop,
                severity_pct=None,
                image_quality_flags=ImageQualityFlags(low_resolution=self.low_res, blurry=self.blurry),
                ood_flag=self.ood_flag,
                escalate_to_human=True,
                model_version="mock_v1",
                temperature=1.0,
            )

        return VisionOutput(
            top_label=self.top_label,
            confidence=self.confidence,
            top_k_probs=self.top_k,
            can_diagnose=self.can_diagnose,
            single_class_crop=self.single_class_crop,
            crop=crop,
            severity_pct=None,
            image_quality_flags=ImageQualityFlags(low_resolution=self.low_res, blurry=self.blurry),
            ood_flag=self.ood_flag,
            escalate_to_human=self.escalate_to_human,
            model_version="mock_v1",
            temperature=1.0,
        )


class MockWeatherAgent:
    def __init__(
        self,
        available: bool = True,
        triggered_rules: Optional[List[str]] = None,
        raise_error: Optional[Exception] = None,
        weather_score: float = 0.20,
    ):
        self.available = available
        self.triggered_rules = triggered_rules or ["W01"]
        self.raise_error = raise_error
        self.weather_score = weather_score

    def get_weather(self, latitude=None, longitude=None, place_name=None, detected_disease=None, **kwargs) -> WeatherOutput:
        if self.raise_error:
            raise self.raise_error

        if not self.available:
            return WeatherOutput(
                weather_available=False,
                weather_partial=False,
                disease_risk_flags=[],
                weather_data_source=None,
                unavailable_reason="Simulated weather service outage",
            )

        flags = [
            WeatherRiskFlag(
                condition_name=r,
                triggered=True,
                current_value="mock_value",
                threshold="mock_threshold",
                threshold_source="mock_citation [UNVERIFIED — I14]",
            )
            for r in self.triggered_rules
        ]

        return WeatherOutput(
            weather_available=True,
            weather_partial=False,
            current_weather=CurrentWeather(
                temperature_c=22.0,
                relative_humidity_pct=92.0,
                precipitation_mm=2.5,
                wind_speed_kmh=10.0,
                cloud_cover_pct=80.0,
                dew_point_c=20.5,
            ),
            disease_risk_flags=flags,
            weather_data_source="mock_open_meteo",
            weather_score=self.weather_score,
        )


class MockRAGAgent:
    def __init__(
        self,
        found: bool = True,
        kb_gap: bool = False,
        raise_error: Optional[Exception] = None,
    ):
        self.found = found
        self.kb_gap = kb_gap
        self.raise_error = raise_error

    def query_candidate(self, model_class: str, crop: Optional[str] = None, include_doses: bool = False) -> RAGOutput:
        if self.raise_error:
            raise self.raise_error

        if self.kb_gap or not self.found:
            return RAGOutput(
                crop=crop or "Unknown",
                model_class=model_class,
                disease_name=model_class.replace("___", " "),
                found=False,
                kb_gap=True,
                reason="Simulated KB Gap",
                rag_available=False,
                knowledge_gap_flag=True,
            )

        return RAGOutput(
            crop=crop or "Tomato",
            model_class=model_class,
            disease_name="Tomato Late Blight",
            found=True,
            kb_gap=False,
            what_it_is="Late blight is caused by the oomycete Phytophthora infestans.",
            why_it_happens="Cool, wet conditions favour sporulation.",
            what_to_do_first="Remove infected lower foliage immediately.",
            citations=[
                Citation(tag="FAO2019", title="FAO Tomato Manual", url="https://example.org/fao")
            ],
            chemical_options=[],
            rag_available=True,
            knowledge_gap_flag=False,
        )


class MockRiskAgent:
    def __init__(
        self,
        risk_level: str = "moderate",
        risk_score: float = 0.55,
        raise_error: Optional[Exception] = None,
    ):
        self.risk_level = risk_level
        self.risk_score = risk_score
        self.raise_error = raise_error

    def score(self, vision, weather, rag, monitoring_disease_status=None) -> RiskOutput:
        if self.raise_error:
            raise self.raise_error

        return RiskOutput(
            risk_score=self.risk_score,
            risk_level=self.risk_level,
            uncertainty_level="high_confidence" if vision.confidence >= 0.70 else "low_confidence",
            factors=[
                RiskFactor(source="threat", value="mock_threat_val", contribution=0.35, tag="mock_threat"),
                RiskFactor(source="weather", value="mock_weather_val", contribution=0.20, tag="mock_weather"),
            ],
            escalate_to_human=bool(vision.confidence < 0.60 or self.risk_level == "severe"),
            uncertainty_signals_triggered=[] if vision.confidence >= 0.60 else ["U1_low_confidence"],
            weather_unavailable=not weather.weather_available,
            rag_unavailable=not rag.rag_available,
            progression_note="Mock progression note",
            risk_computed_at=datetime.now(timezone.utc),
        )


class MockRecommendationAgent:
    def __init__(
        self,
        urgency: str = "within_24h",
        llm_used: bool = True,
        raise_error: Optional[Exception] = None,
    ):
        self.urgency = urgency
        self.llm_used = llm_used
        self.raise_error = raise_error

    def recommend(self, vision=None, rag=None, weather=None, risk=None, include_doses=False, **kwargs) -> RecommendationOutput:
        if self.raise_error:
            raise self.raise_error

        return RecommendationOutput(
            summary="Promptly remove affected foliage and treat with approved copper fungicide.",
            urgency=self.urgency,
            what_to_do_first="Remove infected lower leaves to limit fungal spread.",
            steps=["Remove infected leaves.", "Avoid overhead irrigation."],
            actions=["Remove infected leaves.", "Avoid overhead irrigation."],
            citations=["FAO Tomato Manual"],
            rag_grounded=True,
            llm_used=self.llm_used,
            escalate_to_human=bool(risk.escalate_to_human if risk else False),
        )


# =============================================================================
# Test Suites
# =============================================================================

class TestHappyPath:
    """Happy path end-to-end for a diseased class."""

    def test_diseased_happy_path(self):
        v_mock = MockVisionAgent(top_label="Tomato___Late_blight", confidence=0.92)
        w_mock = MockWeatherAgent(available=True)
        r_mock = MockRAGAgent(found=True)
        rk_mock = MockRiskAgent(risk_level="moderate", risk_score=0.55)
        rec_mock = MockRecommendationAgent(urgency="within_24h", llm_used=True)

        resp = run_case(
            image=DUMMY_JPEG,
            crop="Tomato",
            location=(19.75, 75.71),
            include_doses=False,
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            risk_agent=rk_mock,
            recommendation_agent=rec_mock,
            use_memory=True,
        )

        assert isinstance(resp, FinalResponse)
        assert str(resp.case_id)
        assert resp.crop == "Tomato"
        assert resp.severity_pct is None  # D2 / R3 invariant
        assert "Severity percentage is not computed" in resp.severity_note

        # Sub-summaries populated
        assert resp.vision is not None
        assert resp.vision.top_label == "Tomato___Late_blight"
        assert resp.vision.confidence == 0.92
        assert resp.vision.can_diagnose is True

        assert resp.weather is not None
        assert resp.weather.weather_available is True
        assert "W01" in resp.weather.triggered_rules

        assert resp.rag is not None
        assert resp.rag.found is True
        assert len(resp.rag.sources) == 1

        assert resp.risk is not None
        assert resp.risk.risk_level == "moderate"

        assert resp.recommendation is not None
        assert resp.recommendation.urgency == "within_24h"
        assert resp.recommendation.llm_used is True

        # Timing measured per node
        assert len(resp.timing_ms) > 0
        assert "vision_runner" in resp.timing_ms
        assert "weather_runner" in resp.timing_ms
        assert "rag_runner" in resp.timing_ms


class TestSpecialPaths:
    """Healthy class, RAG gap, and single-class crop paths."""

    def test_healthy_class_path(self):
        v_mock = MockVisionAgent(top_label="Tomato___healthy", confidence=0.96)
        w_mock = MockWeatherAgent(available=True)
        r_mock = MockRAGAgent(found=False, kb_gap=False)
        rk_mock = MockRiskAgent(risk_level="low", risk_score=0.10)
        rec_mock = MockRecommendationAgent(urgency="monitor", llm_used=False)

        resp = run_case(
            image=DUMMY_JPEG,
            crop="Tomato",
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            risk_agent=rk_mock,
            recommendation_agent=rec_mock,
            use_memory=True,
        )

        assert resp.vision.top_label == "Tomato___healthy"
        assert resp.risk.risk_level == "low"
        assert resp.recommendation.urgency == "monitor"
        assert resp.severity_pct is None

    def test_squash_powdery_mildew_rag_gap(self):
        """Squash___Powdery_mildew is an expected KB gap."""
        v_mock = MockVisionAgent(top_label="Squash___Powdery_mildew", confidence=0.88)
        w_mock = MockWeatherAgent(available=True)
        r_mock = MockRAGAgent(found=False, kb_gap=True)
        rk_mock = MockRiskAgent(risk_level="moderate", risk_score=0.50)
        rec_mock = MockRecommendationAgent(urgency="consult_agronomist", llm_used=False)

        resp = run_case(
            image=DUMMY_JPEG,
            crop="Squash",
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            risk_agent=rk_mock,
            recommendation_agent=rec_mock,
            use_memory=True,
        )

        assert resp.rag.kb_gap is True
        assert "T6" in resp.escalation.triggered_triggers
        assert resp.escalation.escalate_to_human is True
        assert resp.recommendation.urgency == "consult_agronomist"

    def test_single_class_crop_path(self):
        """Single-class crop sets can_diagnose=False and triggers SINGLE_CLASS_CROP escalation."""
        v_mock = MockVisionAgent(top_label="Squash___Powdery_mildew", confidence=0.95)
        w_mock = MockWeatherAgent(available=True)
        r_mock = MockRAGAgent(found=False, kb_gap=True)
        rk_mock = MockRiskAgent(risk_level="moderate", risk_score=0.50)
        rec_mock = MockRecommendationAgent(urgency="consult_agronomist")

        resp = run_case(
            image=DUMMY_JPEG,
            crop="Squash",  # Single-class crop
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            risk_agent=rk_mock,
            recommendation_agent=rec_mock,
            use_memory=True,
        )

        assert resp.vision.can_diagnose is False
        assert resp.vision.single_class_crop is True
        assert "SINGLE_CLASS_CROP" in resp.escalation.triggered_triggers
        assert resp.escalation.priority in ("high", "urgent")


class TestGracefulDegradation:
    """Degradation tests per spec 4e error handling table."""

    def test_weather_down_degrades_gracefully(self):
        v_mock = MockVisionAgent()
        w_mock = MockWeatherAgent(raise_error=ConnectionError("Open-Meteo connection timed out"))
        r_mock = MockRAGAgent()
        rk_mock = MockRiskAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG,
            crop="Tomato",
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            risk_agent=rk_mock,
            recommendation_agent=rec_mock,
            use_memory=True,
        )

        assert resp.partial_result is True
        assert "WEATHER_UNAVAILABLE" in resp.flags
        assert resp.weather.weather_available is False
        assert "weather_runner" in resp.errors

    def test_rag_not_found_degrades_gracefully(self):
        v_mock = MockVisionAgent()
        w_mock = MockWeatherAgent()
        r_mock = MockRAGAgent(raise_error=RuntimeError("Vector DB index corrupted"))
        rk_mock = MockRiskAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG,
            crop="Tomato",
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            risk_agent=rk_mock,
            recommendation_agent=rec_mock,
            use_memory=True,
        )

        assert resp.partial_result is True
        assert "RAG_UNAVAILABLE" in resp.flags
        assert resp.rag.found is False
        assert "rag_runner" in resp.errors

    def test_llm_down_degrades_gracefully(self):
        v_mock = MockVisionAgent()
        w_mock = MockWeatherAgent()
        r_mock = MockRAGAgent()
        rk_mock = MockRiskAgent()
        rec_mock = MockRecommendationAgent(raise_error=TimeoutError("Gemini API call timed out"))

        resp = run_case(
            image=DUMMY_JPEG,
            crop="Tomato",
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            risk_agent=rk_mock,
            recommendation_agent=rec_mock,
            use_memory=True,
        )

        assert resp.partial_result is True
        assert "RECOMMENDATION_FALLBACK" in resp.flags
        assert resp.recommendation.urgency == "consult_agronomist"
        assert resp.recommendation.llm_used is False

    def test_vision_error_degrades_gracefully(self):
        v_mock = MockVisionAgent(raise_error=RuntimeError("CUDA out of memory in model forward"))
        w_mock = MockWeatherAgent()
        r_mock = MockRAGAgent()
        rk_mock = MockRiskAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG,
            crop="Tomato",
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            risk_agent=rk_mock,
            recommendation_agent=rec_mock,
            use_memory=True,
        )

        assert resp.fatal_error == "vision_unavailable"
        assert "FATAL_VISION_UNAVAILABLE" in resp.flags
        assert "vision_runner" in resp.errors

    def test_invalid_image_degrades_gracefully(self):
        resp = run_case(
            image=b"",  # Empty image
            crop="Tomato",
            use_memory=True,
        )

        assert resp.fatal_error == "invalid_image"
        assert "FATAL_INVALID_IMAGE" in resp.flags


class TestEscalationTriggers:
    """Escalation triggers T1–T11 evaluation."""

    def test_trigger_t1_low_confidence(self):
        v = VisionOutput(
            top_label="Tomato___Late_blight",
            confidence=0.55,  # < 0.60
            top_k_probs=[ClassProb(label="Tomato___Late_blight", prob=0.55), ClassProb(label="Tomato___Early_blight", prob=0.30)],
            severity_pct=None,
            severity_note="",
            image_quality_flags=ImageQualityFlags(),
            ood_flag=False,
            model_version="v1",
            temperature=1.0,
        )
        esc = evaluate_escalation(vision=v)
        assert "T1" in esc.triggered_triggers
        assert esc.escalate_to_human is True
        assert esc.priority == "medium"

    def test_trigger_t2_narrow_margin(self):
        v = VisionOutput(
            top_label="Tomato___Late_blight",
            confidence=0.70,
            top_k_probs=[ClassProb(label="Tomato___Late_blight", prob=0.52), ClassProb(label="Tomato___Early_blight", prob=0.48)],  # gap 0.04 < 0.15
            severity_pct=None,
            severity_note="",
            image_quality_flags=ImageQualityFlags(),
            ood_flag=False,
            model_version="v1",
            temperature=1.0,
        )
        esc = evaluate_escalation(vision=v)
        assert "T2" in esc.triggered_triggers
        assert esc.escalate_to_human is True
        assert esc.priority == "medium"

    def test_trigger_t3_low_resolution(self):
        v = VisionOutput(
            top_label="Tomato___Late_blight",
            confidence=0.85,
            top_k_probs=[ClassProb(label="Tomato___Late_blight", prob=0.85), ClassProb(label="Tomato___Early_blight", prob=0.10)],
            severity_pct=None,
            severity_note="",
            image_quality_flags=ImageQualityFlags(low_resolution=True),
            ood_flag=False,
            model_version="v1",
            temperature=1.0,
        )
        esc = evaluate_escalation(vision=v)
        assert "T3" in esc.triggered_triggers
        assert esc.escalate_to_human is True

    def test_trigger_t4_blurry(self):
        v = VisionOutput(
            top_label="Tomato___Late_blight",
            confidence=0.85,
            top_k_probs=[ClassProb(label="Tomato___Late_blight", prob=0.85), ClassProb(label="Tomato___Early_blight", prob=0.10)],
            severity_pct=None,
            severity_note="",
            image_quality_flags=ImageQualityFlags(blurry=True),
            ood_flag=False,
            model_version="v1",
            temperature=1.0,
        )
        esc = evaluate_escalation(vision=v)
        assert "T4" in esc.triggered_triggers
        assert esc.escalate_to_human is True

    def test_trigger_t5_ood(self):
        v = VisionOutput(
            top_label="Tomato___Late_blight",
            confidence=0.35,  # < 0.40
            top_k_probs=[ClassProb(label="Tomato___Late_blight", prob=0.35)],
            severity_pct=None,
            severity_note="",
            image_quality_flags=ImageQualityFlags(),
            ood_flag=True,
            model_version="v1",
            temperature=1.0,
        )
        esc = evaluate_escalation(vision=v)
        assert "T5" in esc.triggered_triggers
        assert esc.priority == "urgent"  # Spec 4d-iii: T5 -> urgent

    def test_trigger_t6_rag_gap(self):
        r = RAGOutput(
            crop="Squash",
            model_class="Squash___Powdery_mildew",
            found=False,
            kb_gap=True,
            rag_available=False,
            knowledge_gap_flag=True,
        )
        esc = evaluate_escalation(rag=r)
        assert "T6" in esc.triggered_triggers
        assert esc.escalate_to_human is True

    def test_trigger_t7_severe_risk_async(self):
        rk = RiskOutput(
            risk_score=0.85,
            risk_level="severe",
            uncertainty_level="high_confidence",
            factors=[],
            escalate_to_human=False,
            progression_note="Severe risk progression observed.",
            risk_computed_at=datetime.now(timezone.utc),
        )
        esc = evaluate_escalation(risk=rk)
        assert "T7" in esc.triggered_triggers
        assert esc.priority == "high"
        assert esc.payload is not None

    def test_monitoring_triggers_t8_to_t11_reachable_with_synthetic_monitoring(self):
        """
        Documenting C7 vs C8 status:
        In C7 initial flow, MonitoringAgent (C8) is a marked no-op slot passing history=None.
        Therefore, T8–T11 do not fire during initial single-visit requests.
        However, evaluate_escalation implements T8–T11 in full. This test passes synthetic
        MonitoringOutput to verify triggers T8–T11 fire with correct priorities.
        """
        # T8: Different disease detected at follow-up -> urgent
        mon_t8 = MonitoringOutput(
            case_id=uuid.uuid4(),
            visit_number=2,
            baseline_label="Tomato___Early_blight",
            baseline_confidence=0.85,
            followup_label="Tomato___Late_blight",
            followup_confidence=0.80,
            label_same=False,
            disease_still_detected=True,
            new_label_is_healthy=False,
            confidence_raw_delta=-0.05,
            disease_status="different_disease_detected_at_followup",
            escalate_to_human=True,
            followup_vision=MockVisionAgent().predict(DUMMY_JPEG),
        )
        esc_t8 = evaluate_escalation(monitoring=mon_t8)
        assert "T8" in esc_t8.triggered_triggers
        assert esc_t8.priority == "urgent"

        # T9: Same disease persists on visit 2 -> high; visit 3 -> urgent
        mon_t9_v2 = mon_t8.model_copy(update={"disease_status": "same_disease_still_detected", "visit_number": 2})
        esc_t9_v2 = evaluate_escalation(monitoring=mon_t9_v2)
        assert "T9" in esc_t9_v2.triggered_triggers
        assert esc_t9_v2.priority == "high"

        mon_t9_v3 = mon_t8.model_copy(update={"disease_status": "same_disease_still_detected", "visit_number": 3})
        esc_t9_v3 = evaluate_escalation(monitoring=mon_t9_v3)
        assert "T9" in esc_t9_v3.triggered_triggers
        assert esc_t9_v3.priority == "urgent"

        # T10: Poor quality follow-up -> medium
        mon_t10 = mon_t8.model_copy(update={"disease_status": "same_disease_still_detected", "poor_quality_flag": True, "visit_number": 1})
        esc_t10 = evaluate_escalation(monitoring=mon_t10)
        assert "T10" in esc_t10.triggered_triggers

        # T11: Crop mismatch -> high
        mon_t11 = mon_t8.model_copy(update={"disease_status": "same_disease_still_detected", "crop_mismatch_flag": True, "visit_number": 1})
        esc_t11 = evaluate_escalation(monitoring=mon_t11)
        assert "T11" in esc_t11.triggered_triggers
        assert esc_t11.priority in ("high", "urgent")


class TestCheckpointPersistence:
    """Checkpointing and reloading by case_id."""

    def test_run_and_reload_case(self):
        from src.orchestrator.store import close_sqlite_connections
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                tmp_path = Path(tmp_dir)
                cases_dir = tmp_path / "cases"
                db_path = tmp_path / "checkpoints.db"

                case_id = uuid.uuid4()
                v_mock = MockVisionAgent(top_label="Tomato___Late_blight", confidence=0.89)
                w_mock = MockWeatherAgent()
                r_mock = MockRAGAgent()
                rk_mock = MockRiskAgent()
                rec_mock = MockRecommendationAgent()

                # Run case
                resp = run_case(
                    image=DUMMY_JPEG,
                    crop="Tomato",
                    case_id=case_id,
                    vision_agent=v_mock,
                    weather_agent=w_mock,
                    rag_agent=r_mock,
                    risk_agent=rk_mock,
                    recommendation_agent=rec_mock,
                    db_path=db_path,
                    case_store_dir=cases_dir,
                )

                assert resp.case_id == case_id

                # Reload FinalResponse by case_id
                reloaded_resp = load_case_response(case_id, case_store_dir=cases_dir)
                assert reloaded_resp is not None
                assert reloaded_resp.case_id == case_id
                assert reloaded_resp.vision.top_label == "Tomato___Late_blight"
                assert reloaded_resp.severity_pct is None

                # Reload CaseState by case_id
                reloaded_case = load_case_state(case_id, case_store_dir=cases_dir)
                assert reloaded_case is not None
                assert reloaded_case.case_id == case_id
                assert reloaded_case.initial_vision.top_label == "Tomato___Late_blight"
            finally:
                close_sqlite_connections()


class TestSecurityAndSanitization:
    """Ensure no API keys, secrets, or raw tracebacks leak into user-facing outputs."""

    def test_sanitize_error_strips_secrets(self):
        leaky_msg = (
            "Error calling API with key AIzaSyB123456789012345678901234567890\n"
            "OpenAI token sk-abc1234567890abcdefghijklmnopqrstuvwxyz\n"
            "Authorization: Bearer mySecretToken123\n"
            "Traceback (most recent call last):\n"
            "  File 'c:/secret/path.py', line 12, in <module>\n"
            "ValueError: Invalid key"
        )
        clean = sanitize_error(leaky_msg)

        assert "AIzaSyB" not in clean
        assert "sk-abc" not in clean
        assert "mySecretToken123" not in clean
        assert "Traceback" not in clean
        assert "[REDACTED_API_KEY]" in clean
        assert "ValueError: Invalid key" in clean

    def test_orchestrator_error_sanitization(self):
        secret_exc = RuntimeError("Failure connecting to API key AIzaSyD987654321098765432109876543210")
        w_mock = MockWeatherAgent(raise_error=secret_exc)

        resp = run_case(
            image=DUMMY_JPEG,
            crop="Tomato",
            weather_agent=w_mock,
            use_memory=True,
        )

        # Confirm no API key in FinalResponse errors
        error_text = json.dumps(resp.errors)
        assert "AIzaSyD" not in error_text
        assert "[REDACTED_API_KEY]" in error_text


class TestConfigThresholdChanges:
    """Verify that changes in orchestrator_config.yaml take effect."""

    def test_threshold_change_takes_effect(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "escalation": {
                        "confidence_threshold": 0.95,  # Increased from 0.60
                        "prob_gap_threshold": 0.15,
                        "ood_threshold": 0.40,
                    }
                },
                f,
            )
            custom_config_path = f.name

        try:
            v_mock = MockVisionAgent(confidence=0.85)  # Would not trigger at 0.60, but triggers at 0.95!
            w_mock = MockWeatherAgent()
            r_mock = MockRAGAgent()
            rk_mock = MockRiskAgent()
            rec_mock = MockRecommendationAgent()

            resp = run_case(
                image=DUMMY_JPEG,
                crop="Tomato",
                vision_agent=v_mock,
                weather_agent=w_mock,
                rag_agent=r_mock,
                risk_agent=rk_mock,
                recommendation_agent=rec_mock,
                config_path=custom_config_path,
                use_memory=True,
            )

            # At threshold=0.95, confidence 0.85 triggers T1
            assert "T1" in resp.escalation.triggered_triggers
            assert resp.escalation.escalate_to_human is True

        finally:
            Path(custom_config_path).unlink(missing_ok=True)


class TestRiskFormulaExactCompliance:
    """Verify exact formula compliance: R = 0.70*T + 0.20*W + 0.10*H through run_case."""

    @pytest.mark.parametrize(
        "label,crop,expected_t,weather_score,history_score",
        [
            ("Corn___Northern_Leaf_Blight", "Corn", 0.85, 0.20, 0.0),   # HIGH tier
            ("Tomato___Early_blight", "Tomato", 0.50, 0.20, 0.0),        # MEDIUM tier
            ("Squash___Powdery_mildew", "Squash", 0.50, 0.20, 0.0),      # GAP row (DISEASE_DETECTED fallback)
        ],
    )
    def test_risk_formula_exact_recompute(self, label, crop, expected_t, weather_score, history_score):
        v_mock = MockVisionAgent(top_label=label, confidence=0.98)
        w_mock = MockWeatherAgent(weather_score=weather_score)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG,
            crop=crop,
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            use_memory=True,
        )

        expected_r = 0.70 * expected_t + 0.20 * weather_score + 0.10 * history_score
        assert resp.risk is not None
        assert abs(resp.risk.risk_score - expected_r) < 1e-6
        # Assert individual factor contributions
        factors_dict = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors_dict.get("threat_table", 0.0) - round(0.70 * expected_t, 6)) < 1e-6
        assert abs(factors_dict.get("weather", 0.0) - round(0.20 * weather_score, 6)) < 1e-6
        assert abs(factors_dict.get("history", 0.0) - round(0.10 * history_score, 6)) < 1e-6

