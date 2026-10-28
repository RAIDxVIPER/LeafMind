"""
tests/unit/test_monitoring_agent.py
===================================
Unit tests for LeafMind Stage C8 Monitoring Agent and Orchestrator Integration.

Verifies:
  - Spec 4c: disease_status labels reachability (5 spec labels)
  - Spec 4c-iv & D2/D4/I21: confidence_raw_delta computed correctly, severity_delta_pct strictly None
  - Spec 4c-iii: Plausibility checks (same-image MD5, blurry/low-res, crop mismatch, too soon)
  - Graceful degradation: missing previous case, corrupt JSON, crop mismatch (history=None + flags, never crash)
  - Spec 4d: Triggers T8–T11 end-to-end through run_case
  - Risk formula exact compliance with history score (R = 0.70*T + 0.20*W + 0.10*H)
  - First-visit regression invariant (previous_case_id=None -> monitoring=None, H=0.0)
  - Security & sanitization (no secret leaks or raw tracebacks)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import pytest

from src.agents.monitoring_agent import MonitoringAgent, load_monitoring_config
from src.monitoring.comparator import (
    check_crop_mismatch,
    check_poor_quality,
    check_same_image,
    check_too_soon,
    compute_image_md5,
    compute_monitoring_delta,
)
from src.orchestrator.pipeline import run_case
from src.orchestrator.store import safe_model_dump_json, save_case_record
from src.schemas.schemas import (
    CaseState,
    Citation,
    ClassProb,
    CurrentWeather,
    FarmerRequest,
    FollowUpRecord,
    ImageQualityFlags,
    MonitoringOutput,
    RAGOutput,
    RecommendationOutput,
    RiskOutput,
    VisionOutput,
    WeatherOutput,
)

# 1x1 dummy JPEG bytes for tests
DUMMY_JPEG_1 = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08"
    b"\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13"
    b"\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05"
    b"\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07"
    b"\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
)

# Different dummy image bytes to simulate non-identical image
DUMMY_JPEG_2 = DUMMY_JPEG_1 + b"\x00\x00\x00"


# =============================================================================
# Helper Fixtures & Mock Generators
# =============================================================================

def make_vision_output(
    top_label: str = "Tomato___Early_blight",
    confidence: float = 0.90,
    can_diagnose: bool = True,
    blurry: bool = False,
    low_res: bool = False,
    ood_flag: bool = False,
    escalate: bool = False,
) -> VisionOutput:
    probs = [
        ClassProb(label=top_label, prob=confidence),
        ClassProb(label="Tomato___healthy", prob=round(1.0 - confidence, 4)),
    ]
    flags = ImageQualityFlags(blurry=blurry, low_resolution=low_res)
    return VisionOutput(
        crop=top_label.split("___")[0],
        top_label=top_label,
        confidence=confidence,
        top_k_probs=probs,
        top_3_candidates=probs,
        image_quality_flags=flags,
        ood_flag=ood_flag,
        escalate_to_human=escalate or blurry or low_res or ood_flag,
        can_diagnose=can_diagnose,
        severity_pct=None,
        model_version="mock_v1",
        temperature=1.0,
    )


def make_baseline_case(
    case_id: Optional[uuid.UUID] = None,
    crop: str = "Tomato",
    top_label: str = "Tomato___Early_blight",
    confidence: float = 0.85,
    submitted_at: Optional[datetime] = None,
    image_bytes: bytes = DUMMY_JPEG_1,
    follow_ups: Optional[List[FollowUpRecord]] = None,
) -> CaseState:
    c_uuid = case_id or uuid.uuid4()
    dt = submitted_at or (datetime.now(timezone.utc) - timedelta(days=5))

    req = FarmerRequest(
        farmer_id="farmer_test",
        image_bytes=image_bytes,
        image_mime_type="image/jpeg",
        crop_type=crop,
        submitted_at=dt,
        is_followup=False,
        case_id=c_uuid,
    )
    v_out = make_vision_output(top_label=top_label, confidence=confidence)
    w_out = WeatherOutput(
        weather_available=True,
        weather_score=0.20,
        weather_data_source="mock",
    )
    r_out = RAGOutput(
        found=True,
        disease_name=top_label,
        citations=[Citation(tag="kb_01", title="Test Guide", source_id="kb_01")],
    )
    rk_out = RiskOutput(
        risk_score=0.41,
        risk_level="moderate",
        uncertainty_level="high_confidence",
        factors=[],
        escalate_to_human=False,
        uncertainty_signals_triggered=[],
        weather_unavailable=False,
        rag_unavailable=False,
        progression_note="Baseline progression note",
        risk_computed_at=dt,
    )
    rec_out = RecommendationOutput(
        summary="Apply preventative organic spray",
        immediate_action="Apply preventative organic spray",
        advisory=["Follow labeled dosage instructions."],
        steps=["Follow labeled dosage instructions."],
        actions=["Follow labeled dosage instructions."],
        timing="Morning",
        monitoring_instruction="Check in 3 days",
        urgency="within_48h",
        confidence_justification="High confidence diagnosis",
        evidence_references=["Test Guide"],
        citations=["Test Guide"],
        caveats=[],
        escalate_to_human=False,
        rag_grounded=True,
    )

    return CaseState(
        case_id=c_uuid,
        farmer_id="farmer_test",
        created_at=dt,
        crop_type=crop,
        initial_request=req,
        initial_vision=v_out,
        initial_weather=w_out,
        initial_rag=r_out,
        initial_risk=rk_out,
        initial_recommendation=rec_out,
        follow_ups=follow_ups or [],
        status="active",
    )


class MockVisionAgent:
    def __init__(self, vision_output: VisionOutput):
        self._v_out = vision_output

    def predict(self, image_bytes: bytes, crop: Optional[str] = None) -> VisionOutput:
        return self._v_out


class MockWeatherAgent:
    def __init__(self, weather_score: float = 0.20):
        self.weather_score = weather_score

    def get_weather(self, latitude=None, longitude=None, place_name=None, detected_disease=None, **kwargs) -> WeatherOutput:
        return WeatherOutput(
            weather_available=True,
            weather_score=self.weather_score,
            weather_data_source="mock",
        )


class MockRAGAgent:
    def query_candidate(self, model_class: str, crop: Optional[str] = None, include_doses: bool = False) -> RAGOutput:
        return RAGOutput(
            found=True,
            disease_name=model_class,
            citations=[Citation(tag="mock_1", title="Mock KB Citation", source_id="mock_1")],
            what_to_do_first="Mock action",
        )


class MockRecommendationAgent:
    def recommend(self, vision=None, rag=None, weather=None, risk=None, include_doses=False, **kwargs) -> RecommendationOutput:
        return RecommendationOutput(
            summary="Mock summary",
            immediate_action="Mock immediate advisory action",
            advisory=["Mock step 1"],
            steps=["Mock step 1"],
            actions=["Mock step 1"],
            timing="within_48h",
            monitoring_instruction="Take photo in 3 days",
            urgency="within_48h",
            confidence_justification="Mock justified",
            evidence_references=["Mock KB Citation"],
            citations=["Mock KB Citation"],
            caveats=[],
            escalate_to_human=risk.escalate_to_human if risk else False,
            rag_grounded=True,
        )


# =============================================================================
# Test Suite 1: Reachability of all 5 Spec Disease Status Labels
# =============================================================================

class TestDiseaseStatusReachability:
    """Verify each of the 5 spec disease_status labels is reachable with a synthetic pair."""

    def test_same_disease_still_detected(self):
        baseline = make_baseline_case(top_label="Tomato___Early_blight", confidence=0.85)
        followup_v = make_vision_output(top_label="Tomato___Early_blight", confidence=0.88)
        now = baseline.created_at + timedelta(days=4)

        m_out = compute_monitoring_delta(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Tomato",
            current_submitted_at=now,
        )
        assert m_out.disease_status == "same_disease_still_detected"
        assert m_out.label_same is True
        assert m_out.disease_still_detected is True
        assert m_out.new_label_is_healthy is False

    def test_no_disease_detected_at_followup(self):
        baseline = make_baseline_case(top_label="Tomato___Early_blight", confidence=0.85)
        followup_v = make_vision_output(top_label="Tomato___healthy", confidence=0.92)
        now = baseline.created_at + timedelta(days=4)

        m_out = compute_monitoring_delta(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Tomato",
            current_submitted_at=now,
        )
        assert m_out.disease_status == "no_disease_detected_at_followup"
        assert m_out.label_same is False
        assert m_out.disease_still_detected is False
        assert m_out.new_label_is_healthy is True

    def test_different_disease_detected_at_followup(self):
        baseline = make_baseline_case(top_label="Tomato___Early_blight", confidence=0.85)
        followup_v = make_vision_output(top_label="Tomato___Late_blight", confidence=0.89)
        now = baseline.created_at + timedelta(days=4)

        m_out = compute_monitoring_delta(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Tomato",
            current_submitted_at=now,
        )
        assert m_out.disease_status == "different_disease_detected_at_followup"
        assert m_out.label_same is False
        assert m_out.disease_still_detected is True
        assert m_out.new_label_is_healthy is False

    def test_not_applicable_healthy_baseline(self):
        baseline = make_baseline_case(top_label="Tomato___healthy", confidence=0.95)
        followup_v = make_vision_output(top_label="Tomato___Early_blight", confidence=0.80)
        now = baseline.created_at + timedelta(days=4)

        m_out = compute_monitoring_delta(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Tomato",
            current_submitted_at=now,
        )
        assert m_out.disease_status == "not_applicable_healthy_baseline"

    def test_unknown_status_when_inconclusive(self):
        baseline = make_baseline_case(top_label="Tomato___Early_blight", confidence=0.85)
        followup_v = make_vision_output(top_label="unknown", confidence=0.30, can_diagnose=False)
        now = baseline.created_at + timedelta(days=4)

        m_out = compute_monitoring_delta(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Tomato",
            current_submitted_at=now,
        )
        assert m_out.disease_status == "unknown"


# =============================================================================
# Test Suite 2: Confidence Raw Delta and Severity Null Invariant (D2/D4/I21)
# =============================================================================

class TestConfidenceRawDeltaAndSeverity:
    """Assert confidence_raw_delta is computed correctly and severity stays null."""

    def test_confidence_raw_delta_computation(self):
        baseline = make_baseline_case(top_label="Tomato___Early_blight", confidence=0.8000)
        followup_v = make_vision_output(top_label="Tomato___Early_blight", confidence=0.9250)
        now = baseline.created_at + timedelta(days=3)

        agent = MonitoringAgent()
        m_out = agent.evaluate(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Tomato",
            current_submitted_at=now,
        )

        # 0.9250 - 0.8000 = 0.1250
        assert abs(m_out.confidence_raw_delta - 0.1250) < 1e-4
        assert m_out.baseline_confidence == 0.8000
        assert m_out.followup_confidence == 0.9250

    def test_severity_is_strictly_none_with_mandatory_note(self):
        baseline = make_baseline_case()
        followup_v = make_vision_output()
        now = baseline.created_at + timedelta(days=3)

        agent = MonitoringAgent()
        m_out = agent.evaluate(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Tomato",
            current_submitted_at=now,
        )

        assert m_out.severity_delta_pct is None
        assert "Numeric severity delta is not computable" in m_out.severity_note
        assert "No per-leaf severity labels or masks exist" in m_out.severity_note

    def test_plant_identity_unverified_disclaimer(self):
        baseline = make_baseline_case()
        followup_v = make_vision_output()
        agent = MonitoringAgent()
        m_out = agent.evaluate(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Tomato",
        )
        assert "CANNOT verify that follow-up images show the same physical plant" in m_out.plant_identity_note


# =============================================================================
# Test Suite 3: Plausibility Checks (Spec 4c-iii)
# =============================================================================

class TestPlausibilityChecks:
    """Verify each plausibility check fires on a crafted case."""

    def test_same_image_resubmission_flag(self):
        baseline = make_baseline_case(image_bytes=DUMMY_JPEG_1)
        followup_v = make_vision_output()

        agent = MonitoringAgent()
        m_out = agent.evaluate(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_1,  # Exact same image bytes
            current_crop="Tomato",
        )

        assert m_out.same_image_flag is True
        assert m_out.disease_status == "unknown"
        assert m_out.escalate_to_human is True

    def test_poor_quality_flag_on_blurry_or_low_res(self):
        baseline = make_baseline_case()
        followup_blurry = make_vision_output(blurry=True)

        agent = MonitoringAgent()
        m_out = agent.evaluate(
            current_vision=followup_blurry,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Tomato",
        )

        assert m_out.poor_quality_flag is True
        assert m_out.disease_status == "unknown"
        assert m_out.escalate_to_human is True

    def test_crop_mismatch_flag(self):
        baseline = make_baseline_case(crop="Tomato", top_label="Tomato___Early_blight")
        followup_v = make_vision_output(top_label="Corn___Common_rust")

        agent = MonitoringAgent()
        m_out = agent.evaluate(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Corn",
        )

        assert m_out.crop_mismatch_flag is True
        assert m_out.disease_status == "unknown"
        assert m_out.escalate_to_human is True

    def test_too_soon_flag_same_calendar_day(self):
        same_day = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)
        baseline = make_baseline_case(submitted_at=same_day)
        followup_v = make_vision_output()
        followup_time = datetime(2026, 6, 15, 16, 30, tzinfo=timezone.utc)

        agent = MonitoringAgent()
        m_out = agent.evaluate(
            current_vision=followup_v,
            baseline_case=baseline,
            current_image_bytes=DUMMY_JPEG_2,
            current_crop="Tomato",
            current_submitted_at=followup_time,
        )

        assert m_out.too_soon_flag is True


# =============================================================================
# Test Suite 4: Graceful Degradation
# =============================================================================

class TestGracefulDegradation:
    """Verify graceful degrade on missing/corrupt previous case or crop mismatch."""

    def test_missing_previous_case_id_degrades_gracefully(self, tmp_path):
        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___Early_blight", confidence=0.88))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        missing_id = str(uuid.uuid4())
        resp = run_case(
            image=DUMMY_JPEG_1,
            crop="Tomato",
            previous_case_id=missing_id,
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring is None
        assert "PREVIOUS_CASE_NOT_FOUND" in resp.flags
        assert resp.partial_result is True
        # History score is 0.0 (first_visit / unavailable)
        factors_dict = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors_dict.get("history", 0.0) - 0.0) < 1e-6

    def test_corrupt_previous_case_json_degrades_gracefully(self, tmp_path):
        corrupt_id = uuid.uuid4()
        corrupt_file = tmp_path / f"{corrupt_id}.json"
        corrupt_file.write_text("{corrupt: [invalid json, missing bracket", encoding="utf-8")

        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___Early_blight", confidence=0.88))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_1,
            crop="Tomato",
            previous_case_id=str(corrupt_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring is None
        assert "PREVIOUS_CASE_NOT_FOUND" in resp.flags
        assert resp.partial_result is True
        factors_dict = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors_dict.get("history", 0.0) - 0.0) < 1e-6

    def test_crop_mismatch_degrades_history_score_to_zero(self, tmp_path):
        # Baseline case is Tomato
        baseline = make_baseline_case(crop="Tomato", top_label="Tomato___Early_blight")
        save_case_record(baseline, case_store_dir=tmp_path)

        # Follow-up submitted with Corn
        v_mock = MockVisionAgent(make_vision_output(top_label="Corn___Common_rust", confidence=0.88))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Corn",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring is not None
        assert resp.monitoring.crop_mismatch_flag is True
        assert "CROP_MISMATCH" in resp.flags
        # T11 fired
        assert "T11" in resp.escalation.triggered_triggers
        assert resp.escalation.escalate_to_human is True
        # History score must NOT be credited; remains 0.0
        factors_dict = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors_dict.get("history", 0.0) - 0.0) < 1e-6


# =============================================================================
# Test Suite 5: Triggers T8–T11 and Exact Risk Formula Compliance
# =============================================================================

class TestMonitoringTriggersAndRiskFormula:
    """Verify triggers T8–T11 fire through run_case and R = 0.70*T + 0.20*W + 0.10*H exactly."""

    def test_t8_different_disease_urgent_escalation(self, tmp_path):
        baseline = make_baseline_case(top_label="Tomato___Early_blight", confidence=0.85)
        save_case_record(baseline, case_store_dir=tmp_path)

        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___Late_blight", confidence=0.90))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring.disease_status == "different_disease_detected_at_followup"
        assert "T8" in resp.escalation.triggered_triggers
        assert resp.escalation.priority == "urgent"
        assert resp.escalation.escalate_to_human is True

        # Formula check: H = 0.0 for different_disease_detected_at_followup
        factors_dict = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors_dict.get("history", 0.0) - 0.0) < 1e-6
        expected_r = round(factors_dict.get("threat_table", 0.0) + factors_dict.get("weather", 0.0) + 0.0, 4)
        assert abs(resp.risk.risk_score - expected_r) < 1e-6

    def test_t9_persistent_disease_visit_2_high_and_risk_formula(self, tmp_path):
        baseline = make_baseline_case(top_label="Tomato___Early_blight", confidence=0.85, follow_ups=[])
        save_case_record(baseline, case_store_dir=tmp_path)

        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___Early_blight", confidence=0.90))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring.disease_status == "same_disease_still_detected"
        assert resp.monitoring.visit_number == 2
        assert "T9" in resp.escalation.triggered_triggers
        assert resp.escalation.priority == "high"
        assert resp.escalation.escalate_to_human is True

        # History score is 0.10 for persistent disease
        factors_dict = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors_dict.get("history", 0.0) - round(0.10 * 0.10, 6)) < 1e-6
        expected_r = round(factors_dict.get("threat_table", 0.0) + factors_dict.get("weather", 0.0) + round(0.10 * 0.10, 6), 4)
        assert abs(resp.risk.risk_score - expected_r) < 1e-6

    def test_t9_persistent_disease_visit_3_urgent(self, tmp_path):
        # Baseline with 1 previous follow up -> current run will be visit 3
        fu1 = FollowUpRecord(
            visit_number=2,
            submitted_at=datetime.now(timezone.utc) - timedelta(days=3),
            monitoring_output=MonitoringOutput(
                case_id=uuid.uuid4(),
                visit_number=2,
                baseline_label="Tomato___Early_blight",
                baseline_confidence=0.85,
                followup_label="Tomato___Early_blight",
                followup_confidence=0.88,
                label_same=True,
                disease_still_detected=True,
                new_label_is_healthy=False,
                confidence_raw_delta=0.03,
                disease_status="same_disease_still_detected",
                escalate_to_human=True,
                followup_vision=make_vision_output(),
            ),
            risk_output=RiskOutput(
                risk_score=0.42,
                risk_level="moderate",
                uncertainty_level="high_confidence",
                factors=[],
                escalate_to_human=False,
                uncertainty_signals_triggered=[],
                weather_unavailable=False,
                rag_unavailable=False,
                progression_note="Followup progression note",
                risk_computed_at=datetime.now(timezone.utc),
            ),
            recommendation_output=RecommendationOutput(
                summary="Check again",
                immediate_action="Check again",
                advisory=[],
                steps=[],
                actions=[],
                timing="now",
                monitoring_instruction="Photo",
                urgency="within_48h",
                confidence_justification="Fine",
                evidence_references=[],
                citations=[],
                caveats=[],
                escalate_to_human=True,
                rag_grounded=True,
            ),
        )

        baseline = make_baseline_case(
            top_label="Tomato___Early_blight",
            confidence=0.85,
            follow_ups=[fu1],
        )
        save_case_record(baseline, case_store_dir=tmp_path)

        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___Early_blight", confidence=0.90))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring.visit_number == 3
        assert "T9" in resp.escalation.triggered_triggers
        assert resp.escalation.priority == "urgent"
        assert resp.escalation.escalate_to_human is True

    def test_t10_poor_quality_medium_escalation(self, tmp_path):
        baseline = make_baseline_case(top_label="Tomato___Early_blight", confidence=0.85)
        save_case_record(baseline, case_store_dir=tmp_path)

        # Blurry follow-up triggers poor quality
        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___Early_blight", confidence=0.70, blurry=True))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring.poor_quality_flag is True
        assert "T10" in resp.escalation.triggered_triggers
        assert resp.escalation.escalate_to_human is True

    def test_t11_crop_mismatch_high_escalation(self, tmp_path):
        baseline = make_baseline_case(crop="Tomato", top_label="Tomato___Early_blight")
        save_case_record(baseline, case_store_dir=tmp_path)

        v_mock = MockVisionAgent(make_vision_output(top_label="Corn___Northern_Leaf_Blight", confidence=0.92))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Corn",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring.crop_mismatch_flag is True
        assert resp.monitoring.disease_status == "unknown"
        assert "T8" not in resp.escalation.triggered_triggers
        assert "T11" in resp.escalation.triggered_triggers
        assert resp.escalation.priority == "high"
        assert resp.escalation.escalate_to_human is True

    def test_recovered_case_negative_history_score_and_risk_formula(self, tmp_path):
        baseline = make_baseline_case(top_label="Tomato___Early_blight", confidence=0.85)
        save_case_record(baseline, case_store_dir=tmp_path)

        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___healthy", confidence=0.95))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring.disease_status == "no_disease_detected_at_followup"
        factors_dict = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors_dict.get("history", 0.0) - round(0.10 * (-0.05), 6)) < 1e-6
        raw_r = factors_dict.get("threat_table", 0.0) + factors_dict.get("weather", 0.0) + round(0.10 * (-0.05), 6)
        expected_r = max(0.0, min(1.0, round(raw_r, 4)))
        assert abs(resp.risk.risk_score - expected_r) < 1e-6


# =============================================================================
# Test Suite 6: First-Visit Regression Invariant
# =============================================================================

class TestFirstVisitRegression:
    """Verify first-visit run_case behavior is strictly unchanged vs C7."""

    def test_first_visit_has_no_monitoring_and_zero_history(self):
        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___Early_blight", confidence=0.92))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_1,
            crop="Tomato",
            previous_case_id=None,  # First visit
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            use_memory=True,
        )

        assert resp.monitoring is None
        assert resp.previous_case_id is None
        assert resp.is_followup is False
        factors_dict = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors_dict.get("history", 0.0) - 0.0) < 1e-6
        # R = 0.70*0.50 + 0.20*0.20 + 0.10*0.0 = 0.35 + 0.04 = 0.39
        assert abs(resp.risk.risk_score - 0.39) < 1e-6


# =============================================================================
# Test Suite 7: Security and Sanitization
# =============================================================================

class TestSecurityAndSanitization:
    """Ensure no API keys, secret strings, or tracebacks leak into state or storage."""

    def test_no_keys_or_tracebacks_in_followup(self, tmp_path):
        baseline = make_baseline_case(top_label="Tomato___Early_blight", confidence=0.85)
        save_case_record(baseline, case_store_dir=tmp_path)

        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___Early_blight", confidence=0.90))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        resp_json = safe_model_dump_json(resp)
        assert "AIza" not in resp_json
        assert "GEMINI_API_KEY" not in resp_json
        assert "Traceback (most recent call last)" not in resp_json

        # Check persisted file on disk as well
        saved_file = tmp_path / f"{resp.case_id}_response.json"
        if saved_file.is_file():
            content = saved_file.read_text(encoding="utf-8")
            assert "AIza" not in content
            assert "GEMINI_API_KEY" not in content


# =============================================================================
# Test Suite 8: C8 Fix Pass Verifications
# =============================================================================

class TestC8FixPass:
    """
    Verifies C8 logic fixes:
      - Duplicate images do not count as persistence (T9/T10 suppressed, SAME_IMAGE fires, H=0.0)
      - Resubmission does not increment the visit count used by T9/T10
      - Crop mismatch produces disease_status='unknown', suppresses T8, fires T11, H=0.0
      - Poor quality produces disease_status='unknown', suppresses T8/T9, fires T10, H=0.0
      - Valid disease change on same crop fires T8
      - Exact R = 0.70*T + 0.20*W + 0.10*H identity holds across all cases
    """

    def test_same_image_twice_at_visit_2_and_3_suppresses_t9_t10_and_fires_same_image(self, tmp_path):
        baseline = make_baseline_case(
            top_label="Tomato___Early_blight",
            confidence=0.85,
            image_bytes=DUMMY_JPEG_1,
        )
        save_case_record(baseline, case_store_dir=tmp_path)

        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___Early_blight", confidence=0.85))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        # Visit 2: Same image as baseline
        resp2 = run_case(
            image=DUMMY_JPEG_1,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp2.monitoring.same_image_flag is True
        assert resp2.monitoring.disease_status == "unknown"
        assert "SAME_IMAGE" in resp2.escalation.triggered_triggers
        assert "T9" not in resp2.escalation.triggered_triggers
        assert "T10" not in resp2.escalation.triggered_triggers

        factors2 = {f.source: f.contribution for f in resp2.risk.factors}
        assert abs(factors2.get("history", 0.0) - 0.0) < 1e-6
        expected_r2 = round(factors2.get("threat_table", 0.0) + factors2.get("weather", 0.0) + 0.0, 4)
        assert abs(resp2.risk.risk_score - expected_r2) < 1e-6

        # Visit 3: Same image submitted again
        resp3 = run_case(
            image=DUMMY_JPEG_1,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp3.monitoring.same_image_flag is True
        assert resp3.monitoring.disease_status == "unknown"
        assert "SAME_IMAGE" in resp3.escalation.triggered_triggers
        assert "T9" not in resp3.escalation.triggered_triggers
        assert "T10" not in resp3.escalation.triggered_triggers

        factors3 = {f.source: f.contribution for f in resp3.risk.factors}
        assert abs(factors3.get("history", 0.0) - 0.0) < 1e-6
        expected_r3 = round(factors3.get("threat_table", 0.0) + factors3.get("weather", 0.0) + 0.0, 4)
        assert abs(resp3.risk.risk_score - expected_r3) < 1e-6

    def test_valid_followup_after_duplicate_image_counts_as_visit_2_for_persistence(self, tmp_path):
        baseline = make_baseline_case(
            top_label="Tomato___Early_blight",
            confidence=0.85,
            image_bytes=DUMMY_JPEG_1,
        )
        save_case_record(baseline, case_store_dir=tmp_path)

        v_mock = MockVisionAgent(make_vision_output(top_label="Tomato___Early_blight", confidence=0.85))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        # Visit 2: duplicate image
        run_case(
            image=DUMMY_JPEG_1,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        # Visit 3: a real different image with the same disease
        v_diff = MockVisionAgent(make_vision_output(top_label="Tomato___Early_blight", confidence=0.88))
        resp3 = run_case(
            image=DUMMY_JPEG_2,  # Different image!
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_diff,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        # Duplicate visit did NOT count toward persistence: this is effectively visit 2
        assert resp3.monitoring.visit_number == 2
        assert resp3.monitoring.disease_status == "same_disease_still_detected"
        assert "T9" in resp3.escalation.triggered_triggers
        assert resp3.escalation.priority == "high"  # Visit 2 is high priority, not urgent (visit 3)

        factors = {f.source: f.contribution for f in resp3.risk.factors}
        assert abs(factors.get("history", 0.0) - round(0.10 * 0.10, 6)) < 1e-6
        expected_r = round(factors.get("threat_table", 0.0) + factors.get("weather", 0.0) + round(0.10 * 0.10, 6), 4)
        assert abs(resp3.risk.risk_score - expected_r) < 1e-6

    def test_crop_mismatch_status_unknown_no_t8_t11_fires_and_h_zero(self, tmp_path):
        baseline = make_baseline_case(crop="Tomato", top_label="Tomato___Early_blight")
        save_case_record(baseline, case_store_dir=tmp_path)

        v_mock = MockVisionAgent(make_vision_output(top_label="Corn___Northern_Leaf_Blight", confidence=0.90))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Corn",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_mock,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring.crop_mismatch_flag is True
        assert resp.monitoring.disease_status == "unknown"
        assert "T8" not in resp.escalation.triggered_triggers
        assert "T9" not in resp.escalation.triggered_triggers
        assert "T11" in resp.escalation.triggered_triggers

        factors = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors.get("history", 0.0) - 0.0) < 1e-6
        expected_r = round(factors.get("threat_table", 0.0) + factors.get("weather", 0.0) + 0.0, 4)
        assert abs(resp.risk.risk_score - expected_r) < 1e-6

    def test_poor_quality_followup_status_unknown_no_t8_no_t9_t10_fires_and_h_zero(self, tmp_path):
        baseline = make_baseline_case(crop="Tomato", top_label="Tomato___Early_blight")
        save_case_record(baseline, case_store_dir=tmp_path)

        # Blurry image that would otherwise look like different disease or same disease
        v_blurry = MockVisionAgent(make_vision_output(top_label="Tomato___Late_blight", confidence=0.60, blurry=True))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_blurry,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring.poor_quality_flag is True
        assert resp.monitoring.disease_status == "unknown"
        assert "T10" in resp.escalation.triggered_triggers
        assert "T8" not in resp.escalation.triggered_triggers
        assert "T9" not in resp.escalation.triggered_triggers

        factors = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors.get("history", 0.0) - 0.0) < 1e-6
        expected_r = round(factors.get("threat_table", 0.0) + factors.get("weather", 0.0) + 0.0, 4)
        assert abs(resp.risk.risk_score - expected_r) < 1e-6

    def test_valid_diseased_to_different_disease_on_same_crop_fires_t8(self, tmp_path):
        baseline = make_baseline_case(crop="Tomato", top_label="Tomato___Early_blight", confidence=0.85)
        save_case_record(baseline, case_store_dir=tmp_path)

        v_diff = MockVisionAgent(make_vision_output(top_label="Tomato___Late_blight", confidence=0.90))
        w_mock = MockWeatherAgent(weather_score=0.20)
        r_mock = MockRAGAgent()
        rec_mock = MockRecommendationAgent()

        resp = run_case(
            image=DUMMY_JPEG_2,
            crop="Tomato",
            previous_case_id=str(baseline.case_id),
            vision_agent=v_diff,
            weather_agent=w_mock,
            rag_agent=r_mock,
            recommendation_agent=rec_mock,
            case_store_dir=tmp_path,
            use_memory=True,
        )

        assert resp.monitoring.disease_status == "different_disease_detected_at_followup"
        assert resp.monitoring.crop_mismatch_flag is False
        assert resp.monitoring.poor_quality_flag is False
        assert "T8" in resp.escalation.triggered_triggers
        assert resp.escalation.priority == "urgent"

        factors = {f.source: f.contribution for f in resp.risk.factors}
        assert abs(factors.get("history", 0.0) - 0.0) < 1e-6
        expected_r = round(factors.get("threat_table", 0.0) + factors.get("weather", 0.0) + 0.0, 4)
        assert abs(resp.risk.risk_score - expected_r) < 1e-6

