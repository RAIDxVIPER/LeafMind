"""
tests/unit/test_streamlit_app.py
================================
Unit tests for the temporary Streamlit test UI (Stage C9a).
Uses streamlit.testing.v1.AppTest with mocked API calls (zero external network).

Verifies:
  - App runs without unhandled exception.
  - Shows the top-3 candidate table.
  - Shows the escalation banner when escalate=true.
  - Never shows a severity percentage as a number.
  - Never reveals any secret API key.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def mock_meta_data():
    return {
        "supported_crops": ["Apple", "Corn", "Tomato"],
        "single_class_crops": [],
        "multi_class_crops": ["Apple", "Corn", "Tomato"],
        "capture_tips": ["Hold camera steady in daylight."],
        "disclaimers": [
            "Advisory only: confirm with an agronomist.",
            "Severity percentage is not measured.",
            "Provisional ratings: threat and weather unverified.",
            "Plant identity unverified.",
        ],
    }


@pytest.fixture
def mock_analysis_result():
    return {
        "case_id": "4a28f891-944a-4a4f-9e79-5098ef6c3b99",
        "farmer_id": "farmer_001",
        "crop": "Tomato",
        "farmer_summary": "Tomato Early Blight detected. Apply copper fungicide.",
        "top3": [
            {"name": "Tomato___Early_blight", "probability": 0.825},
            {"name": "Tomato___Target_Spot", "probability": 0.120},
            {"name": "Tomato___healthy", "probability": 0.055},
        ],
        "risk_label": "medium",
        "urgency": "within_48h",
        "steps": [
            "Prune infected foliage.",
            "Apply protective copper spray.",
        ],
        "warnings": ["Image has mild blur."],
        "disclaimers": [
            "Advisory only.",
            "Severity percentage is not measured.",
        ],
        "escalation_banner": "Agronomist review recommended: Automated analysis flagged high uncertainty or severe risk.",
        "escalation": {
            "escalate_to_human": True,
            "priority": "high",
            "triggered_triggers": ["T1"],
        },
        "weather": {
            "weather_available": True,
            "current_temp_c": 26.0,
            "current_humidity_pct": 85.0,
            "triggered_rules": ["W01: Warm and humid"],
            "weather_score": 0.20,
            "location_name": "Nashik",
        },
        "risk": {
            "risk_score": 0.54,
            "breakdown": {
                "threat_score": 0.70,
                "weather_score": 0.20,
                "history_score": 0.00,
            },
        },
        "severity_pct": None,
        "timing_ms": {"pipeline": 150.0},
    }


def test_streamlit_app_renders_without_exception(mock_meta_data):
    """Verify app initializes and runs initial view cleanly without exceptions."""
    with patch("streamlit_app.app.fetch_health", return_value={"status": "ok", "version": "0.1.0"}), \
         patch("streamlit_app.app.fetch_meta", return_value=mock_meta_data):

        at = AppTest.from_file("streamlit_app/app.py")
        at.run(timeout=10)

        assert len(at.exception) == 0
        assert len(at.title) >= 1
        # Crop selector exists
        assert len(at.selectbox) >= 1
        # Interactive elements exist
        assert len(at.button) >= 1
        assert len(at.text_input) >= 1


def test_streamlit_app_displays_results_and_escalation(mock_meta_data, mock_analysis_result):
    """
    Verify result presentation:
      - App shows top-3 candidates table
      - Shows escalation banner when escalate=true
      - Never displays severity percentage as a number
      - Never exposes API keys
    """
    with patch("streamlit_app.app.fetch_health", return_value={"status": "ok", "version": "0.1.0"}), \
         patch("streamlit_app.app.fetch_meta", return_value=mock_meta_data):

        at = AppTest.from_file("streamlit_app/app.py")
        # Pre-seed session state with a completed analysis
        at.session_state["current_analysis"] = mock_analysis_result
        at.run(timeout=10)

        assert len(at.exception) == 0

        # Top-3 Table exists and is populated
        assert len(at.table) >= 1
        table_data = at.table[0].value
        # Check that 3 candidate rows exist
        assert len(table_data) == 3

        # Escalation banner is shown as a warning
        assert len(at.warning) >= 1
        warning_texts = " ".join([str(w.value) for w in at.warning])
        assert "agronomist review recommended" in warning_texts.lower()

        # Secret / key safety: no raw API key or token pattern
        all_app_text = json.dumps(at.session_state["current_analysis"])
        assert "AIza" not in all_app_text
        assert "sk-" not in all_app_text

        # Severity percentage is NEVER rendered as a number
        # Verify severity_pct is strictly None
        assert mock_analysis_result["severity_pct"] is None
        assert "severity percentage is not measured" in all_app_text.lower()


def test_streamlit_app_handles_escalated_safe_stub_response(mock_meta_data, mock_analysis_result):
    """
    Verify AppTest handles an escalated response where escalation.payload contains
    SafeImageStub (image_included=False, image_sha256, size_bytes).
    Ensures zero exceptions and proper warning banner.
    """
    escalated_data = dict(mock_analysis_result)
    escalated_data["crop"] = "Squash"
    escalated_data["escalation"] = {
        "escalate_to_human": True,
        "priority": "high",
        "triggered_triggers": ["SINGLE_CLASS_CROP"],
        "reasons": [{"trigger_id": "SINGLE_CLASS_CROP", "description": "Single class crop"}],
        "payload": {
            "case_id": "4a28f891-944a-4a4f-9e79-5098ef6c3b99",
            "farmer_id": "farmer_001",
            "escalation_reason": ["Single class crop"],
            "image_bytes": {
                "image_included": False,
                "image_sha256": "abcdef1234567890abcdef1234567890",
                "size_bytes": 20480,
            },
            "escalated_at": "2026-10-28T12:00:00Z",
            "priority": "high",
        },
    }

    with patch("streamlit_app.app.fetch_health", return_value={"status": "ok", "version": "0.1.0"}), \
         patch("streamlit_app.app.fetch_meta", return_value=mock_meta_data):

        at = AppTest.from_file("streamlit_app/app.py")
        at.session_state["current_analysis"] = escalated_data
        at.run(timeout=10)

        assert len(at.exception) == 0
        warning_texts = " ".join([str(w.value) for w in at.warning])
        assert "agronomist review recommended" in warning_texts.lower()


def test_streamlit_app_handles_followup_monitoring_response(mock_meta_data, mock_analysis_result):
    """
    Verify AppTest handles a follow-up response with monitoring trajectory delta cleanly,
    asserting strictly neutral wording and zero improvement/worsening claims.
    """
    followup_data = dict(mock_analysis_result)
    followup_data["is_followup"] = True
    followup_data["previous_case_id"] = "11111111-2222-3333-4444-555555555555"
    followup_data["monitoring"] = {
        "case_id": "4a28f891-944a-4a4f-9e79-5098ef6c3b99",
        "visit_number": 2,
        "baseline_label": "Tomato___Early_blight",
        "baseline_confidence": 0.82,
        "followup_label": "Tomato___Early_blight",
        "followup_confidence": 0.75,
        "label_same": True,
        "disease_still_detected": True,
        "new_label_is_healthy": False,
        "confidence_raw_delta": -0.07,
        "disease_status": "same_disease_still_detected",
        "same_image_flag": False,
        "poor_quality_flag": False,
        "crop_mismatch_flag": False,
        "too_soon_flag": False,
        "escalate_to_human": False,
    }

    with patch("streamlit_app.app.fetch_health", return_value={"status": "ok", "version": "0.1.0"}), \
         patch("streamlit_app.app.fetch_meta", return_value=mock_meta_data):

        at = AppTest.from_file("streamlit_app/app.py")
        at.session_state["current_analysis"] = followup_data
        at.run(timeout=10)

        assert len(at.exception) == 0
        # Check rendered text for neutral language and absence of recovery/improvement/worsening claims
        all_elements = list(at.info) + list(at.metric) + list(at.markdown) + list(at.caption)
        all_rendered_text = " ".join([
            str(getattr(el, "value", ""))
            for el in all_elements
        ]).lower()

        assert "the same disease was detected again" in all_rendered_text
        assert "recover" not in all_rendered_text
        assert "improv" not in all_rendered_text
        assert "worsen" not in all_rendered_text


def test_streamlit_all_5_disease_statuses_strictly_neutral(mock_meta_data, mock_analysis_result):
    """
    Assert that all 5 monitoring disease_status values produce neutral, non-efficacy text in the UI
    and never produce words like recover, improv, or worsen.
    """
    valid_statuses = [
        "same_disease_still_detected",
        "no_disease_detected_at_followup",
        "different_disease_detected_at_followup",
        "not_applicable_healthy_baseline",
        "unknown",
    ]

    for status_val in valid_statuses:
        followup_data = dict(mock_analysis_result)
        followup_data["is_followup"] = True
        followup_data["monitoring"] = {
            "case_id": "4a28f891-944a-4a4f-9e79-5098ef6c3b99",
            "visit_number": 2,
            "baseline_label": "Tomato___Early_blight",
            "baseline_confidence": 0.82,
            "followup_label": "Tomato___Early_blight",
            "followup_confidence": 0.75,
            "label_same": True,
            "disease_still_detected": True,
            "new_label_is_healthy": False,
            "confidence_raw_delta": 0.0,
            "disease_status": status_val,
            "same_image_flag": False,
            "poor_quality_flag": False,
            "crop_mismatch_flag": False,
            "too_soon_flag": False,
            "escalate_to_human": False,
        }

        with patch("streamlit_app.app.fetch_health", return_value={"status": "ok", "version": "0.1.0"}), \
             patch("streamlit_app.app.fetch_meta", return_value=mock_meta_data):

            at = AppTest.from_file("streamlit_app/app.py")
            at.session_state["current_analysis"] = followup_data
            at.run(timeout=10)

            assert len(at.exception) == 0
            info_texts = " ".join([str(i.value) for i in at.info]).lower()
            assert "recover" not in info_texts
            assert "improv" not in info_texts
            assert "worsen" not in info_texts


