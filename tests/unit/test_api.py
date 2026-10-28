"""
tests/unit/test_api.py
======================
Unit and integration tests for LeafMind FastAPI backend (Stage C9a).

Tests:
  - GET /health
  - GET /meta (supported_crops match classifier, single-class crops, tips, disclaimers)
  - POST /analyze happy path (top3, risk identity R = 0.70*T + 0.20*W + 0.10*H,
    severity null invariant, no internal source tags in farmer text, citations intact)
  - Validation failures (text renamed .jpg, corrupt bytes, oversize >10MB, tiny <64px) -> 422
  - Blurry image -> 200 with warning
  - Unknown crop -> 422; single-class crop -> 200 with can_diagnose=False explained
  - Follow-up flow with previous_case_id; missing previous ID degrades gracefully
  - GET /case/{case_id} -> 200 and 404
  - Secret & path sanitization: no API key, env var, path, or traceback in any response
  - Timeout -> 504 Gateway Timeout
  - CORS: allowed vs disallowed origin
  - Healthy guard warning (runner-up disease >= threshold) & healthy class R = 0 (Task 3a, 3b)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import uuid

from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter
import pytest

from src.api.app import app, create_app
from src.api.formatting import clean_farmer_text
from src.api.validation import (
    MULTI_CLASS_CROPS,
    SINGLE_CLASS_CROPS,
    SUPPORTED_CROPS,
)
from src.orchestrator.state import (
    EscalationSummary,
    FinalResponse,
    RAGSummary,
    TriggerReason,
    VisionSummary,
    WeatherSummary,
)
from src.schemas.schemas import (
    Citation,
    ClassProb,
    CurrentWeather,
    FarmerRequest,
    FarmerResponse,
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


# =============================================================================
# Helper Utilities & Fixtures
# =============================================================================

def create_test_image_bytes(
    width: int = 128,
    height: int = 128,
    color: str = "green",
    fmt: str = "JPEG",
) -> bytes:
    """Generate in-memory valid image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


def make_sample_final_response(
    crop: str = "Tomato",
    top_label: str = "Tomato___Early_blight",
    top_prob: float = 0.88,
    runner_up_label: str = "Tomato___Target_Spot",
    runner_up_prob: float = 0.08,
    threat_score: float = 0.50,
    weather_score: float = 0.20,
    history_score: float = 0.00,
    escalate: bool = False,
    is_healthy: bool = False,
    blurry: bool = False,
    single_class_crop: bool = False,
    citations: Optional[List[str]] = None,
    what_to_do_first: str = "Prune infected lower foliage immediately (T2).",
    steps: Optional[List[str]] = None,
) -> FinalResponse:
    """Build a deterministic FinalResponse fixture for testing."""
    if is_healthy:
        top_label = f"{crop}___healthy"
        threat_score = 0.0
        weather_score = 0.0
        history_score = 0.0

    if single_class_crop:
        top_label = "Squash___Powdery_mildew"

    c_uuid = uuid.uuid4()
    r_score = round(0.70 * threat_score + 0.20 * weather_score + 0.10 * history_score, 4)

    candidates = [
        ClassProb(label=top_label, prob=top_prob),
        ClassProb(label=runner_up_label, prob=runner_up_prob),
        ClassProb(label=f"{crop}___healthy", prob=max(0.0, 1.0 - top_prob - runner_up_prob)),
    ]

    vis_summary = VisionSummary(
        top_label=top_label,
        confidence=top_prob,
        top_3=candidates,
        can_diagnose=not single_class_crop,
        single_class_crop=single_class_crop,
        ood_flag=False,
        uncertainty_flags=ImageQualityFlags(blurry=blurry),
        severity_pct=None,
        severity_note="Severity percentage is not computed.",
    )

    w_summary = WeatherSummary(
        weather_available=True,
        current_temp_c=25.0,
        current_humidity_pct=80.0,
        triggered_rules=["W01"],
        weather_score=weather_score,
        data_source="mock_weather",
        location_name="Pune, Maharashtra",
    )

    c_list = citations or ["TNAU Agritech Portal: Tomato Diseases"]
    rag_sources = [
        Citation(tag="T2", title=c, url="https://example.com")
        for c in c_list
    ]

    rag_summary = RAGSummary(
        found=True,
        kb_gap=False,
        disease_name=top_label.replace("___", " "),
        sources=rag_sources,
        citations=c_list,
        chemical_options_count=1,
        what_to_do_first=what_to_do_first,
        doses_included=False,
    )

    risk_factors = [
        RiskFactor(source="threat_table", value=f"threat={threat_score:.2f}", contribution=round(0.70 * threat_score, 4), tag="[ENG]"),
        RiskFactor(source="weather", value=f"weather={weather_score:.2f}", contribution=round(0.20 * weather_score, 4), tag="[ENG]"),
        RiskFactor(source="history", value=f"history={history_score:.2f}", contribution=round(0.10 * history_score, 4), tag="[ENG]"),
    ]
    r_label = "severe" if r_score >= 0.60 else ("moderate" if r_score >= 0.25 else "low")
    risk_output = RiskOutput(
        risk_score=r_score,
        risk_level=r_label,
        factors=risk_factors,
        escalate_to_human=escalate,
        uncertainty_level="high_confidence",
        uncertainty_signals_triggered=[],
        progression_note="Standard progression note",
        risk_computed_at=datetime.now(timezone.utc),
    )

    rec_steps = steps or [
        "Prune and discard infected leaves (T2).",
        "Apply protective copper spray (VER).",
    ]
    rec_output = RecommendationOutput(
        summary=f"Confirmed diagnosis: {top_label} (CAR1).",
        what_to_do_first=what_to_do_first,
        steps=rec_steps,
        urgency="within_48h",
        sources=c_list,
    )

    esc_summary = EscalationSummary(
        escalate_to_human=escalate,
        priority="high" if escalate else "low",
        triggered_triggers=["T1"] if escalate else [],
        reasons=[TriggerReason(trigger_id="T1", description="High uncertainty")] if escalate else [],
    )

    return FinalResponse(
        case_id=c_uuid,
        farmer_id="farmer_test",
        submitted_at=datetime.now(timezone.utc),
        crop=crop,
        is_followup=False,
        vision=vis_summary,
        weather=w_summary,
        rag=rag_summary,
        risk=risk_output,
        recommendation=rec_output,
        escalation=esc_summary,
        timing_ms={"node_a": 10.5, "node_b": 2.1},
        severity_pct=None,
        severity_note="Severity percentage is not computed.",
    )


# =============================================================================
# 1. /health and /meta Tests
# =============================================================================

def test_health_endpoint(client: TestClient):
    """GET /health must return status ok and version 0.1.0."""
    resp = client.get("/health")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_meta_endpoint_crop_taxonomy(client: TestClient):
    """
    GET /meta must return supported crops exactly matching the classifier taxonomy,
    single-class crops, farmer capture tips, and standard disclaimers.
    """
    resp = client.get("/meta")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()

    assert set(data["supported_crops"]) == set(SUPPORTED_CROPS)
    assert set(data["single_class_crops"]) == set(SINGLE_CLASS_CROPS)
    assert set(data["multi_class_crops"]) == set(MULTI_CLASS_CROPS)

    # Disclaimers must include advisory, severity, provisional, plant identity
    disclaimers_text = " ".join(data["disclaimers"]).lower()
    assert "advisory only" in disclaimers_text
    assert "severity percentage is not measured" in disclaimers_text
    assert "provisional ratings" in disclaimers_text
    assert "plant identity unverified" in disclaimers_text

    # Capture tips must be provided
    assert len(data["capture_tips"]) >= 3


# =============================================================================
# 2. Image Validation & Error Handling (HTTP 422)
# =============================================================================

def test_invalid_image_text_renamed_jpg(client: TestClient):
    """Uploading plain text renamed as .jpg must fail PIL decode with 422."""
    fake_bytes = b"This is a text file that was renamed to picture.jpg"
    files = {"image": ("fake.jpg", fake_bytes, "image/jpeg")}
    data = {"crop": "Tomato"}

    resp = client.post("/analyze", files=files, data=data)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = resp.json()["detail"]
    assert "could not decode image file" in detail.lower()
    assert "capture tip" in detail.lower()


def test_corrupt_image_bytes(client: TestClient):
    """Corrupt image header/bytes must return 422 with a capture tip."""
    corrupt_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00corrupt_payload_here"
    files = {"image": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    data = {"crop": "Tomato"}

    resp = client.post("/analyze", files=files, data=data)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "corrupt" in resp.json()["detail"].lower() or "not a valid" in resp.json()["detail"].lower()


def test_oversize_image_rejected(client: TestClient):
    """Uploading file exceeding 10 MB must be rejected with 422."""
    # 11 MB dummy bytes
    oversize_bytes = b"0" * (11 * 1024 * 1024)
    files = {"image": ("large.jpg", oversize_bytes, "image/jpeg")}
    data = {"crop": "Tomato"}

    resp = client.post("/analyze", files=files, data=data)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "maximum" in resp.json()["detail"].lower() or "exceeds" in resp.json()["detail"].lower()


def test_tiny_image_dimensions_rejected(client: TestClient):
    """Image below minimum dimension (e.g. 32x32 < 64px) must be rejected with 422."""
    tiny_bytes = create_test_image_bytes(width=32, height=32)
    files = {"image": ("tiny.jpg", tiny_bytes, "image/jpeg")}
    data = {"crop": "Tomato"}

    resp = client.post("/analyze", files=files, data=data)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = resp.json()["detail"]
    assert "too small" in detail.lower()
    assert "64" in detail


def test_blurry_image_not_rejected(client: TestClient):
    """Blurry image must NOT be rejected; it returns 200 with a warning."""
    blurry_response = make_sample_final_response(blurry=True, escalate=True)

    with patch("src.api.routes.run_case", return_value=blurry_response):
        valid_bytes = create_test_image_bytes(width=128, height=128)
        files = {"image": ("leaf.jpg", valid_bytes, "image/jpeg")}
        data = {"crop": "Tomato"}

        resp = client.post("/analyze", files=files, data=data)
        assert resp.status_code == status.HTTP_200_OK
        res_json = resp.json()
        assert any("blurry" in w.lower() for w in res_json["warnings"])
        assert res_json["escalation_banner"] is not None


# =============================================================================
# 3. Crop Validation Tests
# =============================================================================

def test_unknown_crop_returns_422(client: TestClient):
    """Unknown crop must return 422 listing supported crops."""
    valid_bytes = create_test_image_bytes()
    files = {"image": ("leaf.jpg", valid_bytes, "image/jpeg")}
    data = {"crop": "Mango"}

    resp = client.post("/analyze", files=files, data=data)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = resp.json()["detail"]
    assert "unknown crop 'mango'" in detail.lower()
    assert "tomato" in detail.lower()


def test_single_class_crop_can_diagnose_false(client: TestClient):
    """
    Single-class crops (e.g. Squash) return 200 with can_diagnose=False
    and an explanatory farmer-facing note.
    """
    single_crop_response = make_sample_final_response(
        crop="Squash",
        single_class_crop=True,
    )

    with patch("src.api.routes.run_case", return_value=single_crop_response):
        valid_bytes = create_test_image_bytes()
        files = {"image": ("leaf.jpg", valid_bytes, "image/jpeg")}
        data = {"crop": "Squash"}

        resp = client.post("/analyze", files=files, data=data)
        assert resp.status_code == status.HTTP_200_OK
        res_json = resp.json()
        assert res_json["vision"]["can_diagnose"] is False
        assert res_json["vision"]["single_class_crop"] is True
        assert "single registered class" in res_json["farmer_summary"].lower()


# =============================================================================
# 4. POST /analyze Happy Path & Contract Invariants
# =============================================================================

def test_analyze_happy_path_contract(client: TestClient):
    """
    Verify complete AnalyzeResponse contract on happy path:
      - top3 present with name and probability
      - risk identity R = 0.70*T + 0.20*W + 0.10*H holds exactly
      - severity_pct is null (never a number)
      - internal tags like (T2), (VER), (CAR1) stripped from farmer text
      - citations in rag.citations intact
      - escalation banner null when escalate_to_human=False
    """
    mock_final = make_sample_final_response(
        threat_score=0.50,
        weather_score=0.20,
        history_score=0.00,
        escalate=False,
    )

    with patch("src.api.routes.run_case", return_value=mock_final):
        valid_bytes = create_test_image_bytes()
        files = {"image": ("leaf.jpg", valid_bytes, "image/jpeg")}
        data = {"crop": "Tomato", "location": "Nashik"}

        resp = client.post("/analyze", files=files, data=data)
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()

        # Check top3
        assert len(data["top3"]) == 3
        assert data["top3"][0]["name"] == "Tomato___Early_blight"
        assert abs(data["top3"][0]["probability"] - 0.88) < 1e-4

        # Risk Identity R = 0.70*T + 0.20*W + 0.10*H holds on the response factors
        risk = data["risk"]
        factors = risk["factors"]
        t_contrib = next(f["contribution"] for f in factors if f["source"] == "threat_table")
        w_contrib = next(f["contribution"] for f in factors if f["source"] == "weather")
        h_contrib = next(f["contribution"] for f in factors if f["source"] == "history")
        expected_r = round(t_contrib + w_contrib + h_contrib, 4)
        assert abs(risk["risk_score"] - expected_r) < 1e-4

        # Severity invariant (D2/R3): severity_pct MUST be null
        assert data["severity_pct"] is None
        assert data["vision"]["severity_pct"] is None
        assert "not computed" in data["severity_note"].lower()

        # Farmer-facing text formatting: internal tags stripped
        farmer_summary = data["farmer_summary"]
        assert "(CAR1)" not in farmer_summary
        assert "(T2)" not in farmer_summary
        assert "(VER)" not in farmer_summary

        for step in data["steps"]:
            assert "(T2)" not in step
            assert "(VER)" not in step
            assert "(CAR1)" not in step

        # Citations intact
        assert len(data["rag"]["citations"]) >= 1
        assert "TNAU Agritech Portal: Tomato Diseases" in data["rag"]["citations"][0]

        # Escalation banner
        assert data["escalation_banner"] is None


# =============================================================================
# 5. Task 3 Backlog Fixes: Healthy Guard & Zero Weather Risk
# =============================================================================

def test_healthy_guard_runner_up_warning(client: TestClient):
    """
    Task 3a (I40): When top class is healthy AND runner-up disease has probability >= 0.25,
    attach warning 'possible missed disease: <runner-up> (<prob>)'.
    """
    mock_resp = make_sample_final_response(
        crop="Tomato",
        is_healthy=True,
        top_prob=0.65,
        runner_up_label="Tomato___Early_blight",
        runner_up_prob=0.30,  # >= 0.25 threshold
    )

    with patch("src.api.routes.run_case", return_value=mock_resp):
        valid_bytes = create_test_image_bytes()
        files = {"image": ("leaf.jpg", valid_bytes, "image/jpeg")}
        data = {"crop": "Tomato"}

        resp = client.post("/analyze", files=files, data=data)
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()

        warnings_text = " ".join(data["warnings"]).lower()
        assert "possible missed disease: tomato___early_blight" in warnings_text
        assert "0.30" in warnings_text


def test_healthy_class_zero_weather_risk(client: TestClient):
    """
    Task 3b (C5 spec): Healthy classes must receive weather_score = 0.0
    so that composite R = 0.0.
    """
    mock_resp = make_sample_final_response(
        crop="Tomato",
        is_healthy=True,
        threat_score=0.0,
        weather_score=0.0,
        history_score=0.0,
    )

    with patch("src.api.routes.run_case", return_value=mock_resp):
        valid_bytes = create_test_image_bytes()
        files = {"image": ("leaf.jpg", valid_bytes, "image/jpeg")}
        data = {"crop": "Tomato"}

        resp = client.post("/analyze", files=files, data=data)
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()

        assert data["risk"]["risk_score"] == 0.0
        w_contrib = next(f["contribution"] for f in data["risk"]["factors"] if f["source"] == "weather")
        assert w_contrib == 0.0


# =============================================================================
# 6. Follow-Up Flow & Case Retrieval (/case/{id})
# =============================================================================

def test_followup_and_case_retrieval_flow(client: TestClient, tmp_path: Path):
    """
    Verify /case/{case_id} returns 200 for stored case and 404 for missing case.
    Verify follow-up with non-existent previous_case_id degrades gracefully.
    """
    # 1. Non-existent case returns 404
    missing_uuid = str(uuid.uuid4())
    resp = client.get(f"/case/{missing_uuid}")
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in resp.json()["detail"].lower()

    # 2. Existing case retrieval returns 200
    mock_final = make_sample_final_response()
    case_id_str = str(mock_final.case_id)

    with patch("src.api.routes.load_case_record", return_value=None), \
         patch("src.api.routes.load_final_response", return_value=mock_final):
        resp = client.get(f"/case/{case_id_str}")
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json()["case_id"] == case_id_str

    # 3. Follow-up with missing previous_case_id degrades gracefully (200, not 500)
    with patch("src.api.routes.run_case", return_value=mock_final):
        valid_bytes = create_test_image_bytes()
        files = {"image": ("leaf.jpg", valid_bytes, "image/jpeg")}
        data = {"crop": "Tomato", "previous_case_id": missing_uuid}

        resp = client.post("/analyze", files=files, data=data)
        assert resp.status_code == status.HTTP_200_OK


# =============================================================================
# 7. Security, Traceback Redaction & CORS
# =============================================================================

def test_no_secrets_or_tracebacks_in_errors(client: TestClient):
    """Errors must never leak API keys, system paths, or raw tracebacks."""
    fake_key = "AIzaSyFakeKeySecret1234567890abcdef"
    error_msg = f"Database connection error to C:\\Users\\ronit\\AppData\\secret with key={fake_key}"

    with patch("src.api.routes.run_case", side_effect=RuntimeError(error_msg)):
        valid_bytes = create_test_image_bytes()
        files = {"image": ("leaf.jpg", valid_bytes, "image/jpeg")}
        data = {"crop": "Tomato"}

        resp = client.post("/analyze", files=files, data=data)
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        detail = resp.json()["detail"]

        # Ensure fake key is redacted
        assert fake_key not in detail
        assert "[REDACTED" in detail or "secret" not in detail.lower()
        # Ensure raw Windows path is not returned in full
        assert "C:\\Users\\ronit" not in detail
        assert "[REDACTED_PATH]" in detail


def test_cors_allowed_and_disallowed_origins():
    """CORS middleware returns headers for allowed origins and rejects others."""
    test_app = create_app()
    with TestClient(test_app) as cl:
        # Allowed origin
        resp_allowed = cl.get("/health", headers={"Origin": "http://localhost:8501"})
        assert resp_allowed.status_code == status.HTTP_200_OK
        assert resp_allowed.headers.get("access-control-allow-origin") == "http://localhost:8501"

        # Disallowed origin
        resp_disallowed = cl.get("/health", headers={"Origin": "http://malicious-attacker.com"})
        assert resp_disallowed.status_code == status.HTTP_200_OK
        assert "access-control-allow-origin" not in resp_disallowed.headers


# =============================================================================
# 8. Timeout 504 Handling
# =============================================================================

def test_analyze_timeout_returns_504(client: TestClient):
    """When pipeline execution exceeds server timeout, HTTP 504 is returned."""
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
        valid_bytes = create_test_image_bytes()
        files = {"image": ("leaf.jpg", valid_bytes, "image/jpeg")}
        data = {"crop": "Tomato"}

        resp = client.post("/analyze", files=files, data=data)
        assert resp.status_code == status.HTTP_504_GATEWAY_TIMEOUT
        assert "timed out" in resp.json()["detail"].lower()


# =============================================================================
# 9. Real Image JSON Serialization & Byte Leak Regression Tests (DO Step 4)
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RE_BASE64_HEURISTIC = re.compile(r"^[A-Za-z0-9+/=\r\n]+$")


def assert_no_bytes_and_no_large_base64(data: Any, max_b64_len: int = 1000, path: str = "root") -> None:
    """
    Recursively verify that:
    1. No field in the response is raw bytes.
    2. No response string is longer than max_b64_len chars of base64-looking data.
    3. json.dumps succeeds strictly with no default= serializer.
    """
    assert not isinstance(data, bytes), f"Found raw bytes at {path}!"
    if isinstance(data, str):
        if len(data) >= max_b64_len:
            sample = data[:300]
            assert not RE_BASE64_HEURISTIC.match(sample), (
                f"Field at {path} contains a large base64-like blob (len={len(data)})!"
            )
    elif isinstance(data, dict):
        for k, v in data.items():
            assert_no_bytes_and_no_large_base64(v, max_b64_len=max_b64_len, path=f"{path}.{k}")
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            assert_no_bytes_and_no_large_base64(item, max_b64_len=max_b64_len, path=f"{path}[{i}]")

    # Strictly JSON serializable with standard library json and no default handler
    if path == "root":
        try:
            dumped = json.dumps(data)
            assert isinstance(dumped, str)
        except Exception as exc:
            pytest.fail(f"json.dumps(data) failed with no default=: {exc}")


def test_real_jpeg_escalated_single_class_crop(client: TestClient):
    """
    Regression test with REAL JPEG bytes:
    Squash single-class crop triggers escalation (T6) and generates a HumanEscalationPayload.
    Response must return HTTP 200 with valid JSON, and no raw image bytes or base64 blobs.
    """
    squash_path = REPO_ROOT / "Dataset_2" / "Train" / "Squash___Powdery_mildew" / "PV__0056878d-a244-4653-ad56-7e97080c7fd2___UMD_Powd.M 9962.JPG"
    assert squash_path.is_file(), f"Test image not found at {squash_path}"
    with open(squash_path, "rb") as f:
        real_bytes = f.read()

    # Verify real JPEG starts with 0xFF 0xD8 (non-UTF8 sequence)
    assert real_bytes.startswith(b"\xff\xd8")

    files = {"image": ("squash.jpg", real_bytes, "image/jpeg")}
    data = {"crop": "Squash"}

    resp = client.post("/analyze", files=files, data=data)
    assert resp.status_code == status.HTTP_200_OK

    resp_json = resp.json()
    assert_no_bytes_and_no_large_base64(resp_json)

    # Check escalation payload has safe stub instead of raw bytes
    assert resp_json["escalation"]["escalate_to_human"] is True
    payload = resp_json["escalation"]["payload"]
    assert payload is not None
    assert isinstance(payload["image_bytes"], dict)
    assert payload["image_bytes"]["image_included"] is False
    assert payload["image_bytes"]["size_bytes"] > 0
    assert payload["image_bytes"]["image_sha256"] is not None


def test_real_jpeg_healthy_guard_field_ood(client: TestClient):
    """
    Regression test with REAL JPEG bytes:
    Apple scab field OOD photo with crop Apple triggers the healthy-guard and escalation.
    Response must return HTTP 200 with valid JSON and safe image metadata.
    """
    apple_path = REPO_ROOT / "Dataset_2" / "Test_OOD" / "Apple___Apple_scab" / "PD__Apple Scab Leaf (1).jpg"
    assert apple_path.is_file(), f"Test image not found at {apple_path}"
    with open(apple_path, "rb") as f:
        real_bytes = f.read()

    assert real_bytes.startswith(b"\xff\xd8")

    files = {"image": ("apple.jpg", real_bytes, "image/jpeg")}
    data = {"crop": "Apple"}

    resp = client.post("/analyze", files=files, data=data)
    assert resp.status_code == status.HTTP_200_OK

    resp_json = resp.json()
    assert_no_bytes_and_no_large_base64(resp_json)
    assert resp_json["crop"] == "Apple"


def test_real_jpeg_followup_case(client: TestClient):
    """
    Regression test with REAL JPEG bytes:
    Follow-up case using previous_case_id comparing against persisted baseline.
    Response must return HTTP 200 with valid JSON and no raw bytes.
    """
    tomato_path = REPO_ROOT / "Dataset_2" / "Test_ID" / "Tomato___Early_blight" / "PV__0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG"
    assert tomato_path.is_file(), f"Test image not found at {tomato_path}"
    with open(tomato_path, "rb") as f:
        real_bytes = f.read()

    # Step 1: Initial visit
    resp_base = client.post(
        "/analyze",
        files={"image": ("baseline.jpg", real_bytes, "image/jpeg")},
        data={"crop": "Tomato"},
    )
    assert resp_base.status_code == status.HTTP_200_OK
    base_case_id = resp_base.json()["case_id"]

    # Step 2: Follow-up visit with previous_case_id
    resp_fu = client.post(
        "/analyze",
        files={"image": ("followup.jpg", real_bytes, "image/jpeg")},
        data={"crop": "Tomato", "previous_case_id": base_case_id},
    )
    assert resp_fu.status_code == status.HTTP_200_OK

    fu_json = resp_fu.json()
    assert_no_bytes_and_no_large_base64(fu_json)
    assert fu_json["is_followup"] is True
    assert fu_json["previous_case_id"] == base_case_id
    assert fu_json["monitoring"] is not None


def test_real_jpeg_blurry_image(client: TestClient):
    """
    Regression test with REAL JPEG bytes:
    Blurry image triggering quality flag and escalation T7.
    Response must return HTTP 200 with valid JSON and warning notices.
    """
    tomato_path = REPO_ROOT / "Dataset_2" / "Test_ID" / "Tomato___Early_blight" / "PV__0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG"
    with open(tomato_path, "rb") as f:
        real_bytes = f.read()

    pil_img = Image.open(io.BytesIO(real_bytes))
    blurry_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=15))
    buf = io.BytesIO()
    blurry_pil.save(buf, format="JPEG")
    blurry_bytes = buf.getvalue()

    resp = client.post(
        "/analyze",
        files={"image": ("blurry.jpg", blurry_bytes, "image/jpeg")},
        data={"crop": "Tomato"},
    )
    assert resp.status_code == status.HTTP_200_OK

    resp_json = resp.json()
    assert_no_bytes_and_no_large_base64(resp_json)
    warnings = resp_json.get("warnings", [])
    assert any("blurry" in w.lower() for w in warnings)


def test_real_png_image(client: TestClient):
    """
    Regression test with REAL PNG bytes:
    Upload a real PNG formatted image (starts with PNG signature bytes).
    Response must return HTTP 200 with valid JSON and no raw bytes.
    """
    tomato_path = REPO_ROOT / "Dataset_2" / "Test_ID" / "Tomato___Early_blight" / "PV__0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG"
    pil_img = Image.open(tomato_path)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    resp = client.post(
        "/analyze",
        files={"image": ("leaf.png", png_bytes, "image/png")},
        data={"crop": "Tomato"},
    )
    assert resp.status_code == status.HTTP_200_OK

    resp_json = resp.json()
    assert_no_bytes_and_no_large_base64(resp_json)


def test_get_case_safe_from_raw_bytes(client: TestClient):
    """
    Regression test for GET /case/{case_id}:
    Loads persisted case record for an escalated request.
    Must return HTTP 200 with valid JSON, and no raw image bytes or large base64 strings.
    """
    squash_path = REPO_ROOT / "Dataset_2" / "Train" / "Squash___Powdery_mildew" / "PV__0056878d-a244-4653-ad56-7e97080c7fd2___UMD_Powd.M 9962.JPG"
    with open(squash_path, "rb") as f:
        real_bytes = f.read()

    resp_post = client.post(
        "/analyze",
        files={"image": ("squash.jpg", real_bytes, "image/jpeg")},
        data={"crop": "Squash"},
    )
    assert resp_post.status_code == status.HTTP_200_OK
    case_id = resp_post.json()["case_id"]

    resp_get = client.get(f"/case/{case_id}")
    assert resp_get.status_code == status.HTTP_200_OK

    case_json = resp_get.json()
    assert_no_bytes_and_no_large_base64(case_json)
    assert case_json["case_id"] == case_id
    if "initial_request" in case_json and "image_bytes" in case_json["initial_request"]:
        img_stub = case_json["initial_request"]["image_bytes"]
        assert isinstance(img_stub, dict)
        assert img_stub.get("image_included") is False


def test_serialization_failure_guard_returns_clean_500(client: TestClient):
    """
    Guard test:
    If response serialization fails inside route, return clean HTTP 500
    with a generic message and no internals/tracebacks.
    """
    from pydantic_core import PydanticSerializationError

    with patch(
        "src.api.routes.AnalyzeResponse.model_dump",
        side_effect=PydanticSerializationError("Error serializing to JSON: invalid utf-8 sequence"),
    ):
        valid_bytes = create_test_image_bytes()
        files = {"image": ("leaf.jpg", valid_bytes, "image/jpeg")}
        data = {"crop": "Tomato"}

        resp = client.post("/analyze", files=files, data=data)
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        body = resp.json()
        assert "detail" in body
        # Must be clean and generic, without leaking invalid utf-8 or internals
        assert "invalid utf-8" not in body["detail"]
        assert "pydantic" not in body["detail"].lower()
        assert "traceback" not in body["detail"].lower()

