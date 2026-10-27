"""
tests/unit/test_recommendation_agent.py
=======================================
Unit tests for Recommendation Agent (Stage C6).

Requirements:
  - No network calls; no real API key; mocked LLM provider.
  - Grounding: validation rejects ungrounded doses, products, and numbers -> fallback.
  - Fallback path on: missing key, timeout, provider error, empty output.
  - Key never appears in logs or exceptions (captured and asserted).
  - Uncertainty / escalation wording present; healthy-leaf note present.
  - RAG found=False and Squash___Powdery_mildew handled without invention.
  - GAP-risk rows show provisional note (I19); weather note marked unverified (I14).
  - Provider switch via LLM_PROVIDER (gemini, ollama, unknown).
  - include_doses=False: no doses in LLM prompt and none in output.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import List
from unittest.mock import patch

import pytest
import requests

from src.agents.recommendation_agent import RecommendationAgent
from src.recommendation.fallback import build_fallback_recommendation
from src.recommendation.prompt import build_recommendation_prompt
from src.recommendation.provider import (
    GeminiProvider,
    MockLLMProvider,
    OllamaProvider,
    get_provider,
    sanitize_key,
)
from src.schemas.schemas import (
    ChemicalOption,
    Citation,
    ClassProb,
    ImageQualityFlags,
    RAGOutput,
    RecommendationInput,
    RecommendationOutput,
    RiskFactor,
    RiskOutput,
    RuleTriggerResult,
    VisionOutput,
    WeatherOutput,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_vision(
    top_label: str = "Tomato___Late_blight",
    confidence: float = 0.88,
    top_k: List[tuple] = None,
    blurry: bool = False,
    can_diagnose: bool = True,
    escalate_to_human: bool = False,
) -> VisionOutput:
    if top_k is None:
        top_k = [
            (top_label, confidence),
            ("Tomato___Early_blight", 0.08),
            ("Tomato___Leaf_Mold", 0.04),
        ]
    top_k_probs = [ClassProb(label=label, prob=prob) for label, prob in top_k]
    return VisionOutput(
        top_label=top_label,
        confidence=confidence,
        top_k_probs=top_k_probs,
        top_3_candidates=top_k_probs[:3],
        image_quality_flags=ImageQualityFlags(blurry=blurry),
        can_diagnose=can_diagnose,
        escalate_to_human=escalate_to_human,
        model_version="efficientnet_b3_v1",
        temperature=1.0,
    )


def make_rag(
    model_class: str = "Tomato___Late_blight",
    found: bool = True,
    disease_name: str = "Late Blight",
    what_to_do_first: str = "Remove and destroy all infected foliage immediately.",
    include_chemical_options: bool = False,
    kb_gap: bool = False,
) -> RAGOutput:
    chemical_options = []
    if include_chemical_options:
        chemical_options = [
            ChemicalOption(
                product="copper oxychloride 50% WP",
                dose="2.5 g/L",
                verification_tag="V1",
            )
        ]

    return RAGOutput(
        model_class=model_class,
        disease_name=disease_name,
        found=found,
        kb_gap=kb_gap,
        knowledge_gap_flag=kb_gap,
        reason="no_kb_chapter" if not found else None,
        what_it_is="Late blight is a destructive fungal-like oomycete infection.",
        why_it_happens="Caused by Phytophthora infestans favoured by cool, wet weather.",
        what_to_do_first=what_to_do_first,
        prevention="Use certified seed, ensure adequate spacing, and avoid overhead watering.",
        treatment_options=["Apply protective copper fungicide sprays before disease spreads."],
        preventive_measures=["Prune lower leaves to improve airflow."],
        chemical_options=chemical_options,
        citations=[
            Citation(tag="T1", title="Tomato Production Manual", url="https://example.org/tomato")
        ],
    )


def make_weather(
    available: bool = True,
    score: float = 0.5,
    triggered_rules: List[str] = None,
) -> WeatherOutput:
    rules = []
    if triggered_rules:
        for r_name in triggered_rules:
            rules.append(
                RuleTriggerResult(
                    rule_id="W01",
                    condition_name=r_name,
                    triggered=True,
                    measured_values="RH 92%",
                    threshold="RH >= 90%",
                    time_window="past_24h",
                    threshold_source="Table 4b",
                )
            )

    return WeatherOutput(
        weather_available=available,
        weather_score=score,
        rules_triggered=rules,
        unavailable_reason=None if available else "Weather API unavailable",
    )


def make_risk(
    risk_level: str = "severe",
    risk_score: float = 0.80,
    is_gap_row: bool = False,
    escalate_to_human: bool = False,
) -> RiskOutput:
    tag = "[ENG assumption][GAP — threat table empty]" if is_gap_row else "[KB-derived draft]"
    val = "GAP fallback" if is_gap_row else "HIGH threat"
    return RiskOutput(
        risk_level=risk_level,
        risk_score=risk_score,
        escalate_to_human=escalate_to_human,
        uncertainty_level="low_confidence" if escalate_to_human else "high_confidence",
        uncertainty_signals_triggered=["U1_low_confidence"] if escalate_to_human else [],
        factors=[
            RiskFactor(
                source="threat_table",
                value=val,
                contribution=0.56,
                tag=tag,
            )
        ],
        progression_note="Prompt action is advisable. [ENG heuristic]",
        risk_computed_at=datetime.now(timezone.utc),
    )


# ── Grounding Tests ────────────────────────────────────────────────────────────

class TestGroundingValidation:
    def test_mock_llm_adding_dose_fails_validation_and_triggers_fallback(self):
        """Mock LLM tries to add a chemical dose -> validation rejects -> fallback used."""
        hallucinated_response = json.dumps({
            "summary": "Apply 2.5 ml/L of copper oxychloride immediately.",
            "what_to_do_first": "Spray at 2.5 ml/L per acre.",
            "steps": ["Apply 2.5 ml/L copper fungicide"],
            "weather_note": None,
            "uncertainty_note": None,
        })
        provider = MockLLMProvider(handler=hallucinated_response)
        agent = RecommendationAgent(provider=provider)

        vision = make_vision()
        rag = make_rag()
        weather = make_weather()
        risk = make_risk()

        rec = agent.recommend(vision=vision, rag=rag, weather=weather, risk=risk, include_doses=False)

        assert rec.llm_used is False
        assert rec.fallback_reason is not None
        assert "validation_failed" in rec.fallback_reason
        assert "dose_pattern_detected" in rec.fallback_reason
        # Fallback output does not contain the hallucinated dose
        assert "2.5 ml/L" not in rec.summary
        assert len(rec.steps) > 0

    def test_mock_llm_adding_ungrounded_product_fails_validation(self):
        """Mock LLM introduces a chemical product not present in input -> rejected."""
        hallucinated_response = json.dumps({
            "summary": "Late blight detected. Spray chlorothalonil to manage infection.",
            "what_to_do_first": "Apply chlorothalonil now.",
            "steps": ["Apply chlorothalonil spray."],
            "weather_note": None,
            "uncertainty_note": None,
        })
        provider = MockLLMProvider(handler=hallucinated_response)
        agent = RecommendationAgent(provider=provider)

        vision = make_vision()
        rag = make_rag()  # Does NOT contain chlorothalonil
        weather = make_weather()
        risk = make_risk()

        rec = agent.recommend(vision=vision, rag=rag, weather=weather, risk=risk)

        assert rec.llm_used is False
        assert rec.fallback_reason is not None
        assert "validation_failed" in rec.fallback_reason
        assert "ungrounded_product" in rec.fallback_reason

    def test_mock_llm_adding_ungrounded_number_fails_validation(self):
        """Mock LLM introduces an ungrounded numeric claim -> rejected."""
        hallucinated_response = json.dumps({
            "summary": "Late blight detected. Expect 99 days of quarantine.",
            "what_to_do_first": "Remove leaves within 99 days.",
            "steps": ["Wait 99 days"],
            "weather_note": None,
            "uncertainty_note": None,
        })
        provider = MockLLMProvider(handler=hallucinated_response)
        agent = RecommendationAgent(provider=provider)

        rec = agent.recommend(
            vision=make_vision(),
            rag=make_rag(),
            weather=make_weather(),
            risk=make_risk(),
        )

        assert rec.llm_used is False
        assert rec.fallback_reason is not None
        assert "validation_failed" in rec.fallback_reason
        assert "ungrounded_number" in rec.fallback_reason

    def test_valid_grounded_output_is_accepted(self):
        """Mock LLM provides reworded advice using input concepts -> accepted."""
        valid_response = json.dumps({
            "summary": "Identified Late blight with severe risk. Immediate foliage sanitation is recommended.",
            "what_to_do_first": "Remove and destroy all infected foliage immediately.",
            "steps": [
                "Remove and destroy all infected foliage immediately.",
                "Ensure adequate plant spacing to improve airflow.",
                "Avoid overhead irrigation to minimize leaf wetness."
            ],
            "weather_note": "[UNVERIFIED — I14] Weather conditions may favour disease spread.",
            "uncertainty_note": None,
        })
        provider = MockLLMProvider(handler=valid_response)
        agent = RecommendationAgent(provider=provider)

        rec = agent.recommend(
            vision=make_vision(),
            rag=make_rag(),
            weather=make_weather(available=True, triggered_rules=["High humidity"]),
            risk=make_risk(risk_level="severe", risk_score=0.80),
        )

        assert rec.llm_used is True
        assert rec.fallback_reason is None
        assert "Identified Late blight" in rec.summary
        assert rec.urgency == "within_24h"
        assert len(rec.steps) == 3
        assert rec.rag_grounded is True
        assert len(rec.citations) == 1
        assert "T1: Tomato Production Manual" in rec.citations[0]


# ── Fallback Path Tests ────────────────────────────────────────────────────────

class TestFallbackPaths:
    def test_fallback_on_missing_api_key(self, monkeypatch):
        """When GEMINI_API_KEY is not set, agent cleanly falls back without crashing."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setattr("src.recommendation.provider.load_local_env_safely", lambda *a, **kw: None)

        # Create agent with no mock provider; initialization fails due to missing key
        agent = RecommendationAgent()

        rec = agent.recommend(
            vision=make_vision(),
            rag=make_rag(),
            weather=make_weather(),
            risk=make_risk(),
        )

        assert rec.llm_used is False
        assert rec.fallback_reason == "missing_api_key"
        assert rec.summary is not None
        assert rec.urgency == "within_24h"
        assert len(rec.steps) > 0

    def test_fallback_on_timeout(self):
        """When provider raises TimeoutError, agent catches and returns fallback."""
        provider = MockLLMProvider(handler=TimeoutError("Request timed out after 30s"))
        agent = RecommendationAgent(provider=provider)

        rec = agent.recommend(
            vision=make_vision(),
            rag=make_rag(),
            weather=make_weather(),
            risk=make_risk(),
        )

        assert rec.llm_used is False
        assert rec.fallback_reason == "timeout"
        assert rec.summary is not None

    def test_fallback_on_provider_error(self):
        """When provider raises HTTP or network error, agent falls back."""
        provider = MockLLMProvider(handler=RuntimeError("HTTP 503: Service Unavailable"))
        agent = RecommendationAgent(provider=provider)

        rec = agent.recommend(
            vision=make_vision(),
            rag=make_rag(),
            weather=make_weather(),
            risk=make_risk(),
        )

        assert rec.llm_used is False
        assert rec.fallback_reason is not None
        assert "provider_error" in rec.fallback_reason
        assert "503" in rec.fallback_reason

    def test_fallback_on_empty_llm_output(self):
        """When provider returns empty or whitespace-only response, agent falls back."""
        provider = MockLLMProvider(handler="   ")
        agent = RecommendationAgent(provider=provider)

        rec = agent.recommend(
            vision=make_vision(),
            rag=make_rag(),
            weather=make_weather(),
            risk=make_risk(),
        )

        assert rec.llm_used is False
        assert rec.fallback_reason is not None
        assert "validation_failed" in rec.fallback_reason
        assert "empty_llm_output" in rec.fallback_reason


# ── Secret Protection & Key Scrubbing ──────────────────────────────────────────

class TestApiKeyProtection:
    def test_api_key_never_appears_in_logs_or_exceptions(self, caplog):
        """Verify that an API key is completely redacted and never leaks to logs or tracebacks."""
        fake_secret_key = "AIzaSyFakeSecretKeyTesting1234567890"

        # Test sanitize_key helper
        raw_msg = f"Error calling endpoint with key {fake_secret_key} failed"
        clean = sanitize_key(raw_msg, fake_secret_key)
        assert fake_secret_key not in clean
        assert "[REDACTED_KEY]" in clean

        # Test provider sanitization under error
        with caplog.at_level(logging.DEBUG):
            provider = GeminiProvider(api_key=fake_secret_key)
            with patch("requests.post") as mock_post:
                mock_post.side_effect = requests.exceptions.RequestException(
                    f"403 Forbidden for {fake_secret_key}"
                )
                with pytest.raises(RuntimeError) as exc_info:
                    provider.generate("Test prompt")

                assert fake_secret_key not in str(exc_info.value)
                assert fake_secret_key not in caplog.text


# ── Uncertainty & Escalation Tests ─────────────────────────────────────────────

class TestUncertaintyAndEscalation:
    def test_uncertainty_wording_when_escalate_to_human(self):
        """When escalate_to_human=True, differential top-3 and agronomist referral are mandatory."""
        vision = make_vision(
            top_label="Tomato___Early_blight",
            confidence=0.45,
            top_k=[
                ("Tomato___Early_blight", 0.45),
                ("Tomato___Late_blight", 0.35),
                ("Tomato___Leaf_Mold", 0.15),
            ],
            blurry=True,
            escalate_to_human=True,
        )
        risk = make_risk(risk_level="moderate", risk_score=0.45, escalate_to_human=True)
        rag = make_rag(model_class="Tomato___Early_blight", disease_name="Early Blight")

        agent = RecommendationAgent(provider=MockLLMProvider("{}"))
        rec = agent.recommend(vision=vision, rag=rag, weather=make_weather(), risk=risk)

        assert rec.urgency == "consult_agronomist"
        assert rec.escalate_to_human is True
        assert rec.uncertainty_note is not None
        assert "uncertain" in rec.uncertainty_note.lower()
        assert "Tomato___Early_blight" in rec.uncertainty_note
        assert "Tomato___Late_blight" in rec.uncertainty_note
        assert "agronomist" in rec.uncertainty_note.lower()
        assert "ADVISORY ONLY" in rec.what_to_do_first
        assert "ESCALATE_TO_HUMAN" in rec.flags

    def test_single_class_crop_cannot_diagnose(self):
        """When can_diagnose=False (single-class crop), agronomist escalation is enforced."""
        vision = make_vision(top_label="Squash___Powdery_mildew", can_diagnose=False)
        risk = make_risk(risk_level="moderate", escalate_to_human=True)
        rag = make_rag(model_class="Squash___Powdery_mildew", found=False)

        agent = RecommendationAgent(provider=MockLLMProvider("{}"))
        rec = agent.recommend(vision=vision, rag=rag, weather=make_weather(), risk=risk)

        assert rec.urgency == "consult_agronomist"
        assert rec.escalate_to_human is True

    def test_healthy_leaf_note_present(self):
        """A healthy leaf prediction must carry a note that diseased leaves are sometimes missed (I40)."""
        vision = make_vision(top_label="Tomato___healthy", confidence=0.95)
        rag = make_rag(model_class="Tomato___healthy", found=False, disease_name="Healthy Tomato")
        risk = make_risk(risk_level="low", risk_score=0.0)

        agent = RecommendationAgent(provider=MockLLMProvider("{}"))
        rec = agent.recommend(vision=vision, rag=rag, weather=make_weather(), risk=risk)

        assert rec.urgency == "monitor"
        assert rec.uncertainty_note is not None
        assert "miss" in rec.uncertainty_note.lower()
        assert any("miss" in c.lower() for c in rec.caveats)


# ── RAG Not Found & Squash Powdery Mildew Tests ────────────────────────────────

class TestRAGNotFoundHandling:
    def test_rag_not_found_handled_without_invention(self):
        """When RAG found=False, say so plainly and do not invent advice (spec 4b)."""
        rag = RAGOutput(
            model_class="Squash___Powdery_mildew",
            found=False,
            kb_gap=True,
            reason="no_kb_chapter",
        )
        risk = make_risk(risk_level="moderate", is_gap_row=True, escalate_to_human=True)
        vision = make_vision(top_label="Squash___Powdery_mildew", can_diagnose=False)

        agent = RecommendationAgent(provider=MockLLMProvider("{}"))
        rec = agent.recommend(vision=vision, rag=rag, weather=make_weather(), risk=risk)

        assert rec.llm_used is False
        assert rec.fallback_reason == "rag_not_found"
        assert rec.urgency == "consult_agronomist"
        assert rec.citations == []
        assert rec.rag_grounded is False
        assert "RAG_CHAPTER_NOT_FOUND" in rec.flags
        assert "Squash___Powdery_mildew" in rec.summary
        assert "Consult a local agronomist" in rec.what_to_do_first


# ── Provisional Threat & Unverified Weather Tests ──────────────────────────────

class TestProvisionalAndUnverifiedNotes:
    def test_gap_risk_row_shows_provisional_note(self):
        """Risk derived from a GAP threat row must show a provisional note (I19)."""
        risk = make_risk(risk_level="moderate", is_gap_row=True)
        rag = make_rag(model_class="Apple___Cedar_apple_rust")

        agent = RecommendationAgent(provider=MockLLMProvider("{}"))
        rec = agent.recommend(
            vision=make_vision("Apple___Cedar_apple_rust"),
            rag=rag,
            weather=make_weather(),
            risk=risk,
        )

        assert "PROVISIONAL_SEVERITY_GAP_ROW" in rec.flags
        assert any("provisional" in c.lower() for c in rec.caveats)

    def test_weather_note_marked_unverified(self):
        """Weather advice must carry the [UNVERIFIED — I14] tag."""
        weather = make_weather(available=True, score=0.6, triggered_rules=["High humidity + cool nights"])
        agent = RecommendationAgent(provider=MockLLMProvider("{}"))

        rec = agent.recommend(
            vision=make_vision(),
            rag=make_rag(),
            weather=weather,
            risk=make_risk(),
        )

        assert rec.weather_note is not None
        assert "[UNVERIFIED — I14]" in rec.weather_note
        assert "WEATHER_UNVERIFIED_I14" in rec.flags


# ── Provider Switch Tests ──────────────────────────────────────────────────────

class TestProviderSwitch:
    def test_provider_switch_gemini_and_ollama(self, monkeypatch):
        """Provider switch via LLM_PROVIDER or get_provider selects gemini or ollama."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-12345678901234567890")

        # Test Gemini selection
        prov_gemini = get_provider("gemini")
        assert isinstance(prov_gemini, GeminiProvider)

        # Test Ollama selection
        prov_ollama = get_provider("ollama")
        assert isinstance(prov_ollama, OllamaProvider)

    def test_unknown_provider_raises_or_falls_back(self):
        """Unknown provider name raises ValueError from factory or falls back in agent."""
        with pytest.raises(ValueError) as exc:
            get_provider("unknown_provider_xyz")
        assert "unknown_provider_xyz" in str(exc.value)

        # Agent with unknown provider falls back gracefully without crashing
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown_provider_xyz"}):
            agent = RecommendationAgent()
            rec = agent.recommend(
                vision=make_vision(),
                rag=make_rag(),
                weather=make_weather(),
                risk=make_risk(),
            )
            assert rec.llm_used is False
            assert "unknown_provider_xyz" in rec.fallback_reason


# ── Dose Scrubbing & Hiding Tests ──────────────────────────────────────────────

class TestDoseHidingAndScrubbing:
    def test_include_doses_false_no_doses_in_prompt_and_output(self):
        """When include_doses=False, prompt has no chemical rates and output is scrubbed."""
        rag = make_rag(
            model_class="Tomato___Late_blight",
            what_to_do_first="Spray copper oxychloride at 2.5 g/L per acre.",
            include_chemical_options=True,
        )
        vision = make_vision()
        weather = make_weather()
        risk = make_risk()

        # Check prompt text
        prompt_dict = build_recommendation_prompt(
            vision=vision,
            rag=rag,
            weather=weather,
            risk=risk,
            include_doses=False,
        )
        assert "2.5 g/L" not in prompt_dict["prompt"]
        assert "Chemical Options" not in prompt_dict["prompt"]

        # Run recommendation with mock LLM
        valid_response = json.dumps({
            "summary": "Late blight detected. Cultural sanitation recommended.",
            "what_to_do_first": "Remove and destroy all infected foliage immediately.",
            "steps": ["Prune lower leaves."],
            "weather_note": None,
            "uncertainty_note": None,
        })
        provider = MockLLMProvider(handler=valid_response)
        agent = RecommendationAgent(provider=provider)

        rec = agent.recommend(
            vision=vision,
            rag=rag,
            weather=weather,
            risk=risk,
            include_doses=False,
        )

        assert "DOSES_SCRUBBED" in rec.flags
        assert "2.5 g/L" not in rec.what_to_do_first
        assert "2.5 g/L" not in " ".join(rec.steps)
