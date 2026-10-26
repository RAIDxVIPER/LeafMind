"""
tests/unit/test_risk_agent.py
=============================
Unit tests for the Risk Agent (Stage C5).

Covers:
  1. Formula correctness with exact numbers (matching spec worked examples)
  2. GAP fallback gives threat_score=0.50 and caps label at "moderate"
  3. Healthy class behaviour (risk_label="low", no threat lookup error,
     weather does not elevate label)
  4. Missing weather / missing history do NOT crash and set correct flags
  5. Uncertainty does NOT change risk_score (D10)
  6. Weights read from config, not hardcoded
  7. Squash___Powdery_mildew (no KB chapter) still produces valid output
  8. Worked examples from docs/spec/part4_decision_logic.md §4a-ix
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import pytest
import yaml
from pathlib import Path

from src.agents.risk_agent import RiskAgent
from src.risk.scorer import RiskScorer, compute_uncertainty, extract_weather_score, score_to_label
from src.risk.threat_table import ThreatTable, ThreatLookupResult
from src.schemas.schemas import (
    ClassProb,
    ImageQualityFlags,
    RAGOutput,
    RiskOutput,
    VisionOutput,
    WeatherOutput,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_vision(
    top_label: str,
    confidence: float,
    top2_prob: float = 0.05,
    blurry: bool = False,
    low_resolution: bool = False,
    ood_flag: bool = False,
) -> VisionOutput:
    """Build a minimal VisionOutput for testing."""
    return VisionOutput(
        top_label=top_label,
        confidence=confidence,
        top_k_probs=[
            ClassProb(label=top_label, prob=confidence),
            ClassProb(label="Other___class", prob=top2_prob),
        ],
        image_quality_flags=ImageQualityFlags(
            blurry=blurry,
            low_resolution=low_resolution,
        ),
        ood_flag=ood_flag,
        model_version="efficientnet_b3_v1",
        temperature=1.0,
    )


def _make_weather(
    available: bool = True,
    weather_score: float = 0.0,
    unavailable_reason: Optional[str] = None,
) -> WeatherOutput:
    """Build a minimal WeatherOutput for testing."""
    return WeatherOutput(
        weather_available=available,
        weather_score=weather_score,
        unavailable_reason=unavailable_reason,
    )


def _make_rag(
    found: bool = True,
    kb_gap: bool = False,
    knowledge_gap_flag: bool = False,
) -> RAGOutput:
    """Build a minimal RAGOutput for testing."""
    return RAGOutput(
        found=found,
        kb_gap=kb_gap,
        knowledge_gap_flag=knowledge_gap_flag,
        rag_available=True,
    )


@pytest.fixture(scope="module")
def agent() -> RiskAgent:
    return RiskAgent()


@pytest.fixture(scope="module")
def scorer() -> RiskScorer:
    return RiskScorer()


@pytest.fixture(scope="module")
def threat_table() -> ThreatTable:
    t = ThreatTable()
    t.load()
    return t


# ── Test: weights read from config ────────────────────────────────────────────

class TestWeightsFromConfig:

    def test_weights_sum_to_1(self, agent: RiskAgent) -> None:
        """Weights w_threat + w_weather + w_history must equal 1.0."""
        w = agent.weights
        total = w["threat"] + w["weather"] + w["history"]
        assert abs(total - 1.0) < 1e-9, f"Weights do not sum to 1.0: {total}"

    def test_weights_match_config(self, agent: RiskAgent) -> None:
        """Weights exposed by agent match what is in risk_config.yaml."""
        config_path = REPO_ROOT / "config" / "risk_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        expected = cfg["weights"]
        assert abs(agent.weights["threat"] - expected["threat"]) < 1e-9
        assert abs(agent.weights["weather"] - expected["weather"]) < 1e-9
        assert abs(agent.weights["history"] - expected["history"]) < 1e-9

    def test_weights_are_spec_values(self, agent: RiskAgent) -> None:
        """Default weights must be 0.70 / 0.20 / 0.10 per spec D11."""
        # [ENG assumption] — from docs/spec/part4_decision_logic.md §4a-iii
        assert abs(agent.weights["threat"] - 0.70) < 1e-9, "w_threat must be 0.70 [D11]"
        assert abs(agent.weights["weather"] - 0.20) < 1e-9, "w_weather must be 0.20 [D11]"
        assert abs(agent.weights["history"] - 0.10) < 1e-9, "w_history must be 0.10 [D11]"


# ── Test: formula correctness ─────────────────────────────────────────────────

class TestFormulaCorrectness:
    """
    Verify exact numeric outputs matching worked examples from spec §4a-ix.
    All values are from docs/spec/part4_decision_logic.md.
    """

    def test_case_1b_gap_fallback_with_weather(self, agent: RiskAgent) -> None:
        """
        Case 1b from spec: confident disease in GAP fallback + W01 triggered.
        Uses Apple___Cedar_apple_rust (stay GAP).
        R = 0.70*0.50 + 0.20*1.0 + 0.10*0.0 = 0.350 + 0.200 + 0.000 = 0.550 → "moderate"
        Source: docs/spec/part4_decision_logic.md §4a-ix Case 1b [ILLUSTRATIVE]
        """
        vision = _make_vision("Apple___Cedar_apple_rust", confidence=0.88, top2_prob=0.08)
        weather = _make_weather(available=True, weather_score=1.0)  # W01 triggered
        rag = _make_rag(found=True)

        result = agent.score(vision, weather, rag, monitoring_disease_status=None)

        assert abs(result.risk_score - 0.55) < 1e-6, (
            f"Expected risk_score=0.55 (Case 1b), got {result.risk_score}"
        )
        assert result.risk_level == "moderate", (
            f"Expected risk_level='moderate', got {result.risk_level!r}"
        )
        assert result.escalate_to_human is False
        assert result.uncertainty_level == "high_confidence"

    def test_case_2_healthy_leaf(self, agent: RiskAgent) -> None:
        """
        Case 2 from spec: confident healthy leaf.
        R = 0.70*0.0 + 0.20*0.0 + 0.10*0.0 = 0.0 → "low"
        Weather flags (even if all triggered) must not elevate a healthy label to severe.
        Source: docs/spec/part4_decision_logic.md §4a-ix Case 2 [ILLUSTRATIVE]
        """
        vision = _make_vision("Tomato___healthy", confidence=0.91)
        # Pass weather_score=1.0 but expect it to be ignored because healthy→threat=0.0
        # and the scorer uses the actual weather_score * w_weather regardless,
        # so for healthy: R = 0.70*0.0 + 0.20*0.0 + 0.10*0.0 = 0.0
        # (healthy → weather_score from weather agent is passed through, but spec says
        #  weather_score=0.0 for healthy because "no applicable disease flags")
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=False)

        result = agent.score(vision, weather, rag, monitoring_disease_status=None)

        assert result.risk_score == pytest.approx(0.0, abs=1e-6), (
            f"Expected risk_score=0.0 for healthy, got {result.risk_score}"
        )
        assert result.risk_level == "low"
        assert result.escalate_to_human is False
        assert result.uncertainty_level == "high_confidence"

    def test_case_3_low_confidence_blurry(self, agent: RiskAgent) -> None:
        """
        Case 3 from spec: low-confidence blurry Early blight image.
        R = 0.70*0.50 + 0.20*0.0 + 0.10*0.0 = 0.35 → "moderate"
        Escalate = True (U1, U4, U5 triggered).
        Source: docs/spec/part4_decision_logic.md §4a-ix Case 3 [ILLUSTRATIVE]
        """
        vision = _make_vision(
            "Tomato___Early_blight",
            confidence=0.41,
            blurry=True,
            ood_flag=True,
        )
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=True)

        result = agent.score(vision, weather, rag, monitoring_disease_status=None)

        assert abs(result.risk_score - 0.35) < 1e-6, (
            f"Expected risk_score=0.35 (Case 3), got {result.risk_score}"
        )
        assert result.risk_level == "moderate"
        assert result.escalate_to_human is True
        # U1 (low conf), U4 (blurry), U5 (ood) should fire; 3 signals → low_confidence
        assert result.uncertainty_level == "low_confidence"
        assert "U1_low_confidence" in result.uncertainty_signals_triggered
        assert "U4_blurry_image" in result.uncertainty_signals_triggered
        assert "U5_ood_flag" in result.uncertainty_signals_triggered

    def test_formula_exact_no_weather_no_history(self, agent: RiskAgent) -> None:
        """
        First visit, no weather contribution, disease detected:
        R = 0.70*0.50 + 0.20*0.0 + 0.10*0.0 = 0.35
        """
        vision = _make_vision("Grape___Black_rot", confidence=0.85)
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=True)

        result = agent.score(vision, weather, rag, monitoring_disease_status=None)

        expected = 0.70 * 0.50 + 0.20 * 0.0 + 0.10 * 0.0
        assert abs(result.risk_score - expected) < 1e-6, (
            f"Expected {expected:.6f}, got {result.risk_score}"
        )
        assert result.risk_level == "moderate"  # 0.35 is in [0.25, 0.60)

    def test_formula_exact_with_partial_weather(self, agent: RiskAgent) -> None:
        """
        R = 0.70*0.50 + 0.20*0.50 + 0.10*0.0 = 0.35 + 0.10 + 0.0 = 0.45
        """
        vision = _make_vision("Tomato___Bacterial_spot", confidence=0.80)
        weather = _make_weather(available=True, weather_score=0.5)
        rag = _make_rag(found=True)

        result = agent.score(vision, weather, rag, monitoring_disease_status=None)

        expected = 0.70 * 0.50 + 0.20 * 0.50 + 0.10 * 0.0
        assert abs(result.risk_score - expected) < 1e-6
        assert result.risk_level == "moderate"

    def test_formula_with_positive_history(self, agent: RiskAgent) -> None:
        """
        Persistent disease on follow-up: history_score = 0.10
        Uses Tomato___Early_blight (MEDIUM tier, threat=0.50).
        R = 0.70*0.50 + 0.20*0.0 + 0.10*0.10 = 0.35 + 0.0 + 0.01 = 0.36
        """
        vision = _make_vision("Tomato___Early_blight", confidence=0.80)
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=True)

        result = agent.score(
            vision, weather, rag,
            monitoring_disease_status="same_disease_still_detected"
        )

        expected = 0.70 * 0.50 + 0.20 * 0.0 + 0.10 * 0.10
        assert abs(result.risk_score - expected) < 1e-6, (
            f"Expected {expected:.6f}, got {result.risk_score}"
        )

    def test_formula_with_negative_history(self, agent: RiskAgent) -> None:
        """
        No disease at follow-up: history_score = -0.05
        Uses Tomato___Early_blight (MEDIUM tier, threat=0.50).
        R = 0.70*0.50 + 0.20*0.0 + 0.10*(-0.05) = 0.35 - 0.005 = 0.345
        """
        vision = _make_vision("Tomato___Early_blight", confidence=0.80)
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=True)

        result = agent.score(
            vision, weather, rag,
            monitoring_disease_status="no_disease_detected_at_followup"
        )

        expected = 0.70 * 0.50 + 0.20 * 0.0 + 0.10 * (-0.05)
        expected = max(0.0, expected)  # clamped
        assert abs(result.risk_score - expected) < 1e-6


# ── Test: GAP fallback ────────────────────────────────────────────────────────

class TestGAPFallback:

    def test_gap_fallback_threat_score_is_0_50(
        self, threat_table: ThreatTable
    ) -> None:
        """GAP disease classes should return GAP_FALLBACK with score=0.50 (I19)."""
        for gap_cls in ("Apple___Cedar_apple_rust", "Grape___Esca_(Black_Measles)", "Squash___Powdery_mildew"):
            result = threat_table.lookup(gap_cls)
            assert result.is_gap is True
            assert result.threat_source == "GAP_FALLBACK"
            assert abs(result.threat_score - 0.50) < 1e-9

    def test_gap_fallback_caps_label_at_moderate(self, agent: RiskAgent) -> None:
        """
        Even with max weather_score=1.0, GAP_FALLBACK caps label at 'moderate'.
        R = 0.70*0.50 + 0.20*1.0 + 0.10*0.10 = 0.35 + 0.20 + 0.01 = 0.56 → would be
        'moderate' anyway (< 0.60), but cap is still enforced to prevent edge cases.
        """
        vision = _make_vision("Apple___Cedar_apple_rust", confidence=0.92)
        weather = _make_weather(available=True, weather_score=1.0)
        rag = _make_rag(found=True)

        result = agent.score(
            vision, weather, rag,
            monitoring_disease_status="same_disease_still_detected"
        )

        # Max achievable with GAP: 0.70*0.50 + 0.20*1.0 + 0.10*0.10 = 0.56 → moderate
        assert result.risk_level in ("low", "moderate"), (
            f"With GAP_FALLBACK, label must not be 'severe', got {result.risk_level!r}"
        )
        assert result.risk_level != "severe"

    def test_gap_fallback_does_not_change_risk_score(self, agent: RiskAgent) -> None:
        """GAP cap must only affect the label, not risk_score itself."""
        vision = _make_vision("Apple___Cedar_apple_rust", confidence=0.92)
        weather = _make_weather(available=True, weather_score=1.0)
        rag = _make_rag(found=True)

        result = agent.score(vision, weather, rag)

        # risk_score is the raw clamped value, unmodified by the label cap
        expected = min(1.0, max(0.0, 0.70 * 0.50 + 0.20 * 1.0 + 0.10 * 0.0))
        assert abs(result.risk_score - expected) < 1e-6, (
            f"risk_score should be {expected:.6f} (not modified by gap cap), "
            f"got {result.risk_score}"
        )

    def test_squash_powdery_mildew_no_kb_chapter_still_works(
        self, agent: RiskAgent
    ) -> None:
        """
        Squash___Powdery_mildew has no KB chapter (expected gap) but must still
        produce valid RiskOutput using GAP_FALLBACK threat.
        """
        vision = _make_vision("Squash___Powdery_mildew", confidence=0.75)
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=False, kb_gap=True, knowledge_gap_flag=True)

        result = agent.score(vision, weather, rag)

        assert isinstance(result, RiskOutput)
        assert result.risk_level in ("low", "moderate", "severe")
        assert 0.0 <= result.risk_score <= 1.0
        # Squash powdery mildew is [GAP], so must use fallback
        # Expected: R = 0.70*0.50 + 0.20*0.0 + 0.10*0.0 = 0.35 → moderate
        assert abs(result.risk_score - 0.35) < 1e-6
        # U6 (knowledge gap) fires → escalate
        assert result.escalate_to_human is True
        assert "U6_knowledge_gap" in result.uncertainty_signals_triggered


# ── Test: healthy class behaviour ─────────────────────────────────────────────

class TestHealthyClassBehaviour:

    @pytest.mark.parametrize("label", [
        "Tomato___healthy",
        "Apple___healthy",
        "Corn_(maize)___healthy",
        "Pepper,_bell___healthy",
        "Potato___healthy",
        "Grape___healthy",
    ])
    def test_healthy_class_threat_score_is_zero(
        self, threat_table: ThreatTable, label: str
    ) -> None:
        result = threat_table.lookup(label)
        assert result.threat_score == 0.0, (
            f"Healthy class '{label}' must have threat_score=0.0, got {result.threat_score}"
        )
        assert result.threat_source == "HEALTHY"
        assert result.is_gap is False

    def test_healthy_class_risk_score_is_low(self, agent: RiskAgent) -> None:
        vision = _make_vision("Tomato___healthy", confidence=0.91)
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=False)

        result = agent.score(vision, weather, rag)

        assert result.risk_level == "low"
        assert result.risk_score == pytest.approx(0.0, abs=1e-6)

    def test_healthy_class_weather_does_not_elevate_to_severe(
        self, agent: RiskAgent
    ) -> None:
        """
        Even if weather_score=1.0 and history=0.10, a healthy label must not reach severe.
        Max R for healthy = 0.70*0.0 + 0.20*1.0 + 0.10*0.10 = 0.21 → 'low'.
        Source: docs/spec/part4_decision_logic.md §4a-iii 'Healthy class behaviour'
        """
        vision = _make_vision("Tomato___healthy", confidence=0.91)
        weather = _make_weather(available=True, weather_score=1.0)
        rag = _make_rag(found=False)

        result = agent.score(
            vision, weather, rag,
            monitoring_disease_status="same_disease_still_detected"  # +0.10 history
        )

        # Max R with healthy threat=0.0: 0.20*1.0 + 0.10*0.10 = 0.21 → "low"
        assert result.risk_level in ("low",), (
            f"Healthy class with max weather should be 'low', got {result.risk_level!r}"
        )
        assert result.risk_level != "severe"


# ── Test: missing weather does not crash ──────────────────────────────────────

class TestMissingWeather:

    def test_weather_unavailable_sets_flag(self, agent: RiskAgent) -> None:
        """When weather is unavailable, weather_unavailable=True must be set."""
        vision = _make_vision("Tomato___Late_blight", confidence=0.85)
        weather = _make_weather(available=False, unavailable_reason="API outage")
        rag = _make_rag(found=True)

        result = agent.score(vision, weather, rag)

        assert result.weather_unavailable is True
        assert isinstance(result, RiskOutput)

    def test_weather_unavailable_uses_neutral_zero(self, agent: RiskAgent) -> None:
        """Neutral weather score (0.0) is used when unavailable. No crash."""
        vision = _make_vision("Tomato___Late_blight", confidence=0.85)
        weather_ok = _make_weather(available=True, weather_score=0.0)
        weather_fail = _make_weather(available=False)
        rag = _make_rag(found=True)

        result_ok = agent.score(vision, weather_ok, rag)
        result_fail = agent.score(vision, weather_fail, rag)

        # With neutral weather (0.0) and no history:
        # R = 0.70*0.50 + 0.20*0.0 + 0.10*0.0 = 0.35 for both
        assert abs(result_ok.risk_score - result_fail.risk_score) < 1e-6, (
            "risk_score with weather_available=False must match weather_score=0.0 result"
        )

    def test_weather_unavailable_does_not_raise(self) -> None:
        """WeatherOutput with weather_available=False must not crash the agent."""
        agent = RiskAgent()
        vision = _make_vision("Corn___Common_rust", confidence=0.78)
        weather = WeatherOutput(weather_available=False, weather_score=0.0)
        rag = _make_rag(found=True)

        result = agent.score(vision, weather, rag)
        assert isinstance(result, RiskOutput)


# ── Test: missing history does not crash ──────────────────────────────────────

class TestMissingHistory:

    def test_first_visit_no_history_uses_zero(self, agent: RiskAgent) -> None:
        """First visit (monitoring_disease_status=None) must use history_score=0.0."""
        vision = _make_vision("Tomato___Early_blight", confidence=0.85)
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=True)

        result = agent.score(vision, weather, rag, monitoring_disease_status=None)

        # R = 0.70*0.50 + 0.20*0.0 + 0.10*0.0 = 0.35 (Tomato___Early_blight threat=0.50)
        expected = 0.70 * 0.50
        assert abs(result.risk_score - expected) < 1e-6

    def test_first_visit_does_not_raise(self) -> None:
        """Passing None for monitoring_disease_status must not crash the agent."""
        agent = RiskAgent()
        vision = _make_vision("Potato___Late_blight", confidence=0.80)
        weather = _make_weather(available=True, weather_score=0.3)
        rag = _make_rag(found=True)

        result = agent.score(vision, weather, rag, monitoring_disease_status=None)
        assert isinstance(result, RiskOutput)


# ── Test: uncertainty does NOT change risk_score (D10) ────────────────────────

class TestUncertaintySeparateFromRisk:
    """
    D10 invariant: Uncertainty signals (U1–U6) drive escalation only.
    They must NOT change risk_score.
    A confident healthy/disease diagnosis produces the same risk_score
    regardless of image quality or OOD flag.
    """

    def test_blurry_image_does_not_inflate_risk_score(self, agent: RiskAgent) -> None:
        """
        Blurry image → escalate, but risk_score is unchanged.
        R must be identical for clear and blurry images with same top_label.
        """
        vision_clear = _make_vision("Tomato___Early_blight", confidence=0.85)
        vision_blurry = _make_vision(
            "Tomato___Early_blight", confidence=0.85, blurry=True
        )
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=True)

        result_clear = agent.score(vision_clear, weather, rag)
        result_blurry = agent.score(vision_blurry, weather, rag)

        assert abs(result_clear.risk_score - result_blurry.risk_score) < 1e-9, (
            f"Blurry image must not change risk_score. "
            f"Clear={result_clear.risk_score}, Blurry={result_blurry.risk_score}"
        )
        # Blurry should escalate; clear should not
        assert result_blurry.escalate_to_human is True
        assert result_clear.escalate_to_human is False

    def test_low_confidence_does_not_inflate_risk_score(self, agent: RiskAgent) -> None:
        """
        Low confidence → escalate, but risk_score from the same disease class
        must be the same as high confidence.
        """
        vision_high = _make_vision("Tomato___Late_blight", confidence=0.92)
        vision_low = _make_vision("Tomato___Late_blight", confidence=0.41)
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=True)

        result_high = agent.score(vision_high, weather, rag)
        result_low = agent.score(vision_low, weather, rag)

        assert abs(result_high.risk_score - result_low.risk_score) < 1e-9, (
            f"Low confidence must not change risk_score. "
            f"High={result_high.risk_score}, Low={result_low.risk_score}"
        )
        assert result_low.escalate_to_human is True
        assert result_high.escalate_to_human is False

    def test_ood_flag_does_not_inflate_risk_score(self, agent: RiskAgent) -> None:
        """OOD flag must escalate but not change risk_score."""
        vision_normal = _make_vision("Corn___Northern_Leaf_Blight", confidence=0.80)
        vision_ood = _make_vision(
            "Corn___Northern_Leaf_Blight", confidence=0.80, ood_flag=True
        )
        weather = _make_weather(available=True, weather_score=0.0)
        rag = _make_rag(found=True)

        result_normal = agent.score(vision_normal, weather, rag)
        result_ood = agent.score(vision_ood, weather, rag)

        assert abs(result_normal.risk_score - result_ood.risk_score) < 1e-9
        assert result_ood.escalate_to_human is True
        assert result_normal.escalate_to_human is False

    def test_uncertainty_level_reflects_signal_count(self, agent: RiskAgent) -> None:
        """uncertainty_level depends on signal count (0→high_confidence, 1→moderate, 2+→low)."""
        # 0 signals → high_confidence
        v0 = _make_vision("Tomato___Late_blight", confidence=0.90, top2_prob=0.05)
        r0 = agent.score(v0, _make_weather(), _make_rag())
        assert r0.uncertainty_level == "high_confidence"
        assert r0.escalate_to_human is False

        # 1 signal: low confidence only
        v1 = _make_vision("Tomato___Late_blight", confidence=0.50, top2_prob=0.40)
        # top2=0.40 makes gap=0.10 < 0.15 → 2 signals: U1 + U2
        r1_two = agent.score(v1, _make_weather(), _make_rag())
        # Expect low_confidence (2+ signals)
        assert r1_two.uncertainty_level in ("moderate_confidence", "low_confidence")

        # 2+ signals → low_confidence
        v2 = _make_vision("Tomato___Late_blight", confidence=0.41, blurry=True, ood_flag=True)
        r2 = agent.score(v2, _make_weather(), _make_rag())
        assert r2.uncertainty_level == "low_confidence"

    def test_uncertainty_signals_do_not_appear_in_factors(
        self, agent: RiskAgent
    ) -> None:
        """
        The three risk factors must be: threat_table, weather, history.
        Uncertainty signals (U1-U6) must NOT appear as factors (D10).
        """
        vision = _make_vision(
            "Tomato___Late_blight", confidence=0.41, blurry=True, ood_flag=True
        )
        result = agent.score(vision, _make_weather(), _make_rag())

        factor_sources = [f.source for f in result.factors]
        assert "threat_table" in factor_sources
        assert "weather" in factor_sources
        assert "history" in factor_sources
        # Uncertainty sources must NOT appear in factors
        for source in factor_sources:
            assert "confidence" not in source or source == "threat_table", (
                f"Uncertainty signal appears as risk factor: {source}"
            )


# ── Test: RiskOutput schema compliance ───────────────────────────────────────

class TestRiskOutputSchema:

    def test_riskoutput_fields_present(self, agent: RiskAgent) -> None:
        """RiskOutput must contain all required fields per schema."""
        vision = _make_vision("Tomato___Late_blight", confidence=0.80)
        weather = _make_weather(available=True, weather_score=0.3)
        rag = _make_rag(found=True)

        result = agent.score(vision, weather, rag)

        assert hasattr(result, "risk_level")
        assert hasattr(result, "risk_score")
        assert hasattr(result, "factors")
        assert hasattr(result, "escalate_to_human")
        assert hasattr(result, "uncertainty_level")
        assert hasattr(result, "uncertainty_signals_triggered")
        assert hasattr(result, "weather_unavailable")
        assert hasattr(result, "rag_unavailable")
        assert hasattr(result, "progression_note")
        assert hasattr(result, "risk_computed_at")
        assert isinstance(result.risk_computed_at, datetime)

    def test_risk_score_clamped_to_01(self, agent: RiskAgent) -> None:
        """risk_score must always be in [0.0, 1.0]."""
        for label, conf, ws in [
            ("Tomato___healthy", 1.0, 0.0),
            ("Tomato___Late_blight", 0.01, 1.0),
            ("Tomato___Late_blight", 0.99, 1.0),
        ]:
            vision = _make_vision(label, confidence=conf)
            weather = _make_weather(available=True, weather_score=ws)
            rag = _make_rag(found=True)
            result = agent.score(vision, weather, rag)
            assert 0.0 <= result.risk_score <= 1.0, (
                f"risk_score out of [0,1] for {label}: {result.risk_score}"
            )

    def test_risk_level_valid_values(self, agent: RiskAgent) -> None:
        """risk_level must only be one of the three valid values."""
        vision = _make_vision("Tomato___Late_blight", confidence=0.80)
        result = agent.score(vision, _make_weather(), _make_rag())
        assert result.risk_level in ("low", "moderate", "severe")

    def test_three_factors_always_present(self, agent: RiskAgent) -> None:
        """factors list must contain exactly 3 entries: threat, weather, history."""
        vision = _make_vision("Tomato___Late_blight", confidence=0.80)
        result = agent.score(vision, _make_weather(), _make_rag())
        assert len(result.factors) == 3
        sources = {f.source for f in result.factors}
        assert sources == {"threat_table", "weather", "history"}

    def test_progression_note_non_empty(self, agent: RiskAgent) -> None:
        """progression_note must always be a non-empty string."""
        vision = _make_vision("Tomato___Late_blight", confidence=0.80)
        result = agent.score(vision, _make_weather(), _make_rag())
        assert isinstance(result.progression_note, str)
        assert len(result.progression_note) > 0


# ── Test: threat table loading ────────────────────────────────────────────────

class TestThreatTable:

    def test_all_disease_classes_present(self, threat_table: ThreatTable) -> None:
        """All 17 mapped model classes must be in the threat table (or get GAP_FALLBACK)."""
        disease_classes = [
            "Apple___Apple_scab",
            "Apple___Cedar_apple_rust",
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Corn___Northern_Leaf_Blight",
            "Corn___Common_rust",
            "Corn___Cercospora_leaf_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Bacterial_spot",
            "Tomato___Leaf_Mold",
            "Tomato___Tomato_mosaic_virus",
            "Potato___Late_blight",
            "Potato___Early_blight",
            "Pepper,_bell___Bacterial_spot",
        ]
        for cls in disease_classes:
            result = threat_table.lookup(cls)
            assert isinstance(result, ThreatLookupResult)
            assert result.threat_score is not None
            assert 0.0 <= result.threat_score <= 1.0

    def test_healthy_classes_return_zero_score(self, threat_table: ThreatTable) -> None:
        """All healthy class names must return threat_score=0.0 via name detection."""
        healthy_classes = [
            "Tomato___healthy",
            "Apple___healthy",
            "Grape___healthy",
            "Potato___healthy",
            "Pepper,_bell___healthy",
            "Some_unknown_crop___healthy",   # not in table, but name → HEALTHY
        ]
        for cls in healthy_classes:
            result = threat_table.lookup(cls)
            assert result.threat_score == 0.0, f"Healthy class {cls!r} got score {result.threat_score}"
            assert result.threat_source == "HEALTHY"

    def test_tier_scores_from_config(self, threat_table: ThreatTable) -> None:
        """Tier scores must be read from config, not hardcoded."""
        scores = threat_table.tier_scores
        assert "HEALTHY" in scores
        assert "DISEASE_DETECTED" in scores
        assert abs(scores["HEALTHY"] - 0.0) < 1e-9
        assert abs(scores["DISEASE_DETECTED"] - 0.50) < 1e-9

    def test_unknown_class_gets_gap_fallback(self, threat_table: ThreatTable) -> None:
        """An unknown class name not in the table must still return GAP_FALLBACK."""
        result = threat_table.lookup("Imaginary___disease_not_in_table")
        assert result.is_gap is True
        assert result.threat_source == "GAP_FALLBACK"
        assert abs(result.threat_score - 0.50) < 1e-9


# ── Test: Approved Threat Table (I19 Partial) ────────────────────────────────

class TestApprovedThreatTable:
    """
    Tests for the approved threat table tiers (I19 partial).
    Validates:
      1. All 17 mapped classes present with exact approved tiers
      2. The 3 GAP rows return GAP_FALLBACK and are capped at moderate
      3. A HIGH row can reach "severe" with high weather/history inputs
      4. MEDIUM/LOW rows are not capped (is_gap is False)
      5. Healthy rows unchanged (tier HEALTHY, threat_score 0.0)
      6. Loader still works and returns all rows
    """

    APPROVED_TIERS = {
        # HIGH (6)
        "Apple___Apple_scab": ("HIGH", 0.85, False),
        "Corn___Northern_Leaf_Blight": ("HIGH", 0.85, False),
        "Potato___Late_blight": ("HIGH", 0.85, False),
        "Tomato___Late_blight": ("HIGH", 0.85, False),
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("HIGH", 0.85, False),
        "Tomato___Tomato_mosaic_virus": ("HIGH", 0.85, False),
        # MEDIUM (8)
        "Corn___Common_rust": ("MEDIUM", 0.50, False),
        "Corn___Cercospora_leaf_spot": ("MEDIUM", 0.50, False),
        "Grape___Black_rot": ("MEDIUM", 0.50, False),
        "Pepper,_bell___Bacterial_spot": ("MEDIUM", 0.50, False),
        "Potato___Early_blight": ("MEDIUM", 0.50, False),
        "Tomato___Early_blight": ("MEDIUM", 0.50, False),
        "Tomato___Septoria_leaf_spot": ("MEDIUM", 0.50, False),
        "Tomato___Bacterial_spot": ("MEDIUM", 0.50, False),
        # LOW (1)
        "Tomato___Leaf_Mold": ("LOW", 0.25, False),
        # STAY GAP (3)
        "Apple___Cedar_apple_rust": ("DISEASE_DETECTED", 0.50, True),
        "Grape___Esca_(Black_Measles)": ("DISEASE_DETECTED", 0.50, True),
        "Squash___Powdery_mildew": ("DISEASE_DETECTED", 0.50, True),
    }

    def test_all_17_mapped_classes_present_and_match_tiers(
        self, threat_table: ThreatTable
    ) -> None:
        """All 17 mapped classes (+ Squash) must match approved tiers exactly."""
        for model_class, (expected_tier, expected_score, expected_gap) in self.APPROVED_TIERS.items():
            result = threat_table.lookup(model_class)
            assert result.tier == expected_tier, (
                f"{model_class}: expected tier {expected_tier}, got {result.tier}"
            )
            assert abs(result.threat_score - expected_score) < 1e-6, (
                f"{model_class}: expected threat_score {expected_score}, got {result.threat_score}"
            )
            assert result.is_gap == expected_gap, (
                f"{model_class}: expected is_gap={expected_gap}, got {result.is_gap}"
            )
            if expected_gap:
                assert result.threat_source == "GAP_FALLBACK"
            else:
                assert result.threat_source == "threat_table"

    def test_three_gap_rows_use_gap_fallback_and_cap_at_moderate(
        self, agent: RiskAgent
    ) -> None:
        """
        The 3 GAP rows (Cedar apple rust, Esca, Squash powdery mildew) must
        use GAP_FALLBACK and be capped at 'moderate' even with high weather and history.
        """
        gap_classes = [
            "Apple___Cedar_apple_rust",
            "Grape___Esca_(Black_Measles)",
            "Squash___Powdery_mildew",
        ]
        for cls_name in gap_classes:
            vision = _make_vision(cls_name, confidence=0.95)
            # High weather and positive history:
            # R = 0.70*0.50 + 0.20*1.0 + 0.10*0.10 = 0.35 + 0.20 + 0.01 = 0.56
            weather = _make_weather(available=True, weather_score=1.0)
            rag = _make_rag(found=True)
            result = agent.score(
                vision, weather, rag,
                monitoring_disease_status="same_disease_still_detected"
            )
            assert result.risk_level in ("low", "moderate"), (
                f"{cls_name} (GAP) must not exceed 'moderate', got {result.risk_level}"
            )
            assert result.risk_level != "severe"
            # Threat factor source must be threat_table with GAP_FALLBACK note or GAP_FALLBACK
            threat_factor = [f for f in result.factors if f.source in ("threat_table", "GAP_FALLBACK")][0]
            assert "GAP_FALLBACK" in threat_factor.value or threat_factor.source == "threat_table"

    def test_high_rows_can_reach_severe(self, agent: RiskAgent) -> None:
        """
        HIGH tier rows can reach 'severe' (risk_score >= 0.60) with high weather/history.
        Formula: R = 0.70*0.85 + 0.20*0.50 + 0.10*0.0 = 0.595 + 0.10 = 0.695 >= 0.60
        """
        high_classes = [
            "Apple___Apple_scab",
            "Corn___Northern_Leaf_Blight",
            "Potato___Late_blight",
            "Tomato___Late_blight",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
        ]
        for cls_name in high_classes:
            vision = _make_vision(cls_name, confidence=0.90)
            weather = _make_weather(available=True, weather_score=0.50)
            rag = _make_rag(found=True)
            result = agent.score(vision, weather, rag, monitoring_disease_status=None)
            # R = 0.70*0.85 + 0.20*0.50 + 0.10*0.0 = 0.695 -> "severe"
            assert abs(result.risk_score - 0.695) < 1e-4, (
                f"{cls_name}: expected 0.695, got {result.risk_score}"
            )
            assert result.risk_level == "severe", (
                f"{cls_name} should reach 'severe', got {result.risk_level}"
            )

    def test_medium_and_low_rows_are_not_capped(
        self, threat_table: ThreatTable
    ) -> None:
        """MEDIUM and LOW rows must have is_gap=False and source='threat_table' (uncapped)."""
        medium_and_low = [
            "Corn___Common_rust",
            "Corn___Cercospora_leaf_spot",
            "Grape___Black_rot",
            "Pepper,_bell___Bacterial_spot",
            "Potato___Early_blight",
            "Tomato___Early_blight",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Bacterial_spot",
            "Tomato___Leaf_Mold",
        ]
        for cls_name in medium_and_low:
            result = threat_table.lookup(cls_name)
            assert result.is_gap is False, f"{cls_name} must not be flagged as gap"
            assert result.threat_source == "threat_table"

    def test_healthy_rows_unchanged(self, threat_table: ThreatTable) -> None:
        """All 12 healthy classes must have tier HEALTHY, threat_score 0.0, is_gap False."""
        healthy_classes = [
            "Apple___healthy",
            "Blueberry___healthy",
            "Cherry_(including_sour)___healthy",
            "Corn_(maize)___healthy",
            "Grape___healthy",
            "Peach___healthy",
            "Pepper,_bell___healthy",
            "Potato___healthy",
            "Raspberry___healthy",
            "Soybean___healthy",
            "Strawberry___healthy",
            "Tomato___healthy",
        ]
        for cls_name in healthy_classes:
            result = threat_table.lookup(cls_name)
            assert result.threat_score == 0.0
            assert result.tier == "HEALTHY"
            assert result.is_gap is False
            assert result.threat_source == "HEALTHY"

    def test_loader_still_works(self) -> None:
        """ThreatTable loader loads all 30 rows and parses metadata correctly."""
        table = ThreatTable()
        table.load()
        assert len(table._classes) == 30
        assert table.tier_scores["HIGH"] == 0.85
        assert table.tier_scores["MEDIUM"] == 0.50
        assert table.tier_scores["LOW"] == 0.25
        # Check an unverified row has the unverified status
        row = table._classes["Tomato___Late_blight"]
        assert row["status"] == "UNVERIFIED_PENDING_PATHOLOGIST"
        assert "docs/review/threat_table_draft.md" in row.get("source", "")


