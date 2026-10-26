"""
src/risk/scorer.py
==================
Risk score computation for the LeafMind Risk Agent (Stage C5).

Formula (D11):
    R = w_threat * threat_score + w_weather * weather_score + w_history * history_score

Weights come from config/risk_config.yaml, NOT hardcoded.

Key design principles:
- D10: Risk (disease identity) and Uncertainty (model/image quality) are NEVER conflated.
  Uncertainty signals (U1–U6) drive escalation only; they do NOT modify R.
- D15: When threat_source="GAP_FALLBACK", use 0.50 and cap label at "moderate" (I26).
- Healthy classes: threat_score=0.0, label="low" or "none", weather score=0.0.
- Weather unavailable: use neutral 0.0 and set flag. Never crash.
- History absent: use 0.0 and set flag. Never crash.
- No LLM is used anywhere in C5 (D13).

Source: docs/spec/part4_decision_logic.md §4a, part6_build_spec.md §6d-iv
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml

from src.risk.threat_table import ThreatLookupResult, ThreatTable
from src.schemas.schemas import (
    RAGOutput,
    RiskFactor,
    RiskOutput,
    VisionOutput,
    WeatherOutput,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RISK_CONFIG_PATH = REPO_ROOT / "config" / "risk_config.yaml"


# ── Helper: load config ──────────────────────────────────────────────────────

def _load_risk_config(config_path: Optional[Path] = None) -> dict:
    """Load risk_config.yaml. Returns a dict; raises on missing file."""
    path = config_path or DEFAULT_RISK_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"risk_config.yaml not found at {path}. "
            "Ensure config/risk_config.yaml exists."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Uncertainty computation (separate from risk score, D10) ──────────────────

@dataclass
class UncertaintyResult:
    """Uncertainty signals — separate from risk, per D10."""
    signals_triggered: List[str]
    uncertainty_level: str   # "high_confidence" | "moderate_confidence" | "low_confidence"
    escalate_to_human: bool


def compute_uncertainty(
    vision: VisionOutput,
    rag: RAGOutput,
    confidence_threshold: float,
    prob_gap_threshold: float,
) -> UncertaintyResult:
    """
    Evaluate U1–U6 uncertainty signals from VisionOutput and RAGOutput.

    These signals drive escalation; they do NOT modify risk_score (D10).

    U1: vision.confidence < confidence_threshold       [ENG assumption — to be calibrated]
    U2: (top1.prob - top2.prob) < prob_gap_threshold   [ENG assumption — to be calibrated]
    U3: image_quality_flags.low_resolution             [ENG]
    U4: image_quality_flags.blurry                     [ENG]
    U5: vision.ood_flag                                [ENG]
    U6: rag.knowledge_gap_flag                         [ENG]
    """
    signals: List[str] = []

    # U1: low confidence
    if vision.confidence < confidence_threshold:
        signals.append("U1_low_confidence")

    # U2: small margin between top-1 and top-2
    if len(vision.top_k_probs) >= 2:
        gap = vision.top_k_probs[0].prob - vision.top_k_probs[1].prob
        if gap < prob_gap_threshold:
            signals.append("U2_low_margin")

    # U3: low resolution
    if vision.image_quality_flags.low_resolution:
        signals.append("U3_low_resolution")

    # U4: blurry
    if vision.image_quality_flags.blurry:
        signals.append("U4_blurry_image")

    # U5: out-of-distribution flag
    if vision.ood_flag:
        signals.append("U5_ood_flag")

    # U6: RAG knowledge gap
    if rag.knowledge_gap_flag or rag.kb_gap:
        signals.append("U6_knowledge_gap")

    n = len(signals)
    if n == 0:
        level = "high_confidence"
    elif n == 1:
        level = "moderate_confidence"
    else:
        level = "low_confidence"

    return UncertaintyResult(
        signals_triggered=signals,
        uncertainty_level=level,
        escalate_to_human=(n > 0),
    )


# ── Weather score extraction ──────────────────────────────────────────────────

def extract_weather_score(
    weather: WeatherOutput,
    neutral_value: float = 0.0,
) -> tuple[float, bool, Optional[str]]:
    """
    Extract the pre-computed weather_score from WeatherOutput.

    The weather agent already computes weather_score per Part 4 §4a-iv.
    The risk agent consumes it directly; it does NOT re-evaluate rules.

    Returns:
        (score, weather_unavailable_flag, reason)
        - score: the weather score to use [0.0, 1.0]
        - weather_unavailable_flag: True when weather is not available
        - reason: explanation string when unavailable
    """
    if not weather.weather_available:
        reason = weather.unavailable_reason or "Weather data was not available."
        logger.info(
            "Weather unavailable — using neutral value %.2f. Reason: %s",
            neutral_value,
            reason,
        )
        return neutral_value, True, reason

    score = float(weather.weather_score)
    score = max(0.0, min(1.0, score))   # clamp defensively
    return score, False, None


# ── History score ──────────────────────────────────────────────────────────────

def compute_history_score(
    monitoring_disease_status: Optional[str],
    history_scores_cfg: dict,
) -> tuple[float, bool]:
    """
    Compute history_score from the previous visit's monitoring disease_status.

    Returns:
        (history_score, history_unavailable_flag)
        - history_score: numeric contribution [typically -0.05, 0.0, or 0.10]
        - history_unavailable_flag: True when no history exists (first visit)

    Source: docs/spec/part4_decision_logic.md §4a-v
    All values come from config, not hardcoded.
    """
    if monitoring_disease_status is None:
        # First visit — no history [ENG default]
        return float(history_scores_cfg.get("first_visit", 0.0)), True

    if monitoring_disease_status == "same_disease_still_detected":
        return float(history_scores_cfg.get("persistent_disease", 0.10)), False

    if monitoring_disease_status == "no_disease_detected_at_followup":
        return float(history_scores_cfg.get("no_disease_at_followup", -0.05)), False

    return float(history_scores_cfg.get("other", 0.0)), False


# ── Label from score ──────────────────────────────────────────────────────────

def score_to_label(
    score: float,
    low_upper: float,
    moderate_upper: float,
    gap_cap: Optional[str] = None,
) -> str:
    """
    Convert numeric risk_score to risk_label string.

    Applies the GAP_FALLBACK label cap (I26) when gap_cap is set to "moderate".
    [ENG assumption — all thresholds to be calibrated]
    """
    if score < low_upper:
        label = "low"
    elif score < moderate_upper:
        label = "moderate"
    else:
        label = "severe"

    if gap_cap and gap_cap == "moderate" and label == "severe":
        logger.info(
            "risk_score=%.4f mapped to 'severe', but threat_source=GAP_FALLBACK "
            "caps label at 'moderate' per I26. risk_score unchanged.",
            score,
        )
        label = "moderate"

    return label


# ── Progression note ──────────────────────────────────────────────────────────

def build_progression_note(
    top_label: str,
    risk_level: str,
    weather_triggered: bool,
) -> str:
    """
    Plain-language progression note heuristic.

    THIS IS NOT A QUANTITATIVE PREDICTION. No temporal data exists in the
    training set. [ENG heuristic][GAP]

    Source: docs/spec/part4_decision_logic.md §4a-viii
    """
    if "healthy" in top_label.lower():
        return (
            "No disease detected in this image. Continue routine monitoring. "
            "[ENG heuristic — not a quantitative prediction]"
        )
    elif risk_level == "severe" and weather_triggered:
        return (
            "Disease identified as potentially high-risk. Current weather conditions "
            "may favour spread. Prompt action is advisable. Verify with a follow-up "
            "image in 3-5 days or consult a local agronomist. "
            "[ENG heuristic — not a quantitative prediction; no temporal data in training set]"
        )
    elif risk_level == "severe":
        return (
            "Disease identified as potentially high-risk. "
            "Prompt treatment is advisable. Verify with a follow-up image. "
            "[ENG heuristic — not a quantitative prediction]"
        )
    elif risk_level == "moderate":
        return (
            "Disease detected at moderate risk level. "
            "Monitor closely; consider treatment if symptoms worsen. "
            "[ENG heuristic — not a quantitative prediction]"
        )
    else:  # low
        return (
            "Low risk level assessed. Continue monitoring. "
            "[ENG heuristic — not a quantitative prediction]"
        )


# ── Main scorer ───────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Computes RiskOutput from VisionOutput, WeatherOutput, RAGOutput, and
    optional monitoring history.

    Weights and thresholds are loaded from config/risk_config.yaml at
    construction time — not hardcoded.

    This class is stateless per call (no caching of prior results).
    """

    def __init__(
        self,
        threat_table: Optional[ThreatTable] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self._cfg = _load_risk_config(config_path)
        self._threat_table = threat_table or ThreatTable()
        self._threat_table.load()

        weights = self._cfg["weights"]
        self._w_threat = float(weights["threat"])    # 0.70 [ENG assumption]
        self._w_weather = float(weights["weather"])  # 0.20 [ENG assumption]
        self._w_history = float(weights["history"])  # 0.10 [ENG assumption]

        # Validate weights sum to 1.0 (within floating point tolerance)
        total = self._w_threat + self._w_weather + self._w_history
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"risk_config.yaml weights must sum to 1.0, got {total:.6f}. "
                "Fix config/risk_config.yaml."
            )

        thresholds = self._cfg["label_thresholds"]
        self._low_upper = float(thresholds["low_upper"])           # 0.25 [ENG]
        self._moderate_upper = float(thresholds["moderate_upper"]) # 0.60 [ENG]
        self._gap_cap = self._cfg.get("gap_fallback_label_cap", "moderate")

        unavail = self._cfg.get("unavailable_defaults", {})
        self._weather_neutral = float(unavail.get("weather_score_neutral", 0.0))
        self._history_neutral = float(unavail.get("history_score_neutral", 0.0))

        uncertainty_cfg = self._cfg.get("uncertainty", {})
        self._conf_threshold = float(uncertainty_cfg.get("confidence_threshold", 0.60))
        self._gap_threshold = float(uncertainty_cfg.get("prob_gap_threshold", 0.15))

        history_cfg = self._cfg.get("history_scores", {})
        self._history_scores_cfg = history_cfg

    @property
    def weights(self) -> dict[str, float]:
        """Expose current weights (read from config)."""
        return {
            "threat": self._w_threat,
            "weather": self._w_weather,
            "history": self._w_history,
        }

    def score(
        self,
        vision: VisionOutput,
        weather: WeatherOutput,
        rag: RAGOutput,
        monitoring_disease_status: Optional[str] = None,
    ) -> RiskOutput:
        """
        Compute the full RiskOutput.

        Args:
            vision: VisionOutput from the Vision Agent
            weather: WeatherOutput from the Weather Agent
            rag: RAGOutput from the RAG Agent
            monitoring_disease_status: Optional[str] — from MonitoringOutput.disease_status
                on follow-up visits; None on first visit.

        Returns:
            RiskOutput with risk_score, risk_level, factors, uncertainty, etc.
        """
        top_label = vision.top_label

        # ── 1. Threat component ────────────────────────────────────────────────
        threat_result: ThreatLookupResult = self._threat_table.lookup(top_label)
        threat_score = threat_result.threat_score
        threat_factor = RiskFactor(
            source="threat_table",
            value=(
                f"class='{top_label}', tier={threat_result.tier}, "
                f"score={threat_score:.4f}, source={threat_result.threat_source}"
            ),
            contribution=round(self._w_threat * threat_score, 6),
            tag=(
                "[ENG assumption — plant pathologist review required (I19)]"
                if threat_result.is_gap
                else "[threat_table — plant pathologist review required]"
            ),
        )

        # ── 2. Weather component ───────────────────────────────────────────────
        weather_score, weather_unavailable, weather_reason = extract_weather_score(
            weather, neutral_value=self._weather_neutral
        )
        weather_factor = RiskFactor(
            source="weather",
            value=(
                f"weather_score={weather_score:.4f}, "
                + ("unavailable" if weather_unavailable else "available")
            ),
            contribution=round(self._w_weather * weather_score, 6),
            tag=(
                "[ASSUMPTION] weather unavailable — neutral 0.0 used [ENG]"
                if weather_unavailable
                else "[UNVERIFIED — I14] weather score from weather agent"
            ),
        )

        # ── 3. History component ───────────────────────────────────────────────
        history_score, history_unavailable = compute_history_score(
            monitoring_disease_status, self._history_scores_cfg
        )
        history_factor = RiskFactor(
            source="history",
            value=(
                f"monitoring_disease_status={monitoring_disease_status!r}, "
                f"history_score={history_score:.4f}"
            ),
            contribution=round(self._w_history * history_score, 6),
            tag=(
                "[ENG default] first visit — no history [ENG assumption]"
                if history_unavailable
                else "[ENG assumption] history score from monitoring output"
            ),
        )

        # ── 4. Risk score formula ──────────────────────────────────────────────
        raw_score = (
            self._w_threat * threat_score
            + self._w_weather * weather_score
            + self._w_history * history_score
        )
        risk_score = max(0.0, min(1.0, raw_score))   # clamp to [0.0, 1.0]

        # ── 5. Risk label (with GAP cap) ───────────────────────────────────────
        gap_cap = self._gap_cap if threat_result.is_gap else None
        risk_level = score_to_label(
            risk_score,
            low_upper=self._low_upper,
            moderate_upper=self._moderate_upper,
            gap_cap=gap_cap,
        )

        # ── 6. Uncertainty signals (separate from risk score, D10) ─────────────
        uncertainty = compute_uncertainty(
            vision=vision,
            rag=rag,
            confidence_threshold=self._conf_threshold,
            prob_gap_threshold=self._gap_threshold,
        )

        # ── 7. Progression note ────────────────────────────────────────────────
        weather_triggered = weather.weather_available and weather_score > 0.0
        progression_note = build_progression_note(top_label, risk_level, weather_triggered)

        # ── 8. Assemble RiskOutput ────────────────────────────────────────────
        return RiskOutput(
            risk_level=risk_level,
            risk_score=round(risk_score, 6),
            factors=[threat_factor, weather_factor, history_factor],
            escalate_to_human=uncertainty.escalate_to_human,
            uncertainty_level=uncertainty.uncertainty_level,
            uncertainty_signals_triggered=uncertainty.signals_triggered,
            weather_unavailable=weather_unavailable,
            rag_unavailable=not rag.rag_available,
            progression_note=progression_note,
            risk_computed_at=datetime.now(timezone.utc),
        )
