"""
src/agents/risk_agent.py
========================
Risk Agent for LeafMind (Stage C5).

Consumes outputs from:
  - src/agents/vision_agent.py → VisionOutput
  - src/agents/weather_agent.py → WeatherOutput
  - src/rag/agent.py → RAGOutput

Produces: RiskOutput

Key design principles (C5):
- D10: Risk ≠ Uncertainty. Vision confidence is a SEPARATE output field.
  It never folds into the risk_score R.
- D11: R = 0.70*threat + 0.20*weather + 0.10*history. Weights from config.
- D13: No LLM is used anywhere in C5.
- D15: GAP fallback (0.50) when threat is null/GAP; flag threat_source="GAP_FALLBACK";
  cap risk_label at "moderate" (I26).
- Healthy classes: threat_score=0.0, label="low"/"none", no threat lookup.
- Weather unavailable: neutral 0.0 with flag. Never crashes.
- History absent (first visit): 0.0 with flag. Never crashes.
- Squash___Powdery_mildew: no KB chapter, but still works via GAP_FALLBACK.

Protected interfaces (DO NOT MODIFY):
  - src/agents/vision_agent.py → VisionOutput
  - src/rag/agent.py → RAGOutput (found, reason fields)
  - src/agents/weather_agent.py → WeatherOutput (rule hits W01–W08, weather_score)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.risk.scorer import RiskScorer
from src.risk.threat_table import ThreatTable
from src.schemas.schemas import RAGOutput, RiskOutput, VisionOutput, WeatherOutput

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class RiskAgent:
    """
    Risk Agent — deterministic composite risk scoring (C5).

    Takes outputs from Vision, Weather, and RAG agents and produces a
    structured RiskOutput with risk_score, risk_label, component breakdown,
    flags, and separate uncertainty fields (D10).

    No LLM, no external calls. All logic is deterministic rule-based. [ENG]
    Weights and thresholds are read from config/risk_config.yaml at init time.

    Usage:
        agent = RiskAgent()
        risk_output = agent.score(vision_output, weather_output, rag_output)

        # On a follow-up visit (to incorporate monitoring history):
        risk_output = agent.score(
            vision_output, weather_output, rag_output,
            monitoring_disease_status="same_disease_still_detected"
        )
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        threat_table_path: Optional[Path] = None,
    ) -> None:
        """
        Initialise the Risk Agent.

        Args:
            config_path: Optional override for config/risk_config.yaml path.
            threat_table_path: Optional override for config/threat_table.json path.
        """
        threat_table = ThreatTable(table_path=threat_table_path)
        self._scorer = RiskScorer(
            threat_table=threat_table,
            config_path=config_path,
        )
        logger.info(
            "RiskAgent initialised with weights: threat=%.2f, weather=%.2f, history=%.2f",
            self._scorer.weights["threat"],
            self._scorer.weights["weather"],
            self._scorer.weights["history"],
        )

    @property
    def weights(self) -> dict[str, float]:
        """Expose config-driven weights for verification (tests, smoke scripts)."""
        return self._scorer.weights

    def score(
        self,
        vision: VisionOutput,
        weather: WeatherOutput,
        rag: RAGOutput,
        monitoring_disease_status: Optional[str] = None,
    ) -> RiskOutput:
        """
        Compute full RiskOutput from agent outputs.

        This is the primary entry point for the Risk Agent.

        Args:
            vision: VisionOutput from vision_agent.py (DO NOT MODIFY interface)
            weather: WeatherOutput from weather_agent.py (DO NOT MODIFY interface)
            rag: RAGOutput from rag/agent.py (found, reason, etc.)
            monitoring_disease_status: Optional status string from MonitoringOutput
                on follow-up visits (e.g. "same_disease_still_detected").
                Pass None on the first visit.

        Returns:
            RiskOutput with:
                - risk_score: R = 0.70*threat + 0.20*weather + 0.10*history
                - risk_level: "low" | "moderate" | "severe"
                - factors: list of RiskFactor (one per component)
                - escalate_to_human: from uncertainty signals U1–U6
                - uncertainty_level: "high_confidence" | "moderate_confidence" | "low_confidence"
                - uncertainty_signals_triggered: list of triggered signal names (e.g. "U1_low_confidence")
                  NOTE: These do NOT affect risk_score (D10).
                - weather_unavailable: True when weather data was unavailable
                - rag_unavailable: True when RAG lookup failed
                - progression_note: plain-language heuristic note
                - risk_computed_at: UTC timestamp

        Design notes:
            D10: vision.confidence is exposed via uncertainty_level and
                 uncertainty_signals_triggered. It is NEVER used to compute risk_score.
                 A confident diagnosis of a serious disease → high risk.
                 A blurry low-confidence image → escalate, not inflate risk.
        """
        try:
            result = self._scorer.score(
                vision=vision,
                weather=weather,
                rag=rag,
                monitoring_disease_status=monitoring_disease_status,
            )
            logger.debug(
                "RiskAgent.score: top_label=%r risk_score=%.4f risk_level=%s "
                "threat_source=%s weather_unavailable=%s history_null=%s",
                vision.top_label,
                result.risk_score,
                result.risk_level,
                next(
                    (f.value for f in result.factors if f.source == "threat_table"), "??"
                ),
                result.weather_unavailable,
                monitoring_disease_status is None,
            )
            return result
        except Exception:
            logger.exception(
                "RiskAgent.score raised an unexpected exception for top_label=%r",
                vision.top_label,
            )
            raise
