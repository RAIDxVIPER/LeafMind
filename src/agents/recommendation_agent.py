"""
src/agents/recommendation_agent.py
==================================
Recommendation Agent for LeafMind (Stage C6).

Consumes outputs from:
  - VisionAgent -> VisionOutput
  - RAGAgent -> RAGOutput
  - WeatherAgent -> WeatherOutput
  - RiskAgent -> RiskOutput

Produces:
  - RecommendationOutput

Key architectural principles (Stage C6):
  - D10: Risk != Uncertainty. Handles uncertainty and escalation separately.
  - D13: LLM is used only to reword and organize RAG/Risk/Weather text.
  - D19: RAG retrieval is extractive; LLM does NOT retrieve or invent disease advice.
  - D23 / D28: Chemical doses remain hidden when include_doses=False.
  - Grounding validation: LLM output is strictly validated against input text.
    If any ungrounded number, chemical, or dose appears, the agent rejects
    the output and falls back to a deterministic template.
  - Fallback guarantee: Never crashes. Missing API key, timeout, provider error,
    empty output, or validation failure all cleanly trigger deterministic fallback
    with llm_used=False and a specific fallback_reason.
  - Provisonal threat note (I19): Risk derived from GAP rows is tagged provisional.
  - Unverified weather (I14): Weather advice is explicitly marked [UNVERIFIED — I14].
  - Uncertainty / Escalation (I40, I42): Low confidence, can_diagnose=False, or
    escalate_to_human=True triggers differential top-3 display and agronomist referral.
    Healthy leaves carry a note that diseased leaves can sometimes be missed.
  - Secret protection: API keys are never printed, logged, or included in errors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.rag.agent import scrub_doses_and_chemicals
from src.recommendation.fallback import (
    build_fallback_recommendation,
    format_citations,
    map_urgency,
)
from src.recommendation.prompt import (
    build_recommendation_prompt,
    format_top_candidates,
)
from src.recommendation.provider import (
    LLMProvider,
    get_provider,
    sanitize_key,
)
from src.recommendation.validation import validate_grounding
from src.schemas.schemas import (
    RAGOutput,
    RecommendationInput,
    RecommendationOutput,
    RiskOutput,
    VisionOutput,
    WeatherOutput,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "recommendation_config.yaml"


class RecommendationAgent:
    """
    Recommendation Agent (Stage C6).

    Synthesizes farmer advisory recommendations using an LLM provider (Gemini/Ollama)
    constrained strictly to extractive RAG and risk/weather inputs, with a guaranteed
    deterministic fallback when LLM is unavailable or fails grounding validation.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        """
        Initialise RecommendationAgent.

        Args:
            config_path: Optional path to recommendation_config.yaml.
            provider: Optional pre-configured LLMProvider instance (e.g. for testing).
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config: Dict[str, Any] = {}
        if self.config_path.is_file():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Could not load recommendation config from %s: %s", self.config_path, e)

        self.provider: Optional[LLMProvider] = provider
        self.init_error: Optional[str] = None

        if self.provider is None:
            try:
                self.provider = get_provider(config=self.config)
            except Exception as e:
                # Capture failure reason (e.g. missing API key, unknown provider) safely
                sanitized = sanitize_key(str(e))
                self.init_error = sanitized
                logger.info("LLM provider initialization deferred/fallback: %s", sanitized)

    def recommend(
        self,
        vision: Optional[VisionOutput] = None,
        rag: Optional[RAGOutput] = None,
        weather: Optional[WeatherOutput] = None,
        risk: Optional[RiskOutput] = None,
        include_doses: bool = False,
        input_data: Optional[RecommendationInput] = None,
    ) -> RecommendationOutput:
        """
        Generate grounded recommendation output.

        Can be called with individual outputs or a single RecommendationInput object.

        Args:
            vision: Vision agent output.
            rag: RAG agent output.
            weather: Weather agent output.
            risk: Risk agent output.
            include_doses: Whether chemical doses are permitted (D23/D28).
            input_data: Optional unified RecommendationInput instance.

        Returns:
            Structured RecommendationOutput.
        """
        # Unpack input_data if provided
        if input_data is not None:
            risk = risk or input_data.risk
            rag = rag or input_data.rag
            vision = vision or input_data.vision
            weather = weather or input_data.weather
            include_doses = include_doses or input_data.include_doses

        if risk is None or rag is None:
            raise ValueError("RecommendationAgent requires at least 'risk' and 'rag' outputs.")

        top_label = vision.top_label if vision else (rag.model_class or "Unknown")
        is_healthy = "healthy" in top_label.lower()
        is_gap_row = any("GAP" in f.tag or "GAP" in f.value or "GAP" in f.source for f in risk.factors)
        escalate = risk.escalate_to_human or (vision and (vision.escalate_to_human or not vision.can_diagnose))

        # 1. RAG not found path -> immediate deterministic fallback (spec 4b)
        if not rag.found:
            return build_fallback_recommendation(
                vision=vision,
                rag=rag,
                weather=weather,
                risk=risk,
                fallback_reason="rag_not_found",
                include_doses=include_doses,
            )

        # 2. Check provider availability
        if self.provider is None:
            reason = self.init_error or "llm_provider_unavailable"
            if "not set" in reason.lower() or "missing" in reason.lower():
                fallback_reason = "missing_api_key"
            elif "unsupported" in reason.lower():
                fallback_reason = reason
            else:
                fallback_reason = "llm_provider_unavailable"
            return build_fallback_recommendation(
                vision=vision,
                rag=rag,
                weather=weather,
                risk=risk,
                fallback_reason=fallback_reason,
                include_doses=include_doses,
            )

        # 3. Build prompts
        prompt_dict = build_recommendation_prompt(
            vision=vision,
            rag=rag,
            weather=weather,
            risk=risk,
            include_doses=include_doses,
        )

        # 4. Invoke LLM provider with safety and timeout guards
        try:
            raw_response = self.provider.generate(
                prompt=prompt_dict["prompt"],
                system_prompt=prompt_dict["system_prompt"],
            )
        except TimeoutError:
            return build_fallback_recommendation(
                vision=vision,
                rag=rag,
                weather=weather,
                risk=risk,
                fallback_reason="timeout",
                include_doses=include_doses,
            )
        except Exception as e:
            sanitized_err = sanitize_key(str(e))
            return build_fallback_recommendation(
                vision=vision,
                rag=rag,
                weather=weather,
                risk=risk,
                fallback_reason=f"provider_error: {sanitized_err}",
                include_doses=include_doses,
            )

        # 5. Output Grounding Validation
        is_valid, fail_reason, parsed = validate_grounding(
            raw_response=raw_response,
            input_prompt=prompt_dict["prompt"],
            include_doses=include_doses,
        )

        if not is_valid or parsed is None:
            return build_fallback_recommendation(
                vision=vision,
                rag=rag,
                weather=weather,
                risk=risk,
                fallback_reason=f"validation_failed: {fail_reason}",
                include_doses=include_doses,
            )

        # 6. Post-process validated LLM output
        summary = parsed["summary"]
        what_to_do_first = parsed["what_to_do_first"]
        steps = parsed["steps"]
        weather_note = parsed["weather_note"]
        uncertainty_note = parsed["uncertainty_note"]

        # Safety net: Re-scrub narrative fields if include_doses=False (D28)
        if not include_doses:
            summary = scrub_doses_and_chemicals(summary) or summary
            if what_to_do_first:
                what_to_do_first = scrub_doses_and_chemicals(what_to_do_first)
            steps = [s for s in (scrub_doses_and_chemicals(step) for step in steps) if s]

        # Deterministic urgency mapping per spec 4b-iii
        urgency = map_urgency(risk=risk, rag=rag, vision=vision)

        # Pass-through citations from RAG
        citations = format_citations(rag.citations)

        # Build flags
        flags: List[str] = []
        if is_gap_row:
            flags.append("PROVISIONAL_SEVERITY_GAP_ROW")
        if weather and weather.weather_available:
            flags.append("WEATHER_UNVERIFIED_I14")
        if not rag.found:
            flags.append("RAG_CHAPTER_NOT_FOUND")
        if escalate:
            flags.append("ESCALATE_TO_HUMAN")
        if not include_doses:
            flags.append("DOSES_SCRUBBED")

        # Assemble caveats
        caveats: List[str] = []
        if is_gap_row:
            caveats.append(
                "Threat severity rating is provisional pending plant pathologist verification (I19)."
            )
        if weather and weather.weather_available:
            caveats.append(
                "Weather disease-risk thresholds are unverified by a plant pathologist (I14)."
            )
        if is_healthy:
            caveats.append(
                "Visual classification models can occasionally miss diseased leaves or early symptoms (I40). Inspect other leaves closely."
            )
        if escalate:
            caveats.append(
                "Automated treatment advice is not confirmed for this case. Expert agronomist verification is required."
            )

        # Ensure weather note is marked unverified [I14]
        if weather and weather.weather_available:
            if weather_note:
                if "[UNVERIFIED — I14]" not in weather_note:
                    weather_note = f"[UNVERIFIED — I14] {weather_note}"
            else:
                weather_note = "[UNVERIFIED — I14] Weather conditions active. Local field verification recommended."
        else:
            weather_note = "[UNVERIFIED — I14] Weather data was unavailable; environmental disease risk could not be assessed."

        # Ensure uncertainty note covers required disclaimers (D10, I40, I42)
        if is_healthy:
            disclaimer = (
                "Note: Visual classification models can occasionally miss diseased leaves or early-stage symptoms (I40). "
                "Continue regular crop scouting and monitoring."
            )
            if not uncertainty_note or "miss" not in uncertainty_note.lower():
                uncertainty_note = disclaimer
        elif escalate:
            candidates_str = format_top_candidates(vision)
            uncertainty_msg = (
                f"Diagnosis is uncertain. Top-3 candidates: {candidates_str}. "
                "Advise confirming with a local agronomist or extension officer before taking action."
            )
            if not uncertainty_note or "uncertain" not in uncertainty_note.lower():
                uncertainty_note = uncertainty_msg

        return RecommendationOutput(
            summary=summary,
            urgency=urgency,
            what_to_do_first=what_to_do_first,
            steps=steps,
            weather_note=weather_note,
            uncertainty_note=uncertainty_note,
            citations=citations,
            flags=flags,
            llm_used=True,
            fallback_reason=None,
            actions=list(steps),
            risk_level=risk.risk_level,
            confidence_justification=(
                f"Risk assessed as {risk.risk_level} (score {risk.risk_score:.2f}). "
                f"Uncertainty: {risk.uncertainty_level}."
            ),
            caveats=caveats,
            rag_grounded=bool(citations),
            escalate_to_human=escalate,
            knowledge_gap_caveated=bool(rag.kb_gap or not rag.found),
        )
