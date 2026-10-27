"""
src/recommendation/fallback.py
==============================
Deterministic fallback recommendation builder (Stage C6).

Constructs complete, structured, and compliant RecommendationOutput objects
directly from RAGOutput, RiskOutput, WeatherOutput, and VisionOutput without LLM calls.

Used when:
  - LLM is disabled or unavailable.
  - API key is missing.
  - LLM times out or returns provider error.
  - LLM output fails grounding validation (hallucinated dose, number, or product).
  - RAG knowledge base chapter is not found.
"""
from __future__ import annotations

import re
from typing import List, Optional

from src.rag.agent import scrub_doses_and_chemicals
from src.recommendation.prompt import format_top_candidates
from src.schemas.schemas import Citation, RAGOutput, RecommendationOutput, RiskOutput, VisionOutput, WeatherOutput


def map_urgency(
    risk: RiskOutput,
    rag: RAGOutput,
    vision: Optional[VisionOutput],
) -> str:
    """
    Map urgency level per spec Part 4b-iii:
      - RAG unavailable or found=False -> consult_agronomist (unless healthy)
      - can_diagnose=False -> consult_agronomist
      - escalate_to_human=True -> consult_agronomist
      - severe + no escalate -> within_24h
      - moderate + no escalate -> within_48h
      - low + no escalate -> monitor
    """
    top_label = vision.top_label if vision else (rag.model_class or "")
    is_healthy = "healthy" in top_label.lower()

    if vision and not vision.can_diagnose:
        return "consult_agronomist"
    if risk.escalate_to_human or (vision and vision.escalate_to_human):
        return "consult_agronomist"
    if is_healthy:
        return "monitor"
    if not rag.found or risk.rag_unavailable:
        return "consult_agronomist"

    r_level = risk.risk_level.lower()
    if r_level == "severe":
        return "within_24h"
    elif r_level == "moderate":
        return "within_48h"
    elif r_level == "low":
        return "monitor"
    else:
        return "consult_agronomist"


def format_citations(citations: List) -> List[str]:
    """Convert RAG citations to string representation."""
    formatted = []
    for c in citations:
        if isinstance(c, Citation):
            s = f"{c.tag}: {c.title}"
            if c.url:
                s += f" ({c.url})"
            formatted.append(s)
        else:
            formatted.append(str(c))
    return formatted


def build_fallback_recommendation(
    vision: Optional[VisionOutput],
    rag: RAGOutput,
    weather: Optional[WeatherOutput],
    risk: RiskOutput,
    fallback_reason: str,
    include_doses: bool = False,
) -> RecommendationOutput:
    """
    Build deterministic recommendation output from structured inputs.

    Args:
        vision: Vision agent output.
        rag: RAG agent output.
        weather: Weather agent output.
        risk: Risk agent output.
        fallback_reason: Explanation of why fallback was used.
        include_doses: Whether chemical doses are permitted.

    Returns:
        RecommendationOutput instance with llm_used=False.
    """
    top_label = vision.top_label if vision else (rag.model_class or "Unknown")
    is_healthy = "healthy" in top_label.lower() or (rag.reason == "healthy_class")
    is_gap_row = any("GAP" in f.tag or "GAP" in f.value or "GAP" in f.source for f in risk.factors)
    escalate = risk.escalate_to_human or (vision and (vision.escalate_to_human or not vision.can_diagnose))

    # Normalize fallback reason for healthy class
    if (is_healthy or rag.reason == "healthy_class") and fallback_reason == "rag_not_found":
        fallback_reason = "healthy_class_deterministic"

    # 1. Map urgency
    urgency = map_urgency(risk=risk, rag=rag, vision=vision)

    # 2. Build flags
    flags: List[str] = []
    if is_gap_row:
        flags.append("PROVISIONAL_SEVERITY_GAP_ROW")
    if weather and weather.weather_available:
        flags.append("WEATHER_UNVERIFIED_I14")
    if is_healthy or rag.reason == "healthy_class":
        flags.append("HEALTHY_CLASS")
    elif not rag.found:
        flags.append("RAG_CHAPTER_NOT_FOUND")
    if escalate:
        flags.append("ESCALATE_TO_HUMAN")
    if not include_doses:
        flags.append("DOSES_HIDDEN")

    # 3. Build caveats
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
    if not rag.found and not is_healthy and rag.reason != "healthy_class":
        caveats.append(
            "Knowledge base chapter was not found for this class. Automatic treatment advice unavailable."
        )

    # 4. Weather note (always marked unverified)
    if weather and weather.weather_available:
        triggered_names = [r.condition_name for r in weather.rules_triggered if r.triggered]
        if triggered_names:
            weather_note = (
                f"[UNVERIFIED — I14] Weather conditions ({', '.join(triggered_names)}) "
                "may favour disease spread. Monitor field conditions closely."
            )
        else:
            weather_note = "[UNVERIFIED — I14] No high-risk weather triggers currently active."
    else:
        weather_note = (
            "[UNVERIFIED — I14] Weather data was unavailable; environmental disease risk could not be assessed."
        )

    # 5. Uncertainty note (D10, I40, I42)
    uncertainty_note: Optional[str] = None
    if is_healthy:
        uncertainty_note = (
            "Note: Visual classification models can occasionally miss diseased leaves or early-stage symptoms (I40). "
            "Continue regular crop scouting and monitoring."
        )
    elif escalate:
        candidates_str = format_top_candidates(vision)
        uncertainty_note = (
            f"Diagnosis is uncertain. Top-3 candidates: {candidates_str}. "
            "Advise confirming with a local agronomist or extension officer before taking action."
        )

    # 6. What to do first
    what_to_do_first: Optional[str] = None
    if is_healthy:
        what_to_do_first = "Continue routine crop scouting and monitoring."
    elif not rag.found:
        what_to_do_first = "Consult a local agronomist or agricultural extension officer."
    elif rag.what_to_do_first:
        text = rag.what_to_do_first
        if not include_doses:
            text = scrub_doses_and_chemicals(text)
        if escalate and text and not text.lower().startswith("advisory only"):
            what_to_do_first = f"ADVISORY ONLY — consult an agronomist to confirm: {text}"
        else:
            what_to_do_first = text
    else:
        what_to_do_first = "Consult a local agronomist or agricultural extension officer."

    # 7. Steps
    steps: List[str] = []
    if is_healthy:
        steps = [
            "Continue routine crop monitoring.",
            "Submit a follow-up image if discolouration, spots, or lesions develop.",
        ]
    elif not rag.found:
        steps = [
            "Consult a local agronomist or agricultural extension officer.",
            "Bring a physical sample of the affected leaf to your local extension office.",
            "Avoid applying unverified chemical treatments before expert confirmation.",
        ]
    else:
        # Assemble from RAG treatment options or preventive measures
        raw_candidates = list(rag.treatment_options) + list(rag.preventive_measures)
        if not raw_candidates and rag.prevention:
            # Split prevention paragraphs or bullets
            for line in rag.prevention.split("\n"):
                clean = line.strip(" •-*")
                if len(clean) > 15:
                    raw_candidates.append(clean)

        for item in raw_candidates:
            clean_item = item.strip()
            if not include_doses:
                clean_item = scrub_doses_and_chemicals(clean_item)
            if not clean_item or len(clean_item) < 10:
                continue
            if escalate and not clean_item.lower().startswith("advisory only"):
                clean_item = f"ADVISORY ONLY — consult an agronomist to confirm: {clean_item}"
            if clean_item not in steps:
                steps.append(clean_item)
            if len(steps) >= 5:
                break

        if not steps and what_to_do_first:
            steps.append(what_to_do_first)

    # 8. Summary
    disease_name = rag.disease_name or rag.model_class or top_label
    if is_healthy:
        summary = "No disease detected in this image. Continue routine monitoring."
    elif not rag.found:
        summary = f"No knowledge base advisory found for {disease_name}. Consult an agronomist or extension officer."
    elif escalate:
        summary = f"Diagnosis is uncertain for {disease_name}. ADVISORY ONLY — consult an agronomist to confirm."
    else:
        summary = f"Identified {disease_name} with {risk.risk_level} risk. Follow immediate cultural management and recommended steps."

    # 9. Citations
    citations = format_citations(rag.citations) if rag.found else []

    # 10. Assemble RecommendationOutput
    return RecommendationOutput(
        summary=summary,
        urgency=urgency,
        what_to_do_first=what_to_do_first,
        steps=steps,
        weather_note=weather_note,
        uncertainty_note=uncertainty_note,
        citations=citations,
        flags=flags,
        llm_used=False,
        fallback_reason=fallback_reason,
        actions=list(steps),
        risk_level=risk.risk_level,
        confidence_justification=(
            f"Risk assessed as {risk.risk_level} (score {risk.risk_score:.2f}). "
            f"Uncertainty: {risk.uncertainty_level}."
        ),
        caveats=caveats,
        rag_grounded=bool(rag.found and citations),
        escalate_to_human=escalate,
        knowledge_gap_caveated=bool(
            rag.kb_gap or (not rag.found and not is_healthy and rag.reason != "healthy_class")
        ),
    )
