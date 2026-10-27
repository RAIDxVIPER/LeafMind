"""
src/recommendation/prompt.py
============================
Prompt construction for Recommendation Agent (Stage C6).

Follows strict grounding rules from spec Part 4b:
  - Rewording and organisation only; no invented facts, chemicals, or numbers.
  - Uncertainty and escalation handling (D10, I40, I42): tentative phrasing and agronomist consultation.
  - Dose hiding (D23, D28): doses and chemical rates are completely excluded from prompt when include_doses=False.
  - Weather marked unverified (I14).
  - GAP threat rows marked provisional (I19).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.rag.agent import scrub_doses_and_chemicals
from src.schemas.schemas import RAGOutput, RiskOutput, VisionOutput, WeatherOutput

SYSTEM_PROMPT = """You are a plant disease advisory assistant for farmers. Your task is to convert a structured disease risk assessment and retrieved knowledge base evidence into farmer-actionable advice.

STRICT GROUNDING RULES:
1. You must ONLY reword and organize the text provided in the INPUT EVIDENCE sections.
2. Do NOT invent, extrapolate, or add ANY new facts, diseases, chemicals, product names, doses, active ingredients, numbers, prices, or treatment steps that are NOT in the input.
3. Every single number and chemical product name in your output MUST appear verbatim in the input text.
4. Citations are attached automatically by the system; do NOT fabricate citations.
5. If the diagnosis is uncertain (escalate_to_human is true or can_diagnose is false), state clearly that the diagnosis is uncertain, list the top-3 candidates, and advise confirming with an agronomist or extension officer before taking action. Do NOT present a single confident diagnosis.
6. If the leaf is identified as healthy, include a note that visual classification models can occasionally miss diseased leaves or early symptoms.
7. Any weather advice MUST carry the exact tag '[UNVERIFIED — I14]'.
8. If the risk assessment indicates a provisional GAP row, state that the severity rating is provisional pending plant pathologist verification (I19).
9. Output ONLY a valid JSON object matching the following structure:
{
  "summary": "<concise advisory summary for farmer>",
  "what_to_do_first": "<single immediate action to take>",
  "steps": ["<actionable step 1>", "<actionable step 2>", ...],
  "weather_note": "<weather advisory context with [UNVERIFIED — I14], or null>",
  "uncertainty_note": "<explanation of diagnostic uncertainty, top-3 candidates, or healthy disclaimer if applicable, else null>"
}
Do NOT include markdown backticks or any prose outside the JSON object."""


def format_top_candidates(vision: Optional[VisionOutput]) -> str:
    """Format top-3 candidates with calibrated probabilities."""
    if not vision:
        return "None provided"

    candidates = vision.top_3_candidates or vision.top_k_probs[:3]
    parts = []
    for c in candidates:
        parts.append(f"{c.label} ({c.prob * 100:.1f}%)")
    return ", ".join(parts)


def build_recommendation_prompt(
    vision: Optional[VisionOutput],
    rag: RAGOutput,
    weather: Optional[WeatherOutput],
    risk: RiskOutput,
    include_doses: bool = False,
) -> Dict[str, str]:
    """
    Construct the system prompt and user prompt for the recommendation LLM.

    Args:
        vision: Optional VisionOutput from vision agent.
        rag: RAGOutput from RAG agent.
        weather: Optional WeatherOutput from weather agent.
        risk: RiskOutput from risk agent.
        include_doses: Whether chemical doses are permitted (D23/D28).

    Returns:
        Dict with "system_prompt" and "prompt".
    """
    lines: List[str] = []

    # 1. Vision & Diagnostic Context
    lines.append("=== INPUT SECTION 1: VISION & DIAGNOSIS ===")
    if vision:
        lines.append(f"Primary Detection: {vision.top_label}")
        lines.append(f"Confidence: {vision.confidence:.2f}")
        lines.append(f"Top 3 Candidates: {format_top_candidates(vision)}")
        lines.append(f"Can Diagnose: {vision.can_diagnose}")
        lines.append(f"Vision Escalation Required: {vision.escalate_to_human}")
        if vision.image_quality_flags.blurry:
            lines.append("Image Quality Flag: Blurry image detected")
        if vision.image_quality_flags.low_resolution:
            lines.append("Image Quality Flag: Low resolution image detected")
        if vision.ood_flag:
            lines.append("OOD Flag: Out-of-distribution leaf image detected")
    else:
        lines.append(f"Disease Class: {rag.model_class or 'Unknown'}")

    # 2. Risk Assessment Context
    lines.append("\n=== INPUT SECTION 2: RISK ASSESSMENT ===")
    lines.append(f"Risk Level: {risk.risk_level}")
    lines.append(f"Risk Score: {risk.risk_score:.2f}")
    lines.append(f"Escalate to Human: {risk.escalate_to_human}")
    lines.append(f"Uncertainty Level: {risk.uncertainty_level}")
    if risk.uncertainty_signals_triggered:
        lines.append(f"Triggered Uncertainty Signals: {', '.join(risk.uncertainty_signals_triggered)}")

    is_gap_row = any("GAP" in f.tag or "GAP" in f.value or "GAP" in f.source for f in risk.factors)
    if is_gap_row:
        lines.append("GAP Alert: Risk rating is derived from a provisional GAP row pending pathologist sign-off [I19]")

    # 3. Weather Context
    lines.append("\n=== INPUT SECTION 3: WEATHER CONTEXT ===")
    if weather and weather.weather_available:
        lines.append(f"Weather Score: {weather.weather_score:.2f}")
        active_rules = [r.condition_name for r in weather.rules_triggered if r.triggered]
        if active_rules:
            lines.append(f"Active Environmental Triggers: {', '.join(active_rules)}")
        else:
            lines.append("Active Environmental Triggers: None")
        lines.append("Note: Weather thresholds are unverified by a pathologist [UNVERIFIED — I14]")
    else:
        lines.append("Weather Data: Unavailable [UNVERIFIED — I14]")

    # 4. RAG Knowledge Base Evidence
    lines.append("\n=== INPUT SECTION 4: KNOWLEDGE BASE EVIDENCE ===")
    lines.append(f"Knowledge Found: {rag.found}")
    lines.append(f"Disease Name: {rag.disease_name or rag.model_class}")

    if not rag.found:
        lines.append(f"Reason: {rag.reason or 'Knowledge base chapter not found for this class.'}")
        lines.append("No treatment advice available in knowledge base. Must advise agronomist consultation.")
    else:
        # Scrub narrative fields if include_doses=False (D28)
        what_it_is = rag.what_it_is
        why_it_happens = rag.why_it_happens
        what_to_do_first = rag.what_to_do_first
        prevention = rag.prevention
        treatment_options = list(rag.treatment_options)
        preventive_measures = list(rag.preventive_measures)

        if not include_doses:
            what_it_is = scrub_doses_and_chemicals(what_it_is)
            why_it_happens = scrub_doses_and_chemicals(why_it_happens)
            what_to_do_first = scrub_doses_and_chemicals(what_to_do_first)
            prevention = scrub_doses_and_chemicals(prevention)
            treatment_options = [
                s for s in (scrub_doses_and_chemicals(t) for t in treatment_options) if s
            ]
            preventive_measures = [
                s for s in (scrub_doses_and_chemicals(p) for p in preventive_measures) if s
            ]

        if what_it_is:
            lines.append(f"Identification: {what_it_is}")
        if why_it_happens:
            lines.append(f"Cause & Spread: {why_it_happens}")
        if what_to_do_first:
            lines.append(f"Immediate Action: {what_to_do_first}")
        if prevention:
            lines.append(f"Cultural & Preventive Management: {prevention}")
        if treatment_options:
            lines.append(f"Treatment Options: {'; '.join(treatment_options)}")
        if preventive_measures:
            lines.append(f"Preventive Measures: {'; '.join(preventive_measures)}")

        # D23: Only include chemical options if include_doses=True
        if include_doses and rag.chemical_options:
            lines.append("Chemical Options:")
            for opt in rag.chemical_options:
                lines.append(f"  - Product: {opt.product}, Dose: {opt.dose or 'N/A'}, Tag: {opt.verification_tag}")

    user_prompt = "\n".join(lines)
    return {
        "system_prompt": SYSTEM_PROMPT,
        "prompt": user_prompt,
    }
