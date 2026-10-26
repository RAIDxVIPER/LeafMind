"""
scripts/smoke_risk.py
=====================
Smoke-test demo for the Risk Agent (Stage C5).

Runs 4 cases:
  1. Tomato___Late_blight (disease, weather available, no history)
  2. A healthy class (Tomato___healthy)
  3. Squash___Powdery_mildew (no KB chapter — expected GAP)
  4. Case with weather unavailable

Output: plain-text summary. No JSON dump, no API keys, no LLM calls.

Usage:
    python scripts/smoke_risk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datetime import datetime, timezone

from src.agents.risk_agent import RiskAgent
from src.schemas.schemas import (
    ClassProb,
    ImageQualityFlags,
    RAGOutput,
    VisionOutput,
    WeatherOutput,
)


def make_vision(
    top_label: str,
    confidence: float,
    top2_prob: float = 0.05,
    blurry: bool = False,
    ood_flag: bool = False,
) -> VisionOutput:
    return VisionOutput(
        top_label=top_label,
        confidence=confidence,
        top_k_probs=[
            ClassProb(label=top_label, prob=confidence),
            ClassProb(label="Other___class", prob=top2_prob),
        ],
        image_quality_flags=ImageQualityFlags(blurry=blurry),
        ood_flag=ood_flag,
        model_version="efficientnet_b3_v1",
        temperature=1.0,
    )


def make_weather(available: bool = True, score: float = 0.5) -> WeatherOutput:
    return WeatherOutput(
        weather_available=available,
        weather_score=score,
        unavailable_reason="API outage during smoke test" if not available else None,
    )


def make_rag(found: bool = True, kb_gap: bool = False) -> RAGOutput:
    return RAGOutput(
        found=found,
        kb_gap=kb_gap,
        knowledge_gap_flag=kb_gap,
        rag_available=True,
    )


def print_result(case_name: str, result) -> None:
    print(f"\n{'='*60}")
    print(f"Case: {case_name}")
    print(f"{'='*60}")
    print(f"  risk_score      : {result.risk_score:.4f}")
    print(f"  risk_level      : {result.risk_level}")
    print(f"  escalate_to_human: {result.escalate_to_human}")
    print(f"  uncertainty_level: {result.uncertainty_level}")
    print(f"  uncertainty_signals: {result.uncertainty_signals_triggered}")
    print(f"  weather_unavailable: {result.weather_unavailable}")
    print(f"  rag_unavailable  : {result.rag_unavailable}")
    print(f"  Factors:")
    for f in result.factors:
        print(f"    [{f.source}] contribution={f.contribution:.4f}")
        print(f"      value: {f.value[:100]}")
        print(f"      tag:   {f.tag[:80]}")
    print(f"  Progression note (first 120 chars):")
    print(f"    {result.progression_note[:120]}")


def main() -> None:
    print("LeafMind Risk Agent — Smoke Test (Stage C5)")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    agent = RiskAgent()
    print(f"\nWeights loaded from config: {agent.weights}")

    # ── Case 1: Tomato Late blight, weather available ──────────────────────────
    vision1 = make_vision("Tomato___Late_blight", confidence=0.88, top2_prob=0.08)
    weather1 = make_weather(available=True, score=0.5)   # partial weather score
    rag1 = make_rag(found=True)
    result1 = agent.score(vision1, weather1, rag1, monitoring_disease_status=None)
    print_result("Tomato___Late_blight (weather available, first visit)", result1)

    # ── Case 2: Healthy class ──────────────────────────────────────────────────
    vision2 = make_vision("Tomato___healthy", confidence=0.91)
    weather2 = make_weather(available=True, score=0.0)
    rag2 = make_rag(found=False)
    result2 = agent.score(vision2, weather2, rag2)
    print_result("Tomato___healthy (expected: low risk, no escalation)", result2)

    # ── Case 3: Squash___Powdery_mildew (no KB chapter) ───────────────────────
    vision3 = make_vision("Squash___Powdery_mildew", confidence=0.75)
    weather3 = make_weather(available=True, score=0.0)
    rag3 = make_rag(found=False, kb_gap=True)
    result3 = agent.score(vision3, weather3, rag3)
    print_result("Squash___Powdery_mildew (no KB chapter, GAP_FALLBACK)", result3)

    # ── Case 4: Weather unavailable ───────────────────────────────────────────
    vision4 = make_vision("Tomato___Early_blight", confidence=0.82)
    weather4 = make_weather(available=False)
    rag4 = make_rag(found=True)
    result4 = agent.score(vision4, weather4, rag4)
    print_result("Tomato___Early_blight (weather UNAVAILABLE)", result4)

    print(f"\n{'='*60}")
    print("Smoke test complete — all 4 cases produced RiskOutput without crash.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
