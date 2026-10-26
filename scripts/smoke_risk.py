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
    print("LeafMind Risk Agent — Smoke Test (Stage C5, Approved Threat Table)")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    agent = RiskAgent()
    print(f"\nWeights loaded from config: {agent.weights}")

    # ── Case 1: Tomato___Late_blight (HIGH tier) ──────────────────────────────
    vision1 = make_vision("Tomato___Late_blight", confidence=0.88, top2_prob=0.08)
    weather1 = make_weather(available=True, score=0.5)
    rag1 = make_rag(found=True)
    result1 = agent.score(vision1, weather1, rag1, monitoring_disease_status=None)
    print_result("Tomato___Late_blight [HIGH tier] (weather=0.5, first visit)", result1)

    # ── Case 2: Tomato___Early_blight (MEDIUM tier) ───────────────────────────
    vision2 = make_vision("Tomato___Early_blight", confidence=0.85, top2_prob=0.05)
    weather2 = make_weather(available=True, score=0.5)
    rag2 = make_rag(found=True)
    result2 = agent.score(vision2, weather2, rag2, monitoring_disease_status=None)
    print_result("Tomato___Early_blight [MEDIUM tier] (weather=0.5, first visit)", result2)

    # ── Case 3: Tomato___Leaf_Mold (LOW tier) ─────────────────────────────────
    vision3 = make_vision("Tomato___Leaf_Mold", confidence=0.82, top2_prob=0.06)
    weather3 = make_weather(available=True, score=0.5)
    rag3 = make_rag(found=True)
    result3 = agent.score(vision3, weather3, rag3, monitoring_disease_status=None)
    print_result("Tomato___Leaf_Mold [LOW tier] (weather=0.5, first visit)", result3)

    # ── Case 4: Apple___Cedar_apple_rust (STAY GAP fallback) ─────────────────
    vision4 = make_vision("Apple___Cedar_apple_rust", confidence=0.80, top2_prob=0.10)
    weather4 = make_weather(available=True, score=0.5)
    rag4 = make_rag(found=True)
    result4 = agent.score(vision4, weather4, rag4, monitoring_disease_status=None)
    print_result("Apple___Cedar_apple_rust [STAY GAP] (weather=0.5, GAP_FALLBACK)", result4)

    # ── Case 5: Tomato___healthy (HEALTHY tier) ───────────────────────────────
    vision5 = make_vision("Tomato___healthy", confidence=0.91)
    weather5 = make_weather(available=True, score=0.0)
    rag5 = make_rag(found=False)
    result5 = agent.score(vision5, weather5, rag5, monitoring_disease_status=None)
    print_result("Tomato___healthy [HEALTHY tier] (weather=0.0, first visit)", result5)

    print(f"\n{'='*60}")
    print("Smoke test complete — all 5 cases produced RiskOutput without crash.")
    print(f"{'='*60}")



if __name__ == "__main__":
    main()
