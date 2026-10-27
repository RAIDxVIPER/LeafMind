"""
scripts/smoke_recommendation.py
===============================
Smoke script for Recommendation Agent (Stage C6).

Runs 5 canonical test cases:
  1. Tomato___Late_blight (High threat, KB found, weather available)
  2. Tomato___Leaf_Mold (Low threat, KB found)
  3. Apple___Cedar_apple_rust (GAP threat row -> provisional note I19)
  4. Tomato___healthy (Healthy leaf -> false-healthy disclaimer I40, urgency=monitor)
  5. Low-confidence / escalated case (Tomato___Early_blight, blurry, top-3 candidates, agronomist consultation)

Modes:
  (a) Fallback mode with no key — ALWAYS runs.
  (b) Live Gemini mode — runs ONLY if GEMINI_API_KEY is present in environment or local .env.

Security note: Never prints the key or request headers.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agents.recommendation_agent import RecommendationAgent
from src.recommendation.provider import GeminiProvider, load_local_env_safely
from src.schemas.schemas import (
    Citation,
    ClassProb,
    ImageQualityFlags,
    RAGOutput,
    RiskFactor,
    RiskOutput,
    RuleTriggerResult,
    VisionOutput,
    WeatherOutput,
)


def make_case_data(case_id: int):
    """Generate realistic inputs for each of the 5 canonical cases."""
    now = datetime.now(timezone.utc)

    if case_id == 1:
        # Case 1: Tomato___Late_blight (HIGH threat, confident)
        vision = VisionOutput(
            top_label="Tomato___Late_blight",
            confidence=0.89,
            top_k_probs=[
                ClassProb(label="Tomato___Late_blight", prob=0.89),
                ClassProb(label="Tomato___Early_blight", prob=0.07),
                ClassProb(label="Tomato___Leaf_Mold", prob=0.03),
            ],
            top_3_candidates=[
                ClassProb(label="Tomato___Late_blight", prob=0.89),
                ClassProb(label="Tomato___Early_blight", prob=0.07),
                ClassProb(label="Tomato___Leaf_Mold", prob=0.03),
            ],
            image_quality_flags=ImageQualityFlags(blurry=False),
            can_diagnose=True,
            escalate_to_human=False,
            model_version="efficientnet_b3_v1",
            temperature=1.0,
        )
        rag = RAGOutput(
            model_class="Tomato___Late_blight",
            disease_name="Tomato Late Blight",
            found=True,
            what_it_is="Late blight is caused by the oomycete Phytophthora infestans.",
            why_it_happens="Favoured by cool, moist conditions with high relative humidity.",
            what_to_do_first="Remove and destroy infected foliage immediately; avoid overhead irrigation.",
            prevention="Use certified disease-free seed, wider plant spacing, and drip irrigation.",
            treatment_options=["Apply protective copper fungicide sprays to healthy foliage."],
            preventive_measures=["Prune lower leaves to enhance canopy ventilation."],
            citations=[
                Citation(tag="T1", title="Tomato Diseases and Disorders Manual", url="https://example.org/tomato")
            ],
        )
        weather = WeatherOutput(
            weather_available=True,
            weather_score=0.8,
            rules_triggered=[
                RuleTriggerResult(
                    rule_id="W01",
                    condition_name="High humidity + cool nights (Late blight)",
                    triggered=True,
                    measured_values="RH 93%, Min Temp 16C",
                    threshold="RH >= 90% and Temp 15-20C",
                    time_window="past_24h",
                    threshold_source="Table 4b",
                )
            ],
        )
        risk = RiskOutput(
            risk_level="severe",
            risk_score=0.78,
            escalate_to_human=False,
            uncertainty_level="high_confidence",
            uncertainty_signals_triggered=[],
            factors=[
                RiskFactor(
                    source="threat_table",
                    value="HIGH threat (Tomato___Late_blight)",
                    contribution=0.595,
                    tag="[KB-derived draft, docs/review/threat_table_draft.md]",
                ),
                RiskFactor(
                    source="weather_W01",
                    value="triggered",
                    contribution=0.160,
                    tag="[ENG assumption — threshold UNVERIFIED I14]",
                ),
            ],
            progression_note="Potentially high-risk disease under favourable weather. Prompt action advisable.",
            risk_computed_at=now,
        )
        case_name = "Tomato___Late_blight (High threat, confident)"

    elif case_id == 2:
        # Case 2: Tomato___Leaf_Mold (LOW threat)
        vision = VisionOutput(
            top_label="Tomato___Leaf_Mold",
            confidence=0.84,
            top_k_probs=[
                ClassProb(label="Tomato___Leaf_Mold", prob=0.84),
                ClassProb(label="Tomato___Early_blight", prob=0.10),
                ClassProb(label="Tomato___healthy", prob=0.04),
            ],
            top_3_candidates=[
                ClassProb(label="Tomato___Leaf_Mold", prob=0.84),
                ClassProb(label="Tomato___Early_blight", prob=0.10),
                ClassProb(label="Tomato___healthy", prob=0.04),
            ],
            image_quality_flags=ImageQualityFlags(blurry=False),
            can_diagnose=True,
            escalate_to_human=False,
            model_version="efficientnet_b3_v1",
            temperature=1.0,
        )
        rag = RAGOutput(
            model_class="Tomato___Leaf_Mold",
            disease_name="Tomato Leaf Mold",
            found=True,
            what_it_is="Leaf mold caused by Passalora fulva manifests as pale green to yellow spots.",
            why_it_happens="Primarily develops in high relative humidity and warm greenhouse/canopy conditions.",
            what_to_do_first="Improve ventilation and reduce canopy humidity.",
            prevention="Circulate air through pruning and avoid wetting foliage during irrigation.",
            treatment_options=["Apply registered preventive fungicides if disease pressure is severe."],
            preventive_measures=["Space plants adequately and ventilate tunnels."],
            citations=[
                Citation(tag="T2", title="Tomato IPM Guide", url="https://example.org/tomato-ipm")
            ],
        )
        weather = WeatherOutput(
            weather_available=True,
            weather_score=0.2,
            rules_triggered=[],
        )
        risk = RiskOutput(
            risk_level="moderate",
            risk_score=0.38,
            escalate_to_human=False,
            uncertainty_level="high_confidence",
            uncertainty_signals_triggered=[],
            factors=[
                RiskFactor(
                    source="threat_table",
                    value="LOW threat (Tomato___Leaf_Mold)",
                    contribution=0.28,
                    tag="[KB-derived draft, docs/review/threat_table_draft.md]",
                )
            ],
            progression_note="Moderate risk assessed. Monitor field conditions.",
            risk_computed_at=now,
        )
        case_name = "Tomato___Leaf_Mold (Low threat)"

    elif case_id == 3:
        # Case 3: Apple___Cedar_apple_rust (GAP threat row)
        vision = VisionOutput(
            top_label="Apple___Cedar_apple_rust",
            confidence=0.86,
            top_k_probs=[
                ClassProb(label="Apple___Cedar_apple_rust", prob=0.86),
                ClassProb(label="Apple___Apple_scab", prob=0.09),
                ClassProb(label="Apple___healthy", prob=0.03),
            ],
            top_3_candidates=[
                ClassProb(label="Apple___Cedar_apple_rust", prob=0.86),
                ClassProb(label="Apple___Apple_scab", prob=0.09),
                ClassProb(label="Apple___healthy", prob=0.03),
            ],
            image_quality_flags=ImageQualityFlags(blurry=False),
            can_diagnose=True,
            escalate_to_human=False,
            model_version="efficientnet_b3_v1",
            temperature=1.0,
        )
        rag = RAGOutput(
            model_class="Apple___Cedar_apple_rust",
            disease_name="Cedar Apple Rust",
            found=True,
            what_it_is="Cedar apple rust caused by Gymnosporangium juniperi-virginianae produces yellow-orange leaf lesions.",
            why_it_happens="Requires alternate hosts (juniper/cedar trees) to complete life cycle.",
            what_to_do_first="Inspect nearby cedar trees for galls and prune affected branches.",
            prevention="Plant rust-resistant apple varieties and remove nearby junipers where feasible.",
            treatment_options=["Apply preventive protective sprays at pink bud stage."],
            preventive_measures=["Prune galls from alternate hosts in late winter."],
            citations=[
                Citation(tag="A1", title="Apple Disease Compendium", url="https://example.org/apple")
            ],
        )
        weather = WeatherOutput(
            weather_available=True,
            weather_score=0.4,
            rules_triggered=[],
        )
        risk = RiskOutput(
            risk_level="moderate",
            risk_score=0.45,
            escalate_to_human=False,
            uncertainty_level="high_confidence",
            uncertainty_signals_triggered=[],
            factors=[
                RiskFactor(
                    source="threat_table",
                    value="GAP fallback (DISEASE_DETECTED=0.50)",
                    contribution=0.35,
                    tag="[ENG assumption][GAP — threat table empty]",
                )
            ],
            progression_note="Disease detected. Severity tier provisional pending pathologist review.",
            risk_computed_at=now,
        )
        case_name = "Apple___Cedar_apple_rust (GAP threat row -> provisional note I19)"

    elif case_id == 4:
        # Case 4: Tomato___healthy (Healthy leaf)
        vision = VisionOutput(
            top_label="Tomato___healthy",
            confidence=0.96,
            top_k_probs=[
                ClassProb(label="Tomato___healthy", prob=0.96),
                ClassProb(label="Tomato___Early_blight", prob=0.03),
                ClassProb(label="Tomato___Late_blight", prob=0.01),
            ],
            top_3_candidates=[
                ClassProb(label="Tomato___healthy", prob=0.96),
                ClassProb(label="Tomato___Early_blight", prob=0.03),
                ClassProb(label="Tomato___Late_blight", prob=0.01),
            ],
            image_quality_flags=ImageQualityFlags(blurry=False),
            can_diagnose=True,
            escalate_to_human=False,
            model_version="efficientnet_b3_v1",
            temperature=1.0,
        )
        rag = RAGOutput(
            model_class="Tomato___healthy",
            disease_name="Healthy Tomato",
            found=False,
            reason="healthy_class",
            what_to_do_first=None,
        )
        weather = WeatherOutput(
            weather_available=True,
            weather_score=0.0,
            rules_triggered=[],
        )
        risk = RiskOutput(
            risk_level="low",
            risk_score=0.0,
            escalate_to_human=False,
            uncertainty_level="high_confidence",
            uncertainty_signals_triggered=[],
            factors=[
                RiskFactor(
                    source="threat_table",
                    value="HEALTHY leaf",
                    contribution=0.0,
                    tag="[ENG assumption]",
                )
            ],
            progression_note="No disease detected in this image. Continue routine monitoring. [ENG heuristic]",
            risk_computed_at=now,
        )
        case_name = "Tomato___healthy (Healthy leaf -> false-healthy disclaimer I40, urgency=monitor)"

    else:
        # Case 5: Low-confidence / escalated case
        vision = VisionOutput(
            top_label="Tomato___Early_blight",
            confidence=0.44,
            top_k_probs=[
                ClassProb(label="Tomato___Early_blight", prob=0.44),
                ClassProb(label="Tomato___Late_blight", prob=0.38),
                ClassProb(label="Tomato___Leaf_Mold", prob=0.14),
            ],
            top_3_candidates=[
                ClassProb(label="Tomato___Early_blight", prob=0.44),
                ClassProb(label="Tomato___Late_blight", prob=0.38),
                ClassProb(label="Tomato___Leaf_Mold", prob=0.14),
            ],
            image_quality_flags=ImageQualityFlags(blurry=True),
            can_diagnose=True,
            escalate_to_human=True,
            model_version="efficientnet_b3_v1",
            temperature=1.0,
        )
        rag = RAGOutput(
            model_class="Tomato___Early_blight",
            disease_name="Tomato Early Blight",
            found=True,
            what_it_is="Early blight caused by Alternaria solani produces concentric ring spots.",
            why_it_happens="Favoured by warm temperatures and alternating wet and dry periods.",
            what_to_do_first="Remove lower diseased leaves and avoid wetting leaves when watering.",
            prevention="Crop rotation, drip irrigation, and mulching around base of plants.",
            treatment_options=["Apply protective copper fungicide before symptoms become widespread."],
            preventive_measures=["Stake plants to keep foliage off the ground."],
            citations=[
                Citation(tag="T3", title="Tomato Early Blight Advisory", url="https://example.org/early-blight")
            ],
        )
        weather = WeatherOutput(
            weather_available=True,
            weather_score=0.4,
            rules_triggered=[],
        )
        risk = RiskOutput(
            risk_level="moderate",
            risk_score=0.45,
            escalate_to_human=True,
            uncertainty_level="low_confidence",
            uncertainty_signals_triggered=["U1_low_confidence", "U4_blurry_image"],
            factors=[
                RiskFactor(
                    source="threat_table",
                    value="MEDIUM threat (Tomato___Early_blight)",
                    contribution=0.35,
                    tag="[KB-derived draft, docs/review/threat_table_draft.md]",
                )
            ],
            progression_note="Disease detected at moderate risk. Blurry image raises uncertainty; human review recommended.",
            risk_computed_at=now,
        )
        case_name = "Tomato___Early_blight (Low confidence / blurry / escalated)"

    return case_name, vision, rag, weather, risk


def print_case_result(rec: RecommendationOutput) -> None:
    """Print readable structured summary of RecommendationOutput."""
    print(f"  Summary         : {rec.summary}")
    print(f"  Urgency         : {rec.urgency}")
    print(f"  What to do first: {rec.what_to_do_first}")
    print(f"  Steps ({len(rec.steps)} total) :")
    for idx, s in enumerate(rec.steps[:3], 1):
        print(f"    {idx}. {s[:110]}{'...' if len(s) > 110 else ''}")
    print(f"  Weather note    : {rec.weather_note}")
    print(f"  Uncertainty note: {rec.uncertainty_note}")
    print(f"  Flags           : {rec.flags}")
    print(f"  Citations       : {len(rec.citations)} sources passed through")
    print(f"  LLM Used        : {rec.llm_used} (fallback_reason: {rec.fallback_reason})")


def run_mode_a() -> None:
    """Run all 5 cases in deterministic fallback mode."""
    print("=" * 78)
    print("MODE (a): FALLBACK MODE (Deterministic fallback, no API key required)")
    print("=" * 78)

    # Force fallback mode by explicitly disabling provider
    agent = RecommendationAgent(provider=None)
    agent.provider = None
    agent.init_error = "missing_api_key (mode_a_fallback)"

    for case_id in range(1, 6):
        case_name, vision, rag, weather, risk = make_case_data(case_id)
        print(f"\n[{case_id}/5] {case_name}")
        print("-" * 78)
        rec = agent.recommend(
            vision=vision,
            rag=rag,
            weather=weather,
            risk=risk,
            include_doses=False,
        )
        print_case_result(rec)


def run_mode_b() -> bool:
    """Run all 5 cases with live Gemini LLM if GEMINI_API_KEY is available."""
    print("\n" + "=" * 78)
    print("MODE (b): LIVE GEMINI LLM MODE")
    print("=" * 78)

    load_local_env_safely()
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("[Mode B: Live Gemini skipped — GEMINI_API_KEY is not set in environment or local .env]")
        return False

    print("GEMINI_API_KEY detected. Connecting to Gemini provider via REST...")
    try:
        agent = RecommendationAgent()
        if agent.provider is None:
            print(f"[Mode B skipped: Provider initialization failed: {agent.init_error}]")
            return False
    except Exception as e:
        print(f"[Mode B skipped: Provider initialization error: {e}]")
        return False

    import time
    for case_id in range(1, 6):
        case_name, vision, rag, weather, risk = make_case_data(case_id)
        print(f"\n[{case_id}/5] {case_name}")
        print("-" * 78)
        try:
            rec = agent.recommend(
                vision=vision,
                rag=rag,
                weather=weather,
                risk=risk,
                include_doses=False,
            )
            print_case_result(rec)
            time.sleep(1.5)  # Pace requests within rate limits
        except Exception as e:
            print(f"  Error running live case {case_id}: {e}")

    return True


def main() -> None:
    print(f"LeafMind Recommendation Agent Smoke Run — {datetime.now(timezone.utc).isoformat()}")
    run_mode_a()
    run_mode_b()


if __name__ == "__main__":
    main()
