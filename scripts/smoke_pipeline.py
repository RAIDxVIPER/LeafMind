"""
scripts/smoke_pipeline.py
=========================
Smoke test running the complete Orchestrator graph (Stage C7) across 4 real local images:
  1. Confident diseased (Tomato Late Blight)
  2. Healthy leaf (Tomato healthy)
  3. Field OOD / blurry leaf (Apple Scab)
  4. Single-class crop (Squash Powdery Mildew)

Supports two execution modes:
  (a) Mocked weather + deterministic recommendation fallback (always runs, 0 network, 0 API key)
  (b) Live Open-Meteo + live Gemini (runs only if GEMINI_API_KEY is set in environment/.env)

No API keys, headers, or raw stack traces are ever printed or logged.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import uuid

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.recommendation_agent import RecommendationAgent
from src.agents.vision_agent import VisionAgent
from src.agents.weather_agent import WeatherAgent
from src.orchestrator.pipeline import run_case
from src.orchestrator.state import FinalResponse
from src.recommendation.provider import GeminiProvider, sanitize_key
from src.schemas.schemas import CurrentWeather, WeatherOutput, WeatherRiskFlag

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class MockWeatherAgent:
    """Mock WeatherAgent for offline/deterministic Mode (a)."""

    def get_weather(self, latitude=None, longitude=None, place_name=None, detected_disease=None, **kwargs) -> WeatherOutput:
        return WeatherOutput(
            weather_available=True,
            weather_partial=False,
            weather_score=0.20,
            current_weather=CurrentWeather(
                temperature_c=25.5,
                relative_humidity_pct=88.0,
                precipitation_mm=1.2,
                wind_speed_kmh=12.0,
                cloud_cover_pct=75.0,
                dew_point_c=23.4,
            ),
            disease_risk_flags=[
                WeatherRiskFlag(
                    condition_name="W01",
                    triggered=True,
                    current_value="RH=88%, temp_min=17C",
                    threshold="RH>90% AND temp_min<18C",
                    threshold_source="Fry (1977) [UNVERIFIED — I14]",
                )
            ],
            weather_data_source="mock-offline",
        )


class DeterministicFallbackRecommendationAgent(RecommendationAgent):
    """RecommendationAgent forced to deterministic fallback mode (no network LLM calls)."""

    def __init__(self):
        super().__init__()
        # Invalidate provider to force deterministic fallback
        self.provider = None


def run_smoke_mode(
    mode_name: str,
    weather_agent,
    rec_agent,
    cases: list,
) -> None:
    print(f"\n{'=' * 75}")
    print(f"RUNNING SMOKE TEST: {mode_name}")
    print(f"{'=' * 75}")

    for idx, (label, img_rel_path, crop, loc) in enumerate(cases, 1):
        full_img_path = REPO_ROOT / img_rel_path
        if not full_img_path.is_file():
            print(f"[{idx}/{len(cases)}] SKIPPED: File not found: {img_rel_path}")
            continue

        print(f"\n[{idx}/{len(cases)}] Case: {label}")
        print(f"  Image Path : {img_rel_path}")
        print(f"  Crop Input : {crop}")
        print(f"  Location   : {loc}")

        case_id = uuid.uuid4()

        try:
            resp: FinalResponse = run_case(
                image=full_img_path,
                crop=crop,
                location=loc,
                include_doses=False,
                case_id=case_id,
                weather_agent=weather_agent,
                recommendation_agent=rec_agent,
                use_memory=True,  # In-memory checkpointer for smoke test
            )

            # Print sanitized output highlights
            v = resp.vision
            w = resp.weather
            r = resp.rag
            rk = resp.risk
            rec = resp.recommendation
            esc = resp.escalation

            print(f"  Case ID          : {resp.case_id}")
            if v:
                print(f"  Vision Result    : {v.top_label} (conf={v.confidence:.3f}, ood={v.ood_flag}, can_diagnose={v.can_diagnose})")
                if v.top_3:
                    top3_str = ", ".join(f"{c.label} ({c.prob:.2f})" for c in v.top_3[:3])
                    print(f"  Top-3 Candidates : {top3_str}")
                if v.uncertainty_flags:
                    flags_fired = [f for f, fired in v.uncertainty_flags.model_dump().items() if fired]
                    if flags_fired:
                        print(f"  Quality Flags    : {flags_fired}")
            if w:
                print(f"  Weather Summary  : available={w.weather_available}, score={w.weather_score:.2f}, temp={w.current_temp_c}C, RH={w.current_humidity_pct}%, rules={w.triggered_rules}")
            if r:
                print(f"  RAG Retrieval    : found={r.found}, gap={r.kb_gap}, sources_count={len(r.sources)}")
            if rk:
                factor_strs = [f"{f.source}={f.contribution:.4f}" for f in rk.factors]
                print(f"  Risk Assessment  : level={rk.risk_level}, score={rk.risk_score:.4f} (factors: {', '.join(factor_strs)})")
            if rec:
                print(f"  Recommendation   : urgency={rec.urgency}, llm_used={rec.llm_used}, action='{rec.what_to_do_first or rec.summary}'")
                if rec.fallback_reason:
                    print(f"  Fallback Reason  : {rec.fallback_reason}")
            print(f"  Escalation       : escalate={esc.escalate_to_human}, priority={esc.priority}, triggers={esc.triggered_triggers}")
            print(f"  Severity Status  : {resp.severity_pct} ({resp.severity_note[:60]}...)")
            print(f"  Execution Timing : total ~{sum(resp.timing_ms.values()):.1f}ms (per-node: {resp.timing_ms})")
            print(f"  Partial Result   : {resp.partial_result}")
            if resp.flags:
                print(f"  Operational Flags: {resp.flags}")

        except Exception as e:
            sanitized = sanitize_key(str(e))
            print(f"  ERROR: Pipeline execution failed: {sanitized}")


def main():
    print("=" * 75)
    print("LeafMind — Stage C7 Orchestrator Smoke Test")
    print("=" * 75)
    print("NOTE: Cases 1-4 are benchmark/laboratory images (PlantVillage distribution).")
    print("      Confidence p=1.000 is expected on these clean lab images and is NOT")
    print("      evidence of real-world field accuracy.")
    print("      Cases 5-7 are real field images from PlantDoc (Test_OOD) illustrating")
    print("      field performance: correct prediction, wrong prediction, and blur.")
    print("=" * 75)

    cases = [
        (
            "1. Benchmark Lab Diseased (Tomato Late Blight - PlantVillage)",
            "Dataset_2/Test_ID/Tomato___Late_blight/PV__005a2c1f-4e15-49e4-9e5c-61dc3ecf9708___RS_Late.B 5096.JPG",
            "Tomato",
            (19.75, 75.71),  # Maharashtra coordinates
        ),
        (
            "2. Benchmark Lab Healthy (Tomato healthy - PlantVillage)",
            "Dataset_2/Test_ID/Tomato___healthy/PV__000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG",
            "Tomato",
            "Pune, Maharashtra",
        ),
        (
            "3. Benchmark Lab Single-Class Crop (Squash Powdery Mildew; can_diagnose=False)",
            "Dataset_2/Test_ID/Squash___Powdery_mildew/PV__0045a100-36d3-45df-b417-d487d6e07eb4___UMD_Powd.M 0003.JPG",
            "Squash",
            None,  # Default Maharashtra
        ),
        (
            "4. Benchmark Lab Multi-Class Diseased (Corn Northern Leaf Blight - PlantVillage)",
            "Dataset_2/Test_ID/Corn___Northern_Leaf_Blight/PV__00a55069-3fa3-405b-8d87-4d3408a6ed98___RS_NLB 3645.JPG",
            "Corn",
            "Nagpur, Maharashtra",
        ),
        (
            "5. Field OOD Multi-Class — PREDICTED CORRECTLY (Apple Scab - PlantDoc)",
            "Dataset_2/Test_OOD/Apple___Apple_scab/PD__Apple Scab Leaf (10).jpg",
            "Apple",
            "Shimla, Himachal Pradesh",
        ),
        (
            "6. Field OOD Multi-Class — PREDICTED WRONG (Apple Scab predicted as Healthy - PlantDoc)",
            "Dataset_2/Test_OOD/Apple___Apple_scab/PD__Apple Scab Leaf (1).jpg",
            "Apple",
            "Shimla, Himachal Pradesh",
        ),
        (
            "7. Field OOD Blurry Leaf (Apple Scab with blur quality flag - PlantDoc)",
            "Dataset_2/Test_OOD/Apple___Apple_scab/PD__Apple Scab Leaf (11).jpg",
            "Apple",
            "Shimla, Himachal Pradesh",
        ),
    ]

    # Mode (a): Mocked weather + deterministic recommendation fallback (Always runs)
    mock_weather = MockWeatherAgent()
    fallback_rec = DeterministicFallbackRecommendationAgent()
    run_smoke_mode(
        mode_name="Mode (a) — Mocked Weather + Deterministic Recommendation Fallback (Offline)",
        weather_agent=mock_weather,
        rec_agent=fallback_rec,
        cases=cases,
    )

    # Mode (b): Real Open-Meteo + Live Gemini (Only if GEMINI_API_KEY is present)
    # Check without printing key
    has_gemini_key = bool(os.getenv("GEMINI_API_KEY"))
    if not has_gemini_key:
        env_file = REPO_ROOT / ".env"
        if env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("GEMINI_API_KEY=") and len(line.split("=", 1)[1].strip()) > 5:
                        has_gemini_key = True
                        break
            except Exception:
                pass

    if has_gemini_key:
        print("\nGEMINI_API_KEY detected. Running Mode (b) with Live Open-Meteo and Live Gemini...")
        live_weather = WeatherAgent()
        live_rec = RecommendationAgent()
        run_smoke_mode(
            mode_name="Mode (b) — Live Open-Meteo + Live Gemini",
            weather_agent=live_weather,
            rec_agent=live_rec,
            cases=cases,
        )
    else:
        print("\nMode (b) SKIPPED: GEMINI_API_KEY not set in environment or .env.")


if __name__ == "__main__":
    main()
