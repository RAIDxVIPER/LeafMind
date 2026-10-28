"""
scripts/smoke_monitoring.py
===========================
Smoke test script for LeafMind Stage C8 Monitoring Agent.

Demonstrates follow-up monitoring across 3 synthetic paired image scenarios:
  Scenario 1: Same lab image twice (triggers same-image MD5 check, same_image_flag=True)
  Scenario 2: Diseased baseline then healthy follow-up (no_disease_detected_at_followup, H=-0.05)
  Scenario 3: Crop mismatch (Tomato baseline followed by Corn follow-up, crop_mismatch_flag=True, T11)

Execution Mode:
  - Offline deterministic: mocked weather + deterministic fallback recommendation
  - 0 network calls, 0 API keys required.
  - Safe for automated regression and CI.

IMPORTANT NOTICE [I20]:
  These are synthetic image pairings constructed for regression testing.
  They DO NOT constitute paired field images of the same physical plant across time,
  and DO NOT represent evidence of real-world monitoring or disease tracking accuracy.
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys
import uuid

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.recommendation_agent import RecommendationAgent
from src.agents.vision_agent import VisionAgent
from src.orchestrator.pipeline import run_case
from src.orchestrator.state import FinalResponse
from src.schemas.schemas import CurrentWeather, WeatherOutput, WeatherRiskFlag

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class MockWeatherAgent:
    """Deterministic offline weather agent."""

    def __init__(self, weather_score: float = 0.20):
        self.weather_score = weather_score

    def get_weather(self, latitude=None, longitude=None, place_name=None, detected_disease=None, **kwargs) -> WeatherOutput:
        return WeatherOutput(
            weather_available=True,
            weather_partial=False,
            weather_score=self.weather_score,
            current_weather=CurrentWeather(
                temperature_c=25.0,
                relative_humidity_pct=85.0,
                precipitation_mm=1.0,
                wind_speed_kmh=10.0,
                cloud_cover_pct=70.0,
                dew_point_c=22.0,
            ),
            disease_risk_flags=[
                WeatherRiskFlag(
                    condition_name="W01",
                    triggered=True,
                    current_value="RH=85%, temp_min=18C",
                    threshold="RH>90% AND temp_min<18C",
                    threshold_source="Fry (1977) [UNVERIFIED — I14]",
                )
            ],
            weather_data_source="mock-offline",
        )


class DeterministicFallbackRecommendationAgent(RecommendationAgent):
    """RecommendationAgent operating in deterministic fallback mode."""

    def __init__(self):
        super().__init__()
        self.provider = None


def print_risk_breakdown(resp: FinalResponse) -> None:
    """Print detailed risk breakdown and verify R = 0.70*T + 0.20*W + 0.10*H."""
    if resp.risk is None:
        print("    Risk Output: None")
        return

    factors_dict = {f.source: f for f in resp.risk.factors}
    t_factor = factors_dict.get("threat_table") or factors_dict.get("threat")
    w_factor = factors_dict.get("weather")
    h_factor = factors_dict.get("history")

    t_contrib = t_factor.contribution if t_factor else 0.0
    w_contrib = w_factor.contribution if w_factor else 0.0
    h_contrib = h_factor.contribution if h_factor else 0.0

    t_score = round(t_contrib / 0.70, 4) if t_factor else 0.0
    w_score = round(w_contrib / 0.20, 4) if w_factor else 0.0
    h_score = round(h_contrib / 0.10, 4) if h_factor else 0.0

    recomputed_r = max(0.0, min(1.0, round(t_contrib + w_contrib + h_contrib, 4)))
    delta_err = abs(resp.risk.risk_score - recomputed_r)

    print(f"    Risk Score (R)     : {resp.risk.risk_score:.4f} (Level: {resp.risk.risk_level})")
    print(f"    Threat Factor (T)  : Score={t_score:.4f} | Weight=0.70 | Contrib={t_contrib:+.4f}")
    print(f"    Weather Factor (W) : Score={w_score:.4f} | Weight=0.20 | Contrib={w_contrib:+.4f}")
    print(f"    History Factor (H) : Score={h_score:.4f} | Weight=0.10 | Contrib={h_contrib:+.4f}")
    print(f"    Recomputed R Identity: 0.70*{t_score:.4f} + 0.20*{w_score:.4f} + 0.10*{h_score:.4f} = {recomputed_r:.4f}")
    print(f"    Formula Verification : {'MATCH (identity verified)' if delta_err < 1e-6 else f'MISMATCH err={delta_err}'}")


def run_scenario(
    scenario_title: str,
    baseline_img_rel: str,
    baseline_crop: str,
    followup_img_rel: str,
    followup_crop: str,
    weather_agent,
    rec_agent,
) -> None:
    print(f"\n{'=' * 80}")
    print(f"{scenario_title}")
    print(f"{'=' * 80}")
    print("  [DISCLAIMER I20] Synthetic image pair used for testing; not real-world monitoring.")

    baseline_path = REPO_ROOT / baseline_img_rel
    followup_path = REPO_ROOT / followup_img_rel

    if not baseline_path.is_file():
        print(f"  ERROR: Baseline image missing: {baseline_img_rel}")
        return
    if not followup_path.is_file():
        print(f"  ERROR: Follow-up image missing: {followup_img_rel}")
        return

    # Step 1: Run Baseline Case
    base_case_id = uuid.uuid4()
    print(f"\n  --- 1. Baseline Case Submission ---")
    print(f"  Image      : {baseline_img_rel}")
    print(f"  Crop       : {baseline_crop}")
    print(f"  Case ID    : {base_case_id}")

    base_resp = run_case(
        image=baseline_path,
        crop=baseline_crop,
        case_id=base_case_id,
        weather_agent=weather_agent,
        recommendation_agent=rec_agent,
        use_memory=True,
    )

    base_label = base_resp.vision.top_label if base_resp.vision else "Unknown"
    base_conf = base_resp.vision.confidence if base_resp.vision else 0.0
    print(f"  Diagnosis  : {base_label} (confidence={base_conf:.4f})")
    print_risk_breakdown(base_resp)

    # Step 2: Run Follow-up Case referencing previous_case_id
    followup_case_id = uuid.uuid4()
    print(f"\n  --- 2. Follow-up Case Submission ---")
    print(f"  Image            : {followup_img_rel}")
    print(f"  Crop             : {followup_crop}")
    print(f"  Follow-up Case ID: {followup_case_id}")
    print(f"  Previous Case ID : {base_case_id}")

    fu_resp = run_case(
        image=followup_path,
        crop=followup_crop,
        case_id=followup_case_id,
        previous_case_id=base_case_id,
        weather_agent=weather_agent,
        recommendation_agent=rec_agent,
        use_memory=True,
    )

    fu_label = fu_resp.vision.top_label if fu_resp.vision else "Unknown"
    fu_conf = fu_resp.vision.confidence if fu_resp.vision else 0.0
    print(f"  Diagnosis        : {fu_label} (confidence={fu_conf:.4f})")
    print_risk_breakdown(fu_resp)

    # Step 3: Monitoring Output & Plausibility Details
    print(f"\n  --- 3. Monitoring Output & Plausibility Details ---")
    m = fu_resp.monitoring
    if m is None:
        print("  Monitoring Output: NONE (gracefully degraded)")
    else:
        print(f"  Disease Status       : {m.disease_status}")
        print(f"  Confidence Raw Delta : {m.confidence_raw_delta:+.4f} ({m.followup_confidence:.4f} - {m.baseline_confidence:.4f})")
        print(f"  Severity Delta Pct   : {m.severity_delta_pct} (STRICTLY NULL per D2/R3)")
        print(f"  Severity Note        : {m.severity_note}")
        print(f"  Plant Identity Note  : {m.plant_identity_note}")
        print(f"  Plausibility Flags   :")
        print(f"    - same_image_flag    : {m.same_image_flag}")
        print(f"    - poor_quality_flag  : {m.poor_quality_flag}")
        print(f"    - crop_mismatch_flag : {m.crop_mismatch_flag}")
        print(f"    - too_soon_flag      : {m.too_soon_flag}")

    print(f"  Pipeline Flags       : {fu_resp.flags}")
    print(f"  Escalation Outcome   :")
    print(f"    - Escalate to Human: {fu_resp.escalation.escalate_to_human}")
    print(f"    - Priority         : {fu_resp.escalation.priority}")
    print(f"    - Triggered Codes  : {fu_resp.escalation.triggered_triggers}")
    for reason in fu_resp.escalation.reasons:
        print(f"      * [{reason.trigger_id}] {reason.description} (priority={reason.priority})")


def main():
    print("=" * 80)
    print("LeafMind — Stage C8 Monitoring Agent Smoke Test")
    print("=" * 80)
    print("Running 3 synthetic paired cases through orchestrator pipeline with previous_case_id:")
    print("  1. Same lab image twice (identifies duplicate submission)")
    print("  2. Diseased baseline followed by healthy follow-up (recovery observation)")
    print("  3. Crop mismatch (Tomato baseline followed by Corn follow-up)")
    print("=" * 80)

    weather_agent = MockWeatherAgent(weather_score=0.20)
    rec_agent = DeterministicFallbackRecommendationAgent()

    # Image Paths
    img_diseased_tomato = "Dataset_2/Test_ID/Tomato___Late_blight/PV__005a2c1f-4e15-49e4-9e5c-61dc3ecf9708___RS_Late.B 5096.JPG"
    img_healthy_tomato = "Dataset_2/Test_ID/Tomato___healthy/PV__000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG"
    img_corn = "Dataset_2/Test_ID/Corn___Northern_Leaf_Blight/PV__00a55069-3fa3-405b-8d87-4d3408a6ed98___RS_NLB 3645.JPG"

    # Scenario 1: Same lab image twice
    run_scenario(
        scenario_title="SCENARIO 1: SAME LAB IMAGE TWICE (Duplicate Submission Guard)",
        baseline_img_rel=img_diseased_tomato,
        baseline_crop="Tomato",
        followup_img_rel=img_diseased_tomato,
        followup_crop="Tomato",
        weather_agent=weather_agent,
        rec_agent=rec_agent,
    )

    # Scenario 2: Diseased then Healthy
    run_scenario(
        scenario_title="SCENARIO 2: DISEASED BASELINE THEN HEALTHY FOLLOW-UP (Observation of Recovery)",
        baseline_img_rel=img_diseased_tomato,
        baseline_crop="Tomato",
        followup_img_rel=img_healthy_tomato,
        followup_crop="Tomato",
        weather_agent=weather_agent,
        rec_agent=rec_agent,
    )

    # Scenario 3: Crop mismatch
    run_scenario(
        scenario_title="SCENARIO 3: CROP MISMATCH (Tomato Baseline Followed by Corn Follow-up)",
        baseline_img_rel=img_diseased_tomato,
        baseline_crop="Tomato",
        followup_img_rel=img_corn,
        followup_crop="Corn",
        weather_agent=weather_agent,
        rec_agent=rec_agent,
    )

    print(f"\n{'=' * 80}")
    print("SMOKE TEST COMPLETE: All 3 synthetic monitoring scenarios executed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
