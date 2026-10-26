"""
scripts/smoke_weather.py
========================
Optional live smoke script / test for Weather Agent (Stage C4).

Calls the real Open-Meteo Forecast and Geocoding APIs for Maharashtra coordinates.
Skipped by default in automated testing and CI.
Requires network connectivity and the environment flag:
    ENABLE_LIVE_WEATHER_TEST=1

Usage:
    # Standalone script execution:
    python scripts/smoke_weather.py

    # Under pytest (only runs when flag is set):
    ENABLE_LIVE_WEATHER_TEST=1 pytest scripts/smoke_weather.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.weather_agent import WeatherAgent
from src.schemas.schemas import WeatherOutput


def run_live_smoke() -> WeatherOutput:
    """Execute live weather call against real Open-Meteo APIs."""
    print("=" * 70)
    print("LeafMind Weather Agent -- Live Smoke Test")
    print("=" * 70)

    agent = WeatherAgent()

    # Test coordinate: Pune, Maharashtra (18.5204, 73.8567)
    test_lat, test_lon = 18.5204, 73.8567
    print(f"1. Fetching live weather for Pune, Maharashtra ({test_lat}, {test_lon})...")
    output = agent.get_weather(
        gps_lat=test_lat,
        gps_lon=test_lon,
        detected_disease="Tomato___Early_blight",
    )

    print(f"\n[Result] Available: {output.weather_available}")
    print(f"[Result] Location Source: {output.location_source}")
    print(f"[Result] Data Source: {output.weather_data_source}")
    print(f"[Result] Fetched At: {output.fetched_at}")
    print(f"[Result] Cache Hit: {output.cache_hit} | Stale: {output.stale}")

    if output.current_weather:
        cw = output.current_weather
        print(f"\nCurrent Weather: {cw.temperature_c} deg C, {cw.relative_humidity_pct}% RH, "
              f"{cw.precipitation_mm}mm rain, {cw.wind_speed_kmh}km/h wind, dew point {cw.dew_point_c} deg C")

    if output.summary_statistics:
        st = output.summary_statistics
        print(f"\nSummary Statistics (past 24h window):")
        print(f"  - Mean Temp: {st.mean_temperature_c} deg C (min: {st.min_temperature_c} deg C, max: {st.max_temperature_c} deg C)")
        print(f"  - Mean RH: {st.mean_relative_humidity_pct}% (min: {st.min_relative_humidity_pct}%, max: {st.max_relative_humidity_pct}%)")
        print(f"  - Total Precipitation: {st.total_precipitation_mm} mm over {st.precipitation_hours} hours")
        print(f"  - Hours RH >= 90%: {st.hours_rh_above_threshold} hrs")
        print(f"  - Leaf Wetness Hours Proxy [ENG proxy]: {st.leaf_wetness_hours_proxy} hrs")
        print(f"  - Max Wind: {st.max_wind_speed_kmh} km/h")

    print(f"\nWeather Score for Tomato Early Blight: {output.weather_score}")
    print("\nFavourable Condition Rules Triggered (W01-W08):")
    for r in output.rules_triggered:
        status = "TRIGGERED" if r.triggered else "not triggered"
        print(f"  [{r.rule_id}] {status:13s} | {r.condition_name}")
        print(f"        Measured: {r.measured_values}")
        print(f"        Threshold: {r.threshold}")

    print("\n2. Testing Geocoding resolution for 'Nashik'...")
    geo_out = agent.get_weather(place_name="Nashik")
    print(f"[Geocoding] Resolved to: {geo_out.location_name}, {geo_out.location_admin} ({geo_out.latitude}, {geo_out.longitude})")
    print(f"[Geocoding] Location Source: {geo_out.location_source}")

    print("\n3. Testing Default Region (no input given)...")
    def_out = agent.get_weather()
    print(f"[Default] Resolved to: {def_out.location_name} ({def_out.latitude}, {def_out.longitude})")
    print(f"[Default] Location Source: {def_out.location_source}")

    print("\n" + "=" * 70)
    print("Live Smoke Test COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return output


@pytest.mark.skipif(
    os.getenv("ENABLE_LIVE_WEATHER_TEST") != "1",
    reason="Live network test skipped by default. Set ENABLE_LIVE_WEATHER_TEST=1 to run.",
)
def test_live_weather_smoke():
    """Optional pytest wrapper for smoke test, skipped by default."""
    out = run_live_smoke()
    assert out.weather_available is True
    assert out.current_weather is not None
    assert len(out.rules_triggered) == 8


if __name__ == "__main__":
    try:
        run_live_smoke()
    except Exception as exc:
        print(f"\nSmoke test encountered error (network may be unavailable): {exc}", file=sys.stderr)
        sys.exit(1)
