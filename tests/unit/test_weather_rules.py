"""
tests/unit/test_weather_rules.py
================================
Unit tests for Weather Disease-Risk Rules (W01–W08) and Part 4 Weather Score (Stage C4).

Covers:
  - Each rule triggers and does not trigger at threshold boundaries:
      W01: Late blight (RH > 90% AND temp_min < 18°C)
      W02: Powdery mildew (RH < 50% AND temp 20–30°C)
      W03: Bacterial spot (temp > 24°C AND precip > 0 AND wind > 15 km/h)
      W04: Early blight (temp > 24°C AND (precip_hours > 2 OR leaf_wetness_proxy > 2h))
      W05: Septoria leaf spot (temp 15–25°C AND RH > 85%)
      W06: Yellow leaf curl virus (temp > 28°C AND RH < 40%)
      W07: Rust (temp 15–22°C AND dew_point within 2°C of temp_min)
      W08: General epidemic risk (precip_sum > 10mm in last 24h)
  - Leaf-wetness proxy derivation labeled [ENG proxy]
  - Weather score computation per Part 4 formula:
      * Worked examples on single and multiple applicable rules
      * Healthy / unmapped diseases return 0.0
      * W08 excluded from scoring per spec Part 4 §4a-iv & Part 6 §6d-v
      * Tagged [UNVERIFIED — I14]
"""
from __future__ import annotations

import pytest

from src.weather.rules import WeatherRuleEngine
from src.schemas.schemas import RuleTriggerResult, WeatherSummaryStats


def make_hourly_series(
    temp: float = 22.0,
    rh: float = 70.0,
    precip: float = 0.0,
    wind: float = 10.0,
    dew: float = 16.0,
    hours: int = 24,
) -> dict:
    """Helper to generate uniform 24-hour baseline data."""
    return {
        "temperature_2m": [temp] * hours,
        "relative_humidity_2m": [rh] * hours,
        "precipitation": [precip] * hours,
        "wind_speed_10m": [wind] * hours,
        "dew_point_2m": [dew] * hours,
    }


class TestWeatherSummaryStats:
    """Verify aggregate summary statistics and leaf-wetness proxy."""

    def test_summary_statistics_computation(self):
        engine = WeatherRuleEngine()
        data = {
            "temperature_2m": [15.0, 20.0, 25.0],
            "relative_humidity_2m": [85.0, 92.0, 95.0],
            "precipitation": [0.0, 2.0, 0.0],
            "wind_speed_10m": [10.0, 15.0, 20.0],
        }
        stats = engine.compute_summary_statistics(data)
        assert stats.mean_temperature_c == 20.0
        assert stats.max_temperature_c == 25.0
        assert stats.min_temperature_c == 15.0
        assert stats.mean_relative_humidity_pct == pytest.approx(90.67, abs=0.01)
        assert stats.max_relative_humidity_pct == 95.0
        assert stats.min_relative_humidity_pct == 85.0
        assert stats.total_precipitation_mm == 2.0
        assert stats.precipitation_hours == 1
        # Hours with RH >= 90% (2 hours: 92% and 95%)
        assert stats.hours_rh_above_threshold == 2
        # Leaf wetness hours: RH>=90% or precip>=0.1mm (hours 1 and 2 match -> 2)
        assert stats.leaf_wetness_hours_proxy == 2
        assert stats.max_wind_speed_kmh == 20.0

    def test_summary_stats_equal_rule_stats_with_forecast_rain_injected(self):
        """
        Verify that compute_summary_statistics and evaluate_rules use the exact same
        rolling 24-hour window ending at current observation time:
        - Heavy rain and high heat injected into historical days (day -3).
        - Heavy rain and high heat injected into forecast days (day +2).
        - Summary stats must strictly equal the 24-hour metrics evaluated by rules.
        """
        from datetime import datetime, timedelta

        engine = WeatherRuleEngine()
        base_time = datetime(2026, 10, 20, 0, 0)
        times = [(base_time + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(240)]

        curr_idx = 168
        curr_time_str = times[curr_idx]

        temps = [25.0] * 240
        rhs = [50.0] * 240
        precips = [0.0] * 240
        winds = [10.0] * 240
        dews = [12.0] * 240

        # Inject historical rain 3 days ago (hours 50-60)
        for h in range(50, 60):
            precips[h] = 15.0
            rhs[h] = 95.0
            temps[h] = 34.0
            winds[h] = 25.0

        # Inject forecast rain 2 days in future (hours 200-210)
        for h in range(200, 210):
            precips[h] = 20.0
            rhs[h] = 98.0
            temps[h] = 35.0
            winds[h] = 30.0

        # In the past 24h window ending at curr_time_str (hours curr_idx-23 to curr_idx)
        for i, h in enumerate(range(curr_idx - 23, curr_idx + 1)):
            temps[h] = 19.5 + (31.0 - 19.5) * (i / 23.0)  # min 19.5, max 31.0
            rhs[h] = 35.0 + (78.0 - 35.0) * (i / 23.0)     # min 35.0, max 78.0
            precips[h] = 0.0                               # 0.0 mm
            winds[h] = 8.0 + (14.0 - 8.0) * (i / 23.0)     # max 14.0
            dews[h] = 11.0

        hourly_data = {
            "time": times,
            "temperature_2m": temps,
            "relative_humidity_2m": rhs,
            "precipitation": precips,
            "wind_speed_10m": winds,
            "dew_point_2m": dews,
        }

        stats = engine.compute_summary_statistics(hourly_data, current_time_str=curr_time_str)
        rule_results = {r.rule_id: r for r in engine.evaluate_rules(hourly_data, summary_stats=stats, current_time_str=curr_time_str)}

        # 1. Assert summary stats strictly exclude past 7d and future forecast
        assert stats.max_temperature_c == 31.0
        assert stats.min_temperature_c == 19.5
        assert stats.max_relative_humidity_pct == 78.0
        assert stats.min_relative_humidity_pct == 35.0
        assert stats.total_precipitation_mm == 0.0
        assert stats.precipitation_hours == 0
        assert stats.leaf_wetness_hours_proxy == 0
        assert stats.max_wind_speed_kmh == 14.0

        # 2. Assert rule evaluations used the exact same 24-hour summary stats
        assert f"RH_max={stats.max_relative_humidity_pct:.1f}%" in rule_results["W01"].measured_values
        assert f"temp_min={stats.min_temperature_c:.1f}°C" in rule_results["W01"].measured_values
        assert f"temp_max={stats.max_temperature_c:.1f}°C" in rule_results["W03"].measured_values
        assert f"precip_sum={stats.total_precipitation_mm:.1f}mm" in rule_results["W03"].measured_values
        assert f"wind_max={stats.max_wind_speed_kmh:.1f}km/h" in rule_results["W03"].measured_values
        assert f"temp_max={stats.max_temperature_c:.1f}°C" in rule_results["W04"].measured_values
        assert f"precip_hours={stats.precipitation_hours}" in rule_results["W04"].measured_values
        assert f"precip_sum_24h={stats.total_precipitation_mm:.1f}mm" in rule_results["W08"].measured_values



class TestRuleBoundaries:
    """Verify triggers and boundary non-triggers for W01–W08."""

    @pytest.fixture
    def engine(self):
        return WeatherRuleEngine()

    # ── W01: Late blight (RH > 90% AND temp_min < 18°C) ──
    def test_w01_triggers(self, engine):
        data = make_hourly_series(temp=17.0, rh=91.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W01"].triggered is True
        assert results["W01"].tag == "[UNVERIFIED — I14]"

    def test_w01_boundary_rh_fails_at_90(self, engine):
        # RH must be strictly > 90%
        data = make_hourly_series(temp=17.0, rh=90.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W01"].triggered is False

    def test_w01_boundary_temp_fails_at_18(self, engine):
        # Temp must be strictly < 18°C
        data = make_hourly_series(temp=18.0, rh=92.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W01"].triggered is False

    # ── W02: Powdery mildew (RH < 50% AND temp 20–30°C) ──
    def test_w02_triggers(self, engine):
        data = make_hourly_series(temp=25.0, rh=45.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W02"].triggered is True

    def test_w02_boundary_rh_fails_at_50(self, engine):
        # RH must be strictly < 50%
        data = make_hourly_series(temp=25.0, rh=50.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W02"].triggered is False

    def test_w02_temp_outside_range_fails(self, engine):
        data_cold = make_hourly_series(temp=19.9, rh=40.0)
        data_hot = make_hourly_series(temp=30.1, rh=40.0)
        assert {r.rule_id: r for r in engine.evaluate_rules(data_cold)}["W02"].triggered is False
        assert {r.rule_id: r for r in engine.evaluate_rules(data_hot)}["W02"].triggered is False

    # ── W03: Bacterial spot (temp > 24°C AND precip > 0 AND wind > 15 km/h) ──
    def test_w03_triggers(self, engine):
        data = make_hourly_series(temp=24.5, precip=0.2, wind=15.5)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W03"].triggered is True

    def test_w03_boundary_fails(self, engine):
        # temp must be > 24
        d1 = make_hourly_series(temp=24.0, precip=0.2, wind=16.0)
        # precip must be > 0
        d2 = make_hourly_series(temp=25.0, precip=0.0, wind=16.0)
        # wind must be > 15
        d3 = make_hourly_series(temp=25.0, precip=0.2, wind=15.0)

        assert {r.rule_id: r for r in engine.evaluate_rules(d1)}["W03"].triggered is False
        assert {r.rule_id: r for r in engine.evaluate_rules(d2)}["W03"].triggered is False
        assert {r.rule_id: r for r in engine.evaluate_rules(d3)}["W03"].triggered is False

    def test_w03_combines_24h_aggregates_across_different_hours(self, engine):
        """
        Verify W03 combines 24-hour summary aggregates:
        Driving rainstorm occurs at night (hour 0: temp=20°C <= 24, precip=1.5mm, wind=20 km/h).
        Warm peak occurs during afternoon (hour 14: temp=27°C > 24, no rain, mild wind).
        Assert W03 triggers because over the past 24h: temp_max > 24, precip_sum > 0, wind_max > 15.
        """
        data = make_hourly_series(temp=22.0, precip=0.0, wind=10.0)
        data["temperature_2m"][14] = 27.0
        data["precipitation"][0] = 1.5
        data["wind_speed_10m"][0] = 20.0
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W03"].triggered is True
        assert "temp_max=27.0°C" in results["W03"].measured_values
        assert "precip_sum=1.5mm" in results["W03"].measured_values
        assert "wind_max=20.0km/h" in results["W03"].measured_values

    # ── W04: Early blight (temp > 24°C AND precip_hours > 2) ──
    def test_w04_triggers_via_precip_hours(self, engine):
        data = make_hourly_series(temp=25.0, precip=0.0, rh=60.0)
        # Add 3 hours of precipitation
        data["precipitation"][0] = 1.0
        data["precipitation"][1] = 1.0
        data["precipitation"][2] = 1.0
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W04"].triggered is True
        assert "precip_hours=3" in results["W04"].measured_values

    def test_w04_leaf_wetness_alone_does_not_trigger_without_rain(self, engine):
        """
        In docs/spec/part6_build_spec.md §6d-v, W04 is strictly 'temp > 24°C AND precip_hours > 2'.
        High relative humidity / leaf wetness proxy without actual rainfall does NOT trigger W04.
        """
        data = make_hourly_series(temp=26.0, precip=0.0, rh=95.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W04"].triggered is False
        assert "precip_hours=0" in results["W04"].measured_values

    def test_w04_boundary_fails_at_2_hours_rain(self, engine):
        # Exactly 2 hours of rain does NOT trigger (> 2 required)
        data = make_hourly_series(temp=25.0, precip=0.0, rh=70.0)
        data["precipitation"][0] = 1.0
        data["precipitation"][1] = 1.0
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W04"].triggered is False

    def test_w04_fails_when_cold(self, engine):
        # 5 hours of rain, but temp <= 24
        data = make_hourly_series(temp=23.0, precip=2.0, rh=95.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W04"].triggered is False

    def test_w04_pune_smoke_values_not_triggered(self, engine):
        """
        Exact Pune smoke values:
        temp_max = 33.6°C, precip_hours = 0.
        Must NOT trigger W04 (Early blight requires temp > 24 AND precip_hours > 2).
        """
        temps = [18.0 + (33.6 - 18.0) * (i / 23.0) for i in range(24)]
        precips = [0.0] * 24
        rhs = [30.0 + (55.0 - 30.0) * (i / 23.0) for i in range(24)]

        data = {
            "temperature_2m": temps,
            "precipitation": precips,
            "relative_humidity_2m": rhs,
            "wind_speed_10m": [10.0] * 24,
            "dew_point_2m": [15.0] * 24,
        }
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        res_w04 = results["W04"]
        assert res_w04.triggered is False
        assert "temp_max=33.6°C" in res_w04.measured_values
        assert "precip_hours=0" in res_w04.measured_values

    def test_w04_clause_failure_permutations(self, engine):
        """
        Test the boolean truth table for W04 (temp > 24 AND precip_hours > 2):
        """
        # Case 1: Temp clause fails alone (temp <= 24, precip_hours > 2)
        d_temp_fails = make_hourly_series(temp=22.0, precip=1.0, rh=95.0)
        assert {r.rule_id: r for r in engine.evaluate_rules(d_temp_fails)}["W04"].triggered is False

        # Case 2: Precip clause fails alone (temp > 24, precip_hours <= 2)
        d_precip_fails = make_hourly_series(temp=28.0, precip=0.0, rh=60.0)
        d_precip_fails["precipitation"][0] = 0.5  # precip_hours = 1
        assert {r.rule_id: r for r in engine.evaluate_rules(d_precip_fails)}["W04"].triggered is False

        # Case 3: Both clauses fail (temp <= 24 AND precip_hours <= 2)
        d_both_fail = make_hourly_series(temp=20.0, precip=0.0, rh=50.0)
        assert {r.rule_id: r for r in engine.evaluate_rules(d_both_fail)}["W04"].triggered is False

        # Case 4: Both clauses pass (temp > 24 AND precip_hours > 2)
        d_pass = make_hourly_series(temp=26.0, precip=0.0, rh=60.0)
        for i in range(3):
            d_pass["precipitation"][i] = 1.0  # precip_hours = 3
        assert {r.rule_id: r for r in engine.evaluate_rules(d_pass)}["W04"].triggered is True



    # ── W05: Septoria leaf spot (temp 15–25°C AND RH > 85%) ──
    def test_w05_triggers(self, engine):
        data = make_hourly_series(temp=20.0, rh=86.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W05"].triggered is True

    def test_w05_boundary_fails(self, engine):
        # RH = 85.0 fails (must be > 85%)
        d1 = make_hourly_series(temp=20.0, rh=85.0)
        d2 = make_hourly_series(temp=14.9, rh=90.0)
        d3 = make_hourly_series(temp=25.1, rh=90.0)
        assert {r.rule_id: r for r in engine.evaluate_rules(d1)}["W05"].triggered is False
        assert {r.rule_id: r for r in engine.evaluate_rules(d2)}["W05"].triggered is False
        assert {r.rule_id: r for r in engine.evaluate_rules(d3)}["W05"].triggered is False

    # ── W06: Yellow leaf curl virus (temp > 28°C AND RH < 40%) ──
    def test_w06_triggers(self, engine):
        data = make_hourly_series(temp=29.0, rh=38.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W06"].triggered is True

    def test_w06_boundary_fails(self, engine):
        # temp = 28.0 fails (> 28 required)
        d1 = make_hourly_series(temp=28.0, rh=35.0)
        # rh = 40.0 fails (< 40 required)
        d2 = make_hourly_series(temp=30.0, rh=40.0)
        assert {r.rule_id: r for r in engine.evaluate_rules(d1)}["W06"].triggered is False
        assert {r.rule_id: r for r in engine.evaluate_rules(d2)}["W06"].triggered is False

    # ── W07: Rust (temp 15–22°C AND dew_point within 2°C of temp_min) ──
    def test_w07_triggers(self, engine):
        data = make_hourly_series(temp=18.0, dew=17.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W07"].triggered is True

    def test_w07_boundary_fails(self, engine):
        # diff > 2°C fails
        d1 = make_hourly_series(temp=18.0, dew=15.5)  # diff = 2.5 > 2.0
        # temp outside 15–22 fails
        d2 = make_hourly_series(temp=23.0, dew=22.0)
        assert {r.rule_id: r for r in engine.evaluate_rules(d1)}["W07"].triggered is False
        assert {r.rule_id: r for r in engine.evaluate_rules(d2)}["W07"].triggered is False

    def test_w07_dew_point_missing_returns_null_and_does_not_trigger(self, engine):
        """
        When dew-point data is missing or None:
        - Must record min_dew_point_diff=null (NOT 999.0).
        - W07 must NOT trigger.
        """
        # Case A: dew_point_2m key omitted entirely
        d_missing = make_hourly_series(temp=18.0)
        d_missing.pop("dew_point_2m", None)
        res_a = {r.rule_id: r for r in engine.evaluate_rules(d_missing)}
        assert res_a["W07"].triggered is False
        assert "min_dew_point_diff=null" in res_a["W07"].measured_values
        assert "999.0" not in res_a["W07"].measured_values

        # Case B: dew_point_2m contains all None values
        d_nones = make_hourly_series(temp=18.0)
        d_nones["dew_point_2m"] = [None] * 24
        res_b = {r.rule_id: r for r in engine.evaluate_rules(d_nones)}
        assert res_b["W07"].triggered is False
        assert "min_dew_point_diff=null" in res_b["W07"].measured_values
        assert "999.0" not in res_b["W07"].measured_values

    def test_past_24h_window_excludes_forecast_and_is_not_calendar_day(self, engine):
        """
        Confirm 'past 24h' means strictly the 24 hours ending at current observation time:
        - Never includes forecast hours (hours after current time).
        - Is a rolling 24-hour window, NOT a calendar day (00:00-23:59).
        """
        from datetime import datetime, timedelta

        # Build a 240-hour timeline: 7 past days (168h) + current hour + 3 forecast days (71h)
        # Starting 2026-10-20T00:00 to 2026-10-29T23:00
        base_time = datetime(2026, 10, 20, 0, 0)
        times = [(base_time + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(240)]

        # Current observation time: 2026-10-27T14:00 (index 182)
        curr_time_str = "2026-10-27T14:00"
        curr_idx = times.index(curr_time_str)

        # Base series: warm and dry
        temps = [28.0] * 240
        rhs = [40.0] * 240
        precips = [0.0] * 240
        winds = [5.0] * 240
        dews = [10.0] * 240

        # Inject heavy rain into forecast hours (after curr_time_str)
        # E.g. hours curr_idx + 10 to curr_idx + 15
        for h in range(curr_idx + 5, curr_idx + 15):
            precips[h] = 10.0
            rhs[h] = 95.0

        hourly_data = {
            "time": times,
            "temperature_2m": temps,
            "relative_humidity_2m": rhs,
            "precipitation": precips,
            "wind_speed_10m": winds,
            "dew_point_2m": dews,
        }

        # Verify window slice isolation directly
        start_idx, end_idx = engine._isolate_past_24h_window(hourly_data, current_time_str=curr_time_str)
        assert end_idx - start_idx == 24
        assert times[end_idx - 1] == curr_time_str
        assert times[start_idx] == times[curr_idx - 23]
        # Rolling window crosses midnight: Oct 26 15:00 to Oct 27 14:00 (NOT calendar day)
        assert times[start_idx].startswith("2026-10-26T15:00")
        assert times[end_idx - 1].startswith("2026-10-27T14:00")

        # Evaluate rules:
        # Forecast has 10h of rain, but past 24h has 0h of rain.
        # W04 and W08 must NOT trigger!
        results = {r.rule_id: r for r in engine.evaluate_rules(hourly_data, current_time_str=curr_time_str)}
        assert results["W04"].triggered is False
        assert results["W08"].triggered is False


    # ── W08: General epidemic risk (precip_sum > 10mm in last 24h) ──
    def test_w08_triggers_and_is_omitted_from_scoring(self, engine):
        # 24 hours of 0.5mm = 12.0mm (> 10mm)
        data = make_hourly_series(precip=0.5)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W08"].triggered is True
        assert results["W08"].omitted_from_scoring is True
        assert "[GAP]" in results["W08"].threshold_source

    def test_w08_fails_at_10mm(self, engine):
        # Exactly 10.0mm fails (> 10 required)
        data = make_hourly_series(precip=10.0 / 24.0)
        results = {r.rule_id: r for r in engine.evaluate_rules(data)}
        assert results["W08"].triggered is False


class TestWeatherScore:
    """Verify Part 4 §4a-iv weather_score formula."""

    @pytest.fixture
    def engine(self):
        return WeatherRuleEngine()

    def test_weather_score_disease_specific_triggered(self, engine):
        """Tomato Late Blight triggers rule W01 -> score = 1.0."""
        data = make_hourly_series(temp=16.0, rh=95.0)
        rule_results = engine.evaluate_rules(data)
        score = engine.compute_weather_score(rule_results, diagnosed_disease="Tomato___Late_blight")
        assert score == 1.0

    def test_weather_score_disease_specific_untriggered(self, engine):
        """Dry warm weather does not trigger W01 for Late Blight -> score = 0.0."""
        data = make_hourly_series(temp=25.0, rh=40.0)
        rule_results = engine.evaluate_rules(data)
        score = engine.compute_weather_score(rule_results, diagnosed_disease="Tomato___Late_blight")
        assert score == 0.0

    def test_weather_score_unmapped_disease_returns_zero(self, engine):
        """Healthy class has 0 applicable rules -> score = 0.0 per Part 4 §4a-iv."""
        data = make_hourly_series(temp=16.0, rh=95.0)
        rule_results = engine.evaluate_rules(data)
        score = engine.compute_weather_score(rule_results, diagnosed_disease="Tomato___healthy")
        assert score == 0.0

    def test_w08_never_counted_in_weather_score(self, engine):
        """Even if W08 triggers, it must NOT inflate weather_score for any disease."""
        # 15mm rain in 24h, warm dry air otherwise
        data = make_hourly_series(temp=32.0, rh=30.0, precip=15.0 / 24.0)
        rule_results = engine.evaluate_rules(data)
        assert any(r.rule_id == "W08" and r.triggered for r in rule_results)

        # For a disease without W08 applicability (e.g. Tomato Early blight, which needs temp>24 and precip_hours>2)
        # If precip_hours is 24, W04 will trigger. Let's make rain happen in 1 single hour of 15mm:
        data_1h = make_hourly_series(temp=20.0, rh=50.0, precip=0.0)
        data_1h["precipitation"][0] = 15.0
        results_1h = engine.evaluate_rules(data_1h)
        assert results_1h[7].rule_id == "W08" and results_1h[7].triggered is True

        # Late Blight applicable is W01 only (which is false here)
        score = engine.compute_weather_score(results_1h, diagnosed_disease="Tomato___Late_blight")
        assert score == 0.0

    def test_generic_weather_score_without_disease(self, engine):
        """Without diagnosed_disease, score is triggered count / 7.0 (W01–W07)."""
        # Trigger only W01 (1 of 7)
        data = make_hourly_series(temp=16.0, rh=95.0, precip=0.0, wind=5.0)
        rule_results = engine.evaluate_rules(data)
        score = engine.compute_weather_score(rule_results, diagnosed_disease=None)
        # W01 triggers, W05 might trigger if temp in 15-25 and RH>85
        # Let's verify score equals triggered_count / 7.0
        active_triggered = sum(1 for r in rule_results if r.triggered and r.rule_id in ("W01", "W02", "W03", "W04", "W05", "W06", "W07"))
        expected = round(active_triggered / 7.0, 4)
        assert score == expected
