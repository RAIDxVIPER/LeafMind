"""
tests/unit/test_weather_agent.py
================================
Full integration unit tests for WeatherAgent (Stage C4).

All HTTP requests mocked — NO live network calls.
Covers:
  - GPS coordinates path
  - Geocoding success path (resolved place name, admin region, coordinates)
  - Geocoding ambiguity path (multiple matches -> ambiguity flag set)
  - Geocoding no match path (weather_available=False, reason recorded)
  - Default region path (Maharashtra [ENG default], location_source="default")
  - Invalid coordinates (ValueError)
  - Cache hit: second call within TTL returns cached data, cache_hit=True, stale=False
  - Stale cache fallback: API failure after cache entry -> serves stale data (stale=True)
  - No cache + API failure -> weather_available=False, no fabricated numbers
  - Pydantic schema validation of output
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock
import httpx
import pytest

from src.agents.weather_agent import WeatherAgent
from src.schemas.schemas import WeatherInput, WeatherOutput


def make_mock_open_meteo_response(
    temp: float = 26.0,
    rh: float = 80.0,
    precip: float = 1.5,
    wind: float = 12.0,
    n_hours: int = 24,
) -> dict:
    """Mock Open-Meteo forecast JSON."""
    return {
        "latitude": 19.75,
        "longitude": 75.71,
        "current": {
            "temperature_2m": temp,
            "relative_humidity_2m": rh,
            "precipitation": precip,
            "wind_speed_10m": wind,
            "cloud_cover": 50.0,
            "dew_point_2m": 22.0,
        },
        "hourly": {
            "time": [f"2026-10-22T{h:02d}:00" for h in range(n_hours)],
            "temperature_2m": [temp] * n_hours,
            "relative_humidity_2m": [rh] * n_hours,
            "precipitation": [precip] * n_hours,
            "wind_speed_10m": [wind] * n_hours,
            "cloud_cover": [50.0] * n_hours,
            "dew_point_2m": [22.0] * n_hours,
        },
        "daily": {
            "time": ["2026-10-22", "2026-10-23", "2026-10-24"],
            "temperature_2m_max": [temp + 4.0] * 3,
            "temperature_2m_min": [temp - 4.0] * 3,
            "relative_humidity_2m_max": [rh + 10.0] * 3,
            "precipitation_sum": [precip * 5.0] * 3,
            "precipitation_hours": [4] * 3,
            "wind_speed_10m_max": [wind + 5.0] * 3,
        },
    }


def make_agent_with_mock_http(
    weather_json: dict | None = None,
    weather_status: int = 200,
    geocoding_json: dict | None = None,
    geocoding_status: int = 200,
) -> Tuple[WeatherAgent, MagicMock]:
    """Factory creating WeatherAgent wired to a mock HTTP client."""
    mock_client = MagicMock(spec=httpx.Client)

    def route_get(url, *args, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "geocoding-api.open-meteo.com" in str(url):
            resp.status_code = geocoding_status
            resp.json.return_value = geocoding_json or {"results": []}
            resp.text = json.dumps(geocoding_json or {})
        else:
            resp.status_code = weather_status
            resp.json.return_value = weather_json or make_mock_open_meteo_response()
            resp.text = json.dumps(weather_json or make_mock_open_meteo_response())
        return resp

    mock_client.get.side_effect = route_get
    agent = WeatherAgent(http_client=mock_client)
    return agent, mock_client


class TestWeatherAgentLocationPaths:
    """Test various location inputs to WeatherAgent."""

    def test_gps_location_path(self):
        agent, _ = make_agent_with_mock_http()
        out = agent.get_weather(gps_lat=18.5204, gps_lon=73.8567)

        assert isinstance(out, WeatherOutput)
        assert out.weather_available is True
        assert out.location_source == "gps"
        assert out.latitude == 18.5204
        assert out.longitude == 73.8567
        assert out.is_ambiguous_location is False
        assert out.ambiguity_flag is False
        assert out.current_weather is not None
        assert len(out.rules_triggered) == 8

    def test_geocoded_success_path(self):
        geo_json = {
            "results": [
                {
                    "name": "Baramati",
                    "latitude": 18.1517,
                    "longitude": 74.5774,
                    "admin1": "Maharashtra",
                    "country": "India",
                    "country_code": "IN",
                }
            ]
        }
        agent, _ = make_agent_with_mock_http(geocoding_json=geo_json)
        out = agent.get_weather(place_name="Baramati")

        assert out.weather_available is True
        assert out.location_source == "geocoded"
        assert out.location_name == "Baramati"
        assert out.location_admin == "Maharashtra"
        assert out.latitude == 18.1517
        assert out.longitude == 74.5774
        assert out.is_ambiguous_location is False

    def test_geocoded_ambiguity_path(self):
        geo_json = {
            "results": [
                {
                    "name": "Satara",
                    "latitude": 17.6805,
                    "longitude": 73.9934,
                    "admin1": "Maharashtra",
                    "country": "India",
                    "country_code": "IN",
                },
                {
                    "name": "Satara Road",
                    "latitude": 17.7833,
                    "longitude": 74.1500,
                    "admin1": "Maharashtra",
                    "country": "India",
                    "country_code": "IN",
                },
            ]
        }
        agent, _ = make_agent_with_mock_http(geocoding_json=geo_json)
        out = agent.get_weather(place_name="Satara")

        assert out.weather_available is True
        assert out.location_source == "geocoded"
        assert out.is_ambiguous_location is True
        assert out.ambiguity_flag is True
        assert out.location_name == "Satara"

    def test_geocoded_no_match_path(self):
        agent, _ = make_agent_with_mock_http(geocoding_json={"results": []})
        out = agent.get_weather(place_name="UnknownPlaceXYZ999")

        assert out.weather_available is False
        assert out.location_source == "geocoded"
        assert out.location_name == "UnknownPlaceXYZ999"
        assert "not found" in str(out.unavailable_reason).lower() or "could not be resolved" in str(out.unavailable_reason).lower()

    def test_default_region_path(self):
        agent, _ = make_agent_with_mock_http()
        out = agent.get_weather()

        assert out.weather_available is True
        assert out.location_source == "default"
        assert out.latitude == 19.7515
        assert out.longitude == 75.7139
        assert "Maharashtra" in str(out.location_name)
        assert out.is_ambiguous_location is False

    def test_invalid_coordinates_raises_value_error(self):
        agent, _ = make_agent_with_mock_http()
        with pytest.raises(ValueError, match="Invalid latitude"):
            agent.get_weather(gps_lat=95.0, gps_lon=75.0)


class TestWeatherAgentCaching:
    """Test fresh cache and stale cache fallback behavior."""

    def test_fresh_cache_hit_avoids_second_network_call(self):
        agent, mock_http = make_agent_with_mock_http()
        query_time = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)

        # Call 1: Miss -> fetches from API
        out1 = agent.get_weather(gps_lat=19.75, gps_lon=75.71, query_datetime=query_time)
        assert out1.cache_hit is False
        assert out1.stale is False
        initial_call_count = mock_http.get.call_count
        assert initial_call_count >= 1

        # Call 2: Hit within TTL -> served from cache
        out2 = agent.get_weather(gps_lat=19.75, gps_lon=75.71, query_datetime=query_time)
        assert out2.cache_hit is True
        assert out2.stale is False
        assert mock_http.get.call_count == initial_call_count  # NO second API call

    def test_stale_cache_fallback_on_api_failure(self):
        agent, mock_http = make_agent_with_mock_http(weather_status=200)
        query_time = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)

        # Call 1: Populates cache
        out1 = agent.get_weather(gps_lat=19.75, gps_lon=75.71, query_datetime=query_time)
        assert out1.weather_available is True

        # Now simulate API going down
        mock_http.get.side_effect = httpx.ConnectError("API is down")

        # Manually expire fresh TTL by modifying cache entry timestamp to be 1 hour old (< 24h max stale)
        key = agent.cache.make_key(19.75, 75.71, query_time.date().isoformat())
        agent.cache._store[key].timestamp -= 3600.0  # 1 hour ago

        # Call 2: Fresh is expired, API fails -> serves stale fallback
        out2 = agent.get_weather(gps_lat=19.75, gps_lon=75.71, query_datetime=query_time)
        assert out2.weather_available is True
        assert out2.cache_hit is True
        assert out2.cache_stale is True
        assert out2.stale is True

    def test_no_cache_and_api_failure_returns_weather_unavailable(self):
        agent, mock_http = make_agent_with_mock_http()
        mock_http.get.side_effect = httpx.ConnectTimeout("Connection timed out")

        out = agent.get_weather(gps_lat=19.75, gps_lon=75.71)
        assert out.weather_available is False
        assert out.current_weather is None
        assert out.forecast_7day is None
        assert out.rules_triggered == []
        assert "unavailable" in str(out.unavailable_reason).lower() or "timed out" in str(out.unavailable_reason).lower()


class TestWeatherAgentOutputSchema:
    """Verify output adherence to Pydantic schema and backward compatibility."""

    def test_weather_output_schema_complete(self):
        agent, _ = make_agent_with_mock_http()
        out = agent.get_weather(gps_lat=19.7515, gps_lon=75.7139, detected_disease="Tomato___Late_blight")

        # Verify all required schema fields
        assert isinstance(out.weather_available, bool)
        assert isinstance(out.stale, bool)
        assert isinstance(out.cache_stale, bool)
        assert out.stale == out.cache_stale
        assert out.latitude == 19.7515
        assert out.longitude == 75.7139
        assert out.location_source in ("gps", "geocoded", "default")
        assert out.fetched_at is not None
        assert out.weather_fetch_timestamp is not None
        assert out.data_window is not None
        assert out.summary_statistics is not None
        assert len(out.rules_triggered) == 8
        assert len(out.disease_risk_flags) == 8
        assert 0.0 <= out.weather_score <= 1.0

        # Verify [UNVERIFIED] tag on rule triggers
        for rule in out.rules_triggered:
            assert "[UNVERIFIED" in rule.tag or "[GAP" in rule.threshold_source

    def test_backward_compatibility_with_weather_input_schema(self):
        agent, _ = make_agent_with_mock_http()
        inp = WeatherInput(gps_lat=19.75, gps_lon=75.71, query_datetime=datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc))
        out = agent.get_weather(inp)
        assert out.weather_available is True
        assert out.latitude == 19.75
