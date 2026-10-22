"""
tests/unit/test_weather_geocoding.py
====================================
Unit tests for location resolution and geocoding in Weather Agent (Stage C4).

All HTTP requests mocked — NO live network calls.
Covers:
  - GPS coordinates path (direct usage, location_source="gps")
  - Geocoding success (Open-Meteo API, India prioritized, location_source="geocoded")
  - Geocoding no match (handled explicitly, error recorded)
  - Geocoding multiple matches (top match returned + ambiguity flag)
  - Default region path (no coords/place name -> Maharashtra [ENG default])
  - Invalid coordinates validation (out of bounds, incomplete -> ValueError)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock
import httpx
import pytest

from src.weather.geocoding import GeocodingClient, ResolvedLocation


def make_mock_client(status_code: int = 200, json_data: dict | None = None) -> httpx.Client:
    """Create a mock httpx.Client returning preset response."""
    client = MagicMock(spec=httpx.Client)
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = json.dumps(json_data or {})
    client.get.return_value = resp
    return client


class TestGeocodingDirectCoords:
    """Test direct coordinate resolution (GPS path)."""

    def test_gps_valid_coordinates(self):
        client = GeocodingClient()
        res = client.resolve(latitude=19.75, longitude=75.71)
        assert res.is_valid is True
        assert res.latitude == 19.75
        assert res.longitude == 75.71
        assert res.location_source == "gps"
        assert res.is_ambiguous is False
        assert res.error is None

    def test_gps_lat_lon_aliases(self):
        client = GeocodingClient()
        res = client.resolve(gps_lat=18.52, gps_lon=73.85)
        assert res.is_valid is True
        assert res.latitude == 18.52
        assert res.longitude == 73.85
        assert res.location_source == "gps"

    def test_gps_bounds_validation_errors(self):
        client = GeocodingClient()
        # Latitude out of bounds
        with pytest.raises(ValueError, match="Invalid latitude"):
            client.resolve(latitude=91.0, longitude=75.0)

        with pytest.raises(ValueError, match="Invalid latitude"):
            client.resolve(latitude=-90.1, longitude=75.0)

        # Longitude out of bounds
        with pytest.raises(ValueError, match="Invalid longitude"):
            client.resolve(latitude=19.0, longitude=180.1)

        with pytest.raises(ValueError, match="Invalid longitude"):
            client.resolve(latitude=19.0, longitude=-180.5)

    def test_incomplete_coordinates_rejected(self):
        client = GeocodingClient()
        with pytest.raises(ValueError, match="Incomplete coordinates"):
            client.resolve(latitude=19.0, longitude=None)

        with pytest.raises(ValueError, match="Incomplete coordinates"):
            client.resolve(latitude=None, longitude=75.0)


class TestGeocodingPlaceName:
    """Test place name resolution via Open-Meteo Geocoding API."""

    def test_geocoding_success_india_match(self):
        mock_data = {
            "results": [
                {
                    "name": "Nashik",
                    "latitude": 19.9975,
                    "longitude": 73.7898,
                    "admin1": "Maharashtra",
                    "country": "India",
                    "country_code": "IN",
                }
            ]
        }
        mock_http = make_mock_client(200, mock_data)
        client = GeocodingClient(http_client=mock_http)

        res = client.resolve(place_name="Nashik")
        assert res.is_valid is True
        assert res.latitude == 19.9975
        assert res.longitude == 73.7898
        assert res.location_name == "Nashik"
        assert res.admin_region == "Maharashtra"
        assert res.country == "India"
        assert res.location_source == "geocoded"
        assert res.is_ambiguous is False

    def test_geocoding_multiple_matches_sets_ambiguity_flag(self):
        mock_data = {
            "results": [
                {
                    "name": "Pune",
                    "latitude": 18.5204,
                    "longitude": 73.8567,
                    "admin1": "Maharashtra",
                    "country": "India",
                    "country_code": "IN",
                },
                {
                    "name": "Pune Cantonment",
                    "latitude": 18.5089,
                    "longitude": 73.8786,
                    "admin1": "Maharashtra",
                    "country": "India",
                    "country_code": "IN",
                },
            ]
        }
        mock_http = make_mock_client(200, mock_data)
        client = GeocodingClient(http_client=mock_http)

        res = client.resolve(place_name="Pune")
        assert res.is_valid is True
        assert res.latitude == 18.5204
        assert res.location_name == "Pune"
        assert res.location_source == "geocoded"
        assert res.is_ambiguous is True
        assert len(res.ambiguity_matches) == 2

    def test_geocoding_prioritizes_india_over_international(self):
        mock_data = {
            "results": [
                {
                    "name": "Delhi",
                    "latitude": 42.2781,
                    "longitude": -74.9157,
                    "admin1": "New York",
                    "country": "United States",
                    "country_code": "US",
                },
                {
                    "name": "Delhi",
                    "latitude": 28.6519,
                    "longitude": 77.2315,
                    "admin1": "Delhi",
                    "country": "India",
                    "country_code": "IN",
                },
            ]
        }
        mock_http = make_mock_client(200, mock_data)
        client = GeocodingClient(http_client=mock_http)

        res = client.resolve(place_name="Delhi")
        assert res.is_valid is True
        # Must pick India match
        assert res.latitude == 28.6519
        assert res.country == "India"
        assert res.admin_region == "Delhi"

    def test_geocoding_no_match_returns_error(self):
        mock_data = {"results": []}
        mock_http = make_mock_client(200, mock_data)
        client = GeocodingClient(http_client=mock_http)

        res = client.resolve(place_name="NonExistentVillage12345")
        assert res.is_valid is False
        assert res.latitude is None
        assert res.longitude is None
        assert res.location_source == "geocoded"
        assert "not found" in str(res.error).lower() or "could not be resolved" in str(res.error).lower()

    def test_geocoding_api_http_error(self):
        mock_http = make_mock_client(500, {})
        client = GeocodingClient(http_client=mock_http)

        res = client.resolve(place_name="Solapur")
        assert res.is_valid is False
        assert "http error" in str(res.error).lower()


class TestGeocodingDefaultRegion:
    """Test default region fallback when no coordinates or place name are given."""

    def test_default_region_when_empty(self):
        client = GeocodingClient()
        res = client.resolve()
        assert res.is_valid is True
        assert res.location_source == "default"
        assert res.latitude == 19.7515  # Maharashtra default [ENG default]
        assert res.longitude == 75.7139
        assert "Maharashtra" in str(res.location_name)
        assert res.is_ambiguous is False

    def test_default_region_when_empty_strings(self):
        client = GeocodingClient()
        res = client.resolve(place_name="   ")
        assert res.location_source == "default"
        assert res.latitude == 19.7515
