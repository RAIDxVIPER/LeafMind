"""
src/weather/geocoding.py
========================
Location input resolution and geocoding for Weather Agent (Stage C4).

Rules and behavior:
  1. GPS Coordinates: If latitude/longitude are provided, validate ranges
     (lat: -90..90, lon: -180..180). Reject invalid coordinates with ValueError;
     never guess or interpolate. location_source="gps".
  2. Place Name: Resolve with Open-Meteo Geocoding API (free, no key required).
     Prioritize matches in India by default. Return resolved coordinates, name,
     and admin region. Handle no-match explicitly (error recorded) and multiple
     matches (top match returned + ambiguity flag). location_source="geocoded".
  3. Default Region: If no coordinates and no place name given, use configured
     default region (e.g. Maharashtra coordinates [ENG default]). location_source="default".
  4. Photo EXIF: Never read or parse photo EXIF.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "weather_config.yaml"


@dataclass
class ResolvedLocation:
    """Resolved geographic location metadata."""
    latitude: Optional[float]
    longitude: Optional[float]
    location_name: Optional[str] = None
    admin_region: Optional[str] = None
    country: Optional[str] = None
    location_source: str = "default"  # "gps" | "geocoded" | "default"
    is_ambiguous: bool = False
    ambiguity_matches: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.latitude is not None
            and self.longitude is not None
            and -90.0 <= self.latitude <= 90.0
            and -180.0 <= self.longitude <= 180.0
            and self.error is None
        )


class GeocodingClient:
    """Resolves coordinates, place names, or default region per LeafMind spec."""

    def __init__(self, config_path: Optional[Path | str] = None, http_client: Optional[httpx.Client] = None):
        cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}

        self.default_loc = self.config.get("default_location", {
            "name": "Maharashtra (default)",
            "admin_region": "Maharashtra",
            "country": "India",
            "country_code": "IN",
            "latitude": 19.7515,
            "longitude": 75.7139,
            "note": "[ENG default] Default region coordinates",
        })

        self.geo_cfg = self.config.get("geocoding", {
            "base_url": "https://geocoding-api.open-meteo.com/v1/search",
            "count": 10,
            "language": "en",
            "default_country_code": "IN",
            "timeout_seconds": 5.0,
            "max_retries": 2,
            "backoff_factor": 0.5,
            "user_agent": "LeafMind-WeatherAgent/1.0 (+https://github.com/RAIDxVIPER/LeafMind)",
        })

        self.http_client = http_client

    def resolve(
        self,
        gps_lat: Optional[float] = None,
        gps_lon: Optional[float] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        place_name: Optional[str] = None,
    ) -> ResolvedLocation:
        """
        Resolve location from GPS coordinates, place name, or default region.

        Priority:
          1. Direct coordinates (gps_lat/gps_lon or latitude/longitude)
          2. Place name (geocoded via Open-Meteo)
          3. Default region from config [ENG default]
        """
        lat = latitude if latitude is not None else gps_lat
        lon = longitude if longitude is not None else gps_lon

        # Case 1: Coordinates provided
        if lat is not None or lon is not None:
            if lat is None or lon is None:
                raise ValueError(f"Incomplete coordinates: lat={lat}, lon={lon}. Both latitude and longitude are required.")
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(f"Invalid latitude: {lat}. Latitude must be between -90.0 and 90.0.")
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(f"Invalid longitude: {lon}. Longitude must be between -180.0 and 180.0.")

            return ResolvedLocation(
                latitude=float(lat),
                longitude=float(lon),
                location_name=place_name.strip() if place_name and place_name.strip() else None,
                admin_region=None,
                country=None,
                location_source="gps",
                is_ambiguous=False,
            )

        # Case 2: Place name provided
        if place_name is not None and place_name.strip():
            clean_name = place_name.strip()
            return self._geocode_place_name(clean_name)

        # Case 3: Nothing provided -> Default region
        logger.info(
            "No location provided; using default region: %s (%.4f, %.4f) [ENG default]",
            self.default_loc.get("name"),
            self.default_loc["latitude"],
            self.default_loc["longitude"],
        )
        return ResolvedLocation(
            latitude=float(self.default_loc["latitude"]),
            longitude=float(self.default_loc["longitude"]),
            location_name=self.default_loc.get("name"),
            admin_region=self.default_loc.get("admin_region"),
            country=self.default_loc.get("country"),
            location_source="default",
            is_ambiguous=False,
        )

    def _geocode_place_name(self, place_name: str) -> ResolvedLocation:
        """Call Open-Meteo Geocoding API to resolve place name."""
        base_url = self.geo_cfg.get("base_url", "https://geocoding-api.open-meteo.com/v1/search")
        timeout = float(self.geo_cfg.get("timeout_seconds", 5.0))
        user_agent = self.geo_cfg.get("user_agent", "LeafMind-WeatherAgent/1.0")
        default_cc = self.geo_cfg.get("default_country_code", "IN").upper()
        count = int(self.geo_cfg.get("count", 10))
        lang = self.geo_cfg.get("language", "en")

        params = {
            "name": place_name,
            "count": count,
            "language": lang,
            "format": "json",
        }
        headers = {"User-Agent": user_agent}

        try:
            if self.http_client is not None:
                resp = self.http_client.get(base_url, params=params, headers=headers, timeout=timeout)
            else:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.get(base_url, params=params, headers=headers)

            if resp.status_code != 200:
                logger.warning("Geocoding API HTTP %d for '%s'", resp.status_code, place_name)
                return ResolvedLocation(
                    latitude=None,
                    longitude=None,
                    location_name=place_name,
                    location_source="geocoded",
                    error=f"Geocoding API HTTP error: {resp.status_code}",
                )

            data = resp.json()
            results = data.get("results") or []

            if not results:
                logger.info("Geocoding returned 0 matches for '%s'", place_name)
                return ResolvedLocation(
                    latitude=None,
                    longitude=None,
                    location_name=place_name,
                    location_source="geocoded",
                    error=f"Place name '{place_name}' could not be resolved by geocoding API.",
                )

            # Prioritize default country (India) matches
            in_matches = [
                r for r in results
                if str(r.get("country_code", "")).upper() == default_cc
                or str(r.get("country", "")).lower() == "india"
            ]

            candidates = in_matches if in_matches else results
            top_match = candidates[0]
            is_ambiguous = len(candidates) > 1

            ambiguity_descriptions = [
                f"{r.get('name')}, {r.get('admin1', '')}, {r.get('country', '')} ({r.get('latitude')}, {r.get('longitude')})"
                for r in candidates[:5]
            ]

            return ResolvedLocation(
                latitude=float(top_match["latitude"]),
                longitude=float(top_match["longitude"]),
                location_name=top_match.get("name", place_name),
                admin_region=top_match.get("admin1"),
                country=top_match.get("country"),
                location_source="geocoded",
                is_ambiguous=is_ambiguous,
                ambiguity_matches=ambiguity_descriptions,
            )

        except Exception as e:
            logger.error("Geocoding lookup failed for '%s': %s", place_name, str(e))
            return ResolvedLocation(
                latitude=None,
                longitude=None,
                location_name=place_name,
                location_source="geocoded",
                error=f"Geocoding network error: {str(e)}",
            )
