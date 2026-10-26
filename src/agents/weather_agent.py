"""
src/agents/weather_agent.py
===========================
Weather Agent for LeafMind (Stage C4).

Deterministic meteorological intelligence component (no LLM).
Provides weather data fetching, geocoding, caching with stale fallback,
favourable-condition rule evaluation (W01–W08), and Part 4 weather scoring.

Key rules:
  - Deterministic code only, no LLM.
  - Location input:
      * coordinates -> use directly, location_source="gps"
      * place name -> Open-Meteo Geocoding API, India prioritized, top match + ambiguity flag
      * none -> default region (Maharashtra coordinates [ENG default]), location_source="default"
      * invalid coordinates -> reject with clear ValueError
      * no photo EXIF reading
  - Open-Meteo primary API (free, no key). OpenWeatherMap backup interface supported.
  - Endpoints and parameters in config (config/weather_config.yaml), not hardcoded.
  - Hourly variables for past 7 days and next 3 days [ENG].
  - Leaf wetness proxy: derived from hours with RH >= 90% or rain, labeled [ENG proxy].
  - Caching by (rounded lat/lon, date window) with TTL from config.
  - Failure handling: serve stale cache only if younger than max_stale_age_hours;
    otherwise return weather_available=False with clear reason. NEVER fabricate numbers.
  - Favourable-condition rules W01–W08 read from config/weather_rules.yaml;
    thresholds tagged [UNVERIFIED — I14].
  - Output schema: WeatherOutput (backward compatible with existing schemas).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import httpx
import yaml

from src.schemas.schemas import (
    CurrentWeather,
    DailyForecast,
    RuleTriggerResult,
    WeatherInput,
    WeatherOutput,
    WeatherRiskFlag,
    WeatherSummaryStats,
    WeatherWindow,
)
from src.weather.cache import WeatherCache
from src.weather.client import WeatherClient, WeatherFetchError
from src.weather.geocoding import GeocodingClient, ResolvedLocation
from src.weather.rules import WeatherRuleEngine

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEATHER_CONFIG = REPO_ROOT / "config" / "weather_config.yaml"
DEFAULT_RULES_CONFIG = REPO_ROOT / "config" / "weather_rules.yaml"


class WeatherAgent:
    """Weather Agent coordinating geocoding, API retrieval, caching, and rule evaluation."""

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        rules_path: Optional[Path | str] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        self.config_path = Path(config_path) if config_path else DEFAULT_WEATHER_CONFIG
        self.rules_path = Path(rules_path) if rules_path else DEFAULT_RULES_CONFIG

        # Load weather configuration
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            logger.warning("Weather config %s not found; using defaults", self.config_path)
            self.config = {}

        # Initialize sub-components
        self.geocoding_client = GeocodingClient(config_path=self.config_path, http_client=http_client)
        self.weather_client = WeatherClient(config_path=self.config_path, http_client=http_client)
        self.rule_engine = WeatherRuleEngine(rules_path=self.rules_path)

        # Cache setup
        cache_cfg = self.config.get("cache", {})
        ttl = float(cache_cfg.get("ttl_seconds", 1800.0))
        max_stale = float(cache_cfg.get("max_stale_age_hours", 24.0))
        max_entries = int(cache_cfg.get("max_entries", 1000))
        rounding = int(cache_cfg.get("coord_rounding_decimals", 2))
        self.cache = WeatherCache(
            ttl_seconds=ttl,
            max_stale_age_hours=max_stale,
            max_entries=max_entries,
            coord_rounding=rounding,
        )

        # Leaf-wetness proxy configuration [ENG proxy]
        lw_cfg = self.config.get("leaf_wetness_proxy", {})
        self.rh_threshold_pct = float(lw_cfg.get("rh_threshold_pct", 90.0))
        self.precip_threshold_mm = float(lw_cfg.get("precip_threshold_mm", 0.1))

    def get_weather(
        self,
        weather_input: Optional[Union[WeatherInput, Dict[str, Any]]] = None,
        gps_lat: Optional[float] = None,
        gps_lon: Optional[float] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        place_name: Optional[str] = None,
        query_datetime: Optional[datetime] = None,
        detected_disease: Optional[str] = None,
    ) -> WeatherOutput:
        """
        Main entry point for Weather Agent.

        Accepts either a WeatherInput instance or direct kwargs.
        Returns a validated WeatherOutput object.
        """
        # 1. Normalize input arguments
        if weather_input is not None:
            if isinstance(weather_input, dict):
                weather_input = WeatherInput(**weather_input)
            lat = weather_input.latitude if weather_input.latitude is not None else weather_input.gps_lat
            lon = weather_input.longitude if weather_input.longitude is not None else weather_input.gps_lon
            p_name = weather_input.place_name
            dt = weather_input.query_datetime or query_datetime
            disease = weather_input.detected_disease or detected_disease
        else:
            lat = latitude if latitude is not None else gps_lat
            lon = longitude if longitude is not None else gps_lon
            p_name = place_name
            dt = query_datetime
            disease = detected_disease

        now_utc = dt or datetime.now(timezone.utc)
        date_str = now_utc.date().isoformat()

        # 2. Resolve location (GPS, Geocoded, or Default region)
        resolved: ResolvedLocation = self.geocoding_client.resolve(
            latitude=lat,
            longitude=lon,
            place_name=p_name,
        )

        # Handle geocoding lookup failure
        if not resolved.is_valid:
            logger.warning("Location resolution failed: %s", resolved.error)
            return WeatherOutput(
                weather_available=False,
                location_source=resolved.location_source,
                location_name=resolved.location_name,
                is_ambiguous_location=resolved.is_ambiguous,
                ambiguity_flag=resolved.is_ambiguous,
                weather_fetch_timestamp=now_utc,
                fetched_at=now_utc,
                unavailable_reason=resolved.error or "Location resolution failed",
            )

        assert resolved.latitude is not None and resolved.longitude is not None
        cur_lat = resolved.latitude
        cur_lon = resolved.longitude

        # 3. Cache lookup (Fresh hit)
        date_window_key = f"{date_str}"
        cached_raw = self.cache.get_fresh(cur_lat, cur_lon, date_window_key)
        if cached_raw is not None:
            logger.info("Serving FRESH cached weather for (%.2f, %.2f)", cur_lat, cur_lon)
            return self._build_weather_output_from_cached(
                cached_raw=cached_raw,
                resolved=resolved,
                fetched_at=now_utc,
                is_stale=False,
                is_cache_hit=True,
                diagnosed_disease=disease,
            )

        # 4. Fetch from Weather API (Open-Meteo primary)
        try:
            fetched_data = self.weather_client.fetch_weather(cur_lat, cur_lon)

            # Process fresh data
            output = self._process_weather_data(
                fetched_data=fetched_data,
                resolved=resolved,
                fetched_at=now_utc,
                is_stale=False,
                is_cache_hit=False,
                diagnosed_disease=disease,
                query_datetime=query_datetime,
            )

            # Store in cache
            self.cache.put(cur_lat, cur_lon, date_window_key, output.model_dump())
            return output

        except Exception as fetch_err:
            logger.warning("Weather fetch failed for (%.2f, %.2f): %s", cur_lat, cur_lon, fetch_err)

            # 5. Stale cache fallback (if entry <= max_stale_age_hours)
            stale_raw = self.cache.get_stale(cur_lat, cur_lon, date_window_key)
            if stale_raw is not None:
                logger.warning("Serving STALE cached weather for (%.2f, %.2f)", cur_lat, cur_lon)
                return self._build_weather_output_from_cached(
                    cached_raw=stale_raw,
                    resolved=resolved,
                    fetched_at=now_utc,
                    is_stale=True,
                    is_cache_hit=True,
                    diagnosed_disease=disease,
                )

            # 6. Unrecoverable outage: return WeatherUnavailable (never fabricate numbers)
            logger.error("No valid or stale weather data available for (%.2f, %.2f)", cur_lat, cur_lon)
            return WeatherOutput(
                weather_available=False,
                latitude=cur_lat,
                longitude=cur_lon,
                location_source=resolved.location_source,
                location_name=resolved.location_name,
                location_admin=resolved.admin_region,
                is_ambiguous_location=resolved.is_ambiguous,
                ambiguity_flag=resolved.is_ambiguous,
                weather_fetch_timestamp=now_utc,
                fetched_at=now_utc,
                unavailable_reason=f"Weather APIs unavailable: {fetch_err}",
            )

    def _process_weather_data(
        self,
        fetched_data: Dict[str, Any],
        resolved: ResolvedLocation,
        fetched_at: datetime,
        is_stale: bool,
        is_cache_hit: bool,
        diagnosed_disease: Optional[str] = None,
        query_datetime: Optional[datetime] = None,
    ) -> WeatherOutput:
        """Evaluate rules, summary statistics, and construct final WeatherOutput."""
        hourly = fetched_data.get("hourly_data") or {}
        raw_resp = fetched_data.get("raw_response", {})
        current_block = raw_resp.get("current", {}) if isinstance(raw_resp, dict) else {}
        current_time_str = current_block.get("time")

        # Compute summary statistics on historical past window (excluding future forecast)
        stats: WeatherSummaryStats = self.rule_engine.compute_summary_statistics(
            hourly_data=hourly,
            rh_threshold_pct=self.rh_threshold_pct,
            precip_threshold_mm=self.precip_threshold_mm,
            current_time_str=current_time_str,
            query_datetime=query_datetime,
        )

        # Evaluate disease-favourable condition rules W01–W08 over past 24h ending at current time
        rule_results: List[RuleTriggerResult] = self.rule_engine.evaluate_rules(
            hourly_data=hourly,
            summary_stats=stats,
            rh_threshold_pct=self.rh_threshold_pct,
            precip_threshold_mm=self.precip_threshold_mm,
            current_time_str=current_time_str,
            query_datetime=query_datetime,
        )

        # Legacy risk flags
        legacy_flags: List[WeatherRiskFlag] = self.rule_engine.to_legacy_risk_flags(rule_results)

        # Compute weather_score per Part 4 formula
        w_score = self.rule_engine.compute_weather_score(
            rule_results=rule_results,
            diagnosed_disease=diagnosed_disease,
        )

        return WeatherOutput(
            weather_available=True,
            weather_partial=False,
            cache_hit=is_cache_hit,
            cache_stale=is_stale,
            stale=is_stale,
            latitude=resolved.latitude,
            longitude=resolved.longitude,
            location_name=resolved.location_name,
            location_admin=resolved.admin_region,
            location_source=resolved.location_source,
            is_ambiguous_location=resolved.is_ambiguous,
            ambiguity_flag=resolved.is_ambiguous,
            weather_fetch_timestamp=fetched_at,
            fetched_at=fetched_at,
            data_window=fetched_data.get("data_window"),
            summary_statistics=stats,
            current_weather=fetched_data.get("current_weather"),
            forecast_7day=fetched_data.get("forecast_7day"),
            disease_risk_flags=legacy_flags,
            rules_triggered=rule_results,
            weather_score=w_score,
            weather_data_source=fetched_data.get("provider", "open-meteo"),
            unavailable_reason=None,
        )

    def _build_weather_output_from_cached(
        self,
        cached_raw: Dict[str, Any],
        resolved: ResolvedLocation,
        fetched_at: datetime,
        is_stale: bool,
        is_cache_hit: bool,
        diagnosed_disease: Optional[str] = None,
    ) -> WeatherOutput:
        """Reconstruct WeatherOutput from cached dictionary."""
        # Update metadata fields
        cached_raw["cache_hit"] = is_cache_hit
        cached_raw["cache_stale"] = is_stale
        cached_raw["stale"] = is_stale
        cached_raw["location_source"] = resolved.location_source
        if resolved.location_name:
            cached_raw["location_name"] = resolved.location_name
        cached_raw["is_ambiguous_location"] = resolved.is_ambiguous
        cached_raw["ambiguity_flag"] = resolved.is_ambiguous

        # If a specific disease was requested, re-evaluate weather_score
        if diagnosed_disease:
            raw_rules = cached_raw.get("rules_triggered", [])
            rule_objs = [
                RuleTriggerResult(**r) if isinstance(r, dict) else r
                for r in raw_rules
            ]
            cached_raw["weather_score"] = self.rule_engine.compute_weather_score(
                rule_objs, diagnosed_disease=diagnosed_disease
            )

        return WeatherOutput(**cached_raw)
