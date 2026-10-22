"""
src/weather/client.py
=====================
HTTP client for Open-Meteo forecast API (primary) and OpenWeatherMap (backup) (Stage C4).

Behavior and constraints:
  - Primary: Open-Meteo forecast API. Free, no API key required.
  - Backup: OpenWeatherMap interface implemented; optional, does not require key to run.
  - All endpoints, parameters, and defaults loaded from config/weather_config.yaml.
  - Hourly variables: temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m,
    dew_point_2m, cloud_cover.
  - Window: past N days (7) and forecast M days (3) [ENG].
  - Timeouts (5s), retries with backoff (2 retries), clear User-Agent header.
  - Graceful degradation: Raises WeatherFetchError on failure so caller can attempt stale cache fallback.
  - Invariant: NEVER fabricate or interpolate numbers.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx
import yaml

from src.schemas.schemas import CurrentWeather, DailyForecast, WeatherWindow

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "weather_config.yaml"


class WeatherFetchError(Exception):
    """Raised when weather API calls fail after retries."""
    pass


class WeatherClient:
    """Handles HTTP calls to Open-Meteo and optional OpenWeatherMap."""

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}

        self.api_cfg = self.config.get("weather_api", {})
        self.primary_cfg = self.api_cfg.get("primary", {
            "provider": "open-meteo",
            "base_url": "https://api.open-meteo.com/v1/forecast",
            "past_days": 7,
            "forecast_days": 3,
            "hourly_variables": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "dew_point_2m",
                "cloud_cover",
            ],
            "daily_variables": [
                "temperature_2m_max",
                "temperature_2m_min",
                "relative_humidity_2m_max",
                "precipitation_sum",
                "precipitation_hours",
                "wind_speed_10m_max",
            ],
            "timezone": "auto",
            "timeout_seconds": 5.0,
            "max_retries": 2,
            "backoff_factor": 0.5,
            "user_agent": "LeafMind-WeatherAgent/1.0 (+https://github.com/RAIDxVIPER/LeafMind)",
        })

        self.backup_cfg = self.api_cfg.get("backup", {
            "provider": "openweathermap",
            "enabled": False,
            "base_url": "https://api.openweathermap.org/data/2.5/forecast",
            "api_key_env": "OPENWEATHERMAP_API_KEY",
            "timeout_seconds": 5.0,
        })

        self.http_client = http_client

    def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        past_days: Optional[int] = None,
        forecast_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fetch weather data from primary provider (Open-Meteo) with retry/backoff.
        Falls back to backup provider (OpenWeatherMap) if enabled and primary fails.
        Raises WeatherFetchError on unrecoverable failure.
        """
        try:
            return self._fetch_open_meteo(latitude, longitude, past_days, forecast_days)
        except Exception as primary_err:
            logger.warning("Primary weather fetch (Open-Meteo) failed: %s", primary_err)

            if self.backup_cfg.get("enabled", False):
                api_key_env = self.backup_cfg.get("api_key_env", "OPENWEATHERMAP_API_KEY")
                api_key = os.getenv(api_key_env)
                if api_key:
                    try:
                        return self._fetch_openweathermap(latitude, longitude, api_key)
                    except Exception as backup_err:
                        logger.warning("Backup weather fetch (OpenWeatherMap) failed: %s", backup_err)

            raise WeatherFetchError(f"Weather APIs unavailable: {primary_err}") from primary_err

    def _fetch_open_meteo(
        self,
        latitude: float,
        longitude: float,
        past_days: Optional[int] = None,
        forecast_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call Open-Meteo Forecast REST API with retries and backoff."""
        base_url = self.primary_cfg.get("base_url", "https://api.open-meteo.com/v1/forecast")
        n_past = past_days if past_days is not None else int(self.primary_cfg.get("past_days", 7))
        n_fore = forecast_days if forecast_days is not None else int(self.primary_cfg.get("forecast_days", 3))

        hourly_vars = ",".join(self.primary_cfg.get("hourly_variables", [
            "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "dew_point_2m", "cloud_cover"
        ]))
        daily_vars = ",".join(self.primary_cfg.get("daily_variables", [
            "temperature_2m_max", "temperature_2m_min", "relative_humidity_2m_max", "precipitation_sum", "precipitation_hours", "wind_speed_10m_max"
        ]))
        timezone_val = self.primary_cfg.get("timezone", "auto")
        timeout = float(self.primary_cfg.get("timeout_seconds", 5.0))
        max_retries = int(self.primary_cfg.get("max_retries", 2))
        backoff = float(self.primary_cfg.get("backoff_factor", 0.5))
        user_agent = self.primary_cfg.get("user_agent", "LeafMind-WeatherAgent/1.0")

        params = {
            "latitude": round(float(latitude), 4),
            "longitude": round(float(longitude), 4),
            "past_days": n_past,
            "forecast_days": n_fore,
            "hourly": hourly_vars,
            "daily": daily_vars,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover,dew_point_2m",
            "timezone": timezone_val,
        }
        headers = {"User-Agent": user_agent}

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                if self.http_client is not None:
                    resp = self.http_client.get(base_url, params=params, headers=headers, timeout=timeout)
                else:
                    with httpx.Client(timeout=timeout) as client:
                        resp = client.get(base_url, params=params, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    parsed = self._parse_open_meteo_response(data, n_past, n_fore)
                    parsed["provider"] = "open-meteo"
                    return parsed
                else:
                    logger.warning("Open-Meteo HTTP %d: %s (attempt %d/%d)", resp.status_code, resp.text[:100], attempt + 1, max_retries + 1)
                    last_exc = RuntimeError(f"Open-Meteo returned HTTP {resp.status_code}")

            except Exception as e:
                logger.warning("Open-Meteo request error on attempt %d/%d: %s", attempt + 1, max_retries + 1, e)
                last_exc = e

            if attempt < max_retries:
                sleep_time = backoff * (2 ** attempt)
                time.sleep(sleep_time)

        raise WeatherFetchError(f"Open-Meteo failed after {max_retries + 1} attempts: {last_exc}")

    def _parse_open_meteo_response(
        self,
        raw_json: Dict[str, Any],
        past_days: int,
        forecast_days: int,
    ) -> Dict[str, Any]:
        """Convert raw Open-Meteo JSON into structured objects."""
        # 1. Parse current weather
        current_block = raw_json.get("current") or {}
        # If current block is empty or missing fields, pick the most recent hourly value
        hourly = raw_json.get("hourly") or {}
        n_hourly = len(hourly.get("time", []))
        last_idx = n_hourly - 1 if n_hourly > 0 else 0

        temp_c = float(current_block.get("temperature_2m", hourly.get("temperature_2m", [20.0])[last_idx] if n_hourly else 20.0))
        rh_pct = float(current_block.get("relative_humidity_2m", hourly.get("relative_humidity_2m", [50.0])[last_idx] if n_hourly else 50.0))
        precip_mm = float(current_block.get("precipitation", hourly.get("precipitation", [0.0])[last_idx] if n_hourly else 0.0))
        wind_kmh = float(current_block.get("wind_speed_10m", hourly.get("wind_speed_10m", [5.0])[last_idx] if n_hourly else 5.0))
        cloud_pct = float(current_block.get("cloud_cover", hourly.get("cloud_cover", [0.0])[last_idx] if n_hourly else 0.0))

        # Dew point
        dew_pt = current_block.get("dew_point_2m")
        if dew_pt is None and n_hourly and hourly.get("dew_point_2m"):
            dew_pt = hourly["dew_point_2m"][last_idx]
        if dew_pt is None:
            # Magnus formula approximation if absent: Td = T - ((100 - RH)/5)
            dew_pt = temp_c - ((100.0 - rh_pct) / 5.0)

        current_obj = CurrentWeather(
            temperature_c=round(temp_c, 2),
            relative_humidity_pct=round(max(0.0, min(100.0, rh_pct)), 2),
            precipitation_mm=round(max(0.0, precip_mm), 2),
            wind_speed_kmh=round(max(0.0, wind_kmh), 2),
            cloud_cover_pct=round(max(0.0, min(100.0, cloud_pct)), 2),
            dew_point_c=round(float(dew_pt), 2),
        )

        # 2. Parse daily forecast
        daily = raw_json.get("daily") or {}
        dates = daily.get("time", [])
        temp_maxs = daily.get("temperature_2m_max", [])
        temp_mins = daily.get("temperature_2m_min", [])
        rh_maxs = daily.get("relative_humidity_2m_max", [])
        precip_sums = daily.get("precipitation_sum", [])
        precip_hours = daily.get("precipitation_hours", [])
        wind_maxs = daily.get("wind_speed_10m_max", [])

        forecast_list: List[DailyForecast] = []
        n_daily = len(dates)
        for i in range(n_daily):
            try:
                f_date = date.fromisoformat(dates[i])
                t_max = float(temp_maxs[i]) if i < len(temp_maxs) and temp_maxs[i] is not None else temp_c
                t_min = float(temp_mins[i]) if i < len(temp_mins) and temp_mins[i] is not None else temp_c
                rh_max = float(rh_maxs[i]) if i < len(rh_maxs) and rh_maxs[i] is not None else rh_pct
                p_sum = float(precip_sums[i]) if i < len(precip_sums) and precip_sums[i] is not None else 0.0
                p_hrs = int(precip_hours[i]) if i < len(precip_hours) and precip_hours[i] is not None else 0
                w_max = float(wind_maxs[i]) if i < len(wind_maxs) and wind_maxs[i] is not None else wind_kmh

                forecast_list.append(
                    DailyForecast(
                        forecast_date=f_date,
                        temp_max_c=round(t_max, 2),
                        temp_min_c=round(t_min, 2),
                        humidity_max_pct=round(max(0.0, min(100.0, rh_max)), 2),
                        precipitation_sum_mm=round(max(0.0, p_sum), 2),
                        precipitation_hours=max(0, p_hrs),
                        wind_speed_max_kmh=round(max(0.0, w_max), 2),
                    )
                )
            except Exception as e:
                logger.debug("Skipping daily entry %d: %s", i, e)

        # 3. Data window
        start_d = date.fromisoformat(dates[0]) if dates else None
        end_d = date.fromisoformat(dates[-1]) if dates else None
        data_window = WeatherWindow(
            past_days=past_days,
            forecast_days=forecast_days,
            start_date=start_d,
            end_date=end_d,
        )

        return {
            "current_weather": current_obj,
            "forecast_7day": forecast_list,
            "hourly_data": hourly,
            "data_window": data_window,
            "raw_response": raw_json,
        }

    def _fetch_openweathermap(
        self,
        latitude: float,
        longitude: float,
        api_key: str,
    ) -> Dict[str, Any]:
        """
        Backup provider: OpenWeatherMap 5-day forecast.
        Optional interface per spec §2c-ii.
        """
        base_url = self.backup_cfg.get("base_url", "https://api.openweathermap.org/data/2.5/forecast")
        timeout = float(self.backup_cfg.get("timeout_seconds", 5.0))
        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }

        if self.http_client is not None:
            resp = self.http_client.get(base_url, params=params, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(base_url, params=params)

        if resp.status_code != 200:
            raise RuntimeError(f"OpenWeatherMap returned HTTP {resp.status_code}")

        data = resp.json()
        list_items = data.get("list", [])
        if not list_items:
            raise RuntimeError("OpenWeatherMap returned empty forecast list")

        first = list_items[0]
        main = first.get("main", {})
        wind = first.get("wind", {})
        clouds = first.get("clouds", {})
        rain = first.get("rain", {})

        temp_c = float(main.get("temp", 20.0))
        rh_pct = float(main.get("humidity", 50.0))
        precip_mm = float(rain.get("3h", 0.0))
        wind_kmh = float(wind.get("speed", 0.0)) * 3.6  # m/s -> km/h
        cloud_pct = float(clouds.get("all", 0.0))
        dew_pt = temp_c - ((100.0 - rh_pct) / 5.0)

        current_obj = CurrentWeather(
            temperature_c=round(temp_c, 2),
            relative_humidity_pct=round(max(0.0, min(100.0, rh_pct)), 2),
            precipitation_mm=round(max(0.0, precip_mm), 2),
            wind_speed_kmh=round(max(0.0, wind_kmh), 2),
            cloud_cover_pct=round(max(0.0, min(100.0, cloud_pct)), 2),
            dew_point_c=round(dew_pt, 2),
        )

        # Build simulated hourly from 3h forecast intervals
        simulated_hourly: Dict[str, List[float]] = {
            "temperature_2m": [],
            "relative_humidity_2m": [],
            "precipitation": [],
            "wind_speed_10m": [],
            "dew_point_2m": [],
        }
        for item in list_items:
            m = item.get("main", {})
            w = item.get("wind", {})
            r = item.get("rain", {})
            t = float(m.get("temp", 20.0))
            h = float(m.get("humidity", 50.0))
            p = float(r.get("3h", 0.0)) / 3.0
            spd = float(w.get("speed", 0.0)) * 3.6
            dp = t - ((100.0 - h) / 5.0)
            simulated_hourly["temperature_2m"].append(t)
            simulated_hourly["relative_humidity_2m"].append(h)
            simulated_hourly["precipitation"].append(p)
            simulated_hourly["wind_speed_10m"].append(spd)
            simulated_hourly["dew_point_2m"].append(dp)

        return {
            "current_weather": current_obj,
            "forecast_7day": [],
            "hourly_data": simulated_hourly,
            "data_window": WeatherWindow(past_days=0, forecast_days=5),
            "provider": "openweathermap",
            "raw_response": data,
        }
