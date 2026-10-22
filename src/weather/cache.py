"""
src/weather/cache.py
====================
In-memory caching with TTL and stale-fallback policy for Weather Agent (Stage C4).

Behavior per spec §2c-iii and §2c-iv:
  1. Cache key: (round(lat, 2), round(lon, 2), date_window)
     Rounding to 2 decimal places ≈ 1.1 km spatial resolution [ENG].
  2. Fresh cache hit: Served when cached entry age <= ttl_seconds (30 min [ENG assumption]).
     Returns cached data with cache_hit=True, stale=False.
  3. Stale cache fallback: When API is down/times out, serve cached entry ONLY if
     entry age <= max_stale_age_hours (24h [ENG assumption]).
     Returns cached data with cache_hit=True, stale=True.
  4. Cache miss / expired beyond max_stale: Returns None, forcing WeatherUnavailable.
     NEVER fabricate or interpolate numbers.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    data: Dict[str, Any]
    timestamp: float  # time.time() when recorded
    latitude: float
    longitude: float
    date_window: str


class WeatherCache:
    """In-memory LRU cache with freshness TTL and stale fallback policy."""

    def __init__(
        self,
        ttl_seconds: float = 1800.0,        # 30 minutes [ENG assumption]
        max_stale_age_hours: float = 24.0,  # 24 hours max stale fallback [ENG assumption]
        max_entries: int = 1000,
        coord_rounding: int = 2,            # 2 decimals ≈ 1.1 km
    ):
        self.ttl_seconds = float(ttl_seconds)
        self.max_stale_seconds = float(max_stale_age_hours * 3600.0)
        self.max_entries = int(max_entries)
        self.coord_rounding = int(coord_rounding)
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    def make_key(self, lat: float, lon: float, date_window: str) -> str:
        """Construct deterministic cache key from rounded coordinates and date window."""
        r_lat = round(float(lat), self.coord_rounding)
        r_lon = round(float(lon), self.coord_rounding)
        return f"{r_lat:.2f}_{r_lon:.2f}_{date_window}"

    def get_fresh(self, lat: float, lon: float, date_window: str) -> Optional[Dict[str, Any]]:
        """Return fresh cached data if age <= ttl_seconds. Otherwise None."""
        key = self.make_key(lat, lon, date_window)
        entry = self._store.get(key)
        if entry is None:
            return None

        age = time.time() - entry.timestamp
        if age <= self.ttl_seconds:
            # Move to end for LRU order
            self._store.move_to_end(key)
            logger.debug("Fresh cache hit for %s (age=%.1fs <= %.1fs)", key, age, self.ttl_seconds)
            return dict(entry.data)

        logger.debug("Cache entry for %s expired (age=%.1fs > %.1fs)", key, age, self.ttl_seconds)
        return None

    def get_stale(self, lat: float, lon: float, date_window: str) -> Optional[Dict[str, Any]]:
        """
        Return stale cached data if age <= max_stale_seconds.
        Used ONLY as fallback when weather APIs are unavailable or timed out.
        """
        key = self.make_key(lat, lon, date_window)
        entry = self._store.get(key)
        if entry is None:
            return None

        age = time.time() - entry.timestamp
        if age <= self.max_stale_seconds:
            self._store.move_to_end(key)
            logger.warning(
                "Serving STALE cache for %s (age=%.1fs <= %.1fs max stale)",
                key,
                age,
                self.max_stale_seconds,
            )
            return dict(entry.data)

        logger.warning(
            "Cache entry for %s discarded: older than max stale age (age=%.1fs > %.1fs)",
            key,
            age,
            self.max_stale_seconds,
        )
        return None

    def put(self, lat: float, lon: float, date_window: str, data: Dict[str, Any]) -> None:
        """Store weather data in cache with current timestamp."""
        key = self.make_key(lat, lon, date_window)
        if key in self._store:
            self._store.move_to_end(key)

        self._store[key] = CacheEntry(
            data=dict(data),
            timestamp=time.time(),
            latitude=round(float(lat), self.coord_rounding),
            longitude=round(float(lon), self.coord_rounding),
            date_window=date_window,
        )

        # Evict oldest entry if capacity exceeded
        if len(self._store) > self.max_entries:
            oldest_key, _ = self._store.popitem(last=False)
            logger.debug("Evicted oldest cache entry: %s", oldest_key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
