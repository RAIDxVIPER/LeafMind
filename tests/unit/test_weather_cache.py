"""
tests/unit/test_weather_cache.py
================================
Unit tests for WeatherCache (Stage C4).

Covers:
  - Fresh cache hit within TTL (30 min [ENG assumption])
  - Fresh cache expiry after TTL
  - Stale cache retrieval when fresh expired but within max_stale (24h [ENG assumption])
  - Stale cache expiry after max_stale_age_hours
  - Coordinate rounding to 2 decimals (~1.1 km radius)
  - LRU eviction when max_entries is exceeded
"""
from __future__ import annotations

import time
import pytest

from src.weather.cache import WeatherCache


class TestWeatherCache:
    """Test suite for WeatherCache functionality."""

    def test_cache_put_and_fresh_hit(self):
        cache = WeatherCache(ttl_seconds=1800.0, max_stale_age_hours=24.0)
        data = {"temperature_c": 26.5, "humidity": 78.0}
        cache.put(19.7515, 75.7139, "2026-10-22", data)

        cached = cache.get_fresh(19.7515, 75.7139, "2026-10-22")
        assert cached is not None
        assert cached["temperature_c"] == 26.5
        assert cached["humidity"] == 78.0

    def test_coord_rounding_groups_nearby_locations(self):
        """Points within ~1 km round to same 2 decimal places."""
        cache = WeatherCache(coord_rounding=2)
        data = {"city": "Pune"}
        # 18.5204 and 18.5249 round to 18.52
        # 73.8567 and 73.8588 round to 73.86
        cache.put(18.5204, 73.8567, "2026-10-22", data)

        hit = cache.get_fresh(18.5249, 73.8588, "2026-10-22")
        assert hit is not None
        assert hit["city"] == "Pune"

    def test_date_window_isolation(self):
        """Different date windows do not collide."""
        cache = WeatherCache()
        cache.put(19.75, 75.71, "2026-10-22", {"day": 1})
        cache.put(19.75, 75.71, "2026-10-23", {"day": 2})

        d1 = cache.get_fresh(19.75, 75.71, "2026-10-22")
        d2 = cache.get_fresh(19.75, 75.71, "2026-10-23")
        assert d1["day"] == 1
        assert d2["day"] == 2

    def test_fresh_expiry_and_stale_fallback(self):
        """Test that after TTL, fresh hit fails but stale lookup succeeds."""
        # TTL: 0.1s, Stale max: 5.0s
        cache = WeatherCache(ttl_seconds=0.1, max_stale_age_hours=5.0 / 3600.0)
        cache.put(19.75, 75.71, "window1", {"value": "cached"})

        # Immediate fresh hit
        assert cache.get_fresh(19.75, 75.71, "window1") is not None

        # Wait for TTL to expire
        time.sleep(0.15)

        # Fresh lookup must return None
        assert cache.get_fresh(19.75, 75.71, "window1") is None

        # Stale lookup must succeed
        stale = cache.get_stale(19.75, 75.71, "window1")
        assert stale is not None
        assert stale["value"] == "cached"

    def test_stale_expiry_returns_none(self):
        """After max_stale_seconds, stale lookup returns None."""
        cache = WeatherCache(ttl_seconds=0.05, max_stale_age_hours=0.1 / 3600.0)
        cache.put(19.75, 75.71, "window1", {"value": "old"})

        # Wait past max_stale (0.15s)
        time.sleep(0.15)

        assert cache.get_fresh(19.75, 75.71, "window1") is None
        assert cache.get_stale(19.75, 75.71, "window1") is None

    def test_lru_eviction(self):
        """Oldest entry evicted when capacity exceeded."""
        cache = WeatherCache(max_entries=2)
        cache.put(10.0, 10.0, "w1", {"k": 1})
        cache.put(20.0, 20.0, "w2", {"k": 2})
        assert len(cache) == 2

        # Access first entry so entry 2 becomes LRU
        _ = cache.get_fresh(10.0, 10.0, "w1")

        # Insert 3rd entry
        cache.put(30.0, 30.0, "w3", {"k": 3})
        assert len(cache) == 2

        # 10.0 and 30.0 should exist; 20.0 should be evicted
        assert cache.get_fresh(10.0, 10.0, "w1") is not None
        assert cache.get_fresh(30.0, 30.0, "w3") is not None
        assert cache.get_fresh(20.0, 20.0, "w2") is None
