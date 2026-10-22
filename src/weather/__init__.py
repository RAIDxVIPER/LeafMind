"""
src/weather package
===================
Weather module for Stage C4: geocoding, caching, rule evaluation, and API client.
"""
from src.weather.geocoding import GeocodingClient, ResolvedLocation
from src.weather.cache import WeatherCache
from src.weather.rules import WeatherRuleEngine
from src.weather.client import WeatherClient, WeatherFetchError

__all__ = [
    "GeocodingClient",
    "ResolvedLocation",
    "WeatherCache",
    "WeatherRuleEngine",
    "WeatherClient",
    "WeatherFetchError",
]
