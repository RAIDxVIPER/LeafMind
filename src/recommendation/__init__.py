"""
src/recommendation/__init__.py
==============================
Recommendation module for LeafMind (Stage C6).
"""
from src.recommendation.fallback import build_fallback_recommendation
from src.recommendation.prompt import build_recommendation_prompt
from src.recommendation.provider import (
    GeminiProvider,
    LLMProvider,
    MockLLMProvider,
    OllamaProvider,
    get_provider,
    sanitize_key,
)
from src.recommendation.validation import validate_grounding

__all__ = [
    "LLMProvider",
    "GeminiProvider",
    "OllamaProvider",
    "MockLLMProvider",
    "get_provider",
    "sanitize_key",
    "build_recommendation_prompt",
    "validate_grounding",
    "build_fallback_recommendation",
]
