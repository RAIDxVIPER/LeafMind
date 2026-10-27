"""
src/recommendation/provider.py
==============================
LLM Provider Layer for Recommendation Agent (Stage C6).

Provides an abstract interface and concrete implementations for:
  - GeminiProvider (Google Gemini API via REST)
  - OllamaProvider (Local Ollama via REST)
  - MockLLMProvider (Deterministic test harness)

Security and Privacy Rules:
  - Gemini reads GEMINI_API_KEY from environment or local .env.
  - NEVER log, print, or write the API key anywhere, including in error
    messages, exception tracebacks, or request headers.
  - All exception messages are scrubbed of any API key patterns before propagation.
  - Model names are configured via config/recommendation_config.yaml, not hardcoded [I25].
"""
from __future__ import annotations

import abc
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import requests

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
API_KEY_REGEX = re.compile(r"AIza[0-9A-Za-z-_]{35}")


def sanitize_key(text: str, key: Optional[str] = None) -> str:
    """
    Sanitize an API key or key pattern from a text string.
    Ensures that real or fake API keys never appear in logs or error messages.
    """
    if not text:
        return text
    clean = text
    if key and len(key) >= 8:
        clean = clean.replace(key, "[REDACTED_KEY]")
    clean = API_KEY_REGEX.sub("[REDACTED_KEY]", clean)
    return clean


def load_local_env_safely(env_path: Optional[Path] = None) -> None:
    """
    Load environment variables from a local .env file safely without printing or logging.
    Does not overwrite existing environment variables.
    """
    if env_path is None:
        env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if "#" in v and not (v.startswith(('"', "'")) and v.endswith(('"', "'"))):
                    v = v.split("#", 1)[0].strip()
                v = v.strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        # Silently ignore read errors to maintain security and prevent crashes
        pass


class LLMProvider(abc.ABC):
    """Abstract base class for recommendation LLM providers."""

    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate completion text from the LLM given a prompt and optional system instruction.

        Args:
            prompt: User-level input text.
            system_prompt: System-level instruction/rules.

        Returns:
            Raw string completion (typically JSON formatted).
        """
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """
    Provider for Google Gemini API via REST.

    Reads GEMINI_API_KEY from environment or local .env safely.
    Model name is loaded from config [I25].
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        cfg = config or {}
        gemini_cfg = cfg.get("llm", {}).get("gemini", {})
        self.model = gemini_cfg.get("model", "gemini-2.5-flash")
        self.temperature = float(gemini_cfg.get("temperature", 0.2))
        self.max_tokens = int(gemini_cfg.get("max_output_tokens", 1024))
        self.timeout = float(gemini_cfg.get("timeout_seconds", 30.0))

        # Resolve API key
        if not api_key:
            load_local_env_safely()
            api_key = os.environ.get("GEMINI_API_KEY")

        self._api_key = api_key
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or local .env")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }

        contents = [{"parts": [{"text": prompt}]}]
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as e:
            sanitized = sanitize_key(str(e), self._api_key)
            logger.warning("Gemini request timed out after %.1fs: %s", self.timeout, sanitized)
            raise TimeoutError(f"Gemini request timed out after {self.timeout}s: {sanitized}") from None
        except Exception as e:
            sanitized = sanitize_key(str(e), self._api_key)
            logger.warning("Gemini network error: %s", sanitized)
            raise RuntimeError(f"Gemini request failed: {sanitized}") from None

        if resp.status_code != 200:
            sanitized_body = sanitize_key(resp.text, self._api_key)
            logger.warning("Gemini API error HTTP %d: %s", resp.status_code, sanitized_body)
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {sanitized_body}")

        try:
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini returned no candidates (possible safety filter block)")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts or "text" not in parts[0]:
                raise ValueError("Gemini response candidate missing text parts")
            return parts[0]["text"]
        except Exception as e:
            sanitized = sanitize_key(str(e), self._api_key)
            raise RuntimeError(f"Failed to parse Gemini response: {sanitized}") from None


class OllamaProvider(LLMProvider):
    """
    Provider for local Ollama via REST.

    Connects to http://localhost:11434 by default.
    Model name is loaded from config [I25].
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        ollama_cfg = cfg.get("llm", {}).get("ollama", {})
        self.model = ollama_cfg.get("model", "llama3:8b")
        self.base_url = ollama_cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self.temperature = float(ollama_cfg.get("temperature", 0.2))
        self.timeout = float(ollama_cfg.get("timeout_seconds", 30.0))

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout as e:
            logger.warning("Ollama request timed out after %.1fs: %s", self.timeout, e)
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s") from None
        except Exception as e:
            logger.warning("Ollama connection error: %s", e)
            raise RuntimeError(f"Ollama connection error: {e}") from None

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"Failed to parse Ollama response: {e}") from None


class MockLLMProvider(LLMProvider):
    """
    Deterministic mock provider for unit testing without network or API keys.
    """

    def __init__(
        self,
        handler: Optional[Union[str, Callable[[str, Optional[str]], str], Exception]] = None,
    ) -> None:
        self.handler = handler
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        if isinstance(self.handler, Exception):
            raise self.handler
        elif callable(self.handler):
            return self.handler(prompt, system_prompt)
        elif isinstance(self.handler, str):
            return self.handler
        elif self.handler is None:
            return "{}"
        else:
            return str(self.handler)


def get_provider(
    provider_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> LLMProvider:
    """
    Factory function to instantiate an LLMProvider.

    Checks provider_name, then LLM_PROVIDER env var, then config.
    Defaults to 'gemini'.

    Raises:
        ValueError if the provider name is unknown or unsupported.
    """
    raw_name = (
        provider_name
        or os.environ.get("LLM_PROVIDER")
        or (config.get("llm", {}).get("provider") if config else None)
        or "gemini"
    )
    if "#" in raw_name:
        raw_name = raw_name.split("#", 1)[0]
    name = raw_name.lower().strip()

    if name == "gemini":
        return GeminiProvider(config=config)
    elif name == "ollama":
        return OllamaProvider(config=config)
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{name}'. Supported providers are 'gemini' or 'ollama'."
        )
