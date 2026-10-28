"""
src/api/config.py
=================
API configuration loader for LeafMind FastAPI backend.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger("leafmind.api.config")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "api_config.yaml"


def load_api_config(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load API configuration with safe fallbacks."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to load api config from %s: %s; using defaults", path, e)

    return {
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "timeout_seconds": 60.0,
        },
        "limits": {
            "max_image_size_mb": 10.0,
            "min_dimension_px": 64,
        },
        "cors": {
            "allowed_origins": [
                "http://localhost:8501",
                "http://127.0.0.1:8501",
                "http://localhost:3000",
                "http://localhost:5173",
            ]
        },
        "guards": {
            "healthy_runner_up_threshold": 0.25,
        },
    }
