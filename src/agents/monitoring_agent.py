"""
src/agents/monitoring_agent.py
==============================
Monitoring Agent for LeafMind Stage C8.

Provides:
  - MonitoringAgent class wrapping plausibility checks and delta computations
  - Config-driven thresholds and caveats (from config/monitoring_config.yaml)
  - Degradation on missing or corrupted baseline CaseState
  - Strict compliance with Spec 4c, D2/R3 (severity stays null), and D4/I21 (no conflating confidence with efficacy)
"""
from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from src.monitoring.comparator import compute_monitoring_delta
from src.schemas.schemas import CaseState, MonitoringOutput, VisionOutput

logger = logging.getLogger("leafmind.monitoring.agent")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "monitoring_config.yaml"


def load_monitoring_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load monitoring configuration from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        logger.warning("Monitoring config not found at %s; using default thresholds.", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to parse monitoring config at %s: %s", path, e)
        return {}


class MonitoringAgent:
    """
    Monitoring Agent — follow-up case monitoring and plausibility evaluator (Stage C8).
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        self.config_path = config_path
        self.config = load_monitoring_config(config_path)
        esc_cfg = self.config.get("escalation", {})
        self.persistent_disease_min_visits = int(esc_cfg.get("persistent_disease_min_visits", 2))
        self.extended_persistence_min_visits = int(esc_cfg.get("extended_persistence_min_visits", 3))

    def evaluate(
        self,
        current_vision: VisionOutput,
        baseline_case: CaseState,
        current_image_bytes: Optional[bytes] = None,
        current_crop: Optional[str] = None,
        current_submitted_at: Optional[datetime] = None,
    ) -> MonitoringOutput:
        """
        Evaluate change between follow-up visit and baseline case.

        Args:
            current_vision: VisionOutput of follow-up image
            baseline_case: Persisted CaseState of baseline case
            current_image_bytes: Optional raw bytes of follow-up image
            current_crop: Optional user-supplied crop
            current_submitted_at: Optional timestamp of follow-up submission

        Returns:
            MonitoringOutput containing disease_status, confidence_raw_delta, and plausibility flags.
        """
        if baseline_case is None:
            raise ValueError("baseline_case cannot be None.")
        if baseline_case.initial_vision is None:
            raise ValueError("baseline_case has no initial_vision.")

        return compute_monitoring_delta(
            current_vision=current_vision,
            baseline_case=baseline_case,
            current_image_bytes=current_image_bytes,
            current_crop=current_crop,
            current_submitted_at=current_submitted_at,
            persistent_disease_min_visits=self.persistent_disease_min_visits,
            extended_persistence_min_visits=self.extended_persistence_min_visits,
        )
