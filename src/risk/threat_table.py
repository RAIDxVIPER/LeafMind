"""
src/risk/threat_table.py
========================
Loads and queries the disease threat table (config/threat_table.json).

All non-healthy disease classes are currently [GAP] — the threat table has not been
filled by a plant pathologist (issue I19). The scorer uses the DISEASE_DETECTED
fallback (threat_score = 0.50) and sets threat_source = "GAP_FALLBACK" for all
such entries, capping risk_label at "moderate" per I26.

DO NOT invent threat values. All tier-to-score mappings are from
docs/spec/part4_decision_logic.md §4a-ii and part6_build_spec.md §6d-iv.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_THREAT_TABLE_PATH = REPO_ROOT / "config" / "threat_table.json"

# Tier-to-score mapping [ENG assumption — values from spec §6d-iv]
# Source: docs/spec/part4_decision_logic.md §4a-ii; part6_build_spec.md §6d-iv
TIER_SCORES: dict[str, float] = {
    "HEALTHY":          0.0,    # [ENG assumption]
    "LOW":              0.25,   # [ENG assumption]
    "MEDIUM":           0.50,   # [ENG assumption]
    "HIGH":             0.85,   # [ENG assumption]
    "DISEASE_DETECTED": 0.50,   # [ENG assumption — fallback when table cell is empty]
}

# GAP fallback tier used when a class entry has status="[GAP]" or tier is null
GAP_FALLBACK_TIER = "DISEASE_DETECTED"
GAP_FALLBACK_SCORE = TIER_SCORES[GAP_FALLBACK_TIER]   # 0.50 [ENG assumption]


@dataclass(frozen=True)
class ThreatLookupResult:
    """Result of a threat table lookup for one disease class."""
    model_class: str
    tier: str                   # e.g. "DISEASE_DETECTED", "HIGH", "HEALTHY"
    threat_score: float         # numeric score [0.0, 1.0]
    threat_source: str          # "threat_table" | "GAP_FALLBACK" | "HEALTHY"
    is_gap: bool                # True when falling back due to empty table (I19)
    notes: str                  # Explanatory note from table entry or fallback note


class ThreatTable:
    """
    Loads config/threat_table.json and performs per-class lookups.

    Healthy class detection is based on the class name containing "healthy"
    (case-insensitive). Healthy classes always return threat_score=0.0 and
    threat_source="HEALTHY" regardless of any other table entry.

    For all non-healthy classes, the table entry is read. If the tier is
    null, "GAP", or "DISEASE_DETECTED" with status "[GAP]", the GAP_FALLBACK
    is applied and is_gap=True is set. [ENG decision — I19, I26]

    The threat score values are read from the table's own _tier_scores block
    at load time, not hardcoded, so they remain overridable via config.
    """

    def __init__(self, table_path: Optional[Path] = None) -> None:
        self._path = table_path or DEFAULT_THREAT_TABLE_PATH
        self._data: dict = {}
        self._classes: dict = {}
        self._tier_scores: dict[str, float] = dict(TIER_SCORES)  # start from defaults
        self._loaded = False

    def load(self) -> None:
        """Load the threat table JSON from disk. Idempotent."""
        if self._loaded:
            return
        if not self._path.exists():
            raise FileNotFoundError(
                f"Threat table not found at {self._path}. "
                "Run the appropriate setup script or check config/."
            )
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        # Override tier scores from table's _tier_scores block if present
        if "_tier_scores" in self._data:
            for tier, score in self._data["_tier_scores"].items():
                self._tier_scores[tier] = float(score)

        self._classes = self._data.get("classes", {})
        self._loaded = True
        logger.debug("ThreatTable loaded: %d classes from %s", len(self._classes), self._path)

    def lookup(self, model_class: str) -> ThreatLookupResult:
        """
        Look up the threat tier and score for a given model class string.

        Behaviour:
        - If model_class contains "healthy" (case-insensitive): returns HEALTHY (score=0.0).
        - If model_class found in table with a non-GAP tier (e.g. HIGH, MEDIUM, LOW):
          returns that tier with its score. threat_source="threat_table".
        - Otherwise (class unknown, or status="[GAP]", or tier=DISEASE_DETECTED+GAP):
          returns GAP_FALLBACK. threat_source="GAP_FALLBACK". is_gap=True.

        Note: Currently ALL non-healthy classes use the GAP_FALLBACK because the
        threat table has not been filled by a plant pathologist (I19).
        """
        self.load()

        # Healthy class detection — primary check (D10: risk is separate from uncertainty)
        if "healthy" in model_class.lower():
            return ThreatLookupResult(
                model_class=model_class,
                tier="HEALTHY",
                threat_score=self._tier_scores.get("HEALTHY", 0.0),
                threat_source="HEALTHY",
                is_gap=False,
                notes="Healthy class: threat_score=0.0. [ENG]",
            )

        entry = self._classes.get(model_class)

        if entry is None:
            # Class not in table at all — use GAP_FALLBACK
            logger.warning(
                "ThreatTable: class '%s' not found — using GAP_FALLBACK. "
                "Update config/threat_table.json when pathologist assigns tier.",
                model_class,
            )
            return ThreatLookupResult(
                model_class=model_class,
                tier=GAP_FALLBACK_TIER,
                threat_score=GAP_FALLBACK_SCORE,
                threat_source="GAP_FALLBACK",
                is_gap=True,
                notes=(
                    f"Class '{model_class}' not found in threat table. "
                    f"Using DISEASE_DETECTED fallback (0.50). [ENG — I19]"
                ),
            )

        tier = entry.get("tier", GAP_FALLBACK_TIER)
        status = entry.get("status", "[GAP]")
        notes = entry.get("notes", "")

        # If status is [GAP] or tier is DISEASE_DETECTED with [GAP] status → GAP_FALLBACK
        is_gap = "[GAP]" in status or tier == GAP_FALLBACK_TIER
        # A legitimate DISEASE_DETECTED without [GAP] still qualifies as a non-gap entry,
        # but per spec all current table entries are [GAP]. Treat consistently.
        # Any entry with [GAP] in status is a gap.

        threat_score = self._tier_scores.get(tier, GAP_FALLBACK_SCORE)
        if is_gap:
            threat_source = "GAP_FALLBACK"
        else:
            threat_source = "threat_table"

        return ThreatLookupResult(
            model_class=model_class,
            tier=tier,
            threat_score=threat_score,
            threat_source=threat_source,
            is_gap=is_gap,
            notes=notes,
        )

    @property
    def tier_scores(self) -> dict[str, float]:
        """Expose tier → score mapping (read from config, not hardcoded)."""
        self.load()
        return dict(self._tier_scores)

    @property
    def all_classes(self) -> dict:
        """Return the raw classes dict from the loaded table."""
        self.load()
        return dict(self._classes)
