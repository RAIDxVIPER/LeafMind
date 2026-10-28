"""
src/orchestrator/escalation.py
==============================
Deterministic evaluation of human-in-the-loop escalation triggers T1–T11 (Stage C7).

Implements:
  - Spec 4d-i to 4d-iv: Deterministic escalation rules T1–T11 and priority mapping
  - Spec 6d-vi: Priority resolution (urgent > high > medium > low)
  - I42: Config-driven thresholds loaded from config/orchestrator_config.yaml
  - Single-class crop / can_diagnose=False escalation
  - HumanEscalationPayload construction for agronomist queue
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import yaml

from src.orchestrator.state import EscalationSummary, TriggerReason
from src.schemas.schemas import (
    HumanEscalationPayload,
    ImageQualityFlags,
    MonitoringOutput,
    RAGOutput,
    RecommendationOutput,
    RiskOutput,
    VisionOutput,
    WeatherOutput,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ORCHESTRATOR_CONFIG = REPO_ROOT / "config" / "orchestrator_config.yaml"

PRIORITY_RANKS = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "urgent": 3,
}
RANK_TO_PRIORITY = {v: k for k, v in PRIORITY_RANKS.items()}


def load_orchestrator_config(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load orchestrator and escalation thresholds from YAML config file."""
    path = Path(config_path) if config_path else DEFAULT_ORCHESTRATOR_CONFIG
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to load orchestrator config from %s: %s", path, e)
    return {}


def evaluate_escalation(
    vision: Optional[VisionOutput] = None,
    rag: Optional[RAGOutput] = None,
    weather: Optional[WeatherOutput] = None,
    risk: Optional[RiskOutput] = None,
    recommendation: Optional[RecommendationOutput] = None,
    monitoring: Optional[MonitoringOutput] = None,
    visit_number: int = 1,
    config_path: Optional[Path | str] = None,
    image_bytes: Optional[bytes] = None,
    case_id: Optional[uuid.UUID | str] = None,
    farmer_id: Optional[str] = "farmer_default",
) -> EscalationSummary:
    """
    Evaluate escalation triggers T1–T11, compute unified priority, and build
    HumanEscalationPayload when escalation is required.

    All thresholds come from config (I42), never hardcoded.
    """
    cfg = load_orchestrator_config(config_path)
    esc_cfg = cfg.get("escalation", {})

    conf_thresh = float(esc_cfg.get("confidence_threshold", 0.60))
    gap_thresh = float(esc_cfg.get("prob_gap_threshold", 0.15))
    ood_thresh = float(esc_cfg.get("ood_threshold", 0.40))
    persist_thresh = int(esc_cfg.get("persistence_visit_threshold", 2))
    extended_thresh = int(esc_cfg.get("extended_visit_threshold", 3))

    triggered_codes: List[str] = []
    reasons: List[TriggerReason] = []
    max_rank = PRIORITY_RANKS["low"]

    def add_trigger(code: str, desc: str, prio: str, val: Any = None, thresh: Any = None) -> None:
        nonlocal max_rank
        triggered_codes.append(code)
        reasons.append(
            TriggerReason(
                trigger_id=code,
                description=desc,
                priority=prio,
                value=val,
                threshold=thresh,
            )
        )
        rank = PRIORITY_RANKS.get(prio, 0)
        if rank > max_rank:
            max_rank = rank

    # ─────────────────────────────────────────────────────────────────────────
    # T1–T5: Vision Triggers
    # ─────────────────────────────────────────────────────────────────────────
    if vision is not None:
        # T1: confidence < threshold
        if vision.confidence < conf_thresh:
            add_trigger(
                "T1",
                f"Classifier confidence ({vision.confidence:.2f}) below threshold ({conf_thresh:.2f}) [ENG]",
                "medium",
                val=vision.confidence,
                thresh=conf_thresh,
            )

        # T2: top1.prob - top2.prob < threshold
        if vision.top_k_probs and len(vision.top_k_probs) >= 2:
            gap = vision.top_k_probs[0].prob - vision.top_k_probs[1].prob
            if gap < gap_thresh:
                add_trigger(
                    "T2",
                    f"Top-2 probability margin ({gap:.2f}) below threshold ({gap_thresh:.2f}) [ENG]",
                    "medium",
                    val=round(gap, 4),
                    thresh=gap_thresh,
                )

        # T3: low resolution
        if vision.image_quality_flags and vision.image_quality_flags.low_resolution:
            add_trigger(
                "T3",
                "Image quality flag: low resolution leaf photo [ENG]",
                "medium",
                val=True,
                thresh=True,
            )

        # T4: blurry
        if vision.image_quality_flags and vision.image_quality_flags.blurry:
            add_trigger(
                "T4",
                "Image quality flag: blurry image detected [ENG]",
                "medium",
                val=True,
                thresh=True,
            )

        # T5: OOD flag or very low confidence
        if vision.ood_flag or vision.confidence < ood_thresh:
            add_trigger(
                "T5",
                f"Out-of-distribution image (ood_flag={vision.ood_flag}, confidence={vision.confidence:.2f} < {ood_thresh:.2f}) [ENG]",
                "urgent",
                val=vision.confidence,
                thresh=ood_thresh,
            )

        # Single-class crop / can_diagnose=False (D17)
        if hasattr(vision, "can_diagnose") and not vision.can_diagnose:
            add_trigger(
                "SINGLE_CLASS_CROP",
                f"Single-class crop '{vision.crop}': classifier cannot provide differential diagnosis [D17]",
                "high",
                val=vision.crop,
                thresh="multi-class required",
            )

        # General vision escalate flag
        if vision.escalate_to_human and "T1" not in triggered_codes and "T2" not in triggered_codes and "T5" not in triggered_codes:
            add_trigger(
                "VISION_UNCERTAIN",
                "Vision agent requested human escalation",
                "medium",
                val=True,
                thresh=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # T6: RAG Knowledge Gap Trigger
    # ─────────────────────────────────────────────────────────────────────────
    if rag is not None:
        if rag.knowledge_gap_flag or rag.kb_gap or not rag.rag_available:
            add_trigger(
                "T6",
                f"RAG knowledge base gap or unavailable guidance for '{rag.model_class}' [GAP]",
                "medium",
                val=rag.kb_gap,
                thresh=False,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # T7: Severe Risk Trigger (Async Notification)
    # ─────────────────────────────────────────────────────────────────────────
    # Per Spec 4d-ii Note: "severe" cases with high confidence (escalate_to_human=False)
    # generate an agronomist notification while farmer receives automated advice.
    is_currently_escalated = (max_rank > PRIORITY_RANKS["low"])
    if risk is not None:
        if risk.risk_level == "severe" and not is_currently_escalated:
            add_trigger(
                "T7",
                "Severe disease risk detected: automated recommendation provided with async agronomist notification [ENG]",
                "high",
                val=risk.risk_level,
                thresh="severe",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # T8–T11: Monitoring Triggers (Evaluated if MonitoringOutput present)
    # ─────────────────────────────────────────────────────────────────────────
    if monitoring is not None:
        # T8: different disease detected at follow-up
        if monitoring.disease_status == "different_disease_detected_at_followup":
            add_trigger(
                "T8",
                "Monitoring: different disease detected at follow-up visit [ENG]",
                "urgent",
                val=monitoring.disease_status,
                thresh="different_disease_detected_at_followup",
            )

        # T9: same disease persists
        if monitoring.disease_status == "same_disease_still_detected":
            curr_visit = monitoring.visit_number or visit_number
            if curr_visit >= extended_thresh:
                add_trigger(
                    "T9",
                    f"Monitoring: disease persists through extended visit {curr_visit} >= {extended_thresh} [ENG]",
                    "urgent",
                    val=curr_visit,
                    thresh=extended_thresh,
                )
            elif curr_visit >= persist_thresh:
                add_trigger(
                    "T9",
                    f"Monitoring: disease persists through visit {curr_visit} >= {persist_thresh} [ENG]",
                    "high",
                    val=curr_visit,
                    thresh=persist_thresh,
                )

        # T10: poor quality on follow-up
        if monitoring.poor_quality_flag:
            add_trigger(
                "T10",
                "Monitoring: poor quality follow-up image [ENG]",
                "medium",
                val=True,
                thresh=True,
            )

        # T11: crop mismatch
        if monitoring.crop_mismatch_flag:
            add_trigger(
                "T11",
                "Monitoring: crop mismatch detected between baseline and follow-up [ENG]",
                "high",
                val=True,
                thresh=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Final Priority & Escalation Decision
    # ─────────────────────────────────────────────────────────────────────────
    final_priority = RANK_TO_PRIORITY[max_rank]
    # T7 alone is an async notification, but agronomist payload is built
    should_escalate = (max_rank > PRIORITY_RANKS["low"])

    # Build agronomist payload if escalation triggered
    payload: Optional[HumanEscalationPayload] = None
    if should_escalate:
        c_id = uuid.UUID(str(case_id)) if case_id else uuid.uuid4()
        f_id = str(farmer_id) if farmer_id else "farmer_default"
        payload_reasons = [r.description for r in reasons] or ["Automated escalation check triggered"]

        # If any agent output is missing, construct minimal valid schema object
        v_out = vision or VisionOutput(
            top_label="Unknown",
            confidence=0.0,
            top_k_probs=[{"label": "Unknown", "prob": 0.0}],
            severity_pct=None,
            severity_note="Severity percentage is not computed.",
            image_quality_flags=ImageQualityFlags(low_resolution=False, blurry=False, non_leaf_suspected=False),
            ood_flag=True,
            model_version="unavailable",
            temperature=1.0,
        )
        w_out = weather or WeatherOutput(
            weather_available=False,
            weather_partial=False,
            disease_risk_flags=[],
            weather_data_source=None,
        )
        r_out = rag or RAGOutput(
            crop="Unknown",
            model_class="Unknown",
            disease_name="Unknown",
            found=False,
            kb_gap=True,
            rag_available=False,
            knowledge_gap_flag=True,
        )
        rk_out = risk or RiskOutput(
            risk_score=0.50,
            risk_level="moderate",
            uncertainty_level="low_confidence",
            factors=[],
            escalate_to_human=True,
            uncertainty_signals_triggered=[],
            weather_unavailable=True,
            rag_unavailable=True,
            progression_note="Automated risk score unavailable; fallback applied.",
            risk_computed_at=datetime.now(timezone.utc),
        )
        rec_out = recommendation or RecommendationOutput(
            summary="Case escalated to agronomist review queue.",
            urgency="consult_agronomist",
            actions=["Consult a local agronomist for manual field inspection."],
            steps=["Consult a local agronomist for manual field inspection."],
            citations=[],
            rag_grounded=False,
            escalate_to_human=True,
        )

        payload = HumanEscalationPayload(
            case_id=c_id,
            farmer_id=f_id,
            escalation_reason=payload_reasons,
            vision_output=v_out,
            weather_output=w_out,
            rag_output=r_out,
            risk_output=rk_out,
            draft_recommendation=rec_out,
            image_bytes=image_bytes or b"",
            escalated_at=datetime.now(timezone.utc),
            priority=final_priority,
        )

    return EscalationSummary(
        escalate_to_human=should_escalate,
        priority=final_priority,
        triggered_triggers=triggered_codes,
        reasons=reasons,
        payload=payload,
    )
