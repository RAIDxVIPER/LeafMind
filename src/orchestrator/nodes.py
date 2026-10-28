"""
src/orchestrator/nodes.py
=========================
Graph node implementations for LeafMind LangGraph orchestrator (Stage C7).

Implements:
  - 16 LangGraph node functions per spec 3c-iii
  - Spec 4e: Graceful degradation on all agent failures (never crash)
  - Spec 4d: Escalation triggers T1–T11 integration
  - D2/R3: severity_pct = None with mandatory explanation string
  - D3: Measured execution time in ms per node
  - D17: Crop-masked vision handling
  - D24: Location resolution default (Maharashtra)
  - Secret protection: All errors sanitized via sanitize_error()
  - Safe concurrent returns: nodes return localized dicts/lists for Annotated reducers
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from src.agents.monitoring_agent import MonitoringAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.risk_agent import RiskAgent
from src.agents.vision_agent import VisionAgent
from src.agents.weather_agent import WeatherAgent
from src.orchestrator.escalation import evaluate_escalation, load_orchestrator_config
from src.orchestrator.state import (
    EscalationSummary,
    FinalResponse,
    LeafMindState,
    RAGSummary,
    VisionSummary,
    WeatherSummary,
    sanitize_error,
)
from src.orchestrator.store import save_case_record, save_final_response, load_case_record
from src.rag.agent import RAGAgent
from src.schemas.schemas import (
    CaseState,
    FarmerRequest,
    FarmerResponse,
    FollowUpRecord,
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


class OrchestratorNodes:
    """
    Container class holding all LangGraph node functions with injectable agents.
    Allows easy dependency injection for unit testing without GPU or network.
    """

    def __init__(
        self,
        vision_agent: Optional[VisionAgent] = None,
        weather_agent: Optional[WeatherAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        risk_agent: Optional[RiskAgent] = None,
        recommendation_agent: Optional[RecommendationAgent] = None,
        monitoring_agent: Optional[MonitoringAgent] = None,
        case_store_dir: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self.vision_agent = vision_agent
        self.weather_agent = weather_agent
        self.rag_agent = rag_agent
        self.risk_agent = risk_agent
        self.recommendation_agent = recommendation_agent
        self.monitoring_agent = monitoring_agent
        self.case_store_dir = case_store_dir
        self.config_path = config_path

    # Lazy agent getters to avoid heavy initialisation during unit test mock setups
    def get_vision_agent(self) -> VisionAgent:
        if self.vision_agent is None:
            cfg = load_orchestrator_config(self.config_path)
            esc_cfg = cfg.get("escalation", {})
            thresh_kwargs = {}
            if "confidence_threshold" in esc_cfg:
                thresh_kwargs["esc_conf"] = float(esc_cfg["confidence_threshold"])
            if "prob_gap_threshold" in esc_cfg:
                thresh_kwargs["esc_gap"] = float(esc_cfg["prob_gap_threshold"])
            if "ood_threshold" in esc_cfg:
                thresh_kwargs["ood_thresh"] = float(esc_cfg["ood_threshold"])

            ckpt_dir = REPO_ROOT / "checkpoints" / "run1"
            if not ckpt_dir.is_dir():
                ckpt_dir = REPO_ROOT / "checkpoints" / "classifier"
            self.vision_agent = VisionAgent.from_checkpoint(ckpt_dir, **thresh_kwargs)
        return self.vision_agent

    def get_weather_agent(self) -> WeatherAgent:
        if self.weather_agent is None:
            self.weather_agent = WeatherAgent()
        return self.weather_agent

    def get_rag_agent(self) -> RAGAgent:
        if self.rag_agent is None:
            self.rag_agent = RAGAgent()
        return self.rag_agent

    def get_risk_agent(self) -> RiskAgent:
        if self.risk_agent is None:
            self.risk_agent = RiskAgent()
        return self.risk_agent

    def get_recommendation_agent(self) -> RecommendationAgent:
        if self.recommendation_agent is None:
            self.recommendation_agent = RecommendationAgent()
        return self.recommendation_agent

    def get_monitoring_agent(self) -> MonitoringAgent:
        if self.monitoring_agent is None:
            self.monitoring_agent = MonitoringAgent()
        return self.monitoring_agent

    # ─────────────────────────────────────────────────────────────────────────
    # 1. input_validator
    # ─────────────────────────────────────────────────────────────────────────
    def input_validator(self, state: LeafMindState) -> Dict[str, Any]:
        """Validate input payload, image bytes, and case parameters."""
        t0 = time.perf_counter()
        img = state.get("image_bytes")
        req = state.get("farmer_request")

        if img is None and req is not None:
            img = req.image_bytes

        if not img or len(img) == 0:
            err = "The image could not be read. Please submit a JPEG or PNG photo of the affected leaf."
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "fatal_error": "invalid_image",
                "errors": {"input_validator": err},
                "node_timings_ms": {"input_validator": elapsed},
                "flags": ["INVALID_IMAGE"],
            }

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "image_bytes": img,
            "node_timings_ms": {"input_validator": elapsed},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. route_request
    # ─────────────────────────────────────────────────────────────────────────
    def route_request(self, state: LeafMindState) -> Dict[str, Any]:
        """Determine whether request is a new case or follow-up visit."""
        t0 = time.perf_counter()
        is_followup = bool(state.get("is_followup", False))
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "is_followup": is_followup,
            "node_timings_ms": {"route_request": elapsed},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 3. case_initialiser
    # ─────────────────────────────────────────────────────────────────────────
    def case_initialiser(self, state: LeafMindState) -> Dict[str, Any]:
        """Initialize new case identity and metadata."""
        t0 = time.perf_counter()
        c_id = state.get("case_id") or str(uuid.uuid4())
        f_id = state.get("farmer_id") or "farmer_001"
        sub_at = state.get("submitted_at") or datetime.now(timezone.utc)

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "case_id": c_id,
            "farmer_id": f_id,
            "submitted_at": sub_at,
            "node_timings_ms": {"case_initialiser": elapsed},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 4. case_loader
    # ─────────────────────────────────────────────────────────────────────────
    def case_loader(self, state: LeafMindState) -> Dict[str, Any]:
        """Load existing CaseState on the follow-up path."""
        t0 = time.perf_counter()
        prev_id = state.get("previous_case_id") or state.get("case_id")
        curr_case_id = state.get("case_id")
        if not curr_case_id or (prev_id and curr_case_id == str(prev_id)):
            curr_case_id = str(uuid.uuid4())

        sub_at = state.get("submitted_at") or datetime.now(timezone.utc)

        if not prev_id:
            logger.warning("Follow-up requested but no previous_case_id or case_id provided; degrading gracefully.")
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "case_id": curr_case_id,
                "submitted_at": sub_at,
                "loaded_case_state": None,
                "flags": ["PREVIOUS_CASE_NOT_SPECIFIED"],
                "partial_result": True,
                "node_timings_ms": {"case_loader": elapsed},
            }

        loaded_case = load_case_record(prev_id, case_store_dir=self.case_store_dir)
        if loaded_case is None:
            logger.warning("Previous case record '%s' not found or corrupt; degrading gracefully.", prev_id)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "case_id": curr_case_id,
                "submitted_at": sub_at,
                "loaded_case_state": None,
                "flags": ["PREVIOUS_CASE_NOT_FOUND"],
                "partial_result": True,
                "node_timings_ms": {"case_loader": elapsed},
            }

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        f_id = state.get("farmer_id") or loaded_case.farmer_id
        return {
            "case_id": curr_case_id,
            "farmer_id": f_id,
            "submitted_at": sub_at,
            "loaded_case_state": loaded_case,
            "crop": state.get("crop") or loaded_case.crop_type,
            "node_timings_ms": {"case_loader": elapsed},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 5. vision_runner
    # ─────────────────────────────────────────────────────────────────────────
    def vision_runner(self, state: LeafMindState) -> Dict[str, Any]:
        """Execute Vision Agent inference with crop masking."""
        t0 = time.perf_counter()
        img = state.get("image_bytes")
        crop = state.get("crop")

        try:
            agent = self.get_vision_agent()
            vision_out: VisionOutput = agent.predict(img, crop=crop)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "vision_output": vision_out,
                "node_timings_ms": {"vision_runner": elapsed},
            }
        except Exception as e:
            sanitized = sanitize_error(e)
            logger.error("Vision Agent failed: %s", sanitized)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            # Per Spec 4e / R9: Vision failure is fatal (abort pipeline)
            return {
                "fatal_error": "vision_unavailable",
                "vision_error": sanitized,
                "errors": {"vision_runner": sanitized},
                "node_timings_ms": {"vision_runner": elapsed},
                "flags": ["VISION_FAILED"],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 6. weather_runner
    # ─────────────────────────────────────────────────────────────────────────
    def weather_runner(self, state: LeafMindState) -> Dict[str, Any]:
        """Execute Weather Agent retrieval and rule evaluations."""
        t0 = time.perf_counter()
        v_out = state.get("vision_output")
        detected_disease = v_out.top_label if v_out else None
        loc = state.get("location")

        lat = None
        lon = None
        place_name = None

        if isinstance(loc, tuple) and len(loc) == 2:
            lat, lon = loc
        elif isinstance(loc, dict):
            lat = loc.get("lat") or loc.get("latitude") or loc.get("gps_lat")
            lon = loc.get("lon") or loc.get("longitude") or loc.get("gps_lon")
            place_name = loc.get("place_name")
        elif isinstance(loc, str):
            place_name = loc

        try:
            agent = self.get_weather_agent()
            w_out: WeatherOutput = agent.get_weather(
                latitude=lat,
                longitude=lon,
                place_name=place_name,
                detected_disease=detected_disease,
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "weather_output": w_out,
                "node_timings_ms": {"weather_runner": elapsed},
            }
        except Exception as e:
            # Per Spec 4e: Graceful degradation — weather unavailable
            sanitized = sanitize_error(e)
            logger.warning("Weather Agent failed; degrading gracefully: %s", sanitized)
            w_fallback = WeatherOutput(
                weather_available=False,
                weather_partial=False,
                disease_risk_flags=[],
                weather_data_source=None,
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "weather_output": w_fallback,
                "weather_error": sanitized,
                "partial_result": True,
                "errors": {"weather_runner": sanitized},
                "node_timings_ms": {"weather_runner": elapsed},
                "flags": ["WEATHER_UNAVAILABLE"],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 7. rag_runner
    # ─────────────────────────────────────────────────────────────────────────
    def rag_runner(self, state: LeafMindState) -> Dict[str, Any]:
        """Execute RAG Agent extractive retrieval."""
        t0 = time.perf_counter()
        v_out = state.get("vision_output")
        top_label = v_out.top_label if v_out else "Unknown"
        crop = state.get("crop")
        include_doses = bool(state.get("include_doses", False))

        try:
            agent = self.get_rag_agent()
            rag_out: RAGOutput = agent.query_candidate(
                model_class=top_label,
                crop=crop,
                include_doses=include_doses,
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "rag_output": rag_out,
                "node_timings_ms": {"rag_runner": elapsed},
            }
        except Exception as e:
            # Per Spec 4e: Graceful degradation — RAG unavailable / gap
            sanitized = sanitize_error(e)
            logger.warning("RAG Agent failed; degrading gracefully: %s", sanitized)
            rag_fallback = RAGOutput(
                crop=crop,
                model_class=top_label,
                found=False,
                kb_gap=True,
                rag_available=False,
                knowledge_gap_flag=True,
                reason=f"Retrieval error: {sanitized}",
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "rag_output": rag_fallback,
                "rag_error": sanitized,
                "partial_result": True,
                "errors": {"rag_runner": sanitized},
                "node_timings_ms": {"rag_runner": elapsed},
                "flags": ["RAG_UNAVAILABLE"],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 8. fan_in_barrier
    # ─────────────────────────────────────────────────────────────────────────
    def fan_in_barrier(self, state: LeafMindState) -> Dict[str, Any]:
        """Fan-in synchronization barrier verifying presence of branch outputs."""
        t0 = time.perf_counter()
        w_out = state.get("weather_output")
        r_out = state.get("rag_output")
        new_flags = []
        partial = False

        if w_out is not None and not w_out.weather_available:
            partial = True
            new_flags.append("WEATHER_UNAVAILABLE")

        if r_out is not None and (not r_out.rag_available or r_out.kb_gap or r_out.knowledge_gap_flag):
            partial = True
            new_flags.append("RAG_GAP")

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "partial_result": partial,
            "flags": new_flags,
            "node_timings_ms": {"fan_in_barrier": elapsed},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 9. monitoring_runner (Stage C8)
    # ─────────────────────────────────────────────────────────────────────────
    def monitoring_runner(self, state: LeafMindState) -> Dict[str, Any]:
        """
        Stage C8 Monitoring runner.
        Evaluates change between follow-up visit and baseline case.
        """
        t0 = time.perf_counter()
        loaded_case = state.get("loaded_case_state")
        v_out = state.get("vision_output")

        if loaded_case is None or v_out is None:
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "monitoring_output": None,
                "node_timings_ms": {"monitoring_runner": elapsed},
            }

        try:
            agent = self.get_monitoring_agent()
            img_bytes = state.get("image_bytes")
            crop = state.get("crop")
            sub_at = state.get("submitted_at")

            m_out: MonitoringOutput = agent.evaluate(
                current_vision=v_out,
                baseline_case=loaded_case,
                current_image_bytes=img_bytes,
                current_crop=crop,
                current_submitted_at=sub_at,
            )

            new_flags = ["PLANT_IDENTITY_UNVERIFIED"]
            if m_out.same_image_flag:
                new_flags.append("SAME_IMAGE_RESUBMISSION")
            if m_out.poor_quality_flag:
                new_flags.append("POOR_QUALITY_FOLLOWUP")
            if m_out.crop_mismatch_flag:
                new_flags.append("CROP_MISMATCH")
            if m_out.too_soon_flag:
                new_flags.append("FOLLOWUP_TOO_SOON")

            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "monitoring_output": m_out,
                "flags": new_flags,
                "escalate_to_human": m_out.escalate_to_human,
                "node_timings_ms": {"monitoring_runner": elapsed},
            }
        except Exception as e:
            sanitized = sanitize_error(e)
            logger.warning("Monitoring Agent failed; degrading gracefully: %s", sanitized)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "monitoring_output": None,
                "flags": ["MONITORING_FAILED"],
                "partial_result": True,
                "errors": {"monitoring_runner": sanitized},
                "node_timings_ms": {"monitoring_runner": elapsed},
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 10. risk_reason_runner
    # ─────────────────────────────────────────────────────────────────────────
    def risk_reason_runner(self, state: LeafMindState) -> Dict[str, Any]:
        """Execute Risk Agent scoring."""
        t0 = time.perf_counter()
        v_out = state.get("vision_output")
        w_out = state.get("weather_output")
        r_out = state.get("rag_output")
        m_out = state.get("monitoring_output")

        # History status is passed only if monitoring evaluated a plausible case.
        # If crop mismatch, same image resubmission, poor quality, or unknown status: history=None.
        m_status = None
        if (
            m_out is not None
            and not m_out.crop_mismatch_flag
            and not m_out.same_image_flag
            and not m_out.poor_quality_flag
            and m_out.disease_status != "unknown"
        ):
            m_status = m_out.disease_status

        try:
            agent = self.get_risk_agent()
            risk_out: RiskOutput = agent.score(
                vision=v_out,
                weather=w_out,
                rag=r_out,
                monitoring_disease_status=m_status,
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "risk_output": risk_out,
                "node_timings_ms": {"risk_reason_runner": elapsed},
            }
        except Exception as e:
            sanitized = sanitize_error(e)
            logger.warning("Risk Agent failed; degrading gracefully: %s", sanitized)
            risk_fallback = RiskOutput(
                risk_score=0.50,
                risk_level="moderate",
                uncertainty_level="low_confidence",
                factors=[],
                escalate_to_human=True,
                uncertainty_signals_triggered=["RISK_ENGINE_ERROR"],
                weather_unavailable=True,
                rag_unavailable=True,
                progression_note="Risk evaluation error; fallback applied.",
                risk_computed_at=datetime.now(timezone.utc),
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "risk_output": risk_fallback,
                "risk_error": sanitized,
                "partial_result": True,
                "errors": {"risk_reason_runner": sanitized},
                "node_timings_ms": {"risk_reason_runner": elapsed},
                "flags": ["RISK_EVAL_ERROR"],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 11. recommendation_runner
    # ─────────────────────────────────────────────────────────────────────────
    def recommendation_runner(self, state: LeafMindState) -> Dict[str, Any]:
        """Execute Recommendation Agent synthesis."""
        t0 = time.perf_counter()
        v_out = state.get("vision_output")
        w_out = state.get("weather_output")
        r_out = state.get("rag_output")
        rk_out = state.get("risk_output")
        include_doses = bool(state.get("include_doses", False))

        try:
            agent = self.get_recommendation_agent()
            rec_out: RecommendationOutput = agent.recommend(
                vision=v_out,
                rag=r_out,
                weather=w_out,
                risk=rk_out,
                include_doses=include_doses,
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "recommendation_output": rec_out,
                "node_timings_ms": {"recommendation_runner": elapsed},
            }
        except Exception as e:
            # Deterministic safe advisory fallback per spec 4e
            sanitized = sanitize_error(e)
            logger.warning("Recommendation Agent failed; using safe advisory fallback: %s", sanitized)
            rec_fallback = RecommendationOutput(
                summary="Automated recommendation temporarily unavailable. Please consult a local agronomist.",
                urgency="consult_agronomist",
                what_to_do_first="Consult a local agronomist or extension officer for field diagnosis.",
                steps=["Consult a local agronomist or extension officer for field diagnosis."],
                actions=["Consult a local agronomist or extension officer for field diagnosis."],
                citations=[],
                rag_grounded=False,
                escalate_to_human=True,
                llm_used=False,
                fallback_reason=f"recommendation_error: {sanitized}",
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "recommendation_output": rec_fallback,
                "recommendation_error": sanitized,
                "partial_result": True,
                "errors": {"recommendation_runner": sanitized},
                "node_timings_ms": {"recommendation_runner": elapsed},
                "flags": ["RECOMMENDATION_FALLBACK"],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 12. escalation_router
    # ─────────────────────────────────────────────────────────────────────────
    def escalation_router(self, state: LeafMindState) -> Dict[str, Any]:
        """Evaluate escalation triggers T1–T11 and route accordingly."""
        t0 = time.perf_counter()
        m_out = state.get("monitoring_output")
        visit_num = m_out.visit_number if m_out else 1
        esc_summary: EscalationSummary = evaluate_escalation(
            vision=state.get("vision_output"),
            rag=state.get("rag_output"),
            weather=state.get("weather_output"),
            risk=state.get("risk_output"),
            recommendation=state.get("recommendation_output"),
            monitoring=m_out,
            visit_number=visit_num,
            config_path=self.config_path,
            image_bytes=state.get("image_bytes"),
            case_id=state.get("case_id"),
            farmer_id=state.get("farmer_id"),
        )

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "escalate_to_human": esc_summary.escalate_to_human,
            "escalation_summary": esc_summary,
            "escalation_payload": esc_summary.payload,
            "node_timings_ms": {"escalation_router": elapsed},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 13. escalation_handler
    # ─────────────────────────────────────────────────────────────────────────
    def escalation_handler(self, state: LeafMindState) -> Dict[str, Any]:
        """Process agronomist queue payload and log escalation event."""
        t0 = time.perf_counter()
        summary = state.get("escalation_summary")
        if summary and summary.payload:
            logger.info(
                "Case %s escalated with priority %s. Triggers: %s",
                summary.payload.case_id,
                summary.payload.priority,
                summary.triggered_triggers,
            )

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "flags": ["ESCALATED_TO_HUMAN"],
            "node_timings_ms": {"escalation_handler": elapsed},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 14. response_builder
    # ─────────────────────────────────────────────────────────────────────────
    def response_builder(self, state: LeafMindState) -> Dict[str, Any]:
        """Construct unified FinalResponse and legacy FarmerResponse."""
        t0 = time.perf_counter()
        c_id = state.get("case_id") or str(uuid.uuid4())
        c_uuid = uuid.UUID(c_id) if isinstance(c_id, str) else c_id
        f_id = state.get("farmer_id") or "farmer_001"
        sub_at = state.get("submitted_at") or datetime.now(timezone.utc)
        crop = state.get("crop")
        is_followup = bool(state.get("is_followup", False))
        partial = bool(state.get("partial_result", False))

        v_out = state.get("vision_output")
        w_out = state.get("weather_output")
        r_out = state.get("rag_output")
        rk_out = state.get("risk_output")
        rec_out = state.get("recommendation_output")
        mon_out = state.get("monitoring_output")
        esc_summary = state.get("escalation_summary")

        if esc_summary is None:
            esc_summary = EscalationSummary(
                escalate_to_human=False,
                priority="low",
                triggered_triggers=[],
                reasons=[],
                payload=None,
            )

        # Vision Summary
        vision_summary = None
        if v_out is not None:
            candidates = v_out.top_3_candidates or v_out.top_k_probs
            vision_summary = VisionSummary(
                top_label=v_out.top_label,
                confidence=v_out.confidence,
                top_3=candidates[:3] if candidates else [],
                can_diagnose=getattr(v_out, "can_diagnose", True),
                single_class_crop=getattr(v_out, "single_class_crop", False),
                ood_flag=v_out.ood_flag,
                uncertainty_flags=v_out.image_quality_flags,
                severity_pct=None,
                severity_note=v_out.severity_note,
            )

        # Weather Summary
        weather_summary = None
        if w_out is not None:
            triggered = [f.condition_name for f in w_out.disease_risk_flags if f.triggered]
            weather_summary = WeatherSummary(
                weather_available=w_out.weather_available,
                current_temp_c=w_out.current_weather.temperature_c if w_out.current_weather else None,
                current_humidity_pct=w_out.current_weather.relative_humidity_pct if w_out.current_weather else None,
                triggered_rules=triggered,
                weather_score=getattr(w_out, "weather_score", 0.0),
                data_source=w_out.weather_data_source,
                cached=getattr(w_out, "cache_hit", False),
                location_name=getattr(w_out, "location_name", None),
            )

        # RAG Summary
        rag_summary = None
        if r_out is not None:
            c_strs = [c.title for c in r_out.citations] if r_out.citations else []
            rag_summary = RAGSummary(
                found=r_out.found,
                kb_gap=r_out.kb_gap,
                disease_name=r_out.disease_name,
                sources=r_out.citations,
                citations=c_strs,
                chemical_options_count=len(r_out.chemical_options),
                what_to_do_first=r_out.what_to_do_first,
                doses_included=bool(state.get("include_doses", False)),
            )

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        timings = dict(state.get("node_timings_ms", {}))
        timings["response_builder"] = elapsed

        final_resp = FinalResponse(
            case_id=c_uuid,
            farmer_id=f_id,
            submitted_at=sub_at,
            crop=crop,
            is_followup=is_followup,
            previous_case_id=state.get("previous_case_id"),
            vision=vision_summary,
            weather=weather_summary,
            rag=rag_summary,
            risk=rk_out,
            recommendation=rec_out,
            escalation=esc_summary,
            monitoring=mon_out,
            raw_vision=v_out,
            raw_weather=w_out,
            raw_rag=r_out,
            partial_result=partial,
            flags=state.get("flags", []),
            timing_ms=timings,
            errors=state.get("errors", {}),
            fatal_error=state.get("fatal_error"),
            severity_pct=None,
        )

        farmer_resp = None
        if v_out and rk_out and rec_out:
            farmer_resp = FarmerResponse(
                case_id=c_uuid,
                submitted_at=sub_at,
                is_followup=is_followup,
                vision=v_out,
                risk=rk_out,
                recommendation=rec_out,
                monitoring=mon_out,
                escalate_to_human=esc_summary.escalate_to_human,
                partial_result=partial,
            )

        return {
            "final_response": final_resp,
            "farmer_response": farmer_resp,
            "node_timings_ms": {"response_builder": elapsed},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 15. persistence_writer
    # ─────────────────────────────────────────────────────────────────────────
    def persistence_writer(self, state: LeafMindState) -> Dict[str, Any]:
        """Persist CaseState and FinalResponse records to storage."""
        t0 = time.perf_counter()
        final_resp = state.get("final_response")
        if final_resp is not None:
            save_final_response(final_resp, case_store_dir=self.case_store_dir)

        # Assemble CaseState if we have full initial data
        c_id = state.get("case_id")
        c_uuid = uuid.UUID(c_id) if isinstance(c_id, str) else c_id
        f_id = state.get("farmer_id") or "farmer_001"
        sub_at = state.get("submitted_at") or datetime.now(timezone.utc)
        crop = state.get("crop")
        is_followup = bool(state.get("is_followup", False))

        v_out = state.get("vision_output")
        w_out = state.get("weather_output")
        r_out = state.get("rag_output")
        rk_out = state.get("risk_output")
        rec_out = state.get("recommendation_output")

        # Check if this was a follow-up visit with loaded baseline
        loaded_case = state.get("loaded_case_state")
        mon_out = state.get("monitoring_output")

        if loaded_case is not None and mon_out is not None and rk_out is not None and rec_out is not None:
            fu_record = FollowUpRecord(
                visit_number=mon_out.visit_number,
                submitted_at=sub_at,
                monitoring_output=mon_out,
                risk_output=rk_out,
                recommendation_output=rec_out,
            )
            loaded_case.follow_ups.append(fu_record)
            if state.get("escalate_to_human"):
                loaded_case.status = "escalated"
            save_case_record(loaded_case, case_store_dir=self.case_store_dir)

        if v_out and w_out and r_out and rk_out and rec_out and c_uuid:
            req = state.get("farmer_request") or FarmerRequest(
                farmer_id=f_id,
                image_bytes=state.get("image_bytes") or b"",
                image_mime_type="image/jpeg",
                crop_type=crop,
                submitted_at=sub_at,
                is_followup=is_followup,
                case_id=c_uuid,
            )
            case_state = CaseState(
                case_id=c_uuid,
                farmer_id=f_id,
                created_at=sub_at,
                crop_type=crop,
                initial_request=req,
                initial_vision=v_out,
                initial_weather=w_out,
                initial_rag=r_out,
                initial_risk=rk_out,
                initial_recommendation=rec_out,
                follow_ups=[],
                status="escalated" if state.get("escalate_to_human") else "active",
            )
            save_case_record(case_state, case_store_dir=self.case_store_dir)

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {"node_timings_ms": {"persistence_writer": elapsed}}

    # ─────────────────────────────────────────────────────────────────────────
    # 16. error_handler
    # ─────────────────────────────────────────────────────────────────────────
    def error_handler(self, state: LeafMindState) -> Dict[str, Any]:
        """Handle fatal errors gracefully without crashing the pipeline."""
        t0 = time.perf_counter()
        fatal = state.get("fatal_error") or "internal_error"
        c_id = state.get("case_id") or str(uuid.uuid4())
        c_uuid = uuid.UUID(c_id) if isinstance(c_id, str) else c_id
        f_id = state.get("farmer_id") or "farmer_001"
        sub_at = state.get("submitted_at") or datetime.now(timezone.utc)

        logger.warning("Pipeline fatal error reached error_handler: %s", fatal)
        esc_summary = EscalationSummary(
            escalate_to_human=True,
            priority="high",
            triggered_triggers=[f"FATAL_{fatal.upper()}"],
            reasons=[],
            payload=None,
        )

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        timings = dict(state.get("node_timings_ms", {}))
        timings["error_handler"] = elapsed

        final_resp = FinalResponse(
            case_id=c_uuid,
            farmer_id=f_id,
            submitted_at=sub_at,
            crop=state.get("crop"),
            is_followup=bool(state.get("is_followup", False)),
            vision=None,
            weather=None,
            rag=None,
            risk=None,
            recommendation=None,
            escalation=esc_summary,
            monitoring=None,
            partial_result=True,
            flags=state.get("flags", []) + [f"FATAL_{fatal.upper()}"],
            timing_ms=timings,
            errors=state.get("errors", {}),
            fatal_error=fatal,
            severity_pct=None,
        )

        return {
            "final_response": final_resp,
            "node_timings_ms": {"error_handler": elapsed},
            "flags": [f"FATAL_{fatal.upper()}"],
        }
