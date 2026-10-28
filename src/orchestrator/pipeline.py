"""
src/orchestrator/pipeline.py
============================
High-level entry point function `run_case` for the LeafMind orchestrator (Stage C7).

Implements:
  - run_case(image, crop, location=None, include_doses=False, case_id=None) -> FinalResponse
  - Automatic reading of image file paths or raw bytes
  - D17: user-supplied crop
  - D24: location handling (GPS, place name, or default Maharashtra)
  - Thread ID tracking via case_id for LangGraph SQLite checkpointing
  - Case reloading via load_case_response / load_case_state
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import uuid

from langgraph.checkpoint.base import BaseCheckpointSaver

from src.agents.monitoring_agent import MonitoringAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.risk_agent import RiskAgent
from src.agents.vision_agent import VisionAgent
from src.agents.weather_agent import WeatherAgent
from src.orchestrator.graph import create_orchestrator_graph
from src.orchestrator.state import FinalResponse, LeafMindState
from src.orchestrator.store import load_case_record, load_final_response
from src.rag.agent import RAGAgent
from src.schemas.schemas import CaseState, FarmerRequest

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_case(
    image: Union[bytes, str, Path],
    crop: Optional[str] = None,
    location: Optional[Union[str, Tuple[float, float], Dict[str, Any]]] = None,
    include_doses: bool = False,
    case_id: Optional[Union[str, uuid.UUID]] = None,
    farmer_id: Optional[str] = "farmer_001",
    is_followup: bool = False,
    previous_case_id: Optional[Union[str, uuid.UUID]] = None,
    vision_agent: Optional[VisionAgent] = None,
    weather_agent: Optional[WeatherAgent] = None,
    rag_agent: Optional[RAGAgent] = None,
    risk_agent: Optional[RiskAgent] = None,
    recommendation_agent: Optional[RecommendationAgent] = None,
    monitoring_agent: Optional[MonitoringAgent] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    use_memory: bool = False,
    db_path: Optional[Union[str, Path]] = None,
    case_store_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> FinalResponse:
    """
    Execute full multi-agent diagnostic and advisory pipeline on a leaf image.

    Args:
        image: raw image bytes or path to JPEG/PNG image file
        crop: optional user-supplied crop name (D17, e.g. "Tomato", "Apple", "Corn", "Squash")
        location: GPS tuple (lat, lon), dict, place name string, or None (defaults to Maharashtra per D24)
        include_doses: whether chemical doses are permitted (D23/D28, default False)
        case_id: optional UUID or string identifying the case
        farmer_id: optional farmer identifier
        is_followup: whether this submission is a follow-up visit (default False)
        vision_agent: optional injected VisionAgent (for unit tests / mock setups)
        weather_agent: optional injected WeatherAgent (for unit tests / mock setups)
        rag_agent: optional injected RAGAgent (for unit tests / mock setups)
        risk_agent: optional injected RiskAgent (for unit tests / mock setups)
        recommendation_agent: optional injected RecommendationAgent (for unit tests / mock setups)
        checkpointer: optional pre-configured BaseCheckpointSaver
        use_memory: whether to use in-memory MemorySaver (for unit tests)
        db_path: path to SQLite checkpoint database
        case_store_dir: directory to persist CaseState JSON records
        config_path: path to orchestrator_config.yaml

    Returns:
        Structured FinalResponse object.
    """
    # 1. Normalize image bytes
    image_bytes: bytes
    if isinstance(image, (str, Path)):
        img_path = Path(image)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image_bytes = img_path.read_bytes()
    elif isinstance(image, bytes):
        image_bytes = image
    else:
        raise TypeError(f"Expected bytes or file path for image, got {type(image)}")

    # 2. Normalize case identity
    case_uuid: uuid.UUID
    if case_id is None:
        case_uuid = uuid.uuid4()
    elif isinstance(case_id, str):
        try:
            case_uuid = uuid.UUID(case_id)
        except ValueError:
            case_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, case_id)
    else:
        case_uuid = case_id
    case_id_str = str(case_uuid)

    now_utc = datetime.now(timezone.utc)
    if previous_case_id:
        is_followup = True
    prev_case_id_str = str(previous_case_id) if previous_case_id else None

    # 3. Compile graph
    compiled_graph = create_orchestrator_graph(
        vision_agent=vision_agent,
        weather_agent=weather_agent,
        rag_agent=rag_agent,
        risk_agent=risk_agent,
        recommendation_agent=recommendation_agent,
        monitoring_agent=monitoring_agent,
        checkpointer=checkpointer,
        use_memory=use_memory,
        db_path=db_path,
        case_store_dir=case_store_dir,
        config_path=config_path,
    )

    # 4. Prepare initial state
    initial_state: LeafMindState = {
        "image_bytes": image_bytes,
        "crop": crop,
        "location": location,
        "include_doses": include_doses,
        "case_id": case_id_str,
        "previous_case_id": prev_case_id_str,
        "farmer_id": farmer_id or "farmer_001",
        "submitted_at": now_utc,
        "is_followup": is_followup,
        "flags": [],
        "errors": {},
        "node_timings_ms": {},
        "partial_result": False,
        "escalate_to_human": False,
    }

    # 5. Execute graph with checkpoint thread_id = case_id
    config = {"configurable": {"thread_id": case_id_str}}
    final_state = compiled_graph.invoke(initial_state, config=config)

    final_resp: Optional[FinalResponse] = final_state.get("final_response")
    if final_resp is None:
        raise RuntimeError("Graph execution finished without producing a FinalResponse.")

    return final_resp


def load_case_response(
    case_id: Union[str, uuid.UUID],
    case_store_dir: Optional[Union[str, Path]] = None,
) -> Optional[FinalResponse]:
    """Reload an existing FinalResponse from local JSON persistence."""
    return load_final_response(case_id, case_store_dir=case_store_dir)


def load_case_state(
    case_id: Union[str, uuid.UUID],
    case_store_dir: Optional[Union[str, Path]] = None,
) -> Optional[CaseState]:
    """Reload an existing CaseState record from local JSON persistence."""
    return load_case_record(case_id, case_store_dir=case_store_dir)
