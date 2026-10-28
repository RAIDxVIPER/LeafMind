"""
src/orchestrator/graph.py
=========================
LangGraph workflow definition for LeafMind orchestrator (Stage C7).

Implements:
  - Spec 3c-iii: Complete DAG with 16 nodes, deterministic transitions, and conditional edges
  - Spec 3c-i: LangGraph StateGraph with checkpointing support
  - Parallel fan-out: [weather_runner, rag_runner] execution converging at fan_in_barrier
  - Graceful degradation: fatal errors route cleanly to error_handler
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.agents.recommendation_agent import RecommendationAgent
from src.agents.risk_agent import RiskAgent
from src.agents.vision_agent import VisionAgent
from src.agents.weather_agent import WeatherAgent
from src.orchestrator.nodes import OrchestratorNodes
from src.orchestrator.state import LeafMindState
from src.orchestrator.store import get_checkpointer
from src.rag.agent import RAGAgent

logger = logging.getLogger(__name__)


def create_orchestrator_graph(
    vision_agent: Optional[VisionAgent] = None,
    weather_agent: Optional[WeatherAgent] = None,
    rag_agent: Optional[RAGAgent] = None,
    risk_agent: Optional[RiskAgent] = None,
    recommendation_agent: Optional[RecommendationAgent] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    use_memory: bool = False,
    db_path: Optional[str | Path] = None,
    case_store_dir: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
):
    """
    Construct and compile the LeafMind StateGraph orchestrator.

    Args:
        vision_agent: Optional VisionAgent instance (real or mock)
        weather_agent: Optional WeatherAgent instance (real or mock)
        rag_agent: Optional RAGAgent instance (real or mock)
        risk_agent: Optional RiskAgent instance (real or mock)
        recommendation_agent: Optional RecommendationAgent instance (real or mock)
        checkpointer: Pre-configured BaseCheckpointSaver or None
        use_memory: If True, uses MemorySaver (fast unit tests)
        db_path: Path to SQLite DB for checkpointing
        case_store_dir: Directory for case records
        config_path: Path to orchestrator_config.yaml

    Returns:
        Compiled LangGraph Pregel executable graph
    """
    nodes = OrchestratorNodes(
        vision_agent=vision_agent,
        weather_agent=weather_agent,
        rag_agent=rag_agent,
        risk_agent=risk_agent,
        recommendation_agent=recommendation_agent,
        case_store_dir=case_store_dir,
        config_path=config_path,
    )

    graph_builder = StateGraph(LeafMindState)

    # 1. Register all 16 nodes
    graph_builder.add_node("input_validator", nodes.input_validator)
    graph_builder.add_node("route_request", nodes.route_request)
    graph_builder.add_node("case_initialiser", nodes.case_initialiser)
    graph_builder.add_node("case_loader", nodes.case_loader)
    graph_builder.add_node("vision_runner", nodes.vision_runner)
    graph_builder.add_node("weather_runner", nodes.weather_runner)
    graph_builder.add_node("rag_runner", nodes.rag_runner)
    graph_builder.add_node("fan_in_barrier", nodes.fan_in_barrier)
    graph_builder.add_node("monitoring_runner", nodes.monitoring_runner)
    graph_builder.add_node("risk_reason_runner", nodes.risk_reason_runner)
    graph_builder.add_node("recommendation_runner", nodes.recommendation_runner)
    graph_builder.add_node("escalation_router", nodes.escalation_router)
    graph_builder.add_node("escalation_handler", nodes.escalation_handler)
    graph_builder.add_node("response_builder", nodes.response_builder)
    graph_builder.add_node("persistence_writer", nodes.persistence_writer)
    graph_builder.add_node("error_handler", nodes.error_handler)

    # 2. Define conditional router functions
    def route_after_input_validation(state: LeafMindState) -> str:
        if state.get("fatal_error"):
            return "error_handler"
        return "route_request"

    def route_after_request_routing(state: LeafMindState) -> str:
        if state.get("is_followup"):
            return "case_loader"
        return "case_initialiser"

    def route_after_case_loader(state: LeafMindState) -> str:
        if state.get("fatal_error"):
            return "error_handler"
        return "vision_runner"

    def route_after_vision(state: LeafMindState) -> List[str]:
        if state.get("fatal_error"):
            return ["error_handler"]
        # Parallel fan-out to weather and rag
        return ["weather_runner", "rag_runner"]

    def route_after_escalation(state: LeafMindState) -> str:
        if state.get("escalate_to_human"):
            return "escalation_handler"
        return "response_builder"

    # 3. Add Edges
    # START -> input_validator
    graph_builder.add_edge(START, "input_validator")

    # input_validator -> route_request OR error_handler
    graph_builder.add_conditional_edges(
        "input_validator",
        route_after_input_validation,
        {
            "route_request": "route_request",
            "error_handler": "error_handler",
        },
    )

    # route_request -> case_initialiser OR case_loader
    graph_builder.add_conditional_edges(
        "route_request",
        route_after_request_routing,
        {
            "case_initialiser": "case_initialiser",
            "case_loader": "case_loader",
        },
    )

    # case_initialiser -> vision_runner
    graph_builder.add_edge("case_initialiser", "vision_runner")

    # case_loader -> vision_runner OR error_handler
    graph_builder.add_conditional_edges(
        "case_loader",
        route_after_case_loader,
        {
            "vision_runner": "vision_runner",
            "error_handler": "error_handler",
        },
    )

    # vision_runner -> [weather_runner, rag_runner] OR error_handler
    graph_builder.add_conditional_edges(
        "vision_runner",
        route_after_vision,
        {
            "weather_runner": "weather_runner",
            "rag_runner": "rag_runner",
            "error_handler": "error_handler",
        },
    )

    # weather_runner & rag_runner -> fan_in_barrier
    graph_builder.add_edge("weather_runner", "fan_in_barrier")
    graph_builder.add_edge("rag_runner", "fan_in_barrier")

    # fan_in_barrier -> monitoring_runner -> risk_reason_runner -> recommendation_runner -> escalation_router
    graph_builder.add_edge("fan_in_barrier", "monitoring_runner")
    graph_builder.add_edge("monitoring_runner", "risk_reason_runner")
    graph_builder.add_edge("risk_reason_runner", "recommendation_runner")
    graph_builder.add_edge("recommendation_runner", "escalation_router")

    # escalation_router -> escalation_handler OR response_builder
    graph_builder.add_conditional_edges(
        "escalation_router",
        route_after_escalation,
        {
            "escalation_handler": "escalation_handler",
            "response_builder": "response_builder",
        },
    )

    # escalation_handler -> response_builder
    graph_builder.add_edge("escalation_handler", "response_builder")

    # response_builder -> persistence_writer -> END
    graph_builder.add_edge("response_builder", "persistence_writer")
    graph_builder.add_edge("persistence_writer", END)

    # error_handler -> END
    graph_builder.add_edge("error_handler", END)

    # 4. Resolve checkpointer and compile
    cp = checkpointer
    if cp is None:
        cp = get_checkpointer(
            db_path=db_path,
            use_memory=use_memory,
            config_path=config_path,
        )

    compiled_graph = graph_builder.compile(checkpointer=cp)
    logger.debug("Compiled LeafMind StateGraph successfully")
    return compiled_graph
