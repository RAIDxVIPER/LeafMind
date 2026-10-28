"""
src/orchestrator/store.py
=========================
Persistence layer for LeafMind Orchestrator (Stage C7).

Provides:
  - LangGraph checkpoint persistence via SqliteSaver (persisting node state across steps)
  - Business Case persistence (CaseState and FinalResponse) to JSON files under data/cases/
  - Reloading by case_id
"""
from __future__ import annotations

import base64
from datetime import date, datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional
import uuid

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.orchestrator.escalation import load_orchestrator_config
from src.orchestrator.state import FinalResponse
from src.schemas.schemas import CaseState

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "checkpoints.db"
DEFAULT_CASES_DIR = REPO_ROOT / "data" / "cases"

# Keep an active connection cache so SQLite does not lock or reconnect unnecessarily
_SQLITE_CONNECTIONS: Dict[str, sqlite3.Connection] = {}


def close_sqlite_connections() -> None:
    """Close and clear all cached SQLite connections (useful for clean teardown on Windows)."""
    for path, conn in list(_SQLITE_CONNECTIONS.items()):
        try:
            conn.close()
        except Exception:
            pass
    _SQLITE_CONNECTIONS.clear()


def safe_model_dump_json(model: Any, indent: int = 2) -> str:
    """
    Serialize Pydantic models to JSON safely handling raw binary bytes
    (encoded to base64) and datetime/UUID objects.
    """
    def convert(obj: Any) -> Any:
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("ascii")
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, "hex") and hasattr(obj, "int"):  # UUID
            return str(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    raw_dict = model.model_dump(mode="python")
    safe_dict = convert(raw_dict)
    return json.dumps(safe_dict, indent=indent)


def get_checkpointer(
    db_path: Optional[str | Path] = None,
    use_memory: bool = False,
    config_path: Optional[Path | str] = None,
) -> BaseCheckpointSaver:
    """
    Get or create a LangGraph checkpointer.

    Args:
        db_path: Optional SQLite file path. If None, loaded from orchestrator_config.yaml
        use_memory: If True, uses in-memory MemorySaver (useful for fast unit tests)
        config_path: Optional override for orchestrator config
    """
    if use_memory:
        logger.debug("Using in-memory LangGraph checkpointer")
        return MemorySaver()

    if db_path is None:
        cfg = load_orchestrator_config(config_path)
        pers_cfg = cfg.get("persistence", {})
        configured_path = pers_cfg.get("sqlite_db_path")
        if configured_path:
            db_path = REPO_ROOT / configured_path
        else:
            db_path = DEFAULT_DB_PATH

    db_str = str(db_path)
    if db_str == ":memory:":
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver

    path_obj = Path(db_path).resolve()
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if str(path_obj) not in _SQLITE_CONNECTIONS:
        conn = sqlite3.connect(str(path_obj), check_same_thread=False)
        _SQLITE_CONNECTIONS[str(path_obj)] = conn
    else:
        conn = _SQLITE_CONNECTIONS[str(path_obj)]

    saver = SqliteSaver(conn)
    saver.setup()
    logger.debug("SqliteSaver ready at %s", path_obj)
    return saver


def get_case_store_dir(
    case_store_dir: Optional[str | Path] = None,
    config_path: Optional[Path | str] = None,
) -> Path:
    """Resolve and ensure the directory for persisting case JSON files."""
    if case_store_dir is not None:
        p = Path(case_store_dir).resolve()
    else:
        cfg = load_orchestrator_config(config_path)
        configured_dir = cfg.get("persistence", {}).get("case_store_dir")
        if configured_dir:
            p = (REPO_ROOT / configured_dir).resolve()
        else:
            p = DEFAULT_CASES_DIR.resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_case_record(
    case_state: CaseState,
    case_store_dir: Optional[str | Path] = None,
) -> Path:
    """Persist a CaseState record to data/cases/{case_id}.json."""
    store_dir = get_case_store_dir(case_store_dir)
    file_path = store_dir / f"{case_state.case_id}.json"
    content = safe_model_dump_json(case_state, indent=2)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug("Saved CaseState record to %s", file_path)
    return file_path


def load_case_record(
    case_id: str | uuid.UUID,
    case_store_dir: Optional[str | Path] = None,
) -> Optional[CaseState]:
    """Load a CaseState record from data/cases/{case_id}.json."""
    store_dir = get_case_store_dir(case_store_dir)
    file_path = store_dir / f"{case_id}.json"
    if not file_path.is_file():
        logger.info("Case record not found: %s", file_path)
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "initial_request" in data and isinstance(data["initial_request"], dict):
            img_b64 = data["initial_request"].get("image_bytes")
            if isinstance(img_b64, str):
                try:
                    data["initial_request"]["image_bytes"] = base64.b64decode(img_b64)
                except Exception:
                    pass
        return CaseState.model_validate(data)
    except Exception as e:
        logger.error("Failed to parse CaseState record at %s: %s", file_path, e)
        return None


def save_final_response(
    response: FinalResponse,
    case_store_dir: Optional[str | Path] = None,
) -> Path:
    """Persist a FinalResponse to data/cases/{case_id}_response.json."""
    store_dir = get_case_store_dir(case_store_dir)
    file_path = store_dir / f"{response.case_id}_response.json"
    content = safe_model_dump_json(response, indent=2)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug("Saved FinalResponse to %s", file_path)
    return file_path


def load_final_response(
    case_id: str | uuid.UUID,
    case_store_dir: Optional[str | Path] = None,
) -> Optional[FinalResponse]:
    """Load a FinalResponse from data/cases/{case_id}_response.json."""
    store_dir = get_case_store_dir(case_store_dir)
    file_path = store_dir / f"{case_id}_response.json"
    if not file_path.is_file():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FinalResponse.model_validate(data)
    except Exception as e:
        logger.error("Failed to load FinalResponse at %s: %s", file_path, e)
        return None
