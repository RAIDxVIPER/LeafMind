"""
src/api/routes.py
=================
FastAPI routes for LeafMind API.
Endpoints:
  - GET /health
  - GET /meta
  - POST /analyze
  - GET /case/{case_id}
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from src.api.config import load_api_config
from src.api.formatting import (
    FARMER_CAPTURE_TIPS,
    STANDARD_DISCLAIMERS,
    build_display_fields,
    build_safe_image_stub,
    sanitize_for_api,
)
from src.api.schemas import (
    AnalyzeResponse,
    HealthResponse,
    MetaResponse,
    Top3Item,
)
from src.api.validation import (
    MULTI_CLASS_CROPS,
    SINGLE_CLASS_CROPS,
    SUPPORTED_CROPS,
    resolve_crop,
    validate_and_sanitize_image,
)
from src.orchestrator.nodes import sanitize_error
from src.orchestrator.pipeline import run_case
from src.orchestrator.store import load_case_record, load_final_response

logger = logging.getLogger("leafmind.api.routes")
router = APIRouter()

# Shared thread pool for running CPU/inference case runs
_THREAD_POOL = ThreadPoolExecutor(max_workers=4)


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Service health check."""
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/meta", response_model=MetaResponse)
async def get_meta() -> MetaResponse:
    """
    Metadata endpoint returning supported crops, capture tips, and disclaimers.
    Single source of truth for UI selectboxes and farmer guidance.
    """
    return MetaResponse(
        supported_crops=list(SUPPORTED_CROPS),
        single_class_crops=list(SINGLE_CLASS_CROPS),
        multi_class_crops=list(MULTI_CLASS_CROPS),
        capture_tips=list(FARMER_CAPTURE_TIPS),
        disclaimers=list(STANDARD_DISCLAIMERS),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_leaf(
    request: Request,
    image: UploadFile = File(..., description="Leaf image file (JPEG, PNG, or WebP)"),
    crop: str = Form(..., description="Crop name (e.g. Tomato, Corn, Apple)"),
    location: Optional[str] = Form(None, description="Place name or region (e.g. Pune, Maharashtra)"),
    lat: Optional[float] = Form(None, description="GPS Latitude"),
    lon: Optional[float] = Form(None, description="GPS Longitude"),
    previous_case_id: Optional[str] = Form(None, description="Previous Case ID for follow-up monitoring"),
    include_doses: bool = Form(False, description="Whether to include chemical dose rates (default: False)"),
) -> AnalyzeResponse:
    """
    Execute full multi-agent diagnostic and advisory pipeline on a leaf photo.
    Returns the FinalResponse contract plus farmer display-ready fields.
    """
    cfg = load_api_config()
    limits_cfg = cfg.get("limits", {})
    max_mb = float(limits_cfg.get("max_image_size_mb", 10.0))
    min_dim = int(limits_cfg.get("min_dimension_px", 64))
    timeout_sec = float(cfg.get("server", {}).get("timeout_seconds", 60.0))
    guard_thresh = float(cfg.get("guards", {}).get("healthy_runner_up_threshold", 0.25))

    # 1. Validate Crop
    resolved_crop = resolve_crop(crop)

    # 2. Read and Validate Image
    try:
        raw_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to read uploaded image file.",
        ) from exc

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded image file is empty. Please capture and upload a leaf photo.",
        )

    sanitized_bytes, w, h, fmt = validate_and_sanitize_image(
        raw_bytes=raw_bytes,
        max_mb=max_mb,
        min_dimension=min_dim,
    )

    # 3. Location handling
    loc_param = None
    if lat is not None and lon is not None:
        loc_param = (lat, lon)
    elif location:
        loc_param = location.strip()

    # 4. Execute Pipeline in Thread Pool with Timeout
    def _execute_run_case():
        return run_case(
            image=sanitized_bytes,
            crop=resolved_crop,
            location=loc_param,
            include_doses=include_doses,
            previous_case_id=previous_case_id,
            use_memory=True,
        )

    loop = asyncio.get_running_loop()
    try:
        final_resp = await asyncio.wait_for(
            loop.run_in_executor(_THREAD_POOL, _execute_run_case),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.error("Analyze request timed out after %.1fs", timeout_sec)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Analysis timed out after {timeout_sec:.0f} seconds. Please retry your request.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        sanitized = sanitize_error(exc)
        logger.error("Internal error during analysis: %s", sanitized)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during case analysis: {sanitized}",
        ) from exc

    # 5. Format display-ready fields
    display_fields = build_display_fields(
        final_resp=final_resp,
        healthy_guard_threshold=guard_thresh,
    )

    top3_items = [
        Top3Item(name=item["name"], probability=item["probability"])
        for item in display_fields["top3"]
    ]

    # Combine FinalResponse model fields with display fields
    raw_resp_dict = final_resp.model_dump(mode="python")

    # Sanitize escalation payload to ensure no raw bytes reach the response
    if raw_resp_dict.get("escalation") and isinstance(raw_resp_dict["escalation"], dict):
        esc_dict = raw_resp_dict["escalation"]
        if esc_dict.get("payload") and isinstance(esc_dict["payload"], dict):
            payload_dict = esc_dict["payload"]
            payload_dict["image_bytes"] = build_safe_image_stub(payload_dict.get("image_bytes"))

    raw_resp_dict.update({
        "farmer_summary": display_fields["farmer_summary"],
        "top3": top3_items,
        "risk_label": display_fields["risk_label"],
        "urgency": display_fields["urgency"],
        "steps": display_fields["steps"],
        "warnings": display_fields["warnings"],
        "disclaimers": display_fields["disclaimers"],
        "escalation_banner": display_fields["escalation_banner"],
    })

    try:
        response_model = AnalyzeResponse(**raw_resp_dict)
        # Pre-verify that response serializes to JSON cleanly without error
        _ = response_model.model_dump(mode="json")
    except Exception as exc:
        sanitized = sanitize_error(exc)
        logger.error("AnalyzeResponse serialization failure: %s", sanitized)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error: Failed to serialize analysis response.",
        ) from exc

    return response_model


@router.get("/case/{case_id}")
async def get_case(case_id: str) -> Dict[str, Any]:
    """Retrieve persisted case record by case_id safely without raw bytes."""
    clean_id = case_id.strip()
    try:
        case_uuid = uuid.UUID(clean_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )

    try:
        # Attempt to load case record from store
        case_record = load_case_record(case_uuid)
        if case_record is not None:
            raw_data = case_record.model_dump(mode="python")
            safe_data = sanitize_for_api(raw_data)
            # Ensure JSON-serializable
            import json
            _ = json.dumps(safe_data)
            return safe_data

        # Attempt to load final response if case record was not found
        final_resp = load_final_response(case_uuid)
        if final_resp is not None:
            raw_data = final_resp.model_dump(mode="python")
            safe_data = sanitize_for_api(raw_data)
            import json
            _ = json.dumps(safe_data)
            return safe_data
    except HTTPException:
        raise
    except Exception as exc:
        sanitized = sanitize_error(exc)
        logger.error("Error retrieving case %s: %s", clean_id, sanitized)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error: Failed to retrieve case record.",
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Case '{case_id}' not found.",
    )

