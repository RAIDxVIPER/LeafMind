"""
src/monitoring/comparator.py
============================
Pre-analysis plausibility checks and delta signal computations for LeafMind
Stage C8 Monitoring Agent.

Strictly follows:
  - Spec Part 4c: disease_status determination & plausibility checks
  - Spec Part 6 6d-vii: Pre-analysis sequence & delta calculation
  - D2/R3: severity_delta_pct is ALWAYS None
  - D4/I21: disease_status and confidence_raw_delta are separate signals;
            confidence is never conflated with efficacy.
  - Plant identity disclaimer: The system cannot verify physical plant identity.
"""
from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import logging
from typing import Any, Dict, Optional, Tuple

from src.schemas.schemas import CaseState, DetectionChange, MonitoringOutput, VisionOutput

logger = logging.getLogger(__name__)


def _normalize_image_bytes(data: bytes) -> bytes:
    """Normalize bytes in case loaded JSON left base64 ASCII encoded bytes."""
    if not data:
        return b""
    try:
        if (data.startswith(b"/9j/") or data.startswith(b"iVBOR") or b"=" in data) and len(data) % 4 == 0:
            decoded = base64.b64decode(data, validate=True)
            if decoded:
                return decoded
    except Exception:
        pass
    return data


def compute_image_md5(data: Optional[bytes]) -> Optional[str]:
    """Compute MD5 hex digest of image bytes."""
    if not data or len(data) == 0:
        return None
    normalized = _normalize_image_bytes(data)
    return hashlib.md5(normalized).hexdigest()


def normalize_crop_name(name: Optional[str]) -> Optional[str]:
    """Normalize crop name or prefix (e.g., 'Corn_(maize)' -> 'corn', 'Tomato' -> 'tomato')."""
    if not name:
        return None
    clean = name.split("___")[0].strip().lower()
    clean = clean.replace("_(maize)", "").replace(",_bell", "")
    return clean


def check_same_image(
    current_bytes: Optional[bytes],
    baseline_bytes: Optional[bytes],
) -> bool:
    """
    Check if current follow-up image is byte-identical (MD5 match) to baseline.
    Spec 4c-iii check 1.
    """
    if not current_bytes or not baseline_bytes:
        return False
    return compute_image_md5(current_bytes) == compute_image_md5(baseline_bytes)


def check_poor_quality(vision: VisionOutput) -> bool:
    """
    Check if follow-up image fails quality gates (blurry or low resolution).
    Spec 4c-iii check 2.
    """
    if vision.image_quality_flags.blurry or vision.image_quality_flags.low_resolution:
        return True
    return False


def check_crop_mismatch(
    current_crop: Optional[str],
    current_label: str,
    baseline_crop: Optional[str],
    baseline_label: str,
) -> bool:
    """
    Check if follow-up image represents a different crop from the baseline.
    Spec 4c-iii check 3.
    """
    c_curr = normalize_crop_name(current_crop) or normalize_crop_name(current_label)
    c_base = normalize_crop_name(baseline_crop) or normalize_crop_name(baseline_label)
    if c_curr and c_base and c_curr != c_base:
        return True
    return False


def check_too_soon(
    current_submitted_at: Optional[datetime],
    baseline_submitted_at: Optional[datetime],
) -> bool:
    """
    Check if follow-up image was submitted on the same calendar day.
    Spec 4c-iii check 5.
    """
    if not current_submitted_at or not baseline_submitted_at:
        return False
    return current_submitted_at.date() == baseline_submitted_at.date()


def compute_monitoring_delta(
    current_vision: VisionOutput,
    baseline_case: CaseState,
    current_image_bytes: Optional[bytes] = None,
    current_crop: Optional[str] = None,
    current_submitted_at: Optional[datetime] = None,
    persistent_disease_min_visits: int = 2,
    extended_persistence_min_visits: int = 3,
) -> MonitoringOutput:
    """
    Execute full pre-analysis plausibility checks, delta computation,
    and disease status assessment per Spec 4c and 6d-vii.

    Args:
        current_vision: VisionOutput of follow-up image
        baseline_case: Persisted CaseState of baseline case
        current_image_bytes: Optional raw bytes of follow-up image for MD5 check
        current_crop: Optional user-supplied crop for follow-up
        current_submitted_at: Optional submission timestamp of follow-up
        persistent_disease_min_visits: Minimum visit count for T9 high escalation
        extended_persistence_min_visits: Minimum visit count for T9 urgent escalation

    Returns:
        Validated MonitoringOutput instance.
    """
    baseline_vision = baseline_case.initial_vision
    if baseline_vision is None:
        raise ValueError("Baseline CaseState has no initial_vision.")

    baseline_label = baseline_vision.top_label
    followup_label = current_vision.top_label
    label_same = (followup_label == baseline_label)

    # Baseline image bytes & timestamps
    baseline_bytes = (
        baseline_case.initial_request.image_bytes
        if baseline_case.initial_request
        else None
    )
    baseline_submitted_at = (
        baseline_case.created_at
        or (baseline_case.initial_request.submitted_at if baseline_case.initial_request else None)
    )

    # 1. Pre-Analysis Plausibility Checks (Spec 4c-iii)
    same_image_flag = check_same_image(current_image_bytes, baseline_bytes)
    poor_quality_flag = check_poor_quality(current_vision)
    crop_mismatch_flag = check_crop_mismatch(
        current_crop=current_crop,
        current_label=followup_label,
        baseline_crop=baseline_case.crop_type,
        baseline_label=baseline_label,
    )
    too_soon_flag = check_too_soon(current_submitted_at, baseline_submitted_at)

    # Determine visit number: only prior valid follow-up visits (not duplicate images,
    # crop mismatches, or poor quality) count toward persistence visit-count.
    valid_prior_followups = sum(
        1 for fu in baseline_case.follow_ups
        if (
            hasattr(fu, "monitoring_output")
            and fu.monitoring_output is not None
            and not getattr(fu.monitoring_output, "same_image_flag", False)
            and not getattr(fu.monitoring_output, "crop_mismatch_flag", False)
            and not getattr(fu.monitoring_output, "poor_quality_flag", False)
            and getattr(fu.monitoring_output, "disease_status", None) not in ("unknown", None)
        ) or (
            not hasattr(fu, "monitoring_output")
            and getattr(fu, "disease_status", None) not in ("unknown", None)
        )
    )
    visit_number = 2 + valid_prior_followups

    # 2. Delta Signal Computation (Spec 4c-iv / D4 / I21)
    disease_was_present = ("healthy" not in baseline_label.lower())
    new_label_is_healthy = ("healthy" in followup_label.lower())
    disease_still_detected = ("healthy" not in followup_label.lower())

    confidence_raw_delta = round(
        float(current_vision.confidence - baseline_vision.confidence), 4
    )

    # 3. Disease Status Determination (Spec 4c-v / 6d-vii)
    # If any plausibility check invalidates the comparison (same image resubmission,
    # crop mismatch, or poor quality image), or if vision cannot diagnose,
    # disease_status is strictly "unknown" (Spec 4c-iii/v, C8 Fix Pass).
    if same_image_flag or crop_mismatch_flag or poor_quality_flag:
        disease_status = "unknown"
    elif not getattr(current_vision, "can_diagnose", True) or followup_label.lower() in ("unknown", "", "none"):
        disease_status = "unknown"
    elif not disease_was_present:
        disease_status = "not_applicable_healthy_baseline"
    elif new_label_is_healthy:
        disease_status = "no_disease_detected_at_followup"
    elif disease_still_detected and label_same:
        disease_status = "same_disease_still_detected"
    elif disease_still_detected and not label_same:
        disease_status = "different_disease_detected_at_followup"
    else:
        disease_status = "unknown"

    detection_change = DetectionChange(
        baseline_detected=disease_was_present,
        followup_detected=disease_still_detected,
    )

    # 4. Escalation Trigger Assessment (Spec 4c-vi)
    escalate_to_human = False
    if same_image_flag:
        # Duplicate image resubmission triggers escalation for human review
        escalate_to_human = True
    elif poor_quality_flag or crop_mismatch_flag:
        # Poor quality (T10) or crop mismatch (T11) triggers escalation
        escalate_to_human = True
    elif disease_status == "different_disease_detected_at_followup":
        escalate_to_human = True
    elif disease_status == "same_disease_still_detected" and visit_number >= persistent_disease_min_visits:
        escalate_to_human = True
    elif current_vision.escalate_to_human:
        escalate_to_human = True

    return MonitoringOutput(
        case_id=baseline_case.case_id,
        visit_number=visit_number,
        baseline_label=baseline_label,
        baseline_confidence=baseline_vision.confidence,
        followup_label=followup_label,
        followup_confidence=current_vision.confidence,
        label_same=label_same,
        disease_still_detected=disease_still_detected,
        new_label_is_healthy=new_label_is_healthy,
        confidence_raw_delta=confidence_raw_delta,
        disease_status=disease_status,
        detection_change=detection_change,
        severity_delta_pct=None,
        same_image_flag=same_image_flag,
        poor_quality_flag=poor_quality_flag,
        crop_mismatch_flag=crop_mismatch_flag,
        too_soon_flag=too_soon_flag,
        escalate_to_human=escalate_to_human,
        followup_vision=current_vision,
    )
