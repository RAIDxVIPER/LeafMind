"""
src/api/formatting.py
=====================
Display-ready field formatting for LeafMind API.
Strips internal source tags (e.g. (T2), (VER), (CAR1), (IH26)) from farmer-facing text
without modifying internal citations or agent outputs.
"""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import re
from typing import Any, Dict, List, Optional
import uuid
from src.orchestrator.state import FinalResponse
from src.schemas.schemas import ClassProb

# Regex to remove citation tags from farmer-facing strings only
RE_INTERNAL_TAG = re.compile(
    r"\s*\((?:[A-Z]+[0-9]+|[0-9]+[A-Z]+|VER|CIBRC|CIB&RC|CAR\d*|IH\d*|FAO\d*|CABI\d*|T\d+|W\d+|V\d+|U\d+|GAP)[^\)]*\)",
    re.IGNORECASE,
)

STANDARD_DISCLAIMERS = [
    "Advisory only: this report provides automated decision support. Consult a qualified local agronomist or extension officer before taking critical crop decisions.",
    "Severity percentage is not measured: no lesion-mask or affected area percentage is computed [D2/R3].",
    "Provisional ratings: disease threat tiers and weather disease-risk rules are engineering approximations pending formal plant pathologist sign-off [I14/I19].",
    "Plant identity unverified: the system cannot verify whether follow-up photos depict the same physical leaf or plant.",
]

FARMER_CAPTURE_TIPS = [
    "Take photos in bright, indirect daylight (avoid harsh direct sunlight or deep shadow).",
    "Hold your camera steady to prevent blur; ensure the leaf surface and lesion margins are in sharp focus.",
    "Frame the leaf so it fills at least 70% of the picture, showing both healthy and symptomatic tissue.",
    "Avoid extreme close-ups that obscure leaf shape, and avoid distant shots of the whole plant.",
]

# Neutral, non-efficacy wording for follow-up monitoring statuses (D4, I21).
# Strictly avoids implying clinical improvement or worsening from classifier confidence.
MONITORING_STATUS_NEUTRAL_LABELS: Dict[str, str] = {
    "same_disease_still_detected": "the same disease was detected again",
    "no_disease_detected_at_followup": "no disease was detected this time",
    "different_disease_detected_at_followup": "a different disease was detected",
    "not_applicable_healthy_baseline": "not applicable (healthy baseline)",
    "unknown": "comparison not possible",
}

# Farmer-facing plain language descriptions for triggered escalation triggers
TRIGGER_FARMER_DESCRIPTIONS: Dict[str, str] = {
    "SAME_IMAGE": "this photo is identical to your earlier photo",
    "SINGLE_CLASS_CROP": "this crop can't be diagnosed from a photo",
    "T1": "the model is not confident",
    "T2": "the model is torn between two similar diseases",
    "T3": "the photo resolution is low",
    "T4": "the photo is blurry",
    "T5": "the photo looks very different from typical leaves",
    "T6": "management guidance is limited for this crop or disease",
    "T7": "severe disease risk",
    "T8": "a different disease was detected at follow-up",
    "T9": "disease has persisted across multiple visits",
    "T10": "follow-up photo quality is degraded",
    "T11": "crop does not match your earlier photo",
    "VISION_UNCERTAIN": "the visual diagnosis is uncertain",
}


def format_monitoring_status(status: Optional[str]) -> str:
    """Format monitoring disease status into strictly neutral, farmer-facing wording."""
    if not status:
        return "comparison not possible"
    return MONITORING_STATUS_NEUTRAL_LABELS.get(status, "comparison not possible")


def clean_farmer_text(text: Optional[str]) -> str:
    """Strip internal tags like (T2), (VER), (CAR1) and normalize whitespace."""
    if not text:
        return ""
    cleaned = RE_INTERNAL_TAG.sub("", text)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_display_fields(
    final_resp: FinalResponse,
    healthy_guard_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Extract and assemble display-ready fields for frontend consumers.
    """
    v_out = final_resp.vision
    rk_out = final_resp.risk
    rec_out = final_resp.recommendation
    esc = final_resp.escalation
    mon = final_resp.monitoring

    # 1. Top 3 Candidates
    top3: List[Dict[str, Any]] = []
    if v_out and v_out.top_3:
        for c in v_out.top_3:
            name = c.label if hasattr(c, "label") else str(c)
            prob = float(c.prob) if hasattr(c, "prob") else 0.0
            top3.append({"name": name, "probability": round(prob, 4)})

    # 2. Risk label and Urgency
    risk_label = rk_out.risk_level if rk_out else "low"
    urgency = rec_out.urgency if rec_out else "monitor"

    # 3. Steps (cleaned of internal tags)
    steps: List[str] = []
    if rec_out and rec_out.steps:
        for s in rec_out.steps:
            cleaned_s = clean_farmer_text(s)
            if cleaned_s and cleaned_s not in steps:
                steps.append(cleaned_s)
    elif rec_out and rec_out.immediate_action:
        steps.append(clean_farmer_text(rec_out.immediate_action))

    # 4. Farmer Summary
    farmer_summary = ""
    if rec_out and rec_out.summary:
        farmer_summary = clean_farmer_text(rec_out.summary)
    elif rec_out and rec_out.what_to_do_first:
        farmer_summary = clean_farmer_text(rec_out.what_to_do_first)
    elif v_out:
        farmer_summary = f"Diagnosis: {v_out.top_label} (Confidence: {v_out.confidence:.1%})."

    # Single-class crop clarification
    if v_out and v_out.single_class_crop:
        crop_name = final_resp.crop or "This crop"
        farmer_summary = (
            f"Advisory note: {crop_name} has only a single registered class in our diagnostic model; "
            "differential disease classification cannot be computed. Routine monitoring is advised."
        )

    # 5. Warnings
    warnings: List[str] = []
    if v_out and v_out.uncertainty_flags:
        if v_out.uncertainty_flags.blurry:
            warnings.append("Image is blurry; capturing a sharper, well-focused photo is recommended.")
        if v_out.uncertainty_flags.low_resolution:
            warnings.append("Image resolution is low; capturing a closer photo of the leaf is recommended.")
    if v_out and v_out.ood_flag:
        warnings.append("Photo appearance differs significantly from typical training leaves (out-of-distribution).")

    # Healthy prediction runner-up guard (Task 3a)
    if v_out and "healthy" in v_out.top_label.lower() and len(top3) > 1:
        runner_up = top3[1]
        if "healthy" not in runner_up["name"].lower() and runner_up["probability"] >= healthy_guard_threshold:
            warnings.append(
                f"possible missed disease: {runner_up['name']} ({runner_up['probability']:.2f})"
            )

    # Monitoring warnings
    if mon:
        if mon.same_image_flag:
            warnings.append("Identical photo resubmission: this image matches the baseline visit exactly.")
        if mon.crop_mismatch_flag:
            warnings.append("Crop mismatch: follow-up photo appears to be a different crop from baseline.")
        if mon.too_soon_flag:
            warnings.append("Follow-up submitted on the same day: symptoms rarely change within 24 hours.")

    # 6. Escalation Banner
    escalation_banner: Optional[str] = None
    if esc and esc.escalate_to_human:
        prio_text = f" (priority: {esc.priority})" if esc.priority != "medium" else ""
        farmer_reasons: List[str] = []
        for trig in getattr(esc, "triggered_triggers", []):
            desc = TRIGGER_FARMER_DESCRIPTIONS.get(trig)
            if desc and desc not in farmer_reasons:
                farmer_reasons.append(desc)

        if farmer_reasons:
            reasons_clause = "; ".join(farmer_reasons)
            escalation_banner = (
                f"Agronomist review recommended{prio_text}: {reasons_clause}. "
                "Please verify this advisory with a local agricultural expert."
            )
        else:
            escalation_banner = (
                f"Agronomist review recommended{prio_text}: Automated analysis flagged uncertainty or "
                "elevated risk. Please verify this advisory with a local agricultural expert."
            )

    return {
        "farmer_summary": farmer_summary,
        "top3": top3,
        "risk_label": risk_label,
        "urgency": urgency,
        "steps": steps,
        "warnings": warnings,
        "disclaimers": list(STANDARD_DISCLAIMERS),
        "escalation_banner": escalation_banner,
    }


# =============================================================================
# API Safe Image Stub & Payload Sanitization
# =============================================================================

RE_BASE64_HEURISTIC = re.compile(r"^[A-Za-z0-9+/=\r\n]+$")


def build_safe_image_stub(image_data: Any) -> Dict[str, Any]:
    """
    Produce a safe metadata stub replacing raw binary image bytes.
    Ensures no raw bytes or large base64 strings ever leak into API responses.
    """
    if not image_data:
        return {"image_included": False, "image_sha256": None, "size_bytes": 0}
    if isinstance(image_data, bytes):
        sha = hashlib.sha256(image_data).hexdigest()
        return {"image_included": False, "image_sha256": sha, "size_bytes": len(image_data)}
    if isinstance(image_data, str):
        # Could be base64 string or placeholder
        sha = hashlib.sha256(image_data.encode("utf-8")).hexdigest()
        return {"image_included": False, "image_sha256": sha, "size_bytes": len(image_data)}
    if isinstance(image_data, dict):
        return {
            "image_included": bool(image_data.get("image_included", False)),
            "image_sha256": image_data.get("image_sha256"),
            "size_bytes": int(image_data.get("size_bytes", 0)),
        }
    return {"image_included": False, "image_sha256": None, "size_bytes": 0}


def is_large_base64_blob(val: str, min_len: int = 500) -> bool:
    """Detect if a string is a long base64 blob that should not be in the API response."""
    if len(val) >= min_len and RE_BASE64_HEURISTIC.match(val[:300]):
        return True
    return False


def sanitize_for_api(obj: Any) -> Any:
    """
    Recursively sanitize any Python object / dictionary so that:
      - No raw `bytes` exist anywhere in the object (converted to safe stub).
      - No large base64 data strings exist (replaced with safe stub).
      - All datetimes, UUIDs, and numbers are strictly JSON-serializable.
    """
    if obj is None:
        return None
    if isinstance(obj, bytes):
        return build_safe_image_stub(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, str):
        if is_large_base64_blob(obj):
            return build_safe_image_stub(obj)
        return obj
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in ("image_bytes", "followup_image_bytes"):
                cleaned[k] = build_safe_image_stub(v)
            else:
                cleaned[k] = sanitize_for_api(v)
        return cleaned
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_api(x) for x in obj]
    if hasattr(obj, "model_dump"):
        return sanitize_for_api(obj.model_dump(mode="python"))
    return obj

