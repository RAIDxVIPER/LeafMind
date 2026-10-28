"""
streamlit_app/app.py
====================
Self-contained Streamlit test UI for LeafMind.
Calls the FastAPI backend over HTTP only (httpx/requests).
Does NOT import from src/ or the pipeline.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx
import streamlit as st

# =============================================================================
# Streamlit Page Configuration
# =============================================================================
st.set_page_config(
    page_title="LeafMind Test Console",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Base URL from env or default
DEFAULT_API_BASE = os.getenv("LEAFMIND_API_BASE", "http://127.0.0.1:8000")


# =============================================================================
# HTTP API Client Helpers
# =============================================================================
def fetch_health(api_base: str) -> Optional[Dict[str, Any]]:
    """Query GET /health endpoint."""
    try:
        resp = httpx.get(f"{api_base.rstrip('/')}/health", timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_meta(api_base: str) -> Dict[str, Any]:
    """Query GET /meta endpoint with fallback defaults."""
    fallback = {
        "supported_crops": [
            "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Peach",
            "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"
        ],
        "single_class_crops": [
            "Blueberry", "Cherry", "Peach", "Raspberry", "Soybean", "Squash", "Strawberry"
        ],
        "multi_class_crops": [
            "Apple", "Corn", "Grape", "Pepper", "Potato", "Tomato"
        ],
        "capture_tips": [
            "Take photos in bright, indirect daylight (avoid harsh direct sunlight or deep shadow).",
            "Hold your camera steady to prevent blur; ensure the leaf surface and lesion margins are in sharp focus.",
            "Frame the leaf so it fills at least 70% of the picture, showing both healthy and symptomatic tissue.",
            "Avoid extreme close-ups that obscure leaf shape, and avoid distant shots of the whole plant.",
        ],
        "disclaimers": [
            "Advisory only: this report provides automated decision support. Consult a qualified local agronomist or extension officer before taking critical crop decisions.",
            "Severity percentage is not measured: no lesion-mask or affected area percentage is computed [D2/R3].",
            "Provisional ratings: disease threat tiers and weather disease-risk rules are engineering approximations pending formal plant pathologist sign-off [I14/I19].",
            "Plant identity unverified: the system cannot verify whether follow-up photos depict the same physical leaf or plant.",
        ],
    }
    try:
        resp = httpx.get(f"{api_base.rstrip('/')}/meta", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return fallback


def post_analyze(
    api_base: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    crop: str,
    location: Optional[str] = None,
    previous_case_id: Optional[str] = None,
    include_doses: bool = False,
) -> tuple[Optional[Dict[str, Any]], Optional[int], Optional[str]]:
    """
    Submit photo and metadata to POST /analyze.
    Returns (data, status_code, error_message).
    """
    url = f"{api_base.rstrip('/')}/analyze"
    files = {"image": (filename, image_bytes, content_type or "image/jpeg")}
    data = {"crop": crop, "include_doses": str(include_doses).lower()}
    if location:
        data["location"] = location.strip()
    if previous_case_id:
        data["previous_case_id"] = previous_case_id.strip()

    try:
        with httpx.Client(timeout=75.0) as client:
            resp = client.post(url, files=files, data=data)
            if resp.status_code == 200:
                return resp.json(), 200, None

            # Attempt to parse error detail
            detail = resp.text
            try:
                err_json = resp.json()
                if "detail" in err_json:
                    detail = err_json["detail"]
            except Exception:
                pass
            return None, resp.status_code, str(detail)
    except httpx.TimeoutException:
        return None, 504, "Analysis request timed out after 75s. Please verify API responsiveness."
    except httpx.ConnectError:
        return None, 503, f"Unable to connect to LeafMind API at {api_base}. Ensure backend is running."
    except Exception as exc:
        return None, 500, f"Unexpected network error: {exc}"


# =============================================================================
# Main Application Flow
# =============================================================================
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("🍃 LeafMind Settings")
    api_base = st.sidebar.text_input(
        "API Base URL",
        value=DEFAULT_API_BASE,
        help="Root URL for the LeafMind FastAPI backend service.",
    )

    # Health Check Indicator
    health = fetch_health(api_base)
    if health and health.get("status") == "ok":
        v = health.get("version", "0.1.0")
        st.sidebar.success(f"🟢 API Online (v{v})")
    else:
        st.sidebar.error("🔴 API Offline / Unreachable")
        st.sidebar.caption(f"Make sure backend is running: `uvicorn src.api.app:app --port 8000`")

    st.sidebar.markdown("---")
    dev_mode = st.sidebar.checkbox(
        "🛠️ Developer view",
        value=False,
        help="Show raw JSON, node execution timings, threat breakdowns, and internal flags.",
    )

    # ── Main Header ──────────────────────────────────────────────────────────
    st.title("🍃 LeafMind Diagnostic & Advisory Test Console")
    st.caption("AI-Powered Multi-Agent Crop Health Decision Support System")

    # Load metadata (from /meta)
    meta = fetch_meta(api_base)
    capture_tips = meta.get("capture_tips", [])
    supported_crops = meta.get("supported_crops", [])
    disclaimers = meta.get("disclaimers", [])

    # Capture Tips Expander
    with st.expander("📷 Leaf Photo Capture Guide (Tips for accurate diagnosis)", expanded=False):
        for tip in capture_tips:
            st.markdown(f"- {tip}")

    st.markdown("### 1. Upload & Field Context")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Crop selector (Single source of truth from /meta)
        crop_options = [""] + supported_crops
        selected_crop = st.selectbox(
            "Select Crop *",
            options=crop_options,
            index=0,
            help="Select the crop plant species shown in the photo.",
        )

        location_input = st.text_input(
            "Farm / Field Location (optional)",
            value="",
            placeholder="e.g. Pune, Maharashtra or Nashik",
            help="City, district, or region for localized weather-driven disease risk scoring.",
        )

        # Follow-up checkbox
        is_followup = st.checkbox("This is a follow-up visit on an earlier case")
        prev_case_id = ""
        if is_followup:
            last_case = st.session_state.get("last_case_id", "")
            prev_case_id = st.text_input(
                "Previous Case ID",
                value=last_case,
                help="UUID of the prior baseline case to compare symptoms against.",
            )

    with col2:
        uploaded_file = st.file_uploader(
            "Upload Leaf Image (JPEG, PNG, WebP) *",
            type=["jpg", "jpeg", "png", "webp"],
            help="High quality, focused leaf image. Max size: 10 MB.",
        )
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Leaf Preview", use_container_width=True)

    # Analyze Button
    st.markdown("---")
    analyze_clicked = st.button("🔍 Analyse Leaf Photo", type="primary", use_container_width=True)

    if analyze_clicked:
        if not selected_crop:
            st.warning("⚠️ Please select a crop from the dropdown.")
            return

        if uploaded_file is None:
            st.warning("⚠️ Please upload a leaf image file to diagnose.")
            return

        with st.spinner("Analyzing leaf with multi-agent pipeline..."):
            image_bytes = uploaded_file.getvalue()
            data, status_code, err_msg = post_analyze(
                api_base=api_base,
                image_bytes=image_bytes,
                filename=uploaded_file.name,
                content_type=uploaded_file.type,
                crop=selected_crop,
                location=location_input if location_input else None,
                previous_case_id=prev_case_id if is_followup and prev_case_id else None,
                include_doses=False,
            )

        if status_code == 200 and data is not None:
            st.session_state["current_analysis"] = data
            st.session_state["last_case_id"] = str(data.get("case_id", ""))
            st.success("✅ Analysis completed successfully!")
        elif status_code == 422:
            st.warning(f"⚠️ Validation Notice: {err_msg}")
            return
        elif status_code == 504:
            st.error(f"⏱️ Request Timeout (504): {err_msg}")
            return
        else:
            st.error(f"❌ Analysis Request Failed ({status_code}): {err_msg}")
            return

    # ── Results Presentation ─────────────────────────────────────────────────
    data = st.session_state.get("current_analysis")
    if not data:
        st.info("👋 Upload a leaf photo and click **Analyse Leaf Photo** to inspect diagnosis, risk, and advisory.")
        return

    st.markdown("## 📋 Diagnostic Assessment")

    # 1. Escalation Banner
    esc_banner = data.get("escalation_banner")
    esc_obj = data.get("escalation", {})
    if esc_banner or esc_obj.get("escalate_to_human"):
        banner_text = esc_banner or "Agronomist review recommended: Automated analysis flagged high uncertainty or severe risk. Please verify this advisory with a local agricultural expert."
        st.warning(f"⚠️ **{banner_text}**")

    # 2. Top-3 Candidates Table
    st.markdown("### Differential Diagnosis (Top-3 Candidates)")
    st.caption("LeafMind presents top-3 diagnostic candidates with calibrated probabilities; never a single overconfident answer.")

    top3 = data.get("top3", [])
    if top3:
        table_rows = []
        for rank, item in enumerate(top3, 1):
            name = item.get("name", "Unknown")
            prob = float(item.get("probability", 0.0))
            is_healthy = "healthy" in name.lower()
            kind = "Healthy Tissue" if is_healthy else "Disease Candidate"
            table_rows.append({
                "Rank": f"#{rank}",
                "Candidate Label": name.replace("___", " - ").replace("_", " "),
                "Probability": f"{prob:.1%}",
                "Category": kind,
            })
        st.table(table_rows)
    else:
        st.write("No candidate probabilities returned.")

    # 3. Risk Label & Urgency
    c_risk, c_urg, c_crop = st.columns(3)
    with c_risk:
        risk_label = str(data.get("risk_label", "low")).upper()
        st.metric(
            label="Risk Level (Provisional / Unverified)",
            value=risk_label,
            help="Composite score: 70% Threat + 20% Weather + 10% History. Provisional engineering ratings [I14/I19].",
        )
    with c_urg:
        urgency = str(data.get("urgency", "monitor")).replace("_", " ").upper()
        st.metric(label="Action Urgency", value=urgency)
    with c_crop:
        st.metric(label="Crop Species", value=str(data.get("crop", "Unknown")))

    # 4. Recommendation & Immediate Action
    st.markdown("### 🌾 Management & Advisory")
    summary = data.get("farmer_summary", "")
    if summary:
        st.info(f"**Summary Guidance**: {summary}")

    steps = data.get("steps", [])
    if steps:
        st.markdown("**Actionable Steps (What to do first):**")
        for idx, step in enumerate(steps, 1):
            st.markdown(f"{idx}. {step}")

    # 5. Weather Context Note
    weather = data.get("weather")
    if weather and weather.get("weather_available"):
        temp = weather.get("current_temp_c")
        humidity = weather.get("current_humidity_pct")
        rules = weather.get("triggered_rules", [])
        w_score = weather.get("weather_score", 0.0)
        loc_str = weather.get("location_name") or "Local Field"

        with st.container():
            st.markdown(f"**Weather Context for {loc_str} (Provisional / Unverified)**")
            temp_str = f"{temp:.1f} °C" if temp is not None else "N/A"
            hum_str = f"{humidity:.1f} %" if humidity is not None else "N/A"
            st.caption(f"Temperature: {temp_str} | Relative Humidity: {hum_str} | Weather Disease Factor: {w_score:.2f}")
            if rules:
                for r in rules:
                    st.caption(f"• Rule triggered: {r}")

    # 6. Warnings & Guard Alerts
    warnings = data.get("warnings", [])
    if warnings:
        st.markdown("### ⚠️ Quality & Diagnostic Notices")
        for w in warnings:
            st.warning(f"• {w}")

    # 7. Follow-up / Monitoring Section
    mon = data.get("monitoring")
    if mon:
        st.markdown("### 🔄 Follow-Up Monitoring Delta")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            raw_status = mon.get("disease_status", "unknown")
            neutral_desc = {
                "same_disease_still_detected": "the same disease was detected again",
                "no_disease_detected_at_followup": "no disease was detected this time",
                "different_disease_detected_at_followup": "a different disease was detected",
                "not_applicable_healthy_baseline": "not applicable (healthy baseline)",
                "unknown": "comparison not possible",
            }.get(raw_status, "comparison not possible")
            delta = mon.get("confidence_raw_delta", mon.get("confidence_delta", 0.0))
            st.metric(label="Follow-Up Disease Status", value=raw_status)
            st.info(f"**Follow-Up Observation:** {neutral_desc.capitalize()}")
            st.metric(label="Classifier Confidence Delta (non-efficacy)", value=f"{delta:+.1%}")
        with col_m2:
            st.markdown("**Plausibility Checks:**")
            st.markdown(f"- Identical Photo: `{'YES' if mon.get('same_image_flag') else 'NO'}`")
            st.markdown(f"- Crop Mismatch: `{'YES' if mon.get('crop_mismatch_flag') else 'NO'}`")
            st.markdown(f"- Same Day Resubmission: `{'YES' if mon.get('too_soon_flag') else 'NO'}`")
            st.markdown(f"- Poor Quality Follow-up: `{'YES' if mon.get('poor_quality_flag') else 'NO'}`")
        st.caption("ℹ️ *Plant identity unverified: the system cannot verify whether follow-up photos depict the same physical leaf or plant.*")

    # 8. Case ID (Copy friendly)
    st.markdown("### 🔖 Case Tracking")
    case_id_val = str(data.get("case_id", ""))
    st.code(case_id_val, language="text")
    st.caption("Keep this Case ID to reference or perform follow-up tracking on future visits.")

    # 9. Citations
    rag_data = data.get("rag")
    if rag_data and rag_data.get("citations"):
        with st.expander("📚 Knowledge Base Citations & Reference Sources", expanded=False):
            for c in rag_data.get("citations", []):
                st.markdown(f"- {c}")

    # 10. Regulatory Disclaimers
    with st.expander("⚖️ Standard Disclaimers", expanded=False):
        for disc in data.get("disclaimers", disclaimers):
            st.markdown(f"- {disc}")

    # ── Developer View ───────────────────────────────────────────────────────
    if dev_mode:
        st.markdown("---")
        st.markdown("## 🛠️ Developer Diagnostics (Testing Only)")

        # Risk Breakdown
        risk_obj = data.get("risk", {})
        breakdown = risk_obj.get("breakdown", {})
        t_score = breakdown.get("threat_score", 0.0)
        w_score = breakdown.get("weather_score", 0.0)
        h_score = breakdown.get("history_score", 0.0)
        r_score = risk_obj.get("risk_score", 0.0)
        recomputed_r = 0.70 * t_score + 0.20 * w_score + 0.10 * h_score

        st.markdown("#### Risk Factor Breakdown ($R = 0.70 \\cdot T + 0.20 \\cdot W + 0.10 \\cdot H$)")
        c_t, c_w, c_h, c_r = st.columns(4)
        c_t.metric("Threat (T)", f"{t_score:.2f}")
        c_w.metric("Weather (W)", f"{w_score:.2f}")
        c_h.metric("History (H)", f"{h_score:.2f}")
        c_r.metric("Reported R / Recomputed", f"{r_score:.2f} / {recomputed_r:.2f}")

        # Node Execution Timings
        timings = data.get("timing_ms", {})
        if timings:
            st.markdown("#### Pipeline Node Timings (ms)")
            timing_cols = st.columns(len(timings))
            for col, (node, dur) in zip(timing_cols, timings.items()):
                col.metric(node, f"{dur:.1f} ms")

        # Flags and Fallbacks
        flags = data.get("flags", [])
        st.markdown(f"**Pipeline Flags ({len(flags)}):** {', '.join(flags) if flags else 'None'}")

        rec_obj = data.get("recommendation", {})
        llm_used = rec_obj.get("llm_used") if rec_obj else None
        fb_reason = rec_obj.get("fallback_reason") if rec_obj else None
        st.markdown(f"**LLM Used:** `{llm_used}` | **Fallback Reason:** `{fb_reason}`")

        # Full Raw JSON
        st.markdown("#### Raw API Response JSON")
        st.json(data)


if __name__ == "__main__":
    main()
