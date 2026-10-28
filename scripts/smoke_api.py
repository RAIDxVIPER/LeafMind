"""
scripts/smoke_api.py
====================
Smoke test script for LeafMind FastAPI Backend (Stage C9a).

Actions:
  1. Starts uvicorn backend on 127.0.0.1 in a subprocess.
  2. Polls GET /health until ready.
  3. Sends 4 requests:
     (1) Lab diseased image (Tomato Early Blight)
     (2) Lab healthy image (Tomato Healthy)
     (3) Dataset_2/Test_OOD field image of a multi-class crop (Tomato Early Blight field photo)
     (4) Invalid file (plain text file renamed .jpg)
  4. Prints status codes and compact view of each response JSON (zero secret leaks).
  5. Shuts down the subprocess cleanly.
  6. Prints exact Windows PowerShell commands to run backend and UI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
API_HOST = "127.0.0.1"
API_PORT = 8005
BASE_URL = f"http://{API_HOST}:{API_PORT}"


def compact_json_summary(data: Dict[str, Any]) -> str:
    """Format a compact, key-safe summary of an AnalyzeResponse or Error JSON."""
    if "detail" in data:
        return f"Detail: {data['detail']}"

    top3 = data.get("top3", [])
    top3_str = ", ".join([f"{c['name']} ({c['probability']:.1%})" for c in top3[:3]])
    risk = data.get("risk", {})
    risk_score = risk.get("risk_score", "N/A")
    risk_label = data.get("risk_label", "N/A")
    urgency = data.get("urgency", "N/A")
    summary = (data.get("farmer_summary") or "")[:80] + "..."
    esc = data.get("escalation_banner")
    warnings = data.get("warnings", [])

    lines = [
        f"Case ID: {data.get('case_id')}",
        f"Top Candidates: {top3_str}",
        f"Risk Score / Level: {risk_score} / {risk_label}",
        f"Action Urgency: {urgency}",
        f"Farmer Summary: {summary}",
        f"Escalation Banner: {esc or 'None (normal advisory)'}",
        f"Warnings: {warnings if warnings else 'None'}",
        f"Severity: {data.get('severity_pct')} (Invariant: strictly null)",
    ]
    return " | ".join(lines)


def run_smoke():
    print(f"=== Starting LeafMind API Smoke Test on {BASE_URL} ===")

    # 1. Start uvicorn server in subprocess
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.app:app",
        "--host",
        API_HOST,
        "--port",
        str(API_PORT),
    ]

    print(f"Launching backend server: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    server_log_path = REPO_ROOT / "scratch" / "uvicorn_smoke.log"
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    server_log_file = open(server_log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=server_log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        # Wait for server /health to become available (cold start timeout 45s)
        ready = False
        t0 = time.time()
        print("Waiting for API service to initialize...")
        while time.time() - t0 < 45.0:
            if server_proc.poll() is not None:
                server_log_file.close()
                out = server_log_path.read_text(encoding="utf-8", errors="replace")
                print("Server failed to start:\n", out)
                sys.exit(1)
            try:
                r = httpx.get(f"{BASE_URL}/health", timeout=2.0)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(1.0)

        if not ready:
            print("Timed out waiting for API server.")
            server_proc.terminate()
            sys.exit(1)

        print(f"Backend ready in {time.time() - t0:.1f}s.\n")

        # Define test images
        diseased_img = next(
            (REPO_ROOT / "Dataset_2" / "Test_ID" / "Tomato___Early_blight").glob("*.JPG"),
            None
        )
        healthy_img = next(
            (REPO_ROOT / "Dataset_2" / "Test_ID" / "Tomato___healthy").glob("*.JPG"),
            None
        )
        ood_img = next(
            (REPO_ROOT / "Dataset_2" / "Test_OOD" / "Tomato___Early_blight").glob("*.jpg"),
            None
        )
        squash_img = next(
            (REPO_ROOT / "Dataset_2" / "Train" / "Squash___Powdery_mildew").glob("*.JPG"),
            None
        )

        dummy_invalid_path = REPO_ROOT / "scratch" / "dummy_invalid.jpg"
        dummy_invalid_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_invalid_path.write_text("This is an invalid plain text payload renamed as jpg.", encoding="utf-8")

        test_cases = [
            ("1. Lab Diseased Image (Tomato Early Blight)", diseased_img, "Tomato", "Nashik, Maharashtra", False),
            ("2. Lab Healthy Image (Tomato Healthy)", healthy_img, "Tomato", None, False),
            ("3. Field OOD Image (Tomato Early Blight field photo)", ood_img, "Tomato", "Pune", False),
            ("4. Invalid Non-Image File (text renamed .jpg)", dummy_invalid_path, "Tomato", None, False),
            ("5. Escalated Single-Class Crop (Squash Powdery Mildew)", squash_img, "Squash", None, False),
            ("6. Follow-up Visit (comparing against baseline visit 1)", diseased_img, "Tomato", None, True),
        ]

        saved_case_id: Optional[str] = None

        with httpx.Client(timeout=60.0) as client:
            for idx, (title, img_path, crop, loc, is_fu) in enumerate(test_cases, 1):
                print(f"--- Running Request {idx}: {title} ---")
                if img_path is None or not img_path.exists():
                    print(f"Skipping: test file {img_path} not found.")
                    continue

                with open(img_path, "rb") as f:
                    file_payload = {"image": (img_path.name, f.read(), "image/jpeg")}

                data_payload: Dict[str, Any] = {"crop": crop}
                if loc:
                    data_payload["location"] = loc
                if is_fu and saved_case_id:
                    data_payload["previous_case_id"] = saved_case_id

                resp = client.post(f"{BASE_URL}/analyze", files=file_payload, data=data_payload)
                print(f"HTTP Status: {resp.status_code}")
                try:
                    resp_json = resp.json()
                    if idx == 1 and resp.status_code == 200:
                        saved_case_id = resp_json.get("case_id")
                    summary_str = compact_json_summary(resp_json)
                    print(f"Summary: {summary_str}\n")
                except Exception as e:
                    print(f"Raw Output: {resp.text[:200]}... (parse error: {e})\n")

    finally:
        print("Shutting down uvicorn backend server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        try:
            server_log_file.close()
        except Exception:
            pass
        print("Backend server stopped.")

    print("\n" + "=" * 60)
    print("Exact Windows PowerShell commands to run the application:")
    print("  Terminal 1 (Backend API):")
    print("    python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000")
    print("  Terminal 2 (Streamlit Test UI):")
    print("    streamlit run streamlit_app/app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_smoke()
