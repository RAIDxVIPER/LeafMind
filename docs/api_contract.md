# LeafMind API Contract (Stage C9)

This document defines the permanent HTTP REST API contract for the LeafMind Backend.
It serves as the specification for frontend clients, including the future Vercel production website (Stage C9b) and the temporary Streamlit test UI (Stage C9a).

---

## 1. Overview & General Principles

- **Base URL**: Configurable, default `http://127.0.0.1:8000` (or `http://localhost:8000`).
- **Data Exchange**: JSON (`application/json`) for request bodies (where applicable) and responses. `POST /analyze` uses `multipart/form-data`.
- **CORS Support**: Handled via FastAPI CORS middleware. The allowed origins are loaded from the `ALLOWED_ORIGINS` environment variable (comma-separated). Default origins:
  - `http://localhost:8501` (Streamlit test UI)
  - `http://localhost:3000` (Local Next.js / React dev server)
  - `http://localhost:5173` (Local Vite dev server)
  *Production note for C9b*: The production Vercel frontend domain is added via `ALLOWED_ORIGINS` in production environment configuration without altering code. Wildcard (`*`) is never combined with credentials.
- **Security & Privacy**:
  - No API keys, internal file paths, environment variables, or stack traces are ever exposed in any response or error detail. All errors pass through `sanitize_error()`.
  - Client-uploaded photos have EXIF GPS metadata stripped before any internal storage or processing.
  - Image validation is performed via binary PIL decoding, not file extension. Corrupt or unparseable files are rejected with HTTP 422.
- **Severity Policy (D2 / R3)**:
  - Severity percentage (`severity_pct`) is **always `null`**. No percentage or numerical lesion mask is computed. An accompanying `severity_note` provides the required explanation.
- **Farmer-Facing Text Formatting**:
  - Display fields (`farmer_summary`, `top3`, `risk_label`, `urgency`, `steps`, `warnings`, `disclaimers`, `escalation_banner`) are pre-formatted for direct farmer consumption. Internal citation codes such as `(T2)`, `(VER)`, `(CAR1)`, `(IH26)`, `(V1)` are stripped from display text.
  - Full citations with titles and URLs remain intact in `rag.sources` and `raw_rag.citations` for audit and transparency.

---

## 2. Endpoints

### 2.1 `GET /health`

Liveness and readiness health probe.

- **Request**: None
- **Response**: `200 OK`
  ```json
  {
    "status": "ok",
    "version": "0.1.0"
  }
  ```

---

### 2.2 `GET /meta`

Returns diagnostic metadata, supported crops (single source of truth matching classifier code), farmer capture guidance, and standard regulatory disclaimers.

- **Request**: None
- **Response**: `200 OK`
  ```json
  {
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
      "Avoid extreme close-ups that obscure leaf shape, and avoid distant shots of the whole plant."
    ],
    "disclaimers": [
      "Advisory only: this report provides automated decision support. Consult a qualified local agronomist or extension officer before taking critical crop decisions.",
      "Severity percentage is not measured: no lesion-mask or affected area percentage is computed [D2/R3].",
      "Provisional ratings: disease threat tiers and weather disease-risk rules are engineering approximations pending formal plant pathologist sign-off [I14/I19].",
      "Plant identity unverified: the system cannot verify whether follow-up photos depict the same physical leaf or plant."
    ]
  }
  ```

---

### 2.3 `POST /analyze`

Analyzes an uploaded leaf photo using the full multi-agent diagnostic, weather-risk, retrieval-augmented advisory, and monitoring comparison pipeline.

- **Content-Type**: `multipart/form-data`
- **Request Parameters**:
  | Field | Type | Required | Description |
  | :--- | :--- | :--- | :--- |
  | `image` | `File` (binary) | **Yes** | Leaf image file. Accepted formats: JPEG, PNG, WebP. Maximum size: 10 MB. Minimum dimension: 64 px. |
  | `crop` | `string` | **Yes** | Crop name (case-insensitive, e.g. `Tomato`, `Corn`, `Apple`). Must match a supported crop. |
  | `location` | `string` | Optional | Free-form place name or region (e.g. `"Nashik, Maharashtra"`, `"Des Moines, Iowa"`). |
  | `lat` | `float` | Optional | GPS latitude (e.g. `19.9975`). If provided with `lon`, overrides place name geocoding. |
  | `lon` | `float` | Optional | GPS longitude (e.g. `73.7898`). |
  | `previous_case_id`| `string` | Optional | UUID of a prior case for monitoring delta evaluation. |
  | `include_doses` | `boolean` | Optional | Whether to retrieve chemical dosage rates (default: `false`). |

- **Response**: `200 OK` (Application/JSON: `AnalyzeResponse`)
  Returns the complete `FinalResponse` object supplemented with display-ready fields.

#### Example Response Body:
```json
{
  "case_id": "e4b6d920-8012-4c28-b808-8df042ce405c",
  "farmer_id": "farmer_e4b6d920",
  "submitted_at": "2026-10-28T09:15:30.123456Z",
  "crop": "Tomato",
  "is_followup": false,
  "previous_case_id": null,
  "farmer_summary": "Diagnosis: Tomato Early Blight (Alternaria solani). Apply copper-based fungicide or chlorothalonil according to label instructions. Improve spacing to allow air circulation.",
  "top3": [
    {
      "name": "Tomato___Early_blight",
      "probability": 0.8842
    },
    {
      "name": "Tomato___Target_Spot",
      "probability": 0.0815
    },
    {
      "name": "Tomato___healthy",
      "probability": 0.0343
    }
  ],
  "risk_label": "medium",
  "urgency": "within_48h",
  "steps": [
    "Prune and discard infected lower leaves showing concentric rings.",
    "Apply protective bio-fungicide or copper spray in the early morning.",
    "Avoid overhead irrigation to minimize leaf wetness duration."
  ],
  "warnings": [],
  "disclaimers": [
    "Advisory only: this report provides automated decision support. Consult a qualified local agronomist or extension officer before taking critical crop decisions.",
    "Severity percentage is not measured: no lesion-mask or affected area percentage is computed [D2/R3].",
    "Provisional ratings: disease threat tiers and weather disease-risk rules are engineering approximations pending formal plant pathologist sign-off [I14/I19].",
    "Plant identity unverified: the system cannot verify whether follow-up photos depict the same physical leaf or plant."
  ],
  "escalation_banner": null,
  "vision": {
    "top_label": "Tomato___Early_blight",
    "confidence": 0.8842,
    "top_3": [
      {"label": "Tomato___Early_blight", "prob": 0.8842},
      {"label": "Tomato___Target_Spot", "prob": 0.0815},
      {"label": "Tomato___healthy", "prob": 0.0343}
    ],
    "can_diagnose": true,
    "single_class_crop": false,
    "ood_flag": false,
    "uncertainty_flags": {
      "blurry": false,
      "low_resolution": false,
      "low_contrast": false,
      "severe_blur": false,
      "low_light": false
    },
    "severity_pct": null,
    "severity_note": "Severity percentage is not computed. No per-leaf severity labels or lesion masks exist in the training data."
  },
  "weather": {
    "weather_available": true,
    "current_temp_c": 24.5,
    "current_humidity_pct": 82.0,
    "triggered_rules": [
      "Moderate temperature (20-30°C) with elevated humidity (>80%) favors Alternaria spore germination"
    ],
    "weather_score": 0.20,
    "data_source": "open-meteo",
    "cached": false,
    "location_name": "Nashik, Maharashtra"
  },
  "rag": {
    "found": true,
    "kb_gap": false,
    "disease_name": "Tomato Early Blight",
    "sources": [
      {
        "source_id": "T2",
        "title": "TNAU Agritech Portal: Tomato Diseases",
        "url": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop_diseases_veg_tomato_1.html",
        "doc_type": "portal"
      }
    ],
    "citations": ["TNAU Agritech Portal: Tomato Diseases"],
    "chemical_options_count": 2,
    "what_to_do_first": "Prune and destroy lower diseased foliage immediately.",
    "doses_included": false
  },
  "risk": {
    "risk_score": 0.39,
    "risk_level": "medium",
    "uncertainty_signals": [],
    "escalate_to_human": false,
    "threat_tier": "MEDIUM",
    "breakdown": {
      "threat_score": 0.50,
      "weather_score": 0.20,
      "history_score": 0.00,
      "weights": {"threat": 0.70, "weather": 0.20, "history": 0.10}
    }
  },
  "recommendation": {
    "summary": "Tomato Early Blight confirmed with medium risk. Prune lower foliage and apply protective copper spray.",
    "what_to_do_first": "Prune and destroy lower diseased foliage immediately.",
    "steps": [
      "Prune and discard infected lower leaves showing concentric rings.",
      "Apply protective bio-fungicide or copper spray in the early morning.",
      "Avoid overhead irrigation to minimize leaf wetness duration."
    ],
    "urgency": "within_48h",
    "chemical_options": [],
    "non_chemical_options": ["Sanitation", "Crop rotation", "Drip irrigation"],
    "sources": ["TNAU Agritech Portal: Tomato Diseases"]
  },
  "escalation": {
    "escalate_to_human": false,
    "priority": "low",
    "triggered_triggers": [],
    "reasons": [],
    "payload": null
  },
  "monitoring": null,
  "partial_result": false,
  "flags": [],
  "timing_ms": {
    "request_normalizer": 0.42,
    "vision_runner": 112.5,
    "weather_runner": 45.2,
    "rag_runner": 2.1,
    "fan_in_barrier": 0.15,
    "risk_reason_runner": 3.8,
    "recommendation_runner": 12.4,
    "escalation_evaluator": 1.2,
    "response_builder": 2.5
  },
  "errors": {},
  "fatal_error": null,
  "severity_pct": null,
  "severity_note": "Severity percentage is not computed. No per-leaf severity labels or lesion masks exist in the training data."
}
```

---

### 2.4 `GET /case/{case_id}`

Retrieves the persisted diagnostic record by its unique `case_id`.

- **Parameters**: `case_id` (UUID path parameter)
- **Response**: `200 OK` (JSON object of stored CaseState or FinalResponse). Image binary payload is omitted from JSON responses.
- **Error**: `404 Not Found` if the case ID does not exist in local storage.

---

## 3. Error Codes & Format

All error responses return a JSON object with a `detail` field. No tracebacks, system file paths, or credentials are ever returned.

| Status Code | Reason | Example Scenario / Response Detail |
| :--- | :--- | :--- |
| `400 Bad Request` | Malformed request parameters | Missing required query/form parameters. |
| `404 Not Found` | Resource not found | Case ID not found: `{"detail": "Case '00000000-...' not found."}` |
| `422 Unprocessable Entity` | Validation failure | Unsupported crop: `{"detail": "Unsupported crop 'Mango'. Supported crops: Apple, Blueberry, Cherry, Corn, Grape, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato"}`<br><br>Corrupt/non-image: `{"detail": "Uploaded file is not a valid or readable image. Capture tip: Take a clear photo of the leaf in daylight and upload as JPG or PNG."}`<br><br>Oversized/undersized: `{"detail": "Image dimensions (32x32) are below minimum 64x64 pixels. Capture tip: Move closer to the leaf."}` |
| `500 Internal Server Error` | Unexpected backend error | Sanitized error: `{"detail": "An error occurred during case analysis: [sanitized description]"}` |
| `504 Gateway Timeout` | Pipeline timeout exceeded | `{"detail": "Analysis timed out after 60 seconds. Please retry your request."}` |

---

## 4. Special Behavioral Guarantees

1. **Single-Class Crops**:
   - For crops with only a single registered class (e.g. `Squash` having only `Squash___Powdery_mildew`), the API returns HTTP `200 OK` with `can_diagnose: false` and an advisory note explaining that differential disease classification cannot be computed.
2. **Healthy-Prediction Runner-Up Guard (I40)**:
   - When the top predicted label is `*___healthy` but a disease candidate is runner-up with probability $\ge 0.25$ (configurable), the API automatically attaches a warning:
     `possible missed disease: <runner-up> (<prob>)`.
3. **Healthy Class Zero Risk (C5 Spec)**:
   - Healthy classes always receive weather contribution $0.0$, ensuring $R = 0.0$ and low risk.
4. **Follow-Up Resubmissions**:
   - If `previous_case_id` matches an existing case, the monitoring agent compares symptom deltas, flags identical photos or crop mismatches, and appends monitoring status to the response.
