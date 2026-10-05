# LeafMind Technical Specification — Part 6: Validity Review, Limitations, and Final Build Specification
Tags: [PPT]=explicit in slides | [DATA]=verified from datasets | [ENG]=engineering proposal | [GAP]=missing or ambiguous
Date: 2026-10-05

Component types: **ML model** = trained neural net | **LLM** = large language model | **Deterministic** = rule-based / algorithmic

---

## 6a. Research Validity Review

### 6a-i. Component Classification

Every component in the system is classified below. The classification determines what claims can be made about it.

| Component | Type | Justification | What CAN be claimed | What MUST NOT be claimed without experiments |
|-----------|------|---------------|---------------------|----------------------------------------------|
| **EfficientNet-B3 classifier** | ML model | Trained CNN; deterministic at inference given weights and input | Architecture choice; training protocol; measured metrics on Test_ID and Test_OOD after training | Accuracy, F1, or any performance number before training is complete; "state-of-the-art" performance; superiority over other architectures without ablation |
| **YOLOv8s detector** (optional) | ML model | Trained CNN; deterministic at inference | Architecture choice; training protocol; measured mAP on carved test set after training | Detection accuracy before training; "real-time" performance without latency measurement; per-class AP for class 28 (2 annotations) |
| **Temperature scaling** | Deterministic (post-processing) | Single scalar T optimised on D2_Val; no learned parameters beyond T | Calibration methodology; measured ECE after calibration | "Well-calibrated" without reporting measured ECE; any ECE number before calibration is run |
| **Image quality flags** (blurry, low_resolution) | Deterministic | Threshold-based heuristics (Laplacian variance, pixel count) | Flag logic; threshold definitions | Flag accuracy or reliability without a labelled quality test set [GAP] |
| **OOD flag** | Deterministic | Threshold on max softmax probability (< 0.40 [ENG assumption]) | Flag logic; threshold definition | OOD detection accuracy without a labelled OOD test set [GAP] |
| **Weather Agent** | Deterministic | API fetch + cache + rule evaluation; no learned parameters | API choice; caching design; rule definitions with literature citations | Rule accuracy without plant pathologist verification [I14]; weather-based risk improvement without ablation |
| **Disease-risk rules W01–W07** | Deterministic | Literature-cited threshold rules | Rule existence; literature sources | Threshold validity without plant pathologist review [I14]; rule W08 as valid (no citation) |
| **RAG retrieval** (embed + search + rerank) | Deterministic | Embedding similarity + cross-encoder reranking; no learned task-specific parameters | Pipeline design; model choices | Retrieval precision/recall before knowledge base is built and evaluation queries are labelled [GAP, I16] |
| **RAG synthesis** | LLM | Generates natural-language treatment advice from retrieved chunks | Grounding methodology; hallucination controls; prompt design | Faithfulness, hallucination rate, or recommendation quality without expert evaluation [GAP] |
| **Risk/Reason Agent** | Deterministic | Weighted rule-based formula; no ML or LLM | Formula design; weight rationale; worked examples (illustrative) | Risk accuracy or calibration without expert-labelled scenario set [GAP, I19]; claim that weights are "optimal" |
| **Recommendation Agent** | LLM | Generates farmer-facing advice grounded to RAG output | Grounding methodology; fallback design; prompt constraints | Recommendation quality, actionability, or safety without agronomist evaluation [GAP] |
| **Monitoring Agent** | ML model + Deterministic | Re-runs Vision Agent (ML) + deterministic delta computation | Delta computation design; plausibility checks; escalation trigger logic | Treatment efficacy; disease progression measurement; "recovery detection" — no paired temporal data exists [GAP, I20] |
| **Orchestrator** | Deterministic | LangGraph state graph; routing, fan-out, persistence | Architecture design; state management; failure handling | Latency claims without measurement; throughput without load testing |
| **Human escalation logic** | Deterministic | Threshold-based trigger evaluation (T1–T11) | Trigger definitions; priority mapping | Escalation appropriateness without agronomist validation |
| **Case persistence** | Deterministic | JSON file / SQLite storage | Storage design; schema | Reliability or durability claims without stress testing |

---

### 6a-ii. Claims That Require Experiments Before Making

The following statements **appear in the PPT or could be inferred from the spec** but MUST NOT be made in any paper, report, or demo without the corresponding experiment being conducted and the result measured.

| Claim | Required experiment | Data needed | Status |
|-------|-------------------|-------------|--------|
| "The system achieves X% accuracy on plant disease classification" | Train EfficientNet-B3 on D2_Train; evaluate on D2_Test_ID and D2_Test_OOD | D2 splits (available) | Blocked until C2 (training) is complete |
| "Sub-4-second end-to-end latency" [PPT] | Measure wall-clock time on 100 test requests; report median and p95 | Running system + test images | Blocked until C9 (API) is complete |
| "The system generalises to field conditions" | Measure macro-F1 on D2_Test_OOD (after D1 exclusions); compare to Test_ID | D2 splits + exclusion list | Blocked until C2 |
| "Confidence scores are well-calibrated" | Measure ECE on D2_Val after temperature scaling | Trained model + D2_Val | Blocked until C2 |
| "The RAG agent provides grounded treatment advice" | Measure faithfulness and hallucination rate via expert review | Knowledge base + agronomist reviewer | Blocked by I16 + expert availability |
| "Weather risk flags improve diagnostic accuracy" | Ablation: compare risk scoring with and without weather flags on expert-labelled scenarios | Expert scenario set | Blocked by I14 + I19 |
| "The monitoring loop tracks treatment efficacy" [PPT] | Would require paired before/after images with known treatment outcomes | No such data exists [I20] | **Cannot be claimed with current data** |
| "Severity estimation of X% leaf area affected" [PPT] | Would require lesion-level segmentation masks or per-leaf severity annotations | No such data exists [I2] | **Cannot be claimed with current data** |
| "The system detects lesion boundaries" | D1 bboxes are whole-leaf, not lesion-level [DATA] | Would require lesion-annotated dataset | **Cannot be claimed with current data** |
| "The threat table accurately ranks disease severity" | Expert review of tier assignments by plant pathologist | Empty threat table [I19] | Blocked by I19 |

---

### 6a-iii. What CAN Be Claimed (with proper qualification)

These are defensible claims that can be made based on the spec and available data:

1. **Architecture novelty**: "We propose an Orchestrator-Specialist multi-agent architecture for agricultural diagnostics that integrates visual classification, meteorological context, and knowledge retrieval." [PPT] — This is a design contribution.
2. **Risk/uncertainty separation**: "The system separates disease risk (how harmful the disease is) from diagnostic uncertainty (how reliable the classification is), preventing conflation that could lead to dangerous misinformation." [ENG] — This is a design principle.
3. **Data hygiene**: "We identified and documented critical cross-dataset contamination (87.8% of OOD test images present in the detection training set) and specified exclusion protocols." [DATA] — This is a methodological contribution.
4. **Honest limitations**: "Severity estimation is not supported by the available data; the system outputs null with an explanation rather than fabricating a percentage." [DATA] — This is an honest scientific practice.

---

## 6b. Limitations

### 6b-i. Data Limitations

| ID | Limitation | Impact | Mitigation |
|----|-----------|--------|------------|
| L1 | No severity/lesion annotations in either dataset [DATA] | Cannot compute "% leaf area affected" as PPT claims; severity_pct is always null | Output null with explanation; flag as GAP for future data acquisition |
| L2 | No treatment/knowledge documents in workspace [DATA] | RAG Agent is fully specified but cannot be tested or deployed; Recommendation Agent falls back to "consult agronomist" | Phase B: collect and licence FAO, CABI, extension documents |
| L3 | No paired before/after temporal images [DATA] | Monitoring Agent cannot validate disease progression or treatment efficacy claims | Monitoring outputs observations only ("disease still detected"), never claims about efficacy |
| L4 | D2_Test_OOD effective size after D1 exclusions is ~299 images [DATA] | Per-class OOD metrics are unreliable; macro-F1 on OOD has wide confidence intervals | Report with explicit caveat; do not draw per-class conclusions on OOD |
| L5 | D1 classes 27 (11 boxes) and 28 (2 boxes) are near-unusable [DATA] | Per-class detector performance is unreliable for these classes | Flag class 28 as unreliable in all outputs; do not gate on its AP |
| L6 | D1 bboxes are whole-leaf, not lesion-level [DATA] | Detector finds "where is the leaf in the scene", not "where is the disease on the leaf" | Document in all detector-related outputs; do not claim lesion localisation |
| L7 | Classifier is trained on PlantVillage (lab images); OOD test is field images [DATA] | Significant domain gap expected between training and real-world deployment | Measure OOD gap explicitly; do not hide behind in-distribution metrics |
| L8 | No soil, crop growth stage, or field-condition metadata in any dataset [DATA] | PPT-mentioned "soil conditions" and "crop growth stage" inputs cannot be supported | Omitted from all schemas; logged as I15/GAP |

### 6b-ii. Model and Algorithm Limitations

| ID | Limitation | Impact | Mitigation |
|----|-----------|--------|------------|
| L9 | Disease threat table is entirely empty [GAP, I19] | Risk formula falls back to DISEASE_DETECTED (0.50); all diseases treated equally; max first-visit R = 0.55 ("moderate"); "severe" is unreachable on first visit | Document as known limitation; require pathologist review before production |
| L10 | Escalation thresholds (0.60 confidence, 0.15 gap) are engineering assumptions [ENG] | May over- or under-escalate depending on actual model calibration | Calibrate on D2_Val after first training run; specify as engineering starting points only |
| L11 | Risk formula weights (0.70/0.20/0.10) are engineering assumptions [ENG] | Risk scores are plausible but not calibrated against expert judgement | Fit weights via logistic regression on expert-labelled scenarios when available |
| L12 | Weather risk thresholds (W01–W07) are unverified by plant pathologist [I14] | Disease-favourable condition flags may be inaccurate | Tag all weather contributions as [UNVERIFIED]; require pathologist review |
| L13 | LLM in RAG synthesis and Recommendation Agent is non-deterministic | Same inputs may produce different outputs; testing requires multiple runs | Use temperature=0 where available; enforce structured JSON output; validate citations |
| L14 | OOD flag threshold (max_softmax < 0.40) is uncalibrated [ENG] | Unknown false-positive and false-negative rates for out-of-distribution detection | Construct labelled OOD test set; calibrate threshold |
| L15 | Image quality flags (blurry, low_resolution) are uncalibrated [ENG] | Unknown accuracy of quality detection | Construct labelled quality set; calibrate thresholds |

### 6b-iii. System Limitations

| ID | Limitation | Impact | Mitigation |
|----|-----------|--------|------------|
| L16 | No offline mode implemented [PPT future scope] | System requires internet for Weather API and LLM API calls | Deterministic fallback paths handle API unavailability; offline mode deferred |
| L17 | No multilingual support [PPT future scope] | Advice is in English only; excludes many target farmers | Future scope; not in Phase 1 |
| L18 | Monitoring Agent cannot verify same-plant identity | Follow-up may show a different plant entirely; no plant identity model exists | Crop-prefix mismatch flag is the only check available |
| L19 | No cost estimates in recommendations [GAP] | PPT implies cost data; none exists in RAG knowledge base | cost_estimate_note = null until cost data collected |

---

## 6c. Additional Data and APIs Required

### 6c-i. Data Still Needed

| # | Data | Purpose | Priority | Source | Status |
|---|------|---------|----------|--------|--------|
| AD1 | Agricultural treatment documents (30 disease classes) | RAG knowledge base | CRITICAL (blocks C3, C6) | FAO, CABI, PlantVillage, university extensions | NOT COLLECTED [I16] |
| AD2 | Disease threat tier assignments | Threat table for risk scoring | CRITICAL (blocks non-fallback risk) | Plant pathologist review | NOT POPULATED [I19] |
| AD3 | Expert-labelled risk scenarios (≥30 cases) | Risk formula validation | HIGH | Plant pathologist / agronomist | NOT CONSTRUCTED |
| AD4 | Labelled OOD test images (non-plant + corrupted) | OOD flag calibration | MEDIUM | ImageNet non-plant classes (open licence) | NOT CONSTRUCTED |
| AD5 | Labelled blurry/sharp image pairs | Quality flag calibration | MEDIUM | Synthetic from D2 images | NOT CONSTRUCTED |
| AD6 | Pesticide registration data (active ingredients, approved crops) | RAG enrichment | MEDIUM | National pesticide registries | NOT COLLECTED |
| AD7 | Regional treatment practices | RAG localisation | MEDIUM | Extension partners, NGOs | NOT COLLECTED |
| AD8 | Lesion-level segmentation masks | Future severity estimation | LOW (future) | Expert annotation campaign | NOT AVAILABLE |
| AD9 | Paired before/after treatment images | Monitoring validation | LOW (future) | Field data collection | NOT AVAILABLE |
| AD10 | Soil and crop growth-stage metadata | Environmental context | LOW (future) | IoT sensors, farmer input | NOT AVAILABLE |

### 6c-ii. APIs Required

| API | Purpose | Phase | Key required? | Status |
|-----|---------|-------|---------------|--------|
| Open-Meteo | Primary weather data | Phase 1 | No | Available; free; no key |
| OpenWeatherMap | Backup weather data | Phase 1 | Yes (free tier) | Available; key via `OPENWEATHERMAP_API_KEY` |
| Google Generative AI (Gemini) | LLM for RAG synthesis + Recommendation | Phase 1 (if chosen) | Yes | Available; key via `GEMINI_API_KEY` |
| Ollama (local) | LLM alternative | Phase 1 (if chosen) | No (local) | Available; requires local installation |

---

## 6d. Final Build Specification

> **This section is self-contained.** A coding AI can use Section 6d alone to build the LeafMind system without needing Parts 1–5. All necessary schemas, folder structure, stack, build order, and coding rules are embedded here. Where data is unavailable, stubs are marked clearly. No fake model results.

---

### 6d-0. Project Identity

**Name**: LeafMind
**Purpose**: Multi-agent crop leaf disease detection and farmer advisory system.
**Core pipeline**: Farmer submits leaf image + GPS → Orchestrator dispatches Vision, Weather, RAG agents in parallel → Risk/Reason Agent synthesises → Recommendation Agent generates advice → Farmer receives structured JSON response.
**Follow-up pipeline**: Farmer submits follow-up image + case_id → Monitoring Agent computes delta → Risk re-evaluated → Updated recommendation.

---

### 6d-i. Folder Structure

```
LeafMind/
│
├── .env.example                   ← committed; template of required environment variables
├── .gitignore                     ← committed; excludes Dataset/, Dataset_2/, .env, checkpoints/,
│                                     vector_db/, *.pth, *.pt, *.onnx, __pycache__
├── plan.md                        ← committed; project plan and progress tracker
├── README.md                      ← committed; project overview, setup, how to run
│
├── docs/                          ← committed; all specification and audit documents
│   ├── spec/
│   │   ├── part1_audit.md
│   │   ├── part2_ml_knowledge.md
│   │   ├── part3_agents.md
│   │   ├── part4_decision_logic.md
│   │   ├── part5_eval_stack_roadmap.md
│   │   └── part6_build_spec.md    ← this document
│   ├── dataset_audit.md
│   ├── ppt_extract.md
│   └── bbox_samples/              ← 20 sample bbox images reviewed in Part 1
│
├── config/                        ← committed; configuration files and static data
│   ├── exclusion_list.json        ← MD5 hashes of D1 files to exclude from OOD evaluation
│   ├── class_map_d2.json          ← D2 class index → label mapping (30 classes)
│   ├── class_map_d1.json          ← D1 class index → label mapping (29 classes)
│   └── threat_table.json          ← initially all [GAP]; plant pathologist fills this
│
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── vision_agent.py
│   │   ├── weather_agent.py
│   │   ├── rag_agent.py
│   │   ├── risk_agent.py
│   │   ├── recommendation_agent.py
│   │   └── monitoring_agent.py
│   ├── models/
│   │   ├── classifier.py          ← EfficientNet-B3 load + infer + temperature scaling
│   │   └── detector.py            ← YOLOv8s load + infer (optional)
│   ├── schemas/
│   │   └── schemas.py             ← ALL Pydantic schemas (embedded below in 6d-iii)
│   ├── rag/
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── synthesiser.py
│   ├── risk/
│   │   ├── threat_table.py        ← loads config/threat_table.json
│   │   └── scorer.py              ← Risk formula R = 0.70×threat + 0.20×weather + 0.10×history
│   ├── monitoring/
│   │   └── delta.py               ← Monitoring delta computation + plausibility checks
│   ├── persistence/
│   │   └── case_store.py          ← CaseState read/write (JSON Phase 1)
│   ├── graph/
│   │   └── workflow.py            ← LangGraph StateGraph definition
│   └── api/
│       ├── app.py                 ← FastAPI application entry point
│       └── routes.py              ← API route definitions
│
├── tests/
│   ├── unit/
│   │   ├── test_risk_scorer.py
│   │   ├── test_monitoring_delta.py
│   │   ├── test_schemas.py
│   │   ├── test_image_quality.py
│   │   └── test_exclusion_list.py
│   ├── integration/
│   │   ├── test_new_case_flow.py
│   │   └── test_followup_flow.py
│   ├── fixtures/                  ← small test images, mock API responses
│   └── stubs/                     ← mock agents for integration tests (CLEARLY MARKED)
│
├── scripts/
│   ├── inspect/                   ← data audit scripts
│   ├── train_classifier.py
│   ├── train_detector.py          ← optional
│   ├── calibrate.py               ← temperature scaling on D2_Val
│   ├── build_exclusion_list.py
│   ├── ingest_rag_docs.py
│   └── evaluate.py
│
├── data/                          ← git-ignored
│   ├── cases/                     ← CaseState JSON files
│   └── checkpoints.db             ← LangGraph SQLite checkpoint store
│
├── checkpoints/                   ← git-ignored; model checkpoints
│   ├── classifier/                ← .pth + temperature.json
│   └── detector/                  ← .pt file
│
├── vector_db/                     ← git-ignored; ChromaDB persistent storage
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

### 6d-ii. Tech Stack

| Package | Purpose | Version policy |
|---------|---------|----------------|
| Python 3.11 | Runtime | Pin minor |
| PyTorch | ML framework for classifier and detector | Latest stable |
| timm | EfficientNet-B3 pretrained weights and fine-tuning | Latest stable |
| ultralytics | YOLOv8s training and inference | Latest stable |
| Pillow | Image decoding, mode conversion, quality checks | Latest stable |
| langgraph | Orchestrator state graph, checkpointing, conditional edges | Latest stable |
| pydantic v2 | All agent message schemas; data validation | Pin major (v2) |
| FastAPI | Async HTTP API | Latest stable |
| uvicorn | ASGI server | Latest stable |
| Streamlit | Phase 1 demo UI | Latest stable |
| sentence-transformers | Embedding (all-MiniLM-L6-v2) + reranking (ms-marco-MiniLM-L-6-v2) | Latest stable |
| chromadb | Vector DB (Phase 1; embedded, no server) | Latest stable |
| google-generativeai | Gemini LLM access (when LLM_PROVIDER=gemini) | Latest stable |
| ollama (Python SDK) | Ollama LLM access (when LLM_PROVIDER=ollama) | Latest stable |
| httpx | Async HTTP client for weather API | Latest stable |
| python-dotenv | Load .env in development | Latest stable |
| pytest + pytest-asyncio | Testing | Latest stable |
| Docker + docker-compose | Containerisation | Latest stable |

**NOT included**: LangChain (not needed; direct SDK calls), TensorFlow, Redis (Phase 2), React/Node.js (Phase 2 if Streamlit insufficient).

**Required environment variables** (`.env.example`):
```bash
LLM_PROVIDER=                  # "gemini" or "ollama"
GEMINI_API_KEY=                # required when LLM_PROVIDER=gemini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OPENWEATHERMAP_API_KEY=        # backup weather API
CASE_STORE_PATH=data/cases
CHECKPOINT_DB_PATH=data/checkpoints.db
VECTOR_DB_PATH=vector_db/
LOG_LEVEL=INFO
```

---

### 6d-iii. Pydantic Schemas (Complete — Copy Into `src/schemas/schemas.py`)

```python
"""
LeafMind Pydantic Schemas — v1.0
All inter-agent message contracts. Field names and types are final.
This file is specification code, not application logic.
"""
from __future__ import annotations
from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import uuid


# ── Farmer Request ─────────────────────────────────────────────────────────────

class FarmerRequest(BaseModel):
    """Initial or follow-up submission from the farmer."""
    farmer_id: str = Field(..., description="Unique farmer identifier")
    image_bytes: bytes = Field(..., description="Raw image bytes (JPEG or PNG)")
    image_mime_type: str = Field(..., pattern=r"^image/(jpeg|png)$")
    gps_lat: float = Field(..., ge=-90.0, le=90.0)
    gps_lon: float = Field(..., ge=-180.0, le=180.0)
    crop_type: Optional[str] = Field(None, description="Optional farmer-supplied crop name")
    submitted_at: datetime
    is_followup: bool = Field(False)
    case_id: Optional[uuid.UUID] = Field(None, description="Required when is_followup=True")

    @field_validator("case_id")
    @classmethod
    def case_id_required_for_followup(cls, v, info):
        if info.data.get("is_followup") and v is None:
            raise ValueError("case_id is required when is_followup=True")
        return v


# ── Vision Agent ───────────────────────────────────────────────────────────────

class ClassProb(BaseModel):
    label: str
    prob: float = Field(..., ge=0.0, le=1.0)

class DetectedLeaf(BaseModel):
    class_id: int
    class_name: str
    bbox: List[float] = Field(..., min_length=4, max_length=4,
                              description="[cx, cy, w, h] normalised 0-1 (YOLO format)")
    det_conf: float = Field(..., ge=0.0, le=1.0)

class ImageQualityFlags(BaseModel):
    low_resolution: bool = False
    blurry: bool = False
    non_leaf_suspected: bool = False

class VisionOutput(BaseModel):
    """
    Output of the Vision Agent.
    severity_pct is always null: no per-leaf severity labels or masks exist in
    the training data. severity_note carries the mandatory explanation.
    """
    top_label: str
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Calibrated (temperature-scaled) softmax probability")
    top_k_probs: List[ClassProb] = Field(..., min_length=1, max_length=10)
    severity_pct: None = Field(
        None,
        description="Always null. Severity percentage is unsupported by current data."
    )
    severity_note: str = Field(
        "Severity percentage is not computed. No per-leaf severity labels or lesion masks "
        "exist in the training data. The PPT design target is acknowledged but cannot be "
        "implemented without expert-annotated mask data.",
        description="Mandatory explanation accompanying severity_pct=null"
    )
    image_quality_flags: ImageQualityFlags
    ood_flag: bool = Field(False, description="True when max confidence < 0.40 [ENG assumption]")
    leaf_detected: bool = Field(
        False, description="True if optional YOLOv8s detector found at least one leaf box"
    )
    detected_leaves: List[DetectedLeaf] = Field(default_factory=list)
    escalate_to_human: bool = Field(
        False,
        description="True when confidence < 0.60 OR (top1-top2) < 0.15 [ENG assumption]"
    )
    model_version: str
    temperature: float = Field(..., description="Temperature scalar applied to logits")


# ── Weather Agent ──────────────────────────────────────────────────────────────

class WeatherInput(BaseModel):
    gps_lat: float = Field(..., ge=-90.0, le=90.0)
    gps_lon: float = Field(..., ge=-180.0, le=180.0)
    query_datetime: datetime

class CurrentWeather(BaseModel):
    temperature_c: float
    relative_humidity_pct: float = Field(..., ge=0.0, le=100.0)
    precipitation_mm: float = Field(..., ge=0.0)
    wind_speed_kmh: float = Field(..., ge=0.0)
    cloud_cover_pct: float = Field(..., ge=0.0, le=100.0)
    dew_point_c: float

class DailyForecast(BaseModel):
    forecast_date: date
    temp_max_c: float
    temp_min_c: float
    humidity_max_pct: float = Field(..., ge=0.0, le=100.0)
    precipitation_sum_mm: float = Field(..., ge=0.0)
    precipitation_hours: int = Field(..., ge=0)
    wind_speed_max_kmh: float = Field(..., ge=0.0)

class WeatherRiskFlag(BaseModel):
    condition_name: str
    triggered: bool
    current_value: str          # Human-readable description of observed value(s)
    threshold: str              # Human-readable threshold expression
    threshold_source: str       # Citation; "[UNVERIFIED — I14]" appended where applicable

class WeatherOutput(BaseModel):
    weather_available: bool
    weather_partial: bool = False
    cache_hit: bool = False
    cache_stale: bool = False
    current_weather: Optional[CurrentWeather] = None
    forecast_7day: Optional[List[DailyForecast]] = None
    disease_risk_flags: List[WeatherRiskFlag] = Field(default_factory=list)
    weather_data_source: Optional[str] = None   # "open-meteo" | "openweathermap" | null
    weather_fetch_timestamp: Optional[datetime] = None


# ── RAG Agent ──────────────────────────────────────────────────────────────────

class RAGInput(BaseModel):
    disease_label: str          # e.g. "Tomato___Early_blight"
    top_k_probs: List[ClassProb]  # Passed for context

class RAGOutput(BaseModel):
    """
    Output of the RAG Agent.
    When rag_available=False or knowledge_gap_flag=True, all treatment fields are empty/null
    and the Recommendation Agent must produce a "consult agronomist" response.
    cost_estimate_note is null until RAG documents containing cost data are collected.
    """
    rag_available: bool
    knowledge_gap_flag: bool = False
    synthesis_failed: bool = False
    treatment_options: List[str] = Field(default_factory=list)
    active_ingredients: List[str] = Field(default_factory=list)
    application_timing: Optional[str] = None
    preventive_measures: List[str] = Field(default_factory=list)
    cost_estimate_note: None = Field(
        None,
        description="Always null until RAG knowledge base contains cost data [GAP]"
    )
    citations: List[str] = Field(default_factory=list,
                                 description="source_document values from chunk metadata")
    retrieval_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Mean cross-encoder reranker score of top-3 chunks"
    )
    raw_chunks: Optional[List[str]] = Field(
        None,
        description="Set when synthesis_failed=True; raw retrieved text as fallback"
    )


# ── Risk/Reason Agent ──────────────────────────────────────────────────────────

class RiskFactor(BaseModel):
    source: str                 # e.g. "vision_confidence", "weather_W03"
    value: str                  # Human-readable observed value
    contribution: float         # Signed contribution to risk_score [ENG assumption]
    tag: str                    # Provenance tag, e.g. "[ENG assumption]"

class RiskInput(BaseModel):
    vision: VisionOutput
    weather: WeatherOutput
    rag: RAGOutput

class RiskOutput(BaseModel):
    """
    Risk score is computed from disease threat + weather + history.
    Uncertainty (confidence, image quality, OOD) is kept SEPARATE and drives escalation only.
    A confident high-threat disease always scores high risk.
    A blurry low-confidence image always escalates, not inflates the risk score.
    """
    risk_level: str = Field(..., pattern=r"^(low|moderate|severe)$")
    risk_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Threat-driven composite score [ENG assumption]; "
                    "independent of classifier confidence"
    )
    factors: List[RiskFactor]
    escalate_to_human: bool
    uncertainty_level: str = Field(
        ...,
        pattern=r"^(high_confidence|moderate_confidence|low_confidence)$",
        description="Separate from risk_score; driven by U1-U6 uncertainty signals"
    )
    uncertainty_signals_triggered: List[str] = Field(
        default_factory=list,
        description="e.g. ['U1_low_confidence', 'U4_blurry_image']; does NOT affect risk_score"
    )
    weather_unavailable: bool = False
    rag_unavailable: bool = False
    progression_note: str = Field(
        ...,
        description="Text heuristic only. NOT a quantitative prediction. [ENG][GAP]"
    )
    risk_computed_at: datetime


# ── Recommendation Agent ───────────────────────────────────────────────────────

class RecommendationInput(BaseModel):
    risk: RiskOutput
    rag: RAGOutput

class RecommendationOutput(BaseModel):
    """
    Farmer-facing recommendation. All treatment content must be traceable to RAGOutput.citations.
    cost_estimate_note is null until knowledge base is built.
    """
    actions: List[str]          # Empty list + escalate_to_human=True when RAG unavailable
    urgency: str = Field(..., pattern=r"^(immediate|within_24h|within_48h|within_week|monitor|consult_agronomist)$")
    risk_level: str
    confidence_justification: str   # Plain-language explanation of confidence for farmer
    citations: List[str]
    caveats: List[str]              # Any limitations or warnings
    cost_estimate_note: None = Field(None, description="Always null [GAP — no cost data in RAG]")
    rag_grounded: bool              # False when RAG unavailable
    escalate_to_human: bool
    knowledge_gap_caveated: bool    # True when knowledge_gap_flag was set


# ── Monitoring Agent ───────────────────────────────────────────────────────────

class MonitoringInput(BaseModel):
    case_id: uuid.UUID
    farmer_id: str
    followup_image_bytes: bytes
    followup_submitted_at: datetime
    visit_number: int = Field(..., ge=2, description="Visit 1 = initial; follow-ups start at 2")

class DetectionChange(BaseModel):
    baseline_detected: bool
    followup_detected: bool

class MonitoringOutput(BaseModel):
    """
    'disease_status' characterises change between visits; NOT a clinical efficacy judgement.
    Confidence change (confidence_raw_delta) is NOT an efficacy signal.
    severity_delta_pct is always null: no severity data exists.
    """
    case_id: uuid.UUID
    visit_number: int
    baseline_label: str
    baseline_confidence: float
    followup_label: str
    followup_confidence: float
    label_same: bool
    disease_still_detected: bool
    new_label_is_healthy: bool
    confidence_raw_delta: float  # followup_confidence - baseline_confidence; NOT efficacy signal
    disease_status: str = Field(
        ...,
        pattern=r"^(same_disease_still_detected|no_disease_detected_at_followup|"
                r"different_disease_detected_at_followup|not_applicable_healthy_baseline|unknown)$",
        description="Characterises change between visits; NOT a clinical efficacy judgement"
    )
    detection_change: Optional[DetectionChange] = None
    severity_delta_pct: None = Field(
        None,
        description="Always null. Numeric severity delta is not computable."
    )
    severity_note: str = Field(
        "Numeric severity delta is not computable. No per-leaf severity labels or masks exist in "
        "the training data. PPT claim of '% change in affected area' is unsupported by current data.",
        description="Mandatory explanation accompanying severity_delta_pct=null"
    )
    same_image_flag: bool = False
    poor_quality_flag: bool = False
    crop_mismatch_flag: bool = False
    too_soon_flag: bool = False
    escalate_to_human: bool
    followup_vision: VisionOutput


# ── Case State Object (persisted across visits) ────────────────────────────────

class FollowUpRecord(BaseModel):
    visit_number: int
    submitted_at: datetime
    monitoring_output: MonitoringOutput
    risk_output: RiskOutput
    recommendation_output: RecommendationOutput

class CaseState(BaseModel):
    """
    Persisted across days. Loaded by Orchestrator on follow-up path.
    Stored as JSON in SQLite (Phase 1) or a document store (Phase 2).
    """
    case_id: uuid.UUID
    farmer_id: str
    created_at: datetime
    crop_type: Optional[str] = None
    initial_request: FarmerRequest
    initial_vision: VisionOutput
    initial_weather: WeatherOutput
    initial_rag: RAGOutput
    initial_risk: RiskOutput
    initial_recommendation: RecommendationOutput
    follow_ups: List[FollowUpRecord] = Field(default_factory=list)
    status: str = Field("active", pattern=r"^(active|escalated|closed)$")


# ── Final Farmer Response ──────────────────────────────────────────────────────

class FarmerResponse(BaseModel):
    case_id: uuid.UUID
    submitted_at: datetime
    is_followup: bool
    vision: VisionOutput
    risk: RiskOutput
    recommendation: RecommendationOutput
    monitoring: Optional[MonitoringOutput] = None   # Present on follow-up path only
    escalate_to_human: bool
    partial_result: bool = False    # True when one or more parallel branches failed


# ── Human Escalation Payload ───────────────────────────────────────────────────

class HumanEscalationPayload(BaseModel):
    """
    Sent to agronomist review queue when escalate_to_human=True.
    Contains all evidence needed for manual review.
    """
    case_id: uuid.UUID
    farmer_id: str
    escalation_reason: List[str]    # Human-readable list of reasons
    vision_output: VisionOutput
    weather_output: WeatherOutput
    rag_output: RAGOutput
    risk_output: RiskOutput
    draft_recommendation: RecommendationOutput  # Agent's best-effort; agronomist can override
    image_bytes: bytes
    escalated_at: datetime
    priority: str = Field(..., pattern=r"^(low|medium|high|urgent)$")
```

---

### 6d-iv. Risk Scoring Formula (Deterministic — `src/risk/scorer.py`)

```
RISK FORMULA:
  R = w_threat × threat_score(top_label) + w_weather × weather_score(flags) + w_history × history_score(case)

  w_threat  = 0.70   [ENG assumption]
  w_weather = 0.20   [ENG assumption]
  w_history = 0.10   [ENG assumption]

  Constraint: w_threat + w_weather + w_history = 1.0
  R clamped to [0.0, 1.0]

THREAT SCORE (from config/threat_table.json):
  HEALTHY          → 0.0
  LOW              → 0.25   [ENG assumption]
  MEDIUM           → 0.50   [ENG assumption]
  HIGH             → 0.85   [ENG assumption]
  DISEASE_DETECTED → 0.50   [ENG assumption — fallback when table cell is empty]

  NOTE [GAP]: The threat table is currently ENTIRELY EMPTY. All non-healthy classes
  use the DISEASE_DETECTED fallback (0.50). This caps first-visit R at 0.55.
  A plant pathologist must fill the table before "severe" is reachable on first visit.

WEATHER SCORE:
  1. Select disease-applicable rules from W01–W07 (W08 excluded — no citation [GAP])
  2. weather_score = triggered_applicable / total_applicable
  3. If no rules apply: weather_score = 0.0
  4. Tag all contributions "[UNVERIFIED — I14]"

HISTORY SCORE:
  First visit: 0.0
  Follow-up with "same_disease_still_detected": 0.10 [ENG assumption]
  Follow-up with "no_disease_detected_at_followup": -0.05 [ENG assumption]
  Else: 0.0

RISK LEVEL THRESHOLDS:
  R < 0.25   → "low"
  0.25 ≤ R < 0.60 → "moderate"
  R ≥ 0.60   → "severe"
  [ENG assumption — all thresholds to be calibrated]

UNCERTAINTY (separate from R — does NOT modify R):
  U1: vision.confidence < 0.60                   [ENG assumption — to be calibrated]
  U2: (top1.prob - top2.prob) < 0.15             [ENG assumption — to be calibrated]
  U3: image_quality_flags.low_resolution == True
  U4: image_quality_flags.blurry == True
  U5: vision.ood_flag == True
  U6: rag.knowledge_gap_flag == True

  escalate_to_human = ANY(U1..U6)
  uncertainty_level:
    0 signals → "high_confidence"
    1 signal  → "moderate_confidence"
    2+ signals → "low_confidence"

INVARIANT: Risk (disease identity) and Uncertainty (image/model quality) are NEVER conflated.
```

---

### 6d-v. Weather Disease-Risk Rules

Implement rules W01–W07 only. W08 is excluded (no peer-reviewed citation [GAP]).

| Rule | Disease / Pathogen | Condition | Threshold | Source |
|------|-------------------|-----------|-----------|--------|
| W01 | Late blight (P. infestans) | High humidity + cool nights | RH > 90% AND temp_min < 18°C | Fry (1977) |
| W02 | Powdery mildew (general) | Low humidity + warm days | RH < 50% AND temp 20–30°C | Yarwood (1957) |
| W03 | Bacterial spot (Xanthomonas) | Warm, wet, windy | temp > 24°C AND precip > 0 AND wind > 15 km/h | Ritchie & Dittapongpitch (1991) |
| W04 | Early blight (A. solani) | Warm + post-rain | temp > 24°C AND precip_hours > 2 | Jones & Jones (1974) |
| W05 | Septoria leaf spot (S. lycopersici) | Cool + wet | temp 15–25°C AND RH > 85% | Seem & Russo (1984) |
| W06 | Yellow leaf curl virus (whitefly) | Hot + dry | temp > 28°C AND RH < 40% | Byrne & Brown (1996) |
| W07 | Rust (Puccinia spp.) | Moderate + humid nights | temp 15–22°C AND dew_point within 2°C of temp_min | Eversmeyer & Kramer (2000) |

All thresholds are [UNVERIFIED — I14]. Tag every weather contribution in output.

---

### 6d-vi. Escalation Triggers

| Trigger | Condition | Priority |
|---------|-----------|----------|
| T1 | vision.confidence < 0.60 | medium |
| T2 | (top1.prob - top2.prob) < 0.15 | medium |
| T3 | image_quality_flags.low_resolution | medium |
| T4 | image_quality_flags.blurry | medium |
| T5 | vision.ood_flag | high |
| T6 | rag.knowledge_gap_flag | medium |
| T7 | risk_level = "severe" AND escalate_to_human = False (async notification) | high |
| T8 | monitoring.disease_status = "different_disease_detected_at_followup" | high |
| T9 | monitoring.disease_status = "same_disease_still_detected" AND visit ≥ 2 | high |
| T10 | monitoring.poor_quality_flag | medium |
| T11 | monitoring.crop_mismatch_flag | high |

Priority resolution: urgent > high > medium > low. Final = highest triggered.
urgent: T5 OR T8 OR T9 (extended persistence, visit ≥ 3).
high: T7 OR T11.
medium: T1, T2, T3, T4, T6, T10.

---

### 6d-vii. Monitoring Delta Logic

```
Pre-analysis checks (in order; abort at first failure):
  1. Same image (MD5 match) → same_image_flag=True; return early
  2. Image quality (Vision Agent flags) → poor_quality_flag=True; escalate
  3. Missing baseline (no visit_1 in CaseState) → MonitoringError("no_baseline")
  4. Too soon (same calendar day) → too_soon_flag=True; proceed with warning

Delta computation (after checks pass):
  5. Run Vision Agent on follow-up image → new_vision
  6. label_same = (new_vision.top_label == baseline_vision.top_label)
  7. disease_still_detected = ("healthy" not in new_vision.top_label.lower())
  8. new_label_is_healthy = ("healthy" in new_vision.top_label.lower())
  9. confidence_raw_delta = new_vision.confidence - baseline_vision.confidence
     NOTE: This is NOT an efficacy signal.
  10. severity_delta_pct = null  [DATA — always]

Disease status determination:
  if baseline was healthy:
    status = "not_applicable_healthy_baseline"
  elif new label is healthy:
    status = "no_disease_detected_at_followup"
    NOTE: Does NOT imply treatment worked. It is an observation only.
  elif disease still detected AND same label:
    status = "same_disease_still_detected"
  elif disease still detected AND different label:
    status = "different_disease_detected_at_followup"
  else:
    status = "unknown"

Crop plausibility: compare crop prefix of follow-up label to baseline label
  → crop_mismatch_flag=True if different
```

---

### 6d-viii. Build Order with Definition of Done

Each stage below has a definition of done (DoD). A stage is complete only when all DoD items are satisfied. **STUB** and **MOCK** markers indicate where real functionality is replaced by placeholders.

| Stage | What to build | Definition of Done | Required tests | Blocked by Phase B? |
|-------|--------------|-------------------|----------------|---------------------|
| **C1 Data Prep** | Exclusion list; dataset split files; class maps; D2 split verification | `config/exclusion_list.json` generated with all D1↔D2_OOD MD5 hashes. `config/class_map_d2.json` and `config/class_map_d1.json` written. Split image counts match Part 1 audit. `test_exclusion_list.py` passes. | `test_exclusion_list.py`: verify hash count ≥ 2,075; verify a sample of 10 random hashes are present; verify no D2_Train hashes are in the list. | No |
| **C2 Vision** | Train EfficientNet-B3 classifier on D2_Train. Calibrate temperature on D2_Val. Optionally train YOLOv8s on D1_train. | Checkpoint saved to `checkpoints/classifier/`. `temperature.json` saved. Macro-F1 on D2_Test_ID ≥ 0.75 [ENG assumption]. Macro-F1 on D2_Test_OOD ≥ 0.50 [ENG assumption] (after exclusions). ECE on D2_Val ≤ 0.05 [ENG assumption]. `test_classifier.py` passes. | `test_classifier.py`: load checkpoint; infer on 3 test fixture images; verify output matches `VisionOutput` schema; verify temperature > 0. | No |
| **C3 RAG Pipeline** | Ingest documents into ChromaDB. Implement embed → search → rerank pipeline. | Vector DB populated. Retrieval returns top-k results in `RAGOutput` schema. `test_retriever.py` passes. **STUB [MOCK]**: Until real documents collected (I16), use a **stub corpus** of 5–10 manually written disease summaries clearly marked `[STUB — not real treatment data]`. Retrieval precision@5 on stub queries: not gated (stub data). | `test_retriever.py`: insert 3 stub documents; query; verify top-k format matches schema; verify metadata fields present. | **YES — I16**. Build with stub; replace with real corpus later. |
| **C4 Weather Agent** | Implement fetch + cache + W01–W07 rule evaluator. W08 excluded. | `WeatherOutput` returned for valid GPS. Cache hit/miss behaviour verified. All 7 rules evaluated. `test_weather_agent.py` passes. | `test_weather_agent.py`: mock HTTP responses; verify all 7 flags fire correctly for synthetic inputs; verify cache returns same response within TTL; verify `WeatherUnavailable` when both APIs fail. | No |
| **C5 Risk Engine** | Implement `threat_table.py` + `scorer.py`. Load `config/threat_table.json`. | `RiskOutput` produced. All 4 worked examples from Part 4 (Cases 1, 1b, 2, 3) produce correct `risk_level` and `risk_score`. Uncertainty signals separate from risk score. `test_risk_scorer.py` passes. | `test_risk_scorer.py`: 4 deterministic test cases matching Part 4 worked examples; verify risk_level, risk_score, escalate_to_human, uncertainty_level. | **Partially — I19** (uses fallback until filled) |
| **C6 Recommendation** | Implement LLM-grounded recommendation + deterministic fallback. | `RecommendationOutput` produced. When RAG unavailable → `urgency="consult_agronomist"`, `rag_grounded=False`. When LLM unavailable → deterministic fallback fires. `test_recommendation_agent.py` passes. **STUB [MOCK]**: LLM call is **mocked** in tests; stub returns valid JSON with citations matching RAG output. | `test_recommendation_agent.py`: mock LLM; verify field completeness; verify citation present when RAG available; verify consult-agronomist path when RAG absent; verify grounding check rejects uncited claims. | **YES — I16 + I17** for full test |
| **C7 Orchestrator** | Implement LangGraph graph. Parallel fan-out for Vision+Weather+RAG. Case persistence. Follow-up path. | `FarmerResponse` returned for new-case path. Case JSON written to `data/cases/`. Follow-up path loads case and includes `MonitoringOutput`. Partial-failure path returns `partial_result=True`. `test_new_case_flow.py` and `test_followup_flow.py` pass. **STUB [MOCK]**: All agents **stubbed** in integration tests; stubs return valid schema objects with fixed values clearly marked `[STUB]`. | `test_new_case_flow.py`: stub all agents; verify response schema; verify case JSON written; verify thread_id set. `test_followup_flow.py`: load stubbed case; verify MonitoringOutput in response. | No (uses stubs) |
| **C8 Monitoring** | Implement plausibility checks + delta computation. Integrate into follow-up graph path. | All plausibility checks work (same-image, quality, crop-mismatch, too-soon). All disease_status values produced correctly. Escalation triggers T8–T11 fire. `test_monitoring_delta.py` passes. | `test_monitoring_delta.py`: synthetic MonitoringInput pairs covering all 5 disease_status values; verify same_image_flag, crop_mismatch_flag, too_soon_flag; verify T8–T11 escalation. | No |
| **C9 API / UI** | FastAPI routes. Streamlit demo UI. Startup env-var check. | API accepts multipart image upload + JSON fields → returns `FarmerResponse`. Startup raises clear error on missing env var. Streamlit shows disease label, risk level, recommendation. | `test_api.py`: POST test image fixture; verify response matches `FarmerResponse` schema; verify 422 on invalid input. | No |
| **C10 Evaluation** | Run evaluation protocols. Write report with measured numbers only. | Macro-F1 on Test_ID and Test_OOD measured and reported. ECE measured and reported. Reliability diagram generated. Confusion matrices generated. All blocked evaluations (RAG, Risk, Monitoring) marked "not evaluated" with reason. | Run `scripts/evaluate.py`; verify no training data in evaluation splits. | **Partially — I16, I19** for full eval |
| **C11 Docker** | Dockerfile + docker-compose. Volume mounts. | `docker-compose up` starts API + UI without errors. Checkpoints loaded from mounted volume. Env vars injected from `.env`. | Manual: run container, POST one request, verify response. | No |

---

### 6d-ix. Critical Path

```
C1 Data Prep (S: 1–3 days)
  └─→ C2 Vision (L: 1–3 weeks — longest; blocks everything downstream)
        ├─→ C5 Risk Engine (S: 1–3 days; can start once C2 schemas are stable)
        │     └─→ C6 Recommendation (M: 3–7 days; needs C5 + C3/mock + I17)
        │           └─→ C7 Orchestrator (M: 3–7 days; assembles all)
        │                 └─→ C8 Monitoring (S: 1–3 days)
        │                       └─→ C9 API/UI (M: 3–7 days)
        │                             └─→ C10 Evaluation (M: 3–7 days)
        │                                   └─→ C11 Docker (S: 1–3 days)
        └─→ C3 RAG Pipeline (M; parallel with C4/C5 after C1; blocked by I16)
        └─→ C4 Weather Agent (S; parallel with C3/C5 after C1; not blocked)
```

---

### 6d-x. Coding Rules

These rules MUST be followed by any coding agent or developer implementing LeafMind.

#### R1. No Fabricated Numbers
- Never hardcode accuracy, F1, latency, or any performance metric.
- All performance numbers must come from measured experiments.
- Use `"NOT MEASURED"` or `None` as placeholder; never `0.95` or `0.92`.

#### R2. Tag Provenance
- Every engineering assumption in code comments must include `[ENG assumption]`.
- Every threshold (0.60, 0.15, 0.40, 0.25/0.60 risk levels) must have a comment citing its provenance and noting it needs calibration.
- Weather thresholds must include `[UNVERIFIED — I14]`.

#### R3. Severity is Null
- `severity_pct` and `severity_delta_pct` are ALWAYS `None`.
- Never compute, estimate, or stub a severity percentage.
- Always include the `severity_note` explanation string.

#### R4. No Unsupported Claims
- The system "detects" or "identifies" diseases — never "confirms" or "diagnoses".
- Monitoring "observes" change — never claims "treatment worked" or "disease cured".
- "no_disease_detected_at_followup" is an observation, not a recovery claim.

#### R5. Stubs and Mocks Must Be Marked
- Every stub or mock must include `[STUB]` or `[MOCK]` in its docstring, variable name, or comment.
- Stub data must never look like real data. Use clearly fake values: `"STUB_disease_label"`, `confidence=0.00`.
- Integration tests using stubs must have test names containing `_stub_` or `_mock_`.

#### R6. Schema Compliance
- All agent inputs and outputs MUST pass Pydantic validation against the schemas in `src/schemas/schemas.py`.
- Never bypass Pydantic validation with `model_construct()` in production code.
- Test fixtures must also validate against schemas.

#### R7. Exclusion List Enforcement
- Before any evaluation on D2_Test_OOD, filter out all MD5 hashes in `config/exclusion_list.json`.
- Before any Few_Shot → OOD evaluation, remove the 47 overlapping filenames.
- Log the number of excluded files at evaluation time.

#### R8. LLM Grounding
- RAG synthesis and Recommendation Agent LLM calls MUST include the grounding system prompt (verbatim or equivalent from Part 4, section 4b-iv).
- LLM output MUST be validated: non-empty citations when RAG is available; no treatment claims absent from retrieved chunks.
- On LLM failure: use deterministic fallback, never silently skip.

#### R9. Weather Graceful Degradation
- If weather APIs fail, set `weather_available=False` and proceed.
- If RAG fails, set `rag_available=False` and proceed with "consult agronomist".
- Vision failure is fatal — abort the pipeline with a clear error.

#### R10. Threat Table Fallback
- If `config/threat_table.json` has an empty cell for a disease class, use `DISEASE_DETECTED` (0.50).
- Never invent a threat tier. The fallback is the only valid default.
- Log a warning when the fallback is used.

#### R11. Environment Variable Enforcement
- The app must check for `LLM_PROVIDER` at startup and fail with a clear error message if missing.
- `GEMINI_API_KEY` is required only when `LLM_PROVIDER=gemini`.
- All paths (`CASE_STORE_PATH`, `CHECKPOINT_DB_PATH`, `VECTOR_DB_PATH`) must have sensible defaults but be overridable.

#### R12. Test Coverage Requirements
- `test_risk_scorer.py`: must include all 4 worked examples from Part 4 (Cases 1, 1b, 2, 3).
- `test_monitoring_delta.py`: must cover all 5 `disease_status` values + all plausibility flags.
- `test_schemas.py`: must verify that example JSON from Part 3 validates against Pydantic schemas.
- Integration tests: must cover both new-case and follow-up paths.

#### R13. D1 Isolation
- D1 images are NEVER used in classifier training (would invalidate OOD evaluation — I4).
- If both models share a backbone family, they are trained independently from pretrained weights.
- The exclusion list covers D1↔D2 overlaps for evaluation only.

#### R14. Risk ≠ Uncertainty Invariant
- Risk score R is computed from threat_score + weather_score + history_score.
- Classifier confidence does NOT appear in the R formula.
- Uncertainty drives escalation (U1–U6) but NEVER inflates R.
- Verify this invariant in `test_risk_scorer.py`: a high-confidence serious disease must score high R; a low-confidence blurry image must escalate but not score high R.

---

### 6d-xi. Threshold Resolution Register

The following thresholds were left undefined or ambiguous in earlier parts. Each is resolved here as either an [ENG assumption] with a value or marked [GAP].

| Threshold | Location | Resolution | Value | Tag |
|-----------|----------|------------|-------|-----|
| Retrieval precision@5 for C3 DoD | Part 5, C3 row | [ENG assumption]: not gated on stub data; gate on real corpus when available | precision@5 ≥ 0.60 on hand-labelled query set when real corpus is built [GAP until I16 resolved] | [GAP] → [ENG assumption] when corpus exists |
| Classifier macro-F1 on Test_ID | Part 2, quality gate | [ENG assumption] | ≥ 0.75 | [ENG assumption] |
| Classifier macro-F1 on Test_OOD | Part 2, quality gate | [ENG assumption] | ≥ 0.50 | [ENG assumption] |
| ECE target | Part 2, calibration | [ENG assumption] | ≤ 0.05 | [ENG assumption] |
| Per-class recall floor | Part 2, quality gate | [ENG assumption] | ≥ 0.30 | [ENG assumption] |
| OOD flag threshold | Part 2, Vision Agent | [ENG assumption] | max_softmax_prob < 0.40 | [ENG assumption — calibrate on OOD test set] |
| Escalation confidence threshold | Part 3, Vision Agent | [ENG assumption] | confidence < 0.60 | [ENG assumption — calibrate on D2_Val] |
| Escalation gap threshold | Part 3, Vision Agent | [ENG assumption] | (top1 - top2) < 0.15 | [ENG assumption — calibrate on D2_Val] |
| Blurry detection threshold | Part 3, Vision Agent | [ENG assumption] | Laplacian variance < TBD | [GAP — requires labelled quality set to calibrate] |
| Risk level thresholds | Part 4, scorer | [ENG assumption] | low < 0.25; moderate 0.25–0.60; severe ≥ 0.60 | [ENG assumption] |
| Threat tier scores | Part 4, threat table | [ENG assumption] | HEALTHY=0.0, LOW=0.25, MEDIUM=0.50, HIGH=0.85 | [ENG assumption] |
| RAG knowledge gap threshold | Part 2, RAG Agent | [ENG assumption] | retrieval_confidence < 0.40 | [ENG assumption] |
| RAG retrieval minimum threshold | Part 2, RAG Agent | [ENG assumption] | top-1 chunk score < 0.20 → no synthesis | [ENG assumption] |
| Weather cache TTL (current) | Part 2, Weather Agent | [ENG assumption] | 30 minutes | [ENG assumption] |
| Weather cache TTL (forecast) | Part 2, Weather Agent | [ENG assumption] | 3 hours | [ENG assumption] |
| Monitoring "too soon" | Part 4, Monitoring | [ENG assumption] | same calendar day | [ENG assumption] |
| Monitoring persistence escalation | Part 4, T9 | [ENG assumption] | visit ≥ 2 (first follow-up) | [ENG assumption] |
| Monitoring extended persistence | Part 4, T9 note | [ENG assumption] | visit ≥ 3 → urgent priority | [ENG assumption] |
| D1 test carve-out fraction | Part 5, detector eval | [ENG assumption] | 15% of D1_train stratified by class | [ENG assumption] |
| Reranker model | Part 2, RAG | [ENG assumption] | ms-marco-MiniLM-L-6-v2 | [ENG assumption] |

---

### 6d-xii. What Is NOT Built (Explicit Non-Scope)

1. **No application code in this spec**. This document specifies what to build, not the code itself.
2. **No severity estimation**. severity_pct = null. Always.
3. **No lesion detection**. D1 bboxes are whole-leaf. The detector finds leaves in scenes.
4. **No treatment efficacy measurement**. No paired temporal data.
5. **No offline mode**. PPT future scope; not Phase 1.
6. **No multilingual support**. English only.
7. **No supply chain / pricing integration**. PPT future scope.
8. **No soil or growth-stage inputs**. No data or API.
9. **No cost estimates in recommendations**. No cost data in RAG.
10. **No real model results in this document**. All numbers are illustrative or [ENG assumption].

---

*End of Part 6.*
