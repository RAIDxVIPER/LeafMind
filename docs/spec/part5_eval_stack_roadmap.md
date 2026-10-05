# LeafMind Technical Specification — Part 5: Evaluation, Folder Structure, Tech Stack, Roadmap
Tags: [PPT]=explicit in slides | [DATA]=verified from datasets | [ENG]=engineering proposal | [GAP]=missing or ambiguous
Date: 2026-10-01

Component types: **ML model** = trained neural net | **LLM** = large language model | **Deterministic** = rule-based / algorithmic

**No latency, accuracy, or performance numbers are stated in this Part unless marked as measured. All thresholds inherited from Parts 2–4 carry their original provenance tags.**

---

## 5a. Evaluation Framework

### 5a-i. General Principles

- Report only **measured** numbers; mark everything else as "not evaluated" or "target [ENG assumption]".
- Every evaluation split and exclusion list must be stated explicitly before reporting a number.
- No number from this spec may appear in a paper or report without re-deriving it from the trained model.
- All evaluation is blocked by the Phase B prerequisites listed at the end of this section.

---

### 5a-ii. Vision Classifier Evaluation

**Component type**: ML model (EfficientNet-B3, trained on D2_Train only) [ENG][DATA]

#### Metrics

| Metric | Definition | Notes |
|--------|------------|-------|
| Top-1 accuracy | Fraction of test images where argmax(softmax) = true label | Overall headline number; report separately for each split |
| Top-5 accuracy | Fraction where true label is in top-5 softmax positions | Useful for near-miss analysis |
| Per-class precision | TP / (TP + FP) per class | Report all 30 classes; flag classes with < 10 test samples |
| Per-class recall | TP / (TP + FN) per class | Flag classes where recall < 0.30 [ENG quality gate, Part 2] |
| Per-class F1 | Harmonic mean of precision and recall per class | Primary per-class metric |
| Macro-F1 | Mean of per-class F1 scores (equal weight per class) | Primary aggregate metric; not weighted by class frequency |
| Weighted-F1 | Mean of per-class F1 weighted by test-set class frequency | Supplementary; skewed by class imbalance |
| Confusion matrix | 30×30 matrix of predicted vs. true labels | Must be reported; identifies cross-class confusions (e.g. early vs. late blight) |
| Top-k class probabilities | Softmax distribution over top-5 classes | Stored per image for calibration and escalation threshold analysis |

#### Evaluation splits and required exclusions [DATA]

**Test_ID** (D2_Test_ID, in-distribution PlantVillage):
- Remove the **5 files** shared with D2_Train (MD5 exact duplicates confirmed, Part 1) [DATA]
- Remove the **4 files** shared with D2_Val (MD5 exact duplicates confirmed, Part 1) [DATA]
- Effective Test_ID size after exclusions: 5,929 − 5 − 4 = 5,920 [DATA — arithmetic; verify against actual exclusion list]

**Test_OOD** (D2_Test_OOD, field/alternative images):
- Remove the **1,972 files** byte-identical to D1_train [DATA, Part 1 MD5 audit]
- Remove the **103 files** byte-identical to D1_val [DATA, Part 1 MD5 audit]
- Remove any **further D1 overlaps** found on re-audit (the exclusion list must be regenerated at training time from the actual D1 files used) [ENG]
- Effective minimum Test_OOD size after known exclusions: 2,374 − 1,972 − 103 = 299 images [DATA — arithmetic; actual count after regenerated exclusion list may differ]
- **IMPORTANT** [DATA]: With only 299 effective OOD images (minimum), per-class OOD recall is unreliable for most classes. Macro-F1 on Test_OOD should be interpreted with this in mind. Per-class breakdowns on Test_OOD are informative only, not conclusive.

**Test_ID vs. Test_OOD reporting**:
- Must be reported separately. Never merge into a single accuracy number.
- The Test_OOD macro-F1 gap vs. Test_ID macro-F1 is the primary domain-shift measure.

**Few-Shot evaluation** (D2_Few_Shot):
- Only include if a few-shot learning protocol is used (not required in Phase 1).
- If used: remove the **47 filename-overlap files** with Test_OOD before any Few_Shot → OOD generalisation evaluation [DATA, Part 1 — note: byte-identity of these 47 files is NOT confirmed; verify by perceptual hash per A14/I3].
- If not used: mark as "not evaluated in Phase 1".

#### Blocked evaluations
- Per-class OOD recall: partially blocked by small effective Test_OOD size [DATA]
- Few-shot evaluation: blocked until few-shot learning protocol is designed [GAP]

---

### 5a-iii. Detector Evaluation (Optional — YOLOv8s on D1)

**Component type**: ML model [ENG]

**D1_val is used for hyperparameter tuning only, not as a test set.** [ENG][DATA] A held-out test split must be carved from D1_train if the detector is built and evaluated. See I23.

#### Protocol [ENG assumption]

```
1. Before training, stratify D1_train (2,334 images after dedup) by class:
   Carve 15% as D1_test_carved (≈ 350 images), stratified so each of the 29 classes
   is represented proportionally. [ENG assumption — 15% is a reasonable holdout; exact
   fraction to be decided at training time]
2. Train on remaining D1_train_reduced (≈ 1,984 images).
3. Tune on D1_val (236 images): confidence threshold, NMS IoU threshold.
4. Evaluate on D1_test_carved: mAP@0.50, mAP@0.50:0.95, per-class AP.
```

#### Metrics

| Metric | Definition | Notes |
|--------|------------|-------|
| mAP@0.50 | Mean average precision at IoU threshold 0.50 | Standard detection metric |
| mAP@0.50:0.95 | Mean AP averaged over IoU thresholds 0.50–0.95 step 0.05 | Stricter; report as supplementary |
| Per-class AP@0.50 | AP per class at IoU 0.50 | Report all 29 classes |

**Class 27 and 28 note** [DATA][I6]: Class 27 has 11 annotations and class 28 has 2 annotations in D1_train. Per-class AP for these classes is not reliable and must be flagged explicitly. Do not use these classes to make claims about detector performance.

**Not evaluated**: Detector is not evaluated on D2_Test_OOD (would conflate classification and detection performance). [ENG]

---

### 5a-iv. Calibration Evaluation

**Component type**: Deterministic post-processing (temperature scaling) [ENG]

#### Metrics

| Metric | Definition | Notes |
|--------|------------|-------|
| Expected Calibration Error (ECE) | Mean absolute difference between mean confidence and accuracy within M equal-width confidence bins | Standard calibration metric; report with M=15 [ENG assumption] |
| Reliability diagram | Plot of mean accuracy vs. mean confidence per bin | Visual calibration check; must be included in evaluation report |
| Maximum Calibration Error (MCE) | Max of per-bin absolute difference | Supplementary; identifies worst-case miscalibration bins |

**Evaluation split**: D2_Val (after removing 4 Train-Val duplicates) [DATA]

**ECE target**: ≤ 0.05 is stated in Part 2. **This is an engineering assumption, not a validated benchmark.** [ENG — see I24; relabelled from Guo et al. 2017 attribution, which provides no target value.] Report the measured ECE; compare to target after measurement.

**Calibration procedure**: Temperature scalar T is tuned to minimise NLL on D2_Val. T is then applied at inference. ECE is measured after scaling. [ENG]

---

### 5a-v. Unknown / OOD Handling and Image-Quality Gate Evaluation

**Component type**: Deterministic (threshold-based) [ENG]

#### OOD flag accuracy

- `ood_flag` fires when `max_softmax_prob < 0.40` [ENG assumption — threshold to be calibrated]
- Evaluation requires a set of known-OOD images (images of non-plant objects, heavily corrupted images, or images from unseen crops) and a set of known-in-distribution images.
- **No such labelled OOD test set exists in the workspace** [GAP]. Evaluation is blocked until one is constructed.
- Minimum construction method [ENG]: sample 100 non-plant images from an open-licence image set (e.g. ImageNet non-plant classes), and 100 images from D2_Test_OOD for reference. Label manually. Report OOD flag precision and recall on this constructed set.

#### Image-quality gate evaluation

- `blurry` flag fires when Laplacian variance < threshold [ENG assumption — threshold to be calibrated]
- `low_resolution` flag fires when max(W, H) < 200px [ENG assumption]
- Evaluation requires manually labelled blurry/sharp image pairs.
- **No labelled quality test set exists** [GAP]. Blocked until constructed.
- Minimum construction method [ENG]: take 50 images from D2_Test_OOD, apply known Gaussian blur at varying sigma levels, label as "blurry" or "acceptable". Report precision/recall of blurry flag vs. the labelled set.

---

### 5a-vi. RAG Agent Evaluation

**Component type**: Deterministic retrieval + LLM synthesis [ENG]

**BLOCKED** [GAP]: All RAG evaluation is blocked by the absence of the knowledge base (I16). The following metrics and protocols are specified for future use.

#### Retrieval evaluation

| Metric | Definition | How to compute |
|--------|------------|----------------|
| Retrieval precision@k | Fraction of top-k retrieved chunks that are relevant to the query | Requires a hand-labelled query-relevant-chunk set [GAP — does not exist] |
| Retrieval recall@k | Fraction of all relevant chunks that appear in top-k | Requires the same hand-labelled set [GAP] |
| Mean Reciprocal Rank (MRR) | Mean of 1/rank of first relevant chunk | Requires the same hand-labelled set [GAP] |

**Construction of the hand-labelled query set** [ENG]: For each of the 30 disease classes, write 3 evaluation queries (e.g. "what are the treatment options for Tomato Early Blight?"). For each query, manually label all relevant chunks in the vector DB. This requires the knowledge base to exist first.

#### LLM synthesis evaluation

| Metric | Definition | Notes |
|--------|------------|-------|
| Faithfulness | Fraction of claims in the generated recommendation that are supported by a retrieved chunk | Score manually per claim; no automated metric is reliable [GAP] |
| Citation correctness | Fraction of cited documents that actually contain the attributed claim | Manual review |
| Hallucination rate | Fraction of generated recommendations containing at least one unsupported claim | Manual review; primary safety metric |
| Expert agreement | Fraction of recommendations rated as "acceptable" by an agronomist reviewer | Requires expert annotator [GAP] |

**All RAG evaluation is blocked** until: (a) knowledge base is built, (b) query-relevant-chunk labels are collected, (c) an agronomist reviewer is available. [GAP]

---

### 5a-vii. Risk/Reason Agent Evaluation

**Component type**: Deterministic [ENG]

**BLOCKED** [GAP]: Risk/Reason evaluation requires expert-labelled case scenarios (input → expected risk level pairs from an agronomist or plant pathologist). No such set exists.

#### Protocol [ENG]

```
1. Construct a hand-built scenario set: at minimum 30 cases (one per disease class + 3 control cases)
   covering low/moderate/severe expected risk levels.
   Each case specifies: disease_label, confidence, weather flags, history.
   Each case has an expected risk_level reviewed and approved by a plant pathologist.
2. Run the Risk/Reason formula on each scenario.
3. Report:
   - Agreement rate: fraction of scenarios where computed risk_level matches expert label
   - Confusion matrix: 3×3 (low/moderate/severe predicted vs. expert)
4. For continuous scores:
   - Spearman rank correlation between computed risk_score and expert severity ranking (if available)
```

**Calibration check** [ENG]: The formula weights (w_threat=0.70, w_weather=0.20, w_history=0.10) and tier-to-score mappings are engineering assumptions. Once the expert scenario set exists, a plant pathologist should review whether the formula ranks scenarios in the expected order. Formal weight calibration (logistic regression on expert labels) is deferred to after Phase B.

**Blocked by**: expert scenario set, populated threat table (I19), agronomist reviewer [GAP]

---

### 5a-viii. Recommendation Agent Evaluation

**Component type**: LLM (grounded to RAG) [ENG]

**BLOCKED** [GAP]: All recommendation evaluation requires the RAG knowledge base and an agronomist reviewer.

| Metric | Definition | Notes |
|--------|------------|-------|
| Evidence grounding rate | Fraction of action items traceable to a citation in `evidence_references` | Automated: check each action item against retrieved chunks |
| Actionability score | Expert rating of whether the advice is actionable at field level | Manual; agronomist reviewer required [GAP] |
| Hallucination rate | Fraction of recommendations with at least one unsupported treatment claim | Manual review; critical safety metric |
| Urgency calibration | Does urgency mapping (immediate / within_24h / etc.) match agronomist recommendation? | Manual; requires expert scenario set [GAP] |

---

### 5a-ix. Monitoring Agent Evaluation

**Component type**: ML + Deterministic [ENG]

No efficacy metrics are included. Evaluation covers only what can be assessed honestly with available data.

#### Plausibility-check evaluation (constructable) [ENG]

| Check | Evaluation method | Data needed | Status |
|-------|-------------------|-------------|--------|
| Same-image detection (MD5) | Generate 50 same-image pairs, 50 different-image pairs; verify flag | Synthetic — constructable | Can be done |
| Blurry follow-up flag | Same as image-quality gate evaluation (5a-v) | Labelled blurry/sharp pairs | Blocked [GAP] |
| Crop prefix mismatch flag | Generate 20 pairs with matching crop prefix, 20 with mismatching; verify flag | Constructable from D2 | Can be done |
| Too-soon flag | Construct timestamp pairs (same day, different day); verify flag | Constructable | Can be done |

#### Escalation behaviour evaluation [ENG]

- Construct 20 synthetic MonitoringOutput objects spanning all `disease_status` values.
- Verify that the correct escalation triggers (T8–T11) fire for each status and visit count.
- Report: trigger precision (no spurious escalations) and trigger recall (no missed escalations) on the constructed set.

#### What cannot be evaluated [DATA][GAP]

- Whether `disease_status = "no_disease_detected_at_followup"` corresponds to genuine recovery (no paired ground-truth images exist)
- Whether the diagnosis on follow-up is of the same physical plant as the baseline (no plant identity model)
- Treatment efficacy (no outcome data)

---

### 5a-x. System-Level Evaluation

**Measure all numbers; never assume.** [ENG]

| Metric | How to measure | Unit | Notes |
|--------|---------------|------|-------|
| End-to-end latency (new case) | Time from Orchestrator receipt to FarmerResponse | ms | Measure on 100 test requests; report median and 95th percentile separately |
| End-to-end latency (follow-up) | Same, for the follow-up path | ms | Separate from new-case path |
| Vision inference latency | Time inside Vision Agent (both models) | ms | Measure separately; GPU vs. CPU |
| Weather API latency (cache miss) | Time to fetch and parse response | ms | Measure on 50 cache-miss requests |
| RAG retrieval latency | Time for embed → search → rerank | ms | Separate from LLM synthesis time |
| LLM synthesis latency | Time for LLM API call to return | ms | Network-dependent; report separately |
| Agent failure rate | Fraction of requests where any agent raises an unhandled error | % | Measure on 200 test requests |
| Weather API failure rate | Fraction of weather fetch attempts that fail | % | Measure over 48h of test traffic |
| RAG retrieval failure rate | Fraction of queries returning knowledge_gap_flag=True | % | Measure after knowledge base is built |
| Cost per request | LLM token cost + API call cost (weather + embeddings) | USD | Measure on 50 requests at representative query length |
| Throughput | Requests/second with concurrent load | req/s | Load-test with locust or equivalent |

**Sub-4-second latency** [PPT]: This is a design target stated in the PPT. It must be measured, not assumed. Until measured, report as "not evaluated; target per PPT design specification". [ENG]

---

### 5a-xi. Blocked Evaluations Summary

| Evaluation | Blocked by | Phase B prerequisite |
|------------|------------|---------------------|
| RAG retrieval precision/recall/MRR | No knowledge base (I16) | Build corpus |
| LLM synthesis faithfulness, hallucination | No knowledge base, no agronomist | Build corpus; recruit reviewer |
| Risk/Reason agreement with expert | No expert scenario set, no threat table | Pathologist fills table; scenario set construction |
| Recommendation actionability, urgency calibration | No knowledge base, no expert | Both |
| OOD flag accuracy | No labelled OOD set | Construct manually |
| Image-quality gate accuracy | No labelled quality set | Construct manually |
| Monitoring efficacy / recovery detection | No paired temporal images | Field data collection |
| Few-shot generalisation | No protocol defined | Define protocol |
| ECE target validation | Not yet a validated benchmark | Accept as ENG assumption or replace with measured calibration |

---

## 5b. Project Folder Structure

**Student-friendly**: every folder has a clear, single purpose. No ambiguous names. [ENG]

```
LeafMind/                          ← project root (move OUT of OneDrive, see I8)
│
├── .env.example                   ← committed; template of required environment variables
│                                     (no values; values ONLY in .env which is git-ignored)
├── .gitignore                     ← exists; excludes Dataset/, Dataset_2/, .env, checkpoints/,
│                                     vector_db/, *.pth, *.pt, *.onnx, audit JSON, __pycache__
├── plan.md                        ← committed; project plan and progress tracker
├── README.md                      ← committed; project overview, setup steps, how to run
│
├── docs/                          ← committed; all specification and audit documents
│   ├── spec/
│   │   ├── part1_audit.md
│   │   ├── part2_ml_knowledge.md
│   │   ├── part3_agents.md
│   │   ├── part4_decision_logic.md
│   │   └── part5_eval_stack_roadmap.md
│   ├── dataset_audit.md
│   ├── ppt_extract.md
│   └── bbox_samples/              ← 20 sample bbox images reviewed in Part 1
│
├── config/                        ← committed; configuration files and static data
│   ├── exclusion_list.json        ← committed; MD5 hashes of D1 files to exclude from
│   │                                 Test_OOD/Few_Shot evaluation (small, ~2k entries)
│   ├── class_map_d2.json          ← committed; D2 class index → label mapping
│   ├── class_map_d1.json          ← committed; D1 class index → label mapping
│   └── threat_table.json          ← committed (initially all [GAP]); plant pathologist fills this
│
├── src/                           ← committed; all source code
│   ├── __init__.py
│   ├── agents/                    ← one file per agent
│   │   ├── orchestrator.py
│   │   ├── vision_agent.py
│   │   ├── weather_agent.py
│   │   ├── rag_agent.py
│   │   ├── risk_agent.py
│   │   ├── recommendation_agent.py
│   │   └── monitoring_agent.py
│   ├── models/                    ← model loading, inference wrappers (no training code here)
│   │   ├── classifier.py          ← EfficientNet-B3 load + infer + temperature scaling
│   │   └── detector.py            ← YOLOv8s load + infer (optional)
│   ├── schemas/                   ← all Pydantic schemas from Part 3
│   │   └── schemas.py
│   ├── rag/                       ← RAG pipeline components
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── synthesiser.py
│   ├── risk/                      ← Risk/Reason deterministic logic
│   │   ├── threat_table.py        ← loads config/threat_table.json
│   │   └── scorer.py
│   ├── monitoring/
│   │   └── delta.py               ← Monitoring delta computation
│   ├── persistence/
│   │   └── case_store.py          ← CaseState read/write (JSON Phase 1 / SQLite Phase 2)
│   ├── graph/
│   │   └── workflow.py            ← LangGraph StateGraph definition (nodes, edges)
│   └── api/
│       ├── app.py                 ← FastAPI application entry point
│       └── routes.py              ← API route definitions
│
├── tests/                         ← committed; all tests
│   ├── unit/                      ← unit tests for individual components
│   │   ├── test_risk_scorer.py
│   │   ├── test_monitoring_delta.py
│   │   ├── test_schemas.py
│   │   ├── test_image_quality.py
│   │   └── test_exclusion_list.py
│   ├── integration/               ← end-to-end tests using mocks/stubs
│   │   ├── test_new_case_flow.py
│   │   └── test_followup_flow.py
│   ├── fixtures/                  ← small test images, mock API responses (committed)
│   └── stubs/                     ← mock agents for integration tests (clearly marked as stubs)
│
├── scripts/                       ← committed; all scripts
│   ├── inspect/                   ← throwaway data audit scripts (see plan.md rules)
│   │   ├── inspect_images.py
│   │   ├── read_audit.py
│   │   └── write_audit_v2.py
│   ├── train_classifier.py        ← Phase C2; calls src/models/ only; no agent code
│   ├── train_detector.py          ← Phase C2 (optional)
│   ├── calibrate.py               ← temperature scaling on D2_Val
│   ├── build_exclusion_list.py    ← regenerates config/exclusion_list.json from D1 + D2 paths
│   ├── ingest_rag_docs.py         ← chunks, embeds, inserts docs into vector DB
│   └── evaluate.py                ← runs evaluation protocols from 5a
│
├── data/                          ← git-ignored (see .gitignore)
│   ├── cases/                     ← CaseState JSON files (Phase 1 persistence)
│   └── checkpoints.db             ← LangGraph SQLite checkpoint store
│
├── checkpoints/                   ← git-ignored; model checkpoints
│   ├── classifier/                ← EfficientNet-B3 .pth file + temperature scalar .json
│   └── detector/                  ← YOLOv8s .pt file
│
├── vector_db/                     ← git-ignored; ChromaDB persistent storage
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

**Notes**:
- `config/exclusion_list.json` is committed (small JSON, ~2k MD5 hashes) and kept outside `data/` so it is not git-ignored. [ENG]
- `config/threat_table.json` is committed as an initially-empty structure; the plant pathologist edits it directly. [ENG]
- `data/` and `checkpoints/` are git-ignored to avoid storing large binary or auto-generated files. [ENG]
- `.env.example` is committed and documents every required environment variable. `.env` is git-ignored. [ENG]
- Training scripts live in `scripts/`; they are separate from the source code to prevent training logic from being imported by the API. [ENG]

---

## 5c. Tech Stack

**Rule**: only include what is necessary; one-line reason each. Nothing included without a specific justified use. Secrets only via environment variables; the app must raise a clear error if a required variable is missing at startup. [ENG]

### Core dependencies

| Package | Version policy | Reason |
|---------|---------------|--------|
| **Python 3.11** | Pin minor version | Type-hint improvements (`TypedDict` inline annotations), better async support, broad ecosystem compatibility [ENG] |
| **PyTorch** (not TensorFlow) | Latest stable | EfficientNet-B3 via `timm`; YOLOv8s via `ultralytics` (both PyTorch-native); single framework; TensorFlow not needed [ENG] |
| **timm** | Latest stable | `efficientnet_b3` pretrained weights, standard fine-tuning API; widely used in plant disease literature [ENG] |
| **ultralytics** | Latest stable | YOLOv8s training and inference; official package; handles YOLO format natively [ENG] |
| **Pillow** | Latest stable | Image decoding, mode conversion (CMYK→RGB, RGBA→RGB), quality checks [ENG] |

### Orchestration and agents

| Package | Version policy | Reason |
|---------|---------------|--------|
| **langgraph** | Latest stable | Chosen over custom orchestrator in Part 3c; parallel fan-out, checkpointing, conditional edges; no LangChain chains used [ENG] |
| **pydantic v2** | Pin major version | All agent message schemas (Part 3b); data validation at agent boundaries; JSON serialisation; v2 for performance and `model_validate` API [ENG] |

### API and UI

| Package | Version policy | Reason |
|---------|---------------|--------|
| **FastAPI** | Latest stable | Async HTTP API; Pydantic schema integration; auto-generated OpenAPI docs for testing; lightweight [ENG] |
| **uvicorn** | Latest stable | ASGI server for FastAPI [ENG] |
| **Streamlit** (not React) | Latest stable | Phase 1 demo UI; student-friendly, minimal JS; React justified only if the UI requires complex state management that Streamlit cannot handle; defer React to Phase 2 if needed [ENG] |

### RAG pipeline

| Package | Version policy | Reason |
|---------|---------------|--------|
| **sentence-transformers** | Latest stable | `all-MiniLM-L6-v2` embedding model (Part 2b); local inference, no API key needed; well-benchmarked for retrieval tasks [ENG] |
| **chromadb** | Latest stable | Vector DB (Phase 1); embedded mode, no separate server; stores in `vector_db/`; git-ignored [ENG] |
| **cross-encoder (sentence-transformers)** | Latest stable | `ms-marco-MiniLM-L-6-v2` reranker (Part 2b); improves retrieval precision [ENG] |

### LLM provider

| Decision | Rationale |
|----------|-----------|
| **Configurable via `LLM_PROVIDER` env var**: `"gemini"` or `"ollama"` | I17 is open; the provider must not be hardcoded. The app reads `LLM_PROVIDER` at startup and selects the client. Missing variable → clear startup error. [ENG] |
| **gemini-1.5-flash** (if `LLM_PROVIDER=gemini`) | Fast, low-cost, JSON-mode output; API key via `GEMINI_API_KEY`. **Model name must be verified against the Google AI SDK before coding — model names change.** [ENG — I25] |
| **llama3** via Ollama (if `LLM_PROVIDER=ollama`) | Local inference, no API cost; requires Ollama running locally; model name via `OLLAMA_MODEL` env var (default: `"llama3"`). **Verify available model names in Ollama before hardcoding.** [ENG — I25] |
| **google-generativeai** Python SDK | For Gemini access; direct SDK call (no LangChain wrapper) [ENG] |
| **ollama** Python SDK | For Ollama access; direct SDK call [ENG] |

### Storage and caching

| Package | Version policy | Reason |
|---------|---------------|--------|
| **SQLite** (Phase 1) | stdlib `sqlite3` | CaseState persistence; LangGraph `SqliteSaver`; no server required for development [ENG] |
| **PostgreSQL + asyncpg** (Phase 2 only) | When needed | Production persistence and LangGraph `PostgresSaver`; defer until Phase 1 is complete [ENG] |
| **In-memory dict** (Phase 1 weather cache) | No package | Simple TTL cache; replace with Redis if multi-worker deployment is needed [ENG] |

### Utilities

| Package | Version policy | Reason |
|---------|---------------|--------|
| **python-dotenv** | Latest stable | Load `.env` into environment variables in development; verify `os.environ` values at startup [ENG] |
| **httpx** | Latest stable | Async HTTP client for Weather API calls (Open-Meteo, OpenWeatherMap); replaces requests in async context [ENG] |
| **pytest** | Latest stable | Test runner; all tests in `tests/` [ENG] |
| **pytest-asyncio** | Latest stable | Async test support for FastAPI and agent async functions [ENG] |

### Containerisation

| Tool | Reason |
|------|--------|
| **Docker** | Reproducible environment; `checkpoints/` and `vector_db/` mounted as volumes, not copied into image [ENG] |
| **docker-compose** | Defines service (API), volumes, and environment variable injection from `.env`; development convenience [ENG] |

### Dropped packages (not included, with reason)

| Package | Reason dropped |
|---------|----------------|
| LangChain | Not needed; direct SDK calls for all LLM, embedding, vector DB operations (Part 3c decision) [ENG] |
| TensorFlow / Keras | Not needed; PyTorch covers all ML needs [ENG] |
| Redis | Deferred to Phase 2; not required for Phase 1 [ENG] |
| React / Node.js | Deferred to Phase 2 if Streamlit proves insufficient [ENG] |
| numpy (standalone) | Transitively available through PyTorch/timm; only import directly if needed [ENG] |

### Required environment variables

The app must check for these at startup and fail clearly if any are missing. [ENG]

```bash
# .env.example  (committed — no real values here)

# LLM provider selection ("gemini" or "ollama")
LLM_PROVIDER=

# Gemini (only required when LLM_PROVIDER=gemini)
GEMINI_API_KEY=

# Ollama (only required when LLM_PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Weather APIs
OPENWEATHERMAP_API_KEY=       # backup weather API
# Open-Meteo needs no key

# Storage paths (override defaults if needed)
CASE_STORE_PATH=data/cases
CHECKPOINT_DB_PATH=data/checkpoints.db
VECTOR_DB_PATH=vector_db/

# App settings
LOG_LEVEL=INFO
```

---

## 5d. Roadmap and Workplan

### 5d-i. Stage Definitions

All effort labels are relative: **S** = 1–3 days, **M** = 3–7 days, **L** = 1–3 weeks (one student, full-time). These are rough calibrations only. [ENG assumption]

No calendar dates are given. Stages must be completed in dependency order; see the critical path below.

| Stage | What to build | Input | Output | Dependencies | Definition of Done | Required tests | Effort | Parallel? | Blocked by Phase B? |
|-------|--------------|-------|--------|-------------|-------------------|----------------|--------|-----------|---------------------|
| **C1 Data Prep** | Exclusion list; dataset split files; class maps; verify D2 splits | Dataset/, Dataset_2/, config/ | config/exclusion_list.json, verified split manifests | Part 1 audit complete | Exclusion list generated; all known D1-D2 overlaps present; split manifests have correct image counts; test_exclusion_list.py passes | test_exclusion_list.py (verify counts, check a sample of excluded files by MD5) | S | No — gates C2 | No |
| **C2 Vision** | Train EfficientNet-B3 classifier; train (optional) YOLOv8s detector; calibrate temperature | D2_Train (after C1 exclusions), D2_Val, D1_train (optional) | checkpoints/classifier/, checkpoints/detector/ | C1 done | Quality-gate thresholds in Part 2 met on Test_ID and Test_OOD (macro-F1, ECE); test_classifier.py passes | test_classifier.py (load checkpoint, infer on 3 test images, check schema compliance); test_calibration.py (verify T > 0 and finite) | L | No — gates everything | No |
| **C3 RAG pipeline** | Ingest corpus into ChromaDB; test retrieval and reranking | RAG documents (Phase B) | vector_db/, scripts/ingest_rag_docs.py | C1 done; knowledge base collected (I16) | Retrieval precision@5 > threshold on hand-labelled query set; test_retriever.py passes | test_retriever.py (mock DB; verify top-k format, schema compliance) | M | Yes (parallel with C4, C5 after C1) | **YES — blocked by I16** |
| **C4 Weather agent** | Implement fetch + cache + rule evaluator; write tests | Open-Meteo API, config/weather_rules (rules W01–W07 embedded in code, W08 excluded) | WeatherOutput (Pydantic) | C1 done; OWM API key if used | test_weather_agent.py passes with mocked API; cache hit/miss behaviour correct; W08 absent from rule list | test_weather_agent.py (mock HTTP; verify all flags, cache, fallback, WeatherOutput schema) | S | Yes (parallel with C3, C5 after C1) | No (uses ENG rules; I14 is a production concern not a build blocker) |
| **C5 Risk engine** | Implement threat_table.py + scorer.py; load config/threat_table.json | VisionOutput, WeatherOutput, RAGOutput, config/threat_table.json | RiskOutput (Pydantic) | C1 done; schemas from C2 path | test_risk_scorer.py passes all worked examples from Part 4 (Cases 1, 1b, 2, 3) | test_risk_scorer.py (deterministic; 4 worked-example cases; verify risk_level, escalation, uncertainty separation) | S | Yes (parallel with C3, C4 after C1) | **Partially — HIGH/MEDIUM/LOW tiers need I19 filled for non-fallback behaviour** |
| **C6 Recommendation** | Implement LLM-grounded recommendation with system prompt constraints; implement deterministic fallback | RiskOutput, RAGOutput, LLM provider | RecommendationOutput (Pydantic) | C5 done; C3 done (or mocked); LLM_PROVIDER set (I17) | test_recommendation_agent.py passes with mocked LLM (stub returns valid JSON); grounding check: no treatment claim absent from mocked RAG chunks | test_recommendation_agent.py (mock LLM; verify field completeness, citation present when RAG available, consult-agronomist path when RAG absent) | M | No — needs C5 | **YES — full test needs I16 and I17** |
| **C7 Orchestrator** | Implement LangGraph graph (nodes, edges, parallel fan-out, checkpointing); implement case persistence | All agent outputs; FarmerRequest schema | FarmerResponse; case JSON written to data/cases/ | C2–C6 done (or stubbed) | test_new_case_flow.py and test_followup_flow.py pass with stubs; case JSON written and reloadable; partial-failure path returns FarmerResponse with partial_result=True | test_new_case_flow.py (stubs for all agents; verify response schema, case written, thread_id set); test_followup_flow.py (load written case, verify MonitoringOutput in response) | M | No — assembles all components | No (can use stubs) |
| **C8 Monitoring** | Implement plausibility checks + delta computation; integrate into follow-up graph path | MonitoringInput, CaseState | MonitoringOutput (Pydantic) | C7 done; C2 done | test_monitoring_delta.py passes all plausibility checks and disease_status values; escalation triggers T8–T11 fire correctly | test_monitoring_delta.py (synthetic MonitoringInput pairs covering all disease_status branches; same-image/blurry/crop-mismatch flags) | S | No — extends C7 | No (can test with synthetic pairs) |
| **C9 API / UI** | FastAPI routes; Streamlit demo UI; startup env-var check | FarmerRequest (multipart); FarmerResponse | HTTP API; running Streamlit app | C7 done | API returns valid FarmerResponse on a real test image; startup raises clear error on missing env var; Streamlit shows disease label, risk level, and recommendation | test_api.py (integration; real request with test image fixture; check response schema) | M | No — wraps C7 | No |
| **C10 Evaluation** | Run all evaluation protocols from 5a on the trained model; write report | Trained checkpoints; test splits; config/exclusion_list.json | Evaluation report with measured numbers only | C1, C2 done; optionally C3–C9 done | Macro-F1 on Test_ID and Test_OOD measured and reported; ECE measured and reported; reliability diagram generated; all other metrics reported as "not evaluated" if blocked | No new tests — run evaluate.py; verify no training data leaked into evaluation splits | M | No — after C2 minimum | **YES — RAG/Risk/Monitoring evaluations blocked by I16, I19 [GAP]** |
| **C11 Docker** | Dockerfile + docker-compose; volume mounts for checkpoints and vector_db | Dockerfile, docker-compose.yml | Runnable container | C9 done | `docker-compose up` starts API and UI without errors; checkpoints and vector_db loaded from mounted volumes; env vars injected from .env | Manual: run container, POST one request, check response | S | No — after C9 | No |

---

### 5d-ii. Critical Path

```
C1 Data Prep
  └─→ C2 Vision (L — longest stage; blocks everything else)
        └─→ C5 Risk engine (S — can start immediately after C2 schemas are stable)
              └─→ C6 Recommendation (M — needs C5 and C3/mock)
                    └─→ C7 Orchestrator (M — assembles all)
                          └─→ C8 Monitoring (S)
                                └─→ C9 API/UI (M)
                                      └─→ C10 Evaluation (M)
                                            └─→ C11 Docker (S)

Parallel paths (can run alongside C2 once C1 is done):
  C1 → C3 RAG pipeline        [blocked by I16 — Phase B prerequisite]
  C1 → C4 Weather agent       [not blocked; safe to build early]
  C1 → C5 Risk engine         [can start in parallel with C2 once schemas are finalised]

Total critical-path length (rough): C1(S) + C2(L) + C5(S) + C6(M) + C7(M) + C8(S) + C9(M) + C10(M) + C11(S)
```

---

### 5d-iii. Milestones

| Milestone | Stages complete | Demo-able output |
|-----------|----------------|-----------------|
| **M1 — Vision and data prep done** | C1, C2 | Post a leaf image → get disease label + calibrated confidence + uncertainty flag. Evaluation numbers on Test_ID and Test_OOD visible in terminal. |
| **M2 — End-to-end on stubs** | C1, C2, C4, C5, C7 (with C3/C6 mocked) | Full pipeline response with real vision output, real weather output, mocked RAG and LLM. FarmerResponse schema complete. Follow-up path persists case and loads it. |
| **M3 — Monitoring loop + evaluation** | C1–C10 (C3 + C6 may still be partial if Phase B incomplete) | End-to-end with real risk scoring, monitoring delta, and (partial) evaluation report. RAG and recommendation fully functional only if Phase B is complete. |

---

### 5d-iv. Top Risks per Stage

| Stage | Top risk | Fallback |
|-------|---------|---------|
| C1 Data Prep | Exclusion list regeneration misses a file (D1-D2 overlap not caught) | Rerun build_exclusion_list.py on the actual training and test sets at training time; verify with MD5 spot check |
| C2 Vision | EfficientNet-B3 macro-F1 on Test_OOD below 0.50 [ENG quality gate] | Try ConvNeXt-Tiny as alternative backbone (Part 2a); increase augmentation; accept lower OOD performance with explicit documentation |
| C2 Vision | D2 class imbalance causes very low recall on rare classes | Apply class-weighted loss or oversampling (Part 2a); flag classes below 0.30 recall and exclude from automated use |
| C3 RAG | Knowledge base not collected before C3 deadline | Build a minimal stub corpus (~10 manually written disease summaries) sufficient to test the pipeline; mark as stub, not production |
| C4 Weather | Open-Meteo API changes structure or becomes unavailable | OpenWeatherMap backup implemented; if both fail, weather_available=False path is fully handled |
| C5 Risk | Threat table still empty at demo time | System operates on DISEASE_DETECTED fallback; add a banner to the UI: "Risk level is provisional until plant pathologist completes the threat table" |
| C6 Recommendation | LLM provider not chosen (I17) | Build with both providers behind the LLM_PROVIDER env var; test with the Ollama local provider first (no API cost); switch to Gemini for final demo |
| C7 Orchestrator | LangGraph API changes between versions | Pin langgraph to the version used during development; test graph nodes independently before integration |
| C8 Monitoring | No real follow-up images to test with | Test with synthetic pairs constructed from D2_Test_OOD (different images of the same class treated as baseline/follow-up); label test as "synthetic, not clinical" |
| C9 API/UI | Streamlit not sufficient for the demo | Identify limitation early; add minimal HTML form as a fallback if Streamlit's file upload limitations cause issues |
| C10 Evaluation | Test_OOD effective size of ~299 images gives unreliable per-class metrics | Report Macro-F1 on the full set with a clear note on the small OOD sample; do not draw per-class conclusions |
| C11 Docker | Large model checkpoint makes image too large | Mount checkpoints as Docker volumes, not inside the image; document the volume mount step in README |

---

### 5d-v. Phase B Blockers per Stage

| Stage | Phase B prerequisite | What is unblocked when resolved |
|-------|---------------------|----------------------------------|
| C3 RAG | RAG knowledge base collected and licensed (I16) | Full retrieval pipeline can be built and tested |
| C5 Risk (non-fallback) | Threat table filled by plant pathologist (I19) | Non-fallback risk scores; possible "severe" on first-visit high-threat diseases |
| C6 Recommendation (full) | RAG knowledge base (I16) + LLM provider key (I17) | Grounded treatment advice with real citations |
| C10 Evaluation (full) | Expert scenario set for Risk/Reason (I19); agronomist reviewer for RAG/Recommendation | Risk calibration and recommendation quality evaluation |
| Production deployment | Weather rules verified by plant pathologist (I14) | Remove [UNVERIFIED] caveat from weather flags in output |
| Production deployment | Threat table complete (I19) | Risk scores are meaningful; deployment advisable |
