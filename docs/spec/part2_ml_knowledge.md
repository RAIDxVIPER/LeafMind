# LeafMind Technical Specification — Part 2: ML and Knowledge Components
Tags: [PPT]=explicit in slides | [DATA]=verified from datasets | [ENG]=engineering proposal | [GAP]=missing or ambiguous
Date: 2026-10-01

---

## 2a. Vision Model Specification

### 2a-i. Task Type Decision

The PPT states (Slide 5): "CNN for image classification and object detection" in the same agent. Two distinct tasks are required:

| Task | Dataset | Classes | Output |
|------|---------|---------|--------|
| **Classification** (primary) | Dataset_2 / D2 | 30 classes | class_id, softmax probabilities, top-1 label, confidence |
| **Detection** (secondary) | Dataset_1 / D1 | 29 classes | bboxes (cx,cy,w,h), class_id per box, confidence per box |

**Rationale** [ENG]: Classification is the primary operational path: a farmer submits one leaf image and receives a disease label. Detection adds localisation (leaf-level, not lesion-level; see `docs/dataset_audit.md` Section on bbox character) for scene-level leaf finding only. The two models are trained on separate datasets (D2 for the classifier, D1 for the optional detector) with incompatible label formats (ImageFolder vs. YOLO bbox); training pipelines must not be mixed.

**Classifier data isolation** [ENG][DATA]: The classifier is trained EXCLUSIVELY on D2_Train (PlantVillage images). D1 images are NOT used in classifier training. This is mandatory because 87.8% of D2_Test_OOD images are byte-identical to D1 images (I4, confirmed by MD5 audit); any use of D1 in the classification training set would invalidate the OOD evaluation entirely. If a shared backbone family (e.g. EfficientNet) is used for both models, the two models are trained independently from their respective pretrained ImageNet weights — there is no shared fine-tuning and no data crossover.

**Critical clarification** [DATA]: D1 bboxes enclose whole leaves, not individual lesions. The detector outputs "where is the diseased leaf in the scene", not "where are the lesions on the leaf". This is confirmed by visual inspection of 20 random train images (docs/bbox_samples/).

---

### 2a-ii. Architecture Choice

#### Classification Model

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base architecture | EfficientNet-B3 or ConvNeXt-Tiny | Both achieve strong plant disease performance on PlantVillage benchmarks in published literature; B3 has 12M params, ConvNeXt-Tiny has 28M — both feasible on consumer GPU [ENG] |
| Pretrained weights | ImageNet-1k (torchvision or timm) | Transfer learning from ImageNet is standard for plant disease tasks; reduces sample requirement [ENG] |
| Input resolution | 224×224 (baseline); upgrade to 384×384 if compute allows | 224 matches PlantVillage standard; 384 improves OOD performance [ENG] |
| Output head | Linear(in_features, 30) + softmax | 30-class classification [DATA] |
| Uncertainty output | Temperature-scaled softmax probabilities | Calibration via temperature scaling on D2_Val; enables A07 escalation threshold [ENG] |

#### Detection Model

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base architecture | YOLOv8-small (ultralytics) | YAML declares YOLOv8 format [DATA]; YOLOv8s balances speed and accuracy at 640×640; pre-trained on COCO [ENG] |
| Pretrained weights | COCO (yolov8s.pt) | Leaf images contain scenes similar to COCO; fine-tuning from COCO is standard practice [ENG] |
| Input resolution | 640×640 (letterbox) | YOLO standard; handles D1's variable resolutions (115–6,000px) via letterboxing [DATA][ENG] |
| Output | Bboxes + class_id (0–28) + confidence per box | YOLOv8 native output format [ENG] |
| NMS threshold | IoU=0.45, conf=0.25 (defaults); tune on D1_val | [ENG] starting point; calibrate per-class on val |

---

### 2a-iii. Preprocessing Pipeline

#### Classification (D2)

```
Step 1  — Mode conversion: force all images to RGB (handles 1 RGBA in D2_Train, mixed modes in OOD/FewShot)
Step 2  — Resize: shortest edge to 256, then centre-crop to 224 (or resize to 384 directly for higher-res variant)
Step 3  — Normalise: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  (ImageNet statistics)
Step 4  — Duplicate removal: exclude 11 within-split duplicate pairs from D2_Train before training
Step 5  — Leakage exclusion: exclude 5 D2_Train↔D2_Test_ID duplicates from test; exclude 4 D2_Train↔D2_Val from val
```

#### Detection (D1)

```
Step 1  — Mode conversion: force all images to RGB (5 CMYK, 4 RGBA, 1 L in D1_train)
Step 2  — Letterbox resize: shortest dimension padded to 640×640 with grey (114,114,114) padding
Step 3  — Normalise: pixel values to [0,1]
Step 4  — Duplicate removal: remove 1 within-split duplicate pair from D1_train
Step 5  — Cross-dataset exclusion (for evaluation only): if evaluating on D2_Test_OOD, exclude the 1,972 D1_train / 103 D1_val exact-MD5 matches
```

---

### 2a-iv. Data Augmentation

#### Classification augmentations (training only — do NOT apply to val/test)

| Augmentation | Parameters | Purpose |
|---|---|---|
| RandomHorizontalFlip | p=0.5 | Orientation invariance |
| RandomVerticalFlip | p=0.3 | Less common but valid for top-down leaf shots |
| RandomRotation | degrees=±30 | Rotational invariance |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1 | Lighting/colour variation between PV and field images |
| RandomResizedCrop | scale=(0.7, 1.0) | Zoom variation |
| RandomGrayscale | p=0.05 | Robustness to desaturated images |
| GaussianBlur | kernel=(3,7), p=0.2 | Mild blur (image quality variation) |
| Normalize | ImageNet mean/std | As above |

**Specifically NOT used**: CutMix or MixUp on detection labels (complex bbox adjustment); heavy geometric transforms (shear, perspective) without verified benefit on leaf datasets [ENG].

#### Detection augmentations (YOLOv8 built-in mosaic augmentation)

YOLOv8 natively applies: Mosaic4 (combines 4 images), RandomFlip, HSV jitter, Scale, Translate. These are tuned for detection tasks and handle D1's multi-leaf scene structure well. Mosaic is especially beneficial given D1's severe class imbalance (413x ratio). Oversampling [ENG]: classes with < 50 annotations (classes 27, 28) should have oversampling multiplier applied before mosaic to prevent near-zero occurrence.

---

### 2a-v. Train/Val/Test Split and Leakage

#### Classification splits (D2 only — D1 images NEVER used for classifier training)

| Split | Images | Source | Notes |
|---|---|---|---|
| Train | 27,694 → 27,683 (after dedup) | D2_Train (100% PV) | Remove 11 within-split duplicate pairs [DATA]; no D1 images used |
| Val | 5,940 → 5,936 (after dedup) | D2_Val (100% PV) | Exclude 4 files shared with Train [DATA] |
| Test_ID | 5,929 → 5,924 (after dedup) | D2_Test_ID (100% PV) | Exclude 5 files shared with Train [DATA] |
| Test_OOD | 2,374 (see note) | D2_Test_OOD (PD+ALT) | **When evaluating the classifier**: must exclude the 1,972 D1_train and 103 D1_val byte-identical files (I4) [DATA]; valid OOD evaluation requires these exclusions |
| Few_Shot | 300 (see note) | D2_Few_Shot (PD+ALT) | Exclude 47 files shared with Test_OOD before any few-shot→OOD eval [DATA] |

**Why D1 images are excluded from classifier training** [DATA][ENG]: 87.8% of D2_Test_OOD images (2,085 of 2,374) are byte-identical to D1_train or D1_val images (confirmed by MD5 audit). Training the classifier on any D1 images would mean the model has seen test images during training, making OOD evaluation results invalid. This is issue I4 and is a hard constraint, not a preference.

#### Detection splits (D1 only — classifier OOD evaluation is separate)

| Split | Images | Notes |
|---|---|---|
| Train | 2,334 (after dedup) | D1_train minus 1 duplicate pair [DATA] |
| Val | 236 | D1_val; 0 duplicates [DATA] |
| Test | ABSENT | [DATA] No test split in D1; D1_val is used as proxy |

**Detector ↔ OOD contamination** [ENG]: The D1-trained detector is NOT evaluated on D2_Test_OOD. Its only evaluation split is D1_val. When the detector is used at inference time on a new farmer image, it operates on that image alone and does not interact with any evaluation set. Therefore the D1↔D2_OOD overlap (I4) does not affect detector evaluation validity, only classifier evaluation validity.

**No D1 train-to-val leakage** [DATA]: 0 filename overlaps confirmed between D1_train and D1_val.

---

### 2a-vi. Class Imbalance Strategy

#### Classification (D2_Train imbalance ratio 35.4×)

1. **Weighted cross-entropy loss** [ENG]: compute per-class weight = total_samples / (n_classes × class_count); apply as `weight` argument to `nn.CrossEntropyLoss`.
2. **WeightedRandomSampler** [ENG]: sample mini-batches to achieve near-uniform class frequency.
3. **Report per-class F1 and precision/recall** in addition to top-1 accuracy — overall accuracy can be misleading under class imbalance.

#### Detection (D1_train imbalance ratio 413.5×)

1. **YOLOv8 class weights**: set `cls_pw` per-class based on inverse frequency.
2. **Oversample rare classes** (27, 28) via image-level oversampling before mosaic.
3. **Do NOT claim valid per-class mAP for class 28** (2 train annotations) [ENG]. Flag class 28 results as unreliable in all evaluation outputs.

---

### 2a-vii. Evaluation Metrics

#### Classification

| Metric | When | Notes |
|---|---|---|
| Top-1 Accuracy | All splits | Primary metric; report per-class and macro-averaged |
| Macro F1 | All splits | Corrects for imbalance |
| Per-class Precision/Recall | Test_ID, Test_OOD | Required to identify failure modes |
| Confusion Matrix | Test_ID, Test_OOD | Identify visually similar disease confusions |
| ECE (Expected Calibration Error) | Val | After temperature scaling; target ECE < 0.05 [ENG] |
| OOD Gap | (Test_ID acc − Test_OOD acc) / Test_ID acc | Measures domain shift severity |

#### Detection

| Metric | When | Notes |
|---|---|---|
| mAP@50 | D1_val | Primary YOLO metric |
| mAP@50:95 | D1_val | Stricter IoU metric |
| Per-class AP | D1_val | Report separately; flag class 28 as unreliable |
| Recall@conf=0.25 | D1_val | Ensure low false-negative rate in detection |

---

### 2a-viii. Calibration

**Method** [ENG]: Temperature scaling (Platt calibration variant).  
**Procedure**:
1. Train classifier to convergence on D2_Train.
2. Run inference on D2_Val; collect logits.
3. Optimise scalar temperature T via NLL minimisation: calibrated_prob = softmax(logits / T).
4. Evaluate ECE on D2_Val before and after; target ECE < 0.05.
5. T is a single scalar; no retraining required.
6. Apply T to all inference calls; include in saved model artifact as `temperature` field.

**Why this matters**: The escalation trigger (A07) uses `max_softmax_prob < 0.60` as a threshold. Uncalibrated neural networks are systematically overconfident; temperature scaling corrects this so the threshold is meaningful.

**Threshold provenance** [ENG assumption — not dataset-derived]: The values `0.60` (max softmax) and `0.15` (top1−top2 gap) are engineering starting points with no empirical basis yet. They must be calibrated on D2_Val after the first training run by measuring the precision/recall trade-off of the escalation trigger against a held-out agronomist decision set. Until calibration is done, these numbers MUST NOT be cited as validated thresholds.

---

### 2a-ix. Quality Gating (Model Release Criteria)

The following criteria must ALL be met before a model version is accepted for use by the Vision Agent:

| Gate | Threshold | Provenance |
|------|-----------|------------|
| Macro F1 on D2_Test_ID | ≥ 0.75 | [ENG assumption] Conservative floor consistent with published PlantVillage classification baselines; not derived from this project's data |
| OOD Macro F1 on D2_Test_OOD | ≥ 0.50 | [ENG assumption] Chosen to be stricter than random (1/30 ≈ 0.033) and feasible given known PV→field domain shift; not dataset-derived |
| ECE on D2_Val (post-calibration) | ≤ 0.05 | [ENG assumption] Standard calibration quality target in the ML calibration literature (Guo et al., 2017); not measured on this dataset |
| Per-class recall for any class | ≥ 0.30 | [ENG assumption] Minimum operability floor; classes below this are flagged for human review, not silently dropped |
| Class 28 (spider mites) detector AP | Not gated | [DATA] Only 2 training annotations; no reliable AP can be computed |

**All thresholds above are engineering assumptions**. None are derived from this project's data or from an agronomist consultation study. They must be re-evaluated after the first training run and ideally against a user-study measuring farmer/agronomist outcomes [ENG][GAP].

---

### 2a-x. OOD Handling

- **Primary OOD test**: D2_Test_OOD (PlantDoc/ALT field images vs. PlantVillage training distribution).
- **OOD gap reporting** (mandatory): report and discuss the ID→OOD accuracy drop; do not hide behind macro averages.
- **Mitigation strategies considered** [ENG]:
  - Style augmentation (ColorJitter, RandAugment) during training.
  - Domain-adaptive fine-tuning on Few_Shot (10 images/class, PD distribution).
  - MixUp between PV and PD training images (requires additional PD training images — not currently available [GAP]).
- **Offline inference note** [PPT]: PPT mentions future offline mode. The classification model (EfficientNet-B3, ~48MB) is feasible for mobile on-device deployment; the detection model (YOLOv8s, ~22MB) is also mobile-feasible with ONNX export [ENG].

---

### 2a-xi. Severity Estimation — UNSUPPORTED; Output is null [DATA]

**Ground truth severity annotations: ABSENT from both datasets** [DATA].

The PPT states (Slide 5) that the Vision Agent produces a "severity estimate (% leaf area affected)". This claim is **not supportable with the current data** for the following reasons, established in `docs/spec/part1_audit.md` (sections 1d, 1e ambiguity A02) and `docs/dataset_audit.md`:

1. **D1 (YOLO) bounding boxes enclose whole leaves, not lesions** [DATA]. Confirmed by visual inspection of 20 randomly sampled training images (docs/bbox_samples/). Bbox area / image area measures the fraction of the frame occupied by the leaf, which is determined by zoom level and shooting distance — not by disease extent.
2. **D2 (ImageFolder) has no spatial annotations at all** [DATA]. Class label only; no bounding box, no segmentation mask, no severity score.
3. **No per-leaf severity score, lesion mask, or percentage-diseased annotation exists in either dataset** [DATA].

**Decision (superseding A02)** [ENG][DATA]: Numeric severity output is **dropped** from Phase 1. The Vision Agent severity field is set to `null` with an explicit explanation string.

```
severity_pct: null
severity_note: "Severity percentage is not computed. No per-leaf severity labels 
or lesion masks exist in the training data. The PPT design target is acknowledged 
but cannot be implemented without expert-annotated mask data."
```

**What to output instead**:
- The disease label and classifier confidence score remain the primary outputs.
- The detector (D1) provides leaf localisation in scene images — its bbox coordinates identify which leaf in the scene is diseased, not how severely.
- Risk/Reason Agent uses the detector's presence/absence signal and classification confidence as indirect severity proxies in its rule-based scoring (specified in Part 4).

**Path to real severity** [GAP]: Severity estimation requires expert-annotated segmentation masks (pixel-level lesion delineation per leaf image). This is a data acquisition task, not an engineering task. Until such masks exist:
- Do NOT output a severity percentage.
- Do NOT use bbox area ratio as a severity proxy (it correlates with zoom, not disease).
- Do flag the severity gap in every downstream output that mentions it.

---

## 2b. RAG (Retrieval-Augmented Generation) Specification

### 2b-i. Purpose and Scope [PPT][ENG]

The RAG Agent retrieves treatment knowledge in response to a disease label output by the Vision Agent. The datasets contain **no treatment knowledge** [DATA]. The entire knowledge base must be built from external sources.

**RAG Agent input**: disease_label (string, one of the 30 D2 class names or 29 D1 class names)  
**RAG Agent output**: ranked list of relevant knowledge chunks with citations; structured treatment options

---

### 2b-ii. Target Knowledge Sources [ENG — DATA STILL TO BE COLLECTED]

> **IMPORTANT** [GAP]: **Zero knowledge documents exist in the current workspace.** The sources listed below are targets for acquisition. No document has been downloaded, licensed, cleaned, chunked, or indexed yet. This section specifies what to collect, not what is already available.

| Priority | Source | Type | License / Access | Collection Status |
|----------|---------|------|-----------------|------------------|
| 1 | FAO Plant Production and Protection Papers (disease management guides relevant to the 30 D2 classes) | PDF | Open access — verify per-document licence before redistribution | **NOT COLLECTED** [GAP] |
| 2 | CABI Crop Protection Compendium disease datasheets | Web/PDF | **Subscription required** — institutional access or licence needed; cannot use without licence | **NOT COLLECTED** [GAP] |
| 3 | PlantVillage disease guides (plantvillage.psu.edu) | HTML/PDF | Open access — verify terms of service for scraping | **NOT COLLECTED** [GAP] |
| 4 | University extension publications (UC Davis, Purdue, Cornell) | PDF | Open access — check each publication's licence | **NOT COLLECTED** [GAP] |
| 5 | EPPO pest datasheets (gd.eppo.int) | HTML | Open access — terms of use apply to redistribution | **NOT COLLECTED** [GAP] |
| 6 | Pesticide manufacturer application guides | PDF | Publicly available — check each document's terms | **NOT COLLECTED** [GAP] |

**Phase 1 stub** [ENG]: A small curated corpus (~50–100 documents) covering the 30 D2 classes, assembled by a human from the sources above after licence verification. The RAG pipeline cannot be built or tested until at least a stub corpus exists. This is a Phase B (Data Acquisition) prerequisite.

**Licence verification required before collection** [ENG]: For each source, confirm: (a) text can be downloaded, (b) redistribution in a private vector DB is permitted, (c) use for automated recommendation generation complies with terms of service. CABI in particular requires a paid subscription; a free-tier stub may be available for development.

**What must be covered per disease class** (content requirements): symptoms description, causative agent (fungal/bacterial/viral/pest), treatment options (chemical and biological), application timing, dosage guidance where available, preventive measures, and any known regional availability of treatments.

---

### 2b-iii. Document Cleaning and Chunking [ENG]

```
Cleaning:
  - Strip PDF headers/footers, page numbers, table-of-contents noise
  - Normalise encoding (UTF-8), normalise whitespace
  - Remove purely visual content (figure captions without context, decorative separators)
  - Retain: disease names, treatment names, dosage values, timing windows, source citations

Chunking strategy:
  Method: Hierarchical chunking — document → section → paragraph
  Target chunk size: 300–500 tokens (allows 2–3 chunks in a single RAG context window)
  Overlap: 50 tokens between consecutive chunks (prevents treatment protocol being split mid-sentence)
  Minimum chunk size: 100 tokens (discard shorter fragments)
  Metadata attached to every chunk:
    - source_document: str          (filename or URL)
    - source_type: str              (FAO / CABI / PlantVillage / extension / EPPO / manufacturer)
    - disease_class: List[str]      (which of the 30/29 classes this chunk is relevant to)
    - section_title: str            (e.g., "Chemical control", "Prevention")
    - chunk_index: int
    - char_offset_start: int
    - last_updated: date            (from document metadata where available)
```

---

### 2b-iv. Embeddings [ENG]

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (Phase 1 baseline) | 22M params; 384-dim vectors; fast inference; strong semantic similarity on domain text [ENG] |
| Upgrade path | `BAAI/bge-large-en-v1.5` (768-dim) or `text-embedding-3-small` (OpenAI API) | Better domain performance; decision deferred until baseline is validated [ENG] |
| Embedding dimension | 384 (baseline) | Trade-off: speed vs. recall |
| Batch embedding | 64 documents/batch | Hardware-dependent; adjust for available RAM |
| Query embedding | Same model as corpus | Query and corpus MUST use identical model; mismatching causes silent recall degradation [ENG] |

---

### 2b-v. Vector Database [ENG]

| Decision | Choice |
|----------|--------|
| Phase 1 (development) | **ChromaDB** (local, no server needed, persistent) |
| Phase 2 (production) | **Qdrant** (self-hosted) or **Weaviate** |
| Rationale | ChromaDB requires zero infrastructure; swap to Qdrant for production scaling without changing API surface [ENG] |
| Collection | One collection: `leafmind_knowledge` |
| Distance metric | Cosine similarity |
| Persistence | ChromaDB on-disk persistence; Qdrant with snapshots |

---

### 2b-vi. Retrieval Parameters [ENG]

```
Query construction:
  Input: disease_label (e.g., "Tomato___Early_blight")
  Query string: f"treatment and management for {disease_label.replace('___', ' ').replace('_', ' ')}"
  Optionally append: crop species extracted from label prefix

Top-k retrieval: k=10 (candidate pool)
Reranking: Cross-encoder reranker (ms-marco-MiniLM-L-6-v2 or equivalent)
  - Reranks top-10 candidates; returns top-3 for context
  - Reranking prevents embedding similarity from favouring syntactically similar but semantically irrelevant chunks

Filters (pre-retrieval metadata filters):
  - disease_class CONTAINS disease_label   (primary filter — reduces irrelevant crops)
  - source_type NOT IN [deprecated_list]   (optionally exclude outdated sources)

Final context window: top-3 reranked chunks (≈ 900–1,500 tokens) passed to LLM synthesis stage
```

---

### 2b-vii. LLM Synthesis and Citations [ENG]

The RAG Agent does NOT return raw retrieved chunks to the farmer UI. It uses an LLM to synthesise chunks into structured output.

```
Synthesis model options (Phase 1):
  Option A: gemini-1.5-flash (Google API) — fast, capable, JSON-mode output
  Option B: llama-3-8b-instruct (local via Ollama) — no API cost, slower

System prompt template (abbreviated):
  "You are a plant pathology assistant. Using ONLY the provided documents, answer:
   What treatment options exist for {disease_label}?
   Format your response as JSON with fields: treatment_options (list), active_ingredients (list),
   application_timing (str), preventive_measures (list), citations (list of source_document).
   Do NOT add any information not present in the documents. If the documents do not contain
   treatment information for this disease, state: 'No treatment information found in knowledge base.'"

Output fields:
  treatment_options: List[str]       (e.g., "Apply mancozeb 75WP at 2g/L, repeat every 7 days")
  active_ingredients: List[str]      (e.g., ["mancozeb", "chlorothalonil"])
  application_timing: str            (e.g., "At first symptom appearance; preventive before rainy season")
  preventive_measures: List[str]
  cost_estimate_note: str | None     (if cost data found in source; otherwise null)
  citations: List[str]               (exact source_document values from chunk metadata)
  retrieval_confidence: float        (mean reranker score of top-3 chunks, 0–1)
  knowledge_gap_flag: bool           (true if retrieval_confidence < 0.40)
```

---

### 2b-viii. Hallucination Controls [ENG]

| Control | Implementation |
|---------|---------------|
| Grounding constraint | System prompt explicitly: "Use ONLY the provided documents" |
| Citation mandate | LLM must list source documents; output is rejected if citations list is empty |
| Knowledge gap flag | If reranker score < 0.40, set `knowledge_gap_flag=True`; Recommendation Agent must note uncertainty |
| Fallback message | If top-1 chunk score < 0.20 (very low recall): return structured error "No relevant treatment information retrieved" rather than hallucinated advice |
| Human escalation trigger | knowledge_gap_flag=True OR retrieval_confidence < 0.40 → pass to agronomist escalation path |
| No unsupported numbers | LLM prompt explicitly: "Do not state dosages, prices, or timing that are not in the retrieved documents" |

---

### 2b-ix. Additional Data Needed [GAP]

The following are required to build the production RAG knowledge base. They are NOT present in the current repository:

| Item | Gap Severity | Source |
|------|-------------|--------|
| Treatment protocol documents for all 30 D2 classes | CRITICAL | FAO, CABI, PlantVillage, extension publications |
| Pesticide registration data (active ingredients, approved crops) | HIGH | National pesticide registries; EPA, EU pesticide database |
| Regional/local treatment practices (for smallholder farmers) | HIGH | Agricultural extension partners; NGO field guides |
| Cost estimates for treatments (by region) | MEDIUM | Market price databases; extension bulletins |
| Disease symptom descriptions (for validation) | MEDIUM | CABI, EPPO |
| Biological control options | LOW | IOBC guidelines; supplier catalogues |

---

## 2c. Weather Agent Specification

### 2c-i. Inputs and Outputs [PPT][ENG]

**Inputs** (from Orchestrator):
```
farmer_location: {lat: float, lon: float}   # GPS coordinates from device
query_datetime: datetime                      # Current UTC datetime
```

**Outputs** (structured response to Risk/Reason Agent):
```
current_weather:
  temperature_c: float
  relative_humidity_pct: float
  precipitation_mm: float          (last 1 hour)
  wind_speed_kmh: float
  cloud_cover_pct: float
  dew_point_c: float               (derived)

forecast_7day: List[DailyForecast]
  - date: date
  - temp_max_c: float
  - temp_min_c: float
  - humidity_max_pct: float
  - precipitation_sum_mm: float
  - precipitation_hours: int
  - wind_speed_max_kmh: float

disease_risk_flags: List[WeatherRiskFlag]
  Each flag:
    condition_name: str
    triggered: bool
    current_value: float
    threshold: float
    threshold_source: str          # Citation for the threshold value

weather_data_source: str
weather_fetch_timestamp: datetime
cache_hit: bool
```

---

### 2c-ii. API Selection [ENG]

**Primary**: [Open-Meteo](https://open-meteo.com/) (decision A06)
- Free, no API key required, REST JSON
- Provides: current weather + hourly/daily forecast up to 16 days
- Endpoint: `https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,precipitation_sum,precipitation_hours,wind_speed_10m_max&timezone=auto&forecast_days=7`
- Rate limit: ~10,000 calls/day (free tier, no key)

**Backup**: OpenWeatherMap free tier
- Requires API key; 60 calls/minute; provides similar fields
- Used only when Open-Meteo returns HTTP error

---

### 2c-iii. Caching [ENG]

```
Cache backend: In-memory dict (Phase 1); Redis (Phase 2 / production)  [ENG]
Cache key: f"{round(lat,2)}_{round(lon,2)}_{date.today()}"
  Rounding to 2 decimal places ≈ 1.1 km radius — acceptable for weather similarity  [ENG]
TTL:
  Current weather: 30 minutes   [ENG assumption] — balances freshness against API call volume;
                                  not derived from weather variability measurement;
                                  adjust based on observed API rate-limit behaviour
  7-day forecast: 3 hours       [ENG assumption] — daily forecasts change infrequently;
                                  value is a reasonable engineering default, not empirically validated
Eviction: LRU; max 1,000 cache entries (Phase 1 in-memory limit)  [ENG]
On cache hit: return cached response; set cache_hit=True in output
On cache miss: fetch API; store response; set cache_hit=False
```

---

### 2c-iv. Failure Handling [ENG]

| Failure Mode | Action |
|---|---|
| Primary API timeout (> 5s) | Retry once; if still failing, fall back to OpenWeatherMap |
| Both APIs unavailable | Return `WeatherUnavailable` flag; Risk/Reason Agent proceeds without weather context (graceful degradation per PPT Slide 10) |
| Invalid GPS coordinates | Return error to Orchestrator; do NOT call API with invalid coordinates |
| API rate limit hit | Return cached stale data (flag as `stale=True`); log for monitoring |
| Response missing required fields | Return `WeatherPartial` flag with available fields only; Risk/Reason Agent handles partial input |

**Key design principle** [PPT]: "One agent failure (e.g. weather API outage) does not block pipeline." The Risk/Reason Agent MUST handle `WeatherUnavailable` gracefully.

---

### 2c-v. Disease-Favourable Condition Rules [ENG]

The following rules flag weather conditions that favour rapid disease spread. All thresholds are sourced from plant pathology literature; each is tagged with its provenance.

**IMPORTANT**: These are engineering proposals based on published pathogen biology. They MUST be reviewed by a plant pathologist before production deployment [ENG][GAP].

| Rule ID | Disease / Pathogen | Condition | Threshold | Source |
|---------|-------------------|-----------|-----------|--------|
| W01 | Late blight (Phytophthora infestans) | High humidity + cool nights | RH > 90% AND temp_min < 18°C | Fry, W.E. (1977) — Phytophthora infestans epidemiology |
| W02 | Powdery mildew (general) | Low humidity + warm days | RH < 50% AND temp 20–30°C | Yarwood, C.E. (1957) — Powdery mildew biology |
| W03 | Bacterial spot (Xanthomonas) | Warm, wet, windy | temp > 24°C AND precipitation > 0 AND wind > 15 km/h | Ritchie & Dittapongpitch (1991) |
| W04 | Early blight (Alternaria solani) | Warm + post-rain leaf wetness | temp > 24°C AND precipitation_hours > 2 | Jones & Jones (1974) — Alternaria epidemiology |
| W05 | Septoria leaf spot (Septoria lycopersici) | Cool + wet | temp 15–25°C AND RH > 85% | Seem & Russo (1984) |
| W06 | Yellow leaf curl virus (vector: whitefly) | Hot + dry (whitefly peak) | temp > 28°C AND RH < 40% | Byrne & Brown (1996) — Bemisia tabaci activity |
| W07 | Rust (Puccinia spp.) | Moderate temp + humid nights | temp 15–22°C AND dew_point within 2°C of temp_min | Eversmeyer & Kramer (2000) |
| W08 | General epidemic risk (post-rain) | Wet period | precipitation_sum > 10mm in last 24h | [GAP] General agronomic heuristic — no peer-reviewed citation; threshold is approximate; omit from production until a plant pathologist confirms or cites a source |

Each triggered flag in `disease_risk_flags` must include:
- `condition_name`: human-readable description of the rule
- `triggered`: bool
- `current_value`: observed metric value
- `threshold`: the threshold value from the rule
- `threshold_source`: citation string from the table above

**MUST NOT**: Invent thresholds without citations. If a threshold cannot be sourced, the rule must be marked `[GAP]` and omitted from production until verified [ENG].

---
