# LeafMind Technical Specification -- Part 1: Understanding and Data Audit
Tags: [PPT]=explicit in slides | [DATA]=verified from datasets | [ENG]=engineering proposal | [GAP]=missing or ambiguous
Date: 2026-09-30

---

## 1a. PPT Extraction Summary

### Problem Statement [PPT]
Crop diseases cause 20-40% annual yield loss globally. Farmers in rural areas lack access to agronomists,
miss critical treatment windows, and frequently misidentify diseases. Manual diagnosis is slow, expensive,
and inaccessible at smallholder scale.

### Objectives [PPT]
1. Enable disease identification from a smartphone photo in seconds
2. Incorporate environmental context (weather, soil conditions, crop growth stage) into every risk assessment
3. Deliver actionable, confidence-scored recommendations
4. Monitor treatment progress through follow-up images and adaptive re-evaluation

### Agent Roster and Responsibilities [PPT]

| Agent              | Role                                                                         | Primary I/O                                                   |
|--------------------|------------------------------------------------------------------------------|---------------------------------------------------------------|
| Orchestrator       | Receives farmer request; logs unique case ID; dispatches parallel tasks      | Image + location -> case_id + dispatched subtasks             |
| Vision Agent       | CNN disease classification + detection; outputs label, confidence, severity  | Image -> disease_label, confidence (0-1), severity_proxy_pct  |
| Weather Agent      | Queries weather API (current + 7-day); flags disease-favourable conditions   | GPS + datetime -> weather struct + risk_flags                 |
| RAG Agent          | Semantic search in agricultural knowledge base for treatment guidance         | disease_label -> treatment_options, citation list             |
| Risk/Reason Agent  | Synthesises Vision+Weather+RAG into risk level and progression estimate      | All agent outputs -> risk_level, risk_score, factors          |
| Recommendation Agent| Converts risk assessment into farmer-actionable JSON advice with urgency    | risk_assessment -> recommendation_JSON                        |
| Monitoring Agent   | Receives follow-up image; computes severity delta vs baseline; triggers re-eval | follow_up_image + case_history -> severity_delta, updated_risk |

### Data Flow (verbatim from PPT) [PPT]
  Farmer Leaf Image
    -> Orchestrator (logs case ID)
      -> [PARALLEL] Vision Agent
      -> [PARALLEL] Weather Agent
      -> [PARALLEL] RAG Agent
    -> Risk/Reason Agent (synthesises all outputs)
    -> Recommendation Agent
    -> Farmer/UI

  Follow-up path:
    Follow-up Image -> Monitoring Agent -> Risk/Reason Agent -> Updated Recommendation

### Latency Timeline (PPT design TARGETS, not experimentally verified) [PPT]
T=0s:      Farmer uploads leaf image; Orchestrator logs unique case ID
T=0-2s:    Vision, Weather, RAG agents execute in parallel
T=2-3s:    Risk/Reason Agent synthesises outputs; computes risk score
T=3-4s:    Recommendation Agent generates advice; UI displays to farmer
T=+Days:   Follow-up image triggers Monitoring Agent; Risk/Reason re-evaluates

IMPORTANT: "Sub-4-second" is a design target stated in the PPT. It has NOT been experimentally
verified. The spec and any downstream code MUST NOT present this as a measured result.

### Key PPT Design Principles [PPT]
- Unique case IDs per farmer interaction
- Longitudinal case history with follow-up image linking
- Confidence/uncertainty as CORE design principle (explicitly stated in PPT slide 7)
- Human-expert (agronomist) escalation on low confidence
- Graceful degradation: one agent failure must not block the full pipeline
- Offline mode (mentioned as future scope, not current requirement)

### Research Contributions [PPT]
1. Novel Orchestrator-Specialist agent architecture for agricultural diagnostics
2. Integration of visual, meteorological, and knowledge-retrieval signals into a unified risk model
3. Closed-loop monitoring enabling treatment efficacy tracking over time
4. Explainable, confidence-scored outputs for low-literacy, resource-constrained farmers

### PPT-Stated Limitations [PPT]
- Poor image quality (lighting/resolution) degrades Vision Agent accuracy
- Visually similar disease symptoms increase misclassification risk
- RAG database may lack local/regional treatment practices
- Rural intermittent internet disrupts real-time agent execution

### Expected System Outputs [PPT]
- Structured JSON recommendation displayed in farmer UI
- Risk level (low/moderate/severe) with numeric score
- Disease label with confidence score and uncertainty bounds
- Severity percentage (leaf area affected)
- Treatment options with evidence citations (RAG-grounded)
- Monitoring updates with severity delta and trend
- Human-escalation payload when confidence is insufficient

---

## 1b. Per-Dataset Audit

### DATASET 1: Dataset/ (PlantDoc Object Detection)

Name: PlantDoc Object Detection Dataset [DATA]
Source (from YAML): Kaggle plantdoc-object-detection-dataset [DATA]
Format declared in YAML: YOLOv8 object detection [DATA]
Format verified from files: YOLOv8 bounding-box (5 fields: class_id cx cy w h, all normalised 0-1) [DATA]
  - Confirmed NOT segmentation masks or polygon coordinates (5-field format only)
  - Multiple bounding boxes per image observed (e.g., train_00073.txt has 32 bboxes)

File Counts [DATA]:
  Train: 2,335 images / 2,335 label files  (1:1 match, confirmed)
  Val:     236 images /   236 label files  (1:1 match, confirmed)
  Test:   ABSENT from filesystem
  Total: 2,571 labelled images

Total bounding-box annotations in train set: 8,444 [DATA]

Classes [DATA]:
  YAML claims nc=29 (classes 0-28). ALL 29 classes verified present in training labels.
  Val labels contain only classes 0-26. Classes 27 (Potato leaf, 11 boxes) and 28
  (Tomato two spotted spider mites, 2 boxes) are absent from validation split.
  Train/Val filename overlap: 0 files (no leakage) [DATA]

Class Distribution (train annotation counts) [DATA]:
  Class 12 Blueberry leaf:                    827 (max)
  Class 11 Tomato leaf yellow virus:           787
  Class 14 Raspberry leaf:                     539
   ...
  Class 27 Potato leaf:                         11
  Class 28 Tomato two spotted spider mites:      2 (min)
  Imbalance ratio: 827/2 = 413.5 (severe)

Image Properties [DATA]:
  Format: JPEG
  Resolution: highly variable (file sizes 2 KB to 12 MB -- indicates mixed source resolutions)
  Channels: assumed RGB (JPEG standard); not verified with PIL in this pass [GAP]

Quality Issues [DATA]:
  - Severe class imbalance (413x ratio)
  - Classes 27 and 28 near-unusable for supervised training
  - No test split present
  - Very high resolution variance -- preprocessing must standardise to fixed size
  - Some label-to-image alignments not verified at pixel level [GAP]

---

### DATASET 2: Dataset_2/ (PlantVillage + PlantDoc Hybrid Classification)

Name: Multi-source hybrid leaf disease classification dataset [DATA]
Sources verified from filename prefixes [DATA]:
  PV__  = PlantVillage (lab-controlled, uniform background)
  PD__  = PlantDoc (field images, variable backgrounds)
  ALT__ = Third-party alternative datasets

Format: ImageFolder classification (folder name = class label) [DATA]
  - NO bounding boxes, NO polygon masks, NO severity scores
  - Label information is class identity ONLY

Splits [DATA]:
  Train:    27,694 images  |  30 classes  |  Source: 100% PlantVillage
  Val:       5,940 images  |  30 classes  |  Source: 100% PlantVillage
  Test_ID:   5,929 images  |  30 classes  |  Source: 100% PlantVillage
  Test_OOD:  2,374 images  |  30 classes  |  Source: 88.3% PlantDoc + 11.7% ALT
  Few_Shot:    300 images  |  30 classes  |  Source: 89% PlantDoc + 11% ALT
  Total:    42,237 images

Classes (30) [DATA]:
  Apple (3): Apple_scab, Cedar_apple_rust, healthy
  Blueberry (1): healthy
  Cherry (1): healthy
  Corn (4): healthy, Cercospora_leaf_spot, Common_rust, Northern_Leaf_Blight
  Grape (3): Black_rot, Esca_(Black_Measles), healthy
  Peach (1): healthy
  Pepper_bell (2): Bacterial_spot, healthy
  Potato (3): Early_blight, Late_blight, healthy
  Raspberry (1): healthy
  Soybean (1): healthy
  Squash (1): Powdery_mildew
  Strawberry (1): healthy
  Tomato (9): Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot,
               Tomato_Yellow_Leaf_Curl_Virus, Tomato_mosaic_virus, healthy, (two_spotted_spider_mites absent)

Class Distribution (Train) [DATA] -- Top and Bottom 5:
  Most:
    Tomato___Tomato_Yellow_Leaf_Curl_Virus: 3,750
    Soybean___healthy:                      3,563
    Tomato___Bacterial_spot:                1,489
    Tomato___Late_blight:                   1,336
    Squash___Powdery_mildew:                1,285
  Least:
    Potato___healthy:          106
    Apple___Cedar_apple_rust:  193
    Peach___healthy:           252
    Raspberry___healthy:       260
    Tomato___Tomato_mosaic_virus: 261
  Imbalance ratio: 3750/106 = 35.4 (moderate-severe)

Cross-Split Overlap [DATA]:
  Train - Val:       0
  Train - Test_ID:   0
  Train - Test_OOD:  0
  Val   - Test_ID:   0
  Val   - Test_OOD:  0
  Test_ID - Test_OOD: 0
  Few_Shot - Train:  0
  Few_Shot - Test_OOD: 47 FILES OVERLAP (CRITICAL)

CRITICAL FINDING: 47 images from Few_Shot also appear in Test_OOD.
Any evaluation combining few-shot adaptation with OOD testing MUST remove these 47 files
from the OOD test set to prevent data leakage. [DATA]

What Each Split Can and Cannot Evaluate [DATA]:
  Test_ID:
    CAN:   In-distribution per-class accuracy (same PlantVillage distribution as Train)
    CANNOT: OOD robustness, real-world field generalization, cross-domain performance

  Test_OOD:
    CAN:   Cross-dataset generalization (PV->PD/ALT domain shift measurement)
    CANNOT: In-distribution accuracy, evaluation involving the 47 overlapping Few_Shot files

  Few_Shot:
    CAN:   Low-data adaptation experiments (~10 examples/class, PD/ALT distribution)
    CANNOT: Standard fine-tuning (too few samples); evaluation on the 47 files it shares with Test_OOD

---

## 1c. Dataset-to-Agent Purpose Mapping

| Dataset | Purpose | Agent Consumer | Input Format | Output | Key Preprocessing Steps |
|---------|---------|----------------|--------------|--------|-------------------------|
| Dataset/images + labels (D1) | Train/eval object detection model (29-class YOLO) | Vision Agent (detection mode) | JPEG + 5-field YOLO txt | Bboxes + class_id + confidence | Resize to 640x640; normalise [0,1]; mosaic augmentation |
| Dataset_2/Train+Val (D2-TV) | Train/validate 30-class image classifier | Vision Agent (classification backbone) | JPEG in class folders | Class logits + probabilities | Resize to 224 or 384; normalise mean/std; augment |
| Dataset_2/Test_ID | Evaluate classifier on in-distribution data | Evaluation pipeline | JPEG | Per-class accuracy metrics | Same as train (no augmentation) |
| Dataset_2/Test_OOD | Evaluate OOD generalisation (lab->field) | Evaluation pipeline | JPEG | OOD accuracy, F1, confusion | Same preprocessing; exclude 47 few-shot files if applicable |
| Dataset_2/Few_Shot | Few-shot adaptation experiments (optional) | Experimental | ~10 JPEG/class | Adapted model | Minimal; no heavy augmentation |
| ABSENT: Agricultural docs | Knowledge base for treatment retrieval | RAG Agent | External text docs | Indexed vector store | Chunking, embedding, indexing |

---

## 1d. Feasibility Matrix

| Task | Status | Evidence / Reason |
|------|--------|-------------------|
| 30-class leaf classification | SUPPORTED | D2 provides 27,694 train images in 30-class ImageFolder format [DATA] |
| 29-class object detection (locate lesions) | SUPPORTED | D1 provides 8,444 YOLO-format bbox annotations across 29 classes [DATA] |
| Severity estimation as ground-truth score | NOT SUPPORTED | No severity annotations or masks in either dataset [DATA] |
| Severity proxy via bbox area ratio | PARTIAL | D1 bbox coordinates allow: severity_proxy = sum(bbox_area)/image_area; this is a PROXY, not validated ground truth [ENG] |
| Treatment recommendation from data | NOT SUPPORTED | No treatment knowledge, dosage, or protocol data in either dataset [DATA] |
| Environmental / weather risk modelling | NOT SUPPORTED (data) | No weather co-variables in either dataset; requires external weather API [DATA] |
| RAG retrieval | NOT SUPPORTED (data) | No knowledge documents in either dataset; must ingest external agricultural literature [DATA] |
| Monitoring: severity delta computation | PARTIAL | No paired before/after images in either dataset; severity delta must be approximated via repeated classification/detection on follow-up images submitted by farmer [ENG] |
| OOD generalization evaluation | SUPPORTED | Test_OOD (D2) provides 2,374 real-world field images separate from PlantVillage train distribution [DATA] |
| Few-shot adaptation experiments | PARTIAL | Few_Shot (D2) provides ~10 images/class; 47-file contamination with Test_OOD must be managed [DATA] |
| Combining D1 and D2 for single model | NOT SUPPORTED (direct) | D1=YOLO-bbox, D2=ImageFolder; incompatible formats; requires format conversion and class name alignment [DATA] |

---

## 1e. Ambiguity Register

| ID | Ambiguity Description | Options | Most Defensible Choice |
|----|----------------------|---------|------------------------|
| A01 | Vision Agent PPT describes both "classification" and "object detection" in same slide | (a) Classification only, (b) Detection only, (c) Classification primary + detection optional | [ENG] Choice (c): primary = 30-class classifier (D2 supported); secondary = detection for lesion localisation and severity proxy (D1 supported); share backbone |
| A02 | Severity estimate PPT says "% leaf area affected" but no severity labels exist | (a) Drop severity output, (b) Bbox area ratio proxy, (c) Segmentation model | [ENG] Choice (b): compute bbox_area_sum / image_area from D1 detector output; label clearly as PROXY; do not claim accuracy without ground-truth severity annotations |
| A03 | Sub-4-second latency target: PPT states this as fact but no experiments done | (a) Remove claim, (b) State as design target | [ENG] Choice (b): retain as "aspirational engineering target"; all code and spec must say "target" not "achieves"; no latency figures in research claims |
| A04 | Monitoring Agent computes "% change in affected area" with no baseline ground truth | (a) Classifier confidence delta, (b) Detection bbox area delta, (c) Both with caveats | [ENG] Choice (c): offer both; primary = D1-style detection area delta; fallback = top-class probability delta; document precision limitations of each |
| A05 | RAG knowledge base has no data source | (a) External FAO/CABI/extension publications, (b) Stub for dev | [ENG] Choice (a) for production spec, (b) stub for Phase 1 development; specify real target sources: FAO extension notes, PlantVillage disease guides, CABI Crop Protection Compendium |
| A06 | Weather API unspecified in PPT | Multiple options | [ENG] Primary: Open-Meteo (free, no key required, REST JSON). Backup: OpenWeatherMap free tier. Cache TTL: 30 minutes for current; 3 hours for forecast |
| A07 | Human escalation threshold unspecified | PPT says "low confidence" without value | [ENG] Propose: escalate if max_softmax_prob < 0.60 OR (top1 - top2) gap < 0.15; tag as engineering assumption; must be calibrated on validation set |
| A08 | Few_Shot - Test_OOD 47-file overlap discovered | (a) Ignore, (b) Remove overlapping files | [ENG] Choice (b): must remove 47 overlapping filenames from Test_OOD evaluation set when any few-shot method is assessed; track overlap file list in dataset_audit |
| A09 | Classes 27 (Potato leaf, 11 boxes) and 28 (spider mites, 2 boxes) in D1 nearly unusable | (a) Train on all 29, (b) Merge rare classes, (c) Remove from training | [ENG] Choice (a) for detection model training with oversampling; for classification evaluation report separately and flag as unreliable; do NOT claim valid per-class performance on class 28 |
| A10 | "Disease progression timeline" in Risk/Reason Agent output has no data support | (a) Estimate from literature, (b) Remove from spec, (c) Rule-based heuristic with citations | [ENG] Choice (c): rule-based heuristic using known pathogen progression rates from plant pathology literature (e.g., Agrios Plant Pathology 2005); every rule tagged with source |
| A11 | D1 class "Potato leaf" (class 27) is healthy; naming inconsistent with disease-only framing | (a) Treat as healthy class, (b) Remap to D2 Potato___healthy | [ENG] Cross-map to D2 taxonomy; see Section 1c mapping table |
| A12 | Grape___Esca_(Black_Measles) exists in D2 but not in D1 | Class present in D2 only | [ENG] This class is classifiable (968 train images in D2) but not detectable via D1 (bbox model); can be included in classifier output; flag in detection model as unsupported |
