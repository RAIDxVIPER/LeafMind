# Dataset Audit (v2 — updated with image inspection results)
> Source of truth: verified from filesystem, label files, and image inspection (Pillow + imagehash).
> Inspection script: scripts/inspect/inspect_images.py
> JSON results: docs/image_audit_raw.json
> Tags: [DATA] = directly observed | [ENG] = engineering inference | [GAP] = not checked

---

## DATASET 1: Dataset/ (PlantDoc Object Detection)

### Source
Path: Dataset/
Metadata: dataset.yaml (claims Kaggle plantdoc-object-detection-dataset, nc=29, YOLO format)
Format declared: YOLOv8 object detection
Format verified: 5-field YOLO bbox per annotation line (class_id cx cy w h, normalised 0-1) [DATA]

### File Counts [DATA]
| Split | Images | Label files | Match |
|-------|--------|-------------|-------|
| train | 2,335  | 2,335       | YES   |
| val   | 236    | 236         | YES   |
| test  | ABSENT | ABSENT      | N/A   |

### Image Dimensions and Channels [DATA]
| Split | W min | W max | W mean | H min | H max | H mean | Modes (count) |
|-------|-------|-------|--------|-------|-------|--------|---------------|
| train | 115   | 6,000 | 1,069  | 85    | 6,000 | 907    | RGB:2325 CMYK:5 RGBA:4 L:1 |
| val   | 115   | 6,000 | 1,055  | 69    | 5,312 | 950    | RGB:236 |

Observations:
- Resolution spans 3 orders of magnitude (115px to 6000px). The dataset is a web scrape with
  no resolution standardisation. All training pipelines MUST apply letterbox or crop-resize
  to a fixed size (e.g. 640x640 for YOLO).
- Non-RGB modes present in train: 5 CMYK, 4 RGBA, 1 L (grayscale). Model loaders that
  assume 3-channel RGB will fail on these files. Must convert to RGB at load time.
- Corrupt images: 0 detected in both splits [DATA]

### Bounding-Box Annotation Character [DATA]
Visual inspection of 20 randomly sampled images with boxes drawn (docs/bbox_samples/):

FINDING: The bounding boxes enclose WHOLE INDIVIDUAL LEAVES, not isolated lesion spots.

Evidence from 20 images:
  - train_00456 (Bell_pepper leaf spot, 1 box, area=0.52): box tightly wraps an entire
    diseased leaf; individual dark spots on the leaf surface are NOT individually boxed.
  - train_00895 (Squash Powdery mildew, 1 box, area=0.93): box fills the full frame;
    the leaf fills the frame; powdery white patches are NOT individually boxed.
  - train_00814 (Corn rust leaf, 1 box, area=0.9982): a close-up of a diseased corn leaf
    filling the whole frame; box = whole image; disease spots are NOT individually boxed.
  - train_00102 (Corn leaf blight, 1 box, area=0.73): the box encloses a whole corn leaf
    with blight streaks; individual blight areas NOT boxed.
  - train_01126 (grape leaf, 2 boxes, avg area=0.39): two individual grape leaves each
    wrapped in a separate box; healthy leaves.
  - train_01003 (Tomato leaf bacterial spot, 12 boxes, avg area=0.01): field scene with
    12 individual tomato leaves each in its own box; individual yellow/affected leaves
    are wrapped leaf-level, not spot-level.
  - train_00952 (Peach leaf, 22 boxes, avg area=0.005): peach tree; 22 small individual
    leaves in a scene; each box is a leaf, not a lesion.
  - train_02232 (Potato leaf early blight, 7 boxes, avg area=0.07): field scene; each
    diseased potato leaf has its own box; individual blight lesions NOT boxed.
  - train_00419 (Corn leaf blight, 1 box, area=0.19): a single diseased corn leaf in
    a scene; box encloses that one leaf.

CONCLUSION: All 20 inspected images confirm whole-leaf bounding boxes. The dataset is a
leaf DETECTION dataset: detect and classify the condition of a leaf in a scene.
Boxes do NOT annotate the area of diseased tissue within a leaf.

### Aggregate Bbox Area Statistics (all 8,444 train annotations) [DATA]
| Statistic | Value (fraction of image area) |
|-----------|-------------------------------|
| min       | 0.00011                       |
| p10       | 0.01235                       |
| p25       | 0.02717                       |
| p50 (median) | 0.06397                   |
| p75       | 0.15497                       |
| p90       | 0.44161                       |
| p95       | 0.72023                       |
| max       | 0.99909                       |
| mean      | 0.14887                       |
| % boxes > 50% of image | 8.7% (close-up single-leaf images) |
| % boxes > 25% of image | 16.3% |
| Mean box width  | not computed (see JSON) |
| Mean box height | not computed (see JSON) |

The wide area range reflects two image types in the dataset:
(a) close-up single-leaf: box ~50-100% of image (8.7% of annotations)
(b) field/scene images with multiple leaves: many small boxes (91.3% of annotations)

### Severity Annotation Status [DATA]
NONE. No per-leaf severity score, no lesion mask, no percentage-diseased annotation
exists in either dataset. The bbox is a whole-leaf wrapper, not a lesion boundary.

IMPLICATION (A02 resolution): Bbox area / image area is NOT a severity proxy.
It measures the fraction of the image frame occupied by the leaf, which is determined
by zoom level and shooting distance, not by disease severity.
The PPT's "severity estimate (% leaf area affected)" is UNSUPPORTED by the available data.

### Label Format [DATA]
- 5-field YOLO bbox: class_id cx cy w h (normalised 0-1). Verified from label files.
- No segmentation polygons, no per-pixel masks.
- Multi-bbox per image: YES (up to 32 boxes observed, train_00073.txt).
- Train/val filename overlap: 0 (no leakage).

### Class Coverage [DATA]
YAML claims nc=29 (classes 0-28). ALL 29 verified present in training labels.
Classes 27 (Potato leaf, 11 boxes) and 28 (spider mites, 2 boxes) absent from val.

### Class Distribution (train annotation counts) [DATA]
Class 12 Blueberry leaf:              827 (max)
Class 11 Tomato leaf yellow virus:    787
Class 14 Raspberry leaf:              539
Class  5 Strawberry leaf:             462
Class 21 Tomato Septoria leaf spot:   403
Class  2 Corn leaf blight:            356
Class  9 Potato leaf early blight:    314
Class 25 Bell_pepper leaf:            312
Class  1 Peach leaf:                  610
...
Class 27 Potato leaf:                  11
Class 28 Tomato spider mites:           2 (min)
Imbalance ratio: 827/2 = 413.5 (severe)

### Exact Duplicates (MD5) [DATA]
Within-split duplicates:
  D1_train: 1 duplicate pair (2 files with identical MD5)
  D1_val:   0 duplicates

Cross-split MD5 duplicates (same file in both datasets):
  D1_train <-> D2_Test_OOD:  1,972 exact duplicate files  *** CRITICAL ***
  D1_train <-> D2_Few_Shot:    238 exact duplicate files   *** CRITICAL ***
  D1_val   <-> D2_Test_OOD:   103 exact duplicate files   *** CRITICAL ***
  D1_val   <-> D2_Few_Shot:    25 exact duplicate files   *** CRITICAL ***
  D1_train + D1_val <-> D2_Test_OOD: 10 (appear in all three)
  D1_train + D1_val <-> D2_Few_Shot:  1

CRITICAL: D2_Test_OOD and D2_Few_Shot contain PlantDoc images that are IDENTICAL
(byte-for-byte) to images in D1_train and D1_val. These are the same PlantDoc source
images present in both datasets. Any cross-dataset evaluation that uses D1 for training
and D2_Test_OOD for evaluation is confounded by this overlap.

Required action: create a cross-dataset exclusion list of the 2,085 D1_train/D1_val files
that also appear in D2_Test_OOD or D2_Few_Shot before any such evaluation.

### Perceptual Hash (Near-Duplicate) Results [DATA]
Scope: D1_train, D1_val, D2_Test_OOD, D2_Few_Shot
Threshold: Hamming distance <= 8 (phash 64-bit)
D2 PlantVillage splits (Train/Val/Test_ID): NOT CHECKED (39K+ images; phash per image
~0.1s = >1 hour; not feasible in inspection phase) [GAP]

Results:
  Exact phash matches (hamming=0) across scoped splits: 2,310 groups
  All 2,310 exact phash groups are cross-split (D1 <-> D2 OOD/FewShot)
  Near-duplicates D1 train+val vs D2_Test_OOD + D2_Few_Shot (hamming<=8): 2,508 pairs
  Perceptual hash errors: 0

Interpretation: The 2,508 near-duplicate pairs are consistent with the 2,085 MD5-exact
cross-dataset duplicates (some pairs may be the same image with trivial JPEG re-encoding
causing slight pixel differences, hence hamming > 0 but <= 8).

---

## DATASET 2: Dataset_2/ (PlantVillage + PlantDoc Hybrid Classification)

### Source [DATA]
Path: Dataset_2/
Sources (from filename prefixes):
  PV__  = PlantVillage (lab-controlled, uniform background, standardised 256x256 JPEG)
  PD__  = PlantDoc (field images, variable backgrounds, variable resolution)
  ALT__ = Third-party alternative sources (similar characteristics to PlantDoc)
Format: ImageFolder classification — folder name = class label, no annotation files [DATA]

### File Counts [DATA]
| Split    | Images | Classes | Source |
|----------|--------|---------|--------|
| Train    | 27,694 | 30      | 100% PlantVillage (PV__) |
| Val      | 5,940  | 30      | 100% PlantVillage (PV__) |
| Test_ID  | 5,929  | 30      | 100% PlantVillage (PV__) |
| Test_OOD | 2,374  | 30      | 88.3% PlantDoc (PD__) + 11.7% ALT__ |
| Few_Shot | 300    | 30      | 89% PlantDoc (PD__) + 11% ALT__ |

### Image Dimensions and Channels [DATA]
| Split    | W     | H     | Modes |
|----------|-------|-------|-------|
| Train    | 256x256 fixed | 256x256 fixed | RGB:27693 RGBA:1 |
| Val      | 256x256 fixed | 256x256 fixed | RGB:5940 |
| Test_ID  | 256x256 fixed | 256x256 fixed | RGB:5929 |
| Test_OOD | 115–6000, mean 1164 | 85–5184, mean 973 | RGB:2365 CMYK:4 RGBA:4 L:1 |
| Few_Shot | 150–5616, mean 1182 | 148–6000, mean 1000 | RGB:300 |

Observations:
- PlantVillage splits are uniformly 256x256 RGB. All images are pre-resized and standardised.
  No preprocessing needed beyond normalisation and augmentation for these splits.
- Test_OOD and Few_Shot inherit the variable resolution and non-RGB modes from PlantDoc/ALT.
  Mode conversion (to RGB) required at load time. Resize to model input dimensions required.
- The 1 RGBA image in D2_Train and the mixed modes in OOD/FewShot must be handled.
- Corrupt images: 0 across all 5 splits [DATA]

### Class List (30 classes) [DATA]
Apple (3):         Apple_scab, Cedar_apple_rust, healthy
Blueberry (1):     healthy
Cherry (1):        healthy
Corn (4):          healthy (Corn maize), Cercospora_leaf_spot, Common_rust, Northern_Leaf_Blight
Grape (3):         Black_rot, Esca_(Black_Measles), healthy
Peach (1):         healthy
Pepper_bell (2):   Bacterial_spot, healthy
Potato (3):        Early_blight, Late_blight, healthy
Raspberry (1):     healthy
Soybean (1):       healthy
Squash (1):        Powdery_mildew
Strawberry (1):    healthy
Tomato (8):        Bacterial_spot, Early_blight, Late_blight, Leaf_Mold,
                   Septoria_leaf_spot, Tomato_Yellow_Leaf_Curl_Virus,
                   Tomato_mosaic_virus, healthy
NOTE: Tomato has 8 classes (not 9 — two_spotted_spider_mites is absent from Dataset_2).

### Class Distribution (Train) [DATA]
Most represented:
  Tomato___Tomato_Yellow_Leaf_Curl_Virus: 3,750
  Soybean___healthy:                      3,563
  Tomato___Bacterial_spot:                1,489
  Tomato___Late_blight:                   1,336
  Squash___Powdery_mildew:                1,285
Least represented:
  Potato___healthy:          106
  Apple___Cedar_apple_rust:  193
  Peach___healthy:           252
Imbalance ratio: 3750/106 = 35.4 (moderate-severe)

### Exact Duplicates (MD5) [DATA]
Within-split:
  D2_Train:    11 duplicate pairs (22 files). Likely images duplicated across class subfolders
               or within a class due to augmented copies in the source dataset.
  D2_Val:       0 duplicates
  D2_Test_ID:   0 duplicates
  D2_Test_OOD:  6 duplicate pairs (12 files) — within OOD split
  D2_Few_Shot:  0 duplicates

Cross-split MD5 duplicates involving D2 splits:
  D2_Test_ID <-> D2_Train: 5 exact duplicate files
  D2_Train   <-> D2_Val:   4 exact duplicate files
  D2_Test_ID <-> D2_Val:   1 exact duplicate file
  D2_Train/Val/Test_ID <-> D2_Test_OOD: 0 (different sources: PV vs PD/ALT, consistent)
  D2 cross-split PV-PV leakage: small (5+4+1=10 files total); note for evaluation.

### Cross-Dataset MD5 Duplicates (D1 <-> D2) [DATA]
See Dataset 1 section above. 2,359 hash groups involving D1 images appearing in D2 OOD/FewShot.
This is the most significant data hygiene issue found in the audit.

### Perceptual Hash Results [DATA]
Same as Dataset 1 section: 2,508 near-duplicate pairs between D1 and D2_OOD/FewShot.
D2 PlantVillage splits: NOT CHECKED [GAP]

### Split Evaluation Validity [DATA]
  Test_ID:    Evaluates in-distribution PV accuracy. CAN: per-class accuracy.
              CANNOT: OOD robustness; real-world field generalisation.
  Test_OOD:   Evaluates cross-domain generalisation (PV->PD/ALT).
              CAN: OOD accuracy per class. CANNOT: valid comparison when D1 trained model
              is used without excluding the 103 D1_val / 1972 D1_train duplicates.
  Few_Shot:   Evaluates low-data adaptation (~10 examples/class).
              47-file filename overlap with Test_OOD MUST be excluded when evaluating
              any model adapted using Few_Shot on Test_OOD.

---

## Cross-Dataset Summary

### Label Format Incompatibility [DATA]
Dataset_1: YOLO detection (bbox coordinates per annotation). Cannot directly combine.
Dataset_2: ImageFolder classification (class = folder name). No bbox.
These CANNOT be combined without format conversion. Must use separate training pipelines.

### What Is NOT Covered By Either Dataset [GAP]
1. Agricultural treatment knowledge (treatment protocols, fungicide dosages, application
   timing, cost) — ABSENT; RAG knowledge base must be built from external sources.
2. Per-leaf severity scores / lesion masks — ABSENT from both datasets.
3. Temporal sequences (paired before/after treatment images) — ABSENT.
4. Weather, GPS, soil, or crop growth-stage metadata — ABSENT.
5. Human expert severity annotations — ABSENT.
6. Supply chain / pesticide pricing data — ABSENT.
7. Near-duplicate check for D2 PlantVillage splits (Train/Val/Test_ID) — NOT CHECKED [GAP].

---

## Data Hygiene Action Items

| # | Issue | Severity | Action Required |
|---|-------|----------|-----------------|
| H1 | D1_train shares 1,972 exact-MD5 files with D2_Test_OOD | CRITICAL | Exclude these files from D2_Test_OOD before any cross-dataset eval |
| H2 | D1_train shares 238 exact-MD5 files with D2_Few_Shot | CRITICAL | Exclude before few-shot + OOD eval |
| H3 | D1_val shares 103 exact-MD5 files with D2_Test_OOD | HIGH | Exclude from OOD test set |
| H4 | D1_val shares 25 exact-MD5 files with D2_Few_Shot | HIGH | Exclude |
| H5 | D2_Train has 11 within-split duplicate pairs | MEDIUM | De-duplicate before training |
| H6 | D2_Train <-> D2_Val: 4 duplicate files | MEDIUM | Remove from val set |
| H7 | D2_Test_ID <-> D2_Train: 5 duplicates | MEDIUM | Remove from test set |
| H8 | D2_Test_OOD: 6 within-split duplicate pairs | LOW | De-duplicate |
| H9 | D1_train: 1 within-split duplicate pair | LOW | De-duplicate |
| H10 | Non-RGB images (CMYK, RGBA, L) in D1_train and D2_OOD | HIGH | Force-convert to RGB at dataloader |
| H11 | D2_Few_Shot vs D2_Test_OOD: 47 filename-overlap files (already known) | HIGH | Exclude from OOD eval when few-shot model used |
