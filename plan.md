# PLAN.md — Multi-Agent Crop Leaf Disease Detection & Farmer Advisory

> Living progress tracker. The agent MUST read this file first in every new chat and update it before stopping (including when a usage limit is near).

## Agent rules (read first)

1. On start: read this file, then inspect the repo and any existing spec files to verify what is actually done. Fix any wrong status below before continuing.
2. Never mark `[x]` unless the output file exists and is complete. Partially written = `[~]`.
3. Work on ONE part at a time. After finishing a part, update this file, then stop and wait for the user to say "continue".
4. Save each part's output to `docs/spec/partN_*.md` so work survives a chat reset. Do not keep deliverables only in chat.
5. Keep all ground rules from the original prompt: tag statements `[PPT]` / `[DATA]` / `[ENG]` / `[GAP]`, no fabricated numbers, no final application code, verify the PPT architecture and report mismatches.
6. Append a line to the **Session Log** at the end of every session.

Legend: `[ ]` not started · `[~]` in progress / partial · `[x]` done · `[!]` blocked

## Inputs

| Item | Path | Status |
|---|---|---|
| PPT 1 | `AI-Powered-Multi-Agent-System-for-Crop-Leaf-Disease-Detection-and-Farmer-Advisory.pptx` | [x] read |
| PPT 2 | `Agent-Roles-Weather-Agent.pptx` | [x] read |
| PPT text extract | `docs/ppt_extract.md` | [x] extracted |
| Dataset 1 (PlantDoc detection) | `Dataset/` | [x] inspected |
| Dataset 2 (PlantVillage+PlantDoc classification) | `Dataset_2/` | [x] inspected |
| Dataset audit | `docs/dataset_audit.md` | [x] complete |

## Part 1 — Understanding and data audit → `docs/spec/part1_audit.md`

- [x] 1a. PPT extraction (problem, objectives, agents, data flow, feedback loop, contribution, limitations, outputs)
- [x] 1a. PPT architecture verified vs. expected flow; mismatches reported
- [x] 1b. Dataset audit with real code (structure, counts, classes, distribution, missing, duplicates, image sizes/channels, splits)
- [x] 1c. Table: Dataset | Purpose | Used by agent | Input | Output | Preprocessing
- [x] 1d. Feasibility matrix (classification, detection, severity, treatment, env. risk, RAG, monitoring)
- [x] 1e. Ambiguity register (A01–A12)

## Part 2 — ML and knowledge components → `docs/spec/part2_ml_knowledge.md`

- [x] 2a. Vision model (task type, architecture, preprocessing, augmentation, split/leakage, imbalance, metrics, calibration, quality gating, OOD, severity method)
- [x] 2b. RAG (documents, cleaning, chunking, embeddings, vector DB, top-k, filters, reranking, citations, hallucination controls, extra data needed)
- [x] 2c. Weather (inputs/outputs, API, caching, failure handling, favourable-condition rules with threshold provenance)

## Part 3 — Agents, schemas, orchestration → `docs/spec/part3_agents.md`

- [x] 3a. Per-agent spec: Orchestrator, Vision, Weather, RAG, Risk/Reason, Recommendation, Monitoring (with example I/O)
- [x] 3b. Pydantic schemas (request, vision, weather, RAG, risk, recommendation, monitoring, final response, escalation)
- [x] 3c. Orchestration choice + state/Case object + persistence
- [x] 3d. Mermaid workflow diagram

## Part 4 — Decision logic → `docs/spec/part4_decision_logic.md`

- [x] 4a. Risk/Reason scoring (formula, threat table, weather score, history score, worked examples)
- [x] 4b. Recommendation logic (grounding rules, urgency mapping, LLM constraints)
- [x] 4c. Monitoring (disease_status, delta signals, plausibility checks, escalation triggers)
- [x] 4d. Human-in-the-loop triggers + agronomist payload (T1–T11, priority mapping)
- [x] 4e. Error handling table

## Part 5 — Evaluation, structure, stack, roadmap → `docs/spec/part5_eval_stack_roadmap.md`

- [x] 5a. Evaluation framework (classifier, detector, calibration, OOD, RAG, Risk, Recommendation, Monitoring, system-level)
- [x] 5b. Folder structure
- [x] 5c. Tech stack (with env vars, dropped packages)
- [x] 5d. Staged roadmap (C1–C11, critical path, milestones, risks, Phase B blockers)

## Part 6 — Validity review and build spec → `docs/spec/part6_build_spec.md`

- [x] 6a. Research validity check (component classification, claims requiring experiments, defensible claims)
- [x] 6b. Limitations (data L1–L8, model L9–L15, system L16–L19)
- [x] 6c. Additional data/APIs (AD1–AD10, 4 APIs)
- [x] 6d. FINAL BUILD SPECIFICATION (schemas, risk formula, weather rules, escalation triggers, build order C1–C11 with DoD, coding rules R1–R14, threshold register)

## Implementation stages (only after Part 6 is approved)

- [x] Data prep (C1) — 2026-10-10, all DoD items satisfied, 62/63 tests pass (1 skipped)
- [~] Vision (C2) -- code + smoke test done; full GPU training pending
- [ ] RAG (C3)
- [ ] Weather (C4)
- [ ] Risk (C5)
- [ ] Recommendation (C6)
- [ ] Orchestrator (C7)
- [ ] Monitoring (C8)
- [ ] API / UI (C9)
- [ ] Evaluation (C10)
- [ ] Docker (C11)

## Open issues / blockers

| ID | Severity | Description | Status |
|---|---|---|---|
| I1 | CRITICAL | RAG knowledge base does not exist; zero documents collected | Open |
| I2 | CRITICAL | No severity / lesion-mask annotations in either dataset; PPT severity claims unsupported | Resolved — severity_pct set to null with explanation (Part 2a-xi) |
| I3 | HIGH | 47-file filename overlap between D2_Few_Shot and D2_Test_OOD; byte-identity not confirmed by perceptual hash | Resolved — verified with perceptual hash; 0 of 47 are near-duplicates (different images with same filename). |
| I4 | CRITICAL | 87.8% of D2_Test_OOD byte-identical to D1 images (1,972 train + 103 val); cross-dataset evaluation is confounded | Resolved — exclusion list specified; classifier MUST NOT train on D1 |
| I5 | HIGH | D2_Train has 11 within-split duplicate pairs; D2_Train↔Val 4 leaks; D2_Train↔Test_ID 5 leaks | Resolved — dedup counts specified in preprocessing pipeline (Part 2a-iii) |
| I6 | MEDIUM | D1 classes 27 (11 boxes) and 28 (2 boxes) are near-unusable for training and evaluation | Resolved — flagged; class 28 AP not gated (Part 2a-ix) |
| I7 | MEDIUM | D1 has no test split; D1_val used as proxy | Open — carve D1_test from D1_train at training time (Part 5a-iii) |
| I8 | LOW | Project is on OneDrive — may cause path and sync issues | Open — recommend moving to local path |
| I9 | HIGH | Non-RGB images in D1_train (5 CMYK, 4 RGBA, 1 L) and D2_Test_OOD (4 CMYK, 4 RGBA, 1 L) | Resolved — force-convert to RGB at load time (Part 2a-iii) |
| I10 | MEDIUM | D1 bbox character: whole-leaf, not lesion-level; PPT implies lesion-level detection | Resolved — documented in dataset_audit.md and Part 2a-xi |
| I11 | LOW | D1 resolution spans 115–6000px; extreme variance | Resolved — letterbox to 640×640 (Part 2a-iii) |
| I12 | MEDIUM | Effective OOD test set after exclusions is only ~299 images; per-class OOD metrics unreliable | Open — effective OOD after exclusion is 283 images covering 7 of 30 classes (23 classes have zero). Note: Part 2/5 arithmetic 2,374-1,972-103=299 double-counted 10 files. Accepted limitation; report with caveat. |
| I13 | LOW | D2 PlantVillage splits (Train/Val/Test_ID) not checked for near-duplicates via perceptual hash | Open (GAP) — 39K images, time-intensive |
| I14 | HIGH | Weather disease-risk rules W01–W08 thresholds unverified by plant pathologist | Open — all weather contributions tagged [UNVERIFIED] |
| I15 | MEDIUM | PPT mentions soil conditions and crop growth stage; no data or API exists | Open (GAP) — omitted from all schemas |
| I16 | CRITICAL | RAG knowledge base not collected; blocks C3, C6 (full), C10 (full) | Open — Phase B prerequisite |
| I17 | MEDIUM | LLM provider not chosen (Gemini vs Ollama) | Open — both supported via LLM_PROVIDER env var |
| I18 | HIGH | Part 3 draft used `1.0 - confidence` as risk base (inverted risk/uncertainty) | Resolved — corrected in Part 4 (Change C1) |
| I19 | CRITICAL | Disease threat table entirely empty; no plant pathologist has filled tiers | Open — all disease classes are [GAP]; fallback to DISEASE_DETECTED=0.50 |
| I20 | MEDIUM | No paired before/after images for monitoring validation | Open (GAP) — field data collection required |
| I21 | HIGH | MonitoringOutput trend labels conflated confidence with disease status | Resolved — replaced with disease_status + confidence_raw_delta (Part 4, Change C2) |
| I22 | HIGH | RiskOutput risk_score folded uncertainty into risk | Resolved — separated; risk from threat table, uncertainty from U1–U6 (Part 4, Change C3) |
| I23 | MEDIUM | D1_val used for tuning, not as held-out test; detector test set must be carved | Open — specified protocol in Part 5a-iii |
| I24 | LOW | ECE ≤ 0.05 target attributed to Guo et al. 2017 but that paper provides no target | Resolved — relabelled as [ENG assumption] |
| I25 | LOW | LLM model names (gemini-1.5-flash, llama3) may change; verify before coding | Open |
| I26 | HIGH | Empty threat table caps max first-visit R at 0.55 ("moderate"); cannot reach "severe" | Open — documented limitation; blocks accurate risk scoring until I19 resolved |
| I27 | LOW | Windows cp1252 console encoding errors on ← ≥ ✓ symbols in logging; scripts exit code 1 despite correct output | Resolved — replaced Unicode arrows/checkmarks with ASCII in both build scripts |
| I28 | HIGH | Manifest paths use Windows backslashes; fails on Linux/Mac/Kaggle; D2Dataset previously substituted blank image silently | Resolved — resolve_image_path helper added; load failures >0.5% threshold raises RuntimeError with first 5 paths logged |
| I29 | HIGH | calibrate.py used entry["label"] instead of entry["class"] | Resolved — fixed key to entry["class"] matching manifests |
| I30 | HIGH | calibrate_temperature created temperature parameter on CPU against CUDA logits | Resolved — device-agnostic calibration on CPU logits; scaler moved back to model device |
| I31 | HIGH | Checkpointing: model.pth not updated with calibrated T; LeafClassifier.from_checkpoint did not set scaler temperature after load_state_dict | Resolved — best weights saved to model.pth with calibrated T baked into state dict; from_checkpoint overrides from temperature.json; round-trip verified |
| I32 | HIGH | GaussianBlur augmentation evaluated once at transform build-time via random.random() | Resolved — replaced with transforms.RandomApply per-image evaluation; textured checkerboard regression test added |
| I33 | MEDIUM | metrics.json lacked per-split confusion matrix, per-class precision/recall/f1/support, and OOD zero-support classes | Resolved — per-class metrics dictionary keyed by class name, per-split confusion matrix CSVs, and OOD zero_support_classes added |
| I34 | MEDIUM | --max-train-images used first-N slice; class weights computed on subsample instead of full manifest; --max-eval-images missing | Resolved — seeded stratified random sample guaranteeing >=1 image/class implemented; class weights computed from full manifest; --max-eval-images added |
| I35 | LOW | OOD ECE reported without explicit note that T was fitted on lab-style D2_Val | Resolved — caveat printed in summary, stored in metrics.json, and logged |
| I36 | HIGH | calibrate_temperature in train_classifier.py diverged to T=14.13 (ECE 0.617) due to LBFGS(lr=0.1) on raw T without line search | Resolved — implemented robust fit_temperature in classifier.py: optimizes log_T with LBFGS(lr=1, strong_wolfe) on CPU float64; clamps T in [0.05, 20]; checks NLL < NLL(T=1) else falls back to 1.0; shared across train_classifier and calibrate.py; pre/post NLL and ECE recorded in metrics.json under "calibration" |

## Decisions made

| # | Decision | Tag | Date | Ref |
|---|---|---|---|---|
| D1 | Classification is primary task; detection is optional/secondary | [ENG] | 2026-10-01 | A01, Part 2a-i |
| D2 | Severity output set to null; bbox area is NOT a severity proxy | [DATA][ENG] | 2026-10-01 | A02, Part 2a-xi |
| D3 | Sub-4-second latency is a design target, not a measured result | [PPT][ENG] | 2026-10-01 | A03 |
| D4 | Monitoring uses disease_status + confidence_raw_delta, not severity_delta | [DATA][ENG] | 2026-10-01 | A04, I21, Part 4c |
| D5 | RAG stub for Phase 1; real sources = FAO, CABI, PlantVillage, extensions | [ENG] | 2026-10-01 | A05 |
| D6 | Weather API: Open-Meteo primary, OpenWeatherMap backup | [ENG] | 2026-10-01 | A06, Part 2c |
| D7 | Escalation thresholds: confidence < 0.60 OR gap < 0.15 (to be calibrated) | [ENG] | 2026-10-01 | A07 |
| D8 | Remove 47 overlapping files from Test_OOD when evaluating few-shot | [DATA] | 2026-10-01 | A08 |
| D9 | Keep all 29 D1 classes; oversample rare ones; flag class 28 as unreliable | [ENG] | 2026-10-01 | A09 |
| D10 | Risk ≠ Uncertainty: risk from disease identity, uncertainty from model quality | [ENG] | 2026-10-01 | Part 4a-i |
| D11 | Risk formula: R = 0.70×threat + 0.20×weather + 0.10×history | [ENG] | 2026-10-01 | Part 4a-iii |
| D12 | LangGraph chosen over custom orchestrator for checkpoint/resume support | [ENG] | 2026-10-01 | Part 3c-i |
| D13 | LLM used only for RAG synthesis and Recommendation; all else deterministic or ML | [ENG] | 2026-10-01 | Part 3 principle |
| D14 | Classifier trained exclusively on D2_Train; D1 images never in classifier pipeline | [DATA][ENG] | 2026-10-01 | I4, Part 2a-v |
| D15 | Threat table uses GAP fallback (DISEASE_DETECTED=0.50) until pathologist fills it | [ENG][GAP] | 2026-10-01 | Part 4a-ii, I19 |
| D16 | After applying config/exclusion_list.json, D2_Test_OOD has only 283 images covering 7 classes. Because the classifier is never trained on D1 (D14/R13), D1<->OOD overlap cannot leak into classifier evaluation. Primary OOD evaluation uses full deduped Test_OOD set. Exclusion-filtered set (283 images) is a secondary report only. Deviates from R7 as written. Exclusion filtering stays mandatory if any D1-trained model (e.g. YOLO) is evaluated on Test_OOD. | [ENG] | 2026-10-10 | I12 |

## Session Log

| Date | Agent/Session | What was done | Next step |
|---|---|---|---|
| 2026-10-01 | New chat after limit reset | plan.md created; awaiting repo check | Verify existing work, update statuses, resume at first unchecked item |
| 2026-10-01 | Session 2 | Parts 1–5 written across multiple sessions | Part 6 |
| 2026-10-10 | Session 4 (C1) | Implemented Stage C1 (Data Prep) in full. Folder structure created. schemas, threat table (healthy classes fixed), exclusion_list.json (2,349 hashes), and split manifests generated. Addressed 5 fixes: Val<->Test_ID leak removed (Test_ID=5,923), Test_OOD label-conflicting duplicates removed (Test_OOD=2,363), few_shot_test_ood_overlap.json verified via pHash (0 near-duplicates, 47 different images), absolute paths removed from manifests. Unit tests fully passing (62 passed, 1 skipped due to missing D1 files). | C2 (Vision model training) when user says 'continue' |
| 2026-10-17 | Session 5 (C2 resume) | Resumed C2 after interruption. State: all C2 source files existed (classifier.py, vision_agent.py, vision.py, train_classifier.py, smoke_test_classifier.py). Missing: scripts/calibrate.py. ECE test was already correct (conf=acc=0.7 -> ECE=0). Created scripts/calibrate.py (temperature scaling on D2_Val, LBFGS, writes temperature.json + calibration_metrics.json; D14 enforced, I9 enforced, I27 enforced). Smoke test PASSED (RGB/CMYK/RGBA/L inputs, severity_pct=None, low_resolution flag, VisionOutput schema). Full pytest: 118 passed, 1 skipped (C1 baseline was 63p/1s; C2 added 55 tests). No absolute paths in src/scripts/tests. | C3 (RAG stub) when user says 'continue' |
| 2026-10-18 | Session 6 (C2 bug fixes) | Fixed 8 pre-GPU bugs (I28-I35): Bug1=Windows backslash paths (resolve_image_path helper + D2Dataset load-failure guard with 0.5% RuntimeError); Bug2=calibrate.py entry["class"]; Bug3=device-agnostic calibrate_temperature; Bug4=from_checkpoint sets scaler T from temperature.json after load_state_dict; model_best.pth+model.pth saved with calibrated T; round-trip test; Bug5=RandomApply per-image GaussianBlur; Bug6=metrics.json per-class precision/recall/f1/support + CSVs + zero-support classes; Bug7=seeded stratified random sample for --max-train-images/--max-eval-images + class weights from full manifest; Bug8=OOD ECE caveat on D2_Val fitted T. Pytest suite: 139 passed, 1 skipped (0 failed). | Kaggle/GPU training when user says 'continue' |
| 2026-10-18 | Session 7 (C2 calibration fix) | Resolved I36: fixed temperature divergence bug in calibrate_temperature. Implemented robust fit_temperature in src/models/classifier.py (optimizes log_T via LBFGS lr=1.0 with strong_wolfe line search on CPU float64 logits, clamped to [0.05, 20.0], checks NLL < NLL(T=1) else falls back to 1.0). Shared across train_classifier and calibrate.py. Recorded pre/post NLL and ECE on D2_Val in metrics.json under "calibration". Added synthetic tests. Pytest suite: 146 passed, 1 skipped (0 failed). | Kaggle/GPU training when user says 'continue' |
