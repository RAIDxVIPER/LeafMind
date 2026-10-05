# LeafMind Technical Specification — Part 4: Decision Logic
Tags: [PPT]=explicit in slides | [DATA]=verified from datasets | [ENG]=engineering proposal | [GAP]=missing or ambiguous
Date: 2026-10-01

Component types: **ML model** = trained neural net | **LLM** = large language model | **Deterministic** = rule-based / algorithmic

**Overall component type for this Part: Deterministic** (Risk/Reason, Recommendation fallback, Monitoring delta, all error handling). The Recommendation Agent uses an LLM but is grounded to retrieved evidence; its constraints are defined here.

**Corrections to Part 3 applied**: Issues I18, I21, I22 (see plan.md) required changes to `docs/spec/part3_agents.md`; those edits are listed at the end of this document and have been applied.

---

## 4a. Risk/Reason Agent — Full Specification

### 4a-i. Core Design Principle: Risk ≠ Uncertainty

These two concepts are kept strictly separate throughout the pipeline. Conflating them (as Part 3's draft did by using `1.0 - confidence` as the risk base) produces the wrong behaviour: a confident diagnosis of a serious disease would score *low* risk, and an unreadable blurry image would score *high* risk. Both outcomes are wrong and potentially dangerous.

**Definitions used throughout this spec**:

| Concept | Definition | Drives |
|---------|------------|--------|
| **Risk** | How harmful the disease is to the crop if the diagnosis is correct | Risk score, risk level, recommendation urgency |
| **Uncertainty** | How confident the system is that the diagnosis is correct | Escalation flag, caveats, expert referral |

**Invariant**: A confident diagnosis of a genuinely harmful disease must produce a **high** risk score. A low-quality image with an uncertain diagnosis must **escalate**, not produce a high risk score. [ENG]

---

### 4a-ii. Disease Threat Table (Option b — GAP table with fallback)

**Decision** [ENG]: Use Option (b) — a per-disease threat table left as `[GAP]` for plant pathologist input, with the pipeline falling back to Option (a) (a coarse healthy / disease-detected split) while cells are empty.

**Rationale**: Option (a) alone (healthy vs. disease-detected) is data-safe but clinically coarse — it treats a mild blemish the same as late blight. Option (b) allows the risk score to differentiate disease severity once a plant pathologist populates the table, without requiring any data that does not currently exist. The fallback to Option (a) while cells are empty prevents fabricated scores. [ENG][GAP]

**No disease-aggressiveness data exists in the workspace** [DATA]. The threat values below are ALL placeholders. They must be populated by a plant pathologist or derived from collected RAG documents before production deployment.

```
DISEASE THREAT TABLE (part4_threat_table)
Format: disease_label → (threat_tier, notes)
Tier values: HIGH | MEDIUM | LOW | HEALTHY
All non-HEALTHY, non-HEALTHY entries are [GAP — to be filled by plant pathologist]

"healthy" classes (any crop___healthy):  threat_tier = HEALTHY

Non-healthy classes (30 D2 classes — all [GAP]):
  Apple___Apple_scab                     → [GAP]
  Apple___Cedar_apple_rust               → [GAP]
  Corn___Cercospora_leaf_spot            → [GAP]
  Corn___Common_rust                     → [GAP]
  Corn___Northern_Leaf_Blight            → [GAP]
  Grape___Black_rot                      → [GAP]
  Grape___Esca_(Black_Measles)           → [GAP]
  Pepper,_bell___Bacterial_spot          → [GAP]
  Potato___Early_blight                  → [GAP]
  Potato___Late_blight                   → [GAP]
  Squash___Powdery_mildew                → [GAP]
  Tomato___Bacterial_spot                → [GAP]
  Tomato___Early_blight                  → [GAP]
  Tomato___Late_blight                   → [GAP]
  Tomato___Leaf_Mold                     → [GAP]
  Tomato___Septoria_leaf_spot            → [GAP]
  Tomato___Tomato_Yellow_Leaf_Curl_Virus → [GAP]
  Tomato___Tomato_mosaic_virus           → [GAP]
  ... (remaining D2 disease classes)     → [GAP]

Fallback (while any cell is empty): use Option (a):
  If top_label contains "healthy" anywhere → threat_tier = HEALTHY
  Otherwise                                → threat_tier = DISEASE_DETECTED (treated as MEDIUM)
```

**When cells are populated**: the threat tier drives the `threat_score` term in the risk formula. Threat tier mapping [ENG assumption — to be reviewed]:
```
HEALTHY         → threat_score = 0.0
LOW             → threat_score = 0.25   [ENG assumption]
MEDIUM          → threat_score = 0.50   [ENG assumption]
HIGH            → threat_score = 0.85   [ENG assumption]
DISEASE_DETECTED (fallback) → threat_score = 0.50   [ENG assumption]
```
All tier-to-score mappings are engineering assumptions with no data derivation; they must be reviewed by a plant pathologist alongside the threat table. [ENG][GAP]

---

### 4a-iii. Risk Score Formula

**Component type**: Deterministic [ENG]

The risk score `R` is a weighted combination of three independent signals:

```
R = w_threat × threat_score(top_label)
  + w_weather × weather_score(disease_risk_flags)
  + w_history × history_score(case_history)

where:
  threat_score  = from the disease threat table (4a-ii); fallback = 0.50 for DISEASE_DETECTED
  weather_score = fraction of applicable disease-specific weather flags that are triggered (see 4a-iv)
  history_score = follow-up signal (see 4a-v); = 0.0 on first visit

  w_threat  = 0.70   [ENG assumption — dominant term; disease identity should drive risk]
  w_weather = 0.20   [ENG assumption — contextual; low because thresholds are unverified (I14)]
  w_history = 0.10   [ENG assumption — small; strengthens or weakens risk on follow-ups]

  Constraint: w_threat + w_weather + w_history = 1.0
  R is clamped to [0.0, 1.0]

Risk level thresholds:
  R < 0.25   → risk_level = "low"
  0.25 ≤ R < 0.60 → risk_level = "moderate"
  R ≥ 0.60   → risk_level = "severe"
  [ENG assumption — all thresholds to be calibrated once threat table is populated and field data is available]
```

**Weight justification** [ENG]:
- `w_threat = 0.70`: Disease identity is the primary determinant of risk. A confirmed late blight is dangerous regardless of weather. This term must dominate.
- `w_weather = 0.20`: Weather flags add meaningful contextual urgency (wet weather accelerates spread) but W01–W07 thresholds are unverified (I14); a low weight prevents unverified flags from overriding the disease signal.
- `w_history = 0.10`: Repeated detection after treatment is a weak additional signal; kept small because the monitoring signal itself has uncertainty (see 4c).

**How weights would be chosen or learned** [ENG][GAP]:
1. Once the threat table is populated and a set of expert-labelled cases is available (disease + weather + observed outcome), the weights can be estimated by fitting a logistic regression against the expert labels. This is a supervised calibration step, not part of Phase 1.
2. Until then, the values above are engineering defaults chosen to reflect the ordering `w_threat ≫ w_weather > w_history` as a reasonable prior.

**Healthy class behaviour**: If `threat_score = 0.0` (HEALTHY), then `R ≤ w_weather × weather_score + w_history × history_score ≤ 0.30`, which maps to `low` or at most the bottom of `moderate`. Weather flags do not elevate a healthy diagnosis to `severe`. [ENG]

---

### 4a-iv. Weather Score Computation

**Component type**: Deterministic [ENG]

```
weather_score(disease_risk_flags):
  1. Select only flags WHERE triggered=True
  2. Select only flags applicable to the detected disease_class
     (i.e., condition_name references the diagnosed disease or a related pathogen)
  3. applicable_flags = count of disease-applicable rules in W01–W07
     (W08 omitted — [GAP] per plan.md)
  4. triggered_applicable = count of triggered flags among applicable ones
  5. If applicable_flags == 0: weather_score = 0.0  (no rule applies to this disease)
  6. Else: weather_score = triggered_applicable / applicable_flags  ∈ [0.0, 1.0]
  7. Tag all weather contributions with "[UNVERIFIED — I14]" in the factors list
```

**CONSTRAINT** [ENG]: Weather flags alone can never raise risk_level from "low" to "severe". Maximum weather contribution = `w_weather × 1.0 = 0.20`. This is below the "severe" threshold (0.60) even if all applicable flags fire. Only a high threat_score (high-tier disease) can produce "severe" risk.

---

### 4a-v. History Score Computation (Follow-Up Visits)

**Component type**: Deterministic [ENG]

```
history_score(case_history):
  On first visit (no prior monitoring output): history_score = 0.0

  On follow-up visit N:
    Load MonitoringOutput from visit N-1
    If monitoring_output.trend == "persistent_disease" (disease still detected at visit ≥ 2):
      history_score = 0.10   [ENG assumption — small boost; to be tuned]
    Else if monitoring_output.trend == "label_changed_to_healthy":
      history_score = -0.05  [ENG assumption — slight reduction; uncertain]
    Else:
      history_score = 0.0

NOTE: history_score intentionally has a small range. The most reliable signal is still
the current visit's vision output and disease identity. [ENG]
```

---

### 4a-vi. Uncertainty — Separate from Risk

Uncertainty is computed independently and drives the `escalate_to_human` flag. It does NOT affect `risk_score`.

```
uncertainty_signals (computed from VisionOutput + RAGOutput + image quality):
  U1: vision.confidence < 0.60                        [ENG assumption — to be calibrated]
  U2: (top_k_probs[0].prob - top_k_probs[1].prob) < 0.15   [ENG assumption — to be calibrated]
  U3: vision.image_quality_flags.low_resolution == True
  U4: vision.image_quality_flags.blurry == True
  U5: vision.ood_flag == True
  U6: rag.knowledge_gap_flag == True  (RAG has insufficient evidence)

escalate_to_human = ANY(U1, U2, U3, U4, U5, U6)

uncertainty_level (for display in FarmerResponse):
  0 signals → "high_confidence"
  1 signal  → "moderate_confidence"
  2+ signals → "low_confidence"
  [ENG assumption — labels only; not a probability]
```

**Why uncertainty is kept separate** [ENG]: A blurry image (U3 or U4) raises uncertainty but must not raise the risk score. If the classifier happens to output "Tomato___Late_blight" with low confidence on a blurry image, the correct response is to escalate (ask for a clearer photo), not to score the case as "severe" risk and alarm the farmer.

---

### 4a-vii. Healthy Class Handling

If `top_label` contains "healthy" (in any class name from either dataset):
- `threat_score = 0.0` → `R ≤ 0.30` → `risk_level ∈ {low, moderate_low}` [ENG]
- If confidence is high (U1–U5 all false): `escalate_to_human = False`; return "no disease detected" recommendation
- If confidence is low (any U true): escalate — image may not be representative
- `progression_note = "No disease detected in this image. Continue routine monitoring."` [ENG heuristic]

---

### 4a-viii. Progression Estimate

**This is a text heuristic only. No data supports a quantitative timeline.** [ENG][GAP]

```python
def progression_note(top_label: str, risk_level: str, weather_triggered: bool) -> str:
    if "healthy" in top_label.lower():
        return ("No disease detected in this image. "
                "Continue routine monitoring. [ENG heuristic]")
    elif risk_level == "severe" and weather_triggered:
        return ("Disease identified as potentially high-risk. Current weather conditions "
                "may favour spread. Prompt action is advisable. Verify with a follow-up "
                "image in 3–5 days or consult a local agronomist. "
                "[ENG heuristic — not a quantitative prediction; no temporal data in training set]")
    elif risk_level == "severe":
        return ("Disease identified as potentially high-risk. "
                "Prompt treatment is advisable. Verify with a follow-up image. "
                "[ENG heuristic — not a quantitative prediction]")
    elif risk_level == "moderate":
        return ("Disease detected at moderate risk level. "
                "Monitor closely; consider treatment if symptoms worsen. "
                "[ENG heuristic — not a quantitative prediction]")
    else:  # low
        return ("Low risk level assessed. Continue monitoring. "
                "[ENG heuristic — not a quantitative prediction]")
```

---

### 4a-ix. Worked Examples

**All values below are ILLUSTRATIVE, not measured. They are constructed to verify that the formula behaves correctly in each scenario, not to represent real model outputs.**

---

#### Case 1: Confident serious disease with favourable weather [ILLUSTRATIVE, not measured]

**Inputs**:
```
top_label = "Tomato___Late_blight"
vision.confidence = 0.88
disease_risk_flags: W01 triggered (High humidity + cool nights) [UNVERIFIED — I14]
threat_table: "Tomato___Late_blight" → HIGH → threat_score = 0.85  [GAP — placeholder]
visit = 1 (no history)
```

**Computation**:
```
threat_score   = 0.85
weather_score  = 1/1 = 1.0   (1 applicable flag, W01, triggered)
history_score  = 0.0          (first visit)

R = 0.70 × 0.85 + 0.20 × 1.0 + 0.10 × 0.0
  = 0.595 + 0.200 + 0.000
  = 0.795 → clamped to 0.795 → risk_level = "severe"

Uncertainty: U1=False (0.88≥0.60), U2=True if top2=0.08 → (0.88-0.08=0.80 ≥ 0.15 → U2=False)
             All image quality flags False; ood_flag=False; knowledge_gap_flag=False
escalate_to_human = False
```

**Output (illustrative)**:
```json
{
  "risk_level": "severe",
  "risk_score": 0.795,
  "escalate_to_human": false,
  "uncertainty_level": "high_confidence",
  "factors": [
    {"source": "threat_table", "label": "Tomato___Late_blight", "threat_tier": "HIGH [GAP]", "threat_score": 0.85, "contribution": 0.595, "tag": "[ENG assumption — plant pathologist review required]"},
    {"source": "weather_W01", "value": "triggered", "weather_score": 1.0, "contribution": 0.200, "tag": "[ENG assumption — threshold UNVERIFIED I14]"},
    {"source": "history", "value": "first_visit", "history_score": 0.0, "contribution": 0.000, "tag": "[ENG assumption]"}
  ],
  "progression_note": "Disease identified as potentially high-risk. Current weather conditions may favour spread. Prompt action is advisable. Verify with a follow-up image in 3–5 days or consult a local agronomist. [ENG heuristic — not a quantitative prediction]"
}
```

**Verification**: A confident diagnosis of a serious disease (Late blight at 88% confidence) correctly produces "severe" risk. The system does NOT score this low because confidence is high. ✓

> **IMPORTANT LIMITATION (I26)**: Case 1 above uses a placeholder `HIGH` tier for Late blight. That tier is `[GAP]`; the threat table is currently **empty**. Case 1b below shows what the pipeline actually does today.

---

#### Case 1b: Confident Late blight — EMPTY threat table (fallback) [ILLUSTRATIVE, not measured]

This is the case the system will produce in Phase 1, before a plant pathologist fills the threat table.

**Inputs**:
```
top_label = "Tomato___Late_blight"
vision.confidence = 0.88
disease_risk_flags: W01 triggered [UNVERIFIED — I14]
threat_table: "Tomato___Late_blight" → [GAP] → fallback = DISEASE_DETECTED → threat_score = 0.50
visit = 1 (no history)
```

**Computation**:
```
threat_score   = 0.50   (DISEASE_DETECTED fallback — table not populated)
weather_score  = 1/1 = 1.0   (W01 triggered, 1 applicable flag)
history_score  = 0.0          (first visit)

R = 0.70 × 0.50 + 0.20 × 1.0 + 0.10 × 0.0
  = 0.350 + 0.200 + 0.000
  = 0.550 → risk_level = "moderate"

Uncertainty: U1=False (0.88≥0.60); all quality flags False; ood_flag=False; knowledge_gap_flag=False
escalate_to_human = False
```

**Output (illustrative)**:
```json
{
  "risk_level": "moderate",
  "risk_score": 0.55,
  "escalate_to_human": false,
  "uncertainty_level": "high_confidence",
  "factors": [
    {
      "source": "threat_table",
      "label": "Tomato___Late_blight",
      "threat_tier": "DISEASE_DETECTED [GAP fallback — threat table not populated]",
      "threat_score": 0.50,
      "contribution": 0.350,
      "tag": "[ENG assumption — plant pathologist review required (I19)]"
    },
    {
      "source": "weather_W01",
      "value": "triggered",
      "weather_score": 1.0,
      "contribution": 0.200,
      "tag": "[ENG assumption — threshold UNVERIFIED I14]"
    },
    {
      "source": "history",
      "value": "first_visit",
      "history_score": 0.0,
      "contribution": 0.000,
      "tag": "[ENG assumption]"
    }
  ],
  "progression_note": "Disease detected at moderate risk level. Monitor closely; consider treatment if symptoms worsen. [ENG heuristic — not a quantitative prediction]"
}
```

**Finding**: With the empty threat table, a confident Late blight + one triggered weather flag scores **R = 0.55 → "moderate"**, not "severe". This is a direct consequence of the DISEASE_DETECTED fallback (0.50) capping the threat signal. [ENG][DATA]

**Plain statement of the constraint** [ENG][GAP]: **Until a plant pathologist populates the threat table, no first-visit diagnosis can reach "severe" unless the weather score is at its maximum AND history_score is also elevated.** Maximum first-visit R with the fallback = 0.70×0.50 + 0.20×1.0 + 0.10×0.0 = 0.55 → "moderate". This means the system will under-warn farmers about genuinely serious diseases until the table is filled. This is a known, deliberate limitation documented as I26 and I19.

**Implication for deployment**: The risk score with an empty threat table is an engineering placeholder. It must not be presented to farmers as a calibrated risk assessment. The Recommendation Agent's caveats must include a note that the risk level is based on a default tier. A plant pathologist completing the threat table is a **Phase B prerequisite before production deployment**. [ENG][GAP]

---

#### Case 2: Confident healthy leaf [ILLUSTRATIVE, not measured]

**Inputs**:
```
top_label = "Tomato___healthy"
vision.confidence = 0.91
disease_risk_flags: W01 triggered (but healthy leaf, so no applicable disease flags)
threat_table: "healthy" → threat_score = 0.0
visit = 1
```

**Computation**:
```
threat_score   = 0.0
weather_score  = 0.0   (W01 applies to Late blight, not to a healthy diagnosis;
                         no disease-applicable flags for "healthy" label)
history_score  = 0.0

R = 0.70 × 0.0 + 0.20 × 0.0 + 0.10 × 0.0 = 0.0 → risk_level = "low"

Uncertainty: all signals False (confidence 0.91, good image, not OOD)
escalate_to_human = False
```

**Output (illustrative)**:
```json
{
  "risk_level": "low",
  "risk_score": 0.0,
  "escalate_to_human": false,
  "uncertainty_level": "high_confidence",
  "factors": [
    {"source": "threat_table", "label": "Tomato___healthy", "threat_tier": "HEALTHY", "threat_score": 0.0, "contribution": 0.0, "tag": "[ENG assumption]"},
    {"source": "weather", "value": "no disease-applicable flags", "contribution": 0.0, "tag": "[ENG assumption]"}
  ],
  "progression_note": "No disease detected in this image. Continue routine monitoring. [ENG heuristic]"
}
```

**Verification**: A confident healthy diagnosis correctly scores "low" risk. Weather flags (which fired) do not contaminate a healthy outcome. ✓

---

#### Case 3: Low-confidence image (poor quality) [ILLUSTRATIVE, not measured]

**Inputs**:
```
top_label = "Tomato___Early_blight"   (classifier's best guess on a blurry image)
vision.confidence = 0.41
vision.image_quality_flags.blurry = True
ood_flag = True   (confidence < 0.40 — borderline; say ood_flag = True due to blurry)
disease_risk_flags: none triggered
threat_table: fallback (threat table not yet populated) → DISEASE_DETECTED → threat_score = 0.50
```

**Computation**:
```
threat_score   = 0.50   (DISEASE_DETECTED fallback)
weather_score  = 0.0    (no flags triggered)
history_score  = 0.0

R = 0.70 × 0.50 + 0.20 × 0.0 + 0.10 × 0.0 = 0.35 → risk_level = "moderate"

Uncertainty:
  U1 = True (confidence 0.41 < 0.60)
  U4 = True (blurry = True)
  U5 = True (ood_flag = True)
escalate_to_human = True
```

**Output (illustrative)**:
```json
{
  "risk_level": "moderate",
  "risk_score": 0.35,
  "escalate_to_human": true,
  "uncertainty_level": "low_confidence",
  "factors": [
    {"source": "threat_table", "label": "Tomato___Early_blight", "threat_tier": "DISEASE_DETECTED [GAP fallback]", "threat_score": 0.50, "contribution": 0.35, "tag": "[ENG assumption — threat table not populated]"},
    {"source": "weather", "value": "no flags triggered", "contribution": 0.0, "tag": "[ENG assumption]"}
  ],
  "progression_note": "Disease detected at moderate risk level. Monitor closely; consider treatment if symptoms worsen. [ENG heuristic — not a quantitative prediction]",
  "uncertainty_signals_triggered": ["U1_low_confidence", "U4_blurry_image", "U5_ood_flag"]
}
```

**Verification**: The low-confidence blurry-image case escalates to a human (correct) and scores "moderate" risk from the disease-detected fallback, NOT "high" risk from the low confidence. The system does not treat "I can't tell what this is" as "this is definitely dangerous." ✓

---

## 4b. Recommendation Agent — Constraints and Logic

### 4b-i. Component type and role

**Component type**: LLM (strictly grounded to RiskOutput + RAGOutput) [ENG]

The Recommendation Agent **never diagnoses**. It only converts an already-computed `RiskOutput` (disease identity, risk level, risk score, factors) and `RAGOutput` (treatment evidence) into farmer-facing structured advice. It does not re-interpret the image or modify the risk assessment. [ENG]

---

### 4b-ii. Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `immediate_action` | `str \| null` | Single most urgent step; null if risk_level=low or RAG unavailable |
| `advisory` | `List[str]` | Treatment or monitoring steps; MUST be traceable to RAGOutput.citations |
| `timing` | `str \| null` | Application timing from RAG; null if not in retrieved documents |
| `monitoring_instruction` | `str` | When to take a follow-up image; ALWAYS populated |
| `urgency` | `str` | `immediate / within_24h / within_48h / within_week / monitor / consult_agronomist` |
| `confidence_justification` | `str` | Plain-language explanation of confidence level for farmer |
| `evidence_references` | `List[str]` | = RAGOutput.citations; empty → "consult agronomist" path |
| `caveats` | `List[str]` | Any limitations; MUST include escalation caveat if escalate_to_human=True |
| `escalate_to_human` | `bool` | Passed through from RiskOutput; if True, recommendation is advisory only |
| `rag_grounded` | `bool` | True only when evidence_references is non-empty |

---

### 4b-iii. Urgency Mapping [ENG assumption]

```
risk_level = "severe"  AND escalate_to_human = False: urgency = "within_24h"
risk_level = "severe"  AND escalate_to_human = True:  urgency = "consult_agronomist"
risk_level = "moderate" AND escalate_to_human = False: urgency = "within_48h"
risk_level = "moderate" AND escalate_to_human = True:  urgency = "consult_agronomist"
risk_level = "low"     AND escalate_to_human = False: urgency = "monitor"
risk_level = "low"     AND escalate_to_human = True:  urgency = "consult_agronomist"
rag.rag_available = False (any risk level):            urgency = "consult_agronomist"
```

All urgency thresholds are engineering assumptions. [ENG]

---

### 4b-iv. Grounding Rules (LLM prompt constraints)

Every treatment statement in `advisory` must be traceable to a chunk in `RAGOutput.raw_chunks` or `RAGOutput.treatment_options` with a matching citation in `RAGOutput.citations`.

LLM system prompt must include (verbatim or equivalent):

```
"You are a plant disease advisory assistant. Your task is to convert a structured risk assessment
and retrieved treatment evidence into farmer-actionable advice.

RULES:
1. Do NOT invent any treatment, dosage, active ingredient, cost, or timing not present in the
   provided evidence sections. If evidence is absent, say 'consult a local agronomist or
   extension officer'.
2. Every item in 'advisory' must be directly traceable to the retrieved documents.
3. If escalate_to_human is True, begin every advisory item with 'ADVISORY ONLY — consult an
   agronomist to confirm:'.
4. If rag_available is False or knowledge_gap_flag is True, set advisory = [] and set
   immediate_action = 'Consult a local agronomist or agricultural extension officer'.
5. Do NOT claim the disease is 'confirmed'. The diagnosis is a classifier output with
   associated uncertainty; use 'detected' or 'identified' language only.
6. Output JSON only. Do not add explanatory prose outside the JSON fields."
```

---

### 4b-v. Low Confidence / No RAG Evidence Path

When `escalate_to_human=True` OR `rag.rag_available=False` OR `rag.knowledge_gap_flag=True`:

```json
{
  "immediate_action": "Consult a local agronomist or agricultural extension officer",
  "advisory": [],
  "timing": null,
  "monitoring_instruction": "Take a clear, close-up photo of the affected leaf in good natural light and submit again, or bring a sample to your local extension office.",
  "urgency": "consult_agronomist",
  "confidence_justification": "<explanation of why confidence is low or evidence is unavailable>",
  "evidence_references": [],
  "caveats": ["Automated treatment advice is not available for this case. Expert verification is required."],
  "escalate_to_human": true,
  "rag_grounded": false
}
```

---

### 4b-vi. Monitoring Instruction (always populated)

`monitoring_instruction` is always populated regardless of RAG availability. Default:

```
"Take a follow-up photo of the same leaf or plant area in 3–5 days and submit using your case ID
to track changes. [ENG assumption — 3–5 day interval is a heuristic with no data support]"
```

If `risk_level = "severe"`: interval shortened to 1–2 days [ENG assumption]. [GAP — interval should be validated by agronomist]

---

## 4c. Monitoring Agent — Redefined Logic

### 4c-i. Component type

**Component type**: ML model (re-runs Vision Agent) + Deterministic (all delta and plausibility logic) [ENG]

---

### 4c-ii. Core Redefinition (supersedes Part 3 draft)

The Monitoring Agent's job is to characterise **change between visits**, not to claim efficacy. "Disease still detected after N days" is an **escalation trigger**, not a recovery measure. The agent makes no claim about whether treatment worked.

**All trend labels and escalation counts (N) are engineering assumptions to be tuned.** [ENG]

---

### 4c-iii. Pre-Analysis Plausibility Checks

Before computing any delta, the agent runs these checks in order:

| Check | What can be checked with available models | What cannot be checked | Action on fail |
|-------|------------------------------------------|----------------------|----------------|
| **Same image resubmission** | MD5 hash comparison with baseline image | — | Set `same_image_flag=True`; return early; do not compute delta |
| **Image quality** | Laplacian variance (blurry flag from Vision Agent) | — | Set `poor_quality_flag=True`; escalate; note in output |
| **Crop plausibility** | The classifier top label's crop prefix (e.g. "Tomato___") should match the baseline crop prefix | Cannot verify actual physical plant identity; classifier may be wrong | Set `crop_mismatch_flag=True` if prefixes differ; warn, do not abort |
| **Missing baseline** | Check if visit_1 VisionOutput exists in CaseState | — | Return `MonitoringError(code="no_baseline")` |
| **Too soon (same day)** | Check if `followup_submitted_at.date() == initial_request.submitted_at.date()` | — | Set `too_soon_flag=True`; proceed but warn |

**What cannot be checked** [GAP]:
- Whether the follow-up image shows the same physical leaf or plant (no plant identity model exists)
- Whether the disease has progressed in real terms (no severity data exists)
- Treatment adherence (not captured by any current input)

---

### 4c-iv. Delta Signal Computation (replaces Part 3 trend labels)

**Replacing Part 3's `improving_confidence` / `declining_confidence` labels** (I21 resolution):

Those labels conflate two independent signals: (a) whether the disease is still present and (b) how confident the classifier is in the current image. A "declining confidence" on a blurry follow-up image says nothing about the disease; "improving confidence" on a healthy label says nothing about treatment efficacy. The new design separates these:

```
From new_vision (follow-up VisionOutput) and baseline_vision (visit_1 VisionOutput):

delta.label_at_followup        = new_vision.top_label
delta.label_at_baseline        = baseline_vision.top_label
delta.label_same               = (new_vision.top_label == baseline_vision.top_label)
delta.disease_still_detected   = ("healthy" not in new_vision.top_label.lower())
delta.disease_was_present      = ("healthy" not in baseline_vision.top_label.lower())
delta.new_label_is_healthy     = ("healthy" in new_vision.top_label.lower())
delta.confidence_at_baseline   = baseline_vision.confidence
delta.confidence_at_followup   = new_vision.confidence
delta.confidence_raw_delta     = new_vision.confidence - baseline_vision.confidence
  (NOTE: this is a raw model output change; it is NOT evidence of improvement or worsening)

severity_delta_pct             = null   [DATA — always; no severity data exists]
severity_note                  = "Numeric severity delta is not computable..."
```

---

### 4c-v. Disease Status Assessment (Deterministic) [ENG]

```
disease_status_at_followup:
  if NOT delta.disease_was_present:
    status = "not_applicable_healthy_baseline"
    # Healthy baseline: monitor for new onset only
  elif delta.new_label_is_healthy:
    status = "no_disease_detected_at_followup"
    # Cautious: could be camera angle, lighting, or genuine recovery
    # Do NOT claim "treatment worked" — this is an observation only
  elif delta.disease_still_detected AND delta.label_same:
    status = "same_disease_still_detected"
  elif delta.disease_still_detected AND NOT delta.label_same:
    status = "different_disease_detected_at_followup"
    # Classifier identified a different disease class — could be secondary infection,
    # misclassification on either visit, or different viewing angle
  else:
    status = "unknown"

NOTE: status = "no_disease_detected_at_followup" does NOT imply treatment was effective.
It is an observation that the model did not detect disease in this image.
The farmer should be advised to submit another follow-up before concluding recovery. [ENG]
```

---

### 4c-vi. Escalation Triggers from Monitoring [ENG assumption — all N values to be tuned]

| Trigger | Condition | N / Count | Escalation reason |
|---------|-----------|-----------|-------------------|
| Disease persists | `same_disease_still_detected` on visit ≥ 2 | visit ≥ 2 [ENG assumption — first follow-up] | Possible treatment failure |
| Disease persists — extended | `same_disease_still_detected` on visit ≥ 3 | 3 visits [ENG assumption] | Escalate urgently |
| Different disease detected | `different_disease_detected_at_followup` | Any | Secondary infection or misclassification; agronomist needed |
| Poor image quality | `poor_quality_flag=True` | Any | Cannot compute reliable delta; resubmit |
| Crop mismatch | `crop_mismatch_flag=True` | Any | Possible wrong plant; agronomist needed |
| Vision escalation | `new_vision.escalate_to_human=True` | Any | Low confidence on follow-up |

All N values above are engineering assumptions. They should be tuned based on disease-specific progression knowledge from agronomists or literature once the RAG knowledge base is built. [ENG][GAP]

---

### 4c-vii. MonitoringOutput Schema Amendments (Part 3 corrections — see end of document)

The `trend` field with values `improving_confidence / declining_confidence / label_changed / stable` is replaced by the `disease_status` field and the `delta` sub-object above. This is a schema correction to Part 3 (see Section "Part 3 Corrections" at end of this document).

---

## 4d. Human-in-the-Loop — Triggers and Payload

### 4d-i. Component type: Deterministic [ENG]

The escalation decision is a deterministic rule evaluation. No LLM is involved in deciding whether to escalate. [ENG]

---

### 4d-ii. Escalation Triggers (complete list)

| Trigger ID | Condition | Source | Priority |
|------------|-----------|--------|----------|
| T1 | `vision.confidence < 0.60` | VisionOutput.uncertainty U1 | medium |
| T2 | `top_k_probs[0].prob - top_k_probs[1].prob < 0.15` | VisionOutput.uncertainty U2 | medium |
| T3 | `vision.image_quality_flags.low_resolution = True` | VisionOutput.uncertainty U3 | medium |
| T4 | `vision.image_quality_flags.blurry = True` | VisionOutput.uncertainty U4 | medium |
| T5 | `vision.ood_flag = True` | VisionOutput.uncertainty U5 | high |
| T6 | `rag.knowledge_gap_flag = True` | RAGOutput | medium |
| T7 | `risk.risk_level = "severe" AND escalate_to_human = False` | RiskOutput | high [auto-include] |
| T8 | `monitoring.disease_status = "different_disease_detected"` | MonitoringOutput | high |
| T9 | `monitoring.disease_status = "same_disease_still_detected" AND visit ≥ 2` | MonitoringOutput | high |
| T10 | `monitoring.poor_quality_flag = True` | MonitoringOutput | medium |
| T11 | `monitoring.crop_mismatch_flag = True` | MonitoringOutput | high |

**Note on T7**: "severe" cases with high confidence (escalate_to_human=False) still generate an escalation notification to the agronomist queue as a monitoring record, but the farmer receives the automated recommendation. This is a design choice [ENG]; agronomists can review severe-risk high-confidence cases asynchronously.

All trigger thresholds (0.60, 0.15, visit count) are engineering assumptions. [ENG]

---

### 4d-iii. Escalation Priority Mapping [ENG assumption]

```
urgent:  T5 (OOD) OR T8 (different disease) OR T9 (extended persistence)
high:    T7 (severe risk) OR T11 (crop mismatch)
medium:  T1, T2, T3, T4, T6, T10
low:     (reserved for future use)

Final priority = highest priority among all triggered T-codes
```

---

### 4d-iv. HumanEscalationPayload (exact JSON sent to agronomist queue)

This is the exact schema. All fields are mandatory unless explicitly marked optional.

```json
{
  "case_id": "uuid4-string",
  "farmer_id": "string",
  "escalated_at": "ISO-8601 datetime",
  "priority": "urgent | high | medium | low",
  "escalation_reasons": [
    {
      "trigger_id": "T1",
      "description": "Classifier confidence (0.41) below threshold (0.60) [ENG assumption]",
      "value": 0.41,
      "threshold": 0.60
    }
  ],
  "diagnosis_summary": {
    "top_label": "Tomato___Early_blight",
    "confidence": 0.41,
    "top_k_probs": [{"label": "...", "prob": 0.41}, "..."],
    "ood_flag": true,
    "image_quality_flags": {"low_resolution": false, "blurry": true, "non_leaf_suspected": false}
  },
  "risk_summary": {
    "risk_level": "moderate",
    "risk_score": 0.35,
    "uncertainty_level": "low_confidence",
    "factors": [{"source": "...", "contribution": 0.35, "tag": "[ENG assumption]"}]
  },
  "weather_summary": {
    "weather_available": true,
    "triggered_flags": [{"condition_name": "W03", "triggered": true, "threshold_source": "Ritchie & Dittapongpitch 1991 [UNVERIFIED — I14]"}]
  },
  "rag_summary": {
    "rag_available": true,
    "knowledge_gap_flag": false,
    "retrieval_confidence": 0.55,
    "treatment_options_retrieved": 2,
    "citations": ["fao_tomato_...pdf"]
  },
  "draft_recommendation": {
    "immediate_action": "ADVISORY ONLY — consult an agronomist to confirm: Apply mancozeb...",
    "advisory": ["ADVISORY ONLY — ..."],
    "urgency": "consult_agronomist",
    "rag_grounded": true
  },
  "monitoring_summary": null,
  "image_available": true,
  "image_reference": "cases/uuid4-string/visit_1_image.jpg",
  "visit_number": 1,
  "crop_type": "Tomato"
}
```

**Note**: `image_bytes` is NOT included directly in the payload; it is referenced by path in the case storage. This avoids transmitting large binary data through the escalation queue. The agronomist system retrieves the image from case storage using `image_reference`. [ENG]

---

## 4e. Error Handling Table

**Component type**: Deterministic [ENG]

All error handling is deterministic rule-based logic. No LLM is involved in error detection or recovery. [ENG]

| Failure | Detection | System Fallback | Farmer-Visible Behaviour |
|---------|-----------|-----------------|--------------------------|
| **Vision model unavailable** (model file missing, corrupt, OOM) | VisionError raised by model loader | Return `StructuredError(code="vision_unavailable")`; abort pipeline | "Unable to analyse your image at this time. Please try again in a few minutes or visit your local extension office." |
| **Invalid image** (not decodable, wrong mime type, 0 bytes) | PIL/Pillow decode failure or size check | Return `StructuredError(code="invalid_image")`; abort pipeline | "The image could not be read. Please submit a JPEG or PNG photo of the affected leaf." |
| **Low-quality image** (blurry, low resolution) | Laplacian variance < threshold; resolution check in Vision Agent | Set quality flags; set escalate_to_human=True; proceed with reduced confidence | "The image appears to be blurry or low resolution. The assessment may be unreliable. Please try taking a clearer photo in good natural light." (shown as caveat alongside the result) |
| **Unknown disease / OOD image** (ood_flag=True) | max softmax < 0.40 [ENG assumption] | Set ood_flag=True; escalate_to_human=True; proceed with top label | "The system was unable to confidently identify a disease in this image. A local agronomist will review your case." |
| **Vision model timeout** (inference > N seconds [ENG assumption]) | asyncio timeout wrapper | Retry once; on second timeout raise VisionError(code="timeout") | Same as vision_unavailable |
| **Weather API unavailable** (both Open-Meteo and OWM fail) | HTTPError / ConnectionError on both; timeout > 5s | Set `weather_available=False`; proceed without weather; set `weather_unavailable=True` in risk factors | "Weather data could not be retrieved. Risk assessment proceeds without weather context." (shown as caveat) |
| **Weather API partial response** (missing fields) | Pydantic validation of API response fails on required fields | Set `weather_partial=True`; use available fields only | No user-visible message; caveat in risk factors |
| **RAG failure — vector DB unavailable** | ChromaDB connection error | Set `rag_available=False`; proceed; recommendation = "consult agronomist" | "Treatment guidance could not be retrieved. Please consult a local agronomist or agricultural extension officer." |
| **RAG failure — empty results** (no chunks above threshold) | top-1 chunk score < 0.20 | Set `knowledge_gap_flag=True`; set `rag_available=True` but no synthesis | Same as above |
| **LLM API unavailable** (Recommendation Agent) | API connection error / timeout | Return deterministic fallback: if risk=severe → "Consult agronomist immediately"; else → "Consult local advisory service" | "Automated recommendation not available. Please consult a local agronomist." + risk level shown |
| **LLM returns invalid JSON** | JSON parse failure | Retry once; if still invalid, use deterministic fallback (see above) | Same as LLM unavailable |
| **LLM response missing citations** | `citations == []` AND `rag_available=True` | Reject; retry once; on second failure use raw_chunks as fallback display | "Treatment guidance is limited; please consult an agronomist for full advice." |
| **Case not found** (follow-up path; case_id not in persistence) | File/DB lookup returns None | Return `StructuredError(code="case_not_found")` | "Case ID not recognised. Please submit a new request or check your case ID." |
| **Conflicting outputs** (Vision says healthy; Risk says severe due to non-empty threat table; logically inconsistent) | Risk engine checks: if threat_score=0.0 AND risk_score>0.50 → internal inconsistency | Log the inconsistency; override risk_level to "low" for healthy labels; flag in `factors` | No change to farmer output; caveats list notes "Internal consistency check applied." |
| **Pipeline timeout** (full Orchestrator timeout) | Global asyncio timeout wrapper [ENG assumption — timeout value to be set] | Kill pending branches; return partial result if any branch completed; else return StructuredError | "The analysis took longer than expected. Partial results may be shown. Please retry if the result seems incomplete." |
| **Network failure** (farmer device loses connection mid-upload) | HTTP connection drop detected by API layer | Depends on API layer (FastAPI / HTTP server); return 503 with Retry-After header | Client shows "Upload failed — please check your connection and try again." |

---

## Part 3 Corrections Required (applied to docs/spec/part3_agents.md)

The following changes are required for Part 3 to remain consistent with Part 4. They have been applied to `docs/spec/part3_agents.md`.

### Change C1 — Risk/Reason Agent processing logic (3a-5) [resolves I18, I22]

**Problem**: Part 3 used `base = 1.0 - vision.confidence` as the base risk score, which causes a confident disease diagnosis to score LOW risk. This directly contradicts the Risk ≠ Uncertainty principle.

**Old text** (Part 3, section 3a-5, steps 2–5):
```
2. Base score from vision confidence:
   base = 1.0 - vision.confidence   (low confidence → higher base risk) [ENG assumption]
3. Disease severity modifier: since severity_pct = null, use detection presence signal:
   if vision.leaf_detected: presence_boost = 0.05 [ENG assumption — small; must be calibrated]
   else: presence_boost = 0.0
4. Weather modifier: for each triggered flag in weather.disease_risk_flags: add a weight
   (exact weights to be specified in Part 4; treat as ENG assumptions until calibrated)
5. RAG modifier: if rag.knowledge_gap_flag: add uncertainty penalty [ENG assumption]
```

**New text** (now consistent with Part 4, section 4a-iii):
```
2. Threat score: look up top_label in disease threat table (Part 4, 4a-ii)
   threat_score = threat_table[top_label]   (fallback = 0.50 if table not populated)
3. Weather score: apply 4a-iv formula (disease-applicable flags only, w_weather=0.20)
4. History score: apply 4a-v formula (0.0 on first visit)
5. R = 0.70 × threat_score + 0.20 × weather_score + 0.10 × history_score
   Clamp R to [0.0, 1.0]
6. Escalation (SEPARATE from R): compute uncertainty signals U1–U6 per 4a-vi
   escalate_to_human = ANY(U1, U2, U3, U4, U5, U6)
```

### Change C2 — MonitoringOutput schema and trend field (3b + 3a-7) [resolves I21]

**Problem**: `trend` field with values `improving_confidence / declining_confidence / label_changed / stable` conflates confidence change with disease status.

**Old schema field**:
```python
trend: str = Field(..., pattern=r"^(improving_confidence|declining_confidence|label_changed|stable)$")
```

**New schema fields** (replacing `trend`):
```python
disease_status: str = Field(
    ...,
    pattern=r"^(same_disease_still_detected|no_disease_detected_at_followup|"
            r"different_disease_detected_at_followup|not_applicable_healthy_baseline|unknown)$"
)
confidence_raw_delta: float  # new_confidence - baseline_confidence; NOT an efficacy signal
poor_quality_flag: bool = False
crop_mismatch_flag: bool = False
too_soon_flag: bool = False
```

**Old processing logic steps 3d–4** in 3a-7 are replaced by Part 4 sections 4c-iv through 4c-vi.

### Change C3 — RiskOutput schema [resolves I22]

**Problem**: `risk_score` was computed from confidence, meaning uncertainty was folded into the risk number.

Add new fields to `RiskOutput`:
```python
uncertainty_level: str = Field(
    ...,
    pattern=r"^(high_confidence|moderate_confidence|low_confidence)$"
)
uncertainty_signals_triggered: List[str] = Field(default_factory=list)
# e.g. ["U1_low_confidence", "U4_blurry_image"]
```

Remove the single `risk_score` computation that used `1.0 - confidence` as base; use the threat-table formula from Part 4 instead.
