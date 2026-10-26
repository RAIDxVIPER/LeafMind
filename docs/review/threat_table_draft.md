# Disease Threat Table — Pathologist Review Draft (Issue I19)

> **Status:** `[UNVERIFIED — needs human reviewer sign-off, I19]`  
> **Date:** 2026-10-27  
> **Scope:** Initial baseline tier proposal for 17 mapped diseased classes in LeafMind Knowledge Base, plus 1 unmapped model gap (`Squash___Powdery_mildew`).  
> **Rule Enforcement:** 15 approved tiers applied as UNVERIFIED_PENDING_PATHOLOGIST in `config/threat_table.json`. 3 classes (`Apple___Cedar_apple_rust`, `Grape___Esca_(Black_Measles)`, `Squash___Powdery_mildew`) remain `[GAP]` with fallback `DISEASE_DETECTED = 0.50`.

---

## 1. Evaluation & Assignment Rubric

Tiers must correspond strictly to valid values defined in `config/risk_config.yaml` and `src/risk/threat_table.py`:
- `HIGH` (Threat Score = `0.85`)
- `MEDIUM` (Threat Score = `0.50`)
- `LOW` (Threat Score = `0.25`)
- `INSUFFICIENT EVIDENCE — keep GAP` (Fallback Score = `0.50`, label capped at "moderate" per I26)

### Criteria for Tiers

| Tier | Score | Rubric Criteria |
|---|---|---|
| **HIGH** | `0.85` | **Severe / Catastrophic Threat:**<br>1. Reported yield loss $\ge 30\text{--}50\%$ up to $100\%$ under epiphytotic or untreated conditions.<br>2. Explosive epidemic spread (e.g., destroys foliage or entire fields within days to a week).<br>3. Systemic, incurable pathogen (e.g., viruses where infected plants do not recover and vectors spread rapidly).<br>4. Explicitly identified by ICAR, CPRI, NHB, or state universities as the "most dreaded", "most destructive", or primary national threat for the crop. |
| **MEDIUM** | `0.50` | **Moderate / Manageable Threat:**<br>1. Causes notable foliar blighting, extensive defoliation, or fruit blemishes that degrade marketable quality without causing immediate field death.<br>2. Moderate or conditional yield losses ($10\text{--}30\%$), manageable via standard protectant/systemic spray schedules or cultural rotation.<br>3. Identified as a "predominant" seasonal disease (e.g., late kharif maize rust) or "main bacterial disease", but lacking explosive field-collapse dynamics.<br>4. Disease predominantly targeting older or senescing leaves. |
| **LOW** | `0.25` | **Minor / Localized / Slow Threat:**<br>1. Confined primarily to specific protected environments (e.g., polyhouses/greenhouses) and rarely destructive in open field cultivation.<br>2. Single-cycle non-reinfecting pathogens on the primary host (pathogen cannot complete full life cycle or reinfect without secondary host).<br>3. Readily checked by basic cultural sanitation, pruning, and airflow with negligible direct yield/economic impact on fruit. |
| **INSUFFICIENT EVIDENCE — keep GAP** | `0.50` | **No Clear Severity Evidence:**<br>1. The chapter provides diagnostic morphology and chemical options but contains no usable severity, destructiveness, or yield loss statements.<br>2. Text explicitly documents that economic impact is difficult to quantify and no Indian loss data exist.<br>3. Must **NOT** be guessed; retained as `[GAP]` until pathologist evaluation. |

### Confidence Levels
- **High:** Verbatim text provides direct quantitative loss data ($\%$) or explicit superlative severity ranking from Indian agronomic institutes (ICAR, CPRI, TNAU, NHB, State Universities).
- **Medium:** Clear qualitative statements of widespread defoliation, fruit rotting, or predominance, but without direct loss percentages, or relying partly on supplementary international data.
- **Low:** Inferred from life cycle constraints (e.g., inability to reinfect host) or conflicting evidence across regional sources without Indian loss figures.

---

## 2. Proposed Threat Tiers (17 Mapped Classes + 1 Model Gap)

All proposed tiers below are `[UNVERIFIED — needs human reviewer sign-off, I19]`.

### 1. `Apple___Apple_scab`
- **Crop:** Apple
- **KB Chapter:** Apple Scab (*Venturia inaequalis*)
- **Proposed Tier:** `HIGH` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** High
- **Verbatim Evidence from KB:**
  1. *[Chunk `Apple__1_glance`]*: "The most dreaded foliar and fruit disease of apple in the north-western Himalayas. Cool, wet spring weather drives infection."
  2. *[Chunk `Apple__1_sec2`]*: "Economic loss: up to 70 % of production value from fruit and pedicel infection with poor control, and up to 100 % with no control (VER)."
  3. *[Chunk `Apple__1_sec4`]*: "Apple scab is the most dreaded disease on commercial apple cultivars in the Kashmir valley and a major problem in Himachal Pradesh (VER, GK)."
- **Reviewer Note:** Extreme economic loss potential (up to 100% without control) and near-zero tolerance for scabbed fruit justify `HIGH`.

---

### 2. `Apple___Cedar_apple_rust`
- **Crop:** Apple
- **KB Chapter:** Cedar Apple Rust (*Gymnosporangium juniperi-virginianae*)
- **Proposed Tier:** `GAP` (DISEASE_DETECTED fallback, `0.50`) `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** Low `[ENG judgement]`
- **Reason:** Changed from LOW to GAP per reviewer direction. Pathogen requires an alternate juniper host (eastern red cedar), is single-cycle non-reinfecting on apple, and has zero reported occurrences or loss studies in Indian apple orchards. Retained as `GAP` pending direct pathologist review.
- **Verbatim Evidence from KB:**
  1. *[Chunk `Apple__9_glance`]*: "It is a major disease where junipers grow near apple orchards in North America. No Indian report or Indian-labelled recommendation was found, so this chapter is mainly diagnostic and preventive information."
  2. *[Chunk `Apple__9_sec2`]*: "Spores are carried by wind; the spore stage produced on apple cannot reinfect apple (CAR1)."
  3. *[Chunk `Apple__9_sec4`]*: "No Indian report of cedar apple rust on apple was found in the sources reviewed. The principal juniper host named in the literature, eastern red cedar, is a North American tree; the disease needs a susceptible juniper near the orchard."

---

### 3. `Corn___Northern_Leaf_Blight`
- **Crop:** Corn (Maize)
- **KB Chapter:** Turcicum Leaf Blight (Northern Corn Leaf Blight) (*Exserohilum turcicum*)
- **Proposed Tier:** `HIGH` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** High
- **Verbatim Evidence from KB:**
  1. *[Chunk `Corn__1_glance`]*: "Long grey-green to tan, cigar-shaped lesions that start on lower leaves and can 'burn' the whole canopy. One of the four nationally important maize diseases; listed by MPKV Rahuri as predominant in kharif Maharashtra."
  2. *[Chunk `Corn__1_sec2`]*: "Losses: grain yield losses of 25 to 90% are reported in epiphytotics (IIMR); avoidable losses of 52% grain and 40% stover were estimated in a susceptible genotype in Karnataka (UAS)."
  3. *[Chunk `Corn__1_sec1`]*: "damage is greatest when infection reaches the crop around silking under favourable weather (T3)."
- **Reviewer Note:** Up to 90% grain loss in epiphytotics and designation as one of India's four nationally important maize diseases firmly justify `HIGH`.

---

### 4. `Corn___Common_rust`
- **Crop:** Corn (Maize)
- **KB Chapter:** Common Rust (*Puccinia sorghi*)
- **Proposed Tier:** `MEDIUM` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** Medium
- **Verbatim Evidence from KB:**
  1. *[Chunk `Corn__4_glance`]*: "Powdery cinnamon-brown pustules on both leaf surfaces in cool, moist weather. AICRP Rahuri lists rust as predominant in late kharif Maharashtra."
  2. *[Chunk `Corn__4_sec1`]*: "In severe cases infection spreads to sheaths, tassels and ears, and leaves yellow and tatter in wind (T6, IIMR)."
  3. *[Chunk `Corn__4_sec2`]*: "Primary spread through the alternate host; secondary spread by wind-borne urediniospores (T6)."
- **Reviewer Note:** Predominant late kharif disease in Maharashtra that causes leaf tattering and spreading in severe cases, but without the catastrophic 90% grain collapse reported for Turcicum leaf blight. Managed with targeted sprays.

---

### 5. `Corn___Cercospora_leaf_spot`
- **Crop:** Corn (Maize)
- **KB Chapter:** Gray Leaf Spot (*Cercospora zeae-maydis* / *C. zeina*)
- **Proposed Tier:** `MEDIUM` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** Medium
- **Verbatim Evidence from KB:**
  1. *[Chunk `Corn__13_glance`]*: "A yield-limiting disease of highland Asia, but it is not among IIMR's national or regional priority diseases, so this chapter rests on international and supplementary sources. No maize claim in CIB&RC."
  2. *[Chunk `Corn__13_sec1`]*: "Heavy infection before grain filling can cause large yield loss and weakened stalks that lodge (GL1)."
  3. *[Chunk `Corn__13_sec2`]*: "Early infection leads to the highest losses (GL2)."
- **Reviewer Note:** Described as "yield-limiting" and causing lodging when infection precedes grain filling, but explicitly omitted from IIMR national priority lists in India. Rated `MEDIUM`.

---

### 6. `Grape___Black_rot`
- **Crop:** Grape
- **KB Chapter:** Black Rot (*Guignardia bidwellii*)
- **Proposed Tier:** `MEDIUM` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** Medium
- **Verbatim Evidence from KB:**
  1. *[Chunk `Grape__8_glance`]*: "A disease of warm, moist weather that attacks leaves, stems, tendrils, flowers and berries, ending in hard black mummified berries."
  2. *[Chunk `Grape__8_sec1`]*: "Crop / stage: leaves, stems, flowers and berries; all new growth is prone to attack during the growing season (NHB)... Affected berries shrivel and become hard black mummies (NHB)."
  3. *[Chunk `Grape__8_sec4`]*: "Listed among diseases of national significance in the NIPHM grape package (NG); no state-specific data found."
- **Reviewer Note:** Causes direct fruit loss (mummification of berries) and is listed as a disease of national significance by NIPHM, but lacks quantitative epiphytotic loss data in Indian vineyards.

---

### 7. `Grape___Esca_(Black_Measles)`
- **Crop:** Grape
- **KB Chapter:** Esca (Black Measles) (Trunk disease complex)
- **Proposed Tier:** `INSUFFICIENT EVIDENCE — keep GAP` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** Low `[ENG judgement]`
- **Verbatim Evidence from KB:**
  1. *[Chunk `Grape__11_sec2`]*: "Losses: UC notes that the economic impact on grape production is difficult to measure accurately (OC4H). No Indian loss data were found."
  2. *[Chunk `Grape__11_sec2`]*: "Once a pruning wound is infected, the pathogen establishes a permanent, localized wood infection that cannot be eradicated by fungicide applications (UE)."
  3. *[Chunk `Grape__11_glance`]*: "The fungi enter through pruning wounds and cannot be removed by fungicides once established, so wound protection and sanitation are the measures. Esca is not described in the Indian sources used for the other chapters of this document"
- **Reviewer Note:** Although the wood infection is permanent and incurable once established, the manual explicitly notes that economic loss is difficult to measure and that zero Indian loss data were found. Per the rubric, this is held as `INSUFFICIENT EVIDENCE — keep GAP`.

---

### 8. `Pepper,_bell___Bacterial_spot`
- **Crop:** Pepper, bell (Capsicum)
- **KB Chapter:** Bacterial Spot (Bacterial Leaf Spot) (*Xanthomonas euvesicatoria* / *X. campestris pv. vesicatoria*)
- **Proposed Tier:** `MEDIUM` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** Medium
- **Verbatim Evidence from KB:**
  1. *[Chunk `Pepper,_bell__1_glance`]*: "The main bacterial disease of capsicum and chilli: small water-soaked leaf spots with yellow halos, leaf drop and raised scabby fruit spots in warm, humid, rainy weather."
  2. *[Chunk `Pepper,_bell__1_sec1`]*: "The disease was found in moderate to severe form in bell pepper areas of Solan, Shimla and Sirmour districts of Himachal Pradesh (GP)."
  3. *[Chunk `Pepper,_bell__1_sec2`]*: "Seed transmissibility is high (AUS)."
- **Reviewer Note:** Acknowledged as the primary bacterial disease of capsicum causing defoliation and unmarketable scabby fruit lesions, documented in "moderate to severe form" in Indian bell pepper tracts.

---

### 9. `Potato___Late_blight`
- **Crop:** Potato
- **KB Chapter:** Late Blight (*Phytophthora infestans*)
- **Proposed Tier:** `HIGH` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** High
- **Verbatim Evidence from KB:**
  1. *[Chunk `Potato__1_glance`]*: "The most destructive potato disease in India: water-soaked leaf lesions with white mildew beneath that can destroy a crop within a week in cool, humid weather. ICAR-CPRI estimates average annual losses of about 15% in India."
  2. *[Chunk `Potato__1_sec1`]*: "Under favourable conditions the whole crop looks blighted and can be destroyed within a week (CP1)."
  3. *[Chunk `Potato__1_sec2`]*: "Losses: about 15% average annual loss in India, equal to about 8.0 million tonnes (CP1); yield losses of 11 to 74% depending on climate are cited (DEY)."
- **Reviewer Note:** Unanimously cited as the single most destructive disease of potato in India (annual loss ~8 million tonnes, up to 74% yield destruction within 7 days). Definitive `HIGH`.

---

### 10. `Potato___Early_blight`
- **Crop:** Potato
- **KB Chapter:** Early Blight (Alternaria Leaf Spot) (*Alternaria solani* / *A. alternata*)
- **Proposed Tier:** `MEDIUM` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** Medium
- **Verbatim Evidence from KB:**
  1. *[Chunk `Potato__2_glance`]*: "Dry, angular dark-brown leaf spots with concentric rings (bull's-eye) that spread from lower leaves in warm, humid weather with alternate wet and dry spells. It also dents tuber quality."
  2. *[Chunk `Potato__2_sec1`]*: "spots can coalesce and the field then looks severely blighted (CP1)."
  3. *[Chunk `Potato__2_sec2`]*: "In Israeli work, early blight was described as principally a disease of senescing plants (SHT, supplementary)."
- **Reviewer Note:** While capable of blighting fields and reducing tuber quality under stress, it primarily attacks senescing lower foliage and lacks the explosive total-crop devastation of late blight.

---

### 11. `Tomato___Early_blight`
- **Crop:** Tomato
- **KB Chapter:** Early Blight (*Alternaria solani*)
- **Proposed Tier:** `MEDIUM` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** High
- **Verbatim Evidence from KB:**
  1. *[Chunk `Tomato__2_glance`]*: "Brown target-board spots on older leaves, stem collar and fruit. Favoured by warm, humid, rainy weather. Second most serious tomato disease after leaf curl in an ICAR assessment."
  2. *[Chunk `Tomato__2_sec2`]*: "spores are splashed/blown to upper foliage and fruit; severe cases cause leaf drop and fruit rot at the calyx."
  3. *[Chunk `Tomato__2_sec4`]*: "MPKV Rahuri's Tomato Improvement Scheme lists early blight resistance as a breeding objective (MP)."
- **Reviewer Note:** Major foliar and fruit-rot threat ("second most serious tomato disease after leaf curl in an ICAR assessment"), but can be kept in check with protectant sprays, unlike viral or oomycete collapse.

---

### 12. `Tomato___Late_blight`
- **Crop:** Tomato
- **KB Chapter:** Late Blight (*Phytophthora infestans*)
- **Proposed Tier:** `HIGH` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** High
- **Verbatim Evidence from KB:**
  1. *[Chunk `Tomato__3_glance`]*: "Fast-spreading blight of leaves, stems and fruit in cool, wet, foggy weather. Can destroy a field in days. Preventive sprays and resistant hybrids give the best protection."
  2. *[Chunk `Tomato__3_sec2`]*: "infection can spread widely in 4-5 days (IH26)."
  3. *[Chunk `Tomato__3_sec4`]*: "in two years of natural epidemics the average yield loss was 12.84 % vs 75.5 % (Arka Rakshak) and 79.5 % (NS 501) (RED, AA)."
- **Reviewer Note:** Extreme destructive potential (destroys field in days; up to 79.5% yield loss in natural epidemics on susceptible varieties). Definitive `HIGH`.

---

### 13. `Tomato___Septoria_leaf_spot`
- **Crop:** Tomato
- **KB Chapter:** Septoria Leaf Spot (*Septoria lycopersici*)
- **Proposed Tier:** `MEDIUM` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** Medium
- **Verbatim Evidence from KB:**
  1. *[Chunk `Tomato__4_glance`]*: "Many small round spots with grey centres and dark margins on leaves, leading to blighting and defoliation. Spread by rain splash in humid weather."
  2. *[Chunk `Tomato__4_sec1`]*: "Small, round to irregular spots with a grey centre and dark margin on leaves; spots coalesce, leaves blight and may defoliate completely; stems and flowers can be affected (T4)."
  3. *[Chunk `Tomato__4_sec1`]*: "Fruit is rarely infected (UME)."
- **Reviewer Note:** Can lead to extensive leaf blighting and complete defoliation under high humidity, but fruit is rarely infected directly, preventing total crop loss.

---

### 14. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
- **Crop:** Tomato
- **KB Chapter:** Tomato Leaf Curl Disease (TYLCV / Begomoviruses)
- **Proposed Tier:** `HIGH` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** High
- **Verbatim Evidence from KB:**
  1. *[Chunk `Tomato__8_glance`]*: "Stunting with downward rolling and crinkling of leaves, spread by whiteflies. One of the most damaging tomato diseases in India. Vector control, resistant hybrids and protected nurseries are key."
  2. *[Chunk `Tomato__8_sec1`]*: "Severe stunting of plants; downward rolling and crinkling of leaves (T6)."
  3. *[Chunk `Tomato__8_sec5`]*: "These insecticides target the VECTOR, not the virus; infected plants do not recover."
- **Reviewer Note:** Rated as one of the most damaging diseases in India; infected plants do not recover, internodes shorten drastically, and young infections lead to complete loss of marketable yield.

---

### 15. `Tomato___Bacterial_spot`
- **Crop:** Tomato
- **KB Chapter:** Bacterial Leaf Spot (Bacterial Spot) (*Xanthomonas axonopodis pv. vesicatoria*)
- **Proposed Tier:** `MEDIUM` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** Medium
- **Verbatim Evidence from KB:**
  1. *[Chunk `Tomato__10_glance`]*: "Small water-soaked leaf spots with a yellow halo, scabby fruit spots and defoliation of older leaves, spread by rain splash. TNAU's recommendation is disease-free seed plus streptocycline sprays"
  2. *[Chunk `Tomato__10_sec1`]*: "Small, brown, water-soaked, circular spots with a yellow halo on leaves; older leaves may defoliate (T11)... Small, water-soaked spots on green fruit; the centres become irregular, light brown, sunken and scabby (T11)."
  3. *[Chunk `Tomato__10_sec5`]*: "No chemical is a cure once lesions are established; the aim is protection of healthy tissue."
- **Reviewer Note:** Defoliates older leaves and produces unmarketable scabby fruit blemishes, but does not cause systemic death of the entire plant canopy.

---

### 16. `Tomato___Leaf_Mold`
- **Crop:** Tomato
- **KB Chapter:** Leaf Mould (*Passalora fulva* / *Cladosporium fulvum*)
- **Proposed Tier:** `LOW` `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** Medium
- **Verbatim Evidence from KB:**
  1. *[Chunk `Tomato__11_glance`]*: "A foliar disease of tomato, especially in humid polyhouses and high tunnels. Yellow patches on the upper leaf surface with velvety olive mould underneath. Airflow, low humidity, sanitation and resistant varieties are the controls."
  2. *[Chunk `Tomato__11_sec1`]*: "Infected blossoms turn black and fall; fruit can show bronze or sunken leathery lesions but this is uncommon (LM1, LM2). Leaves eventually drop prematurely (LM2)."
  3. *[Chunk `Tomato__11_sec4`]*: "No Maharashtra-specific data were found. The disease is most relevant to polyhouse tomato."
- **Reviewer Note:** Primarily an issue in enclosed structures (polyhouses/high tunnels) with inadequate airflow. Fruit lesions are explicitly described as "uncommon". Readily controlled by managing humidity and spacing.

---

### 17. `Tomato___Tomato_mosaic_virus`
- **Crop:** Tomato
- **KB Chapter:** Tomato Mosaic Virus (and related mosaic viruses — ToMV / ToMMV)
- **Proposed Tier:** `HIGH` `[UNVERIFIED — reviewer judgement call, I19]`
- **Confidence:** Medium `[reviewer judgement call]`
- **Verbatim Evidence from KB:**
  1. *[Chunk `Tomato__12_glance`]*: "Mottled green-yellow mosaic, distorted leaves and small or deformed fruit. Tobamoviruses spread by seed, hands, tools and weeds, so clean seed, sanitation and resistant varieties are the main controls. No chemical cure."
  2. *[Chunk `Tomato__12_sec1`]*: "ToMMV in India: mild mosaic that progresses to severe leaf shoestring, with small, hard, unripe or deformed fruit; more than 50 % of plants showed symptoms in an experimental field (MV1)."
  3. *[Chunk `Tomato__12_sec2`]*: "Press reports from 2023 attributed mosaic crop losses to CMV in Maharashtra and to ToMV in Karnataka and other southern states (MV3, secondary source)."
- **Reviewer Note:** Reviewer judgement call: incurable systemic viral infection transmitted mechanically and through seed. Causes severe shoestring, stunted deformed unmarketable fruit, and has demonstrated ability to overcome established host resistance (ToMMV overcoming ToMV resistance genes).

---

### 18. `Squash___Powdery_mildew` (Model Disease Gap)
- **Crop:** Squash
- **KB Chapter:** None (Expected Gap)
- **Proposed Tier:** `GAP` (DISEASE_DETECTED fallback, `0.50`) `[UNVERIFIED — needs human reviewer sign-off, I19]`
- **Confidence:** High (Unmapped Gap)
- **Reason:** No manual or chapter exists for Squash in `data/knowledge_base/`. Formally catalogued in `config/kb_class_map.yaml` under `model_classes_without_kb`. No verified text available to extract severity quotes. Must remain `GAP` until a crop manual is ingested.

---

## 3. Summary Table for Reviewer Sign-Off

| # | Model Class | Crop | Proposed Tier `[UNVERIFIED]` | Tier Score | Confidence | Primary Basis |
|---|---|---|---|---|---|---|
| 1 | `Apple___Apple_scab` | Apple | **HIGH** | 0.85 | High | Up to 100% loss; most dreaded apple disease |
| 2 | `Apple___Cedar_apple_rust` | Apple | **GAP** *(fallback)* | 0.50 | Low | Single-cycle on apple; no Indian reports; retained as GAP |
| 3 | `Corn___Northern_Leaf_Blight` | Corn | **HIGH** | 0.85 | High | 25–90% grain loss in epiphytotics; national priority |
| 4 | `Corn___Common_rust` | Corn | **MEDIUM** | 0.50 | Medium | Predominant late kharif; leaf tattering in severe cases |
| 5 | `Corn___Cercospora_leaf_spot` | Corn | **MEDIUM** | 0.50 | Medium | Yield-limiting in Asia; not national priority |
| 6 | `Grape___Black_rot` | Grape | **MEDIUM** | 0.50 | Medium | Berries shrivel to black mummies; national significance |
| 7 | `Grape___Esca_(Black_Measles)` | Grape | **INSUFFICIENT EVIDENCE — keep GAP** | 0.50 | Low | Text notes loss is difficult to measure & zero Indian data |
| 8 | `Pepper,_bell___Bacterial_spot` | Pepper | **MEDIUM** | 0.50 | Medium | Main bacterial disease; moderate-to-severe form in HP |
| 9 | `Potato___Late_blight` | Potato | **HIGH** | 0.85 | High | Most destructive potato disease; destroys crop in 7 days |
| 10 | `Potato___Early_blight` | Potato | **MEDIUM** | 0.50 | Medium | Dents tuber quality; mainly disease of senescing plants |
| 11 | `Tomato___Early_blight` | Tomato | **MEDIUM** | 0.50 | High | 2nd most serious tomato disease; fruit rot & leaf drop |
| 12 | `Tomato___Late_blight` | Tomato | **HIGH** | 0.85 | High | Up to 79.5% loss; destroys field in days |
| 13 | `Tomato___Septoria_leaf_spot` | Tomato | **MEDIUM** | 0.50 | Medium | Complete defoliation; fruit rarely infected |
| 14 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato | **HIGH** | 0.85 | High | Incurable vector-borne; one of most damaging in India |
| 15 | `Tomato___Bacterial_spot` | Tomato | **MEDIUM** | 0.50 | Medium | Severe defoliation & scabby fruit; no field collapse |
| 16 | `Tomato___Leaf_Mold` | Tomato | **LOW** | 0.25 | Medium | Primarily polyhouse issue; fruit lesions uncommon |
| 17 | `Tomato___Tomato_mosaic_virus` | Tomato | **HIGH** | 0.85 | Medium | Reviewer judgement call: incurable, shoestring & deformed fruit |
| 18 | `Squash___Powdery_mildew` | Squash | **GAP** | 0.50 | High | No KB chapter in data/knowledge_base/ |

---
*Draft compiled strictly from extractive retrieval of `data/knowledge_base/` manuals via `src/rag/retriever.py` and `vector_db/kb_chunks.json`. Live table `config/threat_table.json` has applied these 15 tiers as UNVERIFIED_PENDING_PATHOLOGIST.*
