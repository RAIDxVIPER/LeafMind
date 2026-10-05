# PPT Text Extraction
Extracted via python-pptx. Timestamp: 2026-09-30

## FILE 1: AI-Powered-Multi-Agent-System-for-Crop-Leaf-Disease-Detection-and-Farmer-Advisory.pptx

### SLIDE 1
Title: AI-Powered Multi-Agent System for Crop Leaf Disease Detection and Farmer Advisory
Subtitle: An intelligent, modular architecture for rapid diagnosis, risk assessment, and actionable recommendations - delivered to farmers in seconds.

### SLIDE 2 - The Problem
- 20-40% Annual Yield Loss: Crop diseases devastate global harvests
- Late Diagnosis: Farmers miss critical treatment windows
- Scarce Expertise: Agronomist consultations inaccessible for smallholder farmers
- Misidentification Risk: Wrong diagnoses waste resources and worsen outcomes

### SLIDE 3 - Motivation and Objectives
1. Rapid, Accurate Diagnosis: disease identification from smartphone photo in seconds
2. Environmental Context Integration: weather, soil conditions, crop growth stage
3. Farmer-Friendly Recommendations: confidence-scored advice
4. Continuous Feedback Loop: follow-up images and adaptive re-evaluation

### SLIDE 4 - System Architecture High-Level
Flow: Upload -> Orchestration -> Parallel Processing (Vision, Weather, RAG) -> Risk Synthesis -> Recommendation
The Orchestrator coordinates all specialist agents, parallel processing reduces end-to-end latency to under 4 seconds. Monitoring Agent closes feedback loop.

### SLIDE 5 - Agent Roles: Vision Agent
- Input: Leaf image (JPEG/PNG) from farmer device
- Process: CNN for image classification and object detection
- Output: Disease label, confidence score (0-100%), severity estimate (% leaf area affected)
- Design: Trained on agricultural datasets; outputs include uncertainty bounds

### SLIDE 6 - Agent Roles: Weather Agent and RAG Agent
Weather Agent:
- Input: Farmer GPS location and date/time
- Process: Queries weather API for current + 7-day forecast
- Output: Temperature, humidity, rainfall, wind speed; disease-favorable conditions flagged
- Design: Caches recent queries; alerts when conditions favor rapid disease spread

RAG Agent:
- Input: Disease label from Vision Agent
- Process: Semantic search across vector database (extension docs, treatment guides, pesticide DBs)
- Output: Treatment options, fungicide recommendations, application timing, cost estimates
- Design: Grounds all recommendations in peer-reviewed agricultural knowledge

### SLIDE 7 - Agent Roles: Risk/Reason and Recommendation Agents
Risk/Reason Agent: Synthesizes Vision + Weather + RAG outputs. Assesses severity (low/moderate/severe) and estimates disease progression timeline.
Recommendation Agent: Converts risk assessment to actionable steps (e.g. Apply fungicide X within 48 hours), urgency level, confidence justification. Outputs structured JSON for UI.

### SLIDE 8 - Agent Roles: Monitoring Agent
- Input: Follow-up leaf image + full previous case history
- Process: Compares current severity vs baseline; computes delta (improvement or deterioration)
- Output: Updated risk assessment and recommendation adjustment
- Design: Long-term case tracking and treatment efficacy measurement

### SLIDE 9 - Data Flow Timeline
T=0s: Farmer uploads leaf image; Orchestrator logs unique case ID
T=0-2s: Vision, Weather, RAG execute in parallel; each returns structured output independently
T=2-3s: Risk/Reason Agent synthesizes outputs; computes risk score and disease progression estimate
T=3-4s: Recommendation Agent generates actionable advice; UI displays result
T=+Days: Farmer uploads follow-up; Monitoring Agent compares severity; re-triggers Risk/Reason Agent
NOTE: Sub-4-second claim is a DESIGN TARGET, not an experimentally verified metric.

### SLIDE 10 - Why Multi-Agent Architecture?
- Modularity: Single well-defined responsibility per agent
- Parallelization: Vision/Weather/RAG run simultaneously
- Explainability: Each agent output independently interpretable
- Scalability: Agents deploy and load-balance independently
- Robustness: One agent failure (e.g. weather API outage) does not block pipeline

### SLIDE 11 - Research Contribution
Key Contributions:
- Novel Orchestrator-Specialist agent architecture for agricultural diagnostics
- Integration of visual, meteorological, and knowledge retrieval signals into a unified risk model
- Closed-loop monitoring enabling treatment efficacy tracking
- Explainable, confidence-scored outputs for low-literacy, resource-constrained farmers

## FILE 2: Agent-Roles-Weather-Agent.pptx

### SLIDE 1 - Weather Agent (SYSTEM ARCHITECTURE)
- Input: Farmer GPS location, date/time stamp
- Process: Queries weather API for current + 7-day forecast
- Output: Temperature, humidity, rainfall, wind speed; disease-favorable conditions flagged
- Design: Caches recent queries. Alerts Orchestrator when conditions (high humidity, warm nights, post-rain windows) favor rapid pathogen spread.

### SLIDE 2 - RAG Agent (RETRIEVAL-AUGMENTED GENERATION)
- Input: Disease label from Vision Agent
- Process: Semantic search across vector DB of agricultural extension docs, treatment guides, pesticide registries
- Output: Relevant treatment options, fungicide/pesticide recommendations, application timing, cost estimates
- Design: Grounding prevents hallucination; retrieval traceable to source documents

### SLIDE 3 - Risk/Reason and Recommendation Agents
Risk/Reason Agent: Synthesizes Vision+Weather+RAG into unified assessment. Risk level: low/moderate/severe. Estimates disease progression timeline.
Recommendation Agent: Clear actionable steps (e.g. Apply fungicide within 48 hours). Urgency level and plain-language confidence justification. Structured JSON for farmer UI.

### SLIDE 4 - Monitoring Agent (FEEDBACK LOOP)
Steps:
1. Follow-up Image Uploaded -> attach new image to case history
2. Compare Against Baseline -> align and contrast with initial image
3. Compute Severity Delta -> calculate percent change in affected area
4. Update Risk and Adjust Recommendation -> revise risk score and treatment advice
By computing percentage change in affected leaf area over time: enables long-term treatment efficacy measurement; triggers escalation when disease continues despite intervention.

### SLIDE 5 - Data Flow (repeated)
Same as main PPT slide 9 timeline.

### SLIDE 6 - Why Multi-Agent (repeated)
Same as main PPT slide 10.

### SLIDE 7 - Handling Uncertainty and Confidence
Presenting confidence ranges (not binary) builds farmer trust and supports regulatory compliance.
System never overstates certainty: low-confidence detections surface a human agronomist for manual verification.
Uncertainty quantification is a core design principle, not a post-hoc addition.

### SLIDE 8 - Monitoring and Feedback Loop
1. Case Tracking: unique case ID per farmer; follow-ups linked to original baseline
2. Severity Trending: percentage change in affected leaf area; quantitative severity trajectory
3. Adaptive Recommendations: if disease worsens despite treatment, escalate urgency or propose alternatives

### SLIDE 9 - Expected Challenges and Limitations
- Image Quality: poor lighting/resolution reduces Vision accuracy. Mitigation: in-app guidance for photo capture
- Disease Overlap: visually similar symptoms increase misclassification. Mitigation: confidence thresholds trigger agronomist escalation
- Regional Knowledge Gaps: RAG may lack local treatment practices. Mitigation: curation with regional extension partners
- Connectivity: rural intermittent internet disrupts real-time agents. Mitigation: offline mode for image capture and lightweight on-device inference

### SLIDE 10 - Future Scope
Near-Term: Push-based monitoring (proactive alerts), multilingual support, offline mode
Long-Term: Crop-specific tuning (wheat/rice/maize), supply chain integration (pesticide availability and pricing)

### SLIDE 11 - Conclusion
- Scalable Diagnosis at scale across diverse crops and geographies
- Farmer Trust via transparency in confidence scoring and chain-of-reasoning
- Continuous Improvement: monitoring feedback loop for treatment efficacy tracking
- Democratizing Expertise: accessible to resource-constrained farmers
