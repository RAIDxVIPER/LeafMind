"""
src/models/vision.py
====================
Re-exports from src.models.classifier for backward compatibility.
The real implementation lives in classifier.py (C2).

This file is kept so that any existing import of src.models.vision
continues to work.
"""
# Re-export everything from the real classifier module
from src.models.classifier import (  # noqa: F401
    EfficientNetB3Classifier,
    LeafClassifier,
    TemperatureScaler,
    compute_class_weights,
    compute_ece,
    preprocess_image,
    check_image_quality,
    resolve_image_path,
    fit_temperature,
    SEVERITY_NOTE,
)

# Backward-compat alias: VisionAgent is in src.agents.vision_agent
from src.agents.vision_agent import VisionAgent  # noqa: F401
