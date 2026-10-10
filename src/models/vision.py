"""
src/models/vision.py
====================
Vision inference wrapper. Mocks PyTorch EfficientNet-B3 classification for pipeline testing.
"""

from pathlib import Path
from src.schemas.schemas import VisionOutput

class VisionAgent:
    def __init__(self, model_path: str = None):
        """
        [STUB] Initializes the EfficientNet-B3 vision model with temperature scaling.
        Currently mocked to prevent heavy PyTorch loads during pipeline testing.
        """
        self.model_path = model_path
        
    def predict(self, image_path: str) -> VisionOutput:
        """
        [MOCK] Simulates running an image through the Vision pipeline.
        Returns a valid VisionOutput instance.
        """
        # As per R3 and Part 2a-xi: severity_pct must always be None.
        return VisionOutput(
            detected_disease="Apple___Apple_scab",
            confidence=0.92,
            severity_pct=None,
            severity_note="Severity tracking omitted due to lack of lesion-level annotations (Part 2a-xi).",
            is_ood=False,
            temperature_scaled=True
        )
