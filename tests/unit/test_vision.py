"""
tests/unit/test_vision.py
=========================
Tests for the vision pipeline interface (C2).
"""

from src.models.vision import VisionAgent
from src.schemas.schemas import VisionOutput

def test_vision_agent_predicts_valid_schema():
    agent = VisionAgent()
    output = agent.predict("dummy/path.jpg")
    
    assert isinstance(output, VisionOutput)
    assert output.severity_pct is None, "R3: severity_pct MUST be None"
    assert "omitted" in output.severity_note.lower()
    assert 0.0 <= output.confidence <= 1.0
