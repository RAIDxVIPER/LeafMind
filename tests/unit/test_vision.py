"""
tests/unit/test_vision.py
=========================
Smoke-level test for the VisionAgent interface (C2).
Full classifier tests are in test_classifier.py.
This file verifies VisionAgent wraps LeafClassifier correctly and
the output satisfies all C2 schema constraints.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.schemas.schemas import VisionOutput

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_rgb_jpeg(w: int = 224, h: int = 224) -> bytes:
    arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _build_mock_classifier():
    """Build a LeafClassifier with untrained model (no network access in CI)."""
    from src.models.classifier import EfficientNetB3Classifier, LeafClassifier

    with open(REPO_ROOT / "config" / "class_map_d2.json") as f:
        raw_map = json.load(f)
    idx_to_label = {int(k): v for k, v in raw_map["classes"].items()}

    model = EfficientNetB3Classifier(num_classes=len(idx_to_label), pretrained=False)
    return LeafClassifier(
        model=model,
        idx_to_label=idx_to_label,
        model_version="test_v0",
        device=torch.device("cpu"),
    )


@pytest.fixture(scope="module")
def vision_agent():
    """VisionAgent backed by untrained model."""
    from src.agents.vision_agent import VisionAgent
    clf = _build_mock_classifier()
    return VisionAgent(classifier=clf, model_version="test_v0")


def test_vision_agent_predicts_valid_schema(vision_agent):
    """VisionAgent.predict() returns a valid VisionOutput."""
    out = vision_agent.predict(_make_rgb_jpeg())
    assert isinstance(out, VisionOutput)


def test_severity_pct_always_none(vision_agent):
    """R3: severity_pct MUST be None in every output."""
    out = vision_agent.predict(_make_rgb_jpeg())
    assert out.severity_pct is None, "CRITICAL: severity_pct must be None [R3/D2]"


def test_severity_note_nonempty(vision_agent):
    """severity_note explains why severity_pct is null [R3]."""
    out = vision_agent.predict(_make_rgb_jpeg())
    assert out.severity_note, "severity_note must be non-empty [R3]"
    # Should mention the core reason
    note_lower = out.severity_note.lower()
    assert any(word in note_lower for word in ["severity", "null", "mask", "label"]), \
        "severity_note should mention why severity is not computed"


def test_confidence_in_range(vision_agent):
    """Confidence is in [0, 1]."""
    out = vision_agent.predict(_make_rgb_jpeg())
    assert 0.0 <= out.confidence <= 1.0


def test_top_label_is_valid_class(vision_agent):
    """top_label is one of the 30 D2 classes [DATA]."""
    with open(REPO_ROOT / "config" / "class_map_d2.json") as f:
        raw_map = json.load(f)
    valid_labels = set(raw_map["classes"].values())
    out = vision_agent.predict(_make_rgb_jpeg())
    assert out.top_label in valid_labels, \
        f"top_label '{out.top_label}' not in D2 class list"


def test_temperature_positive(vision_agent):
    """Temperature must be positive."""
    out = vision_agent.predict(_make_rgb_jpeg())
    assert out.temperature > 0


def test_leaf_detected_false_c2(vision_agent):
    """leaf_detected is always False in C2 (YOLOv8 out of scope)."""
    out = vision_agent.predict(_make_rgb_jpeg())
    assert not out.leaf_detected
