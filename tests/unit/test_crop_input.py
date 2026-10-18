"""
tests/unit/test_crop_input.py
==============================
Unit tests for user-supplied crop input (C2 follow-up 2).
All tests run on CPU only [ENG].

Tests cover:
  1. Masking correctness: logit masking restricts predictions strictly to crop classes.
  2. Renormalisation sums to 1: softmax over allowed classes sums to 1.0.
  3. Top-3 ordering: top_3_candidates are sorted descending by probability.
  4. Unknown crop error: unknown crop raises ValueError listing supported crops.
  5. Single-class flag: Blueberry, Cherry, Peach, Raspberry, Soybean, Strawberry, Squash
     set single_class_crop=True and can_diagnose=False; multi-class crops set False/True.
  6. Corn alias: Corn, Corn_(maize), corn, Corn (maize) all map to the same 4 classes.
  7. crop=None regression: None produces exact unmasked behaviour, top_3_candidates=None.
  8. VisionAgent integration: VisionAgent forwards crop parameter correctly.
  9. Candidate unpacking & indexing: ClassProb supports tuple unpacking and indexing.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.models.classifier import LeafClassifier, TemperatureScaler, extract_crop_from_label
from src.agents.vision_agent import VisionAgent
from src.schemas.schemas import ClassProb, VisionOutput

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Fixtures & Helpers ────────────────────────────────────────────────────────

def _make_dummy_jpeg(w: int = 224, h: int = 224) -> bytes:
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class MockModel(nn.Module):
    """Mock PyTorch module that returns controllable logits."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = nn.Parameter(logits.clone(), requires_grad=False)
        self.scaler = TemperatureScaler(temperature=1.0)

    def forward(self, x: torch.Tensor, calibrate: bool = True) -> torch.Tensor:
        batch_size = x.shape[0]
        # Repeat the logits for batch size
        return self.logits.unsqueeze(0).repeat(batch_size, 1)

    def get_temperature(self) -> float:
        return 1.0


@pytest.fixture(scope="module")
def full_idx_to_label() -> Dict[int, str]:
    with open(REPO_ROOT / "config" / "class_map_d2.json") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["classes"].items()}


@pytest.fixture
def mock_classifier(full_idx_to_label) -> LeafClassifier:
    """Classifier with uniform zero logits across all 30 classes."""
    logits = torch.zeros(len(full_idx_to_label), dtype=torch.float32)
    model = MockModel(logits)
    return LeafClassifier(
        model=model,
        idx_to_label=full_idx_to_label,
        model_version="test_crop_v1",
        device=torch.device("cpu"),
    )


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestCropResolutionAndAliases:
    """Test crop resolution, alias mapping, and error reporting."""

    def test_extract_crop_from_label(self):
        assert extract_crop_from_label("Apple___Apple_scab") == "Apple"
        assert extract_crop_from_label("Tomato___healthy") == "Tomato"
        assert extract_crop_from_label("Corn_(maize)___healthy") == "Corn"
        assert extract_crop_from_label("Corn___Common_rust") == "Corn"
        assert extract_crop_from_label("Pepper,_bell___healthy") == "Pepper,_bell"

    def test_supported_crops_populated(self, mock_classifier):
        # 13 crops in total across the 30 classes
        expected_crops = {
            "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Peach",
            "Pepper,_bell", "Potato", "Raspberry", "Soybean", "Squash",
            "Strawberry", "Tomato"
        }
        assert set(mock_classifier.crop_to_indices.keys()) == expected_crops

    def test_corn_alias_resolution(self, mock_classifier):
        for alias in ["Corn", "Corn_(maize)", "corn", "Corn (maize)", "maize"]:
            resolved = mock_classifier._resolve_crop(alias)
            assert resolved == "Corn", f"Failed for alias: {alias}"
            # 4 classes for Corn
            assert len(mock_classifier.crop_to_indices[resolved]) == 4

    def test_unknown_crop_raises_value_error_listing_crops(self, mock_classifier):
        dummy_img = _make_dummy_jpeg()
        with pytest.raises(ValueError) as exc_info:
            mock_classifier.predict(dummy_img, crop="Mango")

        err_msg = str(exc_info.value)
        assert "Unknown crop 'Mango'" in err_msg
        assert "Supported crops:" in err_msg
        assert "Apple" in err_msg
        assert "Corn" in err_msg
        assert "Tomato" in err_msg

    def test_empty_crop_raises_value_error(self, mock_classifier):
        dummy_img = _make_dummy_jpeg()
        with pytest.raises(ValueError) as exc_info:
            mock_classifier.predict(dummy_img, crop="   ")
        assert "Supported crops:" in str(exc_info.value)


class TestMaskingCorrectness:
    """Test that predictions are strictly masked to the given crop."""

    def test_masking_overrides_dominant_cross_crop_logits(self, full_idx_to_label):
        # Set a massive logit on Tomato___Early_blight (idx 23)
        logits = torch.zeros(len(full_idx_to_label), dtype=torch.float32)
        logits[23] = 100.0  # Tomato___Early_blight has massive logit

        # Set moderate logits on Corn classes (indices 5, 6, 7, 8)
        # Corn___Northern_Leaf_Blight (idx 8) has highest among corn
        logits[8] = 10.0
        logits[6] = 5.0

        model = MockModel(logits)
        clf = LeafClassifier(
            model=model,
            idx_to_label=full_idx_to_label,
            device=torch.device("cpu"),
        )

        dummy_img = _make_dummy_jpeg()

        # Unmasked: predicts Tomato___Early_blight
        unmasked_out = clf.predict(dummy_img, crop=None)
        assert unmasked_out.top_label == "Tomato___Early_blight"

        # Masked to Corn: MUST predict Corn___Northern_Leaf_Blight, NEVER Tomato
        masked_out = clf.predict(dummy_img, crop="Corn")
        assert masked_out.top_label == "Corn___Northern_Leaf_Blight"
        assert masked_out.crop == "Corn"

        # All top candidates must belong to Corn
        for cand in masked_out.top_3_candidates:
            assert cand.label.startswith("Corn"), f"Cross-crop leak: {cand.label}"


class TestRenormalisationAndOrdering:
    """Test softmax renormalisation sums to 1 and descending candidate ordering."""

    def test_renormalisation_sums_to_one(self, full_idx_to_label):
        logits = torch.randn(len(full_idx_to_label), dtype=torch.float32)
        model = MockModel(logits)
        clf = LeafClassifier(
            model=model,
            idx_to_label=full_idx_to_label,
            device=torch.device("cpu"),
        )
        dummy_img = _make_dummy_jpeg()

        # Apple has 3 classes: top-3 candidates MUST sum to 1.0
        out_apple = clf.predict(dummy_img, crop="Apple")
        assert len(out_apple.top_3_candidates) == 3
        prob_sum = sum(cand.prob for cand in out_apple.top_3_candidates)
        assert prob_sum == pytest.approx(1.0, rel=1e-5)

        # Pepper,_bell has 2 classes: top-3 candidates has 2 items and sums to 1.0
        out_pepper = clf.predict(dummy_img, crop="Pepper,_bell")
        assert len(out_pepper.top_3_candidates) == 2
        assert sum(cand.prob for cand in out_pepper.top_3_candidates) == pytest.approx(1.0, rel=1e-5)

    def test_top_3_ordering_descending(self, full_idx_to_label):
        logits = torch.zeros(len(full_idx_to_label), dtype=torch.float32)
        # Tomato has 8 classes (indices 22..29)
        # Set specific logits so probabilities are strictly ordered
        logits[22] = 2.0  # Bacterial_spot
        logits[23] = 8.0  # Early_blight -> rank 1
        logits[24] = 6.0  # Late_blight  -> rank 2
        logits[25] = 4.0  # Leaf_Mold    -> rank 3
        logits[26] = 1.0

        model = MockModel(logits)
        clf = LeafClassifier(
            model=model,
            idx_to_label=full_idx_to_label,
            device=torch.device("cpu"),
        )
        dummy_img = _make_dummy_jpeg()

        out = clf.predict(dummy_img, crop="Tomato")
        cands = out.top_3_candidates

        assert len(cands) == 3
        assert cands[0].label == "Tomato___Early_blight"
        assert cands[1].label == "Tomato___Late_blight"
        assert cands[2].label == "Tomato___Leaf_Mold"

        # Strictly descending probabilities
        assert cands[0].prob > cands[1].prob > cands[2].prob
        assert out.confidence == cands[0].prob
        assert out.top_label == cands[0].label


class TestSingleClassCrops:
    """Test single-class crop handling: single_class_crop=True, can_diagnose=False."""

    SINGLE_CLASS_CROPS = [
        "Blueberry",
        "Cherry",
        "Peach",
        "Raspberry",
        "Soybean",
        "Squash",
        "Strawberry",
    ]

    MULTI_CLASS_CROPS = [
        "Apple",
        "Corn",
        "Grape",
        "Pepper,_bell",
        "Potato",
        "Tomato",
    ]

    @pytest.mark.parametrize("crop", SINGLE_CLASS_CROPS)
    def test_single_class_crops_flags(self, mock_classifier, crop):
        dummy_img = _make_dummy_jpeg()
        out = mock_classifier.predict(dummy_img, crop=crop)

        assert out.single_class_crop is True, f"{crop} should have single_class_crop=True"
        assert out.can_diagnose is False, f"{crop} should have can_diagnose=False"
        assert len(out.top_3_candidates) == 1
        assert out.top_3_candidates[0].prob == pytest.approx(1.0)
        assert out.confidence == pytest.approx(1.0)

    @pytest.mark.parametrize("crop", MULTI_CLASS_CROPS)
    def test_multi_class_crops_flags(self, mock_classifier, crop):
        dummy_img = _make_dummy_jpeg()
        out = mock_classifier.predict(dummy_img, crop=crop)

        assert out.single_class_crop is False, f"{crop} should have single_class_crop=False"
        assert out.can_diagnose is True, f"{crop} should have can_diagnose=True"
        assert len(out.top_3_candidates) >= 2


class TestCornAliasConsistency:
    """Verify all corn aliases produce identical predictions and candidates."""

    def test_corn_aliases_identical(self, full_idx_to_label):
        logits = torch.randn(len(full_idx_to_label), dtype=torch.float32)
        model = MockModel(logits)
        clf = LeafClassifier(
            model=model,
            idx_to_label=full_idx_to_label,
            device=torch.device("cpu"),
        )
        dummy_img = _make_dummy_jpeg()

        out_corn = clf.predict(dummy_img, crop="Corn")
        out_maize = clf.predict(dummy_img, crop="Corn_(maize)")
        out_lower = clf.predict(dummy_img, crop="corn")
        out_space = clf.predict(dummy_img, crop="Corn (maize)")

        for out in [out_maize, out_lower, out_space]:
            assert out.top_label == out_corn.top_label
            assert out.confidence == pytest.approx(out_corn.confidence)
            assert len(out.top_3_candidates) == len(out_corn.top_3_candidates)
            for c1, c2 in zip(out.top_3_candidates, out_corn.top_3_candidates):
                assert c1.label == c2.label
                assert c1.prob == pytest.approx(c2.prob)


class TestCropNoneRegression:
    """Verify that crop=None preserves exact v1 unmasked behaviour."""

    def test_crop_none_matches_omitted_crop(self, full_idx_to_label):
        logits = torch.randn(len(full_idx_to_label), dtype=torch.float32)
        model = MockModel(logits)
        clf = LeafClassifier(
            model=model,
            idx_to_label=full_idx_to_label,
            device=torch.device("cpu"),
        )
        dummy_img = _make_dummy_jpeg()

        out_default = clf.predict(dummy_img)
        out_none = clf.predict(dummy_img, crop=None)

        assert out_default.top_label == out_none.top_label
        assert out_default.confidence == pytest.approx(out_none.confidence)
        assert out_none.crop is None
        assert out_none.top_3_candidates is None
        assert out_none.single_class_crop is False
        assert out_none.can_diagnose is True

        # Candidate aliases on VisionOutput return None when crop is None
        assert out_none.crop_candidates is None
        assert out_none.top3_candidates is None


class TestVisionAgentCropIntegration:
    """Verify VisionAgent passes crop argument and preserves output semantics."""

    def test_vision_agent_predict_with_crop(self, mock_classifier):
        agent = VisionAgent(classifier=mock_classifier, model_version="agent_v1")
        dummy_img = _make_dummy_jpeg()

        # With valid crop
        out = agent.predict(dummy_img, crop="Tomato")
        assert isinstance(out, VisionOutput)
        assert out.crop == "Tomato"
        assert out.can_diagnose is True
        assert len(out.top_3_candidates) == 3

        # With single-class crop
        out_single = agent.predict(dummy_img, crop="Cherry")
        assert out_single.single_class_crop is True
        assert out_single.can_diagnose is False

        # With unknown crop
        with pytest.raises(ValueError, match="Supported crops:"):
            agent.predict(dummy_img, crop="UnknownCropXYZ")

    def test_class_prob_unpacking_and_indexing(self):
        cp = ClassProb(label="Corn___Common_rust", prob=0.88)
        # Tuple unpacking
        lbl, prob = cp
        assert lbl == "Corn___Common_rust"
        assert prob == 0.88

        # Indexing
        assert cp[0] == "Corn___Common_rust"
        assert cp[1] == 0.88
        assert cp.class_name == "Corn___Common_rust"
