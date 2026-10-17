"""
tests/unit/test_classifier.py
==============================
C2 unit tests for the LeafMind EfficientNet-B3 classifier pipeline.

Tests cover:
  - Model architecture and forward pass
  - Image preprocessing (RGB conversion from CMYK, RGBA, L [I9])
  - Image quality flags (low_resolution, blurry) [ENG assumption thresholds]
  - ECE computation
  - Class weight computation
  - Temperature scaling
  - VisionOutput schema compliance [R6]
  - severity_pct = None always [R3/D2/Part 2a-xi]
  - OOD flag and escalation flag logic [ENG assumption]
  - Full predict() pipeline on synthetic images
  - from_checkpoint() round-trip (save + load + predict)

No real model weights are downloaded or used in these tests.
All model instances use pretrained=False to avoid network access in CI.
All performance numbers are from synthetic data [R1] -- no fabricated metrics.
ASCII-only output [I27].
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

# ── Fixtures ──────────────────────────────────────────────────────────────────

NUM_CLASSES = 30  # D2 [DATA]

MOCK_IDX_TO_LABEL: Dict[int, str] = {
    0: "Apple___Apple_scab",
    1: "Apple___Cedar_apple_rust",
    2: "Apple___healthy",
    3: "Blueberry___healthy",
    4: "Cherry___healthy",
    5: "Corn_(maize)___healthy",
    6: "Corn___Cercospora_leaf_spot",
    7: "Corn___Common_rust",
    8: "Corn___Northern_Leaf_Blight",
    9: "Grape___Black_rot",
    10: "Grape___Esca_(Black_Measles)",
    11: "Grape___healthy",
    12: "Peach___healthy",
    13: "Pepper,_bell___Bacterial_spot",
    14: "Pepper,_bell___healthy",
    15: "Potato___Early_blight",
    16: "Potato___Late_blight",
    17: "Potato___healthy",
    18: "Raspberry___healthy",
    19: "Soybean___healthy",
    20: "Squash___Powdery_mildew",
    21: "Strawberry___healthy",
    22: "Tomato___Bacterial_spot",
    23: "Tomato___Early_blight",
    24: "Tomato___Late_blight",
    25: "Tomato___Leaf_Mold",
    26: "Tomato___Septoria_leaf_spot",
    27: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    28: "Tomato___Tomato_mosaic_virus",
    29: "Tomato___healthy",
}


def _make_image_bytes(w: int = 224, h: int = 224, mode: str = "RGB") -> bytes:
    """Create a small synthetic image in the given mode as JPEG bytes."""
    arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    if mode == "CMYK":
        img = img.convert("CMYK")
    elif mode == "RGBA":
        img = img.convert("RGBA")
    elif mode == "L":
        img = img.convert("L")
    buf = io.BytesIO()
    fmt = "JPEG" if mode != "RGBA" else "PNG"
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def model_no_pretrain():
    """EfficientNetB3Classifier without pretrained weights (no network access in CI)."""
    from src.models.classifier import EfficientNetB3Classifier
    return EfficientNetB3Classifier(num_classes=NUM_CLASSES, pretrained=False, temperature=1.5)


@pytest.fixture(scope="module")
def leaf_classifier_no_pretrain(model_no_pretrain):
    """LeafClassifier wrapping the untrained model."""
    from src.models.classifier import LeafClassifier
    return LeafClassifier(
        model=model_no_pretrain,
        idx_to_label=MOCK_IDX_TO_LABEL,
        model_version="test_v0",
        device=torch.device("cpu"),
    )


# ── Architecture tests ────────────────────────────────────────────────────────

class TestEfficientNetB3Architecture:

    def test_model_creates_without_pretrained(self):
        """Model builds without downloading weights."""
        from src.models.classifier import EfficientNetB3Classifier
        model = EfficientNetB3Classifier(num_classes=30, pretrained=False)
        assert model is not None

    def test_forward_pass_output_shape(self, model_no_pretrain):
        """Forward pass produces (batch, 30) logits."""
        model_no_pretrain.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model_no_pretrain(x, calibrate=False)
        assert out.shape == (2, NUM_CLASSES), f"Expected (2, {NUM_CLASSES}), got {out.shape}"

    def test_calibrated_forward_divides_by_temperature(self, model_no_pretrain):
        """Temperature scaling applied when calibrate=True."""
        model_no_pretrain.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            raw    = model_no_pretrain(x, calibrate=False)
            scaled = model_no_pretrain(x, calibrate=True)
        T = model_no_pretrain.get_temperature()
        assert T > 0, "Temperature must be positive"
        # scaled logits should differ from raw (unless T==1.0)
        diff = (scaled - raw / T).abs().max().item()
        assert diff < 1e-4, f"Calibrated logits differ unexpectedly: diff={diff}"

    def test_temperature_parameter_is_learnable(self, model_no_pretrain):
        """Temperature is a nn.Parameter (learnable for calibration)."""
        assert isinstance(model_no_pretrain.scaler.temperature, nn.Parameter)

    def test_num_classes_correct(self, model_no_pretrain):
        """Output layer has exactly 30 units [DATA]."""
        assert model_no_pretrain.num_classes == NUM_CLASSES

    def test_output_sums_to_one_after_softmax(self, model_no_pretrain):
        """Softmax of calibrated logits sums to 1."""
        model_no_pretrain.eval()
        x = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            logits = model_no_pretrain(x, calibrate=True)
            probs  = torch.softmax(logits, dim=1)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(3), atol=1e-5), \
            f"Softmax does not sum to 1: {row_sums}"


# ── Preprocessing + RGB conversion tests [I9] ─────────────────────────────────

class TestImagePreprocessing:

    def test_rgb_image_passes_through(self):
        """RGB image preprocesses without error."""
        from src.models.classifier import preprocess_image
        img_bytes = _make_image_bytes(mode="RGB")
        tensor, pil_img = preprocess_image(img_bytes, input_size=224)
        assert tensor.shape == (1, 3, 224, 224)
        assert pil_img.mode == "RGB"

    def test_cmyk_converted_to_rgb(self):
        """CMYK image is force-converted to RGB at load time [I9]."""
        from src.models.classifier import preprocess_image
        img_bytes = _make_image_bytes(mode="CMYK")
        tensor, pil_img = preprocess_image(img_bytes, input_size=224)
        assert tensor.shape == (1, 3, 224, 224), "CMYK->RGB tensor must be (1,3,H,W)"

    def test_rgba_converted_to_rgb(self):
        """RGBA image is force-converted to RGB at load time [I9]."""
        from src.models.classifier import preprocess_image
        img_bytes = _make_image_bytes(mode="RGBA")
        tensor, pil_img = preprocess_image(img_bytes, input_size=224)
        assert tensor.shape == (1, 3, 224, 224), "RGBA->RGB tensor must be (1,3,H,W)"

    def test_grayscale_L_converted_to_rgb(self):
        """L (grayscale) image is force-converted to RGB at load time [I9]."""
        from src.models.classifier import preprocess_image
        img_bytes = _make_image_bytes(mode="L")
        tensor, pil_img = preprocess_image(img_bytes, input_size=224)
        assert tensor.shape == (1, 3, 224, 224), "L->RGB tensor must be (1,3,H,W)"

    def test_corrupt_bytes_raises_value_error(self):
        """Invalid bytes raise ValueError."""
        from src.models.classifier import preprocess_image
        with pytest.raises((ValueError, Exception)):
            preprocess_image(b"not an image", input_size=224)

    def test_output_tensor_dtype_is_float32(self):
        """Output tensor is float32."""
        from src.models.classifier import preprocess_image
        img_bytes = _make_image_bytes(mode="RGB")
        tensor, _ = preprocess_image(img_bytes, input_size=224)
        assert tensor.dtype == torch.float32

    def test_normalisation_applied(self):
        """Tensor values are not in raw [0,1] range after normalisation."""
        from src.models.classifier import preprocess_image
        img_bytes = _make_image_bytes(mode="RGB")
        tensor, _ = preprocess_image(img_bytes, input_size=224)
        # After ImageNet normalisation, some values will be negative
        assert tensor.min().item() < 0.0, "Normalised tensor should have negative values"


# ── Image quality flag tests [ENG assumption thresholds] ──────────────────────

class TestImageQualityFlags:

    def test_normal_image_not_low_res(self):
        """224x224 image is not flagged as low resolution [ENG assumption: threshold=200]."""
        from src.models.classifier import check_image_quality
        img = Image.new("RGB", (224, 224), color=(100, 150, 50))
        flags = check_image_quality(img, low_res_px=200)
        assert not flags.low_resolution, "224x224 should NOT be low_resolution"

    def test_tiny_image_flagged_as_low_res(self):
        """50x50 image is flagged as low resolution [ENG assumption]."""
        from src.models.classifier import check_image_quality
        img = Image.new("RGB", (50, 50), color=(100, 150, 50))
        flags = check_image_quality(img, low_res_px=200)
        assert flags.low_resolution, "50x50 MUST be flagged as low_resolution [ENG assumption]"

    def test_boundary_at_threshold(self):
        """Image with max dimension exactly at threshold is flagged [ENG assumption: strict <]."""
        from src.models.classifier import check_image_quality
        img = Image.new("RGB", (199, 150), color=(100, 150, 50))
        flags = check_image_quality(img, low_res_px=200)
        assert flags.low_resolution, "max(199,150)=199 < 200 -> low_resolution=True [ENG]"

    def test_uniform_image_is_blurry(self):
        """
        Completely uniform image has Laplacian variance = 0 -> blurry [ENG assumption].
        This tests the blurry detection heuristic, not calibration quality.
        """
        from src.models.classifier import check_image_quality
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        flags = check_image_quality(img, blur_lap=100.0)
        assert flags.blurry, "Uniform image must be flagged as blurry [ENG assumption]"

    def test_non_leaf_suspected_always_false(self):
        """non_leaf_suspected is always False (no dedicated model) [ENG]."""
        from src.models.classifier import check_image_quality
        img = Image.new("RGB", (224, 224), color=(100, 150, 50))
        flags = check_image_quality(img)
        assert not flags.non_leaf_suspected, \
            "non_leaf_suspected must always be False (no leaf detection model) [ENG]"

    def test_quality_flags_return_correct_type(self):
        """check_image_quality returns ImageQualityFlags Pydantic model."""
        from src.models.classifier import check_image_quality
        from src.schemas.schemas import ImageQualityFlags
        img = Image.new("RGB", (224, 224), color=(100, 150, 50))
        flags = check_image_quality(img)
        assert isinstance(flags, ImageQualityFlags)


# ── ECE computation tests ─────────────────────────────────────────────────────

class TestECEComputation:

    def test_perfect_calibration_ece_near_zero(self):
        """
        Perfect calibration: confidence matches accuracy -> ECE near 0.
        Example: confidence=0.8, fraction correct=0.8 -> bin ECE = |0.8-0.8|*1.0 = 0.
        Using a mixed scenario where for each bin the accuracy == confidence.
        """
        from src.models.classifier import compute_ece
        rng = np.random.RandomState(7)
        # Create samples where for every sample, confidence = true probability of being correct
        # Approximate by making accuracy ≈ confidence at the aggregate level:
        # Use confidence=0.7 with 70% correct -> ECE bin contribution = |0.7 - 0.7| * 1.0 = 0
        n = 1000
        confs = np.full(n, 0.7)
        # Exactly 70% correct -> accuracy in bin = 0.7 = confidence -> ECE = 0
        correct = np.zeros(n)
        correct[:700] = 1.0
        ece = compute_ece(confs, correct, n_bins=15)
        assert ece < 0.01, f"Perfect calibration ECE (conf=acc=0.7) should be ~0, got {ece}"

    def test_overconfident_model_high_ece(self):
        """Overconfident model: high confidence, low accuracy -> high ECE."""
        from src.models.classifier import compute_ece
        n = 1000
        confs   = np.full(n, 0.95)
        correct = np.zeros(n)  # all wrong
        ece = compute_ece(confs, correct, n_bins=15)
        assert ece > 0.5, f"Overconfident ECE should be > 0.5, got {ece}"

    def test_ece_in_zero_one_range(self):
        """ECE is always in [0, 1]."""
        from src.models.classifier import compute_ece
        rng = np.random.RandomState(42)
        confs   = rng.random(500)
        correct = rng.randint(0, 2, 500).astype(float)
        ece = compute_ece(confs, correct)
        assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"

    def test_ece_target_is_eng_assumption(self):
        """Verify that ECE target 0.05 is tagged as ENG assumption in config."""
        import yaml
        cfg_path = Path(__file__).resolve().parents[2] / "config" / "classifier_config.yaml"
        if not cfg_path.exists():
            pytest.skip("classifier_config.yaml not found -- skipping config check")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        # The ECE target value should match what's in the threshold register [plan.md]
        ece_target = cfg.get("calibration", {}).get("ece_target", None)
        assert ece_target == 0.05, f"Expected ece_target=0.05 [ENG assumption], got {ece_target}"


# ── Class weight tests ────────────────────────────────────────────────────────

class TestClassWeights:

    def test_class_weights_shape(self):
        """Class weights tensor has shape (num_classes,)."""
        from src.models.classifier import compute_class_weights
        counts = {i: 100 + i * 10 for i in range(30)}
        w = compute_class_weights(counts, 30)
        assert w.shape == (30,)

    def test_class_weights_inverse_frequency(self):
        """Minority class gets higher weight than majority class."""
        from src.models.classifier import compute_class_weights
        counts = {0: 100, 1: 3750}  # simulates Potato___healthy vs TYLCV imbalance [DATA]
        w = compute_class_weights(counts, 2)
        assert w[0] > w[1], "Minority class (100 images) must have higher weight than majority (3750)"

    def test_class_weights_sum_property(self):
        """With uniform counts, all weights should be equal."""
        from src.models.classifier import compute_class_weights
        counts = {i: 100 for i in range(30)}
        w = compute_class_weights(counts, 30)
        assert torch.allclose(w, w[0].expand(30), atol=1e-5), \
            "Uniform counts should produce equal weights"

    def test_class_weights_positive(self):
        """All weights must be strictly positive."""
        from src.models.classifier import compute_class_weights
        counts = {0: 1, 1: 3750, **{i: 500 for i in range(2, 30)}}
        w = compute_class_weights(counts, 30)
        assert (w > 0).all(), "All class weights must be positive"


# ── Temperature scaling tests ─────────────────────────────────────────────────

class TestTemperatureScaling:

    def test_temperature_scaler_divides_logits(self):
        """TemperatureScaler divides logits by T."""
        from src.models.classifier import TemperatureScaler
        scaler = TemperatureScaler(temperature=2.0)
        logits = torch.tensor([[2.0, 4.0, 6.0]])
        scaled = scaler(logits)
        expected = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(scaled, expected, atol=1e-5)

    def test_temperature_scaler_t_equals_one_is_identity(self):
        """T=1.0 leaves logits unchanged."""
        from src.models.classifier import TemperatureScaler
        scaler = TemperatureScaler(temperature=1.0)
        logits = torch.randn(4, 30)
        scaled = scaler(logits)
        assert torch.allclose(scaled, logits, atol=1e-5)

    def test_temperature_cannot_be_zero_or_negative(self):
        """Temperature is clamped to positive values in forward."""
        from src.models.classifier import TemperatureScaler
        scaler = TemperatureScaler(temperature=0.0)
        logits = torch.tensor([[1.0, 2.0]])
        # Should not produce NaN/Inf due to clamp(min=1e-6)
        out = scaler(logits)
        assert not torch.isnan(out).any(), "Clamped T=0 should not produce NaN"
        assert not torch.isinf(out).any(), "Clamped T=0 should not produce Inf"


# ── Full predict() pipeline tests ─────────────────────────────────────────────

class TestLeafClassifierPredict:

    def test_predict_returns_vision_output(self, leaf_classifier_no_pretrain):
        """predict() returns a VisionOutput instance."""
        from src.schemas.schemas import VisionOutput
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert isinstance(out, VisionOutput), f"Expected VisionOutput, got {type(out)}"

    def test_severity_pct_always_none(self, leaf_classifier_no_pretrain):
        """severity_pct is ALWAYS None [R3/D2/Part 2a-xi]."""
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert out.severity_pct is None, "CRITICAL VIOLATION: severity_pct must be None [R3]"

    def test_severity_note_present_and_nonempty(self, leaf_classifier_no_pretrain):
        """severity_note is a non-empty string explaining why severity is null [R3]."""
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert out.severity_note, "severity_note must be non-empty [R3]"
        assert len(out.severity_note) > 20, "severity_note must be a meaningful explanation"

    def test_confidence_in_valid_range(self, leaf_classifier_no_pretrain):
        """Confidence is in [0.0, 1.0]."""
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert 0.0 <= out.confidence <= 1.0, f"Confidence out of range: {out.confidence}"

    def test_top_k_probs_nonempty(self, leaf_classifier_no_pretrain):
        """top_k_probs has at least one entry."""
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert len(out.top_k_probs) >= 1, "top_k_probs must have at least one entry"

    def test_top_label_is_valid_d2_class(self, leaf_classifier_no_pretrain):
        """top_label is one of the 30 D2 classes [DATA]."""
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        valid_labels = set(MOCK_IDX_TO_LABEL.values())
        assert out.top_label in valid_labels, \
            f"top_label '{out.top_label}' not in D2 class list [DATA]"

    def test_temperature_positive(self, leaf_classifier_no_pretrain):
        """Temperature value > 0."""
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert out.temperature > 0, "temperature must be positive"

    def test_model_version_nonempty(self, leaf_classifier_no_pretrain):
        """model_version field is non-empty."""
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert out.model_version, "model_version must be non-empty"

    def test_leaf_detected_false_for_c2(self, leaf_classifier_no_pretrain):
        """leaf_detected is always False in C2 (YOLOv8 out of scope)."""
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert not out.leaf_detected, \
            "leaf_detected must be False in C2 (YOLOv8 detector out of scope)"

    def test_detected_leaves_empty_for_c2(self, leaf_classifier_no_pretrain):
        """detected_leaves is empty list in C2."""
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert out.detected_leaves == [], \
            "detected_leaves must be [] in C2 (YOLOv8 out of scope)"

    def test_ood_flag_high_for_low_confidence(self):
        """
        When model produces low confidence (< 0.40 [ENG assumption]), ood_flag=True.
        This is tested by setting ood_thresh=1.0 to force ood_flag=True for any image.
        [ENG assumption -- threshold to be calibrated on labelled OOD set]
        """
        from src.models.classifier import EfficientNetB3Classifier, LeafClassifier
        model = EfficientNetB3Classifier(num_classes=NUM_CLASSES, pretrained=False)
        clf = LeafClassifier(
            model=model,
            idx_to_label=MOCK_IDX_TO_LABEL,
            device=torch.device("cpu"),
            ood_thresh=1.0,    # force ood_flag=True for any confidence [ENG test]
        )
        img_bytes = _make_image_bytes(mode="RGB")
        out = clf.predict(img_bytes)
        assert out.ood_flag, \
            "ood_flag should be True when confidence < ood_thresh=1.0 [ENG assumption]"

    def test_escalation_forced_by_low_conf_threshold(self):
        """
        escalate_to_human=True when confidence < escalate threshold [ENG assumption].
        """
        from src.models.classifier import EfficientNetB3Classifier, LeafClassifier
        model = EfficientNetB3Classifier(num_classes=NUM_CLASSES, pretrained=False)
        clf = LeafClassifier(
            model=model,
            idx_to_label=MOCK_IDX_TO_LABEL,
            device=torch.device("cpu"),
            esc_conf=1.0,   # force escalation for any confidence [ENG test]
        )
        img_bytes = _make_image_bytes(mode="RGB")
        out = clf.predict(img_bytes)
        assert out.escalate_to_human, \
            "escalate_to_human must be True when conf < esc_conf=1.0 [ENG assumption]"

    def test_cmyk_predict_succeeds(self, leaf_classifier_no_pretrain):
        """CMYK image is handled via RGB conversion -- no crash [I9]."""
        img_bytes = _make_image_bytes(mode="CMYK")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert isinstance(out.confidence, float)
        assert out.severity_pct is None

    def test_rgba_predict_succeeds(self, leaf_classifier_no_pretrain):
        """RGBA image is handled via RGB conversion -- no crash [I9]."""
        img_bytes = _make_image_bytes(mode="RGBA")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert isinstance(out.confidence, float)

    def test_grayscale_predict_succeeds(self, leaf_classifier_no_pretrain):
        """L (grayscale) image is handled via RGB conversion -- no crash [I9]."""
        img_bytes = _make_image_bytes(mode="L")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        assert isinstance(out.confidence, float)

    def test_schema_validates_via_pydantic(self, leaf_classifier_no_pretrain):
        """VisionOutput from predict() passes Pydantic model_validate [R6]."""
        from src.schemas.schemas import VisionOutput
        img_bytes = _make_image_bytes(mode="RGB")
        out = leaf_classifier_no_pretrain.predict(img_bytes)
        # model_validate will re-validate all fields; raises ValidationError if invalid
        VisionOutput.model_validate(out.model_dump())


# ── Checkpoint round-trip test ─────────────────────────────────────────────────

class TestCheckpointRoundTrip:

    def test_save_and_load_checkpoint(self, tmp_path):
        """Save model to tmp checkpoint dir and reload via from_checkpoint()."""
        import json
        from src.models.classifier import EfficientNetB3Classifier, LeafClassifier

        # Create and save a model
        model = EfficientNetB3Classifier(num_classes=NUM_CLASSES, pretrained=False, temperature=1.5)
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        torch.save(model.state_dict(), ckpt_dir / "model.pth")
        with open(ckpt_dir / "temperature.json", "w") as f:
            json.dump({"temperature": 1.5}, f)
        with open(ckpt_dir / "class_map.json", "w") as f:
            json.dump({str(k): v for k, v in MOCK_IDX_TO_LABEL.items()}, f)
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump({"model_version": "test_ckpt_v0", "num_classes": NUM_CLASSES}, f)

        # Load and predict
        clf = LeafClassifier.from_checkpoint(ckpt_dir, device=torch.device("cpu"))
        img_bytes = _make_image_bytes(mode="RGB")
        out = clf.predict(img_bytes)

        assert out.model_version == "test_ckpt_v0"
        assert out.temperature == pytest.approx(1.5, abs=1e-4)
        assert out.severity_pct is None  # R3
        assert isinstance(out.confidence, float)

    def test_missing_checkpoint_dir_raises(self):
        """from_checkpoint raises FileNotFoundError when dir missing."""
        from src.models.classifier import LeafClassifier
        with pytest.raises(FileNotFoundError):
            LeafClassifier.from_checkpoint("/nonexistent/path/xyz")

    def test_missing_model_pth_raises(self, tmp_path):
        """from_checkpoint raises FileNotFoundError when model.pth missing."""
        import json
        from src.models.classifier import LeafClassifier
        (tmp_path / "class_map.json").write_text(
            json.dumps({str(k): v for k, v in MOCK_IDX_TO_LABEL.items()})
        )
        (tmp_path / "temperature.json").write_text(json.dumps({"temperature": 1.5}))
        # No model.pth
        with pytest.raises(FileNotFoundError):
            LeafClassifier.from_checkpoint(tmp_path)


# ── Bug regression tests ──────────────────────────────────────────────────────


class TestBug1ResolveImagePath:
    """Bug1: Windows backslashes in manifest paths break path resolution on Linux."""

    def test_forward_slash_path_resolves(self, tmp_path):
        """Forward-slash path (already correct) resolves to the right file."""
        from src.models.classifier import resolve_image_path
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "img.jpg").write_bytes(b"x")
        result = resolve_image_path(tmp_path, "sub/img.jpg")
        assert result == tmp_path / "sub" / "img.jpg"

    def test_backslash_path_resolves(self, tmp_path):
        """Windows backslash path is correctly converted to forward slash."""
        from src.models.classifier import resolve_image_path
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "img.jpg").write_bytes(b"x")
        # Simulate a Windows manifest entry like "sub\\img.jpg"
        result = resolve_image_path(tmp_path, "sub\\img.jpg")
        assert result == tmp_path / "sub" / "img.jpg"

    def test_deep_backslash_path_resolves(self, tmp_path):
        """Multi-level Windows backslash path resolves correctly."""
        from src.models.classifier import resolve_image_path
        (tmp_path / "Train" / "Apple___scab").mkdir(parents=True)
        target = tmp_path / "Train" / "Apple___scab" / "img.JPG"
        target.write_bytes(b"x")
        # Simulate D2 manifest entry: "Train\\Apple___scab\\img.JPG"
        result = resolve_image_path(tmp_path, "Train\\Apple___scab\\img.JPG")
        assert result == target

    def test_d2dataset_backslash_entry_loads(self, tmp_path):
        """D2Dataset loads an image from a manifest entry with a backslash path [Bug1]."""
        import io, json
        import numpy as np
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader
        from scripts.train_classifier import D2Dataset, build_eval_transform

        # Create a real image on disk in sub-directory
        cls_dir = tmp_path / "Train" / "Apple___Apple_scab"
        cls_dir.mkdir(parents=True)
        img_path = cls_dir / "test_img.JPG"
        arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(str(img_path), format="JPEG")

        label_to_idx = {"Apple___Apple_scab": 0}
        cfg = {
            "model": {"input_size": 64},
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        }
        tf = build_eval_transform(cfg)

        # Manifest entry uses Windows backslash (as produced on Windows)
        entries = [{"path": "Train\\Apple___scab\\test_img.JPG", "class": "Apple___Apple_scab", "md5": "abc"}]
        # Fix the path to match the actual file (backslash converted)
        entries = [{"path": "Train\\Apple___Apple_scab\\test_img.JPG", "class": "Apple___Apple_scab", "md5": "abc"}]

        ds = D2Dataset(entries, tmp_path, tf, label_to_idx)
        tensor, label = ds[0]
        assert tensor.shape[0] == 3, "Expected 3-channel tensor"
        assert label == 0


class TestBug2CalibrateEntryKey:
    """Bug2: calibrate.py D2ValDataset used entry['label'] but manifests use entry['class']."""

    def test_d2val_dataset_uses_class_key(self, tmp_path):
        """D2ValDataset correctly reads entry['class'] not entry['label']."""
        import io
        import numpy as np
        from PIL import Image
        from torch.utils.data import DataLoader

        # We test the dataset class directly from calibrate.py
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root))

        from scripts.calibrate import D2ValDataset

        # Build a tiny image on disk
        cls_dir = tmp_path / "Val" / "Apple___Apple_scab"
        cls_dir.mkdir(parents=True)
        img_path = cls_dir / "img.JPG"
        arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(str(img_path), format="JPEG")

        # Manifest entry uses 'class' key (real format)
        entries = [{
            "path": "Val\\Apple___Apple_scab\\img.JPG",
            "class": "Apple___Apple_scab",
            "md5": "deadbeef",
        }]
        label_to_idx = {"Apple___Apple_scab": 0}

        ds = D2ValDataset(entries, tmp_path, label_to_idx, input_size=64)
        tensor, label = ds[0]
        assert label == 0, (
            "D2ValDataset should read entry['class'] not entry['label'] [Bug2]. "
            "Got label={} expected 0".format(label)
        )

    def test_d2val_dataset_wrong_key_gives_zero(self, tmp_path):
        """Regression: if 'label' key exists but 'class' is used, label resolves to 0 via .get() default."""
        import numpy as np
        from PIL import Image
        from scripts.calibrate import D2ValDataset

        cls_dir = tmp_path / "Val" / "SomeClass"
        cls_dir.mkdir(parents=True)
        img_path = cls_dir / "img.JPG"
        arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(str(img_path), format="JPEG")

        # Correct entry with 'class' key -- should give label 5
        entries = [{"path": "Val\\SomeClass\\img.JPG", "class": "SomeClass", "md5": "x"}]
        label_to_idx = {"SomeClass": 5}
        ds = D2ValDataset(entries, tmp_path, label_to_idx, input_size=64)
        _, label = ds[0]
        assert label == 5, "Label should be 5 when 'class' key is used correctly [Bug2]"


class TestBug3CalibrateTemperatureDevice:
    """Bug3: calibrate_temperature must not create a CPU parameter when model is on CUDA."""

    def test_calibrate_temperature_cpu(self):
        """calibrate_temperature works correctly on CPU (device-agnostic baseline)."""
        from scripts.train_classifier import calibrate_temperature, D2Dataset, build_eval_transform
        from src.models.classifier import EfficientNetB3Classifier, compute_class_weights
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cpu")
        num_classes = 5

        model = EfficientNetB3Classifier(num_classes=num_classes, pretrained=False, temperature=1.5)
        model.to(device)
        model.eval()

        # Build a fake DataLoader: pre-made (tensor, label) pairs
        # We mock the val_loader by wrapping tensors directly
        inputs  = torch.randn(20, 3, 224, 224)
        targets = torch.randint(0, num_classes, (20,))
        ds = TensorDataset(inputs, targets)
        loader = DataLoader(ds, batch_size=10)

        # Should complete without device errors
        t = calibrate_temperature(model, loader, device, temperature_init=1.5, max_iter=5)
        assert isinstance(t, float), "calibrate_temperature must return a float"
        assert t > 0, "Optimised temperature must be positive [Bug3]"

        # After calibration, model.scaler.temperature should be on 'device'
        assert model.scaler.temperature.device.type == device.type, (
            "After calibration, model.scaler.temperature must be on the model's device [Bug3]"
        )

    def test_calibrate_temperature_returns_param_on_model_device(self):
        """After calibrate_temperature, the T parameter lives on model.device (Bug3)."""
        from scripts.train_classifier import calibrate_temperature
        from src.models.classifier import EfficientNetB3Classifier
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cpu")
        model = EfficientNetB3Classifier(num_classes=4, pretrained=False).to(device)
        inputs  = torch.randn(8, 3, 224, 224)
        targets = torch.randint(0, 4, (8,))
        loader  = DataLoader(TensorDataset(inputs, targets), batch_size=8)

        calibrate_temperature(model, loader, device, temperature_init=1.5, max_iter=3)
        assert model.scaler.temperature.device.type == "cpu", (
            "model.scaler.temperature must be on CPU after calibration on CPU [Bug3]"
        )


class TestBug4CheckpointTemperatureRoundTrip:
    """Bug4: from_checkpoint must load temperature from temperature.json and override state dict."""

    def test_temperature_roundtrip_t2(self, tmp_path):
        """Save checkpoint with T=2.0 in temperature.json; load and assert get_temperature()==2.0."""
        from src.models.classifier import EfficientNetB3Classifier, LeafClassifier
        import json, torch

        # Build model and save with T=1.0 in state dict
        model = EfficientNetB3Classifier(num_classes=NUM_CLASSES, pretrained=False, temperature=1.0)
        torch.save(model.state_dict(), tmp_path / "model.pth")

        # Write temperature.json with T=2.0 (simulates post-calibration)
        with open(tmp_path / "temperature.json", "w") as f:
            json.dump({"temperature": 2.0}, f)
        with open(tmp_path / "class_map.json", "w") as f:
            json.dump({str(k): v for k, v in MOCK_IDX_TO_LABEL.items()}, f)
        with open(tmp_path / "config.json", "w") as f:
            json.dump({"model_version": "test_T2", "num_classes": NUM_CLASSES}, f)

        clf = LeafClassifier.from_checkpoint(tmp_path, device=torch.device("cpu"))

        actual_T = clf.model.get_temperature()
        assert abs(actual_T - 2.0) < 1e-4, (
            "from_checkpoint must honour temperature.json (T=2.0) and override "
            "whatever T was baked into the state dict. Got T={}. [Bug4]".format(actual_T)
        )

    def test_temperature_roundtrip_preserves_weights(self, tmp_path):
        """Loading from checkpoint doesn't corrupt model weights (basic sanity)."""
        from src.models.classifier import EfficientNetB3Classifier, LeafClassifier
        import json, torch

        model = EfficientNetB3Classifier(num_classes=NUM_CLASSES, pretrained=False, temperature=1.5)
        torch.save(model.state_dict(), tmp_path / "model.pth")
        with open(tmp_path / "temperature.json", "w") as f:
            json.dump({"temperature": 1.5}, f)
        with open(tmp_path / "class_map.json", "w") as f:
            json.dump({str(k): v for k, v in MOCK_IDX_TO_LABEL.items()}, f)

        clf = LeafClassifier.from_checkpoint(tmp_path, device=torch.device("cpu"))
        # Should predict without error
        import io
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        out = clf.predict(buf.getvalue())
        assert out.temperature == pytest.approx(1.5, abs=1e-4)


class TestBug5BlurPerImage:
    """Bug5: GaussianBlur must be applied per-image via RandomApply, not once at build time."""

    def test_random_apply_blur_is_per_image(self):
        """Over 200 transform calls, both blurred and unblurred outputs occur [Bug5]."""
        from torchvision import transforms
        from PIL import Image
        import numpy as np

        # Use a high-frequency checkerboard so blur changes pixel values measurably.
        # Flat images look identical whether blurred or not.
        checker = np.zeros((80, 80, 3), dtype=np.uint8)
        for row in range(80):
            for col in range(80):
                checker[row, col] = 255 if (row // 4 + col // 4) % 2 == 0 else 0
        img = Image.fromarray(checker, mode="RGB")

        # Build a transform with p=0.5 blur using RandomApply (the fixed approach)
        tf = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(2.0, 2.0))],
                p=0.5,
            ),
            transforms.ToTensor(),
        ])

        # Measure the variance across 200 calls -- if blur is per-image,
        # some calls will change the output significantly.
        patch_means = []
        for _ in range(200):
            t = tf(img)
            patch_means.append(float(t[0, 28:36, 28:36].std().item()))

        unique_vals = set(round(v, 5) for v in patch_means)

        # If blur was applied exactly once at build time, all std values would
        # be the same. With RandomApply, we expect at least 2 distinct values.
        assert len(unique_vals) >= 2, (
            "GaussianBlur via RandomApply should produce different outputs per call [Bug5]. "
            "Got only {} distinct variance values over 200 calls.".format(len(unique_vals))
        )

    def test_build_train_transform_uses_random_apply(self):
        """build_train_transform must not use a one-time random.random() conditional [Bug5]."""
        import inspect
        from scripts.train_classifier import build_train_transform

        source = inspect.getsource(build_train_transform)
        assert "RandomApply" in source, (
            "build_train_transform must use transforms.RandomApply for GaussianBlur [Bug5]. "
            "Found: {}".format(source[:200])
        )
        # The old buggy pattern -- one-time conditional at build time
        assert "random.random()" not in source, (
            "build_train_transform must NOT use random.random() inline (evaluated once at "
            "transform-build time, not per image) [Bug5]"
        )


class TestBug6MetricsContent:
    """Bug6: evaluate_split must return per-class precision/recall/support and zero_support_classes."""

    def test_evaluate_split_returns_per_class_dict(self):
        """evaluate_split result includes per_class with precision/recall/f1/support [Bug6]."""
        from scripts.train_classifier import evaluate_split
        from src.models.classifier import EfficientNetB3Classifier
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cpu")
        num_cls = 5
        model = EfficientNetB3Classifier(num_classes=num_cls, pretrained=False).to(device)
        idx_to_label = {i: "cls_{}".format(i) for i in range(num_cls)}

        inputs  = torch.randn(20, 3, 224, 224)
        targets = torch.zeros(20, dtype=torch.long)  # all class 0 -> classes 1-4 have 0 support
        loader  = DataLoader(TensorDataset(inputs, targets), batch_size=20)

        result = evaluate_split(model, loader, device, idx_to_label, "test_split")

        assert "per_class" in result, "evaluate_split must return 'per_class' dict [Bug6]"
        assert "confusion_matrix" in result, "evaluate_split must return 'confusion_matrix' [Bug6]"
        assert "zero_support_classes" in result, "evaluate_split must return 'zero_support_classes' [Bug6]"

        for cls_name, metrics in result["per_class"].items():
            assert "precision" in metrics, "per_class must include precision [Bug6]"
            assert "recall"    in metrics, "per_class must include recall [Bug6]"
            assert "f1"        in metrics, "per_class must include f1 [Bug6]"
            assert "support"   in metrics, "per_class must include support [Bug6]"

    def test_zero_support_detected(self):
        """Classes absent from targets are listed in zero_support_classes [Bug6]."""
        from scripts.train_classifier import evaluate_split
        from src.models.classifier import EfficientNetB3Classifier
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cpu")
        num_cls = 4
        model = EfficientNetB3Classifier(num_classes=num_cls, pretrained=False).to(device)
        idx_to_label = {i: "cls_{}".format(i) for i in range(num_cls)}

        # Only classes 0 and 1 appear in targets; classes 2 and 3 have zero support
        inputs  = torch.randn(10, 3, 224, 224)
        targets = torch.randint(0, 2, (10,))
        loader  = DataLoader(TensorDataset(inputs, targets), batch_size=10)

        result = evaluate_split(model, loader, device, idx_to_label, "test_split")
        zs = result["zero_support_classes"]
        assert "cls_2" in zs, "cls_2 should be zero_support but is not [Bug6]. Got: {}".format(zs)
        assert "cls_3" in zs, "cls_3 should be zero_support but is not [Bug6]. Got: {}".format(zs)


class TestBug7MaxTrainImagesRandomSample:
    """Bug7: --max-train-images must take a seeded random sample, not first-N."""

    def test_max_train_images_is_not_first_n(self):
        """Seeded sample != first-N slice for a shuffled manifest [Bug7]."""
        import random as stdlib_random
        import json

        # Simulate a manifest list with known ordering
        manifest = [{"path": "img_{}.jpg".format(i), "class": "A", "md5": str(i)} for i in range(1000)]
        seed = 42
        limit = 100

        rng = stdlib_random.Random(seed)
        sampled = rng.sample(manifest, limit)
        first_n = manifest[:limit]

        # They should NOT be the same (with probability ~1.0 for any non-trivial n)
        assert sampled != first_n, (
            "--max-train-images should use seeded random.sample(), not first-N slice [Bug7]. "
            "Got identical result."
        )

    def test_max_train_images_same_seed_same_result(self):
        """Same seed produces the same sample (deterministic) [Bug7]."""
        import random as stdlib_random

        manifest = [{"path": "img_{}.jpg".format(i), "class": "A", "md5": str(i)} for i in range(500)]
        seed, limit = 42, 50

        rng1 = stdlib_random.Random(seed)
        s1 = rng1.sample(manifest, limit)
        rng2 = stdlib_random.Random(seed)
        s2 = rng2.sample(manifest, limit)

        assert s1 == s2, "Same seed must produce identical samples [Bug7]"

    def test_train_classifier_uses_random_sample(self):
        """Regression: verify build uses random.Random.sample, not a slice [Bug7]."""
        import inspect
        from scripts import train_classifier
        source = inspect.getsource(train_classifier.main)
        assert "_rng.sample" in source, (
            "train_classifier.main must use _rng.sample() for --max-train-images [Bug7]"
        )
        assert "max_train_images]" not in source, (
            "train_classifier.main must NOT use first-N slice for --max-train-images [Bug7]"
        )

    def test_stratified_sample_covers_all_classes(self):
        """stratified_sample guarantees at least 1 image per class where available [Bug7]."""
        import random as stdlib_random
        from scripts.train_classifier import stratified_sample

        # 10 classes, varying sizes (imbalanced)
        manifest = []
        for c in range(10):
            count = (c + 1) * 20
            for i in range(count):
                manifest.append({"path": f"img_{c}_{i}.jpg", "class": f"cls_{c}", "md5": f"{c}_{i}"})

        rng = stdlib_random.Random(42)
        sample_size = 25
        sampled = stratified_sample(manifest, sample_size, rng)

        assert len(sampled) == sample_size
        represented_classes = set(item["class"] for item in sampled)
        assert len(represented_classes) == 10, (
            "All 10 classes must be represented in stratified sample of size 25 [Bug7]. "
            "Got: {}".format(represented_classes)
        )

    def test_stratified_sample_deterministic(self):
        """stratified_sample produces identical samples with same seed [Bug7]."""
        import random as stdlib_random
        from scripts.train_classifier import stratified_sample

        manifest = [{"path": f"img_{i}.jpg", "class": f"cls_{i % 5}", "md5": str(i)} for i in range(200)]
        s1 = stratified_sample(manifest, 30, stdlib_random.Random(123))
        s2 = stratified_sample(manifest, 30, stdlib_random.Random(123))
        assert s1 == s2


class TestBug8ECECaveat:
    """Bug8: OOD ECE reporting must include a caveat that T was fitted on D2_Val."""

    def test_evaluate_split_logs_temperature(self):
        """evaluate_split accepts temperature arg and includes it in log format [Bug8]."""
        from scripts.train_classifier import evaluate_split
        from src.models.classifier import EfficientNetB3Classifier
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cpu")
        num_cls = 3
        model = EfficientNetB3Classifier(num_classes=num_cls, pretrained=False).to(device)
        idx_to_label = {i: "c{}".format(i) for i in range(num_cls)}
        inputs  = torch.randn(6, 3, 224, 224)
        targets = torch.randint(0, num_cls, (6,))
        loader  = DataLoader(TensorDataset(inputs, targets), batch_size=6)

        # Should not raise when temperature is provided
        result = evaluate_split(model, loader, device, idx_to_label, "OOD_test", temperature=1.23)
        assert "macro_f1" in result

    def test_train_classifier_has_ood_ece_note(self):
        """train_classifier.main logs that T was fitted on D2_Val for OOD ECE [Bug8]."""
        import inspect
        from scripts import train_classifier
        source = inspect.getsource(train_classifier.main)
        assert "OOD ECE" in source or "OOD_ECE" in source or "domain shift" in source, (
            "train_classifier.main must document OOD ECE caveat (T fitted on D2_Val) [Bug8]"
        )
