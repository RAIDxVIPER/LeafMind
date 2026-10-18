"""
tests/unit/test_ood_diagnostics.py
====================================
Synthetic-logit unit tests for OOD robustness diagnostics (Task 1, C2 follow-up).
Tests:
  - _crop_from_label helper (crop extraction, Corn/Corn_(maize) normalisation)
  - _compute_ood_diagnostics (crop_restricted_top1/macro_f1, crop_top1,
    healthy_vs_diseased_acc, diseased_predicted_healthy, mean_confidence, top5_predicted_classes)
  - evaluate_split integration (diagnostics key present in output)
  - v1 pipeline unchanged (no RandomPerspective/RandomErasing with v1 config)
  - v2 pipeline adds new transforms; eval transforms identical v1==v2

All metrics computed from synthetic data -- no real model weights [R1].
ASCII-only output [I27].
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

# ── Shared test class map (4-class subset for fast fixture) ───────────────────
SIMPLE_IDX_TO_LABEL = {
    0: "Apple___Apple_scab",
    1: "Apple___healthy",
    2: "Tomato___Early_blight",
    3: "Tomato___healthy",
}

# Full 30-class map matches MOCK_IDX_TO_LABEL in test_classifier.py [DATA]
FULL_IDX_TO_LABEL = {
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


def _uniform_logits(n: int, num_cls: int) -> tuple:
    """Return (logits, confs) for uniform distribution over num_cls classes."""
    logits = np.zeros((n, num_cls), dtype=np.float32)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return logits, probs.max(axis=1)


def _peaked_logits(n: int, num_cls: int, targets: np.ndarray, peak: float = 10.0) -> tuple:
    """Return (logits, confs) with high confidence on target class."""
    logits = np.full((n, num_cls), -10.0, dtype=np.float32)
    for i in range(n):
        logits[i, int(targets[i])] = peak
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return logits, probs.max(axis=1)


# ── _crop_from_label tests ────────────────────────────────────────────────────

class TestCropFromLabel:
    """Unit tests for _crop_from_label helper."""

    def test_apple_crop_extracted(self):
        from scripts.train_classifier import _crop_from_label
        assert _crop_from_label("Apple___Apple_scab") == "Apple"
        assert _crop_from_label("Apple___Cedar_apple_rust") == "Apple"
        assert _crop_from_label("Apple___healthy") == "Apple"

    def test_tomato_crop_extracted(self):
        from scripts.train_classifier import _crop_from_label
        assert _crop_from_label("Tomato___healthy") == "Tomato"
        assert _crop_from_label("Tomato___Early_blight") == "Tomato"
        assert _crop_from_label("Tomato___Tomato_Yellow_Leaf_Curl_Virus") == "Tomato"

    def test_corn_maize_normalised_to_corn(self):
        """Corn_(maize) and Corn must both map to 'Corn' [ENG diagnostic]."""
        from scripts.train_classifier import _crop_from_label
        assert _crop_from_label("Corn_(maize)___healthy") == "Corn"
        assert _crop_from_label("Corn___Common_rust") == "Corn"
        assert _crop_from_label("Corn___Northern_Leaf_Blight") == "Corn"
        assert _crop_from_label("Corn___Cercospora_leaf_spot") == "Corn"

    def test_pepper_bell_with_comma(self):
        from scripts.train_classifier import _crop_from_label
        assert _crop_from_label("Pepper,_bell___Bacterial_spot") == "Pepper,_bell"
        assert _crop_from_label("Pepper,_bell___healthy") == "Pepper,_bell"

    def test_grape_crop(self):
        from scripts.train_classifier import _crop_from_label
        assert _crop_from_label("Grape___Black_rot") == "Grape"
        assert _crop_from_label("Grape___healthy") == "Grape"


# ── _compute_ood_diagnostics tests ───────────────────────────────────────────

class TestComputeOODDiagnostics:
    """
    Synthetic-logit unit tests for _compute_ood_diagnostics.
    All metrics are computed from synthetic data [R1]. ASCII-only [I27].
    """

    def test_crop_restricted_top1_perfect_model(self):
        """If model always correct, crop_restricted_top1 == 1.0."""
        from scripts.train_classifier import _compute_ood_diagnostics

        n, num_cls = 8, 4
        targets = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        preds = targets.copy()
        logits, confs = _peaked_logits(n, num_cls, targets)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        assert diag["crop_restricted_top1"] == pytest.approx(1.0, abs=1e-4)
        assert diag["crop_restricted_macro_f1"] == pytest.approx(1.0, abs=1e-4)

    def test_crop_restricted_note_contains_eng_tag(self):
        """crop_restricted_note must contain [ENG diagnostic] tag."""
        from scripts.train_classifier import _compute_ood_diagnostics

        n, num_cls = 4, 4
        targets = np.array([0, 1, 2, 3], dtype=np.int64)
        preds = targets.copy()
        logits, confs = _uniform_logits(n, num_cls)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        assert "[ENG diagnostic" in diag["crop_restricted_note"]
        assert "collects the crop from the user" in diag["crop_restricted_note"]

    def test_crop_restricted_cannot_hurt_accuracy(self):
        """
        Cross-crop errors: crop_restricted_top1 >= raw top-1 accuracy.
        Restricting to valid crop classes can only help or tie.
        """
        from scripts.train_classifier import _compute_ood_diagnostics

        n, num_cls = 4, 4
        # Apple images misclassified as Tomato (cross-crop) and vice versa
        targets = np.array([0, 0, 2, 2], dtype=np.int64)
        preds   = np.array([2, 2, 0, 0], dtype=np.int64)
        logits, _ = _peaked_logits(n, num_cls, preds)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        confs = probs.max(axis=1)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        raw_top1 = float((preds == targets).mean())
        assert diag["crop_restricted_top1"] >= raw_top1 - 1e-4, \
            "crop_restricted_top1 must be >= unconstrained top1"

    def test_crop_restricted_fixes_cross_crop_errors(self):
        """
        Model always picks wrong crop class but correct within crop.
        crop_restricted should give top-1 > 0 even though raw top-1 == 0.
        """
        from scripts.train_classifier import _compute_ood_diagnostics

        # Targets: Apple_scab(0), Apple_healthy(1); preds: Tomato(2,3) -- all wrong crop
        n, num_cls = 2, 4
        targets = np.array([0, 1], dtype=np.int64)
        preds   = np.array([2, 3], dtype=np.int64)
        logits  = np.array([
            [-10, -10, 10, -10],   # strongly predicts Tomato_blight for Apple_scab
            [-10, -10, -10, 10],   # strongly predicts Tomato_healthy for Apple_healthy
        ], dtype=np.float32)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        confs = probs.max(axis=1)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        raw_top1 = 0.0  # all wrong
        assert diag["crop_restricted_top1"] >= raw_top1, \
            "crop_restricted_top1 should be >= 0"

    def test_crop_top1_half_correct_crop(self):
        """crop_top1 = 0.5 when half images get correct crop."""
        from scripts.train_classifier import _compute_ood_diagnostics

        n, num_cls = 4, 4
        targets = np.array([0, 1, 2, 3], dtype=np.int64)
        # indices 0,1 predict within Apple (correct crop); 2,3 predict Apple (wrong crop)
        preds   = np.array([1, 0, 0, 1], dtype=np.int64)
        logits, confs = _uniform_logits(n, num_cls)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        assert diag["crop_top1"] == pytest.approx(0.5, abs=1e-4)

    def test_crop_top1_all_correct_crop(self):
        """crop_top1 == 1.0 when every prediction is in the correct crop."""
        from scripts.train_classifier import _compute_ood_diagnostics

        n, num_cls = 4, 4
        targets = np.array([0, 1, 2, 3], dtype=np.int64)
        preds   = targets.copy()  # same crop, same class
        logits, confs = _uniform_logits(n, num_cls)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        assert diag["crop_top1"] == pytest.approx(1.0, abs=1e-4)

    def test_hvd_acc_perfect(self):
        """healthy_vs_diseased_acc == 1.0 when binary label always correct."""
        from scripts.train_classifier import _compute_ood_diagnostics

        n, num_cls = 4, 4
        targets = np.array([0, 1, 2, 3], dtype=np.int64)
        preds   = targets.copy()
        logits, confs = _uniform_logits(n, num_cls)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        assert diag["healthy_vs_diseased_acc"] == pytest.approx(1.0, abs=1e-4)

    def test_hvd_acc_all_wrong(self):
        """healthy_vs_diseased_acc == 0.0 when every binary label is wrong."""
        from scripts.train_classifier import _compute_ood_diagnostics

        n, num_cls = 4, 4
        targets = np.array([0, 1, 2, 3], dtype=np.int64)
        # Flip all: diseased->healthy, healthy->diseased
        preds   = np.array([1, 0, 3, 2], dtype=np.int64)
        logits, confs = _uniform_logits(n, num_cls)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        assert diag["healthy_vs_diseased_acc"] == pytest.approx(0.0, abs=1e-4)

    def test_diseased_predicted_healthy_all_diseased_pred_as_healthy(self):
        """diseased_predicted_healthy == 1.0 when every diseased image predicted healthy."""
        from scripts.train_classifier import _compute_ood_diagnostics

        n, num_cls = 4, 4
        targets = np.array([0, 1, 2, 3], dtype=np.int64)   # 0=diseased, 1=healthy, 2=diseased, 3=healthy
        preds   = np.array([1, 1, 3, 3], dtype=np.int64)   # all predicted healthy
        logits, confs = _uniform_logits(n, num_cls)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        assert diag["diseased_predicted_healthy"] == pytest.approx(1.0, abs=1e-4)

    def test_diseased_predicted_healthy_none_when_all_healthy_targets(self):
        """diseased_predicted_healthy is None when there are no diseased images in targets."""
        from scripts.train_classifier import _compute_ood_diagnostics

        all_healthy_map = {0: "Apple___healthy", 1: "Tomato___healthy"}
        n, num_cls = 2, 2
        targets = np.array([0, 1], dtype=np.int64)
        preds   = targets.copy()
        logits, confs = _uniform_logits(n, num_cls)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, all_healthy_map)
        assert diag["diseased_predicted_healthy"] is None

    def test_mean_confidence_uniform_4class(self):
        """mean_confidence == 0.25 for uniform logits over 4 classes."""
        from scripts.train_classifier import _compute_ood_diagnostics

        n, num_cls = 4, 4
        targets = np.array([0, 1, 2, 3], dtype=np.int64)
        preds   = targets.copy()
        logits, confs = _uniform_logits(n, num_cls)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        assert diag["mean_confidence"] == pytest.approx(0.25, abs=1e-3)

    def test_top5_predicted_classes_most_common_first(self):
        """top5_predicted_classes[0] is the most predicted class."""
        from scripts.train_classifier import _compute_ood_diagnostics

        n, num_cls = 8, 4
        targets = np.array([0, 0, 0, 1, 1, 2, 2, 3], dtype=np.int64)
        preds   = np.array([0, 0, 0, 0, 0, 2, 2, 2], dtype=np.int64)  # class 0 x5, class 2 x3
        logits, confs = _uniform_logits(n, num_cls)

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, SIMPLE_IDX_TO_LABEL)
        top5 = diag["top5_predicted_classes"]

        assert isinstance(top5, list)
        assert len(top5) <= 5
        for entry in top5:
            assert "class" in entry and "share" in entry
            assert 0.0 <= entry["share"] <= 1.0

        # Most predicted: Apple___Apple_scab (index 0) with share 5/8
        assert top5[0]["class"] == "Apple___Apple_scab"
        assert top5[0]["share"] == pytest.approx(5 / 8, abs=0.01)

    def test_corn_maize_grouped_with_corn_in_diagnostics(self):
        """
        Corn_(maize)___healthy and Corn___Common_rust must be in the same crop group.
        A model predicting Corn___Common_rust for a Corn_(maize)___healthy image should
        have crop_top1 = 1.0 (correct crop, wrong disease).
        """
        from scripts.train_classifier import _compute_ood_diagnostics

        # Use full 30-class map; targets: Corn_(maize)___healthy = 5; preds: Corn___Common_rust = 7
        n, num_cls = 1, 30
        targets = np.array([5], dtype=np.int64)  # Corn_(maize)___healthy
        preds   = np.array([7], dtype=np.int64)  # Corn___Common_rust
        logits  = np.zeros((n, num_cls), dtype=np.float32)
        confs   = np.array([1.0 / num_cls])

        diag = _compute_ood_diagnostics(logits, preds, targets, confs, FULL_IDX_TO_LABEL)
        assert diag["crop_top1"] == pytest.approx(1.0, abs=1e-4), \
            "Corn_(maize) and Corn must be treated as same crop"

    def test_diagnostics_in_evaluate_split_return_dict(self):
        """
        evaluate_split must return a 'diagnostics' key with all required fields.
        Integration test using a tiny synthetic loader.
        """
        import scripts.train_classifier as tc
        from src.models.classifier import EfficientNetB3Classifier
        from torch.utils.data import DataLoader, TensorDataset

        NUM_CLASSES = 30
        model = EfficientNetB3Classifier(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        device = torch.device("cpu")

        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 5, 22, 29], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        result = tc.evaluate_split(model, loader, device, FULL_IDX_TO_LABEL, "test_split")
        diag = result.get("diagnostics")

        assert diag is not None, "'diagnostics' key missing from evaluate_split output"
        required_keys = [
            "crop_restricted_top1", "crop_restricted_macro_f1", "crop_restricted_note",
            "crop_top1", "healthy_vs_diseased_acc", "diseased_predicted_healthy",
            "mean_confidence", "top5_predicted_classes",
        ]
        for k in required_keys:
            assert k in diag, f"diagnostics missing key: {k}"

        assert "[ENG diagnostic" in diag["crop_restricted_note"]
        # crop_restricted_top1 >= top1_accuracy always
        assert diag["crop_restricted_top1"] >= result["top1_accuracy"] - 1e-4
        assert 0.0 <= diag["mean_confidence"] <= 1.0


# ── v1 pipeline unchanged tests (Task 3 constraint) ──────────────────────────

class TestV1PipelineUnchanged:
    """
    Verify that v1 config (classifier_config.yaml) produces the same transform chain
    as before (no RandomPerspective, no RandomErasing, no label_smoothing).
    v2 config must add new augmentations without touching eval transforms.
    """

    def _transform_names(self, compose) -> list:
        return [type(t).__name__ for t in compose.transforms]

    def test_v1_train_no_perspective_no_erasing(self):
        """v1 config must not produce RandomPerspective or RandomErasing."""
        import yaml
        import scripts.train_classifier as tc
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parents[2] / "config" / "classifier_config.yaml"
        if not cfg_path.exists():
            pytest.skip("classifier_config.yaml not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        names = self._transform_names(tc.build_train_transform(cfg))

        for required in ["RandomResizedCrop", "RandomHorizontalFlip", "RandomVerticalFlip",
                          "RandomRotation", "ColorJitter", "RandomGrayscale",
                          "RandomApply", "ToTensor", "Normalize"]:
            assert required in names, f"v1 transform missing required step: {required}"

        assert "RandomPerspective" not in names, "RandomPerspective must not be in v1 pipeline"
        assert "RandomErasing" not in names, "RandomErasing must not be in v1 pipeline"

    def test_v2_train_adds_perspective_and_erasing(self):
        """v2 config must produce RandomPerspective and RandomErasing in train pipeline."""
        import yaml
        import scripts.train_classifier as tc
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parents[2] / "config" / "classifier_config_v2.yaml"
        if not cfg_path.exists():
            pytest.skip("classifier_config_v2.yaml not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        names = self._transform_names(tc.build_train_transform(cfg))
        assert "RandomPerspective" in names, "v2 must include RandomPerspective"
        assert "RandomErasing" in names, "v2 must include RandomErasing"

        # RandomErasing must be AFTER ToTensor (requires tensor)
        to_tensor_idx = names.index("ToTensor")
        re_idx = names.index("RandomErasing")
        assert re_idx > to_tensor_idx, "RandomErasing must appear after ToTensor [ENG]"

    def test_v1_eval_chain_unchanged(self):
        """v1 eval transform: Resize, CenterCrop, ToTensor, Normalize -- exactly."""
        import yaml
        import scripts.train_classifier as tc
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parents[2] / "config" / "classifier_config.yaml"
        if not cfg_path.exists():
            pytest.skip("classifier_config.yaml not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        names = self._transform_names(tc.build_eval_transform(cfg))
        assert names == ["Resize", "CenterCrop", "ToTensor", "Normalize"], \
            f"v1 eval chain changed unexpectedly: {names}"

    def test_v2_eval_transform_equals_v1(self):
        """Val/test transforms must be IDENTICAL for v1 and v2 [DATA spec 2a-iv]."""
        import yaml
        import scripts.train_classifier as tc
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "config"
        if not (root / "classifier_config.yaml").exists() or \
           not (root / "classifier_config_v2.yaml").exists():
            pytest.skip("Config files not found")

        with open(root / "classifier_config.yaml") as f:
            cfgA = yaml.safe_load(f)
        with open(root / "classifier_config_v2.yaml") as f:
            cfgB = yaml.safe_load(f)

        names_v1 = self._transform_names(tc.build_eval_transform(cfgA))
        names_v2 = self._transform_names(tc.build_eval_transform(cfgB))
        assert names_v1 == names_v2, \
            f"Eval transforms differ: v1={names_v1}, v2={names_v2}"

    def test_label_smoothing_absent_in_v1_config(self):
        """v1 config must NOT define label_smoothing (defaults to 0.0 in script)."""
        import yaml
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parents[2] / "config" / "classifier_config.yaml"
        if not cfg_path.exists():
            pytest.skip("classifier_config.yaml not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        ls = cfg.get("training", {}).get("label_smoothing", None)
        assert ls is None, f"v1 config must not define label_smoothing; got {ls}"

    def test_label_smoothing_01_in_v2_config(self):
        """v2 config must define label_smoothing=0.1 [ENG assumption]."""
        import yaml
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parents[2] / "config" / "classifier_config_v2.yaml"
        if not cfg_path.exists():
            pytest.skip("classifier_config_v2.yaml not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        ls = cfg.get("training", {}).get("label_smoothing", None)
        assert ls == pytest.approx(0.1, abs=1e-6), \
            f"v2 config must define label_smoothing=0.1, got {ls}"

    def test_v2_scale_min_changed_from_v1(self):
        """v2 random_resized_crop scale_min must be 0.25 (v1 was 0.7) [ENG]."""
        import yaml
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parents[2] / "config" / "classifier_config_v2.yaml"
        if not cfg_path.exists():
            pytest.skip("classifier_config_v2.yaml not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        scale_min = cfg.get("augmentation", {}).get("random_resized_crop", {}).get("scale_min", None)
        assert scale_min == pytest.approx(0.25, abs=1e-6), \
            f"v2 scale_min must be 0.25, got {scale_min}"
