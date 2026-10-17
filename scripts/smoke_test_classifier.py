"""
scripts/smoke_test_classifier.py
=================================
CPU smoke test for the C2 LeafMind classifier pipeline.

PURPOSE: Prove the full training + calibration + evaluation pipeline works correctly
using a tiny subset of data (configurable, default 200 train / 50 val / 50 test images).

This script DOES NOT train a production model -- it trains for 2 epochs on a small
subset to catch import errors, shape mismatches, and schema validation failures.

Run locally with:
    python scripts/smoke_test_classifier.py
    python scripts/smoke_test_classifier.py --help

Expected output: all steps complete without error, metrics are printed,
a smoke checkpoint is saved to checkpoints/smoke_test/.

[R1] All numbers printed are measured from actual data (tiny subset).
[R3] severity_pct = None always verified.
[I27] ASCII-only output.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.classifier import (
    EfficientNetB3Classifier,
    LeafClassifier,
    compute_class_weights,
    compute_ece,
    resolve_image_path,
)
from src.schemas.schemas import VisionOutput

# ASCII-only logging [I27]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smoke_test")

MANIFESTS_DIR  = REPO_ROOT / "config" / "manifests"
CLASS_MAP_FILE = REPO_ROOT / "config" / "class_map_d2.json"
EXCLUSION_FILE = REPO_ROOT / "config" / "exclusion_list.json"


def create_tiny_synthetic_image(w: int = 224, h: int = 224, mode: str = "RGB") -> bytes:
    """
    Create a tiny synthetic leaf-like image as bytes.
    Used when real Dataset_2 files are not accessible (CI / pure unit test context).
    [STUB] -- returns green-tinted random noise, not a real leaf image.

    Format choices:
      RGB -> JPEG
      RGBA -> PNG  (JPEG does not support alpha channel)
      CMYK -> JPEG (CMYK JPEG is supported by Pillow)
      L    -> JPEG
    """
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + 80, 0, 255)  # green tint
    img = Image.fromarray(arr, mode="RGB")
    if mode == "CMYK":
        img = img.convert("CMYK")
    elif mode == "RGBA":
        img = img.convert("RGBA")
    elif mode == "L":
        img = img.convert("L")
    buf = io.BytesIO()
    # RGBA cannot be saved as JPEG; use PNG for RGBA
    fmt = "PNG" if mode == "RGBA" else "JPEG"
    img.save(buf, format=fmt)
    return buf.getvalue()


class SyntheticD2Dataset(torch.utils.data.Dataset):
    """
    Synthetic dataset of random images for smoke testing without real data.
    Uses correct class labels from class_map_d2.json.
    [STUB] -- images are random noise, not real plant data.
    """

    def __init__(
        self,
        label_to_idx: dict,
        n_images: int = 200,
        input_size: int = 224,
        include_non_rgb: bool = True,
    ):
        self.label_to_idx = label_to_idx
        self.labels       = list(label_to_idx.keys())
        self.n_images     = n_images
        self.input_size   = input_size
        # Mix in non-RGB modes to test conversion [I9]
        modes = ["RGB"] * 8 + (["CMYK", "RGBA", "L"] if include_non_rgb else [])
        self.modes = [random.choice(modes) for _ in range(n_images)]

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, idx: int):
        label_name = self.labels[idx % len(self.labels)]
        label_idx  = self.label_to_idx[label_name]
        mode       = self.modes[idx]

        img_bytes = create_tiny_synthetic_image(
            self.input_size, self.input_size, mode=mode
        )
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")  # [I9]
        tensor = TF.to_tensor(img)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, label_idx


def make_real_dataset(
    manifest_name: str,
    data_root: Path,
    label_to_idx: dict,
    max_images: int = 50,
) -> "D2DatasetLite | None":
    """
    Try to build a small real dataset from manifests. Returns None if data unavailable.
    Falls back to SyntheticD2Dataset in smoke test.
    """
    manifest_path = MANIFESTS_DIR / f"d2_{manifest_name}.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        all_images = json.load(f)["images"]

    # Sample balanced subset
    random.seed(42)
    random.shuffle(all_images)
    subset = all_images[:max_images]

    # Verify at least one file is accessible
    if not resolve_image_path(data_root, subset[0]["path"]).exists():
        return None

    return subset


def run_smoke_test(
    data_root: Path | None,
    n_train: int = 200,
    n_val: int = 50,
    n_test: int = 50,
    epochs: int = 2,
    checkpoint_dir: Path = REPO_ROOT / "checkpoints" / "smoke_test",
):
    """Run the full smoke test pipeline."""
    logger.info("=" * 60)
    logger.info("LEAFMIND C2 SMOKE TEST")
    logger.info("=" * 60)
    logger.info("Purpose: verify pipeline correctness on tiny synthetic data.")
    logger.info("NOT a production training run [ENG assumption -- see train_classifier.py]")
    logger.info("All metrics below are from synthetic/small data -- not real [R1].")

    device = torch.device("cpu")  # smoke test is CPU-only
    seed   = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Load class map ────────────────────────────────────────────────────────
    with open(CLASS_MAP_FILE) as f:
        raw_map = json.load(f)
    idx_to_label = {int(k): v for k, v in raw_map["classes"].items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    num_classes  = len(idx_to_label)
    logger.info("Loaded %d classes from class_map_d2.json [DATA]", num_classes)

    # ── Build datasets ────────────────────────────────────────────────────────
    use_real = False
    real_train = None
    if data_root and data_root.exists():
        real_train = make_real_dataset("train", data_root, label_to_idx, n_train)
        if real_train:
            use_real = True
            logger.info("Using REAL D2 images (subset=%d) [DATA]", len(real_train))
        else:
            logger.warning("D2 data not accessible at %s -- using SYNTHETIC [STUB]", data_root)
    else:
        logger.warning("data_root not set or not found -- using SYNTHETIC images [STUB]")

    if use_real:
        from scripts.train_classifier import D2Dataset, build_eval_transform
        cfg = {"model": {"input_size": 224}, "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
        tf = build_eval_transform(cfg)
        # Use eval transform for smoke (no augmentation needed)
        real_val   = make_real_dataset("val",     data_root, label_to_idx, n_val)
        real_test  = make_real_dataset("test_id", data_root, label_to_idx, n_test)
        train_ds = D2Dataset(real_train, data_root, tf, label_to_idx)
        val_ds   = D2Dataset(real_val   or real_train[:n_val],   data_root, tf, label_to_idx)
        test_ds  = D2Dataset(real_test  or real_train[:n_test],  data_root, tf, label_to_idx)
    else:
        train_ds = SyntheticD2Dataset(label_to_idx, n_images=n_train)
        val_ds   = SyntheticD2Dataset(label_to_idx, n_images=n_val)
        test_ds  = SyntheticD2Dataset(label_to_idx, n_images=n_test)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

    # ── Class weights ─────────────────────────────────────────────────────────
    # For real data, use actual counts; for synthetic, uniform (all classes represented)
    class_counts_uniform = {i: max(1, n_train // num_classes) for i in range(num_classes)}
    class_weights = compute_class_weights(class_counts_uniform, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EfficientNetB3Classifier(
        num_classes=num_classes,
        drop_rate=0.3,
        temperature=1.5,
        pretrained=False,  # do NOT download pretrained weights in smoke test
    ).to(device)
    logger.info("Model created: EfficientNet-B3 (%d classes, pretrained=False)", num_classes)

    # ── Training (2 epochs on tiny data) ─────────────────────────────────────
    optimiser = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad()
            logits = model(inputs, calibrate=False)
            loss   = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            total_loss += loss.item()
            n_batches  += 1
        avg_loss = total_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d  avg_loss=%.4f", epoch, epochs, avg_loss)

    # ── Minimal calibration ───────────────────────────────────────────────────
    # LBFGS on synthetic data won't converge meaningfully; just verify T is set
    # and ECE can be computed. Use fixed T=1.5 for smoke test.
    temperature = 1.5  # [ENG default -- do not report as calibrated]
    logger.info(
        "Smoke calibration: T=%.2f (default, not optimised -- synthetic data [STUB])",
        temperature
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_targets, all_confs = [], [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            logits = model(inputs.to(device), calibrate=True)
            probs  = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    confs   = np.array(all_confs)
    correct = (preds == targets).astype(float)
    ece     = compute_ece(confs, correct)
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    logger.info(
        "Val (synthetic/subset): macro_F1=%.4f  ECE=%.4f  [NOT real metrics -- R1]",
        macro_f1, ece
    )

    # ── Save smoke checkpoint ─────────────────────────────────────────────────
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pth")
    with open(checkpoint_dir / "temperature.json", "w") as f:
        json.dump({"temperature": temperature, "tag": "[STUB -- smoke test default]"}, f)
    with open(checkpoint_dir / "class_map.json", "w") as f:
        json.dump({str(k): v for k, v in idx_to_label.items()}, f)
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({
            "model_version": "smoke_test",
            "num_classes": num_classes,
            "input_size": 224,
            "tag": "[STUB] Smoke test checkpoint -- do NOT use for production",
        }, f)
    logger.info("Smoke checkpoint saved to %s", checkpoint_dir)

    # ── Step 5: Load checkpoint and verify VisionOutput schema ────────────────
    logger.info("--- Schema validation ---")
    clf = LeafClassifier.from_checkpoint(checkpoint_dir, device=device)

    # Test with synthetic image (RGB)
    img_rgb  = create_tiny_synthetic_image(mode="RGB")
    out_rgb  = clf.predict(img_rgb)
    _validate_output(out_rgb, "RGB input")

    # Test with CMYK image -- must be converted to RGB [I9]
    img_cmyk = create_tiny_synthetic_image(mode="CMYK")
    out_cmyk = clf.predict(img_cmyk)
    _validate_output(out_cmyk, "CMYK input (RGB-converted)")

    # Test with RGBA image
    img_rgba = create_tiny_synthetic_image(mode="RGBA")
    out_rgba = clf.predict(img_rgba)
    _validate_output(out_rgba, "RGBA input (RGB-converted)")

    # Test with L (grayscale) image
    img_l    = create_tiny_synthetic_image(mode="L")
    out_l    = clf.predict(img_l)
    _validate_output(out_l, "L (grayscale) input (RGB-converted)")

    # Test with tiny low-resolution image -- should set low_resolution=True
    img_tiny = create_tiny_synthetic_image(w=50, h=50, mode="RGB")
    out_tiny = clf.predict(img_tiny)
    assert out_tiny.image_quality_flags.low_resolution, \
        "FAIL: low_resolution flag should be True for 50x50 image [ENG assumption]"
    logger.info("Low-resolution flag: PASS (50x50 -> low_resolution=True)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    print(f"  Train epochs:    {epochs}")
    print(f"  Synthetic data:  {not use_real}")
    print(f"  Val macro_F1:    {macro_f1:.4f} (NOT real -- synthetic data [R1])")
    print(f"  Val ECE:         {ece:.4f} (NOT calibrated -- default T [R1])")
    print(f"  VisionOutput schema: PASS (RGB, CMYK, RGBA, L inputs tested)")
    print(f"  severity_pct=None:   PASS [R3/D2]")
    print(f"  low_resolution flag: PASS (50x50 image)")
    print(f"  Checkpoint dir:  {checkpoint_dir}")
    print("=" * 60)
    print("\nNEXT: Run full GPU training on Kaggle/Colab:")
    print("  python scripts/train_classifier.py --data-root /path/to/Dataset_2 --epochs 30")
    print("=" * 60)


def _validate_output(output: VisionOutput, test_name: str):
    """Validate a VisionOutput against all C2 constraints."""
    assert isinstance(output, VisionOutput), f"Output is not a VisionOutput: {type(output)}"
    # R3: severity_pct ALWAYS None
    assert output.severity_pct is None, f"[{test_name}] FAIL: severity_pct must be None [R3]"
    assert output.severity_note, f"[{test_name}] FAIL: severity_note must be non-empty [R3]"
    # Schema: confidence in [0,1]
    assert 0.0 <= output.confidence <= 1.0, f"[{test_name}] FAIL: confidence out of bounds"
    # Schema: top_k_probs non-empty
    assert len(output.top_k_probs) >= 1, f"[{test_name}] FAIL: top_k_probs is empty"
    # Schema: temperature > 0
    assert output.temperature > 0, f"[{test_name}] FAIL: temperature <= 0"
    # Schema: model_version non-empty
    assert output.model_version, f"[{test_name}] FAIL: model_version is empty"
    logger.info("Schema validation PASS: %s", test_name)


def main():
    parser = argparse.ArgumentParser(
        description="LeafMind C2 smoke test -- CPU, tiny data, verifies pipeline correctness"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Path to Dataset_2 directory. If not set, uses synthetic images [STUB]."
    )
    parser.add_argument(
        "--n-train", type=int, default=200,
        help="Number of training images to use (default=200)"
    )
    parser.add_argument(
        "--n-val", type=int, default=50,
        help="Number of validation images (default=50)"
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Epochs to train (default=2 for smoke test)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default=str(REPO_ROOT / "checkpoints" / "smoke_test"),
        help="Directory for smoke checkpoint output"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else None

    run_smoke_test(
        data_root=data_root,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_val,
        epochs=args.epochs,
        checkpoint_dir=Path(args.checkpoint_dir),
    )


if __name__ == "__main__":
    main()
