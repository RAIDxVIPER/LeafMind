"""
scripts/calibrate.py
=====================
Post-training temperature scaling calibration for the LeafMind classifier.

PURPOSE:
    After full GPU training completes (scripts/train_classifier.py), run this
    script to optimise the single-scalar temperature T on the D2_Val split.
    The calibrated T is written to:
        {checkpoint_dir}/temperature.json
        {checkpoint_dir}/calibration_metrics.json

    This NEVER touches D2_Train or D2_Test_* -- calibration uses ONLY D2_Val [ENG].

METHOD:
    Temperature scaling (Guo et al., 2017 -- methodology only).
    T is optimised via LBFGS to minimise cross-entropy NLL on D2_Val logits.
    Target ECE < 0.05 is an [ENG assumption], not a literature-derived threshold
    (see I24 / plan.md threshold register).

USAGE (after GPU training):
    python scripts/calibrate.py --checkpoint-dir checkpoints/classifier --data-root /path/to/Dataset_2

    Outputs:
        checkpoints/classifier/temperature.json          -- {"temperature": <float>}
        checkpoints/classifier/calibration_metrics.json  -- pre/post ECE, NLL

DO NOT RUN FULL TRAINING LOCALLY. Training is done on Kaggle/Colab GPU.
Calibration is cheap and CAN be run locally on CPU after training completes.

Coding rules:
    [R3]  severity_pct = None always (never computed here)
    [D14] D2_Train images NEVER used in calibration
    [I9]  RGB conversion forced for all image modes
    [I27] ASCII-only logging and output
    [R1]  All printed numbers are measured from actual data

Tags: [ENG] [DATA]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.classifier import (
    EfficientNetB3Classifier,
    TemperatureScaler,
    compute_ece,
    fit_temperature,
    resolve_image_path,   # Bug1: cross-platform backslash fix
)

# ASCII-only logging [I27]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("calibrate")

MANIFESTS_DIR = REPO_ROOT / "config" / "manifests"
CLASS_MAP_FILE = REPO_ROOT / "config" / "class_map_d2.json"


# ── Dataset ───────────────────────────────────────────────────────────────────

class D2ValDataset(Dataset):
    """
    Minimal dataset for D2_Val calibration.

    Reads images listed in config/manifests/d2_val.json.
    D2_Train manifests are NEVER read here [D14].
    RGBA images are saved as PNG in the manifest; all other modes as JPEG.
    RGB conversion is forced at load time [I9].
    """

    def __init__(
        self,
        manifest: list,
        data_root: Path,
        label_to_idx: dict,
        input_size: int = 224,
    ):
        import torchvision.transforms as T
        self.entries = manifest
        self.data_root = data_root
        self.label_to_idx = label_to_idx
        self.input_size = input_size
        # Eval transform: resize + centre-crop + normalise -- NO augmentation [DATA spec 2a-iv]
        self.transform = T.Compose([
            T.Resize(int(input_size * 256 / 224)),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        # Bug1: resolve_image_path converts Windows backslashes for Linux/Mac compat
        img_path = resolve_image_path(self.data_root, entry["path"])
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                logger.debug("Converting %s from %s to RGB [I9]", img_path.name, img.mode)
                img = img.convert("RGB")
        except Exception as exc:
            logger.warning("Cannot load %s: %s -- using blank image", img_path, exc)
            img = Image.new("RGB", (self.input_size, self.input_size), color=(128, 128, 128))
        tensor = self.transform(img)
        # Bug2: key is 'class', not 'label' -- manifests use 'class' [DATA]
        label_idx = self.label_to_idx.get(entry["class"], 0)
        return tensor, label_idx


# ── Logit extraction ──────────────────────────────────────────────────────────

@torch.no_grad()
def extract_logits(
    model: EfficientNetB3Classifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Run model in eval mode WITHOUT temperature scaling to collect raw logits.
    Returns (all_logits, all_labels) as CPU tensors.
    calibrate=False so we get backbone logits before T scaling.
    """
    model.eval()
    all_logits, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs, calibrate=False)  # raw logits -- no T scaling
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# ── Temperature optimisation ──────────────────────────────────────────────────

def optimise_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    init_temp: float = 1.5,
    max_iter: int = 50,
) -> float:
    """
    Optimise temperature scalar T via fit_temperature (shared implementation).
    Method: Guo et al. (2017) -- single-scalar post-hoc calibration.
    """
    return fit_temperature(logits, labels, max_iter=max_iter)



# ── ECE helpers ───────────────────────────────────────────────────────────────

def ece_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    n_bins: int = 15,
) -> float:
    """Compute ECE after applying temperature T to raw logits."""
    scaled = logits / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=1).numpy()
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    correct = (preds == labels.numpy()).astype(float)
    return compute_ece(confs, correct, n_bins=n_bins)


# ── Main calibration routine ──────────────────────────────────────────────────

def run_calibration(
    checkpoint_dir: Path,
    data_root: Path,
    batch_size: int = 64,
    max_iter: int = 50,
    n_bins: int = 15,
    device=None,
) -> dict:
    """
    Full calibration pipeline.

    1. Load trained model from checkpoint_dir.
    2. Load D2_Val manifest (NEVER D2_Train) [D14].
    3. Extract raw logits (no temperature scaling).
    4. Optimise T via LBFGS.
    5. Compute pre/post ECE and NLL.
    6. Write temperature.json and calibration_metrics.json.

    Returns metrics dict for inspection.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load class map
    with open(CLASS_MAP_FILE) as f:
        raw_map = json.load(f)
    idx_to_label = {int(k): v for k, v in raw_map["classes"].items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    num_classes = len(idx_to_label)
    logger.info("Class map loaded: %d classes [DATA]", num_classes)

    # Load D2_Val manifest -- D2_Train is NEVER read here [D14]
    manifest_path = MANIFESTS_DIR / "d2_val.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "D2_Val manifest not found: {}\n"
            "Run scripts/build_split_manifests.py first.".format(manifest_path)
        )
    with open(manifest_path) as f:
        val_data = json.load(f)
    val_entries = val_data["images"]
    logger.info(
        "D2_Val manifest loaded: %d images [DATA -- D2 ONLY, D14 enforced]",
        len(val_entries),
    )

    # Dataset and DataLoader
    val_ds = D2ValDataset(val_entries, data_root, label_to_idx)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Load model -- prefer model_best.pth if present (Bug4)
    best_path = checkpoint_dir / "model_best.pth"
    model_path = best_path if best_path.exists() else checkpoint_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            "model.pth not found in {}.\n"
            "Run full GPU training first (scripts/train_classifier.py).".format(checkpoint_dir)
        )
    if best_path.exists():
        logger.info("Loading best checkpoint: %s [Bug4]", model_path)
    else:
        logger.warning(
            "model_best.pth not found; falling back to model.pth. "
            "Calibration may be sub-optimal if final epoch was not the best."
        )
    model = EfficientNetB3Classifier(num_classes=num_classes, pretrained=False).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded from %s", model_path)

    # Extract raw logits
    logger.info("Extracting raw logits from D2_Val (%d images)...", len(val_ds))
    logits, labels = extract_logits(model, val_loader, device)
    logger.info("Logits extracted: shape=%s", list(logits.shape))

    # Pre-calibration metrics
    pre_ece = ece_from_logits(logits, labels, temperature=1.0, n_bins=n_bins)
    pre_nll = float(nn.CrossEntropyLoss()(logits, labels).item())
    logger.info("Pre-calibration:  ECE=%.4f  NLL=%.4f", pre_ece, pre_nll)

    # Optimise temperature
    logger.info("Optimising temperature (LBFGS, max_iter=%d)...", max_iter)
    opt_temp = optimise_temperature(logits, labels, max_iter=max_iter)
    logger.info("Optimised T=%.4f", opt_temp)

    # Post-calibration metrics
    post_ece = ece_from_logits(logits, labels, temperature=opt_temp, n_bins=n_bins)
    post_nll = float(nn.CrossEntropyLoss()(logits / max(opt_temp, 1e-6), labels).item())
    logger.info("Post-calibration: ECE=%.4f  NLL=%.4f", post_ece, post_nll)

    ece_target = 0.05  # [ENG assumption -- see I24 / plan.md threshold register]
    if post_ece < ece_target:
        logger.info(
            "ECE target ACHIEVED: %.4f < %.2f [ENG assumption -- I24]", post_ece, ece_target
        )
    else:
        logger.warning(
            "ECE target MISSED: %.4f >= %.2f [ENG assumption -- I24]. "
            "Model may be poorly calibrated.",
            post_ece, ece_target,
        )

    # Write outputs
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    temp_out = checkpoint_dir / "temperature.json"
    with open(temp_out, "w") as f:
        json.dump({
            "temperature": opt_temp,
            "tag": "[ENG] temperature scaling, Guo et al. 2017 method",
        }, f, indent=2)
    logger.info("Written: %s  (T=%.4f)", temp_out, opt_temp)

    # Bug4: update model.scaler.temperature and re-save model.pth so state dict
    # is consistent with temperature.json (both are authoritative together).
    model.scaler.temperature.data.fill_(opt_temp)
    torch.save(model.state_dict(), checkpoint_dir / "model.pth")
    logger.info("model.pth re-saved with calibrated T=%.4f in state dict [Bug4]", opt_temp)

    metrics = {
        "pre_calibration": {
            "ece": round(pre_ece, 6),
            "nll": round(pre_nll, 6),
            "temperature": 1.0,
        },
        "post_calibration": {
            "ece": round(post_ece, 6),
            "nll": round(post_nll, 6),
            "temperature": round(opt_temp, 6),
        },
        "ece_target": ece_target,
        "ece_target_tag": "[ENG assumption -- not literature-derived, see I24]",
        "ece_target_met": bool(post_ece < ece_target),
        "val_images": len(val_ds),
        "n_bins": n_bins,
        "dataset": "D2_Val [DATA -- D2 only, D14 enforced]",
        "severity_pct": None,
    }

    metrics_out = checkpoint_dir / "calibration_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Written: %s", metrics_out)

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print("  Val images (D2_Val):   {}  [DATA]".format(len(val_ds)))
    print("  Pre-calibration  ECE:  {:.4f}".format(pre_ece))
    print("  Post-calibration ECE:  {:.4f}".format(post_ece))
    print("  Optimised T:           {:.4f}".format(opt_temp))
    print("  ECE target (<0.05):    {} [ENG assumption -- I24]".format(
        "MET" if post_ece < 0.05 else "MISSED"
    ))
    print("  Output:  {}".format(temp_out))
    print("  Metrics: {}".format(metrics_out))
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Copy checkpoints/classifier/ back to your local machine.")
    print("  2. Run: python -m pytest tests/ -q")
    print("=" * 60)

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "LeafMind post-training temperature scaling calibration. "
            "Runs ONLY on D2_Val -- never D2_Train [D14]. "
            "Run after full GPU training is complete."
        )
    )
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default=str(REPO_ROOT / "checkpoints" / "classifier"),
        help="Directory containing model.pth (output of train_classifier.py)",
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Absolute path to the Dataset_2 directory",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    data_root = Path(args.data_root)

    if not data_root.exists():
        logger.error("data-root does not exist: %s", data_root)
        sys.exit(1)

    run_calibration(
        checkpoint_dir=ckpt_dir,
        data_root=data_root,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
