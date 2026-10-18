"""
scripts/train_classifier.py
============================
Full EfficientNet-B3 training script for LeafMind C2 (Vision Classifier).

Designed for GPU execution on Kaggle or Google Colab.
DO NOT run locally with full data -- use scripts/smoke_test_classifier.py for local testing.

Usage (GPU, e.g. Kaggle/Colab):
    python scripts/train_classifier.py --data-root /path/to/Dataset_2 --epochs 30

Key design decisions (see config/classifier_config.yaml for all values):
  [D14/R13] D1 images NEVER enter the classifier training pipeline
  [ENG]     Weighted cross-entropy loss handles ~35.4x class imbalance
  [ENG]     Augmentation targets lab-to-field domain gap (ColorJitter, blur, etc.)
  [ENG]     Temperature scaling calibrated on D2_Val after training
  [D16]     OOD evaluation uses full deduped Test_OOD as primary;
            exclusion-filtered (283 images) reported as secondary only
  [R1]      No fabricated numbers -- all metrics computed from actual data
  [R3]      severity_pct is ALWAYS None -- never computed or stored
  [I27]     ASCII-only console output

All threshold values are [ENG assumption] and tagged in code comments.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Project imports (repo root must be on PYTHONPATH)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.classifier import (
    EfficientNetB3Classifier,
    compute_class_weights,
    compute_ece,
    fit_temperature,
    resolve_image_path,   # Bug1: cross-platform backslash fix
)

# ── ASCII-only logging [I27] ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_classifier")

# ── Paths (all relative, resolved at runtime -- no absolute paths in code) ────
MANIFESTS_DIR = REPO_ROOT / "config" / "manifests"
CLASS_MAP_FILE = REPO_ROOT / "config" / "class_map_d2.json"
EXCLUSION_FILE = REPO_ROOT / "config" / "exclusion_list.json"


def stratified_sample(images: list, max_n: int, rng: random.Random) -> list:
    """
    Seeded stratified random sample that guarantees at least 1 image per class
    where available, up to max_n total images [Bug7].
    Uses rng.sample internally.
    """
    if len(images) <= max_n:
        return list(images)

    from collections import defaultdict
    by_class = defaultdict(list)
    for img in images:
        by_class[img["class"]].append(img)

    classes = sorted(by_class.keys())
    for cls in classes:
        rng.shuffle(by_class[cls])

    sampled = []
    if max_n >= len(classes):
        for cls in classes:
            if by_class[cls]:
                sampled.append(by_class[cls].pop())
        remaining_pool = []
        for cls in classes:
            remaining_pool.extend(by_class[cls])
        needed = max_n - len(sampled)
        if needed > 0 and remaining_pool:
            sampled.extend(rng.sample(remaining_pool, min(needed, len(remaining_pool))))
    else:
        selected_classes = rng.sample(classes, max_n)
        for cls in selected_classes:
            sampled.append(by_class[cls].pop())

    rng.shuffle(sampled)
    return sampled


# ── Dataset ───────────────────────────────────────────────────────────────────

class D2Dataset(Dataset):
    """
    Dataset backed by the deduplicated split manifests from C1.

    Images are sourced from D2 (PlantVillage + PlantDoc classification dataset) ONLY.
    D1 images are NEVER used in this dataset [D14/R13].
    RGB conversion is forced at load time to handle CMYK/RGBA/L modes [I9].
    """

    def __init__(
        self,
        image_list: list,   # list of {"path": str, "class": str, "md5": str}
        data_root: Path,
        transform=None,
        label_to_idx: dict = None,
    ):
        self.image_list   = image_list
        self.data_root    = Path(data_root)
        self.transform    = transform
        self.label_to_idx = label_to_idx or {}

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int):
        entry      = self.image_list[idx]
        # Bug1: resolve_image_path converts Windows backslashes to forward slashes
        img_path   = resolve_image_path(self.data_root, entry["path"])
        label_name = entry["class"]
        label_idx  = self.label_to_idx[label_name]

        # Force RGB -- handles CMYK, RGBA, L [I9]
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as exc:
            # Bug1: do NOT silently substitute blank image -- count failures
            if not hasattr(self, "_load_failures"):
                self._load_failures = []
            self._load_failures.append(str(img_path))
            if len(self._load_failures) <= 5:
                logger.error("Image load failure %d: %s -- %s", len(self._load_failures), img_path, exc)
            threshold = max(1, int(len(self.image_list) * 0.005))
            if len(self._load_failures) > threshold:
                raise RuntimeError(
                    "Load failures ({}) exceed 0.5%% of dataset ({}). "
                    "First 5 failing paths: {}".format(
                        len(self._load_failures),
                        len(self.image_list),
                        self._load_failures[:5],
                    )
                )
            img = Image.new("RGB", (224, 224), color=0)  # sentinel only

        if self.transform:
            img = self.transform(img)

        return img, label_idx


# ── Augmentation transforms ───────────────────────────────────────────────────

def build_train_transform(cfg: dict) -> transforms.Compose:
    """
    Training augmentation pipeline targeting lab-to-field domain gap [ENG spec 2a-iv].
    Applied ONLY to training images -- never to val/test [DATA].
    """
    aug = cfg.get("augmentation", {})
    cj  = aug.get("color_jitter", {})
    rrc = aug.get("random_resized_crop", {})
    blur_k = aug.get("gaussian_blur_kernel", [3, 7])
    # kernel must be odd; choose one value in range
    kernel = blur_k[0] if blur_k[0] % 2 == 1 else blur_k[0] + 1

    input_size = cfg.get("model", {}).get("input_size", 224)
    resize_to  = int(input_size * 256 / 224)

    # v2 augmentation fields (default p=0.0 = disabled = v1 behaviour) [ENG]
    perspective_p   = float(aug.get("random_perspective_p", 0.0))
    random_erase_p  = float(aug.get("random_erasing_p", 0.0))

    # Build transform list; RandomPerspective before ToTensor, RandomErasing after [ENG]
    tfm_list = [
        # RandomResizedCrop first -- scale variation simulates field zoom [ENG]
        transforms.RandomResizedCrop(
            input_size,
            scale=(rrc.get("scale_min", 0.7), rrc.get("scale_max", 1.0)),
            ratio=(rrc.get("ratio_min", 3 / 4), rrc.get("ratio_max", 4 / 3)),
        ),
        transforms.RandomHorizontalFlip(p=aug.get("horizontal_flip_p", 0.5)),   # [ENG]
        transforms.RandomVerticalFlip(p=aug.get("vertical_flip_p", 0.3)),       # [ENG]
        transforms.RandomRotation(degrees=aug.get("rotation_degrees", 30)),     # [ENG]
        transforms.ColorJitter(                                                  # [ENG] domain gap
            brightness=cj.get("brightness", 0.3),
            contrast=cj.get("contrast", 0.3),
            saturation=cj.get("saturation", 0.3),
            hue=cj.get("hue", 0.1),
        ),
        transforms.RandomGrayscale(p=aug.get("grayscale_p", 0.05)),             # [ENG]
        # Bug5: RandomApply evaluates the p-gate per image, not once at build time.
        # Previously evaluated once at build time so every image in an epoch
        # saw the same choice (always blur or never blur).
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
            p=aug.get("gaussian_blur_p", 0.2),
        ),  # [ENG] kernel_size=5 (odd, required); p applied per image
    ]

    # RandomPerspective: p=0.0 by default (v1 unchanged); v2 sets p=0.3 [ENG]
    if perspective_p > 0.0:
        tfm_list.append(
            transforms.RandomPerspective(distortion_scale=0.5, p=perspective_p)
        )

    tfm_list += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.get("normalization", {}).get("mean", [0.485, 0.456, 0.406]),
            std=cfg.get("normalization", {}).get("std",  [0.229, 0.224, 0.225]),
        ),
    ]

    # RandomErasing: applied after ToTensor (requires tensor) [ENG]; p=0.0 = v1 behaviour
    if random_erase_p > 0.0:
        tfm_list.append(transforms.RandomErasing(p=random_erase_p))

    return transforms.Compose(tfm_list)


def build_eval_transform(cfg: dict) -> transforms.Compose:
    """
    Inference-time transform: resize + centre-crop + normalise. No augmentation.
    Applied identically to Val, Test_ID, Test_OOD.
    """
    input_size = cfg.get("model", {}).get("input_size", 224)
    resize_to  = int(input_size * 256 / 224)
    norm       = cfg.get("normalization", {})
    return transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=norm.get("mean", [0.485, 0.456, 0.406]),
            std=norm.get("std",  [0.229, 0.224, 0.225]),
        ),
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────


def _crop_from_label(label_name: str) -> str:
    """
    Extract the crop name from a class label string.
    Format: '<Crop>___<Disease>'. Returns the text before '___'.
    Special case: 'Corn_(maize)' and 'Corn' are treated as the same crop [ENG].
    """
    crop = label_name.split("___")[0].strip()
    # Normalise: treat 'Corn_(maize)' as 'Corn' [ENG diagnostic]
    if crop.lower().startswith("corn"):
        crop = "Corn"
    return crop


def _compute_ood_diagnostics(
    logits_all: np.ndarray,   # (N, num_classes) -- raw numpy, BEFORE softmax
    preds: np.ndarray,
    targets: np.ndarray,
    confs: np.ndarray,
    idx_to_label: dict,
) -> dict:
    """
    Compute OOD robustness diagnostics for a single evaluation split.
    All metrics are computed from actual predictions -- no numbers are fabricated [R1].

    Metrics returned:
      crop_restricted_top1       -- top-1 after masking logits to true crop classes
      crop_restricted_macro_f1   -- macro-F1 after crop-restricted argmax
      crop_top1                  -- fraction whose predicted class belongs to the true crop
      healthy_vs_diseased_acc    -- binary accuracy: healthy vs diseased
      diseased_predicted_healthy -- fraction of diseased images predicted as healthy
      mean_confidence            -- mean of max softmax across all images
      top5_predicted_classes     -- 5 most-predicted classes with share (fraction)

    NOTE: crop_restricted_* metrics are [ENG diagnostic: valid only if the app
    collects the crop from the user].
    """
    from sklearn.metrics import f1_score

    num_cls = len(idx_to_label)
    n = len(targets)

    # ── Crop-to-class mapping ────────────────────────────────────────────────
    # Build {crop_str: [class_idx, ...]} from idx_to_label
    crop_to_indices: dict = {}
    for idx, name in idx_to_label.items():
        crop = _crop_from_label(name)
        crop_to_indices.setdefault(crop, []).append(idx)

    # ── crop_restricted argmax ───────────────────────────────────────────────
    crop_restricted_preds = np.empty(n, dtype=np.int64)
    for i in range(n):
        true_crop = _crop_from_label(idx_to_label[int(targets[i])])
        valid_indices = crop_to_indices.get(true_crop, list(range(num_cls)))
        # Mask logits: set invalid class logits to -inf, then argmax
        row_logits = logits_all[i].copy().astype(np.float64)
        mask = np.full(num_cls, -np.inf)
        mask[valid_indices] = row_logits[valid_indices]
        crop_restricted_preds[i] = int(np.argmax(mask))

    crop_restricted_top1 = float((crop_restricted_preds == targets).mean())
    crop_restricted_macro_f1 = float(
        f1_score(
            targets, crop_restricted_preds,
            average="macro", zero_division=0,
            labels=list(range(num_cls)),
        )
    )

    # ── crop_top1: fraction whose top-1 pred belongs to true crop ────────────
    crop_correct = np.array([
        _crop_from_label(idx_to_label[int(preds[i])]) ==
        _crop_from_label(idx_to_label[int(targets[i])])
        for i in range(n)
    ], dtype=float)
    crop_top1 = float(crop_correct.mean())

    # ── healthy_vs_diseased_acc ──────────────────────────────────────────────
    def _is_healthy(label_idx: int) -> bool:
        return idx_to_label[int(label_idx)].endswith("___healthy")

    true_healthy  = np.array([_is_healthy(t) for t in targets], dtype=bool)
    pred_healthy  = np.array([_is_healthy(p) for p in preds],   dtype=bool)

    hvd_acc = float((true_healthy == pred_healthy).mean())

    # fraction of DISEASED images predicted as healthy
    diseased_mask = ~true_healthy
    if diseased_mask.sum() > 0:
        diseased_pred_healthy = float(pred_healthy[diseased_mask].mean())
    else:
        diseased_pred_healthy = float("nan")

    # ── mean confidence + top-5 most predicted classes ───────────────────────
    mean_conf = float(confs.mean())

    pred_counter = Counter(int(p) for p in preds)
    top5 = pred_counter.most_common(5)
    top5_classes = [
        {"class": idx_to_label[cls_idx], "share": round(cnt / max(n, 1), 4)}
        for cls_idx, cnt in top5
    ]

    return {
        "crop_restricted_top1": round(crop_restricted_top1, 4),
        "crop_restricted_macro_f1": round(crop_restricted_macro_f1, 4),
        "crop_restricted_note": "[ENG diagnostic: valid only if the app collects the crop from the user]",
        "crop_top1": round(crop_top1, 4),
        "healthy_vs_diseased_acc": round(hvd_acc, 4),
        "diseased_predicted_healthy": (
            round(diseased_pred_healthy, 4)
            if not (isinstance(diseased_pred_healthy, float) and np.isnan(diseased_pred_healthy))
            else None
        ),
        "mean_confidence": round(mean_conf, 4),
        "top5_predicted_classes": top5_classes,
    }


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx_to_label: dict,
    split_name: str,
    temperature: float = None,  # Bug8: for logging which T was used
) -> dict:
    """
    Evaluate model on a single data split.
    Returns dict with macro_f1, per_class_f1/precision/recall/support, ece,
    top1_acc, confusion_matrix, raw arrays, and OOD diagnostics.

    All returned numbers are computed from actual data [R1].
    No numbers are hardcoded or fabricated.
    Bug8: OOD ECE caveat -- T is fitted on D2_Val; OOD ECE expected to be worse.
    """
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    model.eval()
    all_preds, all_targets, all_confs, all_logits = [], [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            logits  = model(inputs, calibrate=True)  # uses calibrated T
            probs   = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())  # store for crop-restricted diagnostics

    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    confs   = np.array(all_confs)
    logits_all = np.concatenate(all_logits, axis=0)  # (N, num_classes)
    correct = (preds == targets).astype(float)

    n       = len(targets)
    num_cls = len(idx_to_label)
    labels  = list(range(num_cls))

    macro_f1   = float(f1_score(targets, preds, average="macro", zero_division=0, labels=labels))
    top1_acc   = float(correct.mean())
    ece        = compute_ece(confs, correct)
    conf_mat   = confusion_matrix(targets, preds, labels=labels)

    # Bug6: per-class precision, recall, f1, support
    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0
    )

    # Build per-class dict keyed by class name [Bug6]
    per_class = {}
    zero_support_classes = []
    supported_f1s = []
    for i in labels:
        name = idx_to_label[i]
        support = int(sup_arr[i])
        per_class[name] = {
            "precision": float(prec_arr[i]),
            "recall":    float(rec_arr[i]),
            "f1":        float(f1_arr[i]),
            "support":   support,
        }
        if support == 0:
            zero_support_classes.append(name)
        else:
            supported_f1s.append(float(f1_arr[i]))

    macro_f1_supported_classes = float(np.mean(supported_f1s)) if supported_f1s else 0.0

    # ── OOD robustness diagnostics ────────────────────────────────────────────
    diag = _compute_ood_diagnostics(logits_all, preds, targets, confs, idx_to_label)

    t_note = "" if temperature is None else " (T=%.4f fitted on D2_Val [ENG])" % temperature
    logger.info(
        "[%s] n=%d top1_acc=%.4f macro_F1=%.4f (supported_F1=%.4f) ECE=%.4f%s",
        split_name, n, top1_acc, macro_f1, macro_f1_supported_classes, ece, t_note
    )
    logger.info(
        "[%s] OOD diagnostics: crop_restricted_top1=%.4f crop_restricted_macro_F1=%.4f "
        "[ENG diagnostic: valid only if the app collects the crop from the user]",
        split_name, diag["crop_restricted_top1"], diag["crop_restricted_macro_f1"],
    )
    logger.info(
        "[%s] OOD diagnostics: crop_top1=%.4f hvd_acc=%.4f "
        "diseased_pred_healthy=%s mean_conf=%.4f",
        split_name, diag["crop_top1"], diag["healthy_vs_diseased_acc"],
        diag["diseased_predicted_healthy"], diag["mean_confidence"],
    )
    logger.info(
        "[%s] top-5 predicted classes: %s",
        split_name,
        ", ".join(
            "{} ({:.1%})".format(e["class"], e["share"])
            for e in diag["top5_predicted_classes"]
        ),
    )
    if zero_support_classes:
        logger.warning(
            "[%s] %d classes have zero support: %s",
            split_name, len(zero_support_classes), zero_support_classes
        )

    return {
        "split": split_name,
        "n_images": n,
        "top1_accuracy": top1_acc,
        "macro_f1": macro_f1,
        "macro_f1_supported_classes": macro_f1_supported_classes,
        "ece": ece,
        "per_class": per_class,
        "per_class_f1": f1_arr.tolist(),          # kept for backward compat
        "confusion_matrix": conf_mat.tolist(),
        "zero_support_classes": zero_support_classes,
        "diagnostics": diag,
        "preds": preds,
        "targets": targets,
        "confs": confs,
    }


# ── Temperature calibration ───────────────────────────────────────────────────

def calibrate_temperature(
    model: EfficientNetB3Classifier,
    val_loader: DataLoader,
    device: torch.device,
    temperature_init: float = 1.5,
    max_iter: int = 50,
) -> float:
    """
    Optimise temperature T on D2_Val by minimising NLL (negative log-likelihood).
    Method: Guo et al. (2017) [ENG -- single scalar post-hoc calibration].
    Uses shared fit_temperature helper from src.models.classifier.
    T is optimised; backbone weights are FROZEN.

    Returns: optimised temperature T
    Target ECE < 0.05 is an [ENG assumption], not a literature-derived threshold [I24].
    """
    logger.info("Starting temperature calibration on D2_Val ...")
    model.eval()

    # Collect raw backbone logits on CPU (before any temperature scaling)
    all_logits, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model.backbone(inputs)   # raw logits, no temperature
            all_logits.append(logits.cpu())   # always move to CPU
            all_targets.append(targets.cpu())

    all_logits  = torch.cat(all_logits)   # (N, num_classes) -- on CPU
    all_targets = torch.cat(all_targets)  # (N,)             -- on CPU

    # Pre-calibration metrics on D2_Val (T=1.0)
    criterion = nn.CrossEntropyLoss()
    pre_nll = float(criterion(all_logits.to(torch.float64), all_targets.to(torch.long)).item())
    probs_pre = torch.softmax(all_logits.to(torch.float64), dim=1).numpy()
    confs_pre = probs_pre.max(axis=1)
    preds_pre = probs_pre.argmax(axis=1)
    corr_pre = (preds_pre == all_targets.numpy()).astype(float)
    pre_ece = compute_ece(confs_pre, corr_pre)

    # Use shared robust fit_temperature
    t_optimised = fit_temperature(all_logits, all_targets, max_iter=max_iter)

    # Post-calibration metrics on D2_Val
    post_nll = float(criterion(all_logits.to(torch.float64) / t_optimised, all_targets.to(torch.long)).item())
    probs_post = torch.softmax(all_logits.to(torch.float64) / t_optimised, dim=1).numpy()
    confs_post = probs_post.max(axis=1)
    preds_post = probs_post.argmax(axis=1)
    corr_post = (preds_post == all_targets.numpy()).astype(float)
    post_ece = compute_ece(confs_post, corr_post)

    logger.info(
        "D2_Val Calibration: T=%.4f | Pre-cal: NLL=%.4f ECE=%.4f | Post-cal: NLL=%.4f ECE=%.4f",
        t_optimised, pre_nll, pre_ece, post_nll, post_ece
    )

    # Move the calibrated parameter back to the model's device so forward passes work
    model.scaler.temperature = nn.Parameter(
        torch.tensor([t_optimised], dtype=torch.float32, device=device)
    )

    # Save calibration stats on model for metrics.json
    model._cal_stats = {
        "temperature": t_optimised,
        "pre_nll": round(pre_nll, 6),
        "post_nll": round(post_nll, 6),
        "pre_ece": round(pre_ece, 6),
        "post_ece": round(post_ece, 6),
    }

    return t_optimised


# ── Main training loop ────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """Load YAML config; fallback to defaults if yaml not installed."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.warning("PyYAML not installed -- using hardcoded defaults")
        return {}


def set_seeds(seed: int):
    """Fixed seed for reproducibility [R1]."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="LeafMind C2 -- EfficientNet-B3 classifier training (GPU)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Path to Dataset_2 directory. Overrides config paths.data_root."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs. Overrides config."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size. Overrides config."
    )
    parser.add_argument(
        "--config", type=str,
        default=str(REPO_ROOT / "config" / "classifier_config.yaml"),
        help="Path to classifier_config.yaml"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Override checkpoint output directory."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint .pth to resume training from."
    )
    parser.add_argument(
        "--no-calibrate", action="store_true",
        help="Skip temperature calibration step."
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, load model_best.pth (or model.pth) from --checkpoint-dir, calibrate and evaluate."
    )
    parser.add_argument(
        "--max-train-images", type=int, default=None,
        help="Limit training set size (seeded random sample -- for 1-epoch real-data dry run)."
    )
    parser.add_argument(
        "--max-eval-images", type=int, default=None,
        help="Limit val/Test_ID/Test_OOD set size (seeded random sample -- for dry run)."
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(Path(args.config))

    # CLI overrides
    if args.data_root:
        cfg.setdefault("paths", {})["data_root"] = args.data_root
    if args.epochs:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.checkpoint_dir:
        cfg.setdefault("paths", {})["checkpoint_dir"] = args.checkpoint_dir

    train_cfg = cfg.get("training", {})
    paths_cfg = cfg.get("paths", {})

    seed       = int(train_cfg.get("seed", 42))
    epochs     = int(train_cfg.get("epochs", 30))
    batch_size = int(train_cfg.get("batch_size", 64))
    lr         = float(train_cfg.get("learning_rate", 1e-4))
    wd         = float(train_cfg.get("weight_decay", 1e-4))
    patience   = int(train_cfg.get("early_stopping_patience", 5))
    warmup_ep  = int(train_cfg.get("warmup_epochs", 2))
    num_workers = int(cfg.get("dataloader", {}).get("num_workers", 4))
    pin_memory  = bool(cfg.get("dataloader", {}).get("pin_memory", True))

    data_root    = Path(paths_cfg.get("data_root", "Dataset_2"))
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    ckpt_dir  = Path(paths_cfg.get("checkpoint_dir", "checkpoints/classifier"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = REPO_ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Seed: %d | Epochs: %d | Batch: %d", device, seed, epochs, batch_size)

    if device.type == "cpu":
        logger.warning(
            "No GPU detected. Full training on CPU is very slow. "
            "Use scripts/smoke_test_classifier.py for local CPU testing. "
            "Run full training on Kaggle/Colab with GPU."
        )

    # ── Load class map ────────────────────────────────────────────────────────
    with open(REPO_ROOT / paths_cfg.get("class_map", "config/class_map_d2.json")) as f:
        raw_map = json.load(f)
    idx_to_label  = {int(k): v for k, v in raw_map["classes"].items()}
    label_to_idx  = {v: k for k, v in idx_to_label.items()}
    num_classes   = len(idx_to_label)
    logger.info("Loaded %d classes from class_map_d2.json", num_classes)

    # ── Load manifests (C1 deduplicated -- D1 images NEVER present) [D14] ────
    def load_manifest(split_name: str) -> list:
        fp = REPO_ROOT / paths_cfg.get("manifests_dir", "config/manifests") / f"d2_{split_name}.json"
        with open(fp) as f:
            return json.load(f)["images"]

    train_images = load_manifest("train")
    val_images   = load_manifest("val")
    test_id_imgs = load_manifest("test_id")
    test_ood_imgs = load_manifest("test_ood")

    # Load exclusion list for OOD filtered evaluation [R7/D16]
    excl_path = REPO_ROOT / paths_cfg.get("exclusion_list", "config/exclusion_list.json")
    with open(excl_path) as f:
        excl_md5s = set(json.load(f)["excluded_md5s"])

    # [D16] Primary OOD evaluation: full deduped Test_OOD (no exclusion)
    # Secondary OOD evaluation: exclusion-filtered set (covering only 7 classes, ~283 images)
    test_ood_filtered = [img for img in test_ood_imgs if img["md5"] not in excl_md5s]
    logger.info(
        "OOD: full=%d | exclusion-filtered=%d (secondary, covers 7 classes [D16/I12])",
        len(test_ood_imgs), len(test_ood_filtered)
    )
    logger.info(
        "CAVEAT: OOD exclusion-filtered set covers only 7 of 30 classes. "
        "Per-class OOD metrics are unreliable [I12]. Primary evaluation uses full deduped OOD set [D16]."
    )

    # ── Class weights for weighted cross-entropy loss [ENG] ───────────────────
    # RATIONALE [ENG]: Weighted CE chosen over WeightedRandomSampler -- see classifier.py
    # ~35.4x imbalance in D2_Train [DATA].
    # Bug7: ALWAYS computed from the FULL training manifest before any dry-run subsampling.
    full_train_counts = Counter(label_to_idx[img["class"]] for img in train_images)
    class_weights = compute_class_weights(dict(full_train_counts), num_classes).to(device)
    # label_smoothing from config (default 0.0 = v1 behaviour [ENG])
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    logger.info(
        "Class weights: min=%.3f max=%.3f (computed from full train manifest [%d images] [ENG/Bug7])",
        class_weights.min().item(), class_weights.max().item(), len(train_images)
    )
    logger.info("label_smoothing=%.2f [ENG; 0.0=v1 default, 0.1=v2]", label_smoothing)

    # Bug7: subset limits use seeded STRATIFIED random sample (not first-N slice)
    # Guarantees at least 1 image per class where available [Bug7].
    # Uses _rng.sample internally via stratified_sample.
    _rng = random.Random(seed)  # deterministic subset selection
    if args.max_train_images and len(train_images) > args.max_train_images:
        train_images = stratified_sample(train_images, args.max_train_images, _rng)
        logger.info(
            "DRY-RUN: stratified-sampled %d training images via _rng.sample (seed=%d)",
            len(train_images), seed
        )
    if args.max_eval_images:
        if len(val_images) > args.max_eval_images:
            val_images = stratified_sample(val_images, args.max_eval_images, _rng)
        if len(test_id_imgs) > args.max_eval_images:
            test_id_imgs = stratified_sample(test_id_imgs, args.max_eval_images, _rng)
        if len(test_ood_imgs) > args.max_eval_images:
            test_ood_imgs = stratified_sample(test_ood_imgs, args.max_eval_images, _rng)
        if len(test_ood_filtered) > args.max_eval_images:
            test_ood_filtered = stratified_sample(test_ood_filtered, args.max_eval_images, _rng)
        logger.info("DRY-RUN: stratified-sampled eval images per split via _rng.sample (seed=%d)", seed)

    logger.info(
        "Split sizes: Train=%d Val=%d Test_ID=%d Test_OOD_full=%d Test_OOD_filtered=%d",
        len(train_images), len(val_images), len(test_id_imgs),
        len(test_ood_imgs), len(test_ood_filtered)
    )

    # ── Transforms ────────────────────────────────────────────────────────────
    train_tf = build_train_transform(cfg)
    eval_tf  = build_eval_transform(cfg)

    # ── Datasets and loaders ──────────────────────────────────────────────────
    train_ds = D2Dataset(train_images, data_root, train_tf, label_to_idx)
    val_ds   = D2Dataset(val_images,   data_root, eval_tf,  label_to_idx)
    test_id_ds  = D2Dataset(test_id_imgs,     data_root, eval_tf, label_to_idx)
    test_ood_ds = D2Dataset(test_ood_imgs,    data_root, eval_tf, label_to_idx)
    test_ood_filt_ds = D2Dataset(test_ood_filtered, data_root, eval_tf, label_to_idx)

    if not args.eval_only:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        train_loader = None
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_id_loader = DataLoader(
        test_id_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_ood_loader = DataLoader(
        test_ood_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_ood_filt_loader = DataLoader(
        test_ood_filt_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = cfg.get("model", {})
    model = EfficientNetB3Classifier(
        num_classes=num_classes,
        drop_rate=float(model_cfg.get("drop_rate", 0.3)),
        temperature=float(cfg.get("calibration", {}).get("temperature_init", 1.5)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    ).to(device)

    if args.resume:
        logger.info("Resuming from %s", args.resume)
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    # Train all params; using AdamW [ENG]
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Linear warmup for warmup_ep epochs, then cosine decay [ENG]
    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_ep:
            return (epoch + 1) / max(warmup_ep, 1)
        progress = (epoch - warmup_ep) / max(epochs - warmup_ep, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=_lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1     = -1.0
    no_improve_cnt  = 0
    best_epoch      = -1
    train_history   = []

    if not args.eval_only:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches  = 0
            t0 = time.time()

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimiser.zero_grad()
                # Training: raw logits (no temperature scaling in loss) [ENG]
                logits = model(inputs, calibrate=False)
                loss   = criterion(logits, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # [ENG]
                optimiser.step()
                epoch_loss += loss.item()
                n_batches  += 1

                if (batch_idx + 1) % 100 == 0:
                    logger.info(
                        "Epoch %d/%d  batch %d/%d  loss=%.4f  lr=%.2e",
                        epoch, epochs, batch_idx + 1, len(train_loader),
                        loss.item(), scheduler.get_last_lr()[0]
                    )

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed  = time.time() - t0

            # ── Validation ────────────────────────────────────────────────────────
            from sklearn.metrics import f1_score as _f1
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    logits = model(inputs, calibrate=True)
                    probs  = torch.softmax(logits, dim=1)
                    preds  = probs.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            val_f1 = float(_f1(
                np.array(val_targets), np.array(val_preds),
                average="macro", zero_division=0, labels=list(range(num_classes))
            ))

            logger.info(
                "Epoch %d/%d  avg_loss=%.4f  val_macro_F1=%.4f  time=%.1fs",
                epoch, epochs, avg_loss, val_f1, elapsed
            )
            train_history.append({"epoch": epoch, "loss": avg_loss, "val_macro_f1": val_f1})

            # ── Early stopping + best model save ─────────────────────────────────
            if val_f1 > best_val_f1:
                best_val_f1    = val_f1
                best_epoch     = epoch
                no_improve_cnt = 0
                torch.save(model.state_dict(), ckpt_dir / "model_best.pth")
                logger.info("Best model saved (epoch=%d val_macro_F1=%.4f)", epoch, best_val_f1)
            else:
                no_improve_cnt += 1
                if no_improve_cnt >= patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs [ENG])",
                        epoch, patience
                    )
                    break

        # Save final model
        torch.save(model.state_dict(), ckpt_dir / "model.pth")
        logger.info("Final model saved to %s/model.pth", ckpt_dir)
    else:
        logger.info("--eval-only specified: skipping training phase.")
        if (ckpt_dir / "config.json").exists():
            try:
                with open(ckpt_dir / "config.json") as f:
                    saved_cfg = json.load(f)
                    best_epoch = saved_cfg.get("best_epoch", best_epoch)
                    best_val_f1 = saved_cfg.get("best_val_macro_f1", best_val_f1)
            except Exception:
                pass

    # Load model for evaluation: model_best.pth or model.pth
    best_path = ckpt_dir / "model_best.pth"
    model_path = best_path if best_path.exists() else ckpt_dir / "model.pth"
    if model_path.exists():
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded checkpoint from %s for evaluation", model_path)
    elif args.eval_only:
        raise FileNotFoundError(
            f"--eval-only specified, but neither model_best.pth nor model.pth found in {ckpt_dir}"
        )

    # ── Temperature calibration on D2_Val ─────────────────────────────────────
    temperature = float(cfg.get("calibration", {}).get("temperature_init", 1.5))
    if not args.no_calibrate:
        cal_cfg  = cfg.get("calibration", {})
        temperature = calibrate_temperature(
            model, val_loader, device,
            temperature_init=float(cal_cfg.get("temperature_init", 1.5)),
            max_iter=int(cal_cfg.get("max_iter", 50)),
        )
        ece_target = float(cal_cfg.get("ece_target", 0.05))
        logger.info(
            "Temperature T=%.4f [ENG assumption target ECE<%.2f -- not literature-derived I24]",
            temperature, ece_target
        )

    # Save temperature.json
    with open(ckpt_dir / "temperature.json", "w") as f:
        json.dump({"temperature": temperature, "tag": "[ENG assumption -- calibrate on D2_Val]"}, f, indent=2)

    # Bug4: Re-save model_best.pth (and model.pth) with the calibrated temperature
    # baked into the state dict, so any checkpoint loaded without temperature.json
    # still has the correct T. temperature.json remains the authoritative source
    # (see LeafClassifier.from_checkpoint which always reads temperature.json).
    torch.save(model.state_dict(), ckpt_dir / "model_best.pth")
    torch.save(model.state_dict(), ckpt_dir / "model.pth")
    logger.info(
        "Checkpoint re-saved with calibrated T=%.4f baked in (model.pth + model_best.pth) [Bug4]",
        temperature
    )

    # Save class_map.json and config.json into checkpoint dir
    with open(ckpt_dir / "class_map.json", "w") as f:
        json.dump({str(k): v for k, v in idx_to_label.items()}, f, indent=2)

    model_version = paths_cfg.get("model_version", "efficientnet_b3_v1")
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({
            "model_version": model_version,
            "num_classes": num_classes,
            "input_size": model_cfg.get("input_size", 224),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "tag": "[NOT MEASURED until GPU training completes]",
        }, f, indent=2)

    # ── Full evaluation (all splits) -- uses calibrated T ─────────────────────
    logger.info("--- Evaluation on all splits ---")
    # Bug8: T was fitted on D2_Val (lab-style images). ECE on OOD splits is
    # expected to be worse due to domain shift. This is not a bug; report with caveat.
    logger.info(
        "NOTE [Bug8]: T=%.4f was fitted on D2_Val only. "
        "OOD ECE is expected to be higher due to domain shift [ENG/DATA].",
        temperature
    )
    results = {}

    results["val"]          = evaluate_split(model, val_loader,           device, idx_to_label, "D2_Val",              temperature)
    results["test_id"]      = evaluate_split(model, test_id_loader,       device, idx_to_label, "D2_Test_ID",          temperature)
    # [D16] Primary OOD: full deduped set
    results["test_ood_full"] = evaluate_split(model, test_ood_loader,     device, idx_to_label, "D2_Test_OOD_full",    temperature)
    # Secondary OOD: exclusion-filtered
    results["test_ood_filt"] = evaluate_split(model, test_ood_filt_loader, device, idx_to_label, "D2_Test_OOD_filtered", temperature)

    # OOD gap [ENG spec 2a-vii]
    id_f1  = results["test_id"]["macro_f1"]
    ood_f1 = results["test_ood_full"]["macro_f1"]
    ood_gap = (id_f1 - ood_f1) / max(id_f1, 1e-6)
    logger.info(
        "OOD gap [D16]: (Test_ID_F1 - Test_OOD_F1) / Test_ID_F1 = (%.4f - %.4f) / %.4f = %.4f",
        id_f1, ood_f1, id_f1, ood_gap
    )
    logger.info(
        "CAVEAT: OOD filtered set covers only ~7 of 30 classes -- per-class OOD metrics unreliable [I12/D16]"
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LEAFMIND C2 TRAINING SUMMARY")
    print("=" * 70)
    for split, r in results.items():
        diag = r.get("diagnostics", {})
        if "ood" in split.lower():
            print(
                "{:30s}  n={:5d}  top1={:.4f}  macro_F1={:.4f} (supported_F1={:.4f})  ECE={:.4f}".format(
                    r["split"], r["n_images"], r["top1_accuracy"], r["macro_f1"], r["macro_f1_supported_classes"], r["ece"]
                )
            )
        else:
            print(
                "{:30s}  n={:5d}  top1={:.4f}  macro_F1={:.4f}  ECE={:.4f}".format(
                    r["split"], r["n_images"], r["top1_accuracy"], r["macro_f1"], r["ece"]
                )
            )
        if diag:
            print(
                "  [diag] crop_restr_top1={:.4f}  crop_restr_mF1={:.4f}  "
                "[ENG diagnostic: valid only if the app collects the crop from the user]".format(
                    diag.get("crop_restricted_top1", float("nan")),
                    diag.get("crop_restricted_macro_f1", float("nan")),
                )
            )
            print(
                "  [diag] crop_top1={:.4f}  hvd_acc={:.4f}  "
                "diseased_pred_healthy={}  mean_conf={:.4f}".format(
                    diag.get("crop_top1", float("nan")),
                    diag.get("healthy_vs_diseased_acc", float("nan")),
                    diag.get("diseased_predicted_healthy"),
                    diag.get("mean_confidence", float("nan")),
                )
            )
            top5_str = ", ".join(
                "{} ({:.1%})".format(e["class"], e["share"])
                for e in diag.get("top5_predicted_classes", [])
            )
            print("  [diag] top-5 predicted: {}".format(top5_str))
    print("OOD gap (primary):  {:.4f}".format(ood_gap))
    print("Temperature T:      {:.4f}  (fitted on D2_Val only [ENG])".format(temperature))
    print("Best epoch:         {}".format(best_epoch))
    # Bug8: explicit caveat that OOD ECE is fitted-on-val
    print("\nNOTE: T was fitted on D2_Val (lab-style images).")
    print("OOD ECE is expected to be higher due to domain shift. [ENG/DATA]")
    # Bug6: zero-support OOD classes
    for ood_split in ("test_ood_full", "test_ood_filt"):
        zs = results[ood_split].get("zero_support_classes", [])
        if zs:
            print("Zero-support classes in {}: {}".format(results[ood_split]["split"], zs))
    print("\nQuality gate checks [ENG assumption thresholds]:")
    print("  Test_ID macro_F1 >= 0.75: {} ({:.4f})".format(
        "PASS" if id_f1 >= 0.75 else "FAIL/NOT MET", id_f1))
    print("  Test_OOD macro_F1 >= 0.50: {} ({:.4f})".format(
        "PASS" if ood_f1 >= 0.50 else "FAIL/NOT MET", ood_f1))
    print("  Val ECE <= 0.05: {} ({:.4f})".format(
        "PASS" if results["val"]["ece"] <= 0.05 else "FAIL/NOT MET", results["val"]["ece"]))
    print("=" * 70)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    cal_stats = getattr(model, "_cal_stats", {})
    metrics_out = {
        "model_version": model_version,
        "temperature": temperature,
        "temperature_note": "T fitted on D2_Val only; OOD ECE expected worse due to domain shift [Bug8/ENG/DATA]",
        "calibration": {
            "temperature": temperature,
            "pre_nll": cal_stats.get("pre_nll"),
            "post_nll": cal_stats.get("post_nll"),
            "pre_ece": cal_stats.get("pre_ece"),
            "post_ece": cal_stats.get("post_ece"),
            "ece_target": float(cfg.get("calibration", {}).get("ece_target", 0.05)),
            "ece_target_met": (
                bool(cal_stats["post_ece"] < float(cfg.get("calibration", {}).get("ece_target", 0.05)))
                if cal_stats.get("post_ece") is not None else None
            ),
        },
        "best_epoch": best_epoch,
        "splits": {},
        "ood_gap_primary": ood_gap,
        "tag": "[ALL NUMBERS COMPUTED from actual data -- R1]",
        "ood_caveat": "Test_OOD filtered set covers only ~7 of 30 classes [I12/D16]. Primary OOD uses full deduped set.",
    }

    # Bug6: full per-class metrics keyed by class name; confusion matrix; zero-support list
    import csv
    for split, r in results.items():
        metrics_out["splits"][split] = {
            "n_images":                   r["n_images"],
            "top1_accuracy":              r["top1_accuracy"],
            "macro_f1":                   r["macro_f1"],
            "macro_f1_supported_classes": r["macro_f1_supported_classes"],
            "ece":                        r["ece"],
            "per_class":                  r["per_class"],
            "per_class_f1":               r["per_class_f1"],      # backward compat
            "confusion_matrix":           r["confusion_matrix"],
            "zero_support_classes":       r["zero_support_classes"],
            "diagnostics":                r.get("diagnostics", {}),
        }
        # Bug6: save per-split confusion matrix as CSV
        csv_path = ckpt_dir / "confusion_matrix_{}.csv".format(split)
        cls_names = [idx_to_label[i] for i in range(num_classes)]
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([""] + cls_names)  # header
            for i, row in enumerate(r["confusion_matrix"]):
                writer.writerow([cls_names[i]] + row)
        logger.info("Confusion matrix CSV: %s", csv_path)

    metrics_path = ckpt_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save training history
    if not args.eval_only:
        with open(ckpt_dir / "train_history.json", "w") as f:
            json.dump(train_history, f, indent=2)


if __name__ == "__main__":
    main()
