"""
src/models/classifier.py
========================
EfficientNet-B3 classifier with temperature scaling for the Vision Agent.

Design constraints (C2 spec):
  - Trained ONLY on D2_Train; D1 images never enter this pipeline [D14/R13]
  - RGB conversion forced at load time to handle CMYK/RGBA/L modes [I9]
  - Class-weighted cross-entropy loss handles ~35x imbalance [2a-vi]
  - Temperature scaling calibrated on D2_Val; stored in temperature.json [2a-viii]
  - severity_pct is ALWAYS None -- never compute or estimate severity [R3/D2]
  - OOD flag: max softmax < 0.40 [ENG assumption -- to be calibrated, see I12/L14]
  - Escalation: confidence < 0.60 OR (top1-top2) < 0.15 [ENG assumption -- to be calibrated]
  - Output MUST validate against VisionOutput Pydantic schema [R6]

All threshold values are [ENG assumption] and tagged in code comments.
No fabricated performance numbers anywhere in this file [R1].
ASCII-only logging and output [I27].
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from PIL import Image, ImageFilter

from src.schemas.schemas import ClassProb, DetectedLeaf, ImageQualityFlags, VisionOutput

logger = logging.getLogger(__name__)

# ── Constants / defaults (all overridable via config) ─────────────────────────

# ImageNet normalisation statistics [ENG] -- standard for pretrained EfficientNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Severity note (mandatory -- R3 / D2 / Part 2a-xi)
SEVERITY_NOTE = (
    "Severity percentage is not computed. No per-leaf severity labels or lesion masks "
    "exist in the training data. The PPT design target is acknowledged but cannot be "
    "implemented without expert-annotated mask data."
)

NUM_CLASSES_D2 = 30  # [DATA]

# Default threshold values -- all [ENG assumption]; calibrate on D2_Val after training
_DEFAULT_OOD_THRESH = 0.40          # max softmax < threshold -> ood_flag [ENG assumption]
_DEFAULT_ESC_CONF   = 0.60          # confidence < threshold -> escalate [ENG assumption]
_DEFAULT_ESC_GAP    = 0.15          # top1-top2 < threshold -> escalate [ENG assumption]
_DEFAULT_LOW_RES_PX = 200           # max(W,H) < threshold -> low_resolution [ENG assumption]
_DEFAULT_BLUR_LAP   = 100.0         # Laplacian variance < threshold -> blurry [ENG assumption]
                                    # NOTE: uncalibrated -- see L15/GAP


# ── Cross-platform path helper [Bug1] ────────────────────────────────────────

def resolve_image_path(data_root: Path | str, rel: str) -> Path:
    """
    Resolve a manifest-relative image path to an absolute Path.

    D2 manifests were generated on Windows and store paths with backslash
    separators (e.g. "Val\\Apple___Apple_scab\\img.JPG"). On Linux/Kaggle/Mac,
    pathlib treats "\\" as a literal filename character rather than a separator,
    so images cannot be found.

    This helper replaces all backslashes with forward slashes before constructing
    the Path, making it safe on every OS.

    Usage:
        img_path = resolve_image_path(data_root, entry["path"])
    """
    return Path(data_root) / rel.replace("\\", "/")


# ── Quality gate helpers ──────────────────────────────────────────────────────

def _laplacian_variance(pil_img: Image.Image) -> float:
    """
    Compute the variance of the Laplacian of the image as a sharpness proxy.
    Higher = sharper. Source: [ENG assumption] commonly used CV heuristic.
    Threshold calibration is a GAP -- see L15 in plan.md.
    """
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    # Approximate Laplacian with finite differences
    laplacian = (
        np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1)
        - 4 * arr
    )
    return float(np.var(laplacian))


def check_image_quality(
    pil_img: Image.Image,
    low_res_px: int = _DEFAULT_LOW_RES_PX,
    blur_lap: float = _DEFAULT_BLUR_LAP,
) -> ImageQualityFlags:
    """
    Run image quality checks and return ImageQualityFlags.

    Thresholds are [ENG assumption] -- calibration requires a labelled quality set [GAP/L15].
    non_leaf_suspected is always False (placeholder -- no dedicated leaf detection model).
    """
    w, h = pil_img.size
    low_resolution = (max(w, h) < low_res_px)  # [ENG assumption]

    lap_var = _laplacian_variance(pil_img)
    blurry = (lap_var < blur_lap)               # [ENG assumption -- uncalibrated, L15]

    logger.debug(
        "Quality check: size=%dx%d lap_var=%.1f low_res=%s blurry=%s",
        w, h, lap_var, low_resolution, blurry
    )

    return ImageQualityFlags(
        low_resolution=low_resolution,
        blurry=blurry,
        non_leaf_suspected=False,  # placeholder -- no leaf-vs-background model [ENG]
    )


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(
    image_bytes: bytes,
    input_size: int = 224,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Decode raw image bytes, force-convert to RGB, and apply inference-time preprocessing.

    RGB conversion handles CMYK, RGBA, and L (grayscale) modes present in both D1 and
    D2_Test_OOD [I9]. Called at every inference -- do NOT skip for any image mode.

    Returns:
        tensor: normalised (1, 3, H, W) float32 tensor ready for the model
        pil_img: the RGB PIL image (for quality checks)
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc

    # Force RGB -- handles CMYK, RGBA, L, P modes [I9]
    if pil_img.mode != "RGB":
        logger.debug("Converting image from mode %s to RGB", pil_img.mode)
        pil_img = pil_img.convert("RGB")

    # Resize: shortest edge to (input_size * 256/224), then centre-crop to input_size
    resize_to = int(math.ceil(input_size * 256 / 224))
    import torchvision.transforms.functional as TF
    w, h = pil_img.size
    if w < h:
        new_w, new_h = resize_to, int(h * resize_to / w)
    else:
        new_w, new_h = int(w * resize_to / h), resize_to
    pil_img_resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    # Centre crop
    left = (new_w - input_size) // 2
    top  = (new_h - input_size) // 2
    pil_cropped = pil_img_resized.crop((left, top, left + input_size, top + input_size))

    # ToTensor + Normalize
    import torchvision.transforms.functional as TF
    tensor = TF.to_tensor(pil_cropped)  # (3, H, W) in [0,1]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0), pil_img  # return original full PIL for quality check


# ── Temperature scaling ───────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Single-scalar temperature scaling for post-hoc calibration.
    Method: Guo et al. (2017) "On Calibration of Modern Neural Networks".
    T is optimised on D2_Val via NLL minimisation [ENG spec 2a-viii].
    T > 1 softens predictions; T < 1 sharpens them.
    """
    def __init__(self, temperature: float = 1.5):
        super().__init__()
        # [ENG assumption] initial T=1.5; optimised on D2_Val
        self.temperature = nn.Parameter(torch.tensor([temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-6)

    def get_temperature(self) -> float:
        return float(self.temperature.item())


# ── Classifier wrapper ────────────────────────────────────────────────────────

class EfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3 fine-tuned for 30-class D2 leaf disease classification.
    Wraps timm model + TemperatureScaler as a single inference unit.

    NEVER used with D1 data during training -- D14/R13 hard constraint.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES_D2,
        drop_rate: float = 0.3,
        temperature: float = 1.5,
        pretrained: bool = True,
    ):
        super().__init__()
        # [ENG] timm EfficientNet-B3; pretrained=True uses ImageNet-1k weights
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        self.scaler = TemperatureScaler(temperature)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, calibrate: bool = True) -> torch.Tensor:
        """
        Forward pass. Returns calibrated logits when calibrate=True (inference),
        raw logits when calibrate=False (training, where loss is computed on raw logits).
        """
        logits = self.backbone(x)
        if calibrate:
            logits = self.scaler(logits)
        return logits

    def get_temperature(self) -> float:
        return self.scaler.get_temperature()


# ── Full inference pipeline ───────────────────────────────────────────────────

class LeafClassifier:
    """
    Full inference pipeline: bytes -> VisionOutput.
    Wraps EfficientNetB3Classifier + quality gate + OOD + escalation flags.

    Usage:
        clf = LeafClassifier.from_checkpoint("checkpoints/classifier")
        output: VisionOutput = clf.predict(image_bytes)
    """

    def __init__(
        self,
        model: EfficientNetB3Classifier,
        idx_to_label: dict,   # {int: str} mapping from class index to label string
        model_version: str = "efficientnet_b3_v1",
        device: Optional[torch.device] = None,
        ood_thresh: float = _DEFAULT_OOD_THRESH,
        esc_conf: float = _DEFAULT_ESC_CONF,
        esc_gap: float = _DEFAULT_ESC_GAP,
        low_res_px: int = _DEFAULT_LOW_RES_PX,
        blur_lap: float = _DEFAULT_BLUR_LAP,
        input_size: int = 224,
        top_k: int = 5,
    ):
        self.model = model
        self.idx_to_label = idx_to_label
        self.model_version = model_version
        self.device = device or torch.device("cpu")
        # Thresholds [ENG assumption] -- see config/classifier_config.yaml
        self.ood_thresh = ood_thresh     # [ENG assumption]
        self.esc_conf   = esc_conf       # [ENG assumption]
        self.esc_gap    = esc_gap        # [ENG assumption]
        self.low_res_px = low_res_px     # [ENG assumption]
        self.blur_lap   = blur_lap       # [ENG assumption -- GAP L15]
        self.input_size = input_size
        self.top_k      = top_k
        self.model.eval()
        self.model.to(self.device)

    def predict(self, image_bytes: bytes) -> VisionOutput:
        """
        Run full inference pipeline on raw image bytes.
        Returns a VisionOutput that validates against the Pydantic schema [R6].

        severity_pct is ALWAYS None [R3/D2].
        D1 images are NEVER processed in training context [D14/R13].
        """
        # Step 1: decode + preprocess
        try:
            tensor, pil_img = preprocess_image(image_bytes, self.input_size)
        except ValueError as exc:
            raise RuntimeError(f"Vision pipeline: cannot decode image -- {exc}") from exc

        # Step 2: image quality flags [ENG assumption thresholds -- GAP L15]
        quality_flags = check_image_quality(pil_img, self.low_res_px, self.blur_lap)

        # Step 3: run classifier (calibrated logits)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            cal_logits = self.model(tensor, calibrate=True)  # temperature-scaled
            probs = torch.softmax(cal_logits, dim=1)[0]      # (num_classes,)

        # Step 4: top-k predictions
        top_probs, top_indices = torch.topk(probs, self.top_k)
        top_k_probs: List[ClassProb] = [
            ClassProb(label=self.idx_to_label[int(idx)], prob=float(prob))
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
        ]

        top_label   = top_k_probs[0].label
        confidence  = top_k_probs[0].prob
        top2_prob   = top_k_probs[1].prob if len(top_k_probs) > 1 else 0.0
        gap         = confidence - top2_prob

        temperature = self.model.get_temperature()

        # Step 5: OOD flag [ENG assumption -- calibrate on labelled OOD set -- L14]
        ood_flag = confidence < self.ood_thresh  # [ENG assumption]

        # Step 6: escalation flag [ENG assumption -- calibrate on D2_Val]
        escalate = (confidence < self.esc_conf) or (gap < self.esc_gap)  # [ENG assumption]

        logger.info(
            "Predict: top=%s conf=%.3f gap=%.3f T=%.3f ood=%s esc=%s",
            top_label, confidence, gap, temperature, ood_flag, escalate
        )

        # Step 7: build VisionOutput -- severity_pct ALWAYS None [R3/D2]
        return VisionOutput(
            top_label=top_label,
            confidence=confidence,
            top_k_probs=top_k_probs,
            severity_pct=None,          # ALWAYS None -- R3 / Part 2a-xi / D2
            severity_note=SEVERITY_NOTE,
            image_quality_flags=quality_flags,
            ood_flag=ood_flag,
            leaf_detected=False,        # YOLOv8 detector out of scope for C2
            detected_leaves=[],
            escalate_to_human=escalate,
            model_version=self.model_version,
            temperature=temperature,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: Optional[torch.device] = None,
        **threshold_kwargs,
    ) -> "LeafClassifier":
        """
        Load a trained classifier from the checkpoint directory.
        Expects:
            {checkpoint_dir}/model.pth        -- full model state dict
            {checkpoint_dir}/temperature.json -- {"temperature": <float>}
            {checkpoint_dir}/class_map.json   -- {"0": "label", ...}
            {checkpoint_dir}/config.json      -- {"model_version": str, ...}
        """
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Load class map
        class_map_path = ckpt_dir / "class_map.json"
        if not class_map_path.exists():
            raise FileNotFoundError(f"class_map.json missing from {ckpt_dir}")
        with open(class_map_path) as f:
            raw_map = json.load(f)
        idx_to_label = {int(k): v for k, v in raw_map.items()}
        num_classes   = len(idx_to_label)

        # Load temperature
        temp_path = ckpt_dir / "temperature.json"
        if temp_path.exists():
            with open(temp_path) as f:
                temperature = float(json.load(f)["temperature"])
        else:
            temperature = 1.5  # [ENG default] -- log warning
            logger.warning(
                "temperature.json not found in %s -- using default T=1.5 [uncalibrated]",
                ckpt_dir
            )

        # Load optional config
        config_path = ckpt_dir / "config.json"
        model_version = "efficientnet_b3_v1"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            model_version = cfg.get("model_version", model_version)

        # Build model and load weights
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = EfficientNetB3Classifier(
            num_classes=num_classes,
            temperature=temperature,
            pretrained=False,   # weights loaded from checkpoint
        )

        model_path = ckpt_dir / "model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"model.pth missing from {ckpt_dir}")

        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        # Bug4: temperature.json is authoritative. After load_state_dict the
        # scaler's temperature may reflect whatever was last saved into the
        # state dict (which could be an uncalibrated default). Explicitly
        # overwrite it from temperature.json so inference always uses the
        # calibrated value, regardless of what is in the .pth file.
        model.scaler.temperature.data.fill_(temperature)
        model.eval()

        logger.info(
            "Loaded classifier from %s (classes=%d T=%.3f device=%s)",
            ckpt_dir, num_classes, temperature, device
        )

        return cls(
            model=model,
            idx_to_label=idx_to_label,
            model_version=model_version,
            device=device,
            **threshold_kwargs,
        )


# ── Class-weight helper (used by training scripts) ────────────────────────────

def compute_class_weights(
    class_counts: dict,   # {class_idx: count}
    num_classes: int,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted cross-entropy loss.
    Formula: weight_i = N_total / (N_classes * count_i)

    RATIONALE [ENG]: Weighted cross-entropy is chosen over WeightedRandomSampler
    because it works with standard shuffled DataLoader, is numerically stable,
    and avoids minority class over-repetition within a single epoch.
    Both approaches handle the ~35.4x D2_Train class imbalance [DATA].

    Reference: scikit-learn compute_class_weight('balanced') formula.
    """
    total = sum(class_counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for idx in range(num_classes):
        count = class_counts.get(idx, 1)  # avoid division by zero
        weights[idx] = total / (num_classes * count)
    return weights


# ── ECE computation helper ────────────────────────────────────────────────────

def compute_ece(
    confidences: np.ndarray,
    correct:     np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE) via equal-width binning.
    Source: Guo et al. (2017) "On Calibration of Modern Neural Networks" -- methodology.
    Target ECE < 0.05 is an [ENG assumption], not a literature-derived threshold.
    See I24 and threshold register in plan.md.

    Args:
        confidences: (N,) array of max softmax probabilities after temperature scaling
        correct:     (N,) bool array (1 if top-1 prediction == ground truth)
        n_bins:      number of equal-width bins [ENG]
    Returns:
        ECE as a float in [0, 1]
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc  = float(correct[mask].mean())
        bin_frac = mask.sum() / n
        ece += abs(avg_conf - avg_acc) * bin_frac
    return float(ece)


# ── Robust temperature fitting (Guo et al. 2017) ──────────────────────────────

def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
) -> float:
    """
    Optimise temperature scalar T via LBFGS to minimise cross-entropy NLL.
    Method: Guo et al. (2017) "On Calibration of Modern Neural Networks".

    Robust fitting to handle extreme eval-mode logits:
    1. Computes per-sample NLL at T=1 and excludes worst 1% of samples (at least 1 if n>=100)
       from the fitting set only. Logs how many were dropped and their NLL statistics.
    2. Optimises log_T (T = exp(log_T)) with LBFGS(lr=1.0, max_iter=max_iter,
       line_search_fn="strong_wolfe") on CPU float64 logits of the trimmed set.
    3. Clamps fitted T to [0.25, 5.0] [ENG assumption] and logs a loud warning pointing
       to scripts/diag_logits.py when clamped.
    4. Computes pre/post NLL and ECE on both all samples and trimmed samples, and logs them.
    5. Guards:
       - If post-fit ECE (on all samples) is higher than ECE at T=1.0, returns 1.0 and warns.
       - If post-fit NLL (on trimmed set) >= pre-fit NLL, returns 1.0 and warns.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)

    # Move to CPU float64 for stable numerical optimization
    logits_64 = logits.detach().cpu().to(torch.float64)
    labels_64 = labels.detach().cpu().to(torch.long)
    n_samples = len(labels_64)

    # 1. Per-sample NLL at T=1.0 to find worst 1%
    criterion_none = nn.CrossEntropyLoss(reduction="none")
    sample_nlls_t1 = criterion_none(logits_64, labels_64)

    if n_samples >= 100:
        k_drop = max(1, int(round(0.01 * n_samples)))
    else:
        k_drop = 0

    if k_drop > 0:
        sorted_indices = torch.argsort(sample_nlls_t1, descending=True)
        dropped_indices = sorted_indices[:k_drop]
        keep_indices = sorted_indices[k_drop:]
        dropped_nlls = sample_nlls_t1[dropped_indices]
        logger.info(
            "Trimming worst %d samples (1%% of %d) from calibration fitting set. "
            "Dropped NLLs: min=%.4f, max=%.4f, mean=%.4f (first 5: %s)",
            k_drop, n_samples,
            float(dropped_nlls.min().item()),
            float(dropped_nlls.max().item()),
            float(dropped_nlls.mean().item()),
            [round(float(v), 3) for v in dropped_nlls[:5].tolist()]
        )
        fit_logits = logits_64[keep_indices]
        fit_labels = labels_64[keep_indices]
    else:
        fit_logits = logits_64
        fit_labels = labels_64

    def _eval_metrics(lg, lb, temp):
        scaled = lg / max(temp, 1e-6)
        loss = float(nn.CrossEntropyLoss()(scaled, lb).item())
        probs = torch.softmax(scaled, dim=1).numpy()
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        corr = (preds == lb.numpy()).astype(float)
        ece = compute_ece(confs, corr)
        return loss, ece

    # Pre-calibration metrics
    all_nll_pre, all_ece_pre = _eval_metrics(logits_64, labels_64, 1.0)
    trim_nll_pre, trim_ece_pre = _eval_metrics(fit_logits, fit_labels, 1.0)

    # 2. Fit T on the trimmed set in log-space
    log_t = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    optimizer = optim.LBFGS(
        [log_t],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )
    criterion_mean = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_t)
        scaled = fit_logits / t
        loss = criterion_mean(scaled, fit_labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception as exc:
        logger.warning("LBFGS optimization error: %s. Falling back to T=1.0", exc)
        return 1.0

    raw_t = float(torch.exp(log_t).item())
    fitted_t = raw_t

    # 3. Clamp T to [0.25, 5.0] [ENG assumption] with loud warning
    clamped = False
    if raw_t < 0.25:
        logger.warning(
            "WARNING: Fitted temperature %.4f clamped to min 0.25 [ENG assumption]. "
            "Extreme logits detected -- run scripts/diag_logits.py to inspect.",
            raw_t
        )
        fitted_t = 0.25
        clamped = True
    elif raw_t > 5.0:
        logger.warning(
            "WARNING: Fitted temperature %.4f clamped to max 5.0 [ENG assumption]. "
            "Extreme logits detected -- run scripts/diag_logits.py to inspect.",
            raw_t
        )
        fitted_t = 5.0
        clamped = True

    # 4. Post-fit metrics
    all_nll_post, all_ece_post = _eval_metrics(logits_64, labels_64, fitted_t)
    trim_nll_post, trim_ece_post = _eval_metrics(fit_logits, fit_labels, fitted_t)

    logger.info(
        "Temperature Calibration: T=%.4f (raw=%.4f, clamped=%s) | "
        "All samples (n=%d): pre NLL=%.4f ECE=%.4f -> post NLL=%.4f ECE=%.4f | "
        "Trimmed (n=%d): pre NLL=%.4f ECE=%.4f -> post NLL=%.4f ECE=%.4f",
        fitted_t, raw_t, clamped,
        n_samples, all_nll_pre, all_ece_pre, all_nll_post, all_ece_post,
        len(fit_labels), trim_nll_pre, trim_ece_pre, trim_nll_post, trim_ece_post
    )

    # 5. Guards
    if all_ece_post > all_ece_pre:
        logger.warning(
            "WARNING: Post-fit ECE on all samples (%.4f) is higher than ECE at T=1.0 (%.4f). "
            "Calibration worsened ECE. Falling back to T=1.0",
            all_ece_post, all_ece_pre
        )
        return 1.0

    if trim_nll_post >= trim_nll_pre:
        logger.warning(
            "WARNING: Post-fit trimmed NLL (%.4f) >= pre-fit trimmed NLL (%.4f). "
            "Optimization failed to lower NLL. Falling back to T=1.0",
            trim_nll_post, trim_nll_pre
        )
        return 1.0

    return fitted_t

