"""
src/agents/vision_agent.py
==========================
Vision Agent: wraps LeafClassifier to produce VisionOutput.

The Vision Agent is the entry point for all image classification in LeafMind.
It enforces:
  - RGB mode conversion for every input image [I9]
  - severity_pct = None always [R3/D2/Part 2a-xi]
  - OOD flag and escalation logic [ENG assumption thresholds]
  - Schema validation via Pydantic VisionOutput [R6]
  - ASCII-only logging [I27]
  - No D1 data in inference context [D14/R13]

YOLOv8 detector: OUT OF SCOPE for C2. leaf_detected is always False.
Full GPU training: handled by scripts/train_classifier.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.models.classifier import LeafClassifier
from src.schemas.schemas import VisionOutput

logger = logging.getLogger(__name__)


class VisionAgent:
    """
    Vision Agent for the LeafMind multi-agent pipeline.

    At inference time, wraps LeafClassifier.predict() to produce a VisionOutput
    that validates against the Pydantic schema defined in src/schemas/schemas.py.

    Modes:
      - Production: loaded from a real trained checkpoint via from_checkpoint()
      - Stub/test: initialised with a pre-built LeafClassifier (mock or real)

    IMPORTANT:
      This agent only wraps the EfficientNet-B3 classifier (C2).
      The YOLOv8 detector (optional, out of scope for C2) is not integrated here.
    """

    def __init__(
        self,
        classifier: LeafClassifier,
        model_version: str = "efficientnet_b3_v1",
    ):
        """
        Args:
            classifier: a LeafClassifier instance (real or mock)
            model_version: version string written into VisionOutput
        """
        self.classifier   = classifier
        self.model_version = model_version
        logger.info("VisionAgent ready (version=%s)", model_version)

    def predict(self, image_bytes: bytes, crop: Optional[str] = None) -> VisionOutput:
        """
        Run classification pipeline on raw image bytes.

        Steps:
          1. Decode + RGB-convert [I9]
          2. Quality gate [ENG assumption thresholds]
          3. EfficientNet-B3 inference with temperature scaling [ENG]
          4. If crop is given: mask logits to allowed classes, re-normalise via softmax,
             compute confidence/OOD/escalation from masked distribution, top-3 candidates.
          5. OOD flag + escalation flag [ENG assumption]
          6. Return VisionOutput (severity_pct=None always [R3])

        Args:
            image_bytes: raw JPEG or PNG image bytes
            crop: optional user-supplied crop name (e.g. "Corn", "Tomato", "Apple")

        Raises:
            RuntimeError: if the image cannot be decoded
            ValueError: if an unknown crop name is provided
        """
        output: VisionOutput = self.classifier.predict(image_bytes, crop=crop)
        logger.debug(
            "VisionAgent.predict: label=%s conf=%.3f crop=%s ood=%s esc=%s can_diag=%s",
            output.top_label, output.confidence, output.crop,
            output.ood_flag, output.escalate_to_human, output.can_diagnose,
        )
        return output

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device=None,
        **kwargs,
    ) -> "VisionAgent":
        """
        Load from a trained checkpoint directory.
        Expects: model.pth, temperature.json, class_map.json, config.json
        """
        clf = LeafClassifier.from_checkpoint(checkpoint_dir, device=device, **kwargs)
        return cls(classifier=clf, model_version=clf.model_version)
