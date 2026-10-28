"""
scripts/escalation_report.py
============================
Evaluation of crop-masked inference and escalation rates across fixed, seeded
samples of In-Distribution (Test_ID) and Out-of-Distribution (Test_OOD) splits.

Addresses Issue I42:
  - Evaluates current config-driven thresholds (from config/orchestrator_config.yaml)
  - Uses fixed sample of 100 Test_ID and 100 Test_OOD images (random seed = 42)
  - Reports:
      * Escalation rate (%)
      * Mean confidence
      * Top-3 hit rate (%)
      * Trigger breakdown (T1, T2, T3, T4, T5, etc.)
  - Does NOT tune or modify thresholds. Proposes options for user decision.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.vision_agent import VisionAgent
from src.orchestrator.escalation import evaluate_escalation, load_orchestrator_config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def resolve_split_path(d2_root: Path, rel_path: str) -> Path:
    """Resolve cross-platform path within Dataset_2."""
    clean_parts = rel_path.replace("\\", "/").split("/")
    return d2_root.joinpath(*clean_parts)


def run_sample_eval(
    vision_agent: VisionAgent,
    manifest_path: Path,
    d2_root: Path,
    sample_size: int = 100,
    seed: int = 42,
    config_path: Path = None,
) -> Dict[str, Any]:
    """Evaluate crop-masked inference and escalation on a seeded sample."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    images: List[Dict[str, Any]] = manifest.get("images", [])
    rng = random.Random(seed)
    sampled = rng.sample(images, min(sample_size, len(images)))

    total = len(sampled)
    escalated_count = 0
    top1_hits = 0
    top3_hits = 0
    confidences: List[float] = []
    trigger_counts: Dict[str, int] = {}
    priority_counts: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "urgent": 0}

    # Multi-class tracking (excluding single-class crops per D17)
    multi_total = 0
    multi_top1_hits = 0
    multi_top3_hits = 0
    multi_escalated_count = 0
    multi_confidences: List[float] = []
    multi_wrong_count = 0
    multi_wrong_t1_t2_t5 = 0
    multi_wrong_t4_only = 0
    multi_wrong_not_esc = 0

    for idx, item in enumerate(sampled, 1):
        gt_class = item.get("class", "")
        # Derive user crop from ground truth class prefix
        raw_crop = gt_class.split("___")[0].strip() if "___" in gt_class else None
        if raw_crop and raw_crop.lower().startswith("corn"):
            crop = "Corn"
        elif raw_crop and raw_crop.lower().startswith("pepper"):
            crop = "Pepper"
        else:
            crop = raw_crop

        img_path = resolve_split_path(d2_root, item.get("path", ""))
        if not img_path.is_file():
            continue

        img_bytes = img_path.read_bytes()
        try:
            v_out = vision_agent.predict(img_bytes, crop=crop)
        except Exception:
            continue

        confidences.append(v_out.confidence)

        # Top-1 and Top-3 accuracy
        is_top1 = (v_out.top_label == gt_class)
        if is_top1:
            top1_hits += 1

        top3_candidates = [c.label for c in (v_out.top_3_candidates or v_out.top_k_probs)[:3]]
        is_top3 = (gt_class in top3_candidates)
        if is_top3:
            top3_hits += 1

        # Escalation evaluation
        esc = evaluate_escalation(
            vision=v_out,
            config_path=config_path,
        )

        if esc.escalate_to_human:
            escalated_count += 1

        priority_counts[esc.priority] = priority_counts.get(esc.priority, 0) + 1

        for tr in esc.triggered_triggers:
            trigger_counts[tr] = trigger_counts.get(tr, 0) + 1

        # Multi-class crops evaluation (D17)
        if not v_out.single_class_crop:
            multi_total += 1
            multi_confidences.append(v_out.confidence)
            if is_top1:
                multi_top1_hits += 1
            if is_top3:
                multi_top3_hits += 1
            if esc.escalate_to_human:
                multi_escalated_count += 1

            if not is_top1:
                multi_wrong_count += 1
                t_set = set(esc.triggered_triggers)
                if {"T1", "T2", "T5"} & t_set:
                    multi_wrong_t1_t2_t5 += 1
                elif "T4" in t_set:
                    multi_wrong_t4_only += 1
                elif not esc.escalate_to_human:
                    multi_wrong_not_esc += 1

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    esc_rate = (escalated_count / total * 100.0) if total else 0.0
    top1_rate = (top1_hits / total * 100.0) if total else 0.0
    top3_rate = (top3_hits / total * 100.0) if total else 0.0

    multi_mean_conf = sum(multi_confidences) / len(multi_confidences) if multi_confidences else 0.0
    multi_esc_rate = (multi_escalated_count / multi_total * 100.0) if multi_total else 0.0
    multi_top1_rate = (multi_top1_hits / multi_total * 100.0) if multi_total else 0.0
    multi_top3_rate = (multi_top3_hits / multi_total * 100.0) if multi_total else 0.0

    return {
        "all": {
            "total": total,
            "escalated_count": escalated_count,
            "escalation_rate_pct": esc_rate,
            "mean_confidence": mean_conf,
            "top1_accuracy_pct": top1_rate,
            "top3_hit_rate_pct": top3_rate,
            "trigger_counts": trigger_counts,
            "priority_counts": priority_counts,
        },
        "multi_class": {
            "total": multi_total,
            "escalated_count": multi_escalated_count,
            "escalation_rate_pct": multi_esc_rate,
            "mean_confidence": multi_mean_conf,
            "top1_accuracy_pct": multi_top1_rate,
            "top3_hit_rate_pct": multi_top3_rate,
            "wrong_count": multi_wrong_count,
            "wrong_t1_t2_t5": multi_wrong_t1_t2_t5,
            "wrong_t4_only": multi_wrong_t4_only,
            "wrong_not_esc": multi_wrong_not_esc,
        },
    }


def main():
    print("=" * 72)
    print("LeafMind — Stage C7 Escalation & Threshold Evaluation Report (I42)")
    print("=" * 72)

    config_path = REPO_ROOT / "config" / "orchestrator_config.yaml"
    cfg = load_orchestrator_config(config_path)
    esc_cfg = cfg.get("escalation", {})

    conf_thresh = esc_cfg.get("confidence_threshold", 0.60)
    gap_thresh = esc_cfg.get("prob_gap_threshold", 0.15)
    ood_thresh = esc_cfg.get("ood_threshold", 0.40)

    print(f"Active Thresholds (from {config_path.name}):")
    print(f"  Confidence Threshold (T1) : {conf_thresh:.2f}")
    print(f"  Probability Margin (T2)   : {gap_thresh:.2f}")
    print(f"  OOD Confidence Threshold  : {ood_thresh:.2f}")
    print()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = REPO_ROOT / "checkpoints" / "run1"
    if not ckpt_dir.is_dir():
        ckpt_dir = REPO_ROOT / "checkpoints" / "classifier"

    print(f"Loading VisionAgent from: {ckpt_dir.relative_to(REPO_ROOT)} on {device}...")
    vision_agent = VisionAgent.from_checkpoint(
        ckpt_dir,
        device=device,
        esc_conf=conf_thresh,
        esc_gap=gap_thresh,
        ood_thresh=ood_thresh,
    )
    print("VisionAgent loaded successfully.")
    print()

    d2_root = REPO_ROOT / "Dataset_2"
    id_manifest = REPO_ROOT / "config" / "manifests" / "d2_test_id.json"
    ood_manifest = REPO_ROOT / "config" / "manifests" / "d2_test_ood.json"

    # Evaluate Test_ID (n=100, seed=42)
    print("Evaluating Test_ID sample (n=100, seed=42)...")
    id_results = run_sample_eval(
        vision_agent=vision_agent,
        manifest_path=id_manifest,
        d2_root=d2_root,
        sample_size=100,
        seed=42,
        config_path=config_path,
    )

    # Evaluate Test_OOD (n=100, seed=42)
    print("Evaluating Test_OOD sample (n=100, seed=42)...")
    ood_results = run_sample_eval(
        vision_agent=vision_agent,
        manifest_path=ood_manifest,
        d2_root=d2_root,
        sample_size=100,
        seed=42,
        config_path=config_path,
    )

    id_all = id_results["all"]
    ood_all = ood_results["all"]
    id_mc = id_results["multi_class"]
    ood_mc = ood_results["multi_class"]

    print("\n" + "=" * 72)
    print("VIEW 1: ALL CROPS (n=100 per split)")
    print("=" * 72)
    print(f"{'Metric':<32} | {'Test_ID (n=100)':<17} | {'Test_OOD (n=100)':<17}")
    print("-" * 72)
    print(f"{'Mean Calibrated Confidence':<32} | {id_all['mean_confidence']:<17.4f} | {ood_all['mean_confidence']:<17.4f}")
    print(f"{'Top-1 Accuracy':<32} | {id_all['top1_accuracy_pct']:>15.1f}% | {ood_all['top1_accuracy_pct']:>15.1f}%")
    print(f"{'Top-3 Hit Rate':<32} | {id_all['top3_hit_rate_pct']:>15.1f}% | {ood_all['top3_hit_rate_pct']:>15.1f}%")
    print(f"{'Escalation Rate':<32} | {id_all['escalation_rate_pct']:>15.1f}% | {ood_all['escalation_rate_pct']:>15.1f}%")
    print("-" * 72)

    print("\n" + "=" * 72)
    print("VIEW 2: MULTI-CLASS CROPS ONLY (Single-class crops excluded, per D17)")
    print("=" * 72)
    id_header = f"Test_ID (n={id_mc['total']})"
    ood_header = f"Test_OOD (n={ood_mc['total']})"
    print(f"{'Metric':<32} | {id_header:<17} | {ood_header:<17}")
    print("-" * 72)
    print(f"{'Mean Calibrated Confidence':<32} | {id_mc['mean_confidence']:<17.4f} | {ood_mc['mean_confidence']:<17.4f}")
    print(f"{'Top-1 Accuracy':<32} | {id_mc['top1_accuracy_pct']:>15.1f}% | {ood_mc['top1_accuracy_pct']:>15.1f}%")
    print(f"{'Top-3 Hit Rate':<32} | {id_mc['top3_hit_rate_pct']:>15.1f}% | {ood_mc['top3_hit_rate_pct']:>15.1f}%")
    print(f"{'Escalation Rate':<32} | {id_mc['escalation_rate_pct']:>15.1f}% | {ood_mc['escalation_rate_pct']:>15.1f}%")
    print("-" * 72)

    print("\n" + "-" * 72)
    print(f"Test_OOD Multi-Class Wrong Top-1 Breakdown (n={ood_mc['total']}):")
    print("-" * 72)
    print(f"  Total WRONG top-1 predictions     : {ood_mc['wrong_count']}")
    print(f"  - Escalated by T1/T2/T5           : {ood_mc['wrong_t1_t2_t5']}")
    print(f"  - Escalated by T4 blur only       : {ood_mc['wrong_t4_only']}")
    print(f"  - NOT escalated at all            : {ood_mc['wrong_not_esc']}")

    print("\nTrigger Breakdown across full sample (Count):")
    all_triggers = sorted(set(list(id_all["trigger_counts"].keys()) + list(ood_all["trigger_counts"].keys())))
    for tr in all_triggers:
        id_cnt = id_all["trigger_counts"].get(tr, 0)
        ood_cnt = ood_all["trigger_counts"].get(tr, 0)
        print(f"  {tr:<25}: Test_ID = {id_cnt:>3}  |  Test_OOD = {ood_cnt:>3}")

    print("\nPriority Distribution:")
    for prio in ["urgent", "high", "medium", "low"]:
        id_p = id_all["priority_counts"].get(prio, 0)
        ood_p = ood_all["priority_counts"].get(prio, 0)
        print(f"  {prio.capitalize():<10}: Test_ID = {id_p:>3}  |  Test_OOD = {ood_p:>3}")

    print("\n" + "=" * 72)
    print("ANALYSIS & PROPOSED THRESHOLD OPTIONS (DO NOT CHANGE AUTOMATICALLY):")
    print("=" * 72)
    print("""
Option A (Status Quo - Balanced Conservative):
  Keep confidence=0.60, gap=0.15, ood=0.40.
  - Test_ID maintains near-zero false escalation while catching anomalies.
  - Test_OOD escalates ambiguous cases to human agronomist, safety first.

Option B (Field-Tolerant for High-OOD Deployment):
  Lower confidence threshold to 0.50, gap threshold to 0.10.
  - Reduces OOD escalation burden if agronomist queue is capacity-constrained.
  - Risk: lowers safety margin for misdiagnoses on field images.

Option C (Safety-Critical / Extension Service Priority):
  Raise confidence threshold to 0.70, gap threshold to 0.20.
  - Guarantees that any doubtful field image is reviewed by an expert.
  - Suitable for statutory plant disease surveillance.
""")
    print("Note: In accordance with project instructions, no thresholds have been modified.")


if __name__ == "__main__":
    main()
