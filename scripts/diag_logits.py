"""
scripts/diag_logits.py  -- DIAGNOSTIC ONLY (reads a checkpoint, writes nothing)

Why: in the C2 dry run the temperature fit went to T=1002 (clamped to 20) and
post-calibration ECE got WORSE (0.11 -> 0.61), while pre-calibration NLL was
15.6 at T=1. That pattern means a few samples have huge raw logits. This script
shows which samples, and whether eval-mode BatchNorm statistics are the cause.

Usage (from repo root, PowerShell):
  python scripts/diag_logits.py --checkpoint-dir checkpoints/dry2 --data-root Dataset_2 --n 400

ASCII-only output [I27]. No fabricated numbers: everything is computed here.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


# --------------------------------------------------------------------------
# Pure numpy helpers (no torch needed; unit-checkable)
# --------------------------------------------------------------------------

def log_softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def ece_np(conf, correct, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += abs(conf[mask].mean() - correct[mask].mean()) * mask.sum() / n
    return float(ece)


def metrics_at_T(logits, labels, T):
    lsm = log_softmax_np(np.asarray(logits, dtype=np.float64) / T)
    n = len(labels)
    nll = -lsm[np.arange(n), labels]
    conf = np.exp(lsm.max(axis=1))
    correct = (lsm.argmax(axis=1) == labels).astype(float)
    k = max(1, int(round(0.01 * n)))
    trimmed = np.sort(nll)[: n - k].mean()
    return {
        "T": T,
        "mean_nll": float(nll.mean()),
        "trimmed_nll_drop_worst_1pct": float(trimmed),
        "ece": ece_np(conf, correct),
        "mean_conf": float(conf.mean()),
        "acc": float(correct.mean()),
    }


def summarize(logits, labels, paths, idx_to_label, title, top_worst=12):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(labels)
    nonfinite = int((~np.isfinite(logits)).any(axis=1).sum())
    lsm = log_softmax_np(logits)
    nll = -lsm[np.arange(n), labels]
    preds = logits.argmax(axis=1)
    srt = np.sort(logits, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    absmax = np.abs(logits).max(axis=1)
    conf = np.exp(lsm.max(axis=1))
    correct = preds == labels

    def pct(a, qs=(50, 90, 99, 100)):
        return " ".join("p%d=%.1f" % (q, np.percentile(a, q)) for q in qs)

    print("")
    print("=" * 70)
    print(title)
    print("=" * 70)
    print("n=%d  non-finite rows=%d  accuracy=%.4f" % (n, nonfinite, correct.mean()))
    print("mean NLL=%.3f  median NLL=%.3f  frac NLL>10=%.3f  frac NLL>20=%.3f" % (
        nll.mean(), np.median(nll), (nll > 10).mean(), (nll > 20).mean()))
    print("max |logit|  : " + pct(absmax))
    print("top1-top2    : " + pct(margin))
    print("confidence   : mean=%.3f  frac>0.99=%.3f" % (conf.mean(), (conf > 0.99).mean()))

    order = np.argsort(-nll)[:top_worst]
    print("")
    print("worst %d samples by NLL:" % len(order))
    for i in order:
        p = Path(str(paths[i]).replace("\\", "/"))
        print("  nll=%8.2f  maxlogit=%8.1f  margin=%7.1f  true=%s  pred=%s  file=%s/%s" % (
            nll[i], srt[i, -1], margin[i],
            idx_to_label[int(labels[i])], idx_to_label[int(preds[i])],
            p.parent.name, p.name[:40]))

    bad = np.where(nll > 10)[0]
    if len(bad):
        cnt = Counter(idx_to_label[int(labels[i])] for i in bad)
        print("")
        print("true classes among samples with NLL>10 (%d samples):" % len(bad))
        for k, v in cnt.most_common(8):
            print("  %-45s %d" % (k, v))


def print_T_grid(logits, labels, title):
    print("")
    print("T grid (%s):" % title)
    print("  %6s %9s %9s %7s %9s" % ("T", "mean_nll", "trim_nll", "ECE", "mean_conf"))
    for T in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0):
        m = metrics_at_T(logits, labels, T)
        print("  %6.2f %9.3f %9.3f %7.4f %9.3f" % (
            T, m["mean_nll"], m["trimmed_nll_drop_worst_1pct"], m["ece"], m["mean_conf"]))


# --------------------------------------------------------------------------
# Main (torch imports kept inside so the helpers above import without torch)
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Diagnose raw D2_Val logits (read-only).")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n", type=int, default=400, help="stratified val sample size")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HERE))
    import train_classifier as tc
    from src.models.classifier import EfficientNetB3Classifier

    cfg = tc.load_config(REPO_ROOT / "config" / "classifier_config.yaml")
    with open(REPO_ROOT / "config" / "class_map_d2.json") as f:
        idx_to_label = {int(k): v for k, v in json.load(f)["classes"].items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}

    with open(REPO_ROOT / "config" / "manifests" / "d2_val.json") as f:
        images = json.load(f)["images"]
    images = tc.stratified_sample(images, args.n, random.Random(42))

    ds = tc.D2Dataset(images, Path(args.data_root), tc.build_eval_transform(cfg), label_to_idx)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = Path(args.checkpoint_dir)
    wpath = ckpt / "model_best.pth"
    if not wpath.exists():
        wpath = ckpt / "model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB3Classifier(num_classes=len(idx_to_label), pretrained=False)
    model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
    model.to(device)
    print("loaded %s on %s; %d val images" % (wpath, device, len(ds)))

    def collect(m):
        outs, labs = [], []
        with torch.no_grad():
            for x, y in loader:
                outs.append(m(x.to(device), calibrate=False).float().cpu().numpy())
                labs.append(y.numpy())
        return np.concatenate(outs), np.concatenate(labs)

    # 1) normal eval mode (what training/calibration used)
    model.eval()
    logits_eval, labels = collect(model)
    paths = [e["path"] for e in images]
    summarize(logits_eval, labels, paths, idx_to_label, "EVAL MODE (BatchNorm running stats)")
    print_T_grid(logits_eval, labels, "eval mode")

    # 2) same weights, BatchNorm using per-batch statistics (dropout off)
    m2 = copy.deepcopy(model)
    m2.eval()
    for mod in m2.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.train()
    logits_bn, labels2 = collect(m2)
    summarize(logits_bn, labels2, paths, idx_to_label, "BATCH-STATS BN (diagnostic only)")
    print_T_grid(logits_bn, labels2, "batch-stats BN")

    print("")
    print("Reading guide:")
    print(" - If eval-mode max|logit| and NLL are huge but batch-stats BN looks sane,")
    print("   BatchNorm running statistics are the cause (short fine-tune / batch size).")
    print(" - If both are huge, look at the worst-sample list: same class / same file type?")
    print(" - Pick a T from the grid by ECE and trimmed NLL, not by mean NLL alone.")


if __name__ == "__main__":
    main()
