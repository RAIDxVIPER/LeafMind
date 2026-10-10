"""
scripts/build_split_manifests.py
=================================
Generates D2 split manifest JSON files in config/manifests/.

PURPOSE:
  Produces canonical lists of image paths for each D2 split AFTER removing known
  within-split and cross-split duplicates identified in the dataset audit (Part 1).

  These manifests are used by the training pipeline (C2) and evaluation scripts (C10)
  to ensure reproducibility and correct data hygiene.

DEDUPLICATION APPLIED (from dataset_audit.md):
  H5: D2_Train within-split: 11 duplicate pairs (22 files → keep 11)
  H6: D2_Train <-> D2_Val: 4 cross-split duplicates → remove from Val
  H7: D2_Test_ID <-> D2_Train: 5 cross-split duplicates → remove from Test_ID
  H8: D2_Test_OOD within-split: 6 duplicate pairs (12 files → keep 6)

  NOTE: D1 exclusions for OOD evaluation are separate (config/exclusion_list.json, R7).
  This script only handles within-D2 deduplication for training/evaluation manifests.

  CONSERVATIVE CHOICE [ENG assumption, logged in plan.md]:
    When removing cross-split duplicates, we keep the file in the split with "higher
    priority" for the duplicate role (Train > Val > Test_ID by priority).
    When removing within-split duplicates:
      - If duplicate group has different class labels, remove ALL copies (label-conflicting).
      - If same class, keep the lexicographically first path.

OUTPUT FILES:
  config/manifests/d2_train.json      — paths after H5 dedup
  config/manifests/d2_val.json        — paths after H6 dedup
  config/manifests/d2_test_id.json    — paths after H7 dedup
  config/manifests/d2_test_ood.json   — paths after H8 dedup (D1 exclusions separate)
  config/manifests/d2_few_shot.json   — no known within-split dups; full list
  config/manifests/summary.json       — counts, dedup log, comparison with Part 1 audit

IMPORTANT: This script NEVER modifies Dataset/ or Dataset_2/ (R13).

Usage:
  python scripts/build_split_manifests.py [--d2-root DATASET_2_DIR]
                                          [--out-dir config/manifests]
                                          [--verbose]
"""

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from PIL import Image
    import imagehash
except ImportError:
    Image = None
    imagehash = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".tif"}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
        stream=sys.stdout,
    )


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_split(split_dir: Path) -> List[Tuple[str, str]]:
    """
    Walk split_dir and return (relative_path, md5) for every image file.
    relative_path is relative to split_dir.parent (the D2 root).
    """
    result = []
    count = 0
    for p in sorted(split_dir.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        md5 = md5_of_file(p)
        rel = str(p.relative_to(split_dir.parent))
        result.append((rel, md5))
        count += 1
        if count % 1000 == 0:
            logging.debug("  %s: hashed %d files ...", split_dir.name, count)
    logging.info("  %s: %d image files found.", split_dir.name, count)
    return result


def dedup_within_split(
    entries: List[Tuple[str, str]], split_name: str
) -> Tuple[List[Tuple[str, str]], List[Dict]]:
    """
    Remove within-split exact duplicates. Keeps the lexicographically first path,
    UNLESS the duplicates have different class labels, in which case ALL are removed.
    Returns (deduplicated entries, log of removed items).
    """
    by_md5 = defaultdict(list)
    for rel, md5 in entries:
        by_md5[md5].append((rel, md5))
        
    kept = []
    removed_log = []
    
    for md5, group in by_md5.items():
        if len(group) == 1:
            kept.append(group[0])
        else:
            classes = {Path(r).parts[1] if len(Path(r).parts) > 1 else "unknown" for r, _ in group}
            if len(classes) > 1:
                # Conflicting labels: remove ALL
                for r, _ in group:
                    removed_log.append({
                        "removed_path": r,
                        "kept_path": None,
                        "md5": md5,
                        "reason": f"label-conflicting duplicate in {split_name}"
                    })
            else:
                # Same class: keep first, remove rest
                kept.append(group[0])
                for r, _ in group[1:]:
                    removed_log.append({
                        "removed_path": r,
                        "kept_path": group[0][0],
                        "md5": md5,
                        "reason": f"within-split duplicate in {split_name}"
                    })
                    
    n_removed = len(removed_log)
    if n_removed > 0:
        logging.info("  %s: removed %d within-split duplicates.", split_name, n_removed)
    return kept, removed_log


def remove_cross_split_duplicates(
    entries: List[Tuple[str, str]],
    reference_md5s: Set[str],
    split_name: str,
    reason: str,
) -> Tuple[List[Tuple[str, str]], List[Dict]]:
    """
    Remove from `entries` any file whose MD5 is in `reference_md5s`.
    Used to remove Val/Test files that also exist in Train.
    """
    kept = []
    removed_log = []
    for rel, md5 in entries:
        if md5 in reference_md5s:
            removed_log.append({"removed_path": rel, "md5": md5, "reason": reason})
        else:
            kept.append((rel, md5))
    if removed_log:
        logging.info(
            "  %s: removed %d cross-split duplicates (%s).",
            split_name,
            len(removed_log),
            reason,
        )
    return kept, removed_log


def write_manifest(
    out_path: Path,
    entries: List[Tuple[str, str]],
    split_name: str,
    d2_root: Path,
    notes: List[str],
) -> None:
    manifest = {
        "_meta": {
            "split": split_name,
            "d2_root": "Dataset_2", # Path relative to workspace or configured by env var
            "image_count": len(entries),
            "notes": notes,
            "generated_by": "scripts/build_split_manifests.py",
        },
        "images": [
            {"path": rel, "md5": md5, "class": Path(rel).parts[1] if len(Path(rel).parts) > 1 else "unknown"}
            for rel, md5 in entries
        ],
    }
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logging.info("  Written: %s (%d images)", out_path.name, len(entries))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build D2 split manifests with within-D2 deduplication."
    )
    parser.add_argument("--d2-root", default="Dataset_2", help="Path to D2 root.")
    parser.add_argument(
        "--out-dir", default="config/manifests", help="Output directory for manifests."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    d2_root = Path(args.d2_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("LeafMind — build_split_manifests.py")
    logging.info("D2 root  : %s", d2_root.resolve())
    logging.info("Output   : %s", out_dir.resolve())
    logging.info("=" * 60)

    # ── Step 1: Hash all splits ────────────────────────────────────────────────
    logging.info("[Step 1] Hashing all D2 splits ...")
    splits_raw: Dict[str, List[Tuple[str, str]]] = {}
    for split in ["Train", "Val", "Test_ID", "Test_OOD", "Few_Shot"]:
        sdir = d2_root / split
        if not sdir.exists():
            logging.warning("Split directory missing: %s", sdir)
            splits_raw[split] = []
        else:
            splits_raw[split] = collect_split(sdir)

    # ── Step 2: Within-split dedup ────────────────────────────────────────────
    logging.info("[Step 2] Within-split deduplication ...")
    dedup_log: List[Dict] = []

    train_dedup, train_removed = dedup_within_split(splits_raw["Train"], "Train")
    dedup_log.extend(train_removed)

    ood_dedup, ood_removed = dedup_within_split(splits_raw["Test_OOD"], "Test_OOD")
    dedup_log.extend(ood_removed)

    # Val, Test_ID, Few_Shot: audit says 0 within-split dups — no action
    val_dedup = splits_raw["Val"]
    test_id_dedup = splits_raw["Test_ID"]
    few_shot_dedup = splits_raw["Few_Shot"]

    # ── Step 3: Cross-split dedup ─────────────────────────────────────────────
    logging.info("[Step 3] Cross-split deduplication ...")
    train_md5s: Set[str] = {md5 for _, md5 in train_dedup}

    # H6: Remove from Val files that exist in Train
    val_final, val_cross_removed = remove_cross_split_duplicates(
        val_dedup,
        train_md5s,
        "Val",
        "H6: D2_Train<->D2_Val duplicate — removed from Val",
    )
    dedup_log.extend(val_cross_removed)

    # H7: Remove from Test_ID files that exist in Train
    test_id_train_dedup, testid_cross_removed = remove_cross_split_duplicates(
        test_id_dedup,
        train_md5s,
        "Test_ID",
        "H7: D2_Test_ID<->D2_Train duplicate — removed from Test_ID",
    )
    dedup_log.extend(testid_cross_removed)

    # Additional fix: Test_ID <-> Val leak
    val_final_md5s = {md5 for _, md5 in val_final}
    test_id_final, testid_val_removed = remove_cross_split_duplicates(
        test_id_train_dedup,
        val_final_md5s,
        "Test_ID",
        "D2_Test_ID<->D2_Val duplicate — removed from Test_ID",
    )
    dedup_log.extend(testid_val_removed)

    # ── Step 4: Audit comparison ──────────────────────────────────────────────
    logging.info("[Step 4] Audit comparison vs Part 1 ...")

    # Raw counts from dataset_audit.md
    AUDIT = {
        "Train": {"raw": 27694, "within_dups_pairs": 11},
        "Val": {"raw": 5940, "cross_dups_with_train": 4},
        "Test_ID": {"raw": 5929, "cross_dups_with_train": 5},
        "Test_OOD": {"raw": 2374, "within_dups_pairs": 6},
        "Few_Shot": {"raw": 300},
    }

    results = {
        "Train": {"raw": len(splits_raw["Train"]), "after_dedup": len(train_dedup)},
        "Val": {"raw": len(splits_raw["Val"]), "after_dedup": len(val_final)},
        "Test_ID": {"raw": len(splits_raw["Test_ID"]), "after_dedup": len(test_id_final)},
        "Test_OOD": {"raw": len(splits_raw["Test_OOD"]), "after_dedup": len(ood_dedup)},
        "Few_Shot": {"raw": len(splits_raw["Few_Shot"]), "after_dedup": len(few_shot_dedup)},
    }

    mismatches = []
    logging.info("-" * 60)
    logging.info("%-12s %8s %10s %8s %10s", "Split", "RawAudit", "RawActual", "Expected", "Final")
    for split_name, audit_data in AUDIT.items():
        raw_actual = results[split_name]["raw"]
        raw_audit = audit_data["raw"]
        final = results[split_name]["after_dedup"]

        if raw_actual != raw_audit:
            msg = (
                f"MISMATCH {split_name}: raw count {raw_actual} != audit {raw_audit}. "
                "Dataset may have changed or path is wrong."
            )
            mismatches.append(msg)
            logging.warning(msg)

        logging.info(
            "%-12s %8d %10d %8s %10d",
            split_name,
            raw_audit,
            raw_actual,
            str(audit_data.get("raw", "?")),
            final,
        )
    logging.info("-" * 60)

    # ── Step 5: 47-file Few_Shot <-> Test_OOD filename overlap ────────────────
    logging.info("[Step 5] Computing Few_Shot <-> Test_OOD filename overlap ...")
    
    ood_paths_by_name = {Path(rel).name: rel for rel, _ in splits_raw["Test_OOD"]}
    fs_paths_by_name = {Path(rel).name: rel for rel, _ in splits_raw["Few_Shot"]}
    
    filename_overlap = sorted(set(ood_paths_by_name.keys()) & set(fs_paths_by_name.keys()))
    logging.info(
        "  Few_Shot <-> Test_OOD filename overlap: %d files (expected ~47).",
        len(filename_overlap),
    )
    
    # Compute pHash for filename overlaps
    verification = {
        "near_duplicates": 0,
        "different_images": 0,
        "details": []
    }
    
    if imagehash and Image:
        for fname in filename_overlap:
            ood_rel = ood_paths_by_name[fname]
            fs_rel = fs_paths_by_name[fname]
            ood_path = d2_root / ood_rel
            fs_path = d2_root / fs_rel
            
            try:
                ood_img = Image.open(ood_path)
                fs_img = Image.open(fs_path)
                ood_hash = imagehash.phash(ood_img)
                fs_hash = imagehash.phash(fs_img)
                dist = ood_hash - fs_hash
                is_near_dup = dist <= 8
                
                if is_near_dup:
                    verification["near_duplicates"] += 1
                else:
                    verification["different_images"] += 1
                    
                verification["details"].append({
                    "filename": fname,
                    "ood_path": ood_rel,
                    "few_shot_path": fs_rel,
                    "hamming_distance": dist,
                    "is_near_duplicate": is_near_dup
                })
            except Exception as e:
                logging.warning(f"Error computing phash for {fname}: {e}")
    else:
        logging.warning("imagehash or PIL not installed, skipping perceptual hash verification")

    if abs(len(filename_overlap) - 47) > 10:
        msg = f"MISMATCH: filename overlap = {len(filename_overlap)}, expected ~47 (I3, D8)."
        mismatches.append(msg)
        logging.warning(msg)

    # Write overlap file
    overlap_out = out_dir.parent / "few_shot_test_ood_overlap.json"
    overlap_data = {
        "_meta": {
            "description": (
                "47 filenames that appear in BOTH D2_Few_Shot and D2_Test_OOD. "
                "These files MUST be excluded from D2_Test_OOD when evaluating any model "
                "that was adapted using D2_Few_Shot (R7, D8, I3). "
                "This is a FILENAME overlap — byte-identity not confirmed per I3."
            ),
            "overlap_count": len(filename_overlap),
            "expected_count": 47,
            "generated_by": "scripts/build_split_manifests.py",
            "issue_ref": "I3",
            "note": "[ENG assumption] byte-identity not confirmed; use filename exclusion conservatively.",
        },
        "filenames": filename_overlap,
        "verification": verification
    }
    with open(overlap_out, "w") as fh:
        json.dump(overlap_data, fh, indent=2)
    logging.info("  Written: %s (%d entries)", overlap_out.name, len(filename_overlap))

    # ── Step 6: Write manifests ────────────────────────────────────────────────
    logging.info("[Step 6] Writing manifests ...")
    write_manifest(
        out_dir / "d2_train.json", train_dedup, "Train", d2_root,
        ["H5: within-split duplicates removed", f"{len(train_removed)} files removed"],
    )
    write_manifest(
        out_dir / "d2_val.json", val_final, "Val", d2_root,
        ["H6: cross-split dups with Train removed", f"{len(val_cross_removed)} files removed"],
    )
    write_manifest(
        out_dir / "d2_test_id.json", test_id_final, "Test_ID", d2_root,
        ["H7: cross-split dups with Train and Val removed", f"{len(testid_cross_removed) + len(testid_val_removed)} files removed",
         "D1 exclusions for OOD eval are separate (config/exclusion_list.json)"],
    )
    write_manifest(
        out_dir / "d2_test_ood.json", ood_dedup, "Test_OOD", d2_root,
        [
            "H8: within-split duplicates removed",
            f"{len(ood_removed)} files removed",
            "IMPORTANT: D1 exclusions (config/exclusion_list.json) are a secondary report only (D16). Primary OOD uses full deduped set.",
        ],
    )
    write_manifest(
        out_dir / "d2_few_shot.json", few_shot_dedup, "Few_Shot", d2_root,
        ["No within-split dups found in audit. Full list."],
    )

    # ── Step 7: Summary ────────────────────────────────────────────────────────
    summary = {
        "_meta": {
            "description": "D2 split manifest generation summary. Counts before and after dedup.",
            "generated_by": "scripts/build_split_manifests.py",
        },
        "counts": {
            split: {
                "raw": results[split]["raw"],
                "after_dedup": results[split]["after_dedup"],
                "removed": results[split]["raw"] - results[split]["after_dedup"],
            }
            for split in results
        },
        "dedup_log": dedup_log,
        "few_shot_test_ood_filename_overlap": len(filename_overlap),
        "audit_mismatches": mismatches,
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logging.info("  Written: summary.json")

    logging.info("=" * 60)
    if mismatches:
        logging.error("Completed with %d mismatch(es):", len(mismatches))
        for m in mismatches:
            logging.error("  • %s", m)
        sys.exit(1)
    else:
        logging.info("Split manifests complete -- no mismatches.")


if __name__ == "__main__":
    main()
