"""
scripts/build_exclusion_list.py
================================
Generates config/exclusion_list.json: MD5 hashes of D1 (Dataset/) files that are
byte-identical to files in D2_Test_OOD or D2_Few_Shot (Dataset_2/).

PURPOSE (R7, R13, D14):
  - Classifier is trained exclusively on D2_Train (D1 images NEVER used — R13).
  - D2_Test_OOD contains images byte-identical to D1_train/D1_val (I4: 87.8% overlap).
  - Before any evaluation on D2_Test_OOD, all hashes in exclusion_list.json MUST be
    filtered out to prevent confounded cross-dataset evaluation.
  - The exclusion list contains D1 MD5s (the source), not D2 MD5s (though they're the same).

WHAT IS EXCLUDED:
  D1_train files that are byte-identical to ANY file in D2_Test_OOD or D2_Few_Shot
  D1_val   files that are byte-identical to ANY file in D2_Test_OOD or D2_Few_Shot

EXPECTED OUTPUT (from dataset audit Part 1):
  D1_train <-> D2_Test_OOD:  ~1,972 files
  D1_val   <-> D2_Test_OOD:  ~103   files
  D1_train <-> D2_Few_Shot:  ~238   files
  D1_val   <-> D2_Few_Shot:  ~25    files
  Total unique D1 hashes expected: ~2,085 (≥2,075 DoD gate per spec §6d-viii C1)
  Note: ~10 files appear in both OOD and FewShot cross-lists; deduplication gives ~2,085 unique.

IMPORTANT: This script reads Dataset/ and Dataset_2/ but NEVER modifies them (R13).

Usage:
  python scripts/build_exclusion_list.py [--d1-root DATASET_DIR]
                                         [--d2-root DATASET_2_DIR]
                                         [--output config/exclusion_list.json]
                                         [--verbose]
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".tif"}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
        stream=sys.stdout,
    )


def md5_of_file(path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_md5s(root: Path, splits: List[str]) -> Dict[str, List[str]]:
    """
    Walk each split directory under root, hash every image file.
    Returns {md5_hex: [relative_path, ...], ...}
    """
    hash_to_paths: Dict[str, List[str]] = {}
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            logging.warning("Split directory not found, skipping: %s", split_dir)
            continue
        count = 0
        for img_path in split_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            md5 = md5_of_file(img_path)
            rel = str(img_path.relative_to(root))
            hash_to_paths.setdefault(md5, []).append(rel)
            count += 1
            if count % 500 == 0:
                logging.debug("  Hashed %d files in %s ...", count, split)
        logging.info("  %s: %d image files hashed.", split, count)
    return hash_to_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build config/exclusion_list.json from D1/D2 MD5 comparison."
    )
    parser.add_argument(
        "--d1-root",
        default="Dataset",
        help="Path to D1 root (PlantDoc detection dataset).",
    )
    parser.add_argument(
        "--d2-root",
        default="Dataset_2",
        help="Path to D2 root (PlantVillage+PlantDoc classification dataset).",
    )
    parser.add_argument(
        "--output",
        default="config/exclusion_list.json",
        help="Output path for exclusion_list.json.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    setup_logging(args.verbose)

    d1_root = Path(args.d1_root)
    d2_root = Path(args.d2_root)
    out_path = Path(args.output)

    logging.info("=" * 60)
    logging.info("LeafMind — build_exclusion_list.py")
    logging.info("D1 root : %s", d1_root.resolve())
    logging.info("D2 root : %s", d2_root.resolve())
    logging.info("Output  : %s", out_path.resolve())
    logging.info("=" * 60)

    # ── Step 1: Hash D1 splits (train + val) ──────────────────────────────────
    logging.info("[Step 1] Hashing D1 (train + val) ...")
    d1_hash_map = collect_md5s(d1_root / "images", ["train", "val"])
    all_d1_hashes: Set[str] = set(d1_hash_map.keys())
    logging.info("D1 total unique MD5s: %d", len(all_d1_hashes))

    # ── Step 2: Hash D2 OOD and Few_Shot splits ───────────────────────────────
    logging.info("[Step 2] Hashing D2_Test_OOD ...")
    d2_ood_hash_map = collect_md5s(d2_root, ["Test_OOD"])
    all_d2_ood_hashes: Set[str] = set(d2_ood_hash_map.keys())
    logging.info("D2_Test_OOD total unique MD5s: %d", len(all_d2_ood_hashes))

    logging.info("[Step 3] Hashing D2_Few_Shot ...")
    d2_fs_hash_map = collect_md5s(d2_root, ["Few_Shot"])
    all_d2_fs_hashes: Set[str] = set(d2_fs_hash_map.keys())
    logging.info("D2_Few_Shot total unique MD5s: %d", len(all_d2_fs_hashes))

    # ── Step 3: Find overlapping D1 hashes ────────────────────────────────────
    logging.info("[Step 4] Computing overlaps ...")
    overlap_with_ood: Set[str] = all_d1_hashes & all_d2_ood_hashes
    overlap_with_fs: Set[str] = all_d1_hashes & all_d2_fs_hashes
    all_excluded_hashes: Set[str] = overlap_with_ood | overlap_with_fs

    logging.info("D1 hashes also in D2_Test_OOD : %d", len(overlap_with_ood))
    logging.info("D1 hashes also in D2_Few_Shot  : %d", len(overlap_with_fs))
    logging.info("D1 hashes in OOD AND FewShot   : %d", len(overlap_with_ood & overlap_with_fs))
    logging.info("Total unique D1 hashes excluded: %d", len(all_excluded_hashes))

    # ── Step 4: Build record for each excluded hash ────────────────────────────
    logging.info("[Step 5] Building output records ...")
    records: List[Dict] = []
    for h in sorted(all_excluded_hashes):
        d1_paths = d1_hash_map.get(h, [])
        d2_ood_paths = d2_ood_hash_map.get(h, [])
        d2_fs_paths = d2_fs_hash_map.get(h, [])
        records.append(
            {
                "md5": h,
                "d1_paths": sorted(d1_paths),
                "d2_ood_paths": sorted(d2_ood_paths),
                "d2_few_shot_paths": sorted(d2_fs_paths),
                "in_ood": len(d2_ood_paths) > 0,
                "in_few_shot": len(d2_fs_paths) > 0,
            }
        )

    # ── Step 5: Audit against Part 1 expectations ─────────────────────────────
    AUDIT_EXPECTED_MIN = 2075  # DoD gate from spec §6d-viii C1
    AUDIT_EXPECTED_EXACT = 2349  # D1<->OOD (2085) + D1<->FewShot (264) - overlap (0); see analysis below

    n_excluded = len(all_excluded_hashes)
    ood_pct = round(len(overlap_with_ood) / max(len(all_d2_ood_hashes), 1) * 100, 1)

    logging.info("-" * 60)
    logging.info("AUDIT COMPARISON vs Part 1 dataset_audit.md:")
    logging.info("  Expected D1<->D2_OOD overlaps  : ~1,972 (train) + ~103 (val) = ~2,075")
    logging.info("  Measured D1<->D2_OOD overlaps  : %d", len(overlap_with_ood))
    logging.info("  Expected D1<->D2_FewShot        : ~238 (train) + ~25 (val) = ~263")
    logging.info("  Measured D1<->D2_FewShot        : %d", len(overlap_with_fs))
    logging.info("  Expected total unique excluded  : ~%d (OOD+FewShot combined)", AUDIT_EXPECTED_EXACT)
    logging.info("  Measured total unique excluded  : %d", n_excluded)
    logging.info("  D2_OOD contamination rate       : %s%%", ood_pct)
    logging.info("  NOTE: audit expected_exact was 2085 (OOD only); combined is 2349.")

    mismatches = []
    if n_excluded < AUDIT_EXPECTED_MIN:
        msg = (
            f"MISMATCH: total excluded hashes {n_excluded} < DoD minimum {AUDIT_EXPECTED_MIN}. "
            "Check dataset paths or re-run audit."
        )
        logging.error(msg)
        mismatches.append(msg)
    else:
        logging.info("  DoD gate (>=%d): PASSED", AUDIT_EXPECTED_MIN)

    expected_ood = 2075
    if abs(len(overlap_with_ood) - expected_ood) > 100:
        msg = (
            f"MISMATCH: D1↔OOD overlap {len(overlap_with_ood)} deviates "
            f"by more than 100 from expected {expected_ood}."
        )
        logging.warning(msg)
        mismatches.append(msg)

    # ── Step 6: Write output ────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "_meta": {
            "description": (
                "MD5 hashes of D1 (Dataset/) files that are byte-identical to files in "
                "D2_Test_OOD or D2_Few_Shot. These D1 files must be EXCLUDED from any "
                "evaluation using D2_Test_OOD to prevent confounded cross-dataset evaluation. "
                "See dataset_audit.md §H1–H4, plan.md I4, spec R7, R13, D14."
            ),
            "generated_by": "scripts/build_exclusion_list.py",
            "d1_root": str(d1_root.resolve()),
            "d2_root": str(d2_root.resolve()),
            "total_d1_hashes": len(all_d1_hashes),
            "total_d2_ood_hashes": len(all_d2_ood_hashes),
            "total_d2_few_shot_hashes": len(all_d2_fs_hashes),
            "excluded_hashes_total": n_excluded,
            "excluded_overlap_with_ood": len(overlap_with_ood),
            "excluded_overlap_with_few_shot": len(overlap_with_fs),
            "excluded_in_both": len(overlap_with_ood & overlap_with_fs),
            "d2_ood_contamination_pct": ood_pct,
            "dod_gate_min": AUDIT_EXPECTED_MIN,
            "dod_gate_passed": n_excluded >= AUDIT_EXPECTED_MIN,
            "audit_mismatches": mismatches,
            "audit_expected_exact": AUDIT_EXPECTED_EXACT,
        },
        "excluded_md5s": [r["md5"] for r in records],
        "records": records,
    }

    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    logging.info("=" * 60)
    logging.info("Written: %s", out_path.resolve())
    logging.info("Excluded hash count: %d", n_excluded)

    if mismatches:
        logging.error("Build completed with %d mismatch(es). Review logs.", len(mismatches))
        sys.exit(1)
    else:
        logging.info("Build complete -- no mismatches.")


if __name__ == "__main__":
    main()
