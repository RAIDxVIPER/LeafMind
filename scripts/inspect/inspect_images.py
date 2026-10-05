"""
scripts/inspect/inspect_images.py
----------------------------------
Inspection script (NOT application code). Runs in ~5 min total.

Optimisations vs previous version:
  - Uses concurrent.futures.ThreadPoolExecutor for I/O-bound PIL reads
  - Perceptual hashing scoped to D1 train+val + D2 OOD + D2 FewShot only
  - D2 PV splits (Train/Val/Test_ID): MD5+dims only, no phash

Writes:
  docs/image_audit_raw.json
  docs/bbox_samples/  (20 annotated JPEGs)
"""

import os, sys, json, hashlib, random, time, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
import imagehash

ROOT     = Path(__file__).resolve().parents[2]
DOCS     = ROOT / "docs"
BBOX_DIR = DOCS / "bbox_samples"
DOCS.mkdir(exist_ok=True)
BBOX_DIR.mkdir(exist_ok=True)

D1_IMG_TRAIN = ROOT / "Dataset" / "images" / "train"
D1_IMG_VAL   = ROOT / "Dataset" / "images" / "val"
D1_LBL_TRAIN = ROOT / "Dataset" / "labels" / "train"
D2_ROOT      = ROOT / "Dataset_2"

YAML_NAMES = {
    0:"Cherry leaf",1:"Peach leaf",2:"Corn leaf blight",3:"Apple rust leaf",
    4:"Potato leaf late blight",5:"Strawberry leaf",6:"Corn rust leaf",
    7:"Tomato leaf late blight",8:"Tomato mold leaf",9:"Potato leaf early blight",
    10:"Apple leaf",11:"Tomato leaf yellow virus",12:"Blueberry leaf",
    13:"Tomato leaf mosaic virus",14:"Raspberry leaf",15:"Tomato leaf bacterial spot",
    16:"Squash Powdery mildew",17:"grape leaf",18:"Corn Gray leaf spot",
    19:"Tomato Early blight leaf",20:"Apple Scab Leaf",21:"Tomato Septoria leaf spot",
    22:"Tomato leaf",23:"Soyabean leaf",24:"Bell_pepper leaf spot",
    25:"Bell_pepper leaf",26:"grape leaf black rot",27:"Potato leaf",
    28:"Tomato spider mites leaf",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def _collect_flat(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})

def _collect_classfolder(split_path):
    """Return list of (path, class_name)."""
    out = []
    for cls_dir in sorted(Path(split_path).iterdir()):
        if cls_dir.is_dir():
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}:
                    out.append((p, cls_dir.name))
    return out

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(131072), b""):
            h.update(chunk)
    return h.hexdigest()

def _probe(path):
    """Return (w, h, mode, corrupt, md5)."""
    digest = _md5(path)
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
        return w, h, mode, False, digest
    except Exception as e:
        return None, None, None, True, digest

def audit_split(name, path_cls_pairs, workers=8):
    """Probe all images in parallel. Returns summary dict + md5_map."""
    records = []
    md5_map = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_meta = {ex.submit(_probe, p): (p, cls)
                       for p, cls in path_cls_pairs}
        done = 0
        for fut in as_completed(fut_to_meta):
            p, cls = fut_to_meta[fut]
            w, h, mode, corrupt, digest = fut.result()
            records.append({"path": str(p), "cls": cls,
                             "w": w, "h": h, "mode": mode,
                             "corrupt": corrupt, "md5": digest})
            md5_map.setdefault(digest, []).append(str(p))
            done += 1
            if done % 2000 == 0:
                print(f"  [{name}] {done}/{len(path_cls_pairs)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    corrupt_list = [r["path"] for r in records if r["corrupt"]]
    widths  = [r["w"] for r in records if r["w"]]
    heights = [r["h"] for r in records if r["h"]]
    modes   = collections.Counter(r["mode"] for r in records if r["mode"])
    exact_dups = {h: ps for h, ps in md5_map.items() if len(ps) > 1}

    elapsed = time.time() - t0
    print(f"  [{name}] DONE: {len(records)} images in {elapsed:.1f}s "
          f"| corrupt={len(corrupt_list)} | exact_dup_groups={len(exact_dups)}", flush=True)

    return {
        "name": name,
        "total": len(records),
        "corrupt_count": len(corrupt_list),
        "corrupt_files": corrupt_list,
        "width_min":  min(widths)  if widths else None,
        "width_max":  max(widths)  if widths else None,
        "width_mean": round(sum(widths)/len(widths), 1) if widths else None,
        "height_min":  min(heights)  if heights else None,
        "height_max":  max(heights)  if heights else None,
        "height_mean": round(sum(heights)/len(heights), 1) if heights else None,
        "mode_dist": dict(modes),
        "exact_dup_groups_within": len(exact_dups),
        "exact_dup_files_within":  sum(len(v) for v in exact_dups.values()),
        "within_dups_sample": {h: ps for h, ps in list(exact_dups.items())[:5]},
        "_md5_map": md5_map,   # used for cross-split check, stripped before JSON save
    }

# ── PASS 1: all splits ───────────────────────────────────────────────────────
print("=== PASS 1: Image dims + MD5 (all images, threaded) ===")
results = {}

# D1 flat
results["D1_train"] = audit_split("D1_train",
    [(p, None) for p in _collect_flat(D1_IMG_TRAIN)])
results["D1_val"] = audit_split("D1_val",
    [(p, None) for p in _collect_flat(D1_IMG_VAL)])

# D2 class-folder splits
for split_name, folder in [
    ("D2_Train",    D2_ROOT / "Train"),
    ("D2_Val",      D2_ROOT / "Val"),
    ("D2_Test_ID",  D2_ROOT / "Test_ID"),
    ("D2_Test_OOD", D2_ROOT / "Test_OOD"),
    ("D2_Few_Shot", D2_ROOT / "Few_Shot"),
]:
    results[split_name] = audit_split(split_name, _collect_classfolder(folder))

# ── PASS 2: cross-split MD5 duplicates ──────────────────────────────────────
print("\n=== PASS 2: Cross-split MD5 duplicates ===")
all_md5 = {}   # md5 -> list of (split, path)
for sname, res in results.items():
    for h, paths in res["_md5_map"].items():
        for p in paths:
            all_md5.setdefault(h, []).append((sname, p))

cross_dups = {h: entries for h, entries in all_md5.items()
              if len(set(e[0] for e in entries)) > 1}
pair_counts = collections.Counter()
for h, entries in cross_dups.items():
    key = tuple(sorted(set(e[0] for e in entries)))
    pair_counts[key] += 1

print(f"  Cross-split MD5 duplicates: {len(cross_dups)} hash groups")
for pair, cnt in pair_counts.most_common(20):
    print(f"  {' vs '.join(pair)}: {cnt}")

# ── PASS 3: perceptual hashing (scoped) ─────────────────────────────────────
print("\n=== PASS 3: Perceptual hashing (D1 train+val + D2 OOD + D2 FewShot) ===")
PHASH_SPLITS = ["D1_train", "D1_val", "D2_Test_OOD", "D2_Few_Shot"]
PHASH_THRESHOLD = 8

def _phash_path(path):
    try:
        with Image.open(path).convert("RGB") as img:
            return str(imagehash.phash(img))
    except Exception:
        return None

phash_records = []   # list of (split, path, phash_str)
phash_errors  = 0
for sname in PHASH_SPLITS:
    paths = [r["path"] for r in []  ]   # placeholder; rebuild from md5_map keys
    # rebuild path list from md5_map values
    all_paths = []
    for paths_list in results[sname]["_md5_map"].values():
        all_paths.extend(paths_list)
    all_paths = list(set(all_paths))   # deduplicate
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_phash_path, p): p for p in all_paths}
        for fut in as_completed(futs):
            p = futs[fut]
            ph = fut.result()
            if ph is None:
                phash_errors += 1
            else:
                phash_records.append((sname, p, ph))
    print(f"  [{sname}] {len(all_paths)} images phashed in {time.time()-t0:.1f}s")

# Group by phash
phash_groups = {}
for sname, p, ph in phash_records:
    phash_groups.setdefault(ph, []).append((sname, p))

exact_phash_dups = {ph: entries for ph, entries in phash_groups.items()
                    if len(entries) > 1}
cross_phash_dups = {ph: entries for ph, entries in exact_phash_dups.items()
                    if len(set(e[0] for e in entries)) > 1}

print(f"  Perceptual errors: {phash_errors}")
print(f"  Exact phash dup groups (in scope): {len(exact_phash_dups)}")
print(f"  Cross-split exact phash dups: {len(cross_phash_dups)}")
print(f"  D2 PlantVillage splits (Train/Val/Test_ID): phash NOT checked (39K images, would take >1h)")

# Hamming near-dups: D1 vs D2_OOD+FewShot
print("  Computing hamming near-dups D1 vs D2_OOD/FewShot...")
d1_hashes   = [(sn, p, ph) for sn, p, ph in phash_records if sn.startswith("D1")]
ood_hashes  = [(sn, p, ph) for sn, p, ph in phash_records if sn in ("D2_Test_OOD","D2_Few_Shot")]

near_dups_d1_ood = []
for s1, p1, ph1 in d1_hashes:
    ih1 = imagehash.hex_to_hash(ph1)
    for s2, p2, ph2 in ood_hashes:
        dist = ih1 - imagehash.hex_to_hash(ph2)
        if dist <= PHASH_THRESHOLD:
            near_dups_d1_ood.append({"split_a":s1,"path_a":p1,
                                      "split_b":s2,"path_b":p2,"hamming":dist})

print(f"  D1 vs D2_OOD/FewShot near-dups (hamming<={PHASH_THRESHOLD}): {len(near_dups_d1_ood)}")

# ── PASS 4: BBox visualisation (20 random D1 train images) ──────────────────
print("\n=== PASS 4: Bounding-box visualisation (20 random D1 train images) ===")
train_imgs = _collect_flat(D1_IMG_TRAIN)
random.seed(42)
sample = random.sample(train_imgs, min(20, len(train_imgs)))

def draw_boxes(img_path, lbl_path, out_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font = ImageFont.truetype("arial.ttf", max(14, H // 35))
    except Exception:
        font = ImageFont.load_default()
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cid  = int(parts[0])
                cx,cy,bw,bh = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
                x1,y1 = int((cx-bw/2)*W), int((cy-bh/2)*H)
                x2,y2 = int((cx+bw/2)*W), int((cy+bh/2)*H)
                draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
                label = YAML_NAMES.get(cid, str(cid))
                draw.text((x1+3,y1+3), label, fill="yellow", font=font)
                boxes.append({"cid":cid,"name":label,
                              "bw":round(bw,4),"bh":round(bh,4),
                              "area_frac":round(bw*bh,4)})
    img.thumbnail((900,900))
    img.save(out_path, quality=85)
    return boxes

bbox_sample_data = []
for img_path in sample:
    lbl_path = D1_LBL_TRAIN / (img_path.stem + ".txt")
    out_path = BBOX_DIR / (img_path.stem + "_vis.jpg")
    boxes = draw_boxes(img_path, lbl_path, out_path)
    avg_area = round(sum(b["area_frac"] for b in boxes)/len(boxes), 4) if boxes else None
    bbox_sample_data.append({"image": img_path.name,
                              "num_boxes": len(boxes),
                              "avg_box_area_frac": avg_area,
                              "boxes": boxes})
    print(f"  {img_path.name}: {len(boxes)} boxes avg_area={avg_area}")

# Aggregate ALL train label box areas
print("  Aggregating bbox areas across all train labels...")
all_areas, all_bw, all_bh = [], [], []
for lf in sorted(D1_LBL_TRAIN.iterdir()):
    for line in lf.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            bw, bh = float(parts[3]), float(parts[4])
            all_areas.append(bw*bh)
            all_bw.append(bw); all_bh.append(bh)

n = len(all_areas)
sa = sorted(all_areas)
def pct(q): return round(sa[int(q*n)], 5)
bbox_agg = {
    "total_annotations": n,
    "area_frac_min":  pct(0.00), "area_frac_p10": pct(0.10),
    "area_frac_p25":  pct(0.25), "area_frac_p50": pct(0.50),
    "area_frac_p75":  pct(0.75), "area_frac_p90": pct(0.90),
    "area_frac_p95":  pct(0.95), "area_frac_max": pct(0.999),
    "area_frac_mean": round(sum(all_areas)/n, 5),
    "pct_boxes_gt_50pct_image": round(sum(1 for a in all_areas if a > 0.5)/n, 4),
    "pct_boxes_gt_25pct_image": round(sum(1 for a in all_areas if a > 0.25)/n, 4),
    "mean_bw": round(sum(all_bw)/n,4),
    "mean_bh": round(sum(all_bh)/n,4),
}
print(f"  n={n} mean_area={bbox_agg['area_frac_mean']} "
      f"p50={bbox_agg['area_frac_p50']} "
      f">50%_image={bbox_agg['pct_boxes_gt_50pct_image']:.1%} "
      f">25%_image={bbox_agg['pct_boxes_gt_25pct_image']:.1%}")

# ── Save JSON ────────────────────────────────────────────────────────────────
print("\n=== Saving image_audit_raw.json ===")

# Strip internal _md5_map before serialising
summary = {}
for k, v in results.items():
    summary[k] = {kk: vv for kk, vv in v.items() if kk != "_md5_map"}

output = {
    "splits_summary": summary,
    "cross_split_md5": {
        "total_cross_dup_groups": len(cross_dups),
        "by_split_pair": {str(k): v for k, v in pair_counts.most_common()},
        "sample": {h: entries[:4] for h, entries in list(cross_dups.items())[:10]},
    },
    "perceptual_hashing": {
        "scope": "D1_train, D1_val, D2_Test_OOD, D2_Few_Shot",
        "d2_pv_splits_checked": False,
        "d2_pv_skipped_reason": "39K+ images; phash per image ~0.1s = >1h; not checked",
        "threshold_hamming": PHASH_THRESHOLD,
        "exact_phash_dup_groups_in_scope": len(exact_phash_dups),
        "cross_split_exact_phash_dups": len(cross_phash_dups),
        "d1_vs_d2_ood_neardup_count": len(near_dups_d1_ood),
        "d1_vs_d2_ood_neardup_sample": near_dups_d1_ood[:10],
        "phash_errors": phash_errors,
        "cross_phash_sample": {ph: entries for ph, entries
                                in list(cross_phash_dups.items())[:5]},
    },
    "bbox_visualisation": {
        "sample_size": len(bbox_sample_data),
        "sample_details": bbox_sample_data,
        "aggregate_all_train_labels": bbox_agg,
    },
}

out_path = DOCS / "image_audit_raw.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)
print(f"Saved: {out_path}")
print("DONE.")
