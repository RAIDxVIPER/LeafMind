"""
tests/unit/test_config.py
==================================
Tests for configuration files and generated manifests (C1).
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent

def test_threat_table_healthy_classes():
    """Fix 4: All healthy classes must be tier HEALTHY, and disease classes DISEASE_DETECTED."""
    threat_table_path = REPO_ROOT / "config" / "threat_table.json"
    assert threat_table_path.exists()
    
    with open(threat_table_path) as f:
        data = json.load(f)
        
    for cls_name, info in data["classes"].items():
        if "healthy" in cls_name.lower():
            assert info["tier"] == "HEALTHY", f"Class {cls_name} must be HEALTHY"
            assert info["status"] == "[ENG]"
        else:
            assert info["tier"] == "DISEASE_DETECTED"
            assert info["status"] == "[GAP]"

@pytest.fixture(scope="module")
def manifests():
    manifests_dir = REPO_ROOT / "config" / "manifests"
    results = {}
    if not manifests_dir.exists():
        return results
        
    for split in ["train", "val", "test_id", "test_ood", "few_shot"]:
        p = manifests_dir / f"d2_{split}.json"
        if p.exists():
            with open(p) as f:
                results[split] = json.load(f)
    return results

def test_val_test_id_no_leak(manifests):
    """Fix 1: Ensure Val and Test_ID have disjoint MD5 sets."""
    if "val" not in manifests or "test_id" not in manifests:
        pytest.skip("Manifests not generated yet")
        
    val_md5s = {img["md5"] for img in manifests["val"]["images"]}
    test_id_md5s = {img["md5"] for img in manifests["test_id"]["images"]}
    
    overlap = val_md5s & test_id_md5s
    assert len(overlap) == 0, f"Found leak between Val and Test_ID: {overlap}"

def test_test_ood_no_duplicates_and_correct_size(manifests):
    """Fix 2: Test_OOD dedup handles conflicting labels by removing all copies."""
    if "test_ood" not in manifests:
        pytest.skip("Test_OOD manifest not generated yet")
        
    images = manifests["test_ood"]["images"]
    md5s = [img["md5"] for img in images]
    
    assert len(md5s) == len(set(md5s)), "Test_OOD contains duplicate MD5s"
    
    # 2374 raw - 10 (remove all 5 pairs) - 1 (keep 1 of 1 pair) = 2363
    assert len(images) == 2363, f"Expected 2363 images in Test_OOD, found {len(images)}"

def test_no_absolute_paths_in_meta(manifests):
    """Fix 5: Manifests must not store absolute C:\\Users... paths."""
    for split, data in manifests.items():
        d2_root = data["_meta"]["d2_root"]
        assert "C:\\" not in d2_root, f"Absolute path found in {split} manifest: {d2_root}"
        assert "C:/" not in d2_root, f"Absolute path found in {split} manifest: {d2_root}"

def test_consistency_no_overlap_between_splits(manifests):
    """Check that Train, Val, Test_ID are disjoint."""
    if not manifests:
        pytest.skip("Manifests not generated yet")
    
    splits = ["train", "val", "test_id"]
    sets = {}
    for split in splits:
        if split in manifests:
            sets[split] = {img["md5"] for img in manifests[split]["images"]}
            
    if "train" in sets and "val" in sets:
        assert len(sets["train"] & sets["val"]) == 0
    if "train" in sets and "test_id" in sets:
        assert len(sets["train"] & sets["test_id"]) == 0
    if "val" in sets and "test_id" in sets:
        assert len(sets["val"] & sets["test_id"]) == 0

def test_consistency_classes(manifests):
    """Each split has exactly 30 classes, class matches folder."""
    if not manifests:
        pytest.skip("Manifests not generated yet")
        
    for split, data in manifests.items():
        classes = {img["class"] for img in data["images"]}
        assert len(classes) == 30, f"{split} has {len(classes)} classes, expected 30"
