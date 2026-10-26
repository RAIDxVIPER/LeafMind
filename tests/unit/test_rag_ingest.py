"""
tests/unit/test_rag_ingest.py
==============================
Unit tests for knowledge base PDF parsing, chunking, and metadata tagging (Stage C3).
Verifies:
  1. All 63 chapters yield all 6 standard sections + "At a glance".
  2. Chunk metadata schema (crop, disease_name, model_class, section, chunk_type, tags, file, page).
  3. "At a glance" chunk has chunk_type=glance.
  4. Statutory caveats and CIB&RC dates attached to every Remedies chunk.
  5. Intro/summary/appendix chunks are stored and excluded from farmer index.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from typing import List

from src.rag.ingest import (
    KBChunk,
    ingest_knowledge_base,
    STANDARD_SECTIONS,
    DEFAULT_KB_DIR,
    DEFAULT_CLASS_MAP_PATH,
)


@pytest.fixture(scope="module")
def ingested_data():
    chunks, reports = ingest_knowledge_base(DEFAULT_KB_DIR, DEFAULT_CLASS_MAP_PATH)
    return chunks, reports


class TestRAGIngestion:

    def test_all_eight_pdfs_present(self, ingested_data):
        chunks, reports = ingested_data
        assert len(reports) == 8, f"Expected 8 PDF reports, got {len(reports)}"
        crops = {r["crop"] for r in reports}
        assert crops == {"Apple", "Blueberry", "Cherry", "Grape", "Corn", "Tomato", "Potato", "Pepper,_bell"}

    EXPECTED_CHAPTER_COUNTS = {
        "Tomato": 12,
        "Corn": 18,
        "Grape": 11,
        "Cherry": 6,
        "Apple": 9,
        "Blueberry": 7,
        "Potato": 10,
        "Pepper,_bell": 7,
    }

    EXPECTED_CHAPTER_TITLES = {
        "Apple": [
            "Apple Scab",
            "Marssonina Leaf Blotch (Premature Leaf Fall)",
            "Powdery Mildew",
            "Collar Rot",
            "White Root Rot",
            "Alternaria Leaf Blotch",
            "Sooty Blotch and Flyspeck",
            "Fire Blight",
            "Cedar Apple Rust",
        ],
        "Blueberry": [
            "Phytophthora Root Rot",
            "Botrytis Blight (Gray Mould)",
            "Anthracnose and Ripe Rot (Fruit Rots)",
            "Phomopsis Twig Blight and Botryosphaeria Stem Blight / Canker",
            "Mummy Berry",
            "Leaf and Fruit Spots (Septoria, Exobasidium, Rust)",
            "Bacterial Leaf Scorch",
        ],
        "Cherry": [
            "Brown Rot",
            "Bacterial Canker (Gummosis)",
            "Cherry Leaf Spot (Shot Hole)",
            "Powdery Mildew",
            "Crown Gall",
            "Collar Rot / Root and Crown Rot (Phytophthora)",
        ],
        "Grape": [
            "Downy Mildew",
            "Powdery Mildew",
            "Anthracnose (Bird's-eye Rot)",
            "Bacterial Leaf Spot",
            "Alternaria Leaf Blight and Bunch Necrosis",
            "Post-harvest Berry Rots (Botrytis, Aspergillus, Rhizopus and others)",
            "Leaf Rust",
            "Black Rot",
            "Bitter Rot",
            "Dead Arm (Phomopsis Cane and Leaf Spot)",
            "Esca (Black Measles)",
        ],
        "Corn": [
            "Turcicum Leaf Blight (Northern Corn Leaf Blight)",
            "Maydis Leaf Blight (Southern Corn Leaf Blight)",
            "Banded Leaf and Sheath Blight",
            "Common Rust",
            "Polysora (Southern) Rust",
            "Downy Mildew (Sorghum Downy Mildew, with Rajasthan and Brown Stripe types)",
            "Charcoal Rot (Post-Flowering Stalk Rot)",
            "Fusarium Stalk Rot (Post-Flowering Stalk Rot)",
            "Bacterial Stalk Rot",
            "Curvularia Leaf Spot",
            "Brown Spot",
            "Zonate Leaf Spot",
            "Gray Leaf Spot",
            "Pythium Stalk Rot",
            "Late Wilt",
            "Ear Rots and Mycotoxin Risk (Fusarium, Gibberella, Aspergillus, Diplodia, Penicillium)",
            "Common Smut and Head Smut",
            "Seed Rot, Seedling Blight and Root Rot",
        ],
        "Tomato": [
            "Damping Off",
            "Early Blight",
            "Late Blight",
            "Septoria Leaf Spot",
            "Fusarium Wilt",
            "Bacterial Wilt",
            "Bacterial Stem and Fruit Canker",
            "Tomato Leaf Curl Disease",
            "Tomato Spotted Wilt / Bud Necrosis",
            "Bacterial Leaf Spot (Bacterial Spot)",
            "Leaf Mould",
            "Tomato Mosaic Virus (and related mosaic viruses)",
        ],
        "Potato": [
            "Late Blight",
            "Early Blight (Alternaria Leaf Spot)",
            "Black Scurf and Stem Canker",
            "Common Scab",
            "Bacterial Wilt (Brown Rot)",
            "Soft Rot and Blackleg",
            "Fusarium Dry Rot",
            "Silver Scurf",
            "Potato Viruses (Leaf Roll, Y, X and Apical Leaf Curl)",
            "Potato Wart (Quarantine Disease)",
        ],
        "Pepper,_bell": [
            "Bacterial Spot (Bacterial Leaf Spot)",
            "Anthracnose, Fruit Rot and Die-back",
            "Powdery Mildew",
            "Cercospora Leaf Spot (Frog-eye)",
            "Damping-off (Nursery)",
            "Fusarium Wilt (Root Wilt)",
            "Chilli Leaf Curl and Other Viral Diseases",
        ],
    }

    def test_all_chapters_yield_all_six_sections(self, ingested_data):
        """Report any chapter where a section is missing instead of failing silently."""
        chunks, reports = ingested_data
        total_chapters = sum(r["chapters_found"] for r in reports)
        assert total_chapters == 80, f"Expected 80 total chapters across 8 PDFs, got {total_chapters}"

        all_missing = {}
        for r in reports:
            if r["missing_sections"]:
                all_missing[r["filename"]] = r["missing_sections"]

        assert not all_missing, f"Chapters with missing sections detected: {all_missing}"

    def test_exact_expected_chapter_counts_per_crop(self, ingested_data):
        """Assert exact expected chapter counts per crop: Tomato 12, Maize 18, Grape 11, Cherry 6, Apple 9, Blueberry 7, Potato 10, Pepper 7."""
        chunks, reports = ingested_data
        counts = {r["crop"]: r["chapters_found"] for r in reports}
        assert counts == self.EXPECTED_CHAPTER_COUNTS, f"Chapter count mismatch: {counts}"

    def test_exact_expected_chapter_titles_per_file(self, ingested_data):
        """Assert exact expected chapter titles in order per PDF file."""
        chunks, reports = ingested_data
        for crop, expected_titles in self.EXPECTED_CHAPTER_TITLES.items():
            seen = []
            for c in chunks:
                if c.crop == crop and c.indexed_for_farmer:
                    if c.disease_name not in seen:
                        seen.append(c.disease_name)
            assert seen == expected_titles, f"Titles mismatch for {crop}:\nFound: {seen}\nExpected: {expected_titles}"

    def test_grape_esca_black_measles_exists_and_mapped(self, ingested_data):
        """Confirm Grape Esca (Black Measles) exists, has all sections, and is mapped to Grape___Esca_(Black_Measles)."""
        chunks, _ = ingested_data
        esca_chunks = [c for c in chunks if c.crop == "Grape" and "esca" in c.disease_name.lower()]
        assert len(esca_chunks) == 7, f"Expected 7 chunks for Esca, got {len(esca_chunks)}"
        for c in esca_chunks:
            assert c.model_class == "Grape___Esca_(Black_Measles)"

    def test_potato_and_pepper_mapped_classes_exist(self, ingested_data):
        """Confirm Potato and Pepper mapped classes exist, have all sections, and are correctly mapped."""
        chunks, _ = ingested_data
        mapped_expected = [
            ("Potato", "Potato___Late_blight", "Late Blight"),
            ("Potato", "Potato___Early_blight", "Early Blight (Alternaria Leaf Spot)"),
            ("Pepper,_bell", "Pepper,_bell___Bacterial_spot", "Bacterial Spot (Bacterial Leaf Spot)"),
        ]
        for crop, model_class, title in mapped_expected:
            matching = [c for c in chunks if c.crop == crop and c.model_class == model_class]
            assert len(matching) == 7, f"Expected 7 chunks for {model_class}, got {len(matching)}"
            sections = {c.section for c in matching}
            assert sections == set(STANDARD_SECTIONS) | {"At a glance"}

    def test_glance_chunk_type_per_chapter(self, ingested_data):
        """Every chapter must yield an 'At a glance' chunk with chunk_type='glance'."""
        chunks, _ = ingested_data
        glance_chunks = [c for c in chunks if c.chunk_type == "glance"]
        assert len(glance_chunks) == 80, f"Expected 80 glance chunks, got {len(glance_chunks)}"
        for g in glance_chunks:
            assert g.section == "At a glance"
            assert len(g.text) > 20
            assert g.indexed_for_farmer is True

    def test_standard_sections_present_per_chapter(self, ingested_data):
        chunks, _ = ingested_data
        by_disease = {}
        for c in chunks:
            if c.indexed_for_farmer:
                by_disease.setdefault((c.crop, c.disease_name), []).append(c)

        assert len(by_disease) == 80
        for (crop, disease), ch_chunks in by_disease.items():
            sections = {c.section for c in ch_chunks}
            expected = set(STANDARD_SECTIONS) | {"At a glance"}
            missing = expected - sections
            assert not missing, f"Crop {crop} Disease {disease} missing sections: {missing}"

    def test_caveats_attached_to_every_remedies_chunk(self, ingested_data):
        """Keep the 'confirm current label' caveat and CIB&RC date note attached to every Remedies chunk."""
        chunks, _ = ingested_data
        remedies = [c for c in chunks if c.section == "Remedies and treatment"]
        assert len(remedies) == 80

        for r in remedies:
            assert len(r.caveats) >= 1, f"Missing caveats in {r.chunk_id}"
            assert "Caveats and regulatory notes:" in r.text
            # Verify confirmation of label or registration is present
            lower_text = r.text.lower()
            assert any(term in lower_text for term in ["label", "cib&rc", "registration", "status"])

    def test_intro_summary_appendix_stored_and_excluded_for_farmer(self, ingested_data):
        """Do not index intro/summary/appendix chunks for farmer answers (store them, but exclude by default)."""
        chunks, _ = ingested_data
        non_farmer = [c for c in chunks if not c.indexed_for_farmer]
        assert len(non_farmer) >= 16  # At least 8 intro + 8 summary table

        for nf in non_farmer:
            assert nf.chunk_type in ("intro", "summary", "appendix", "sources")
            assert nf.indexed_for_farmer is False
            assert len(nf.text) > 0

    def test_verification_tags_extracted(self, ingested_data):
        chunks, _ = ingested_data
        all_v_tags = set()
        for c in chunks:
            all_v_tags.update(c.verification_tags)
        assert "V1" in all_v_tags
        assert "V2" in all_v_tags
        assert "V3" in all_v_tags
