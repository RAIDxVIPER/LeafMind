"""
tests/unit/test_rag_agent.py
============================
Unit tests for RAG Agent advisory generation and extractive assembly (Stage C3).
Verifies:
  1. Chemical doses hidden by default (include_doses=False -> chemical_options=[]).
  2. Chemical doses present with include_doses=True and properly tagged (V1/V2/V3/US/--).
  3. Citations non-empty for every found disease result.
  4. what_to_do_first comes from Prevention and management, is extractive and short.
  5. process_vision_output correctly processes VisionOutput top-3 candidates.
  6. Index rebuild is deterministic (same chunk count twice).
"""
from __future__ import annotations

import pytest

from src.rag.agent import RAGAgent
from src.rag.index import KBIndex
from src.rag.ingest import ingest_knowledge_base
from src.schemas.schemas import ClassProb, ImageQualityFlags, VisionOutput


@pytest.fixture(scope="module")
def agent():
    index = KBIndex()
    index.load()
    return RAGAgent()


class TestRAGAgent:

    def test_doses_hidden_by_default(self, agent):
        """Doses hidden by default, empty unless include_doses=True."""
        ro = agent.query_candidate("Tomato___Late_blight", crop="Tomato", include_doses=False)
        assert ro.found is True
        assert ro.chemical_options == [], "chemical_options MUST be empty when include_doses=False"

    def test_doses_present_with_include_doses_true(self, agent):
        """Chemical doses are returned only when include_doses=True and carry verification tags."""
        ro = agent.query_candidate("Tomato___Late_blight", crop="Tomato", include_doses=True)
        assert ro.found is True
        assert len(ro.chemical_options) > 0, "chemical_options must not be empty when include_doses=True"
        for opt in ro.chemical_options:
            assert opt.verification_tag in ("V1", "V2", "V3", "US", "--")
            assert len(opt.product) > 0

    def test_citations_non_empty_for_every_found_result(self, agent):
        """Citations non-empty for every found result across all 17 mapped classes."""
        mapped_classes = [
            ("Apple___Apple_scab", "Apple"),
            ("Apple___Cedar_apple_rust", "Apple"),
            ("Grape___Black_rot", "Grape"),
            ("Grape___Esca_(Black_Measles)", "Grape"),
            ("Corn___Northern_Leaf_Blight", "Corn"),
            ("Corn___Common_rust", "Corn"),
            ("Corn___Cercospora_leaf_spot", "Corn"),
            ("Tomato___Early_blight", "Tomato"),
            ("Tomato___Late_blight", "Tomato"),
            ("Tomato___Septoria_leaf_spot", "Tomato"),
            ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato"),
            ("Tomato___Bacterial_spot", "Tomato"),
            ("Tomato___Leaf_Mold", "Tomato"),
            ("Tomato___Tomato_mosaic_virus", "Tomato"),
            ("Potato___Late_blight", "Potato"),
            ("Potato___Early_blight", "Potato"),
            ("Pepper,_bell___Bacterial_spot", "Pepper,_bell"),
        ]

        for model_class, crop in mapped_classes:
            ro = agent.query_candidate(model_class, crop=crop)
            assert ro.found is True, f"Expected found=True for {model_class}"
            assert len(ro.citations) > 0, f"Citations must be non-empty for {model_class}"
            for cit in ro.citations:
                assert cit.tag, f"Citation tag missing in {model_class}"
                assert cit.title, f"Citation title missing in {model_class}"
                if cit.url:
                    assert not cit.url.endswith("-"), f"Citation URL must not end with hyphen in {model_class}: {cit.url}"

    def test_no_citation_url_ends_with_hyphen(self, agent):
        """Add a test that no citation URL ends with a hyphen across all 17 mapped classes."""
        mapped_classes = [
            ("Apple___Apple_scab", "Apple"),
            ("Apple___Cedar_apple_rust", "Apple"),
            ("Grape___Black_rot", "Grape"),
            ("Grape___Esca_(Black_Measles)", "Grape"),
            ("Corn___Northern_Leaf_Blight", "Corn"),
            ("Corn___Common_rust", "Corn"),
            ("Corn___Cercospora_leaf_spot", "Corn"),
            ("Tomato___Early_blight", "Tomato"),
            ("Tomato___Late_blight", "Tomato"),
            ("Tomato___Septoria_leaf_spot", "Tomato"),
            ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato"),
            ("Tomato___Bacterial_spot", "Tomato"),
            ("Tomato___Leaf_Mold", "Tomato"),
            ("Tomato___Tomato_mosaic_virus", "Tomato"),
            ("Potato___Late_blight", "Potato"),
            ("Potato___Early_blight", "Potato"),
            ("Pepper,_bell___Bacterial_spot", "Pepper,_bell"),
        ]
        all_urls = []
        for model_class, crop in mapped_classes:
            ro = agent.query_candidate(model_class, crop=crop)
            for cit in ro.citations:
                if cit.url:
                    all_urls.append(cit.url)
                    assert not cit.url.endswith("-"), f"Citation URL ends with hyphen in {model_class}: {cit.url}"
        assert len(all_urls) > 0, "Expected to test at least one citation URL"

    def test_what_to_do_first_extractive_and_short(self, agent):
        """what_to_do_first comes from Prevention and management, extractive and short."""
        mapped_classes = [
            ("Tomato___Late_blight", "Tomato"),
            ("Apple___Apple_scab", "Apple"),
            ("Corn___Common_rust", "Corn"),
            ("Grape___Black_rot", "Grape"),
            ("Potato___Late_blight", "Potato"),
            ("Potato___Early_blight", "Potato"),
            ("Pepper,_bell___Bacterial_spot", "Pepper,_bell"),
        ]

        for model_class, crop in mapped_classes:
            ro = agent.query_candidate(model_class, crop=crop, include_doses=False)
            assert ro.what_to_do_first is not None
            assert len(ro.what_to_do_first) > 10, f"what_to_do_first too short for {model_class}"
            assert len(ro.what_to_do_first) < 400, f"what_to_do_first too long for {model_class}"
            # Verify it does not mention chemical dose formulas
            assert "ml/ha" not in ro.what_to_do_first
            assert "kg/ha" not in ro.what_to_do_first
            if model_class == "Pepper,_bell___Bacterial_spot":
                assert "(TXT, supplementary)" not in ro.what_to_do_first
                assert "Regional note" not in ro.what_to_do_first

    def test_process_vision_output(self, agent):
        """process_vision_output takes VisionOutput and returns RAGOutput per candidate."""
        vo = VisionOutput(
            top_label="Tomato___Late_blight",
            confidence=0.92,
            top_k_probs=[
                ClassProb(label="Tomato___Late_blight", prob=0.92),
                ClassProb(label="Tomato___Early_blight", prob=0.06),
                ClassProb(label="Tomato___healthy", prob=0.02),
            ],
            image_quality_flags=ImageQualityFlags(),
            model_version="efficientnet_b3_v1",
            temperature=1.0,
            crop="Tomato",
            top_3_candidates=[
                ClassProb(label="Tomato___Late_blight", prob=0.92),
                ClassProb(label="Tomato___Early_blight", prob=0.06),
                ClassProb(label="Tomato___healthy", prob=0.02),
            ],
        )

        results = agent.process_vision_output(vo, include_doses=False)
        assert len(results) == 3

        # Candidate 1: Late Blight (found=True)
        assert results[0].model_class == "Tomato___Late_blight"
        assert results[0].found is True
        assert results[0].chemical_options == []

        # Candidate 2: Early Blight (found=True)
        assert results[1].model_class == "Tomato___Early_blight"
        assert results[1].found is True

        # Candidate 3: healthy (found=False, reason="healthy_class")
        assert results[2].model_class == "Tomato___healthy"
        assert results[2].found is False
        assert results[2].reason == "healthy_class"

    def test_index_rebuild_is_deterministic(self):
        """Index rebuild is deterministic (same chunk count twice)."""
        chunks1, reports1 = ingest_knowledge_base()
        chunks2, reports2 = ingest_knowledge_base()

        assert len(chunks1) == len(chunks2) == 584
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2

    def test_dose_scrub_policy_all_17_classes(self, agent):
        """
        Verify dose scrubbing policy [D28]:
        1. When include_doses=False: no narrative field matches dose/rate or named chemical patterns.
        2. When include_doses=True: original text is returned completely unchanged.
        """
        import re

        mapped_classes = [
            ("Apple___Apple_scab", "Apple"),
            ("Apple___Cedar_apple_rust", "Apple"),
            ("Grape___Black_rot", "Grape"),
            ("Grape___Esca_(Black_Measles)", "Grape"),
            ("Corn___Northern_Leaf_Blight", "Corn"),
            ("Corn___Common_rust", "Corn"),
            ("Corn___Cercospora_leaf_spot", "Corn"),
            ("Tomato___Early_blight", "Tomato"),
            ("Tomato___Late_blight", "Tomato"),
            ("Tomato___Septoria_leaf_spot", "Tomato"),
            ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato"),
            ("Tomato___Bacterial_spot", "Tomato"),
            ("Tomato___Leaf_Mold", "Tomato"),
            ("Tomato___Tomato_mosaic_virus", "Tomato"),
            ("Potato___Late_blight", "Potato"),
            ("Potato___Early_blight", "Potato"),
            ("Pepper,_bell___Bacterial_spot", "Pepper,_bell"),
        ]

        banned_patterns = [
            re.compile(r"\b\d+(?:\.\d+)?\s*(?:g|kg|mg|ml|l|L)\s*(?:/|per)\s*\d*\s*(?:kg|l|L|ha|acre|ac|litre|100\s*L)\b", re.I),
            re.compile(r"\b\d+(?:\.\d+)?\s*%\s*(?:SL|SP|EC|WP|WG|SC|DP|FS)\b", re.I),
            re.compile(r"\b\d+(?:\.\d+)?\s*%\s*\+\s*\d+(?:\.\d+)?\s*%\s*(?:SL|SP|EC|WP|WG|SC|DP|FS)?\b", re.I),
            re.compile(r"\b(?:ppm|mg/L|per 100\s*L)\b", re.I),
            re.compile(r"\b(?:kasugamycin|streptocycline|bordeaux paste|5%\s*boric)\b", re.I),
            re.compile(r"\burea\s+spray\b", re.I),
        ]

        fields = [
            "what_it_is",
            "why_it_happens",
            "what_to_do_first",
            "prevention",
            "look_alikes",
            "favourable_conditions",
        ]

        for model_class, crop in mapped_classes:
            ro_scrubbed = agent.query_candidate(model_class, crop=crop, include_doses=False)
            ro_original = agent.query_candidate(model_class, crop=crop, include_doses=True)

            # Check that scrubbed fields contain NO banned dose patterns
            for f in fields:
                val = getattr(ro_scrubbed, f)
                if val:
                    for pat in banned_patterns:
                        m = pat.search(val)
                        assert not m, f"Dose leak in {model_class}.{f}: '{m.group(0)}' in '{val[:80]}...'"

            for item in ro_scrubbed.treatment_options:
                for pat in banned_patterns:
                    m = pat.search(item)
                    assert not m, f"Dose leak in {model_class}.treatment_options: '{m.group(0)}'"

            for item in ro_scrubbed.preventive_measures:
                for pat in banned_patterns:
                    m = pat.search(item)
                    assert not m, f"Dose leak in {model_class}.preventive_measures: '{m.group(0)}'"

            # Check that with include_doses=True, original chemical references remain intact
            if model_class == "Apple___Apple_scab":
                assert "urea spray" in ro_original.prevention
                assert "5 kg per 100 L" in ro_original.prevention
            elif model_class == "Grape___Esca_(Black_Measles)":
                assert "5% boric" in ro_original.prevention
                assert "Bordeaux paste" in ro_original.prevention
            elif model_class == "Tomato___Early_blight":
                assert "4 g/kg" in ro_original.prevention
            elif model_class == "Tomato___Late_blight":
                assert "1.0 kg/ac" in ro_original.prevention
                assert "Maharashtra note" in ro_scrubbed.prevention
            elif model_class == "Tomato___Septoria_leaf_spot":
                assert "4 g/kg" in ro_original.prevention
                assert "Maharashtra note" in ro_scrubbed.prevention
            elif model_class == "Tomato___Bacterial_spot":
                assert "streptocycline" in ro_original.what_it_is
                assert "disease-free seed" in ro_scrubbed.what_it_is
            elif model_class == "Pepper,_bell___Bacterial_spot":
                assert "kasugamycin 3% SL" in ro_original.what_it_is
                assert "streptocycline" in ro_original.what_it_is

    def test_what_to_do_first_reworked_expected_actions(self, agent):
        """
        Verify specific expectations for reworked what_to_do_first:
        1. Pepper bacterial spot: clean seed / healthy transplants.
        2. Apple scab: non-chemical imperative (resistance management / restrict applications).
        """
        ro_pepper = agent.query_candidate("Pepper,_bell___Bacterial_spot", crop="Pepper,_bell", include_doses=False)
        assert ro_pepper.what_to_do_first is not None
        assert "disease-free seed and healthy transplants" in ro_pepper.what_to_do_first.lower()

        ro_apple = agent.query_candidate("Apple___Apple_scab", crop="Apple", include_doses=False)
        assert ro_apple.what_to_do_first is not None
        assert "urea" not in ro_apple.what_to_do_first.lower()
        assert "time sprays" not in ro_apple.what_to_do_first.lower()
        assert "restrict the number of applications" in ro_apple.what_to_do_first.lower()

    def test_no_dangling_because_sentences_when_doses_scrubbed(self, agent):
        """
        Verify that when include_doses=False, no narrative field contains a sentence
        starting with 'Because' that lacks a main action after the comma.
        Ensures chemical sentence drops do not leave dangling subordinate clauses.
        """
        import re

        mapped_classes = [
            ("Apple___Apple_scab", "Apple"),
            ("Apple___Cedar_apple_rust", "Apple"),
            ("Grape___Black_rot", "Grape"),
            ("Grape___Esca_(Black_Measles)", "Grape"),
            ("Corn___Northern_Leaf_Blight", "Corn"),
            ("Corn___Common_rust", "Corn"),
            ("Corn___Cercospora_leaf_spot", "Corn"),
            ("Tomato___Early_blight", "Tomato"),
            ("Tomato___Late_blight", "Tomato"),
            ("Tomato___Septoria_leaf_spot", "Tomato"),
            ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato"),
            ("Tomato___Bacterial_spot", "Tomato"),
            ("Tomato___Leaf_Mold", "Tomato"),
            ("Tomato___Tomato_mosaic_virus", "Tomato"),
            ("Potato___Late_blight", "Potato"),
            ("Potato___Early_blight", "Potato"),
            ("Pepper,_bell___Bacterial_spot", "Pepper,_bell"),
        ]

        fields = [
            "what_it_is",
            "why_it_happens",
            "what_to_do_first",
            "prevention",
            "look_alikes",
            "favourable_conditions",
        ]

        # Valid action verb pattern for main clauses
        action_verbs = re.compile(
            r"\b(is|are|was|were|helps?|speeds?|reduces?|prevents?|remov\w+|clear\w+|bury\w+|protects?|causes?|provides?|should|must|can)\b",
            re.I,
        )

        for model_class, crop in mapped_classes:
            ro = agent.query_candidate(model_class, crop=crop, include_doses=False)
            all_text_blocks = [getattr(ro, f) for f in fields if getattr(ro, f)]
            all_text_blocks.extend(ro.treatment_options)
            all_text_blocks.extend(ro.preventive_measures)

            for block in all_text_blocks:
                # Check each sentence or bullet line
                for segment in re.split(r"(?<=[.!?\n])\s+", block):
                    clean_seg = segment.strip().lstrip("•-* \t")
                    if clean_seg.startswith("Because"):
                        # Must contain a comma separating subordinate clause
                        assert "," in clean_seg, f"Sentence starting with 'Because' lacks comma in {model_class}: {clean_seg}"
                        after_comma = clean_seg.split(",", 1)[1].strip()
                        # Must have a main action verb after the comma
                        assert action_verbs.search(after_comma), (
                            f"Dangling 'Because' fragment with no main action after comma in {model_class}: {clean_seg}"
                        )
                        # Must not end abruptly with hanging preposition or conjunction
                        assert not re.search(r"\b(by|with|of|in|on|at|and|or|for|to)\s*$", after_comma, re.I), (
                            f"Hanging preposition/conjunction in 'Because' clause in {model_class}: {clean_seg}"
                        )

