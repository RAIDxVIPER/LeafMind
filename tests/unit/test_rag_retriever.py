"""
tests/unit/test_rag_retriever.py
================================
Unit tests for KB exact candidate lookup and crop-filtered semantic search (Stage C3).
Verifies:
  1. Every mapped model class retrieves its own chapter and the correct crop.
  2. No cross-crop leakage (e.g. Tomato Early Blight with crop='Potato' rejected).
  3. Missing chapter -> kb_gap=True, found=False.
  4. Healthy model class -> found=False, reason="healthy_class".
  5. Semantic search strictly filtered to crop; never returns chunks from another crop.
  6. Semantic search requires crop (raises ValueError).
"""
from __future__ import annotations

import pytest

from src.rag.retriever import KBRetriever
from src.rag.index import KBIndex


@pytest.fixture(scope="module")
def retriever():
    index = KBIndex()
    index.load()
    return KBRetriever(index=index)


class TestRAGRetriever:

    def test_every_mapped_model_class_retrieval(self, retriever):
        """Every mapped model class retrieves its own chapter and the correct crop."""
        expected_mappings = [
            ("Apple___Apple_scab", "Apple", "Apple Scab"),
            ("Apple___Cedar_apple_rust", "Apple", "Cedar Apple Rust"),
            ("Grape___Black_rot", "Grape", "Black Rot"),
            ("Grape___Esca_(Black_Measles)", "Grape", "Esca (Black Measles)"),
            ("Corn___Northern_Leaf_Blight", "Corn", "Turcicum Leaf Blight (Northern Corn Leaf Blight)"),
            ("Corn___Common_rust", "Corn", "Common Rust"),
            ("Corn___Cercospora_leaf_spot", "Corn", "Gray Leaf Spot"),
            ("Tomato___Early_blight", "Tomato", "Early Blight"),
            ("Tomato___Late_blight", "Tomato", "Late Blight"),
            ("Tomato___Septoria_leaf_spot", "Tomato", "Septoria Leaf Spot"),
            ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato", "Tomato Leaf Curl Disease"),
            ("Tomato___Bacterial_spot", "Tomato", "Bacterial Leaf Spot (Bacterial Spot)"),
            ("Tomato___Leaf_Mold", "Tomato", "Leaf Mould"),
            ("Tomato___Tomato_mosaic_virus", "Tomato", "Tomato Mosaic Virus (and related mosaic viruses)"),
            ("Potato___Late_blight", "Potato", "Late Blight"),
            ("Potato___Early_blight", "Potato", "Early Blight (Alternaria Leaf Spot)"),
            ("Pepper,_bell___Bacterial_spot", "Pepper,_bell", "Bacterial Spot (Bacterial Leaf Spot)"),
        ]

        for model_class, expected_crop, expected_disease in expected_mappings:
            res = retriever.retrieve_for_candidate(model_class, crop=expected_crop)
            assert res.found is True, f"Failed to retrieve mapped class {model_class}"
            assert res.kb_gap is False
            assert res.crop == expected_crop
            assert res.disease_name == expected_disease
            assert res.glance_chunk is not None
            assert res.identification_chunk is not None
            assert res.causes_chunk is not None
            assert res.favourable_chunk is not None
            assert res.prevention_chunk is not None
            assert res.remedies_chunk is not None
            assert res.sources_chunk is not None

    def test_no_cross_crop_leakage(self, retriever):
        """Cross-crop queries must be rejected to prevent leakage (bidirectional Tomato <-> Potato)."""
        # Querying Tomato Early Blight under Potato crop filter must fail
        res = retriever.retrieve_for_candidate("Tomato___Early_blight", crop="Potato")
        assert res.found is False
        assert res.reason == "crop_mismatch"

        # Querying Potato Early Blight under Tomato crop filter must fail
        res_p_eb = retriever.retrieve_for_candidate("Potato___Early_blight", crop="Tomato")
        assert res_p_eb.found is False
        assert res_p_eb.reason == "crop_mismatch"

        # Querying Tomato Late Blight under Potato crop filter must fail
        res_t_lb = retriever.retrieve_for_candidate("Tomato___Late_blight", crop="Potato")
        assert res_t_lb.found is False
        assert res_t_lb.reason == "crop_mismatch"

        # Querying Potato Late Blight under Tomato crop filter must fail
        res_p_lb = retriever.retrieve_for_candidate("Potato___Late_blight", crop="Tomato")
        assert res_p_lb.found is False
        assert res_p_lb.reason == "crop_mismatch"

        # Querying Apple Scab under Grape crop filter must fail
        res_apple = retriever.retrieve_for_candidate("Apple___Apple_scab", crop="Grape")
        assert res_apple.found is False
        assert res_apple.reason == "crop_mismatch"

    def test_missing_chapter_kb_gap(self, retriever):
        """Diseased classes without KB chapters return kb_gap=True and found=False."""
        # Only Squash___Powdery_mildew remains an expected gap
        missing_classes = [
            "Squash___Powdery_mildew",
        ]
        for mc in missing_classes:
            res = retriever.retrieve_for_candidate(mc)
            assert res.found is False, f"{mc} should not be found"
            assert res.kb_gap is True, f"{mc} should set kb_gap=True"
            assert res.reason == "no_kb_chapter"

        # Confirm newly added classes no longer return kb_gap=True
        newly_added = [
            "Pepper,_bell___Bacterial_spot",
            "Potato___Early_blight",
            "Potato___Late_blight",
        ]
        for mc in newly_added:
            res = retriever.retrieve_for_candidate(mc)
            assert res.found is True, f"{mc} should now be found"
            assert res.kb_gap is False, f"{mc} must not return kb_gap=True"

    def test_healthy_class_handling(self, retriever):
        """Healthy classes return found=False, kb_gap=False, reason='healthy_class'."""
        healthy_classes = [
            "Tomato___healthy",
            "Apple___healthy",
            "Blueberry___healthy",
            "Cherry___healthy",
            "Corn_(maize)___healthy",
            "Grape___healthy",
            "Peach___healthy",
            "Pepper,_bell___healthy",
            "Potato___healthy",
            "Raspberry___healthy",
            "Soybean___healthy",
            "Strawberry___healthy",
        ]
        for hc in healthy_classes:
            res = retriever.retrieve_for_candidate(hc)
            assert res.found is False, f"{hc} should return found=False"
            assert res.kb_gap is False, f"{hc} is healthy, not a knowledge gap"
            assert res.reason == "healthy_class"

    def test_crop_aliases_in_candidate_retrieval(self, retriever):
        """Verify crop alias resolution: Corn / maize / Corn_(maize)."""
        res1 = retriever.retrieve_for_candidate("Corn___Common_rust", crop="Corn")
        res2 = retriever.retrieve_for_candidate("Corn___Common_rust", crop="maize")
        res3 = retriever.retrieve_for_candidate("Corn___Common_rust", crop="Corn_(maize)")
        assert res1.found and res2.found and res3.found
        assert res1.disease_name == res2.disease_name == res3.disease_name

    def test_semantic_search_requires_crop(self, retriever):
        """Semantic search must raise ValueError if crop is empty."""
        with pytest.raises(ValueError):
            retriever.semantic_search("yellowing leaves", crop="")

    def test_semantic_search_never_leaks_cross_crop(self, retriever):
        """Semantic search must strictly return chunks matching the specified crop only."""
        crops_to_test = ["Tomato", "Apple", "Corn", "Grape", "Cherry", "Blueberry", "Potato", "Pepper,_bell"]
        query = "spots on leaves and fruit rot with wilting"

        for crop in crops_to_test:
            results = retriever.semantic_search(query, crop=crop, top_k=5)
            assert len(results) > 0, f"Expected semantic results for {crop}"
            for chunk, score in results:
                assert retriever.resolve_crop(chunk.crop) == retriever.resolve_crop(crop), (
                    f"Leakage detected: expected {crop}, got {chunk.crop} (chunk {chunk.chunk_id})"
                )
