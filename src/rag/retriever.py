"""
src/rag/retriever.py — Knowledge Base Retrieval Component
Provides two retrieval pathways:
1. Exact metadata lookup by model_class for disease candidates (primary path; never uses semantic search).
2. Semantic search for optional free-text follow-up questions (strictly filtered to the given crop; never returns cross-crop chunks).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.rag.index import KBIndex
from src.rag.ingest import KBChunk

logger = logging.getLogger("leafmind.rag.retriever")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASS_MAP_PATH = REPO_ROOT / "config" / "kb_class_map.yaml"


@dataclass
class ChapterResult:
    """Exact lookup result for a disease chapter."""
    crop: Optional[str]
    model_class: Optional[str]
    disease_name: Optional[str]
    found: bool
    kb_gap: bool
    reason: Optional[str]
    glance_chunk: Optional[KBChunk] = None
    identification_chunk: Optional[KBChunk] = None
    causes_chunk: Optional[KBChunk] = None
    favourable_chunk: Optional[KBChunk] = None
    prevention_chunk: Optional[KBChunk] = None
    remedies_chunk: Optional[KBChunk] = None
    sources_chunk: Optional[KBChunk] = None
    all_chunks: List[KBChunk] = None

    def __post_init__(self):
        if self.all_chunks is None:
            self.all_chunks = []


class KBRetriever:
    """
    Retriever for knowledge base chunks.
    Ensures zero cross-crop leakage and deterministic exact-match lookup.
    """

    def __init__(
        self,
        index: Optional[KBIndex] = None,
        class_map_path: Optional[Path] = None,
    ):
        self.index = index or KBIndex()
        if not self.index.chunks_by_id:
            loaded = self.index.load()
            if not loaded:
                logger.warning("KBIndex could not be loaded from disk. Run scripts/build_kb_index.py.")

        self.class_map_path = Path(class_map_path or DEFAULT_CLASS_MAP_PATH)
        self.kb_class_map = self._load_class_map()
        self.healthy_classes = set(self.kb_class_map.get("healthy_classes", []))
        self.classes_without_kb = set(self.kb_class_map.get("model_classes_without_kb", []))

    def _load_class_map(self) -> Dict[str, Any]:
        if not self.class_map_path.exists():
            logger.error("Class map not found at %s", self.class_map_path)
            return {}
        with open(self.class_map_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def resolve_crop(self, crop: Optional[str]) -> Optional[str]:
        """Resolve a crop name/alias to canonical crop name."""
        if not crop:
            return None
        c = crop.strip()
        if not c:
            return None
        # Handle Corn aliases
        if c.lower() in ("corn", "corn_(maize)", "corn (maize)", "maize"):
            return "Corn"
        # Handle Pepper aliases
        if c.lower() in ("pepper", "pepper, bell", "pepper,_bell", "pepper_bell", "bell_pepper", "bell pepper"):
            return "Pepper,_bell"

        # Check crops defined in class map
        for canon, data in self.kb_class_map.get("crops", {}).items():
            if c.lower() == canon.lower():
                return canon
            for alias in data.get("aliases", []):
                if c.lower() == alias.lower():
                    return canon

        return c.capitalize()

    def retrieve_for_candidate(
        self,
        model_class: str,
        crop: Optional[str] = None,
    ) -> ChapterResult:
        """
        Primary path: exact metadata lookup by model_class.
        Never uses semantic search to decide which disease it is.
        """
        resolved_crop = self.resolve_crop(crop)

        # 1. Check healthy class
        if model_class in self.healthy_classes or model_class.endswith("___healthy"):
            inferred_crop = resolved_crop or model_class.split("___")[0]
            inferred_crop = self.resolve_crop(inferred_crop)
            return ChapterResult(
                crop=inferred_crop,
                model_class=model_class,
                disease_name="Healthy",
                found=False,
                kb_gap=False,
                reason="healthy_class",
            )

        # 2. Check known KB gap (e.g. Potato, Pepper, Squash until added)
        if model_class in self.classes_without_kb:
            inferred_crop = resolved_crop or model_class.split("___")[0]
            inferred_crop = self.resolve_crop(inferred_crop)
            return ChapterResult(
                crop=inferred_crop,
                model_class=model_class,
                disease_name=model_class.split("___")[-1].replace("_", " "),
                found=False,
                kb_gap=True,
                reason="no_kb_chapter",
            )

        # 3. Retrieve chunks by exact model_class
        chunks = self.index.get_chunks_by_model_class(model_class)
        if not chunks:
            inferred_crop = resolved_crop or model_class.split("___")[0]
            inferred_crop = self.resolve_crop(inferred_crop)
            return ChapterResult(
                crop=inferred_crop,
                model_class=model_class,
                disease_name=model_class.split("___")[-1].replace("_", " "),
                found=False,
                kb_gap=True,
                reason="no_kb_chapter",
            )

        # 4. Enforce crop consistency (guard against cross-crop leakage)
        chapter_crop = chunks[0].crop
        disease_name = chunks[0].disease_name

        if resolved_crop and self.resolve_crop(chapter_crop) != resolved_crop:
            logger.warning(
                "Crop mismatch in retrieval: requested %s, chapter is %s for %s",
                resolved_crop, chapter_crop, model_class
            )
            return ChapterResult(
                crop=resolved_crop,
                model_class=model_class,
                disease_name=disease_name,
                found=False,
                kb_gap=False,
                reason="crop_mismatch",
            )

        # 5. Partition chunks by section
        glance_chunk: Optional[KBChunk] = None
        id_chunk: Optional[KBChunk] = None
        causes_chunk: Optional[KBChunk] = None
        fav_chunk: Optional[KBChunk] = None
        prev_chunk: Optional[KBChunk] = None
        rem_chunk: Optional[KBChunk] = None
        src_chunk: Optional[KBChunk] = None

        for c in chunks:
            if c.chunk_type == "glance" or c.section == "At a glance":
                glance_chunk = c
            elif c.section == "Identification":
                id_chunk = c
            elif c.section == "Causes and progression":
                causes_chunk = c
            elif c.section == "Favourable conditions":
                fav_chunk = c
            elif c.section == "Prevention and management":
                prev_chunk = c
            elif c.section == "Remedies and treatment":
                rem_chunk = c
            elif c.section == "Sources and citations" or c.chunk_type == "sources":
                src_chunk = c

        return ChapterResult(
            crop=chapter_crop,
            model_class=model_class,
            disease_name=disease_name,
            found=True,
            kb_gap=False,
            reason=None,
            glance_chunk=glance_chunk,
            identification_chunk=id_chunk,
            causes_chunk=causes_chunk,
            favourable_chunk=fav_chunk,
            prevention_chunk=prev_chunk,
            remedies_chunk=rem_chunk,
            sources_chunk=src_chunk,
            all_chunks=chunks,
        )

    def semantic_search(
        self,
        query: str,
        crop: str,
        top_k: int = 3,
        include_non_farmer: bool = False,
    ) -> List[Tuple[KBChunk, float]]:
        """
        Secondary path: semantic search for optional free-text follow-up questions.
        STRICTLY filtered to the given crop. Never returns chunks from another crop.
        """
        if not crop:
            raise ValueError("crop is required for semantic search to prevent cross-crop leakage")

        canonical_crop = self.resolve_crop(crop)
        if not canonical_crop:
            raise ValueError(f"Unknown crop: {crop}")

        if self.index.faiss_index is None or not self.index.faiss_chunk_ids:
            logger.warning("FAISS index not loaded for semantic search")
            return []

        # Encode query
        query_vec = self.index.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
        import faiss
        faiss.normalize_L2(query_vec)

        # Retrieve more candidates to allow strict crop filtering
        num_candidates = min(len(self.index.faiss_chunk_ids), max(50, top_k * 10))
        scores, indices = self.index.faiss_index.search(query_vec, num_candidates)

        results: List[Tuple[KBChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.index.faiss_chunk_ids):
                continue
            chunk_id = self.index.faiss_chunk_ids[idx]
            chunk = self.index.get_chunk(chunk_id)
            if not chunk:
                continue

            # Strict crop filter: never return chunks from another crop!
            if self.resolve_crop(chunk.crop) != canonical_crop:
                continue

            if not include_non_farmer and not chunk.indexed_for_farmer:
                continue

            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results
