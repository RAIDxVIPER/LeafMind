"""
src/rag/index.py — Vector and Metadata Index for Knowledge Base
Manages persistence and loading of the FAISS vector index and metadata store
using local embeddings (BAAI/bge-small-en-v1.5 via sentence-transformers).
Runs fully offline once the embedding model is downloaded.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.ingest import KBChunk, ingest_knowledge_base

logger = logging.getLogger("leafmind.rag.index")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VECTOR_DB_DIR = REPO_ROOT / "vector_db"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class KBIndex:
    """
    Knowledge Base Index combining:
    1. Exact metadata lookup by model_class and (crop, disease_name).
    2. FAISS IndexFlatIP cosine similarity index for crop-filtered follow-up questions.
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ):
        self.index_dir = Path(index_dir or DEFAULT_VECTOR_DB_DIR)
        self.model_name = model_name
        self.device = device
        self._embedder: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.chunks_by_id: Dict[str, KBChunk] = {}
        self.chunks_by_model_class: Dict[str, List[str]] = {}
        self.chunks_by_crop: Dict[str, List[str]] = {}
        self.faiss_chunk_ids: List[str] = []

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-loaded sentence transformer with offline fallback."""
        if self._embedder is None:
            # Prefer offline mode if cached
            cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
            model_slug = "models--" + self.model_name.replace("/", "--")
            if (cache_dir / model_slug).exists():
                os.environ["HF_HUB_OFFLINE"] = "1"
            logger.info("Loading embedding model %s on %s (offline=%s)",
                        self.model_name, self.device, os.environ.get("HF_HUB_OFFLINE", "0"))
            self._embedder = SentenceTransformer(self.model_name, device=self.device)
        return self._embedder

    def build(self, chunks: List[KBChunk]) -> Dict[str, Any]:
        """
        Build FAISS vector index and metadata dictionaries from ingested chunks.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        self.chunks_by_model_class = {}
        self.chunks_by_crop = {}
        self.faiss_chunk_ids = []

        farmer_chunks: List[KBChunk] = []
        for c in chunks:
            # Group by crop
            self.chunks_by_crop.setdefault(c.crop, []).append(c.chunk_id)
            # Group by model_class
            if c.model_class:
                self.chunks_by_model_class.setdefault(c.model_class, []).append(c.chunk_id)
            # Filter for semantic search index
            if c.indexed_for_farmer:
                farmer_chunks.append(c)

        logger.info("Embedding %d chunks for farmer search index...", len(farmer_chunks))
        texts = [f"{c.crop} | {c.disease_name} | {c.section}\n{c.text}" for c in farmer_chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Normalize to unit length for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.faiss_index = index
        self.faiss_chunk_ids = [c.chunk_id for c in farmer_chunks]

        # Save artifacts
        self.save()

        return {
            "total_chunks": len(chunks),
            "indexed_for_farmer": len(farmer_chunks),
            "dimension": dim,
            "classes_indexed": len(self.chunks_by_model_class),
            "crops_indexed": len(self.chunks_by_crop),
        }

    def save(self) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = self.index_dir / "kb_faiss.index"
        metadata_path = self.index_dir / "kb_metadata.json"
        chunks_path = self.index_dir / "kb_chunks.json"

        # 1. Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(faiss_path))

        # 2. Save metadata & lookups
        metadata = {
            "model_name": self.model_name,
            "faiss_chunk_ids": self.faiss_chunk_ids,
            "chunks_by_model_class": self.chunks_by_model_class,
            "chunks_by_crop": self.chunks_by_crop,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 3. Save all chunks
        chunks_dict = {cid: c.to_dict() for cid, c in self.chunks_by_id.items()}
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_dict, f, indent=2)

        logger.info("Saved index and metadata to %s", self.index_dir)

    def load(self) -> bool:
        """Load persisted index and metadata from disk."""
        faiss_path = self.index_dir / "kb_faiss.index"
        metadata_path = self.index_dir / "kb_metadata.json"
        chunks_path = self.index_dir / "kb_chunks.json"

        if not (faiss_path.exists() and metadata_path.exists() and chunks_path.exists()):
            logger.warning("Index files missing in %s", self.index_dir)
            return False

        self.faiss_index = faiss.read_index(str(faiss_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.model_name = metadata.get("model_name", self.model_name)
        self.faiss_chunk_ids = metadata.get("faiss_chunk_ids", [])
        self.chunks_by_model_class = metadata.get("chunks_by_model_class", {})
        self.chunks_by_crop = metadata.get("chunks_by_crop", {})

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_dict = json.load(f)
        self.chunks_by_id = {cid: KBChunk.from_dict(d) for cid, d in chunks_dict.items()}

        logger.info("Successfully loaded index from %s (%d chunks, %d indexed)",
                    self.index_dir, len(self.chunks_by_id), len(self.faiss_chunk_ids))
        return True

    def get_chunk(self, chunk_id: str) -> Optional[KBChunk]:
        return self.chunks_by_id.get(chunk_id)

    def get_chunks_by_model_class(self, model_class: str) -> List[KBChunk]:
        chunk_ids = self.chunks_by_model_class.get(model_class, [])
        return [self.chunks_by_id[cid] for cid in chunk_ids if cid in self.chunks_by_id]

    def get_chunks_by_crop(self, crop: str) -> List[KBChunk]:
        chunk_ids = self.chunks_by_crop.get(crop, [])
        return [self.chunks_by_id[cid] for cid in chunk_ids if cid in self.chunks_by_id]
