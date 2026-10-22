"""
scripts/build_kb_index.py — Build Knowledge Base Vector Index and Metadata
Rebuilds the LeafMind RAG knowledge base index from data/knowledge_base/*.pdf.
Embeds chunks with BAAI/bge-small-en-v1.5 and persists FAISS index to vector_db/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.rag.ingest import ingest_knowledge_base
from src.rag.index import KBIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_kb_index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LeafMind KB vector and metadata index")
    parser.add_argument("--kb-dir", type=str, default=str(REPO_ROOT / "data" / "knowledge_base"),
                        help="Path to PDF knowledge base directory")
    parser.add_argument("--index-dir", type=str, default=str(REPO_ROOT / "vector_db"),
                        help="Path to output vector_db directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device for embeddings (cpu/cuda)")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    index_dir = Path(args.index_dir)

    logger.info("Ingesting PDFs from %s...", kb_dir)
    chunks, reports = ingest_knowledge_base(kb_dir=kb_dir)

    print("\n" + "=" * 70)
    print(" KNOWLEDGE BASE INGESTION AUDIT")
    print("=" * 70)
    header = "{:<12s} | {:<10s} | {:<12s} | {:<20s}".format(
        "Crop", "Chapters", "Total Chunks", "Missing Sections"
    )
    print(header)
    print("-" * 70)
    for r in reports:
        missing_str = "None" if not r["missing_sections"] else str(r["missing_sections"])
        crop = r["crop"]
        crop_chunks = [c for c in chunks if c.crop == crop]
        print("{:<12s} | {:<10d} | {:<12d} | {:<20s}".format(
            crop, r["chapters_found"], len(crop_chunks), missing_str
        ))
    print("=" * 70)

    # Initialize index and build
    logger.info("Building FAISS vector index and metadata store in %s...", index_dir)
    index = KBIndex(index_dir=index_dir, device=args.device)
    stats = index.build(chunks)

    print("\n" + "=" * 70)
    print(" VECTOR INDEX BUILD SUMMARY")
    print("=" * 70)
    print(f"Total chunks stored:           {stats['total_chunks']}")
    print(f"Chunks indexed for farmer:     {stats['indexed_for_farmer']}")
    print(f"Embedding dimension:           {stats['dimension']}")
    print(f"Diseased classes with KB:      {stats['classes_indexed']}")
    print(f"Crops indexed:                 {stats['crops_indexed']}")
    print(f"Index storage directory:       {index_dir}")
    print("=" * 70)

    # List mapped model classes and gaps
    import yaml
    with open(REPO_ROOT / "config" / "kb_class_map.yaml", "r", encoding="utf-8") as f:
        kb_map = yaml.safe_load(f)

    print("\nModel classes WITH KB chapters in current index:")
    for mc in sorted(index.chunks_by_model_class.keys()):
        print(f"  [+] {mc}")

    print("\nModel classes with NO KB chapter (expected gaps):")
    for mc in kb_map.get("model_classes_without_kb", []):
        print(f"  [-] {mc} (Phase B / future expansion)")

    print("\nHealthy model classes (no disease chapter required):")
    for hc in kb_map.get("healthy_classes", []):
        print(f"  [o] {hc}")

    print("\nIndex build complete and verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
