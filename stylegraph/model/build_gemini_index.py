"""
Build a FAISS index using Gemini Embedding 2.

This replaces the old OpenCLIP-based builder. Instead of running a local
model, it calls the Gemini API to embed each image, then stores the
resulting vectors in a FAISS flat-IP index.

Usage:
    python -m stylegraph.model.build_gemini_index
    python -m stylegraph.model.build_gemini_index --catalog data/catalog/demo_products.jsonl --output data/index
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from stylegraph.config import (
    ROOT_DIR,
    CATALOG_DIR,
    INDEX_DIR,
    GEMINI_EMBED_DIM,
    GEMINI_EMBED_MODEL,
)
from stylegraph.model.gemini_embed import GeminiEmbedder
from stylegraph.utils.io import ensure_dir, read_jsonl


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path
    candidate = ROOT_DIR / path
    if candidate.exists():
        return candidate
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index with Gemini embeddings.")
    parser.add_argument(
        "--catalog",
        type=str,
        default=str(CATALOG_DIR / "demo_products.jsonl"),
        help="Path to product catalog JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(INDEX_DIR),
        help="Output directory for the FAISS index.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partially-built index.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=GEMINI_EMBED_DIM,
        help="Embedding dimensions.",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    if not catalog_path.exists():
        print(f"❌  Catalog not found: {catalog_path}")
        print("   Run: python -m scripts.prepare_demo_subset")
        return

    # Load catalog
    catalog_rows = list(read_jsonl(catalog_path))
    print(f"📦  Loaded {len(catalog_rows)} items from catalog.")

    # Check for resume
    meta_path = output_dir / "metadata.jsonl"
    existing_ids = set()
    if args.resume and meta_path.exists():
        for row in read_jsonl(meta_path):
            existing_ids.add(row.get("id"))
        print(f"⏩  Resuming: {len(existing_ids)} items already indexed.")

    # Filter out already-indexed items
    pending = [r for r in catalog_rows if r.get("id") not in existing_ids]
    if not pending:
        print("✅  All items already indexed. Nothing to do.")
        return

    print(f"🔄  Indexing {len(pending)} images with Gemini Embedding...")

    # Initialize embedder
    embedder = GeminiEmbedder()

    # Embed all images
    embeddings = []
    indexed_meta = []
    failed = 0

    for row in tqdm(pending, desc="Embedding"):
        image_path = resolve_image_path(row["image_path"])
        if not image_path.exists():
            print(f"  ⚠️  Image not found: {image_path}")
            failed += 1
            continue

        try:
            vec = embedder.embed_image(image_path)
            embeddings.append(vec)
            indexed_meta.append(row)
        except Exception as e:
            print(f"  ⚠️  Failed to embed {image_path.name}: {e}")
            failed += 1
            continue

    if not embeddings:
        print("❌  No images were embedded. Check your GEMINI_API_KEY.")
        return

    all_embeddings = np.vstack(embeddings).astype(np.float32)
    faiss.normalize_L2(all_embeddings)

    # Build or append to FAISS index
    index_path = output_dir / "faiss.index"
    if args.resume and index_path.exists():
        index = faiss.read_index(str(index_path))
        index.add(all_embeddings)
        # Append metadata
        with open(meta_path, "a", encoding="utf-8") as f:
            for row in indexed_meta:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
    else:
        dim = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(all_embeddings)
        # Write metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            for row in indexed_meta:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    # Save FAISS index
    faiss.write_index(index, str(index_path))

    # Save model.json (marks this as a Gemini-built index)
    model_meta = {
        "provider": "gemini",
        "model": GEMINI_EMBED_MODEL,
        "dimensions": args.dim,
    }
    with open(output_dir / "model.json", "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2, ensure_ascii=True)

    print(f"\n✅  FAISS index built ({index.ntotal} vectors, {args.dim}d)")
    print(f"   Index: {index_path}")
    print(f"   Metadata: {meta_path}")
    if failed:
        print(f"   ⚠️  {failed} images failed to embed.")


if __name__ == "__main__":
    main()
