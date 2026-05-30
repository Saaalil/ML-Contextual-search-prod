"""
Contextual Search Engine — 3-layer pipeline:
  1. Query Understanding (Gemini Flash) → structured intent
  2. Vector Retrieval (Gemini Embedding + FAISS) → top candidates
  3. Contextual Re-ranking (Gemini Flash) → final results

Supports both text-to-image and image-to-image search.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from PIL import Image

from stylegraph.config import INDEX_DIR, TOP_K, RERANK_CANDIDATES
from stylegraph.utils.io import load_image, read_jsonl

logger = logging.getLogger(__name__)


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
        
    from stylegraph.config import DEMO_IMAGES_DIR, CATALOG_DIR
    filename = path.name
    
    if (DEMO_IMAGES_DIR / filename).exists():
        return DEMO_IMAGES_DIR / filename
    if (CATALOG_DIR / filename).exists():
        return CATALOG_DIR / filename
        
    return path


class SearchEngine:
    """Contextual fashion search engine powered by Gemini."""

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[dict],
        embedder,
        query_parser=None,
        reranker=None,
        provider: str = "gemini",
    ):
        self.index = index
        self.metadata = metadata
        self.embedder = embedder
        self.query_parser = query_parser
        self.reranker = reranker
        self.provider = provider

    @classmethod
    def from_dir(
        cls,
        index_dir: Optional[Path] = None,
        device: str = "cpu",
        enable_reranking: bool = True,
    ):
        """Load a search engine from an index directory."""
        index_dir = Path(index_dir) if index_dir else INDEX_DIR
        index_path = index_dir / "faiss.index"
        meta_path = index_dir / "metadata.jsonl"
        model_path = index_dir / "model.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        index = faiss.read_index(str(index_path))
        metadata = list(read_jsonl(meta_path))

        # Detect provider from model.json
        provider = "gemini"  # default to gemini
        if model_path.exists():
            with open(model_path, "r", encoding="utf-8") as f:
                model_meta = json.load(f)
            provider = model_meta.get("provider", "openclip")

        if provider == "gemini":
            from stylegraph.model.gemini_embed import GeminiEmbedder
            embedder = GeminiEmbedder()

            query_parser = None
            reranker = None
            if enable_reranking:
                try:
                    from stylegraph.retrieval.query_parser import QueryParser
                    from stylegraph.retrieval.reranker import ContextualReranker
                    query_parser = QueryParser()
                    reranker = ContextualReranker()
                except Exception as e:
                    logger.warning(f"Could not init query parser/reranker: {e}")

            return cls(
                index=index,
                metadata=metadata,
                embedder=embedder,
                query_parser=query_parser,
                reranker=reranker,
                provider="gemini",
            )
        else:
            # Backward compat: load OpenCLIP model
            import torch
            import open_clip

            model_meta_data = {}
            if model_path.exists():
                with open(model_path, "r", encoding="utf-8") as f:
                    model_meta_data = json.load(f)

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_meta_data.get("model_name", "ViT-B-32"),
                pretrained=model_meta_data.get("pretrained", "laion2b_s34b_b79k"),
            )
            device_t = torch.device(device)
            model.to(device_t)
            model.eval()
            tokenizer = open_clip.get_tokenizer(
                model_meta_data.get("model_name", "ViT-B-32")
            )

            # Create a simple wrapper
            class OpenCLIPEmbedder:
                def __init__(self, model, preprocess, tokenizer, device):
                    self._model = model
                    self._preprocess = preprocess
                    self._tokenizer = tokenizer
                    self._device = device

                def embed_text(self, text, **kwargs):
                    import torch
                    with torch.no_grad():
                        tokens = self._tokenizer([text]).to(self._device)
                        features = self._model.encode_text(tokens)
                        features = features / features.norm(dim=-1, keepdim=True)
                    return features.cpu().numpy().squeeze(0)

                def embed_image_pil(self, image, **kwargs):
                    import torch
                    with torch.no_grad():
                        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
                        features = self._model.encode_image(tensor)
                        features = features / features.norm(dim=-1, keepdim=True)
                    return features.cpu().numpy().squeeze(0)

            embedder = OpenCLIPEmbedder(model, preprocess, tokenizer, device_t)
            return cls(
                index=index,
                metadata=metadata,
                embedder=embedder,
                provider="openclip",
            )

    # ── Core Search Methods ──────────────────────────────────────────

    def _vector_search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        """Raw FAISS nearest-neighbor search."""
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def _apply_filters(self, results: list[dict], filters: dict, exclude: list) -> list[dict]:
        """Apply structured filters to search results."""
        filtered = []
        for item in results:
            skip = False
            attrs = item.get("attrs", {})

            # Price filters
            if "price_max" in filters:
                if float(item.get("price", 1e9)) > filters["price_max"]:
                    skip = True
            if "price_min" in filters:
                if float(item.get("price", 0)) < filters["price_min"]:
                    skip = True

            # Category filter
            if "category" in filters and filters["category"]:
                item_cat = (item.get("category", "") or "").lower()
                if filters["category"].lower() not in item_cat:
                    skip = True

            # Attribute filters (color, style, occasion, etc.)
            for key in ["color", "occasion", "style", "season", "material", "fit"]:
                if key in filters and filters[key]:
                    item_val = (attrs.get(key, "") or "").lower()
                    if item_val and filters[key].lower() not in item_val:
                        # Soft filter: don't skip if attr is unknown
                        if item_val != "unknown":
                            skip = True

            # Exclusion list
            for exc in exclude:
                exc_lower = exc.lower()
                item_text = json.dumps(item).lower()
                if exc_lower in item_text:
                    skip = True

            if not skip:
                filtered.append(item)

        return filtered

    def search_text(
        self,
        text: str,
        top_k: int = TOP_K,
        enable_context: bool = True,
    ) -> list[dict]:
        """
        Contextual text search:
          1. Parse query → structured intent
          2. Embed semantic query → FAISS search → candidates
          3. Apply filters → re-rank → top-K results
        """
        parsed = None
        if enable_context and self.query_parser:
            try:
                parsed = self.query_parser.parse(text)
                logger.info(f"Parsed query: {parsed}")
            except Exception as e:
                logger.warning(f"Query parsing failed: {e}")

        # Use semantic query for embedding, or raw text
        embed_text = parsed["semantic_query"] if parsed else text
        vec = self.embedder.embed_text(embed_text)

        # Retrieve more candidates for filtering/re-ranking
        n_candidates = RERANK_CANDIDATES if (parsed and self.reranker) else top_k
        results = self._vector_search(vec, n_candidates)

        # Apply structured filters
        if parsed and parsed.get("filters"):
            filtered_results = self._apply_filters(
                results, parsed["filters"], parsed.get("exclude", [])
            )
            # If the dataset is messy and filters removed everything, fall back to unfiltered results
            if len(filtered_results) > 0:
                results = filtered_results

        # Contextual re-ranking
        if parsed and self.reranker and len(results) > top_k:
            try:
                results = self.reranker.rerank(parsed, results, top_k=top_k)
            except Exception as e:
                logger.warning(f"Re-ranking failed: {e}")
                results = results[:top_k]
        else:
            results = results[:top_k]

        return results

    def search_image(
        self, image: Image.Image, top_k: int = TOP_K
    ) -> list[dict]:
        """Image-to-image search: find visually similar items."""
        vec = self.embedder.embed_image_pil(image)
        return self._vector_search(vec, top_k)

    def load_image(self, path_str: str) -> Image.Image:
        return load_image(resolve_image_path(path_str))
