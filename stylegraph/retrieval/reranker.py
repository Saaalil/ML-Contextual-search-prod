"""
Contextual Re-ranking Layer — uses raw FAISS similarity scores
to rank the final candidates, bypassing Gemini.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ContextualReranker:
    """Re-ranks search candidates using FAISS similarity scores."""

    def __init__(self, api_key: Optional[str] = None):
        # No API key needed for local sorting
        pass

    def rerank(
        self,
        parsed_query: dict,
        candidates: list[dict],
        top_k: int = 20,
    ) -> list[dict]:
        """Re-rank candidates based on their similarity score."""
        if not candidates:
            return []

        # Sort by score descending
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

        reranked = []
        for c in candidates[:top_k]:
            item = dict(c)
            # Maintain compatibility with the frontend that looks for rerank_reason
            item["rerank_score"] = item.get("score", 0)
            item["rerank_reason"] = "Matched based on vector similarity and local keyword filters."
            reranked.append(item)
            
        return reranked
