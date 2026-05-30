"""
Contextual Re-ranking Layer — uses Gemini Flash to score how well each
candidate matches the user's actual intent.

This goes beyond embedding similarity by understanding:
- Occasion appropriateness
- Style coherence
- Exclusion criteria
- Composite concepts
"""

import json
import os
import logging
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

from stylegraph.config import GEMINI_LLM_MODEL

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """You are a fashion search relevance judge. Given the user's search intent
and a list of candidate items, score each item's relevance from 0.0 to 1.0.

Consider:
- Does the item match the requested style/occasion/color?
- Does the item violate any exclusion criteria?
- How well does the item fit the overall intent?

Return ONLY a JSON array of objects: [{"index": 0, "score": 0.95, "reason": "short reason"}, ...]
Sort by score descending. Return ALL items.
"""


class ContextualReranker:
    """Re-ranks search candidates using Gemini Flash for contextual relevance."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        self.client = genai.Client(api_key=self.api_key)

    def rerank(
        self,
        parsed_query: dict,
        candidates: list[dict],
        top_k: int = 20,
    ) -> list[dict]:
        """Re-rank candidates based on contextual relevance to the parsed query."""
        if not candidates:
            return []

        # If only a few candidates, skip re-ranking
        if len(candidates) <= top_k:
            return candidates

        try:
            # Build the prompt
            query_desc = (
                f"User wants: {parsed_query.get('semantic_query', '')}\n"
                f"Filters: {json.dumps(parsed_query.get('filters', {}))}\n"
                f"Exclude: {parsed_query.get('exclude', [])}\n"
                f"Intent: {parsed_query.get('intent', 'find_items')}"
            )

            items_desc = []
            for i, c in enumerate(candidates):
                desc = (
                    f"Item {i}: title='{c.get('title', 'unknown')}', "
                    f"category='{c.get('category', 'unknown')}', "
                    f"price=${c.get('price', 0):.2f}, "
                    f"attrs={json.dumps(c.get('attrs', {}))}"
                )
                items_desc.append(desc)

            prompt = (
                f"{query_desc}\n\n"
                f"Candidates:\n" + "\n".join(items_desc)
            )

            resp = self.client.models.generate_content(
                model=GEMINI_LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_RERANK_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            scores = json.loads(resp.text)
            # Sort by score descending
            scores.sort(key=lambda x: x.get("score", 0), reverse=True)

            reranked = []
            for s in scores[:top_k]:
                idx = s.get("index", 0)
                if 0 <= idx < len(candidates):
                    item = dict(candidates[idx])
                    item["rerank_score"] = s.get("score", 0)
                    item["rerank_reason"] = s.get("reason", "")
                    reranked.append(item)
            return reranked

        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}. Returning original order.")
            return candidates[:top_k]
