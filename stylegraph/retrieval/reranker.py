"""
Contextual Re-ranking Layer (Multimodal) — uses Gemini 2.0 Flash to visually 
score how well each candidate matches the user's actual intent.

This goes beyond embedding similarity by actually *looking* at the images to judge:
- Occasion appropriateness
- Style coherence
- Exclusion criteria
- Detailed visual properties that FAISS might miss
"""

import json
import os
import logging
from typing import Optional
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

from stylegraph.config import GEMINI_LLM_MODEL, ROOT_DIR
from stylegraph.utils.io import load_image

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """You are an expert fashion AI judge. I am giving you the user's natural language search intent, followed by a list of images (the top candidates retrieved by our vector database).

Your job is to look at each image carefully and score its relevance to the user's query from 0.0 to 1.0.

Consider:
- Does the item in the image match the requested style/occasion/color?
- Does the image violate any exclusion criteria?
- How well does the item visually fit the overall intent?

You MUST return ONLY a JSON array of objects in this exact format: 
[{"index": 0, "score": 0.95, "reason": "short explanation of why this visually matches"}, ...]

Sort the output array by score descending. Return an object for ALL images provided.
"""

def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path
    candidate = ROOT_DIR / path
    if candidate.exists():
        return candidate
    return path


class ContextualReranker:
    """Re-ranks search candidates using Gemini 2.0 Flash Multimodal Vision."""

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
        """Re-rank candidates based on visual multimodal relevance to the parsed query."""
        if not candidates:
            return []

        # If only a few candidates, skip re-ranking
        if len(candidates) <= top_k and len(candidates) < 5:
            return candidates

        # Limit how many images we send to Gemini to avoid blowing up tokens
        # 40 images max is a good balance between recall and API limits
        max_candidates = min(len(candidates), 40)
        candidates = candidates[:max_candidates]

        try:
            # Build the text prompt
            query_desc = (
                f"User wants: {parsed_query.get('semantic_query', '')}\n"
                f"Filters: {json.dumps(parsed_query.get('filters', {}))}\n"
                f"Exclude: {parsed_query.get('exclude', [])}\n"
                f"Intent: {parsed_query.get('intent', 'find_items')}"
            )

            contents = [_RERANK_PROMPT + "\n\n" + query_desc + "\n\nCandidates:\n"]
            
            # Append each image to the multimodal payload
            valid_indices = []
            for i, c in enumerate(candidates):
                try:
                    img_path = resolve_image_path(c.get("image_path", ""))
                    img = load_image(img_path)
                    # Thumbnail to save bandwidth and tokens
                    img.thumbnail((384, 384))
                    
                    contents.append(f"Image Index {i}:")
                    contents.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Skipping image {i} for reranking due to load error: {e}")

            if not valid_indices:
                return candidates[:top_k]

            # Call Gemini 2.0 Flash
            resp = self.client.models.generate_content(
                model=GEMINI_LLM_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            scores = json.loads(resp.text)
            
            # Sort by score descending
            scores.sort(key=lambda x: x.get("score", 0), reverse=True)

            reranked = []
            # We want to return exactly top_k items
            for s in scores:
                idx = s.get("index", -1)
                if 0 <= idx < len(candidates):
                    item = dict(candidates[idx])
                    item["rerank_score"] = s.get("score", 0)
                    item["rerank_reason"] = s.get("reason", "")
                    reranked.append(item)
                    if len(reranked) >= top_k:
                        break
            
            # Fill the rest if Gemini returned fewer than requested
            if len(reranked) < top_k:
                seen_ids = {r.get("id") for r in reranked}
                for c in candidates:
                    if c.get("id") not in seen_ids:
                        c_copy = dict(c)
                        c_copy["rerank_score"] = 0
                        c_copy["rerank_reason"] = "Fallback: Gemini missed this item."
                        reranked.append(c_copy)
                        if len(reranked) >= top_k:
                            break

            return reranked

        except Exception as e:
            logger.warning(f"Multimodal Re-ranking failed: {e}. Returning original FAISS order.")
            # Ensure compatibility with frontend expected keys
            for c in candidates:
                c["rerank_score"] = c.get("score", 0)
                c["rerank_reason"] = "Matched based on vector similarity (Reranking failed)"
            return candidates[:top_k]
