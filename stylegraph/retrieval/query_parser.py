"""
Query Understanding Layer — parses natural language queries into structured
search intent using Gemini Flash.

Example:
    Input:  "red summer dress under $50"
    Output: {
        "semantic_query": "red summer dress",
        "filters": {"color": "red", "season": "summer", "category": "dress", "price_max": 50},
        "exclude": [],
        "intent": "find_items"
    }
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

_SYSTEM_PROMPT = """You are a fashion search query parser. Given a natural language query,
extract structured information. Return ONLY valid JSON with these fields:

{
  "semantic_query": "the core search text for embedding similarity (clean, no filters)",
  "filters": {
    "color": "string or null",
    "category": "string or null (e.g. dress, jacket, pants, skirt, top)",
    "occasion": "string or null (e.g. office, casual, party, beach, wedding)",
    "style": "string or null (e.g. bohemian, minimal, streetwear, vintage)",
    "season": "string or null (e.g. summer, winter, spring, fall)",
    "material": "string or null",
    "fit": "string or null (e.g. slim, oversized, regular)",
    "price_max": "number or null",
    "price_min": "number or null"
  },
  "exclude": ["list of things to exclude, e.g. 'sporty', 'formal'"],
  "intent": "find_items | find_similar | browse_category"
}

Rules:
- Keep semantic_query short and focused (3-8 words)
- Extract price constraints like 'under $50' into price_max
- Extract negations like 'not sporty' or 'but not formal' into exclude list
- Use lowercase for all filter values
- Return null (not "null") for unknown fields
"""


class QueryParser:
    """Parses natural language queries into structured fashion search intent."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        self.client = genai.Client(api_key=self.api_key)

    def parse(self, query: str) -> dict:
        """Parse a natural language query into structured intent."""
        try:
            resp = self.client.models.generate_content(
                model=GEMINI_LLM_MODEL,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads(resp.text)
            # Ensure all expected keys exist
            parsed.setdefault("semantic_query", query)
            parsed.setdefault("filters", {})
            parsed.setdefault("exclude", [])
            parsed.setdefault("intent", "find_items")
            # Clean null string values in filters
            filters = parsed["filters"]
            for k, v in list(filters.items()):
                if v is None or v == "null" or v == "":
                    del filters[k]
            return parsed
        except Exception as e:
            logger.warning(f"Query parsing failed: {e}. Falling back to raw query.")
            return {
                "semantic_query": query,
                "filters": {},
                "exclude": [],
                "intent": "find_items",
            }
