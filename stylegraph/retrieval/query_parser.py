"""
Query Understanding Layer — parses natural language queries into structured
search intent using Query Expansion (Vibe Enhancements) and strict constraint extraction.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Vibe dictionary for Query Expansion.
# When a user types a short keyword, we inject rich, semantic descriptors
# into the embedding query to give FAISS more context.
VIBE_DICTIONARY = {
    "office": "professional, formal, structured, elegant, tailored, workwear",
    "beach": "summer, casual, vibrant, lightweight, breezy, tropical, relaxed",
    "party": "bold, stylish, evening, eye-catching, festive, chic",
    "workout": "athletic, breathable, sporty, activewear, comfortable, flexible",
    "gym": "athletic, breathable, sporty, activewear, comfortable, flexible",
    "casual": "relaxed, comfortable, everyday, effortless, simple, laid-back",
    "vintage": "retro, classic, nostalgic, old-school, timeless, authentic",
    "streetwear": "urban, trendy, oversized, edgy, cool, hype, casual",
    "minimalist": "clean, simple, monochrome, sleek, modern, understated",
    "boho": "bohemian, flowy, earthy, pattern, relaxed, free-spirited",
    "wedding": "formal, elegant, sophisticated, gown, suit, celebratory, refined",
    "winter": "warm, cozy, insulated, heavy, layered, cold-weather",
    "summer": "light, breathable, cool, sunny, warm-weather, bright",
}

class QueryParser:
    """Parses and mathematically enhances queries for maximum semantic vector relevance."""

    def __init__(self, api_key: Optional[str] = None):
        pass

    def parse(self, query: str) -> dict:
        """Parse a natural language query into enhanced semantic intent."""
        query_lower = query.lower()
        filters = {}
        exclude = []
        
        # 1. Extract STRICT Mathematical Constraints (Price)
        # Matches: "under $50", "below 100", "< 200", "under 50"
        price_match = re.search(r'(?:under|below|<)\s*\$?\s*(\d+)', query_lower)
        if price_match:
            filters["price_max"] = int(price_match.group(1))
            
        # 2. Extract STRICT Exclusions
        # Matches: "not red", "no jeans"
        not_match = re.search(r'(?:not|no)\s+([a-zA-Z0-9]+)', query_lower)
        if not_match:
            exclude.append(not_match.group(1))

        # 3. Clean the semantic query
        # Remove the strict constraints so they don't confuse the embedding model
        semantic_query = query_lower
        semantic_query = re.sub(r'(?:under|below|<)\s*\$?\s*(\d+)', '', semantic_query)
        semantic_query = re.sub(r'(?:not|no)\s+([a-zA-Z0-9]+)', '', semantic_query)
        semantic_query = semantic_query.strip()
        
        if not semantic_query:
            semantic_query = query

        # 4. Contextual Query Expansion (The Secret Sauce)
        # If the user's query contains a known vibe, append its rich descriptors!
        enhancements = []
        for vibe, descriptors in VIBE_DICTIONARY.items():
            pattern = rf'\b{vibe}\b'
            if re.search(pattern, semantic_query):
                enhancements.append(descriptors)
                
        if enhancements:
            # Append the enhancements in parentheses so the model groups them semantically
            semantic_query += f" (Context: {', '.join(enhancements)})"
            logger.info(f"Expanded Query: {semantic_query}")

        return {
            "semantic_query": semantic_query,
            "filters": filters,  # ONLY strict constraints like price now!
            "exclude": exclude,
            "intent": "find_items"
        }
