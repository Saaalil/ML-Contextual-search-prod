"""
Query Understanding Layer — parses natural language queries into structured
search intent using custom keyword transformation.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

COLORS = {"red", "blue", "green", "black", "white", "pink", "yellow", "orange", "grey", "gray", "brown", "navy", "purple"}
CATEGORIES = {"dress", "jacket", "pants", "skirt", "top", "shirt", "shorts", "jeans", "t-shirt", "coat", "sweater", "hoodie", "blouse", "suit"}
OCCASIONS = {"office", "casual", "party", "beach", "wedding", "formal", "workout", "gym", "lounge"}
MATERIALS = {"cotton", "leather", "silk", "denim", "wool", "linen", "polyester", "nylon"}
FITS = {"slim", "oversized", "regular", "loose", "tight", "skinny"}
SEASONS = {"summer", "winter", "spring", "fall", "autumn"}

class QueryParser:
    """Parses natural language queries into structured fashion search intent using rules."""

    def __init__(self, api_key: Optional[str] = None):
        # API key is no longer needed since we use local keyword parsing
        pass

    def parse(self, query: str) -> dict:
        """Parse a natural language query into structured intent."""
        query_lower = query.lower()
        filters = {}
        
        # Helper to find keywords, accounting for simple plurals (s, es)
        def find_match(vocab: set) -> Optional[str]:
            for word in sorted(list(vocab), key=len, reverse=True):
                pattern = rf'\b{re.escape(word)}(?:s|es)?\b'
                if re.search(pattern, query_lower):
                    return word
            return None

        # 1. Extract Price (e.g., "under $50", "below 100", "< 200")
        price_match = re.search(r'(?:under|below|<)\s*\$?(\d+)', query_lower)
        if price_match:
            filters["price_max"] = int(price_match.group(1))
            
        # 2. Extract Keywords
        color_match = find_match(COLORS)
        if color_match:
            filters["color"] = color_match
            
        cat_match = find_match(CATEGORIES)
        if cat_match:
            filters["category"] = cat_match
            
        occ_match = find_match(OCCASIONS)
        if occ_match:
            filters["occasion"] = occ_match
            
        mat_match = find_match(MATERIALS)
        if mat_match:
            filters["material"] = mat_match
            
        fit_match = find_match(FITS)
        if fit_match:
            filters["fit"] = fit_match
            
        season_match = find_match(SEASONS)
        if season_match:
            filters["season"] = season_match

        # 3. Exclude (simple negations)
        exclude = []
        if "not " in query_lower:
            not_match = re.search(r'not (\w+)', query_lower)
            if not_match:
                exclude.append(not_match.group(1))

        # Semantic query can be cleaned by removing the price and not filters
        semantic_query = query_lower
        semantic_query = re.sub(r'(?:under|below|<)\s*\$?(\d+)', '', semantic_query)
        semantic_query = re.sub(r'not (\w+)', '', semantic_query)
        semantic_query = semantic_query.strip()
        
        if not semantic_query:
            semantic_query = query

        return {
            "semantic_query": semantic_query,
            "filters": filters,
            "exclude": exclude,
            "intent": "find_items"
        }
