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
        
        # 1. Extract Price (e.g., "under $50", "below 100", "< 200")
        price_match = re.search(r'(?:under|below|<)\s*\$?(\d+)', query_lower)
        if price_match:
            filters["price_max"] = int(price_match.group(1))
            
        # 2. Extract Keywords
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        color_match = words.intersection(COLORS)
        if color_match:
            filters["color"] = list(color_match)[0]
            
        cat_match = words.intersection(CATEGORIES)
        if cat_match:
            filters["category"] = list(cat_match)[0]
            
        occ_match = words.intersection(OCCASIONS)
        if occ_match:
            filters["occasion"] = list(occ_match)[0]
            
        mat_match = words.intersection(MATERIALS)
        if mat_match:
            filters["material"] = list(mat_match)[0]
            
        fit_match = words.intersection(FITS)
        if fit_match:
            filters["fit"] = list(fit_match)[0]
            
        season_match = words.intersection(SEASONS)
        if season_match:
            filters["season"] = list(season_match)[0]

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
