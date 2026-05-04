"""
router.py
---------
Determines whether a user query is about a person, a place, or both.

Strategy: Rule-based keyword matching against the known entity names.
  - If query mentions known person names → "person"
  - If query mentions known place names  → "place"
  - If both → "both"
  - Fallback (nothing matched) → "both" (search everything)

This approach is fast, deterministic, and requires no extra model call.
"""

import json
import re
import sys
from pathlib import Path
from typing import Literal

QueryType = Literal["person", "place", "both"]

# Path to entities.json relative to this file
ENTITIES_JSON = Path(__file__).resolve().parent.parent / "data" / "entities.json"


def _load_entities() -> tuple[list[str], list[str]]:
    """
    Loads entity names from entities.json, splitting into persons and places.

    Returns:
        (person_names, place_names) — both lowercase for matching.
    """
    with open(ENTITIES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    person_names = []
    place_names = []
    for entity in data.get("entities", []):
        name_lower = entity["name"].lower()
        if entity["type"] == "person":
            # Add full name and individual surname/firstname variants
            parts = entity["name"].lower().split()
            person_names.append(name_lower)
            person_names.extend(parts)
        else:
            place_names.append(name_lower)
            # Also add shortened versions (e.g., "eiffel" for "eiffel tower")
            place_names.extend(entity["name"].lower().split())

    # Deduplicate while preserving order
    person_names = list(dict.fromkeys(p for p in person_names if len(p) > 2))
    place_names = list(dict.fromkeys(p for p in place_names if len(p) > 2))
    return person_names, place_names


# Load entity names once at module import
_PERSON_NAMES, _PLACE_NAMES = _load_entities()

# Keywords that strongly suggest place queries
_PLACE_KEYWORDS = [
    "where", "located", "location", "city", "country", "tower", "wall",
    "mountain", "canyon", "temple", "church", "mosque", "statue", "pyramid",
    "palace", "castle", "rainforest", "desert", "falls", "ocean", "sea",
    "island", "continent", "river", "lake", "park", "monument", "landmark",
    "ruins", "ancient", "unesco", "heritage site",
]

# Keywords that strongly suggest person queries
_PERSON_KEYWORDS = [
    "who", "born", "died", "invented", "discovered", "wrote", "painted",
    "scientist", "artist", "writer", "musician", "athlete", "footballer",
    "physicist", "mathematician", "philosopher", "politician", "leader",
    "biography", "life", "career", "award", "nobel", "achievement",
]


def classify_query(query: str) -> QueryType:
    """
    Classifies a query as "person", "place", or "both".

    Args:
        query: The user's natural language question.

    Returns:
        "person", "place", or "both".
    """
    q = query.lower()

    # Check for known entity name mentions
    found_person = any(name in q for name in _PERSON_NAMES)
    found_place = any(name in q for name in _PLACE_NAMES)

    if found_person and found_place:
        return "both"
    if found_person:
        return "person"
    if found_place:
        return "place"

    # Fall back to keyword heuristics
    person_score = sum(1 for kw in _PERSON_KEYWORDS if kw in q)
    place_score = sum(1 for kw in _PLACE_KEYWORDS if kw in q)

    if person_score > 0 and place_score == 0:
        return "person"
    if place_score > 0 and person_score == 0:
        return "place"

    # Default: search everything
    return "both"


def get_all_entity_names() -> list[dict]:
    """
    Returns all entity names and types from entities.json.
    Used by other modules to display available entities.
    """
    with open(ENTITIES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entities", [])


if __name__ == "__main__":
    test_queries = [
        "Who was Albert Einstein?",
        "Where is the Eiffel Tower located?",
        "What did Marie Curie discover?",
        "Which famous place is located in Turkey?",
        "Compare Einstein and Nikola Tesla",
        "Compare the Eiffel Tower and the Colosseum",
        "Who is the president of Mars?",
        "Tell me about Taylor Swift",
        "What is Machu Picchu?",
    ]

    print("Query Routing Test:")
    print("-" * 50)
    for q in test_queries:
        result = classify_query(q)
        print(f"  [{result:6s}] {q}")
