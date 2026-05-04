"""
wikipedia_fetcher.py
--------------------
Fetches plain-text content from Wikipedia using only stdlib (urllib + json).
No third-party wikipedia library is used.

Strategy:
  1. Try Wikipedia REST API summary endpoint for a short extract.
  2. Then fetch the full article sections via the Wikimedia Action API
     (action=query&prop=extracts&explaintext=1) to get the full plain text.
"""

import json
import time
import urllib.parse
import urllib.request
from typing import Optional


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "LocalRAGProject/1.0 (educational; contact: student@example.com)"

# Delay between requests to be polite to Wikipedia's servers
REQUEST_DELAY_SECONDS = 1.0


def _make_request(url: str, params: dict) -> dict:
    """Makes a GET request to a URL with query parameters and returns parsed JSON."""
    query_string = urllib.parse.urlencode(params)
    full_url = f"{url}?{query_string}"
    req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def fetch_wikipedia_text(title: str) -> Optional[str]:
    """
    Fetches the full plain-text content of a Wikipedia article by its page title.

    Args:
        title: The Wikipedia page title (e.g., "Albert_Einstein").

    Returns:
        A string with the full article plain text, or None on failure.
    """
    params = {
        "action": "query",
        "titles": title.replace("_", " "),
        "prop": "extracts",
        "explaintext": "1",       # plain text, no HTML or wikitext
        "exsectionformat": "plain",
        "redirects": "1",         # follow redirects automatically
        "format": "json",
        "utf8": "1",
    }

    try:
        data = _make_request(WIKIPEDIA_API, params)
        pages = data.get("query", {}).get("pages", {})

        for page_id, page in pages.items():
            if page_id == "-1":
                print(f"  [WARN] Page not found: {title}")
                return None
            text = page.get("extract", "")
            if not text:
                print(f"  [WARN] Empty extract for: {title}")
                return None
            return text

    except Exception as e:
        print(f"  [ERROR] Failed to fetch '{title}': {e}")
        return None

    return None


def fetch_all_entities(entities: list[dict], delay: float = REQUEST_DELAY_SECONDS) -> list[dict]:
    """
    Fetches Wikipedia text for a list of entity dicts.

    Args:
        entities: List of {"name": ..., "wikipedia_title": ..., "type": ...} dicts.
        delay: Seconds to wait between requests.

    Returns:
        List of {"name", "type", "text"} dicts (only successfully fetched).
    """
    results = []
    total = len(entities)

    for i, entity in enumerate(entities, 1):
        name = entity["name"]
        title = entity["wikipedia_title"]
        entity_type = entity["type"]

        print(f"[{i}/{total}] Fetching: {name} ({title}) ...")
        text = fetch_wikipedia_text(title)

        if text:
            word_count = len(text.split())
            print(f"  OK - {word_count:,} words")
            results.append({
                "name": name,
                "wikipedia_title": title,
                "type": entity_type,
                "text": text,
            })
        else:
            print(f"  SKIPPED - no content retrieved")

        if i < total:
            time.sleep(delay)

    return results


if __name__ == "__main__":
    # Quick smoke test
    text = fetch_wikipedia_text("Albert_Einstein")
    if text:
        print(f"Fetched {len(text):,} characters")
        print(text[:500])
    else:
        print("Failed to fetch test article")
