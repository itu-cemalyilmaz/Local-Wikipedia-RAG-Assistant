"""
chunker.py
----------
Splits a long text into fixed-size overlapping chunks using pure Python.
No LangChain or other chunking libraries are used.

Strategy: Fixed-size character chunks with overlap.
- CHUNK_SIZE: target size in characters (~400 tokens ≈ 1600 chars)
- OVERLAP: overlap in characters (~80 tokens ≈ 320 chars)

We split on sentence/paragraph boundaries when possible to avoid
cutting mid-sentence. Specifically we try to snap to the nearest
newline or period within a tolerance window of the target size.
"""

from typing import Optional

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1500     # characters per chunk
DEFAULT_OVERLAP = 300         # overlap characters between consecutive chunks
SNAP_WINDOW = 200             # characters to look back for a cleaner break point


def _find_break_point(text: str, target: int, window: int) -> int:
    """
    Find the best character index to break the text near `target`.
    Prefers paragraph breaks > sentence ends > word boundaries.

    Args:
        text: The text to search within.
        target: Target break position.
        window: How far before target to search for a natural break.

    Returns:
        The chosen break position (character index).
    """
    start_search = max(0, target - window)
    candidate = text[start_search:target]

    # Prefer double newline (paragraph break)
    idx = candidate.rfind("\n\n")
    if idx != -1:
        return start_search + idx + 2

    # Then single newline
    idx = candidate.rfind("\n")
    if idx != -1:
        return start_search + idx + 1

    # Then sentence end (. followed by space)
    idx = candidate.rfind(". ")
    if idx != -1:
        return start_search + idx + 2

    # Then any space (word boundary)
    idx = candidate.rfind(" ")
    if idx != -1:
        return start_search + idx + 1

    # Fallback: hard cut at target
    return target


def chunk_text(
    text: str,
    source_name: str,
    entity_type: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    Split `text` into overlapping chunks.

    Args:
        text: The full article text.
        source_name: Human-readable source name (e.g., "Albert Einstein").
        entity_type: "person" or "place".
        chunk_size: Target chunk size in characters.
        overlap: Overlap between consecutive chunks in characters.

    Returns:
        List of chunk dicts with keys:
            - chunk_id: unique string ID
            - text: chunk text
            - source: source entity name
            - type: "person" or "place"
            - chunk_index: sequential chunk number (0-based)
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    pos = 0
    chunk_index = 0
    safe_source = source_name.replace(" ", "_").lower()

    while pos < len(text):
        # Determine end of this chunk
        end = pos + chunk_size

        if end >= len(text):
            # Last chunk — take everything remaining
            chunk_text_str = text[pos:].strip()
        else:
            # Snap to a natural break point
            break_pos = _find_break_point(text, end, SNAP_WINDOW)
            chunk_text_str = text[pos:break_pos].strip()
            end = break_pos

        if chunk_text_str:
            chunk_id = f"{safe_source}_{chunk_index:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_str,
                "source": source_name,
                "type": entity_type,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        if end >= len(text):
            break

        # Move forward by (chunk_size - overlap), but at least 1 character
        step = max(1, chunk_size - overlap)
        pos = pos + step

    return chunks


def chunk_entities(
    entities: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    Chunk a list of entity dicts (each with 'text', 'name', 'type').

    Args:
        entities: List of {"name", "type", "text", ...} dicts.
        chunk_size: Characters per chunk.
        overlap: Overlap characters.

    Returns:
        Flat list of all chunk dicts across all entities.
    """
    all_chunks = []
    for entity in entities:
        chunks = chunk_text(
            text=entity["text"],
            source_name=entity["name"],
            entity_type=entity["type"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)
        print(f"  Chunked '{entity['name']}' -> {len(chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    sample = "This is a test. " * 500
    chunks = chunk_text(sample, "Test Entity", "person")
    print(f"Produced {len(chunks)} chunks from {len(sample)} chars")
    for c in chunks[:3]:
        print(f"  [{c['chunk_id']}] {len(c['text'])} chars: {c['text'][:80]}...")
