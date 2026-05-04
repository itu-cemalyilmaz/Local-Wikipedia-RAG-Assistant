"""
vector_store.py
---------------
ChromaDB vector store wrapper for the Wikipedia RAG system.

Design: Single collection 'wiki_rag' with metadata field 'type' = 'person' | 'place'.
This allows flexible filtering per query while keeping management simple.

Collection schema per document:
  - id:         chunk_id string (e.g., "albert_einstein_0001")
  - document:   chunk text
  - metadata:   {"source": "Albert Einstein", "type": "person", "chunk_index": 1}
  - embedding:  handled by OllamaEmbeddingFunction
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings

# Ensure retrieval package is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.embedder import OllamaEmbeddingFunction

COLLECTION_NAME = "wiki_rag_v2"
CHROMA_DB_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db_v2")


def get_client() -> chromadb.PersistentClient:
    """Returns a ChromaDB PersistentClient backed by the local chroma_db/ directory."""
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection(client: Optional[chromadb.PersistentClient] = None) -> chromadb.Collection:
    """
    Returns the 'wiki_rag' collection, creating it if it doesn't exist.

    Args:
        client: Optional existing ChromaDB client. Creates one if not provided.

    Returns:
        ChromaDB Collection object.
    """
    if client is None:
        client = get_client()
    embedding_fn = OllamaEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    return collection


def upsert_chunks(chunks: List[dict], collection: Optional[chromadb.Collection] = None) -> int:
    """
    Upsert a list of chunk dicts into the vector store.

    Args:
        chunks: List of {"chunk_id", "text", "source", "type", "chunk_index"} dicts.
        collection: Optional existing collection. Creates one if not provided.

    Returns:
        Number of chunks upserted.
    """
    if collection is None:
        collection = get_collection()

    if not chunks:
        return 0

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "type": c["type"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def query_store(
    query_text: str,
    entity_type: Optional[str] = None,
    n_results: int = 5,
    collection: Optional[chromadb.Collection] = None,
) -> List[dict]:
    """
    Query the vector store for chunks relevant to `query_text`.

    Args:
        query_text: The user's question or query string.
        entity_type: Optional filter — "person", "place", or None (no filter).
        n_results: Number of results to return.
        collection: Optional existing collection.

    Returns:
        List of result dicts: {"text", "source", "type", "chunk_index", "distance"}
    """
    if collection is None:
        collection = get_collection()

    total_docs = collection.count()
    if total_docs == 0:
        return []

    # Cap n_results to what's available
    n_results = min(n_results, total_docs)

    where_filter = None
    if entity_type in ("person", "place"):
        where_filter = {"type": {"$eq": entity_type}}

    query_kwargs = {
        "query_texts": [query_text],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    output = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        output.append({
            "text": doc,
            "source": meta.get("source", "Unknown"),
            "type": meta.get("type", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": round(dist, 4),
        })

    return output


def get_collection_stats(collection: Optional[chromadb.Collection] = None) -> dict:
    """
    Returns statistics about the vector store collection.

    Returns:
        Dict with total_chunks, person_entities, place_entities counts.
    """
    if collection is None:
        client = get_client()
        try:
            collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=OllamaEmbeddingFunction(),
            )
        except Exception:
            return {"total_chunks": 0, "indexed_sources": []}

    total = collection.count()

    # Get unique sources
    try:
        all_meta = collection.get(include=["metadatas"])
        sources = {}
        for meta in all_meta.get("metadatas", []):
            name = meta.get("source", "?")
            t = meta.get("type", "?")
            sources[name] = t
        indexed = [{"name": k, "type": v} for k, v in sorted(sources.items())]
    except Exception:
        indexed = []

    return {
        "total_chunks": total,
        "indexed_sources": indexed,
    }


if __name__ == "__main__":
    print("Vector store smoke test...")
    stats = get_collection_stats()
    print(f"Total chunks in store: {stats['total_chunks']}")
    if stats["total_chunks"] > 0:
        results = query_store("Who is Albert Einstein?", entity_type="person", n_results=3)
        print(f"\nTop {len(results)} results for 'Who is Albert Einstein?':")
        for r in results:
            print(f"  [{r['source']}] dist={r['distance']} — {r['text'][:100]}...")
