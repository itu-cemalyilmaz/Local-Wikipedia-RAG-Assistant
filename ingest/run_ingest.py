"""
run_ingest.py
-------------
Entry point for the ingestion pipeline.

Pipeline:
  1. Load entity list from data/entities.json
  2. For each entity: fetch Wikipedia text
  3. Chunk the text (fixed-size overlap)
  4. Embed + upsert chunks into ChromaDB
  5. Log ingestion metadata to SQLite (ingest_log.db)

Run with:
    python ingest/run_ingest.py

Options:
    --force     Re-ingest all entities even if already indexed
    --entity    Ingest a specific entity by name (e.g., --entity "Albert Einstein")
    --dry-run   Fetch and chunk without storing (for testing)
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.chunker import chunk_entities
from ingest.wikipedia_fetcher import fetch_wikipedia_text
from retrieval.embedder import OllamaEmbeddingFunction
from retrieval.vector_store import get_collection, upsert_chunks, get_collection_stats

ENTITIES_JSON = ROOT / "data" / "entities.json"
INGEST_LOG_DB = ROOT / "ingest_log.db"


# ─────────────────────────────────────────────
# SQLite ingestion log
# ─────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    """Initializes the SQLite ingestion log database."""
    conn = sqlite3.connect(str(INGEST_LOG_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingest_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            type        TEXT NOT NULL,
            wiki_title  TEXT NOT NULL,
            char_count  INTEGER,
            chunk_count INTEGER,
            status      TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def get_already_ingested(conn: sqlite3.Connection) -> set[str]:
    """Returns set of entity names that were previously successfully ingested."""
    rows = conn.execute(
        "SELECT name FROM ingest_log WHERE status = 'ok'"
    ).fetchall()
    return {row[0] for row in rows}


def log_ingest(conn: sqlite3.Connection, name: str, entity_type: str,
               wiki_title: str, char_count: int, chunk_count: int, status: str):
    """Logs an ingestion result to SQLite."""
    conn.execute(
        """INSERT INTO ingest_log (name, type, wiki_title, char_count, chunk_count, status, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (name, entity_type, wiki_title, char_count, chunk_count, status,
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()


# ─────────────────────────────────────────────
# Main ingestion logic
# ─────────────────────────────────────────────

def load_entities(filter_name: str = None) -> list[dict]:
    """Loads entities from entities.json, optionally filtering by name."""
    with open(ENTITIES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    entities = data.get("entities", [])
    if filter_name:
        entities = [e for e in entities if e["name"].lower() == filter_name.lower()]
        if not entities:
            print(f"[ERROR] Entity not found: '{filter_name}'")
            sys.exit(1)
    return entities


def run_ingestion(
    entities: list[dict],
    collection,
    conn: sqlite3.Connection,
    force: bool = False,
    dry_run: bool = False,
):
    """
    Runs the full ingestion pipeline for a list of entities.

    Args:
        entities: List of entity dicts from entities.json.
        collection: ChromaDB collection to upsert into.
        conn: SQLite connection for logging.
        force: If True, re-ingest even if already in log.
        dry_run: If True, don't actually store in ChromaDB.
    """
    already_ingested = get_already_ingested(conn) if not force else set()
    total = len(entities)
    skipped = 0
    success = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f" Wikipedia RAG - Ingestion Pipeline")
    print(f" Entities to process: {total}")
    print(f" Force re-ingest: {force} | Dry run: {dry_run}")
    print(f"{'='*60}\n")

    for i, entity in enumerate(entities, 1):
        name = entity["name"]
        title = entity["wikipedia_title"]
        entity_type = entity["type"]

        print(f"[{i:02d}/{total}] {name} ({entity_type})")

        # Skip if already ingested
        if name in already_ingested:
            print(f"  -> Skipped (already indexed). Use --force to re-ingest.\n")
            skipped += 1
            continue

        # Step 1: Fetch Wikipedia text
        text = fetch_wikipedia_text(title)
        if not text:
            print(f"  -> FAILED: could not fetch Wikipedia content\n")
            log_ingest(conn, name, entity_type, title, 0, 0, "fetch_failed")
            failed += 1
            time.sleep(0.5)
            continue

        char_count = len(text)
        print(f"  Fetched: {char_count:,} characters")

        # Step 2: Chunk
        chunks = chunk_entities([{"name": name, "type": entity_type, "text": text}])
        chunk_count = len(chunks)
        print(f"  Chunks:  {chunk_count}")

        if dry_run:
            print(f"  -> DRY RUN: not stored.\n")
            success += 1
            continue

        # Step 3: Embed + Upsert
        try:
            upserted = upsert_chunks(chunks, collection=collection)
            print(f"  Stored:  {upserted} chunks in ChromaDB")
            log_ingest(conn, name, entity_type, title, char_count, chunk_count, "ok")
            success += 1
        except Exception as e:
            print(f"  -> FAILED during embedding/storage: {e}\n")
            log_ingest(conn, name, entity_type, title, char_count, 0, f"embed_failed")
            failed += 1
            continue

        print()
        # Polite delay between Wikipedia requests
        if i < total:
            time.sleep(1.0)

    # Summary
    print(f"\n{'='*60}")
    print(f" Ingestion complete!")
    print(f"  Success:  {success}")
    print(f"  Skipped:  {skipped}")
    print(f"  Failed:   {failed}")

    if not dry_run:
        stats = get_collection_stats(collection)
        print(f"  Total chunks in store: {stats['total_chunks']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Wikipedia RAG Ingestion Pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest all entities even if already indexed")
    parser.add_argument("--entity", type=str, default=None,
                        help='Ingest a specific entity by name (e.g., "Albert Einstein")')
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and chunk without storing in ChromaDB")
    args = parser.parse_args()

    # Load entities
    entities = load_entities(filter_name=args.entity)

    # Initialize SQLite log
    conn = init_db()

    # Initialize ChromaDB collection (unless dry run)
    collection = None
    if not args.dry_run:
        print("Connecting to ChromaDB...")
        try:
            collection = get_collection()
            print(f"  Collection ready: '{collection.name}'")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ChromaDB: {e}")
            print("Make sure Ollama is running: ollama serve")
            sys.exit(1)

    # Run ingestion
    run_ingestion(
        entities=entities,
        collection=collection,
        conn=conn,
        force=args.force,
        dry_run=args.dry_run,
    )

    conn.close()


if __name__ == "__main__":
    main()
