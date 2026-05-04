# Product Requirements Document — WikiRAG

**Product Name:** WikiRAG — Local Wikipedia RAG System  
**Version:** 1.0  
**Date:** 2026-04-29  
**Author:** Cemal Yılmaz

---

## 1. Problem Statement

Modern AI assistants rely on cloud-based LLMs with external API access, creating privacy concerns, internet dependency, and usage costs. For educational, offline, or privacy-sensitive environments, a fully local alternative is needed that can answer factual questions about well-known people and places using verified, structured knowledge.

---

## 2. Product Vision

Build a fully local, ChatGPT-style question-answering system that:
- Operates entirely on a laptop without internet access (after initial setup)
- Answers factual questions about 40+ famous people and places
- Retrieves knowledge from structured Wikipedia data
- Generates grounded, verifiable answers using a local LLM
- Clearly indicates when it does not know the answer

---

## 3. Target Users

- Students and researchers working offline
- Developers learning RAG system design
- Privacy-conscious users who cannot use cloud LLMs
- Educators demonstrating AI concepts locally

---

## 4. Core Features

### 4.1 Data Ingestion
- **Wikipedia Fetch:** Retrieve plain-text article content via MediaWiki API
- **Chunking:** Split articles into fixed-size segments (1500 chars, 300 overlap)
- **Embedding:** Generate local embeddings using `nomic-embed-text` via Ollama
- **Storage:** Persist embeddings in ChromaDB with entity type metadata
- **Logging:** Track ingestion state in SQLite to support idempotent re-runs

### 4.2 Retrieval
- Vector similarity search in ChromaDB (cosine distance)
- Metadata-filtered queries (person / place / both)
- Rule-based query router for fast, deterministic type classification
- Configurable result count (default: 5 chunks)

### 4.3 Generation
- Local LLM via Ollama (`llama3.2:3b` default)
- Strict RAG prompt: answer only from context
- "I don't know" fallback when context is insufficient
- Low temperature (0.1) for factual consistency
- Optional streaming output

### 4.4 Chat Interface
- Streamlit web UI at `localhost:8501`
- Chat history with session state
- Query type indicator (👤 Person / 🏛️ Place / 🔍 Both)
- Source chunk viewer (expandable per message)
- Sidebar: system status, entity list, settings
- Example query buttons for quick start
- Clear chat button

---

## 5. Entity Coverage

**20 People:** Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo, Isaac Newton, Stephen Hawking, Elon Musk, Mahatma Gandhi, Nelson Mandela, Cleopatra, Napoleon Bonaparte, Aristotle, Charles Darwin, Wolfgang Amadeus Mozart

**20 Places:** Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest, Stonehenge, Angkor Wat, Amazon Rainforest, Sahara Desert, Niagara Falls, Venice, Tokyo, Sydney Opera House, Acropolis of Athens, Petra

---

## 6. Technical Architecture

```
[Wikipedia REST API]
        │  (one-time ingestion)
        ▼
[wikipedia_fetcher.py] → plain text
        │
        ▼
[chunker.py] → fixed-size chunks with overlap
        │
        ▼
[embedder.py] → nomic-embed-text via Ollama
        │
        ▼
[ChromaDB] ← single collection, metadata: {type: person|place}
        │
        ▼  (at query time)
[router.py] → classify query type
        │
        ▼
[vector_store.py] → similarity search + metadata filter
        │
        ▼
[llm.py] → Ollama llama3.2:3b with RAG prompt
        │
        ▼
[chat_ui.py] → Streamlit chat interface
```

---

## 7. Technical Constraints

| Constraint | Value |
|---|---|
| Runtime | 100% localhost |
| Language | Python 3.10+ |
| LLM | Ollama (`llama3.2:3b`, `phi3`, or `mistral`) |
| Embedding | `nomic-embed-text` via Ollama |
| Vector DB | ChromaDB (persistent) |
| Metadata DB | SQLite |
| UI | Streamlit |
| External APIs | None (Wikipedia fetch is one-time setup only) |

---

## 8. Design Decisions and Rationale

### Vector Store: Option B (Single Collection + Metadata)
**Reason:** Simpler collection management; ChromaDB's native `where` filter handles entity type filtering cleanly; easier to extend with new categories.

### Chunking: Fixed-Size with Overlap
**Reason:** Wikipedia articles can exceed 50,000 characters. Fixed-size chunking is predictable and fast. Overlap (20% of chunk size) preserves context at boundaries, reducing information loss.

### Query Routing: Keyword-Based
**Reason:** Fast (no extra LLM call), deterministic, and sufficient for the known entity set. A semantic router would add 1-2 seconds of latency per query with minimal accuracy improvement.

### Embedding: nomic-embed-text
**Reason:** State-of-the-art local embedding model, 768-dim, available via Ollama with no API key. Outperforms older sentence-transformers models on retrieval benchmarks.

### LLM: llama3.2:3b
**Reason:** Best speed/quality tradeoff for CPU inference. 3B parameters fit in ~6GB RAM, reasonable on most laptops.

---

## 9. Success Criteria

| Metric | Target |
|---|---|
| Entity coverage | ≥ 40 entities indexed |
| Required entities | All 20 required people + places |
| Query accuracy | Correct answer for all required example questions |
| Failure handling | "I don't know" for unknown entities |
| Response time | < 30 seconds per query on CPU |
| Setup time | < 5 minutes following README |

---

## 10. Out of Scope (v1.0)

- Multi-turn memory / conversation history
- User-configurable entity addition via UI
- Automatic model selection based on hardware
- Production deployment (see `recommendation.md`)
- Real-time Wikipedia updates

---

## 11. Future Extensions

- Chat history memory (multi-turn context window)
- Citations with Wikipedia links
- Side-by-side model comparison
- Latency measurement dashboard
- Caching for repeated queries
- Improved re-ranking (BM25 hybrid)
- Entity addition via UI (drag-and-drop Wikipedia URL)
