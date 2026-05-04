# WikiRAG — Local Wikipedia RAG System

> A fully local, ChatGPT-style Q&A system about famous people and places, powered by Ollama and ChromaDB. No internet connection required after initial data ingestion.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Local Model](#running-the-local-model)
- [Ingesting Data](#ingesting-data)
- [Starting the Application](#starting-the-application)
- [Example Queries](#example-queries)
- [Project Structure](#project-structure)

---

## Overview

WikiRAG is a Retrieval-Augmented Generation (RAG) system that:

1. **Ingests** Wikipedia articles for 40 famous people and places
2. **Chunks** them into overlapping text segments
3. **Embeds** each chunk using `mxbai-embed-large` (via Ollama, fully local)
4. **Stores** embeddings in ChromaDB (local vector database)
5. **Retrieves** relevant chunks for any user question
6. **Generates** grounded answers using `llama3.2:3b` (via Ollama, fully local)

Everything runs on `localhost` — no external API calls.

---

## System Architecture

```
User Query
    │
    ▼
[Query Router]          ← keyword-based: person / place / both
    │
    ▼
[ChromaDB Vector Store] ← cosine similarity search, metadata filter
    │  (mxbai-embed-large embeddings)
    │
    ▼
[Retrieved Chunks]      ← top-5 relevant text segments
    │
    ▼
[Ollama LLM]            ← llama3.2:3b, strict RAG prompt
    │
    ▼
[Answer]                ← grounded in context, "I don't know" fallback
```

**Design Decisions:**

| Decision | Choice | Rationale |
|---|---|---|
| Vector store | Single ChromaDB collection + metadata filter | Handles mixed queries natively; simpler management |
| Chunking | Fixed-size 1500 chars, 300-char overlap | Predictable for large docs; overlap preserves context at boundaries |
| Embedding | `mxbai-embed-large` via Ollama | Fully local, 1024-dim, excellent semantic quality. Changed from nomic due to identical-vector bug in some Ollama versions. |
| LLM | `llama3.2:3b` via Ollama | Fast on CPU, good instruction following |
| Query routing | Rule-based keyword matching | Fast, deterministic, zero extra latency |
| UI | Streamlit | Chat interface with session state |

---

## Prerequisites

- **Python** 3.10 or higher
- **Ollama** installed and running — [https://ollama.com/download](https://ollama.com/download)
- **Internet access** (one-time, for Wikipedia ingestion only)

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ai3
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Local Model

> [!IMPORTANT]
> **For the Instructor:** The `chroma_db_v2` database has been pre-computed and included in this repository so you don't have to wait 15-30 minutes for the ingestion script to run! You can skip directly to step 2 after starting Ollama and pulling the models.

### 1. Start Ollama

```bash
ollama serve
```

> Keep this terminal open. Ollama must be running for embedding and generation.

### 2. Pull the required models

```bash
# LLM for answer generation
ollama pull llama3.2:3b

# Embedding model
ollama pull mxbai-embed-large
```

Verify models are available:

```bash
ollama list
```

You should see both `llama3.2:3b` and `mxbai-embed-large` in the list.

---

## Ingesting Data

Run the ingestion pipeline to fetch Wikipedia articles, chunk them, and store embeddings:

```bash
python ingest/run_ingest.py
```

This will:
- Fetch Wikipedia articles for **40 entities** (20 people + 20 places)
- Chunk each article into ~1500-character segments with 300-character overlap
- Embed each chunk using `mxbai-embed-large`
- Store embeddings in `./chroma_db_v2/`
- Log ingestion status to `./ingest_log.db`

**Estimated time:** ~15-30 minutes depending on your hardware (embedding is the bottleneck).

### Options

```bash
# Re-ingest all entities (even if already indexed)
python ingest/run_ingest.py --force

# Ingest a single entity
python ingest/run_ingest.py --entity "Albert Einstein"

# Test fetching and chunking without storing (no Ollama needed)
python ingest/run_ingest.py --dry-run
```

> Ingestion is **idempotent** — running it again will skip already-indexed entities unless `--force` is used.

---

## Starting the Application

```bash
streamlit run app/chat_ui.py
```

Open your browser at **http://localhost:8501**

The sidebar shows:
- ✅ Ollama status
- ✅ ChromaDB status and chunk count
- List of indexed entities
- Model and retrieval settings

---

## Example Queries

### People
```
Who was Albert Einstein and what is he known for?
What did Marie Curie discover?
Why is Nikola Tesla famous?
Compare Lionel Messi and Cristiano Ronaldo
What is Frida Kahlo known for?
When was Leonardo da Vinci born?
What did Ada Lovelace contribute to computing?
```

### Places
```
Where is the Eiffel Tower located?
Why is the Great Wall of China important?
What is Machu Picchu?
What was the Colosseum used for?
Where is Mount Everest?
What is Hagia Sophia?
```

### Mixed
```
Which famous place is located in Turkey?
Which person is associated with electricity?
Compare Albert Einstein and Nikola Tesla
Compare the Eiffel Tower and the Statue of Liberty
```

### Failure cases (should return "I don't know")
```
Who is the president of Mars?
Tell me about John Doe
What is the capital of Atlantis?
```

---

## Project Structure

```
ai3/
├── README.md                  ← This file
├── product_prd.md             ← Product Requirements Document
├── recommendation.md          ← Production deployment recommendations
├── requirements.txt           ← Python dependencies
├── .gitignore
│
├── data/
│   └── entities.json          ← 40 entities (20 people + 20 places)
│
├── ingest/
│   ├── wikipedia_fetcher.py   ← Fetches Wikipedia plain text (stdlib only)
│   ├── chunker.py             ← Fixed-size overlap chunker (pure Python)
│   └── run_ingest.py          ← Ingestion entry point (CLI)
│
├── retrieval/
│   ├── embedder.py            ← Ollama mxbai-embed-large wrapper
│   ├── vector_store.py        ← ChromaDB CRUD operations
│   └── router.py              ← Query type classifier (person/place/both)
│
├── generation/
│   └── llm.py                 ← Ollama LLM chat with RAG prompt
│
├── app/
│   └── chat_ui.py             ← Streamlit chat interface
│
├── chroma_db_v2/              ← Persisted ChromaDB data (pre-computed)
└── ingest_log.db              ← SQLite ingestion log
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot connect to Ollama` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull llama3.2:3b` and `ollama pull mxbai-embed-large` |
| `No data indexed yet` | Run `python ingest/run_ingest.py` |
| Slow responses | Normal — local LLMs are slower than cloud APIs |
| ChromaDB error | Delete `chroma_db_v2/` folder and re-run ingestion |

---

## Demo Video

🎬 [Link to demo video](#) *(replace with your Loom/YouTube link)*
