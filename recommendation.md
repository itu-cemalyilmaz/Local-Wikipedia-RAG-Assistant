# Production Deployment Recommendation — WikiRAG

**Project:** WikiRAG — Local Wikipedia RAG System  
**Version:** 1.0  
**Date:** 2026-04-29

---

## Executive Summary

WikiRAG was designed and optimized for **local, single-user laptop execution**. Moving it to production requires addressing scalability, reliability, security, and cost dimensions that do not exist in the local prototype. This document outlines recommended architectures, technology choices, and tradeoffs for three production scenarios.

---

## 1. Current Architecture Limitations

| Limitation | Impact in Production |
|---|---|
| Ollama on single machine | Cannot handle concurrent users; no load balancing |
| Single ChromaDB instance (file-based) | No horizontal scaling; concurrent writes can corrupt state |
| No authentication | Anyone with network access can use the system |
| No caching | Every query re-runs embedding + retrieval + LLM |
| Static entity list | No mechanism to update Wikipedia data automatically |
| No monitoring | Cannot track latency, errors, or usage patterns |
| Streamlit single-threaded | Not designed for multi-user concurrent sessions |

---

## 2. Recommended Production Stack

### Scenario A: Small Team / Internal Tool (≤50 users)

**Goal:** Minimal infrastructure, maximum simplicity, low cost.

```
┌─────────────────────────────────────────────────────┐
│                  Single VM / Server                  │
│                                                     │
│  [Nginx] ──► [FastAPI Backend] ──► [Ollama Server]  │
│                     │                               │
│              [Chroma HTTP Server]                   │
│                     │                               │
│              [Redis Cache]                          │
│                     │                               │
│              [PostgreSQL + pgvector]  (optional)    │
└─────────────────────────────────────────────────────┘
```

**Key changes from prototype:**

| Component | Prototype | Production |
|---|---|---|
| UI | Streamlit | React/Next.js SPA or Streamlit behind auth |
| API | None | FastAPI REST API |
| LLM | Ollama (local) | Ollama on dedicated GPU VM |
| Vector DB | ChromaDB (file) | Chroma HTTP server or pgvector |
| Cache | None | Redis (cache embeddings + LLM responses) |
| Auth | None | OAuth2 / API key |
| Monitoring | None | Prometheus + Grafana |

**Recommended hardware:** 1× VM with 16GB RAM + NVIDIA GPU (RTX 3060+)

---

### Scenario B: Medium Scale (≤500 users)

**Goal:** Horizontal scalability with managed services.

```
[Load Balancer]
      │
  ┌───┴───┐
  │  API  │  (FastAPI, 2-4 replicas, Kubernetes/Docker Compose)
  └───┬───┘
      ├── [Redis Cache]
      ├── [Chroma Server] (dedicated pod)
      └── [Ollama Pool]   (2+ GPU workers, request queue)
```

**Key additions:**
- **LLM Gateway (LiteLLM):** Load balance across multiple Ollama instances, with automatic failover
- **Vector DB:** Self-hosted Weaviate or Qdrant (better horizontal scaling than Chroma)
- **Message Queue:** Celery + Redis for async embedding jobs during ingestion
- **Monitoring:** OpenTelemetry → Grafana stack for latency, error rates, token usage

---

### Scenario C: Enterprise / High Traffic (500+ users)

**Goal:** Full managed services, SLA guarantees, compliance.

| Component | Recommendation |
|---|---|
| LLM Inference | **vLLM** on dedicated GPU cluster (OpenAI-compatible API) |
| Embedding | Batch embedding service (TEI — Text Embeddings Inference) |
| Vector DB | **Weaviate** (managed cloud) or **Pinecone** |
| App Backend | FastAPI + Kubernetes (HPA auto-scaling) |
| Cache | Redis Enterprise / ElastiCache |
| Data pipeline | Apache Airflow for scheduled Wikipedia refresh |
| Auth | Auth0 / Okta with RBAC |
| Monitoring | Datadog or Grafana Cloud |
| Storage | S3 / GCS for article backups |

---

## 3. Critical Production Changes

### 3.1 Replace Streamlit with a Proper Frontend + API

**Prototype:** Streamlit handles UI + logic in a single Python file (single-threaded, no concurrency).

**Production:** Split into:
- **FastAPI backend** — `/api/chat`, `/api/ingest`, `/api/status` endpoints
- **React/Next.js frontend** — WebSocket for streaming, proper state management

```python
# Example FastAPI endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    query_type = classify_query(request.query)
    chunks = await vector_store.query_async(request.query, entity_type=query_type)
    answer = await llm.generate_async(request.query, chunks)
    return {"answer": answer, "sources": chunks, "query_type": query_type}
```

### 3.2 Add Response Caching

Cache both **embeddings** and **LLM responses** for repeated queries:

```python
import hashlib, redis

r = redis.Redis()

def get_cached_or_embed(text: str) -> list[float]:
    key = f"embed:{hashlib.md5(text.encode()).hexdigest()}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    embedding = embed_text(text)
    r.setex(key, 3600, json.dumps(embedding))  # 1 hour TTL
    return embedding
```

**Expected impact:** 40-60% reduction in LLM latency for repeated questions.

### 3.3 Scheduled Data Refresh

Wikipedia articles change over time. Set up a daily/weekly refresh:

```bash
# Cron job: re-ingest all entities weekly
0 2 * * 0 cd /app && python ingest/run_ingest.py --force
```

In production, use **Apache Airflow** or **Prefect** for orchestrated, monitored pipelines.

### 3.4 Upgrade Vector Database

| Database | Best For | Scaling |
|---|---|---|
| ChromaDB | Local dev, small datasets | Single node only |
| **Qdrant** | Production self-hosted | Horizontal cluster |
| **Weaviate** | Hybrid search (BM25 + vector) | Managed cloud |
| pgvector | Teams already using PostgreSQL | SQL-native |

**Recommendation:** Migrate to **Qdrant** for self-hosted production or **Weaviate** for managed cloud.

### 3.5 LLM Performance Optimization

| Technique | Impact | Effort |
|---|---|---|
| **GPU inference** | 5-10× faster than CPU | Medium |
| **vLLM continuous batching** | 10-100× throughput | High |
| **Quantized models** (GGUF Q4) | 2-4× memory reduction | Low |
| **Response streaming** | Perceived latency -70% | Low |
| **Prompt caching** | 30-50% token reduction | Medium |

---

## 4. Security Considerations

| Risk | Mitigation |
|---|---|
| Unauthenticated API access | OAuth2 / API key authentication |
| Prompt injection | Input sanitization, output filtering |
| Data poisoning | Verified Wikipedia source only, checksums |
| Sensitive data in logs | Anonymize query logs, GDPR compliance |
| Model output liability | Add disclaimer: "AI-generated, verify independently" |

---

## 5. Cost Estimation (Scenario A)

| Resource | Option | Monthly Cost |
|---|---|---|
| GPU VM (1× RTX 3060 equivalent) | AWS g4dn.xlarge | ~$250 |
| Vector DB (Chroma self-hosted) | Included in VM | $0 |
| Redis Cache | ElastiCache t3.micro | ~$15 |
| Load Balancer | AWS ALB | ~$20 |
| **Total** | | **~$285/month** |

> For ≤50 internal users, this is cost-effective. At 500+ users, move to Scenario B/C.

---

## 6. Monitoring Recommended Metrics

| Metric | Target |
|---|---|
| Query latency (p95) | < 15 seconds |
| Embedding latency (p95) | < 500ms |
| Vector search latency | < 100ms |
| LLM generation latency | < 10 seconds |
| "I don't know" rate | < 20% (indicates retrieval quality) |
| Cache hit rate | > 40% |
| Error rate | < 1% |

---

## 7. Summary Recommendations

1. **Short-term:** Keep the current local architecture for development and demo
2. **First production step:** Extract API layer (FastAPI) + add Redis caching
3. **Scale:** Migrate to Qdrant + vLLM + Kubernetes when concurrency exceeds 10 simultaneous users
4. **Always:** Add authentication before any public exposure
5. **Consider:** Moving to a fine-tuned model (LoRA on domain data) for better accuracy on specialized queries
