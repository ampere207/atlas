# Atlas — Distributed Intelligent RAG Engine

## Overview

Atlas is a production-grade **Retrieval-Augmented Generation (RAG)** system that intelligently retrieves knowledge from multiple sources, ranks results intelligently, and generates grounded responses using Google Gemini.

Instead of relying solely on LLM knowledge, Atlas acts as a **knowledge intelligence layer** that:
- Understands your question semantically
- Searches across vector stores, full-text indices, and knowledge graphs
- Combines results intelligently using hybrid ranking
- Optimizes context within token limits
- Grounds Gemini responses in retrieved documents

---

## What It Does

**Query Processing Flow:**
```
Your Question
    ↓
[Query Classification] (Understand query intent)
    ↓
[Strategy Selection] (Choose optimal retrieval approach)
    ↓
[Multi-Source Retrieval] (Search vector DB, Elasticsearch, Neo4j)
    ↓
[Intelligent Ranking] (Combine scores, deduplicate, rerank)
    ↓
[Context Optimization] (Fit results within token budget)
    ↓
[Grounded Response] (Gemini answers using only retrieved context)
    ↓
[Cached] (Store for instant retrieval next time)
```

---

## Key Features

✨ **Query Intelligence** - Classifies queries (factual, conceptual, multi-hop, code, analytics) and selects optimal retrieval strategy

🔍 **Hybrid Retrieval** - Combines vector search (semantic), BM25 (keyword), and knowledge graphs (relationships)

🏆 **Smart Ranking** - Cross-encoder reranking, deduplication, and hybrid score combination

💾 **Semantic Caching** - Instant responses for similar queries using embedding similarity

⚡ **Async Pipelines** - Non-blocking ingestion and query processing for high throughput

🧠 **Grounded Answers** - Gemini LLM constrained to answer only from retrieved context

---

## Tech Stack

- **Framework:** FastAPI (Python 3.11)
- **Vector Store:** Qdrant (semantic search)
- **Full-Text Search:** Elasticsearch (BM25)
- **Knowledge Graph:** Neo4j (entity relationships)
- **Cache:** Redis (semantic cache)
- **LLM:** Google Gemini 2.5 Flash
- **Embeddings:** BAAI/bge-small-en
- **Reranking:** BAAI/bge-reranker-base (cross-encoder)

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (for Qdrant, Elasticsearch, Redis, Neo4j)
- Google Gemini API key


### Test It

```bash
# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample-guardduty.md" \
  -F "source=demo"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is AWS GuardDuty?","top_k":5}'

# View metrics
curl http://localhost:8000/metrics
```

---

## How It Works

1. **Ingestion**: Documents are parsed, chunked, embedded, and indexed in multiple stores
2. **Query Classification**: Gemini classifies query type to determine retrieval strategy
3. **Adaptive Retrieval**: Different queries use different strategies:
   - Short factual → BM25 (fast keyword search)
   - Conceptual → Vector (semantic similarity)
   - Complex → Hybrid (combines multiple signals)
   - Relationships → Graph (entity connections)
4. **Intelligent Ranking**: Results combined via weighted hybrid score, deduplicated, cross-encoded reranked
5. **Context Optimization**: Results trimmed to token budget with Maximal Marginal Relevance (MMR) for diversity
6. **Grounded Response**: Input context to Gemini with prompt: *"Answer using ONLY the provided context"*
7. **Caching**: Response cached by query embedding for instant retrieval

---

## Example Queries

```bash
# Factual lookup (uses BM25)
"What is AWS GuardDuty?"

# Conceptual explanation (uses VECTOR)
"Explain AWS security architecture and how services integrate"

# Multi-hop reasoning (uses HYBRID)
"How does GuardDuty work with Security Hub for threat detection?"

# Code search (uses HYBRID)
"Why do FastAPI import errors happen?"
```

---

## Architecture

**Core Components:**
- `query_intelligence/` - Query classification and strategy selection
- `retrieval/` - Vector, BM25, graph, and SQL retrievers
- `ranking/` - Result aggregation, hybrid ranking, cross-encoder reranking
- `context/` - Context optimization with MMR and deduplication
- `cache/` - Semantic caching with Redis backend
- `ingestion/` - Async document parsing, chunking, embedding, indexing
- `api/` - FastAPI routes for ingestion and querying
- `metrics/` - Performance tracking and analytics

---

## Features in Detail

📚 **See [README1.md](README1.md) for comprehensive documentation** including:
- Full system architecture
- Detailed query flow
- Ingestion pipeline
- Setup instructions
- Architecture diagrams
- Performance metrics

---

## Performance

- **Cache hits:** Instant (< 150ms)
- **Full query:** ~8-10s (classification + retrieval + reranking + LLM)
- **Throughput:** Concurrent async processing
- **Scalability:** Stateless API, distributed backends

---

## Status

✅ Phase 1 - Query classification and routing  
✅ Phase 2 - Working RAG (vector, BM25, reranking, Gemini)  
✅ Phase 3 - Advanced intelligence (graph, MMR, adaptive strategies, metrics)  

---

## Next Steps

- Production deployment
- Fine-tuning hybrid weights based on your data
- Custom embedding models for domain-specific search
- Query logging and continuous improvement

---

**Atlas**: Where your documents meet intelligent retrieval.
