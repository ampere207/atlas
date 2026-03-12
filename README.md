## Atlas — Distributed Intelligent RAG Engine (Phase 2)

Atlas now runs a working hybrid RAG backend with ingestion, vector + BM25 indexing/retrieval,
hybrid aggregation, reranking, context optimization, Gemini answering, and semantic caching.

### Stack

- FastAPI (Python 3.11)
- PostgreSQL (metadata placeholder)
- Redis (semantic cache)
- Qdrant (vector index/search)
- Elasticsearch (BM25 index/search)
- Neo4j (graph placeholder)
- Sentence Transformers (`BAAI/bge-small-en` embeddings)
- Cross-encoder reranker (`BAAI/bge-reranker-base`)
- Gemini API (`google-genai`)

### Configuration

Copy `.env.example` to `.env` and set at minimum:

- `GEMINI_API_KEY`
- `QDRANT_URL`
- `ELASTICSEARCH_URL`
- `REDIS_URL`

Key defaults:

- Qdrant collection: `documents`
- Elasticsearch index: `documents_index`
- Chunking: `500` tokens with `100` overlap
- Query default `top_k`: `10`

### Run

```powershell
uv sync
uv run python main.py
```

### API

- `GET /health`
- `POST /ingest` (multipart file upload)
- `POST /query`

`/query` flow:

1. semantic cache lookup
2. query classification
3. retrieval strategy selection
4. retrieval (vector/BM25/hybrid/sql/graph)
5. result aggregation
6. reranking
7. context optimization
8. Gemini answer generation
9. semantic cache write
