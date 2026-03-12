import asyncio
import logging
from typing import Any

from ingestion.chunking import Chunk
from retrieval.base_retriever import BaseRetriever, RetrievedDocument

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
except ImportError:  # pragma: no cover
    Elasticsearch = None
    bulk = None


class BM25Retriever(BaseRetriever):
    def __init__(self, elasticsearch_url: str, index_name: str = "documents_index") -> None:
        self._elasticsearch_url = elasticsearch_url
        self._index_name = index_name
        self._client: Any | None = None
        self._client_init_attempted = False
        self._index_initialized = False
        self._fallback_docs: list[dict[str, Any]] = []

    async def index_chunks(self, chunks: list[Chunk], metadata: dict[str, Any]) -> None:
        if not chunks:
            return

        await self._ensure_client()
        if self._client is None or bulk is None:
            for chunk in chunks:
                self._fallback_docs.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "chunk_text": chunk.text,
                        "metadata": metadata,
                    }
                )
            return

        await self._ensure_index()

        actions = [
            {
                "_index": self._index_name,
                "_id": chunk.chunk_id,
                "_source": {
                    "document_id": chunk.document_id,
                    "chunk_text": chunk.text,
                    "metadata": metadata,
                },
            }
            for chunk in chunks
        ]

        def _bulk_index() -> None:
            bulk(self._client, actions)

        try:
            await asyncio.to_thread(_bulk_index)
        except Exception as exc:
            logger.warning("Elasticsearch bulk index failed, using in-memory fallback: %s", exc)
            for chunk in chunks:
                self._fallback_docs.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "chunk_text": chunk.text,
                        "metadata": metadata,
                    }
                )

    async def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        await self._ensure_client()
        if self._client is None:
            return self._search_fallback(query, top_k)

        await self._ensure_index()

        def _search() -> dict[str, Any]:
            return self._client.search(
                index=self._index_name,
                size=top_k,
                query={
                    "match": {
                        "chunk_text": {
                            "query": query,
                        }
                    }
                },
            )

        try:
            response = await asyncio.to_thread(_search)
            hits = response.get("hits", {}).get("hits", [])
            docs: list[RetrievedDocument] = []
            for hit in hits:
                source = hit.get("_source", {})
                docs.append(
                    RetrievedDocument(
                        document_id=str(source.get("document_id", hit.get("_id", ""))),
                        content=str(source.get("chunk_text", "")),
                        source="elasticsearch",
                        score=float(hit.get("_score", 0.0)),
                        metadata={"retriever": "bm25", **source.get("metadata", {})},
                    )
                )
            return docs
        except Exception as exc:
            logger.warning("Elasticsearch search failed, using in-memory fallback: %s", exc)
            return self._search_fallback(query, top_k)

    async def _ensure_client(self) -> None:
        if self._client is not None or self._client_init_attempted:
            return

        self._client_init_attempted = True
        if Elasticsearch is None:
            return

        def _build_client() -> Any:
            return Elasticsearch(self._elasticsearch_url)

        try:
            self._client = await asyncio.to_thread(_build_client)
        except Exception as exc:
            logger.warning("Elasticsearch client unavailable: %s", exc)
            self._client = None

    async def _ensure_index(self) -> None:
        if self._client is None or self._index_initialized:
            return

        def _create_if_missing() -> None:
            exists = self._client.indices.exists(index=self._index_name)
            if exists:
                return
            self._client.indices.create(
                index=self._index_name,
                mappings={
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "chunk_text": {"type": "text"},
                        "metadata": {"type": "object", "enabled": True},
                    }
                },
            )

        try:
            await asyncio.to_thread(_create_if_missing)
            self._index_initialized = True
        except Exception as exc:
            logger.warning("Elasticsearch index initialization failed: %s", exc)

    def _search_fallback(self, query: str, top_k: int) -> list[RetrievedDocument]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        if not query_terms:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self._fallback_docs:
            text_terms = {term.lower() for term in str(doc.get("chunk_text", "")).split()}
            overlap = len(query_terms.intersection(text_terms))
            score = float(overlap) / float(max(len(query_terms), 1))
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)

        docs: list[RetrievedDocument] = []
        for score, doc in scored[:top_k]:
            docs.append(
                RetrievedDocument(
                    document_id=str(doc.get("document_id", doc.get("chunk_id", ""))),
                    content=str(doc.get("chunk_text", "")),
                    source="elasticsearch-fallback",
                    score=score,
                    metadata={"retriever": "bm25", **doc.get("metadata", {})},
                )
            )
        return docs
