import asyncio
import logging
from typing import Any

from embeddings.embedding_model import EmbeddingModel
from ingestion.chunking import Chunk
from retrieval.base_retriever import BaseRetriever, RetrievedDocument

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams
except ImportError:  # pragma: no cover
    QdrantClient = None
    Distance = None
    PointStruct = None
    VectorParams = None


class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        qdrant_url: str,
        collection_name: str = "documents",
    ) -> None:
        self._embedding_model = embedding_model
        self._qdrant_url = qdrant_url
        self._collection_name = collection_name
        self._client: Any | None = None
        self._client_init_attempted = False
        self._collection_initialized = False
        self._fallback_points: list[dict[str, Any]] = []

    async def index_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        metadata: dict[str, Any],
    ) -> None:
        if not chunks or not embeddings:
            return

        await self._ensure_client()
        if self._client is None:
            for chunk, vector in zip(chunks, embeddings):
                self._fallback_points.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "chunk_text": chunk.text,
                        "vector": vector,
                        "metadata": metadata,
                    }
                )
            return

        vector_size = len(embeddings[0]) if embeddings and embeddings[0] else 0
        if vector_size <= 0:
            return

        await self._ensure_collection(vector_size)

        def _upsert() -> None:
            points = [
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "chunk_text": chunk.text,
                        "metadata": metadata,
                    },
                )
                for chunk, vector in zip(chunks, embeddings)
                if vector
            ]
            if points:
                self._client.upsert(collection_name=self._collection_name, points=points)

        try:
            await asyncio.to_thread(_upsert)
        except Exception as exc:
            logger.warning("Qdrant upsert failed, falling back in-memory: %s", exc)
            for chunk, vector in zip(chunks, embeddings):
                self._fallback_points.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "chunk_text": chunk.text,
                        "vector": vector,
                        "metadata": metadata,
                    }
                )

    async def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        query_embedding = await self._embedding_model.generate_embedding(query)
        if not query_embedding:
            return []

        await self._ensure_client()
        if self._client is None:
            return self._search_fallback(query_embedding, top_k)

        await self._ensure_collection(len(query_embedding))

        def _search() -> Any:
            return self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
            )

        try:
            response = await asyncio.to_thread(_search)
            points = response.points if hasattr(response, "points") else []
            docs: list[RetrievedDocument] = []
            for point in points:
                payload = point.payload or {}
                docs.append(
                    RetrievedDocument(
                        document_id=str(payload.get("document_id", point.id)),
                        content=str(payload.get("chunk_text", "")),
                        source="qdrant",
                        score=float(point.score or 0.0),
                        metadata={"retriever": "vector", **payload.get("metadata", {})},
                    )
                )
            return docs
        except Exception as exc:
            logger.warning("Qdrant query failed, using in-memory vector fallback: %s", exc)
            return self._search_fallback(query_embedding, top_k)

    async def _ensure_client(self) -> None:
        if self._client is not None or self._client_init_attempted:
            return

        self._client_init_attempted = True
        if QdrantClient is None:
            return

        def _build_client() -> Any:
            return QdrantClient(url=self._qdrant_url)

        try:
            self._client = await asyncio.to_thread(_build_client)
        except Exception as exc:
            logger.warning("Qdrant client unavailable: %s", exc)
            self._client = None

    async def _ensure_collection(self, vector_size: int) -> None:
        if self._client is None or self._collection_initialized:
            return

        def _create_if_missing() -> None:
            try:
                self._client.get_collection(self._collection_name)
            except Exception:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )

        try:
            await asyncio.to_thread(_create_if_missing)
            self._collection_initialized = True
        except Exception as exc:
            logger.warning("Qdrant collection initialization failed: %s", exc)

    def _search_fallback(self, query_embedding: list[float], top_k: int) -> list[RetrievedDocument]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for point in self._fallback_points:
            score = self._cosine_similarity(query_embedding, point.get("vector", []))
            scored.append((score, point))

        scored.sort(key=lambda item: item[0], reverse=True)
        docs: list[RetrievedDocument] = []
        for score, point in scored[:top_k]:
            docs.append(
                RetrievedDocument(
                    document_id=str(point.get("document_id", point.get("chunk_id", ""))),
                    content=str(point.get("chunk_text", "")),
                    source="qdrant-fallback",
                    score=float(max(score, 0.0)),
                    metadata={"retriever": "vector", **point.get("metadata", {})},
                )
            )
        return docs

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(y * y for y in b) ** 0.5
        if mag_a == 0.0 or mag_b == 0.0:
            return -1.0
        return dot / (mag_a * mag_b)
