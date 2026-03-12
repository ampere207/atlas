import hashlib
import json
import logging
import math
from typing import Any

from embeddings.embedding_model import EmbeddingModel
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(
        self,
        redis_client: Redis | None,
        embedding_model: EmbeddingModel,
        ttl_seconds: int = 900,
        similarity_threshold: float = 0.9,
        max_entries: int = 500,
    ) -> None:
        self._redis = redis_client
        self._embedding_model = embedding_model
        self._ttl_seconds = ttl_seconds
        self._similarity_threshold = similarity_threshold
        self._max_entries = max_entries
        self._fallback_store: dict[str, str] = {}
        self._fallback_entries: list[dict[str, Any]] = []

    def _entry_key(self, query: str) -> str:
        digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
        return f"atlas:semantic_cache:entry:{digest}"

    def _entries_list_key(self) -> str:
        return "atlas:semantic_cache:entries"

    async def get_cached_response(self, query: str) -> dict[str, Any] | None:
        query_embedding = await self._embedding_model.generate_embedding(query)
        if not query_embedding:
            try:
                return await self._get_exact_match(query)
            except Exception as exc:
                logger.warning("Semantic exact cache read failed, using in-memory fallback: %s", exc)
                cached = self._fallback_store.get(self._entry_key(query))
                if not cached:
                    return None
                entry = json.loads(cached)
                return entry.get("response")

        if self._redis is None:
            return self._search_fallback_entries(query_embedding)

        try:
            raw_entries = await self._redis.lrange(self._entries_list_key(), 0, self._max_entries - 1)
            best_response: dict[str, Any] | None = None
            best_score = -1.0

            for raw in raw_entries:
                entry = json.loads(raw)
                candidate_embedding = entry.get("embedding", [])
                score = self._cosine_similarity(query_embedding, candidate_embedding)
                if score > best_score and score >= self._similarity_threshold:
                    best_score = score
                    best_response = entry.get("response")
            return best_response
        except Exception as exc:
            logger.warning("Redis read failed, fallback store used: %s", exc)
            return self._search_fallback_entries(query_embedding)

    async def store_response(self, query: str, response: dict[str, Any]) -> None:
        query_embedding = await self._embedding_model.generate_embedding(query)
        entry_payload = {
            "query": query,
            "embedding": query_embedding,
            "response": response,
        }

        key = self._entry_key(query)
        entry_json = json.dumps(entry_payload)

        if self._redis is None:
            self._fallback_store[key] = entry_json
            self._fallback_entries.insert(0, entry_payload)
            self._fallback_entries = self._fallback_entries[: self._max_entries]
            return

        try:
            await self._redis.set(key, entry_json, ex=self._ttl_seconds)
            await self._redis.lpush(self._entries_list_key(), entry_json)
            await self._redis.ltrim(self._entries_list_key(), 0, self._max_entries - 1)
        except Exception as exc:
            logger.warning("Redis write failed, fallback store used: %s", exc)
            self._fallback_store[key] = entry_json
            self._fallback_entries.insert(0, entry_payload)
            self._fallback_entries = self._fallback_entries[: self._max_entries]

    async def _get_exact_match(self, query: str) -> dict[str, Any] | None:
        key = self._entry_key(query)
        if self._redis is None:
            cached = self._fallback_store.get(key)
            if not cached:
                return None
            entry = json.loads(cached)
            return entry.get("response")

        raw = await self._redis.get(key)
        if not raw:
            return None
        entry = json.loads(raw)
        return entry.get("response")

    def _search_fallback_entries(self, query_embedding: list[float]) -> dict[str, Any] | None:
        best_response: dict[str, Any] | None = None
        best_score = -1.0

        for entry in self._fallback_entries:
            score = self._cosine_similarity(query_embedding, entry.get("embedding", []))
            if score > best_score and score >= self._similarity_threshold:
                best_score = score
                best_response = entry.get("response")
        return best_response

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return -1.0
        return dot / (mag_a * mag_b)
