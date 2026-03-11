import hashlib
import json
import logging
from typing import Any

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self, redis_client: Redis | None, ttl_seconds: int = 900) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        self._fallback_store: dict[str, str] = {}

    def _key(self, query: str) -> str:
        digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
        return f"atlas:semantic_cache:{digest}"

    async def get_cached_response(self, query: str) -> dict[str, Any] | None:
        key = self._key(query)
        if self._redis is None:
            cached = self._fallback_store.get(key)
            return json.loads(cached) if cached else None

        try:
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("Redis read failed, fallback store used: %s", exc)
            cached = self._fallback_store.get(key)
            return json.loads(cached) if cached else None

    async def store_response(self, query: str, response: dict[str, Any]) -> None:
        key = self._key(query)
        payload = json.dumps(response)

        if self._redis is None:
            self._fallback_store[key] = payload
            return

        try:
            await self._redis.set(key, payload, ex=self._ttl_seconds)
        except Exception as exc:
            logger.warning("Redis write failed, fallback store used: %s", exc)
            self._fallback_store[key] = payload
