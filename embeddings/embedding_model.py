import asyncio
from typing import Any


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en") -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._init_attempted = False

    async def generate_embedding(self, text: str) -> list[float]:
        if not text.strip():
            return []

        await self._ensure_model()
        if self._model is None:
            return []

        vector = await asyncio.to_thread(self._encode_single, text)
        return [float(value) for value in vector]

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        await self._ensure_model()
        if self._model is None:
            return [[] for _ in texts]

        vectors = await asyncio.to_thread(self._encode_batch, texts)
        return [[float(value) for value in vector] for vector in vectors]

    async def _ensure_model(self) -> None:
        if self._model is not None or self._init_attempted:
            return

        self._init_attempted = True

        def _load() -> Any:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(self._model_name)

        try:
            self._model = await asyncio.to_thread(_load)
        except Exception:
            self._model = None

    def _encode_single(self, text: str) -> list[float]:
        if self._model is None:
            return []
        vector = self._model.encode(text, normalize_embeddings=True)
        return list(vector)

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(item) for item in vectors]
