import asyncio
from typing import Any

from retrieval.base_retriever import RetrievedDocument


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._init_attempted = False

    async def rerank(
        self, query: str, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        if not documents:
            return []

        await self._ensure_model()
        if self._model is None:
            return sorted(documents, key=lambda item: item.score, reverse=True)

        pairs = [(query, document.content) for document in documents]
        scores = await asyncio.to_thread(self._predict_scores, pairs)

        reranked: list[RetrievedDocument] = []
        for document, score in zip(documents, scores):
            reranked.append(
                RetrievedDocument(
                    document_id=document.document_id,
                    content=document.content,
                    source=document.source,
                    score=float(score),
                    metadata=document.metadata,
                )
            )
        return sorted(reranked, key=lambda item: item.score, reverse=True)

    async def _ensure_model(self) -> None:
        if self._model is not None or self._init_attempted:
            return

        self._init_attempted = True

        def _load() -> Any:
            from sentence_transformers import CrossEncoder

            return CrossEncoder(self._model_name)

        try:
            self._model = await asyncio.to_thread(_load)
        except Exception:
            self._model = None

    def _predict_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._model is None:
            return [0.0 for _ in pairs]
        raw_scores = self._model.predict(pairs)
        return [float(score) for score in raw_scores]
