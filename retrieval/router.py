import asyncio

from query_intelligence.query_types import RetrievalStrategy
from retrieval.base_retriever import BaseRetriever, RetrievedDocument


class RetrievalRouter:
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        graph_retriever: BaseRetriever,
        sql_retriever: BaseRetriever,
    ) -> None:
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._graph = graph_retriever
        self._sql = sql_retriever

    async def route(
        self,
        query: str,
        strategy: RetrievalStrategy,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        if strategy == RetrievalStrategy.VECTOR:
            return await self._vector.search(query, top_k=top_k)
        if strategy == RetrievalStrategy.BM25:
            return await self._bm25.search(query, top_k=top_k)
        if strategy == RetrievalStrategy.GRAPH:
            return await self._graph.search(query, top_k=top_k)
        if strategy == RetrievalStrategy.SQL:
            return await self._sql.search(query, top_k=top_k)
        if strategy == RetrievalStrategy.HYBRID:
            results = await asyncio.gather(
                self._vector.search(query, top_k=top_k),
                self._bm25.search(query, top_k=top_k),
            )
            return self._aggregate(results, top_k)
        return await self._vector.search(query, top_k=top_k)

    def _aggregate(
        self, grouped_results: list[list[RetrievedDocument]], top_k: int
    ) -> list[RetrievedDocument]:
        merged: dict[str, RetrievedDocument] = {}
        for result_set in grouped_results:
            for document in result_set:
                existing = merged.get(document.document_id)
                if existing is None or document.score > existing.score:
                    merged[document.document_id] = document
        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
