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
            vector_docs, bm25_docs = await asyncio.gather(
                self._vector.search(query, top_k=top_k),
                self._bm25.search(query, top_k=top_k),
            )
            return vector_docs + bm25_docs
        return await self._vector.search(query, top_k=top_k)
