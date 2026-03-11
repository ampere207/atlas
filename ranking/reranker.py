from retrieval.base_retriever import RetrievedDocument


class Reranker:
    async def rerank(
        self, query: str, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        return sorted(documents, key=lambda item: item.score, reverse=True)
