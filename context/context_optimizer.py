from retrieval.base_retriever import RetrievedDocument


class ContextOptimizer:
    async def optimize(self, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        return documents
