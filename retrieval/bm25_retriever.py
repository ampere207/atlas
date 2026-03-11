from retrieval.base_retriever import BaseRetriever, RetrievedDocument


class BM25Retriever(BaseRetriever):
    async def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                document_id="bm25-1",
                content=f"BM25 keyword match for query: {query}",
                source="elasticsearch-placeholder",
                score=0.89,
                metadata={"retriever": "bm25"},
            )
        ][:top_k]
