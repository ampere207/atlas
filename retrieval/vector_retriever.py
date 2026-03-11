from retrieval.base_retriever import BaseRetriever, RetrievedDocument


class VectorRetriever(BaseRetriever):
    async def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                document_id="vec-1",
                content=f"Vector match for query: {query}",
                source="qdrant-placeholder",
                score=0.86,
                metadata={"retriever": "vector"},
            )
        ][:top_k]
