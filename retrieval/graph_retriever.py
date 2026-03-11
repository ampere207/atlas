from retrieval.base_retriever import BaseRetriever, RetrievedDocument


class GraphRetriever(BaseRetriever):
    async def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                document_id="graph-1",
                content=f"Graph traversal placeholder for query: {query}",
                source="neo4j-placeholder",
                score=0.8,
                metadata={"retriever": "graph"},
            )
        ][:top_k]
