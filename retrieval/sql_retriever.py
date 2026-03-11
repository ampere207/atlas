from retrieval.base_retriever import BaseRetriever, RetrievedDocument


class SQLRetriever(BaseRetriever):
    async def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                document_id="sql-1",
                content=f"SQL retrieval placeholder for query: {query}",
                source="postgres-placeholder",
                score=0.83,
                metadata={"retriever": "sql"},
            )
        ][:top_k]
