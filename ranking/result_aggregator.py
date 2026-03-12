from retrieval.base_retriever import RetrievedDocument


class ResultAggregator:
    async def aggregate(
        self,
        documents: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        merged: dict[str, RetrievedDocument] = {}
        for document in documents:
            dedupe_key = document.metadata.get("chunk_id") or f"{document.document_id}:{document.content[:120]}"
            existing = merged.get(dedupe_key)
            if existing is None:
                merged[dedupe_key] = document
                continue

            if document.score > existing.score:
                merged[dedupe_key] = document

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
