from retrieval.base_retriever import RetrievedDocument


class ContextOptimizer:
    def __init__(self, max_tokens: int = 1800) -> None:
        self._max_tokens = max_tokens

    async def optimize(self, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        deduped: list[RetrievedDocument] = []
        seen_fingerprints: set[str] = set()

        for document in documents:
            fingerprint = f"{document.document_id}:{document.content[:160]}"
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            deduped.append(document)

        selected: list[RetrievedDocument] = []
        used_tokens = 0
        for document in deduped:
            token_estimate = max(1, len(document.content.split()))
            if used_tokens + token_estimate > self._max_tokens:
                break
            selected.append(document)
            used_tokens += token_estimate

        return selected
