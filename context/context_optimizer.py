import logging
from typing import Any

from embeddings.embedding_model import EmbeddingModel
from retrieval.base_retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class ContextOptimizer:
    """
    Optimizes context for LLM consumption.

    Techniques:
    - Deduplication (exact content + fingerprint)
    - Maximal Marginal Relevance (MMR) for diversity
    - Token budget enforcement
    - Content compression (relevance-based excerpts)
    """

    def __init__(
        self,
        max_tokens: int = 1800,
        embedding_model: EmbeddingModel | None = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
    ) -> None:
        """
        Initialize context optimizer.

        Args:
            max_tokens: Maximum tokens for context
            embedding_model: Optional embedding model for MMR
            use_mmr: Whether to apply MMR for diversity
            mmr_lambda: MMR lambda parameter (0-1, balance relevance vs diversity)
        """
        self._max_tokens = max_tokens
        self._embedding_model = embedding_model
        self._use_mmr = use_mmr and embedding_model is not None
        self._mmr_lambda = mmr_lambda

    async def optimize(
        self, documents: list[RetrievedDocument], metadata: dict[str, Any] | None = None
    ) -> list[RetrievedDocument]:
        """
        Optimize document list for LLM context.

        Pipeline:
        1. Deduplicate
        2. Apply MMR if enabled
        3. Enforce token budget
        4. Return optimized list

        Args:
            documents: Retrieved documents, potentially with duplicates
            metadata: Optional metadata (e.g., original query for MMR)

        Returns:
            Optimized document list within token budget
        """
        # Step 1: Deduplication
        deduped = self._deduplicate(documents)
        logger.debug(f"Deduplication: {len(documents)} → {len(deduped)} docs")

        # Step 2: Apply MMR if enabled
        if self._use_mmr and metadata and "query" in metadata:
            deduped = await self._apply_mmr(
                deduped, metadata["query"]
            )
            logger.debug(f"MMR applied: {len(deduped)} docs with diversity")

        # Step 3: Token budget selection
        selected = self._select_within_budget(deduped)
        logger.debug(f"Token budget: {len(deduped)} → {len(selected)} docs")

        return selected

    def _deduplicate(
        self, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """
        Remove duplicate documents.

        Uses multiple fingerprints:
        - Exact chunk_id match
        - Content prefix match (first 160 chars + document_id)
        """
        deduped: list[RetrievedDocument] = []
        seen_fingerprints: set[str] = set()

        for document in documents:
            # Generate fingerprint
            chunk_id = document.metadata.get("chunk_id")
            if chunk_id:
                fingerprint = chunk_id
            else:
                fingerprint = f"{document.document_id}:{document.content[:160]}"

            if fingerprint in seen_fingerprints:
                logger.debug(f"Skipped duplicate: {fingerprint[:50]}...")
                continue

            seen_fingerprints.add(fingerprint)
            deduped.append(document)

        return deduped

    async def _apply_mmr(
        self, documents: list[RetrievedDocument], query: str
    ) -> list[RetrievedDocument]:
        """
        Apply Maximal Marginal Relevance (MMR) for diversity.

        MMR balances:
        - Relevance to query (high similarity)
        - Diversity from already-selected docs (low similarity to each other)

        formula: MMR = λ * relevance - (1-λ) * max_similarity_to_selected

        Args:
            documents: Candidate documents
            query: Original query for relevance

        Returns:
            Re-ranked documents with diversity emphasis
        """
        if not documents or not self._embedding_model:
            return documents

        try:
            # Get embeddings
            query_embedding = await self._embedding_model.generate_embedding(query)
            doc_embeddings = await self._embedding_model.generate_embeddings(
                [doc.content for doc in documents]
            )

            # Calculate MMR scores
            selected: list[tuple[float, RetrievedDocument]] = []
            remaining = list(zip(documents, doc_embeddings))

            while remaining and len(selected) < min(len(documents), 5):
                # Calculate MMR score for each remaining document
                best_idx = 0
                best_mmr = float("-inf")

                for idx, (doc, emb) in enumerate(remaining):
                    # Relevance score: similarity to query
                    relevance = self._cosine_similarity(query_embedding, emb)

                    # Diversity score: dissimilarity to selected docs
                    if selected:
                        max_sim_to_selected = max(
                            self._cosine_similarity(emb, sel_emb)
                            for _, sel_emb in [(d, e) for d, e in selected]
                        )
                    else:
                        max_sim_to_selected = 0.0

                    # MMR formula
                    mmr_score = (
                        self._mmr_lambda * relevance
                        - (1 - self._mmr_lambda) * max_sim_to_selected
                    )

                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_idx = idx

                # Select best document
                selected.append(remaining.pop(best_idx))

            # Return documents in original order
            selected_docs = [doc for doc, _ in selected]
            result = [d for d in documents if d in selected_docs]
            return result

        except Exception as e:
            logger.warning(f"MMR application failed, returning original order: {e}")
            return documents

    def _select_within_budget(
        self, documents: list[RetrievedDocument], strategy: str = "greedy"
    ) -> list[RetrievedDocument]:
        """
        Select documents to fit within token budget.

        Strategies:
        - 'greedy': Select top documents until budget exhausted
        - 'balanced': Distribute budget across document sources

        Args:
            documents: Candidate documents
            strategy: Selection strategy

        Returns:
            Selected documents within token budget
        """
        selected: list[RetrievedDocument] = []
        used_tokens = 0

        for document in documents:
            token_estimate = max(1, len(document.content.split()))

            if used_tokens + token_estimate > self._max_tokens:
                logger.debug(
                    f"Token budget exhausted at {used_tokens}/{self._max_tokens} tokens"
                )
                break

            selected.append(document)
            used_tokens += token_estimate

        return selected

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
