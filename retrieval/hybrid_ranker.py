import logging
from typing import Any

from retrieval.base_retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class HybridRanker:
    """
    Ranks and combines results from multiple retrieval systems.

    Implements weighted scoring to intelligently merge vector, BM25, and graph results.
    """

    def __init__(
        self,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.3,
        graph_weight: float = 0.1,
    ) -> None:
        """
        Initialize HybridRanker with weights.

        Args:
            vector_weight: Weight for vector search results (default 0.6)
            bm25_weight: Weight for BM25 search results (default 0.3)
            graph_weight: Weight for graph search results (default 0.1)
        """
        # Normalize weights
        total = vector_weight + bm25_weight + graph_weight
        self.vector_weight = vector_weight / total
        self.bm25_weight = bm25_weight / total
        self.graph_weight = graph_weight / total

    def rank(
        self, documents: list[RetrievedDocument], top_k: int = 10
    ) -> list[RetrievedDocument]:
        """
        Rank documents by weighted hybrid score.

        Strategy:
        1. Normalize scores from each source (0-1 range)
        2. Apply source-specific weights
        3. Combine scores for documents from multiple sources
        4. Return top_k by combined score

        Args:
            documents: Mixed results from vector, BM25, and graph retrievers
            top_k: Number of top results to return

        Returns:
            Ranked list of top_k documents
        """
        if not documents:
            return []

        # Group documents by chunk_id to combine scores from multiple sources
        doc_map: dict[str, dict[str, Any]] = {}

        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", doc.document_id)

            if chunk_id not in doc_map:
                doc_map[chunk_id] = {
                    "document": doc,
                    "scores": {},
                }

            # Determine source and store score
            source = doc.source
            doc_map[chunk_id]["scores"][source] = doc.score

        # Calculate hybrid scores
        ranked_results: list[tuple[float, RetrievedDocument]] = []

        for chunk_id, entry in doc_map.items():
            doc = entry["document"]
            scores = entry["scores"]

            # Normalize and weight scores
            hybrid_score = 0.0

            # Vector score
            if "vector" in scores or "qdrant" in scores:
                source_key = "vector" if "vector" in scores else "qdrant"
                normalized_vector = min(1.0, scores.get(source_key, 0.0))
                hybrid_score += normalized_vector * self.vector_weight

            # BM25 score (Elasticsearch scores are typically 0-1 after normalization)
            if "elasticsearch" in scores or "bm25" in scores:
                source_key = "elasticsearch" if "elasticsearch" in scores else "bm25"
                normalized_bm25 = min(1.0, scores.get(source_key, 0.0))
                hybrid_score += normalized_bm25 * self.bm25_weight

            # Graph score
            if "graph" in scores:
                normalized_graph = min(1.0, scores.get("graph", 0.0))
                hybrid_score += normalized_graph * self.graph_weight

            # If only one source, use its normalized score
            if hybrid_score == 0.0 and scores:
                max_score = max(scores.values())
                hybrid_score = min(1.0, max_score)

            ranked_results.append((hybrid_score, doc))

        # Sort by score descending
        ranked_results.sort(key=lambda x: x[0], reverse=True)

        # Return top_k with updated scores
        results = []
        for score, doc in ranked_results[:top_k]:
            # Update document score to hybrid score
            doc.score = score
            results.append(doc)

        logger.debug(
            f"Hybrid ranking: {len(documents)} input docs → {len(results)} output docs"
        )
        return results

    def normalize_scores(
        self, documents: list[RetrievedDocument], source: str
    ) -> list[RetrievedDocument]:
        """
        Normalize scores from a specific source to 0-1 range.

        Args:
            documents: Documents with scores from a specific source
            source: Source name ("vector", "bm25", "graph", etc.)

        Returns:
            Documents with normalized scores
        """
        if not documents:
            return []

        # Find min/max scores
        scores = [doc.score for doc in documents if doc.score > 0]
        if not scores:
            return documents

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score or 1.0

        # Normalize each document
        normalized = []
        for doc in documents:
            normalized_score = (doc.score - min_score) / score_range
            doc.score = normalized_score
            normalized.append(doc)

        return normalized
