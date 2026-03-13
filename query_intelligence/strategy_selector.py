import logging
from typing import Any

from metrics.retrieval_metrics import get_metrics
from query_intelligence.query_types import (
    QueryClassification,
    QueryType,
    RetrievalStrategy,
)

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Selects optimal retrieval strategy based on query characteristics and metrics.

    Heuristics:
    - Short factual queries (< 10 words) → BM25 (faster, keyword-focused)
    - Long conceptual queries (> 15 words) → VECTOR (semantic matching)
    - Multi-topic queries (multiple entities) → HYBRID (combines signals)
    - Relationship queries (contains "how", "why", "relationship") → GRAPH (entity connections)
    - Structured queries (SQL syntax evident) → SQL (data queries)
    """

    _MAPPING: dict[QueryType, RetrievalStrategy] = {
        QueryType.FACT_LOOKUP: RetrievalStrategy.BM25,
        QueryType.CONCEPTUAL: RetrievalStrategy.VECTOR,
        QueryType.MULTI_HOP_REASONING: RetrievalStrategy.HYBRID,
        QueryType.CODE_SEARCH: RetrievalStrategy.HYBRID,
        QueryType.ANALYTICS_QUERY: RetrievalStrategy.SQL,
    }

    def __init__(self, use_adaptive_heuristics: bool = True) -> None:
        """
        Initialize strategy selector.

        Args:
            use_adaptive_heuristics: Whether to refine strategy based on query characteristics
        """
        self.use_adaptive_heuristics = use_adaptive_heuristics
        self._metrics = get_metrics()

    def select_strategy(
        self,
        classification: QueryClassification,
        query: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RetrievalStrategy:
        """
        Select retrieval strategy.

        Strategy selection is based on:
        1. Query classification (primary)
        2. Query characteristics (secondary heuristics)
        3. Historical performance metrics (tertiary)

        Args:
            classification: Query classification result
            query: Original query text (optional, for heuristics)
            metadata: Additional metadata (optional)

        Returns:
            Selected retrieval strategy
        """
        # Get base strategy from classification
        base_strategy = self._MAPPING.get(
            classification.query_type, RetrievalStrategy.VECTOR
        )

        if not self.use_adaptive_heuristics or not query:
            return base_strategy

        # Apply adaptive heuristics
        refined_strategy = self._refine_strategy(
            base_strategy, query, classification
        )
        logger.debug(
            f"Strategy refinement: {base_strategy} → {refined_strategy} "
            f"(confidence: {classification.confidence:.2f})"
        )

        return refined_strategy

    def _refine_strategy(
        self,
        base_strategy: RetrievalStrategy,
        query: str,
        classification: QueryClassification,
    ) -> RetrievalStrategy:
        """
        Refine strategy based on query characteristics.

        Args:
            base_strategy: Strategy from query classification
            query: Query text
            classification: Query classification result

        Returns:
            Refined strategy
        """
        query_lower = query.lower()
        word_count = len(query.split())

        # Relationship detection (questions about "how", "why", "relationship")
        relationship_keywords = ["how", "why", "relationship", "connection", "related"]
        if any(kw in query_lower for kw in relationship_keywords):
            # Upgrade to HYBRID or GRAPH if base is BM25 or VECTOR
            if base_strategy in [RetrievalStrategy.BM25, RetrievalStrategy.VECTOR]:
                return RetrievalStrategy.HYBRID

        # Entity-rich queries (multiple capitalized words)
        entity_count = sum(
            1 for word in query.split() if word and word[0].isupper()
        )
        if entity_count >= 3 and base_strategy == RetrievalStrategy.BM25:
            # Multi-entity queries benefit from hybrid search
            return RetrievalStrategy.HYBRID

        # Length-based refinement
        if word_count < 8 and base_strategy == RetrievalStrategy.VECTOR:
            # Very short queries perform better with BM25
            return RetrievalStrategy.BM25

        if word_count > 20 and base_strategy == RetrievalStrategy.BM25:
            # Long queries better served by VECTOR (semantic)
            return RetrievalStrategy.VECTOR

        # SQL indicators
        sql_keywords = ["select", "where", "count", "sum", "group", "order", "join"]
        if any(kw in query_lower for kw in sql_keywords):
            return RetrievalStrategy.SQL

        # Code indicators
        code_keywords = ["function", "class", "import", "def", "async", "return"]
        if any(kw in query_lower for kw in code_keywords):
            return RetrievalStrategy.HYBRID

        # Low confidence queries → use HYBRID (combines multiple signals)
        if classification.confidence < 0.6:
            return RetrievalStrategy.HYBRID

        return base_strategy

    def get_strategy_performance(self) -> dict[str, Any]:
        """Get performance metrics for each strategy."""
        metrics_summary = self._metrics.get_summary()
        return {
            "strategy_usage": metrics_summary.get("strategy_usage", {}),
            "latency_stats": metrics_summary.get("latency_stats", {}),
            "cache_stats": metrics_summary.get("cache_stats", {}),
        }
