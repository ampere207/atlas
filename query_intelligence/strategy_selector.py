from query_intelligence.query_types import (
    QueryClassification,
    QueryType,
    RetrievalStrategy,
)


class StrategySelector:
    _MAPPING: dict[QueryType, RetrievalStrategy] = {
        QueryType.FACT_LOOKUP: RetrievalStrategy.BM25,
        QueryType.CONCEPTUAL: RetrievalStrategy.VECTOR,
        QueryType.MULTI_HOP_REASONING: RetrievalStrategy.HYBRID,
        QueryType.CODE_SEARCH: RetrievalStrategy.HYBRID,
        QueryType.ANALYTICS_QUERY: RetrievalStrategy.SQL,
    }

    def select_strategy(
        self,
        classification: QueryClassification,
    ) -> RetrievalStrategy:
        return self._MAPPING.get(classification.query_type, RetrievalStrategy.VECTOR)
