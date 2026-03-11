from enum import Enum

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    FACT_LOOKUP = "FACT_LOOKUP"
    CONCEPTUAL = "CONCEPTUAL"
    MULTI_HOP_REASONING = "MULTI_HOP_REASONING"
    CODE_SEARCH = "CODE_SEARCH"
    ANALYTICS_QUERY = "ANALYTICS_QUERY"


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    GRAPH = "graph"
    SQL = "sql"
    HYBRID = "hybrid"


class QueryClassification(BaseModel):
    query_type: QueryType
    reasoning: str = Field(default="")
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
