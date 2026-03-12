from pydantic import BaseModel, Field

from query_intelligence.query_types import QueryClassification, RetrievalStrategy


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=20)


class RetrievedDocOut(BaseModel):
    document_id: str
    content: str
    source: str
    score: float


class QueryResponse(BaseModel):
    query: str
    classification: QueryClassification
    strategy: RetrievalStrategy
    cached: bool
    documents: list[RetrievedDocOut]
    answer: str
