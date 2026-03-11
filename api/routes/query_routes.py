from fastapi import APIRouter, Depends

from api.schemas.query_schema import QueryRequest, QueryResponse, RetrievedDocOut
from cache.semantic_cache import SemanticCache
from context.context_optimizer import ContextOptimizer
from core.dependencies import (
    get_cache,
    get_context_optimizer,
    get_llm_provider,
    get_query_classifier,
    get_reranker,
    get_retrieval_router,
    get_strategy_selector,
)
from llm.llm_interface import LLMProvider
from query_intelligence.query_classifier import QueryClassifier
from query_intelligence.strategy_selector import StrategySelector
from ranking.reranker import Reranker
from retrieval.router import RetrievalRouter

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    classifier: QueryClassifier = Depends(get_query_classifier),
    selector: StrategySelector = Depends(get_strategy_selector),
    retrieval_router: RetrievalRouter = Depends(get_retrieval_router),
    reranker: Reranker = Depends(get_reranker),
    context_optimizer: ContextOptimizer = Depends(get_context_optimizer),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    semantic_cache: SemanticCache = Depends(get_cache),
) -> QueryResponse:
    cached = await semantic_cache.get_cached_response(payload.query)
    if cached is not None:
        return QueryResponse.model_validate(cached)

    classification = await classifier.classify(payload.query)
    strategy = selector.select_strategy(classification)

    docs = await retrieval_router.route(
        query=payload.query,
        strategy=strategy,
        top_k=payload.top_k,
    )
    ranked = await reranker.rerank(payload.query, docs)
    optimized = await context_optimizer.optimize(ranked)

    context_text = "\n\n".join(doc.content for doc in optimized)
    prompt = (
        "You are Atlas.\n\n"
        "Answer the question using ONLY the context below.\n\n"
        f"Query:\n{payload.query}\n\n"
        f"Context:\n{context_text}"
    )
    answer = await llm_provider.generate_text(prompt)

    response = QueryResponse(
        query=payload.query,
        classification=classification,
        strategy=strategy,
        cached=False,
        documents=[
            RetrievedDocOut(
                document_id=doc.document_id,
                content=doc.content,
                source=doc.source,
                score=doc.score,
            )
            for doc in optimized
        ],
        answer=answer,
    )
    await semantic_cache.store_response(payload.query, response.model_dump(mode="json"))
    return response
