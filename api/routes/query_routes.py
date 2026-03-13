import logging
import time
from typing import Any

from fastapi import APIRouter, Depends

from api.schemas.query_schema import QueryRequest, QueryResponse, RetrievedDocOut
from cache.semantic_cache import SemanticCache
from context.context_optimizer import ContextOptimizer
from core.dependencies import (
    get_cache,
    get_context_optimizer,
    get_hybrid_ranker,
    get_llm_provider,
    get_query_classifier,
    get_result_aggregator,
    get_reranker,
    get_retrieval_metrics,
    get_retrieval_router,
    get_strategy_selector,
)
from llm.llm_interface import LLMProvider
from metrics.retrieval_metrics import RetrievalMetrics
from query_intelligence.query_classifier import QueryClassifier
from query_intelligence.strategy_selector import StrategySelector
from ranking.result_aggregator import ResultAggregator
from ranking.reranker import Reranker
from retrieval.hybrid_ranker import HybridRanker
from retrieval.router import RetrievalRouter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    classifier: QueryClassifier = Depends(get_query_classifier),
    selector: StrategySelector = Depends(get_strategy_selector),
    retrieval_router: RetrievalRouter = Depends(get_retrieval_router),
    result_aggregator: ResultAggregator = Depends(get_result_aggregator),
    hybrid_ranker: HybridRanker = Depends(get_hybrid_ranker),
    reranker: Reranker = Depends(get_reranker),
    context_optimizer: ContextOptimizer = Depends(get_context_optimizer),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    semantic_cache: SemanticCache = Depends(get_cache),
    metrics: RetrievalMetrics = Depends(get_retrieval_metrics),
) -> QueryResponse:
    """
    Process a query through the full intelligent RAG pipeline.

    Pipeline:
    1. Semantic cache lookup
    2. Query classification
    3. Strategy selection (with adaptive heuristics)
    4. Retrieval from multiple systems (vector, BM25, graph)
    5. Hybrid ranking
    6. Cross-encoder reranking
    7. Context optimization with MMR
    8. Gemini response generation
    9. Cache response
    """
    start_time = time.time()
    cache_hit = False
    retrieval_source = "unknown"

    try:
        # Step 1: Semantic cache lookup
        logger.debug(f"Cache lookup for: {payload.query[:50]}...")
        cached = await semantic_cache.get_cached_response(payload.query)
        if cached is not None:
            cache_hit = True
            logger.info("Cache HIT")
            cached_payload = dict(cached)
            cached_payload["query"] = payload.query
            cached_payload["cached"] = True
            
            # Record cache hit metric
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_retrieval(
                query=payload.query,
                strategy="CACHE",
                retriever_source="redis",
                latency_ms=latency_ms,
                documents_returned=len(cached_payload.get("documents", [])),
                cache_hit=True,
            )
            
            return QueryResponse.model_validate(cached_payload)

        # Step 2: Query classification
        logger.debug(f"Classifying query: {payload.query[:50]}...")
        classification = await classifier.classify(payload.query)
        logger.info(f"Classification: {classification.query_type} (confidence: {classification.confidence:.2f})")

        # Step 3: Strategy selection with adaptive heuristics
        strategy = selector.select_strategy(
            classification=classification,
            query=payload.query,
            metadata={"query_type": classification.query_type},
        )
        logger.info(f"Selected strategy: {strategy}")

        # Step 4: Multi-source retrieval
        retrieval_start = time.time()
        logger.debug(f"Retrieving with strategy: {strategy}")
        docs = await retrieval_router.route(
            query=payload.query,
            strategy=strategy,
            top_k=payload.top_k,
        )
        retrieval_latency_ms = (time.time() - retrieval_start) * 1000
        logger.debug(f"Retrieved {len(docs)} documents in {retrieval_latency_ms:.1f}ms")

        # Determine primary retrieval source
        if docs:
            retrieval_source = docs[0].source

        # Step 5: Result aggregation + hybrid ranking
        logger.debug("Aggregating and hybrid ranking...")
        candidates = await result_aggregator.aggregate(
            docs, top_k=max(payload.top_k * 3, payload.top_k)
        )
        hybrid_ranked = hybrid_ranker.rank(candidates, top_k=payload.top_k)
        logger.debug(f"Hybrid ranked: {len(hybrid_ranked)} docs")

        # Step 6: Cross-encoder reranking
        logger.debug("Reranking with cross-encoder...")
        rerank_start = time.time()
        reranked = await reranker.rerank(payload.query, hybrid_ranked)
        rerank_latency_ms = (time.time() - rerank_start) * 1000
        logger.debug(f"Reranking took {rerank_latency_ms:.1f}ms")

        # Step 7: Context optimization with MMR
        logger.debug("Optimizing context...")
        optimized = await context_optimizer.optimize(
            reranked[: payload.top_k],
            metadata={"query": payload.query},
        )
        logger.debug(f"Context optimized to {len(optimized)} docs")

        # Step 8: LLM response generation
        context_text = "\n\n".join(doc.content for doc in optimized)
        
        # Improved prompt with grounding instructions
        prompt = (
            "You are a knowledgeable assistant. "
            "Answer the question using ONLY the provided context. "
            "If the context does not contain the answer, say you do not know.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{payload.query}"
        )
        
        logger.debug("Generating LLM response...")
        llm_start = time.time()
        answer = await llm_provider.generate_text(prompt)
        llm_latency_ms = (time.time() - llm_start) * 1000
        logger.debug(f"LLM generation took {llm_latency_ms:.1f}ms")

        # Build response
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

        # Step 9: Cache response
        logger.debug("Caching response...")
        await semantic_cache.store_response(
            payload.query, response.model_dump(mode="json")
        )

        # Record metrics
        total_latency_ms = (time.time() - start_time) * 1000
        final_score = float(optimized[0].score) if optimized else 0.0
        metrics.record_retrieval(
            query=payload.query,
            strategy=strategy.value,
            retriever_source=retrieval_source,
            latency_ms=total_latency_ms,
            documents_returned=len(optimized),
            cache_hit=False,
            reranked=True,
            final_score=final_score,
            metadata={
                "classification_confidence": float(classification.confidence),
                "retrieval_ms": retrieval_latency_ms,
                "rerank_ms": rerank_latency_ms,
                "llm_ms": llm_latency_ms,
            },
        )

        logger.info(
            f"Query processed in {total_latency_ms:.1f}ms "
            f"(retrieval: {retrieval_latency_ms:.1f}ms, "
            f"rerank: {rerank_latency_ms:.1f}ms, "
            f"llm: {llm_latency_ms:.1f}ms)"
        )

        return response

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        
        # Record error metric
        total_latency_ms = (time.time() - start_time) * 1000
        metrics.record_retrieval(
            query=payload.query,
            strategy="ERROR",
            retriever_source="error",
            latency_ms=total_latency_ms,
            documents_returned=0,
            cache_hit=False,
            metadata={"error": str(e)},
        )
        raise


@router.get("/metrics")
async def get_metrics_summary(
    metrics: RetrievalMetrics = Depends(get_retrieval_metrics),
) -> dict[str, Any]:
    """Get comprehensive retrieval performance metrics."""
    return metrics.get_summary()
