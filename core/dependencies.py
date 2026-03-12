from functools import lru_cache

from cache.semantic_cache import SemanticCache
from context.context_optimizer import ContextOptimizer
from core.config import Settings, get_settings
from db.redis_client import build_redis_client
from embeddings.embedding_model import EmbeddingModel
from ingestion.chunking import ChunkingService
from ingestion.document_parser import DocumentParser
from ingestion.embedding_pipeline import EmbeddingPipeline
from ingestion.ingestion_service import IngestionService
from llm.gemini_client import GeminiClient
from query_intelligence.query_classifier import QueryClassifier
from query_intelligence.strategy_selector import StrategySelector
from ranking.result_aggregator import ResultAggregator
from ranking.reranker import Reranker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.graph_retriever import GraphRetriever
from retrieval.router import RetrievalRouter
from retrieval.sql_retriever import SQLRetriever
from retrieval.vector_retriever import VectorRetriever


@lru_cache(maxsize=1)
def get_llm_provider() -> GeminiClient:
    settings = get_settings()
    return GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    settings = get_settings()
    return EmbeddingModel(model_name=settings.embedding_model_name)


@lru_cache(maxsize=1)
def get_query_classifier() -> QueryClassifier:
    settings = get_settings()
    return QueryClassifier(
        llm_provider=get_llm_provider(),
        backend=settings.query_classifier_backend,
        use_semantic_classifier=settings.query_classifier_use_semantic,
        semantic_model_name=settings.query_classifier_model,
    )


@lru_cache(maxsize=1)
def get_strategy_selector() -> StrategySelector:
    return StrategySelector()


@lru_cache(maxsize=1)
def get_vector_retriever() -> VectorRetriever:
    settings = get_settings()
    return VectorRetriever(
        embedding_model=get_embedding_model(),
        qdrant_url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
    )


@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    settings = get_settings()
    return BM25Retriever(
        elasticsearch_url=settings.elasticsearch_url,
        index_name=settings.elasticsearch_index_name,
    )


@lru_cache(maxsize=1)
def get_retrieval_router() -> RetrievalRouter:
    return RetrievalRouter(
        vector_retriever=get_vector_retriever(),
        bm25_retriever=get_bm25_retriever(),
        graph_retriever=GraphRetriever(),
        sql_retriever=SQLRetriever(),
    )


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    settings = get_settings()
    return Reranker(model_name=settings.reranker_model_name)


@lru_cache(maxsize=1)
def get_result_aggregator() -> ResultAggregator:
    return ResultAggregator()


@lru_cache(maxsize=1)
def get_context_optimizer() -> ContextOptimizer:
    settings = get_settings()
    return ContextOptimizer(max_tokens=settings.context_max_tokens)


@lru_cache(maxsize=1)
def get_cache() -> SemanticCache:
    settings = get_settings()
    redis_client = build_redis_client(settings)
    return SemanticCache(
        redis_client=redis_client,
        embedding_model=get_embedding_model(),
        ttl_seconds=settings.cache_ttl_seconds,
        similarity_threshold=settings.cache_similarity_threshold,
        max_entries=settings.cache_max_entries,
    )


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    settings = get_settings()

    return IngestionService(
        parser=DocumentParser(),
        chunker=ChunkingService(
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        ),
        embedding_pipeline=EmbeddingPipeline(embedding_model=get_embedding_model()),
        vector_retriever=get_vector_retriever(),
        bm25_retriever=get_bm25_retriever(),
    )


def get_app_settings() -> Settings:
    return get_settings()
