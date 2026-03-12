from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Atlas - Distributed Intelligent RAG Engine"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    query_classifier_backend: str = "gemini"
    query_classifier_use_semantic: bool = True
    query_classifier_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    embedding_model_name: str = "BAAI/bge-small-en"
    reranker_model_name: str = "BAAI/bge-reranker-base"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "documents"

    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index_name: str = "documents_index"

    postgres_dsn: str = Field(
        default="postgresql+asyncpg://atlas:atlas@localhost:5432/atlas"
    )
    redis_url: str = "redis://localhost:6379/0"

    default_top_k: int = 5
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    context_max_tokens: int = 1800
    cache_ttl_seconds: int = 900
    cache_similarity_threshold: float = 0.9
    cache_max_entries: int = 500
    use_placeholder_retrieval: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
