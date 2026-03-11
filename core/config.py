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

    query_classifier_backend: str = "fast"
    query_classifier_use_semantic: bool = True
    query_classifier_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    postgres_dsn: str = Field(
        default="postgresql+asyncpg://atlas:atlas@localhost:5432/atlas"
    )
    redis_url: str = "redis://localhost:6379/0"

    default_top_k: int = 5
    cache_ttl_seconds: int = 900
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
