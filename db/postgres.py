from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from core.config import Settings


def build_postgres_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(settings.postgres_dsn, echo=settings.debug, future=True)
