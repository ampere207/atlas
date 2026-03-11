from datetime import datetime

from fastapi import FastAPI

from api.routes.ingestion_routes import router as ingestion_router
from api.routes.query_routes import router as query_router
from api.schemas.health_schema import HealthResponse
from core.config import get_settings
from core.logging import configure_logging

settings = get_settings()
configure_logging("DEBUG" if settings.debug else "INFO")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Atlas - Distributed Intelligent RAG Engine backend",
)

app.include_router(query_router)
app.include_router(ingestion_router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="atlas",
        version=settings.app_version,
        timestamp=datetime.utcnow(),
    )
