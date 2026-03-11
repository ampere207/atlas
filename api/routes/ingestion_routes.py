from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.schemas.ingestion_schema import IngestResponse
from core.dependencies import get_ingestion_service
from ingestion.ingestion_service import IngestionService

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    source: str = Form(default="manual_upload"),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    file_bytes = await file.read()
    result = await ingestion_service.ingest(
        file_bytes=file_bytes,
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        source=source,
    )
    return IngestResponse(
        status="success",
        message="Document metadata stored. Full ingestion pipeline is pending Phase 2.",
        document_id=result["document_id"],
        filename=result["filename"],
        source=result["source"],
        timestamp=datetime.utcnow(),
    )
