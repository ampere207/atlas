from datetime import datetime

from pydantic import BaseModel


class IngestResponse(BaseModel):
    status: str
    message: str
    document_id: str
    filename: str
    source: str
    chunks_created: int
    characters: int
    timestamp: datetime
