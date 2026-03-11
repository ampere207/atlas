from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class IngestedDocumentMetadata:
    document_id: str
    filename: str
    content_type: str
    source: str
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
