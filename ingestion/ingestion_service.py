import uuid
from typing import Any

from db.models import IngestedDocumentMetadata
from ingestion.chunking import ChunkingService
from ingestion.document_parser import DocumentParser
from ingestion.embedding_pipeline import EmbeddingPipeline


class IngestionService:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: ChunkingService,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedding_pipeline = embedding_pipeline
        self._metadata_store: dict[str, IngestedDocumentMetadata] = {}

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parsed = await self._parser.parse(file_bytes, filename, content_type)
        chunks = await self._chunker.chunk(parsed.text)
        _embeddings = await self._embedding_pipeline.embed_chunks(chunks)

        document_id = str(uuid.uuid4())
        metadata = IngestedDocumentMetadata(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            source=source,
            extra_metadata=extra_metadata or {},
        )
        self._metadata_store[document_id] = metadata

        return {
            "document_id": document_id,
            "filename": filename,
            "content_type": content_type,
            "source": source,
            "chunks_created": len(chunks),
            "status": "accepted",
        }
