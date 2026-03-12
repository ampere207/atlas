import uuid
from typing import Any

from db.models import IngestedDocumentMetadata
from ingestion.chunking import ChunkingService
from ingestion.document_parser import DocumentParser
from ingestion.embedding_pipeline import EmbeddingPipeline
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever


class IngestionService:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: ChunkingService,
        embedding_pipeline: EmbeddingPipeline,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedding_pipeline = embedding_pipeline
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
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
        document_id = str(uuid.uuid4())
        chunks = await self._chunker.chunk(parsed.text, document_id=document_id)
        embeddings = await self._embedding_pipeline.embed_chunks(chunks)

        await self._vector_retriever.index_chunks(
            chunks=chunks,
            embeddings=embeddings,
            metadata={"filename": filename, "source": source, **(extra_metadata or {})},
        )
        await self._bm25_retriever.index_chunks(
            chunks=chunks,
            metadata={"filename": filename, "source": source, **(extra_metadata or {})},
        )

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
            "characters": len(parsed.text),
            "status": "accepted",
        }
