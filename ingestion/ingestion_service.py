import asyncio
import logging
import uuid
from typing import Any

from db.models import IngestedDocumentMetadata
from ingestion.chunking import ChunkingService
from ingestion.document_parser import DocumentParser
from ingestion.embedding_pipeline import EmbeddingPipeline
from retrieval.bm25_retriever import BM25Retriever
from retrieval.graph_builder import GraphBuilder
from retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Orchestrates async document ingestion pipeline.

    Pipeline:
    1. Parse document
    2. Chunk content
    3. Embed chunks
    4. Index in vector store (Qdrant)
    5. Index in BM25 store (Elasticsearch)
    6. Build knowledge graph (Neo4j) - optional
    """

    def __init__(
        self,
        parser: DocumentParser,
        chunker: ChunkingService,
        embedding_pipeline: EmbeddingPipeline,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        graph_builder: GraphBuilder | None = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedding_pipeline = embedding_pipeline
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._graph_builder = graph_builder
        self._metadata_store: dict[str, IngestedDocumentMetadata] = {}

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        source: str,
        extra_metadata: dict[str, Any] | None = None,
        build_graph: bool = False,
    ) -> dict[str, Any]:
        """
        Ingest a document end-to-end.

        Args:
            file_bytes: Raw file content
            filename: Original filename
            content_type: MIME type
            source: Data source label
            extra_metadata: Additional metadata
            build_graph: Whether to build knowledge graph

        Returns:
            Ingestion result summary
        """
        try:
            # Step 1: Parse
            logger.info(f"Parsing {filename}...")
            parsed = await self._parser.parse(file_bytes, filename, content_type)

            # Step 2: Chunk
            logger.info(f"Chunking {filename}...")
            document_id = str(uuid.uuid4())
            chunks = await self._chunker.chunk(parsed.text, document_id=document_id)
            logger.debug(f"Created {len(chunks)} chunks")

            # Step 3: Embed
            logger.info(f"Embedding {len(chunks)} chunks...")
            embeddings = await self._embedding_pipeline.embed_chunks(chunks)

            # Step 4: Index (parallel where possible)
            logger.info("Indexing in vector and BM25 stores...")
            await asyncio.gather(
                self._vector_retriever.index_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata={
                        "filename": filename,
                        "source": source,
                        **(extra_metadata or {}),
                    },
                ),
                self._bm25_retriever.index_chunks(
                    chunks=chunks,
                    metadata={
                        "filename": filename,
                        "source": source,
                        **(extra_metadata or {}),
                    },
                ),
            )

            # Step 5: Build graph (optional)
            if build_graph and self._graph_builder:
                logger.info("Building knowledge graph...")
                try:
                    await self._graph_builder.build_graph(
                        documents=[
                            {
                                "document_id": document_id,
                                "chunks": [
                                    {
                                        "chunk_id": c.chunk_id,
                                        "text": c.text,
                                    }
                                    for c in chunks
                                ],
                            }
                        ],
                        chunk_id_to_doc={c.chunk_id: c.text for c in chunks},
                    )
                except Exception as e:
                    logger.warning(f"Graph building failed: {e}")

            # Store metadata
            metadata = IngestedDocumentMetadata(
                document_id=document_id,
                filename=filename,
                content_type=content_type,
                source=source,
                extra_metadata=extra_metadata or {},
            )
            self._metadata_store[document_id] = metadata

            logger.info(f"Successfully ingested {filename} ({len(parsed.text)} chars)")

            return {
                "document_id": document_id,
                "filename": filename,
                "content_type": content_type,
                "source": source,
                "chunks_created": len(chunks),
                "characters": len(parsed.text),
                "status": "accepted",
            }

        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {e}", exc_info=True)
            raise
