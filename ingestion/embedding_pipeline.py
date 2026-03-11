from ingestion.chunking import Chunk


class EmbeddingPipeline:
    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        # Placeholder embeddings keep interfaces stable until real model integration.
        return [[0.0, 0.1, 0.2] for _ in chunks]
