from ingestion.chunking import Chunk
from embeddings.embedding_model import EmbeddingModel


class EmbeddingPipeline:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._embedding_model = embedding_model

    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        texts = [chunk.text for chunk in chunks]
        return await self._embedding_model.generate_embeddings(texts)
