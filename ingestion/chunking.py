from dataclasses import dataclass
from uuid import uuid4


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str


class ChunkingService:
    def __init__(self, chunk_size_tokens: int = 500, overlap_tokens: int = 100) -> None:
        self._chunk_size_tokens = chunk_size_tokens
        self._overlap_tokens = overlap_tokens

    async def chunk(self, text: str, document_id: str) -> list[Chunk]:
        if not text:
            return []

        tokens = text.split()
        if not tokens:
            return []

        step = max(1, self._chunk_size_tokens - self._overlap_tokens)
        chunks: list[Chunk] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self._chunk_size_tokens]
            if not window:
                continue
            chunk_text = " ".join(window).strip()
            if not chunk_text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:{uuid4()}",
                    document_id=document_id,
                    text=chunk_text,
                )
            )

            if start + self._chunk_size_tokens >= len(tokens):
                break
        return chunks
