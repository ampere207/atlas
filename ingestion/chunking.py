from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    index: int
    text: str


class ChunkingService:
    def __init__(self, chunk_size: int = 500) -> None:
        self._chunk_size = chunk_size

    async def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        chunks: list[Chunk] = []
        for idx, start in enumerate(range(0, len(text), self._chunk_size)):
            chunks.append(Chunk(index=idx, text=text[start : start + self._chunk_size]))
        return chunks
