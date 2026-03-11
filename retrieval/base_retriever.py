from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedDocument:
    document_id: str
    content: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        raise NotImplementedError
