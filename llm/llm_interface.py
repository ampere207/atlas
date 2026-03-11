from abc import ABC, abstractmethod

from query_intelligence.query_types import QueryClassification


class LLMProvider(ABC):
    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def classify_query(self, prompt: str) -> QueryClassification:
        raise NotImplementedError
