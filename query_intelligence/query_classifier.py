import asyncio
import logging
import math
import re
from typing import Any

from llm.llm_interface import LLMProvider
from query_intelligence.query_types import QueryClassification, QueryType

logger = logging.getLogger(__name__)


class QueryClassifier:
    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        backend: str = "fast",
        use_semantic_classifier: bool = True,
        semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._llm_provider = llm_provider
        self._backend = backend.lower().strip()
        self._use_semantic_classifier = use_semantic_classifier
        self._semantic_model_name = semantic_model_name
        self._semantic_encoder: Any | None = None
        self._semantic_init_attempted: bool = False
        self._prototype_vectors: dict[QueryType, list[float]] | None = None

    async def _ensure_semantic_encoder(self) -> None:
        """Lazy-load the sentence-transformers model in a background thread on first use."""
        if self._semantic_init_attempted or not self._use_semantic_classifier:
            return
        self._semantic_init_attempted = True
        try:
            def _load() -> Any:
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer(self._semantic_model_name)

            self._semantic_encoder = await asyncio.to_thread(_load)
            logger.info("Semantic classifier model loaded: %s", self._semantic_model_name)
        except Exception as exc:
            logger.info("Semantic classifier unavailable, using lexical fast path: %s", exc)
            self._semantic_encoder = None

    async def classify(self, query: str) -> QueryClassification:
        if self._backend == "gemini" and self._llm_provider is not None:
            try:
                return await self._llm_provider.classify_query(query)
            except Exception as exc:
                logger.warning("Gemini classification failed, using fast classifier: %s", exc)
        return await self._fast_classify(query)

    async def _fast_classify(self, query: str) -> QueryClassification:
        lexical = self._lexical_classify(query)
        if self._use_semantic_classifier and self._semantic_encoder is None:
            await self._ensure_semantic_encoder()
        if self._semantic_encoder is None:
            return lexical

        semantic_type, semantic_confidence = await self._semantic_vote(query)
        if semantic_type is None:
            return lexical

        # Use semantic signal when confidence is strong enough, else keep lexical speed/rules.
        if semantic_confidence >= 0.62:
            if lexical.query_type != semantic_type and lexical.confidence < semantic_confidence:
                return QueryClassification(
                    query_type=semantic_type,
                    reasoning=(
                        "Fast semantic classifier selected this type using embedding similarity. "
                        f"Lexical hint was {lexical.query_type.value}."
                    ),
                    confidence=min(0.95, semantic_confidence),
                )
            return QueryClassification(
                query_type=lexical.query_type,
                reasoning=(
                    f"Lexical + semantic fast classifiers agree. {lexical.reasoning}"
                ),
                confidence=min(0.95, max(lexical.confidence, semantic_confidence)),
            )
        return lexical

    def _lexical_classify(self, query: str) -> QueryClassification:
        lowered = query.lower().strip()
        token_count = len(lowered.split())

        if any(token in lowered for token in ["compare", "difference", "vs", "versus"]):
            return QueryClassification(
                query_type=QueryType.MULTI_HOP_REASONING,
                reasoning="Comparison intent detected from lexical patterns.",
                confidence=0.83,
            )

        if any(
            token in lowered
            for token in [
                "error",
                "stack trace",
                "exception",
                "fix",
                "debug",
                "traceback",
                "compile",
            ]
        ):
            return QueryClassification(
                query_type=QueryType.CODE_SEARCH,
                reasoning="Code debugging lexical terms detected.",
                confidence=0.86,
            )

        if any(
            token in lowered
            for token in ["count", "sum", "average", "report", "sql", "dashboard", "metric"]
        ) or re.search(r"\b(select|group by|where|join)\b", lowered):
            return QueryClassification(
                query_type=QueryType.ANALYTICS_QUERY,
                reasoning="Analytical/SQL lexical indicators detected.",
                confidence=0.86,
            )

        if any(
            lowered.startswith(prefix)
            for prefix in ["what is", "who is", "define", "when is", "where is"]
        ):
            return QueryClassification(
                query_type=QueryType.FACT_LOOKUP,
                reasoning="Direct fact lookup phrase detected.",
                confidence=0.87,
            )

        if token_count >= 12 and any(
            token in lowered for token in ["relationship", "dependencies", "across", "between"]
        ):
            return QueryClassification(
                query_type=QueryType.MULTI_HOP_REASONING,
                reasoning="Multi-hop relationship query pattern detected.",
                confidence=0.76,
            )

        return QueryClassification(
            query_type=QueryType.CONCEPTUAL,
            reasoning="Default fast conceptual classification.",
            confidence=0.7,
        )

    async def _semantic_vote(self, query: str) -> tuple[QueryType | None, float]:
        if self._semantic_encoder is None:
            return None, 0.0

        if self._prototype_vectors is None:
            self._prototype_vectors = await asyncio.to_thread(self._build_prototype_vectors)

        query_vector = await asyncio.to_thread(self._encode, query)
        if not query_vector:
            return None, 0.0

        best_type: QueryType | None = None
        best_similarity = -1.0
        for query_type, prototype in self._prototype_vectors.items():
            similarity = self._cosine_similarity(query_vector, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_type = query_type

        # Convert cosine similarity in [-1, 1] to rough confidence in [0, 1].
        confidence = max(0.0, min(1.0, (best_similarity + 1.0) / 2.0))
        return best_type, confidence

    def _build_prototype_vectors(self) -> dict[QueryType, list[float]]:
        examples = {
            QueryType.FACT_LOOKUP: [
                "What is AWS GuardDuty",
                "Define zero trust architecture",
                "Who created Kubernetes",
            ],
            QueryType.CONCEPTUAL: [
                "Explain cloud security posture management",
                "How does retrieval augmented generation work",
                "Concept of defense in depth",
            ],
            QueryType.MULTI_HOP_REASONING: [
                "Compare GuardDuty and Security Hub capabilities",
                "How are AWS services related for threat detection",
                "Relationships between IAM roles and policies",
            ],
            QueryType.CODE_SEARCH: [
                "Exact error message fix in FastAPI",
                "Traceback import error module not found",
                "How to debug async SQLAlchemy timeout",
            ],
            QueryType.ANALYTICS_QUERY: [
                "Count incidents by severity last 30 days",
                "SQL query for average response time",
                "Generate monthly security metrics report",
            ],
        }
        vectors: dict[QueryType, list[float]] = {}
        for query_type, texts in examples.items():
            encoded = self._encode_batch(texts)
            if not encoded:
                continue
            vectors[query_type] = self._mean_vector(encoded)
        return vectors

    def _encode(self, text: str) -> list[float]:
        if self._semantic_encoder is None:
            return []
        vector = self._semantic_encoder.encode(text, normalize_embeddings=True)
        return [float(v) for v in vector]

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        if self._semantic_encoder is None:
            return []
        vectors = self._semantic_encoder.encode(texts, normalize_embeddings=True)
        return [[float(v) for v in item] for item in vectors]

    def _mean_vector(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        size = len(vectors[0])
        out = [0.0] * size
        for vector in vectors:
            for idx, value in enumerate(vector):
                out[idx] += value
        count = float(len(vectors))
        return [value / count for value in out]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return -1.0
        return dot / (mag_a * mag_b)
