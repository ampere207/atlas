import asyncio
import json
import logging
import re

from llm.llm_interface import LLMProvider
from query_intelligence.query_types import QueryClassification, QueryType

logger = logging.getLogger(__name__)

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


class GeminiClient(LLMProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        self._enabled = bool(api_key and genai)
        self._model_name = model
        self._client = None

        if self._enabled:
            try:
                self._client = genai.Client(api_key=api_key)
            except Exception as exc:
                logger.warning("Failed to initialize Gemini client: %s", exc)
                self._enabled = False
                self._client = None

    async def generate_text(self, prompt: str) -> str:
        if not self._enabled or self._client is None:
            return "Gemini is not configured. Returning placeholder response."

        def _generate() -> str:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[prompt],
)
            try:
                return response.candidates[0].content.parts[0].text
            except Exception:
                return ""

        try:
            generated = await asyncio.to_thread(_generate)
            return generated or "No generated answer available."
        except Exception as exc:
            logger.exception("Gemini text generation failed")
            return "Gemini call failed. Returning placeholder response."

    async def classify_query(self, prompt: str) -> QueryClassification:
        if not self._enabled or self._client is None:
            return QueryClassification(
                query_type=QueryType.CONCEPTUAL,
                reasoning="Gemini not configured, used default classification.",
                confidence=0.4,
            )

        instruction = (
            "Classify the query into one of: FACT_LOOKUP, CONCEPTUAL, "
            "MULTI_HOP_REASONING, CODE_SEARCH, ANALYTICS_QUERY. "
            "Return ONLY JSON with keys: query_type, reasoning, confidence.\n"
            f"Query: {prompt}"
        )

        def _classify() -> str:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=instruction,
            )
            try:
                return response.candidates[0].content.parts[0].text
            except Exception:
                return ""   

        try:
            raw = await asyncio.to_thread(_classify)
            return self._parse(raw)
        except Exception as exc:
            logger.warning("Gemini classification call failed: %s", exc)
            return QueryClassification(
                query_type=QueryType.CONCEPTUAL,
                reasoning="Fallback classification due to Gemini API failure.",
                confidence=0.5,
            )

    def _parse(self, raw: str) -> QueryClassification:
        try:
            # Extract JSON block from Gemini response with better handling
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not match:
                # Try markdown code block format
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON object found in Gemini response")
                cleaned = json_match.group(1)
            else:
                cleaned = match.group()

            logger.debug("Extracted JSON from Gemini response: %s", cleaned)
            payload = json.loads(cleaned)

            query_type = QueryType(payload["query_type"])

            # Handle confidence: numeric or string values
            confidence_raw = payload.get("confidence", 0.7)

            if isinstance(confidence_raw, str):
                confidence_map = {
                    "high": 0.9,
                    "medium": 0.7,
                    "low": 0.5,
                }
                confidence = confidence_map.get(confidence_raw.lower(), 0.7)
            else:
                confidence = float(confidence_raw)

            logger.debug("Parsed classification: type=%s, confidence=%s", query_type, confidence)

            return QueryClassification(
                query_type=query_type,
                reasoning=payload.get("reasoning", ""),
                confidence=confidence,
            )

        except Exception as exc:
            logger.warning(
                "Failed to parse Gemini classification response. Raw output: %s | Error: %s",
                raw[:500],
                exc,
            )

            return QueryClassification(
                query_type=QueryType.CONCEPTUAL,
                reasoning="Fallback classification due to parse error.",
                confidence=0.5,
            )