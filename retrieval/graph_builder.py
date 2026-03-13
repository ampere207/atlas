import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from db.neo4j_client import Neo4jClient
from llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Entity:
    name: str
    label: str
    description: str


@dataclass(slots=True)
class Relationship:
    from_entity: str
    from_label: str
    to_entity: str
    to_label: str
    relationship_type: str


class GraphBuilder:
    """Builds knowledge graph from document chunks via LLM entity extraction."""

    def __init__(self, gemini_client: GeminiClient, neo4j_client: Neo4jClient) -> None:
        self.gemini_client = gemini_client
        self.neo4j_client = neo4j_client

    async def extract_entities_and_relationships(
        self, text: str, chunk_id: str
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Extract entities and relationships from text using Gemini.

        Returns: (entities, relationships)
        """
        prompt = f"""Extract entities and relationships from the following text.

Text:
{text}

Return JSON with this exact structure:
{{
    "entities": [
        {{"name": "string", "label": "Service|Technology|Concept|Person|Organization|Other", "description": "string"}}
    ],
    "relationships": [
        {{"from": "entity_name", "from_label": "label", "to": "entity_name", "to_label": "label", "type": "RELATED_TO|PART_OF|INTEGRATES_WITH|OWNS|CREATED_BY|OTHER"}}
    ]
}}

Labels can be: Service, Technology, Concept, Person, Organization, Other
Relationship types: RELATED_TO, PART_OF, INTEGRATES_WITH, OWNS, CREATED_BY, OTHER

Extract only meaningful entities and relationships. Be conservative."""

        try:
            response = await self.gemini_client.generate(prompt)
            # Extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            entities = [
                Entity(
                    name=e["name"],
                    label=e["label"],
                    description=e.get("description", ""),
                )
                for e in data.get("entities", [])
            ]

            relationships = [
                Relationship(
                    from_entity=r["from"],
                    from_label=r["from_label"],
                    to_entity=r["to"],
                    to_label=r["to_label"],
                    relationship_type=r["type"],
                )
                for r in data.get("relationships", [])
            ]

            return entities, relationships
        except Exception as e:
            logger.warning(f"Failed to extract entities from chunk {chunk_id}: {e}")
            return [], []

    async def build_graph(
        self, documents: list[dict[str, Any]], chunk_id_to_doc: dict[str, str]
    ) -> None:
        """
        Build knowledge graph from documents.

        Args:
            documents: List of {document_id, chunks} dicts
            chunk_id_to_doc: Mapping of chunk_id to chunk text
        """
        for doc in documents:
            document_id = doc["document_id"]
            chunks = doc.get("chunks", [])

            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", "")
                text = chunk.get("text", "")

                if not text:
                    continue

                # Extract entities and relationships
                entities, relationships = (
                    await self.extract_entities_and_relationships(text, chunk_id)
                )

                # Create nodes in Neo4j
                for entity in entities:
                    try:
                        await self.neo4j_client.create_node(
                            entity.label,
                            {
                                "name": entity.name,
                                "description": entity.description,
                                "chunk_id": chunk_id,
                                "document_id": document_id,
                            },
                        )
                    except Exception as e:
                        logger.debug(
                            f"Failed to create node for entity {entity.name}: {e}"
                        )

                # Create relationships
                for rel in relationships:
                    try:
                        await self.neo4j_client.create_relationship(
                            from_label=rel.from_label,
                            from_prop="name",
                            from_value=rel.from_entity,
                            rel_type=rel.relationship_type,
                            to_label=rel.to_label,
                            to_prop="name",
                            to_value=rel.to_entity,
                        )
                    except Exception as e:
                        logger.debug(
                            f"Failed to create relationship {rel.from_entity} -> {rel.to_entity}: {e}"
                        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text."""
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return "{}"
