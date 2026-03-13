import logging
from typing import Any

from db.neo4j_client import Neo4jClient
from embeddings.embedding_model import EmbeddingModel
from retrieval.base_retriever import BaseRetriever, RetrievedDocument

logger = logging.getLogger(__name__)


class GraphRetriever(BaseRetriever):
    """Retrieves documents via knowledge graph traversal."""

    def __init__(
        self,
        neo4j_client: Neo4jClient | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.neo4j_client = neo4j_client
        self.embedding_model = embedding_model
        self._fallback_docs: dict[str, RetrievedDocument] = {}

    async def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """
        Search for related documents via graph traversal.

        Strategy:
        1. Extract entities from query (via embedding similarity or simple matching)
        2. Search Neo4j for nodes matching entities
        3. Traverse relationships to find connected documents
        4. Rank by connection depth and relationship strength
        """
        if not self.neo4j_client:
            logger.warning("Neo4j client not initialized, returning empty results")
            return []

        try:
            # Extract potential entity names from query (simple approach)
            entities = await self._extract_query_entities(query)
            logger.debug(f"Extracted query entities: {entities}")

            # Search for related nodes
            related_docs: dict[str, RetrievedDocument] = {}

            for entity in entities:
                try:
                    # Search for nodes by name
                    nodes = await self.neo4j_client.search_nodes(
                        label="Service", property_name="name", property_value=entity
                    )
                    if not nodes:
                        nodes = await self.neo4j_client.search_nodes(
                            label="Technology",
                            property_name="name",
                            property_value=entity,
                        )
                    if not nodes:
                        nodes = await self.neo4j_client.search_nodes(
                            label="Concept",
                            property_name="name",
                            property_value=entity,
                        )

                    # Get connected nodes
                    for node in nodes:
                        connected = await self.neo4j_client.get_connected_nodes(
                            start_label="Service",
                            start_property="name",
                            start_value=entity,
                            max_depth=2,
                        )
                        if not connected:
                            connected = (
                                await self.neo4j_client.get_connected_nodes(
                                    start_label="Technology",
                                    start_property="name",
                                    start_value=entity,
                                    max_depth=2,
                                )
                            )
                        if not connected:
                            connected = (
                                await self.neo4j_client.get_connected_nodes(
                                    start_label="Concept",
                                    start_property="name",
                                    start_value=entity,
                                    max_depth=2,
                                )
                            )

                        # Extract chunk_id and document_id from connected nodes
                        for conn_node in connected:
                            chunk_id = conn_node.get("chunk_id", "")
                            document_id = conn_node.get("document_id", "")
                            if chunk_id and document_id:
                                key = f"{document_id}:{chunk_id}"
                                if key not in related_docs:
                                    related_docs[key] = RetrievedDocument(
                                        document_id=document_id,
                                        content=conn_node.get(
                                            "description",
                                            f"Related via {entity}",
                                        ),
                                        source="graph",
                                        score=0.8,  # Base graph match score
                                        metadata={
                                            "chunk_id": chunk_id,
                                            "entity": entity,
                                        },
                                    )
                except Exception as e:
                    logger.debug(f"Error searching for entity {entity}: {e}")

            # Return top_k results
            results = list(related_docs.values())[:top_k]
            logger.debug(f"Graph search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []

    async def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Index chunks in graph (no-op, done by GraphBuilder)."""
        pass

    async def _extract_query_entities(self, query: str) -> list[str]:
        """
        Extract potential entity names from query.

        Simple heuristic: Use embedding similarity against known entity patterns.
        For now, return capitalized words as potential entities.
        """
        # Simple approach: extract capitalized phrases
        words = query.split()
        entities = []

        i = 0
        while i < len(words):
            word = words[i]
            # Check if word starts with capital letter
            if word and word[0].isupper():
                # Collect consecutive capitalized words
                phrase = word
                j = i + 1
                while j < len(words) and words[j] and words[j][0].isupper():
                    phrase += " " + words[j]
                    j += 1
                entities.append(phrase)
                i = j
            else:
                i += 1

        # Deduplicate
        return list(set(entities))
