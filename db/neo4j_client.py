from typing import Any

from neo4j import AsyncDriver, AsyncSession, GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from core.config import Settings


async def build_neo4j_driver(settings: Settings) -> AsyncDriver:
    """Create and return a Neo4j AsyncDriver."""
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    # Test connection
    try:
        async with driver.session(database="neo4j") as session:
            await session.run("RETURN 1")
    except ServiceUnavailable:
        await driver.close()
        raise
    return driver


async def close_neo4j_driver(driver: AsyncDriver) -> None:
    """Close Neo4j driver connection."""
    await driver.close()


class Neo4jClient:
    """Simple Neo4j client wrapper for graph operations."""

    def __init__(self, driver: AsyncDriver) -> None:
        self.driver = driver

    async def create_node(
        self, label: str, properties: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a node with given label and properties."""
        async with self.driver.session(database="neo4j") as session:
            result = await session.run(
                f"CREATE (n:{label} $props) RETURN n",
                props=properties,
            )
            record = await result.single()
            return record.data() if record else {}

    async def create_relationship(
        self,
        from_label: str,
        from_prop: str,
        from_value: str,
        rel_type: str,
        to_label: str,
        to_prop: str,
        to_value: str,
    ) -> dict[str, Any]:
        """Create a relationship between two nodes."""
        async with self.driver.session(database="neo4j") as session:
            query = f"""
            MATCH (a:{from_label} {{{from_prop}: $from_val}})
            MATCH (b:{to_label} {{{to_prop}: $to_val}})
            CREATE (a)-[r:{rel_type}]->(b)
            RETURN r
            """
            result = await session.run(
                query, from_val=from_value, to_val=to_value
            )
            record = await result.single()
            return record.data() if record else {}

    async def get_connected_nodes(
        self,
        start_label: str,
        start_property: str,
        start_value: str,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Get all nodes connected to a starting node within max_depth hops."""
        async with self.driver.session(database="neo4j") as session:
            query = f"""
            MATCH (start:{start_label} {{{start_property}: $value}})
            MATCH (start)-[*1..{max_depth}]-(connected)
            RETURN DISTINCT connected
            """
            result = await session.run(query, value=start_value)
            records = await result.fetch(-1)
            return [record.data() for record in records]

    async def search_nodes(
        self, label: str, property_name: str, property_value: str
    ) -> list[dict[str, Any]]:
        """Search for nodes by label and property."""
        async with self.driver.session(database="neo4j") as session:
            query = f"MATCH (n:{label} {{{property_name}: $value}}) RETURN n"
            result = await session.run(query, value=property_value)
            records = await result.fetch(-1)
            return [record.data() for record in records]

    async def clear_database(self) -> None:
        """Clear all nodes and relationships (use with caution)."""
        async with self.driver.session(database="neo4j") as session:
            await session.run("MATCH (n) DETACH DELETE n")
