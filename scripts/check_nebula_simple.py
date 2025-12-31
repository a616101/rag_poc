#!/usr/bin/env python3
"""
Simple check of NebulaGraph data using basic queries.
"""

import asyncio
import sys
sys.path.insert(0, "src")

async def main():
    from chatbot_graphrag.services.graph.nebula_client import nebula_client

    print("Connecting to NebulaGraph...")
    try:
        await nebula_client.initialize()
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # 1. Get graph statistics
    print("\n=== Graph Statistics ===")
    try:
        # Try submitting a stats job first
        await nebula_client.execute("SUBMIT JOB STATS;")
        await asyncio.sleep(2)
        stats = await nebula_client.execute_json("SHOW STATS;")
        print(f"Stats: {stats}")
    except Exception as e:
        print(f"  Stats error: {e}")

    # 2. Get all tags (vertex types)
    print("\n=== Tags (Vertex Types) ===")
    try:
        result = await nebula_client.execute("SHOW TAGS;")
        print(f"Tags result: {result}")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. Get all edge types
    print("\n=== Edge Types ===")
    try:
        result = await nebula_client.execute("SHOW EDGES;")
        print(f"Edges result: {result}")
    except Exception as e:
        print(f"  Error: {e}")

    # 4. Get all indexes
    print("\n=== Indexes ===")
    try:
        result = await nebula_client.execute("SHOW TAG INDEXES;")
        print(f"Tag indexes: {result}")
    except Exception as e:
        print(f"  Error: {e}")

    # 5. Try to match any entity vertex
    print("\n=== Sample Entity Vertices ===")
    try:
        result = await nebula_client.execute_json("MATCH (n:entity) RETURN n LIMIT 10;")
        print(f"Found {len(result)} entity vertices")
        for r in result[:5]:
            node = r.get("n", {})
            if isinstance(node, dict):
                props = node.get("properties", {}).get("entity", {})
                print(f"  - {props.get('name', 'N/A')} ({props.get('entity_type', 'N/A')})")
    except Exception as e:
        print(f"  Error: {e}")

    # 6. Try to match any chunk vertex
    print("\n=== Sample Chunk Vertices ===")
    try:
        result = await nebula_client.execute_json("MATCH (c:chunk) RETURN c LIMIT 10;")
        print(f"Found {len(result)} chunk vertices")
        for r in result[:5]:
            node = r.get("c", {})
            if isinstance(node, dict):
                props = node.get("properties", {}).get("chunk", {})
                print(f"  - doc_id: {props.get('doc_id', 'N/A')}, type: {props.get('chunk_type', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

    # 7. Try to match any community vertex
    print("\n=== Sample Community Vertices ===")
    try:
        result = await nebula_client.execute_json("MATCH (c:community) RETURN c LIMIT 10;")
        print(f"Found {len(result)} community vertices")
    except Exception as e:
        print(f"  Error: {e}")

    # 8. Try to get any edges
    print("\n=== Sample Edges ===")
    try:
        result = await nebula_client.execute_json("MATCH ()-[e]->() RETURN e LIMIT 10;")
        print(f"Found {len(result)} edges")
        for r in result[:5]:
            edge = r.get("e", {})
            if isinstance(edge, dict):
                print(f"  - {edge.get('src', 'N/A')[:30]}... --[{edge.get('type', 'N/A')}]--> {edge.get('dst', 'N/A')[:30]}...")
    except Exception as e:
        print(f"  Error: {e}")

    # 9. Search for entities containing "吳" or "醫師"
    print("\n=== Searching for entities with '吳' ===")
    try:
        # Use MATCH with CONTAINS for substring search
        result = await nebula_client.execute_json(
            "MATCH (n:entity) WHERE n.entity.name CONTAINS '吳' RETURN n LIMIT 20;"
        )
        print(f"Found {len(result)} entities containing '吳'")
        for r in result[:10]:
            node = r.get("n", {})
            if isinstance(node, dict):
                props = node.get("properties", {}).get("entity", {})
                print(f"  - {props.get('name', 'N/A')} ({props.get('entity_type', 'N/A')})")
    except Exception as e:
        print(f"  Error: {e}")

    # 10. Count entities by type
    print("\n=== Entity Counts by Type ===")
    try:
        result = await nebula_client.execute_json(
            "MATCH (n:entity) RETURN n.entity.entity_type AS type, count(*) AS cnt ORDER BY cnt DESC;"
        )
        for r in result:
            print(f"  - {r.get('type', 'N/A')}: {r.get('cnt', 0)}")
    except Exception as e:
        print(f"  Error: {e}")

    await nebula_client.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
