#!/usr/bin/env python3
"""
Check NebulaGraph for physician data specifically.
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

    # 1. Search for entities containing "明昇"
    print("\n=== Searching for entities with '明昇' ===")
    try:
        result = await nebula_client.execute_json(
            "MATCH (n:entity) WHERE n.entity.name CONTAINS '明昇' RETURN n LIMIT 20;"
        )
        print(f"Found {len(result)} entities containing '明昇'")
        for r in result:
            node = r.get("n", {})
            if isinstance(node, dict):
                props = node.get("properties", {}).get("entity", {})
                print(f"  - {props.get('name', 'N/A')} ({props.get('entity_type', 'N/A')})")
                print(f"    Description: {props.get('description', 'N/A')[:100]}")
                print(f"    VID: {node.get('vid', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

    # 2. List all "person" type entities
    print("\n=== All Person Entities ===")
    try:
        result = await nebula_client.execute_json(
            "MATCH (n:entity) WHERE n.entity.entity_type == 'person' RETURN n LIMIT 50;"
        )
        print(f"Found {len(result)} person entities (showing first 20):")
        for r in result[:20]:
            node = r.get("n", {})
            if isinstance(node, dict):
                props = node.get("properties", {}).get("entity", {})
                print(f"  - {props.get('name', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. Check if there are any physician-related entities
    print("\n=== Searching for '醫師' entities ===")
    try:
        result = await nebula_client.execute_json(
            "MATCH (n:entity) WHERE n.entity.name CONTAINS '醫師' RETURN n LIMIT 20;"
        )
        print(f"Found {len(result)} entities containing '醫師'")
        for r in result:
            node = r.get("n", {})
            if isinstance(node, dict):
                props = node.get("properties", {}).get("entity", {})
                print(f"  - {props.get('name', 'N/A')} ({props.get('entity_type', 'N/A')})")
    except Exception as e:
        print(f"  Error: {e}")

    # 4. Search for cardiology related entities
    print("\n=== Searching for '心臟' entities ===")
    try:
        result = await nebula_client.execute_json(
            "MATCH (n:entity) WHERE n.entity.name CONTAINS '心臟' RETURN n LIMIT 20;"
        )
        print(f"Found {len(result)} entities containing '心臟'")
        for r in result:
            node = r.get("n", {})
            if isinstance(node, dict):
                props = node.get("properties", {}).get("entity", {})
                print(f"  - {props.get('name', 'N/A')} ({props.get('entity_type', 'N/A')})")
    except Exception as e:
        print(f"  Error: {e}")

    # 5. Count edges by type
    print("\n=== Edge Counts ===")
    edge_types = ["belongs_to", "connects_to", "located_at", "member_of",
                  "mentions", "part_of", "performs", "related_to", "requires",
                  "treats", "works_in"]
    for edge_type in edge_types:
        try:
            result = await nebula_client.execute_json(
                f"MATCH ()-[e:{edge_type}]->() RETURN count(e) AS cnt;"
            )
            cnt = result[0].get("cnt", 0) if result else 0
            if cnt > 0:
                print(f"  - {edge_type}: {cnt}")
        except Exception as e:
            pass

    # 6. Check chunk vertices for physician docs
    print("\n=== Physician Document Chunks ===")
    try:
        result = await nebula_client.execute_json(
            "MATCH (c:chunk) WHERE c.chunk.doc_id CONTAINS 'physician' RETURN c LIMIT 10;"
        )
        print(f"Found {len(result)} chunks with 'physician' in doc_id")
        for r in result[:5]:
            node = r.get("c", {})
            if isinstance(node, dict):
                props = node.get("properties", {}).get("chunk", {})
                print(f"  - doc_id: {props.get('doc_id', 'N/A')}")
                print(f"    type: {props.get('chunk_type', 'N/A')}")
                print(f"    preview: {props.get('content_preview', 'N/A')[:80]}...")
    except Exception as e:
        print(f"  Error: {e}")

    # 7. Check all unique doc_ids in chunks
    print("\n=== Unique Doc Types in Chunks ===")
    try:
        result = await nebula_client.execute_json(
            "MATCH (c:chunk) RETURN DISTINCT substring(c.chunk.doc_id, 0, 20) AS prefix LIMIT 30;"
        )
        print(f"Found {len(result)} unique doc_id prefixes:")
        for r in result:
            print(f"  - {r.get('prefix', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

    await nebula_client.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
