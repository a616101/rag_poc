#!/usr/bin/env python3
"""
Check NebulaGraph data for physicians and related entities.
"""

import asyncio
import sys
sys.path.insert(0, "src")

async def main():
    from chatbot_graphrag.services.graph.nebula_client import nebula_client
    from chatbot_graphrag.core.constants import EntityType, NEBULA_ENTITY_TAG

    print("Connecting to NebulaGraph...")
    try:
        await nebula_client.initialize()
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # 1. Count all entities
    print("\n=== Entity Count by Type ===")
    for entity_type in EntityType:
        try:
            results = await nebula_client.find_entities_by_type(entity_type, limit=1000)
            print(f"  {entity_type.value}: {len(results)} entities")
        except Exception as e:
            print(f"  {entity_type.value}: Error - {e}")

    # 2. Search for physician entities
    print("\n=== Physician Entities ===")
    try:
        physician_results = await nebula_client.find_entities_by_type(EntityType.PHYSICIAN, limit=50)
        for r in physician_results[:10]:
            props = r.get("props", {})
            print(f"  - {props.get('name', 'N/A')} (vid: {r.get('vid', 'N/A')[:30]}...)")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. Search for 吳明昇 specifically
    print("\n=== Searching for '吳明昇' ===")
    try:
        results = await nebula_client.search_entities_by_name("吳明昇", limit=10)
        if results:
            for r in results:
                props = r.get("props", {})
                print(f"  Found: {props.get('name', 'N/A')}")
                print(f"    Type: {props.get('entity_type', 'N/A')}")
                print(f"    Description: {props.get('description', 'N/A')[:100]}...")
                print(f"    VID: {r.get('vid', 'N/A')}")
        else:
            print("  No results found via prefix search")
    except Exception as e:
        print(f"  Error: {e}")

    # 4. Check for medical expertise entities
    print("\n=== Medical Condition/Expertise Entities ===")
    try:
        results = await nebula_client.find_entities_by_type(EntityType.MEDICAL_CONDITION, limit=20)
        for r in results[:10]:
            props = r.get("props", {})
            print(f"  - {props.get('name', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

    # 5. Check department entities
    print("\n=== Department Entities ===")
    try:
        results = await nebula_client.find_entities_by_type(EntityType.DEPARTMENT, limit=20)
        for r in results[:10]:
            props = r.get("props", {})
            print(f"  - {props.get('name', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

    # 6. Get relations for a physician (if found)
    if physician_results:
        first_physician = physician_results[0]
        vid = first_physician.get("vid")
        props = first_physician.get("props", {})
        name = props.get("name", "Unknown")

        print(f"\n=== Relations for {name} ===")
        try:
            relations = await nebula_client.get_relations(vid)
            if relations:
                for r in relations[:10]:
                    edge = r.get("e", {})
                    print(f"  - {edge}")
            else:
                print("  No relations found")
        except Exception as e:
            print(f"  Error: {e}")

    # 7. Check chunks in graph
    print("\n=== Chunk Vertices ===")
    try:
        query = f"MATCH (c:chunk) RETURN c LIMIT 10;"
        results = await nebula_client.execute_json(query)
        print(f"  Found {len(results)} chunk vertices")
        for r in results[:5]:
            print(f"    - {r}")
    except Exception as e:
        print(f"  Error (may be expected if no chunks): {e}")

    # 8. Count total vertices and edges
    print("\n=== Graph Statistics ===")
    try:
        stats_query = "SHOW STATS;"
        result = await nebula_client.execute(stats_query)
        print(f"  Stats query result: {result}")
    except Exception as e:
        print(f"  Could not get stats: {e}")

    await nebula_client.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
