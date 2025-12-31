#!/usr/bin/env python3
"""
Test LLM-based query decomposition for improved retrieval.
"""

import asyncio
import sys

sys.path.insert(0, "src")


async def test_llm_decomposition():
    """Test the LLM-based query decomposition logic."""
    from chatbot_graphrag.services.search.query_decomposer import decompose_query_with_llm

    test_queries = [
        # Entity lookup - specific physician
        "吳明昇醫師主治哪些項目",
        "吳明昇醫師的專長是什麼",
        "吳明昇什麼時候有門診",
        # Reverse lookup - find physicians by property
        "哪一些醫師有當過主任？",
        "哪些醫師曾任院長？",
        "心臟內科有哪些醫師",
        # General queries
        "請問掛號流程",
        "如何預約看診",
        "如何掛號?",
    ]

    print("=" * 60)
    print("Testing LLM-based Query Decomposition")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        try:
            result = await decompose_query_with_llm(query)
            print(f"  Query Type: {result.query_type}")
            print(f"  Reasoning: {result.reasoning}")
            print(f"  Physician name: {result.physician_name}")
            print(f"  Department: {result.department}")
            print(f"  Property type: {result.property_type}")
            print(f"  Sub-queries: {result.sub_queries}")
            print(f"  Metadata filters: {result.metadata_filters}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


async def test_retrieval():
    """Test retrieval with LLM-based query decomposition (requires running services)."""
    try:
        from chatbot_graphrag.services.search.hybrid_search import (
            HybridSearchConfig,
            hybrid_search_service,
        )
        from chatbot_graphrag.services.search.query_decomposer import decompose_query_with_llm
        from chatbot_graphrag.services.vector.embedding_service import embedding_service

        print("\n" + "=" * 60)
        print("Testing Retrieval with LLM Query Decomposition")
        print("=" * 60)

        # Initialize services
        print("\nInitializing services...")
        await embedding_service.initialize()
        await hybrid_search_service.initialize()

        # Test reverse lookup query
        query = "哪一些醫師有當過主任？"
        print(f"\nTest query: {query}")

        # Decompose query with LLM
        decomposed = await decompose_query_with_llm(query)
        print(f"\nDecomposition:")
        print(f"  Type: {decomposed.query_type}")
        print(f"  Reasoning: {decomposed.reasoning}")
        print(f"  Sub-queries: {decomposed.sub_queries}")
        print(f"  Metadata filters: {decomposed.metadata_filters}")

        # Collect all queries to search
        queries_to_search = [query]
        if decomposed.sub_queries:
            queries_to_search.extend(decomposed.sub_queries)

        print(f"\nSearching with {len(queries_to_search)} queries...")

        # Generate embeddings
        import asyncio
        embeddings = await asyncio.gather(
            *[embedding_service.embed_text(q) for q in queries_to_search]
        )

        # Create search config with filters
        config = HybridSearchConfig(
            dense_limit=20,
            sparse_limit=20,
            fts_limit=20,
            final_limit=15,
        )

        # No doc_type filters - trust semantic search to find relevant documents

        # Execute searches in parallel
        search_tasks = [
            hybrid_search_service.search(
                query=q,
                query_embedding=emb,
                config=config,
            )
            for q, emb in zip(queries_to_search, embeddings)
        ]

        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Merge results
        seen_chunks = set()
        merged_results = []
        for i, results in enumerate(all_results):
            if isinstance(results, Exception):
                print(f"Search {i} failed: {results}")
                continue
            for r in results:
                if r.chunk_id not in seen_chunks:
                    seen_chunks.add(r.chunk_id)
                    merged_results.append(r)

        # Sort by score
        merged_results.sort(key=lambda r: r.score, reverse=True)
        merged_results = merged_results[:15]

        print(f"\nFound {len(merged_results)} results:")
        for i, r in enumerate(merged_results[:10]):
            print(f"\n  [{i + 1}] Score: {r.score:.4f}")
            print(f"      Doc: {r.metadata.get('title', 'N/A')}")
            print(f"      Section: {r.metadata.get('section_title', 'N/A')}")
            print(f"      Type: {r.metadata.get('chunk_type', 'N/A')}")
            content_preview = r.content[:150].replace("\n", " ")
            print(f"      Content: {content_preview}...")

    except Exception as e:
        print(f"Retrieval test failed (services may not be running): {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # First test LLM decomposition
    asyncio.run(test_llm_decomposition())

    # Then test retrieval (requires services)
    print("\n\nAttempting retrieval test...")
    asyncio.run(test_retrieval())
