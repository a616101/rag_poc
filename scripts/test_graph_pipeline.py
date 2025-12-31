#!/usr/bin/env python3
"""
Test script for GraphRAG pipeline components.

Tests:
1. Entity extraction from sample text
2. Relation extraction between entities
3. Graph batch loading (requires NebulaGraph)
4. Community detection
5. Community summarization

Usage:
    PYTHONPATH=src python scripts/test_graph_pipeline.py
"""

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Sample Chinese text for testing
SAMPLE_TEXT = """
心臟血管科位於門診大樓三樓，提供心臟疾病診斷與治療服務。
吳明昇醫師是心臟血管科的主治醫師，專長為心導管手術及心律不整治療。
心臟超音波檢查需要預約，可至一樓掛號處辦理。
患者需攜帶健保卡及身分證，若有慢性病連續處方籤可至藥局領藥。
"""


async def test_entity_extraction():
    """Test entity extraction from sample text."""
    logger.info("=" * 60)
    logger.info("Testing Entity Extraction")
    logger.info("=" * 60)

    from chatbot_graphrag.services.graph.entity_extractor import entity_extractor

    await entity_extractor.initialize()

    entities = await entity_extractor.extract_entities(
        content=SAMPLE_TEXT,
        chunk_id="test_chunk_001",
        doc_id="test_doc_001",
    )

    logger.info(f"Extracted {len(entities)} entities:")
    for entity in entities:
        logger.info(f"  - {entity.name} ({entity.entity_type.value}): {entity.description}")

    return entities


async def test_relation_extraction(entities):
    """Test relation extraction between entities."""
    logger.info("=" * 60)
    logger.info("Testing Relation Extraction")
    logger.info("=" * 60)

    if len(entities) < 2:
        logger.warning("Not enough entities for relation extraction")
        return []

    from chatbot_graphrag.services.graph.relation_extractor import relation_extractor

    await relation_extractor.initialize()

    relations = await relation_extractor.extract_relations(
        content=SAMPLE_TEXT,
        entities=entities,
        chunk_id="test_chunk_001",
        doc_id="test_doc_001",
    )

    logger.info(f"Extracted {len(relations)} relations:")
    entity_name_map = {e.id: e.name for e in entities}
    for relation in relations:
        src_name = entity_name_map.get(relation.source_id, relation.source_id)
        tgt_name = entity_name_map.get(relation.target_id, relation.target_id)
        logger.info(f"  - {src_name} --[{relation.relation_type.value}]--> {tgt_name}")

    return relations


async def test_community_detection(entities, relations):
    """Test community detection."""
    logger.info("=" * 60)
    logger.info("Testing Community Detection")
    logger.info("=" * 60)

    if len(entities) < 3:
        logger.warning("Not enough entities for community detection")
        return []

    from chatbot_graphrag.services.graph.community_detector import community_detector

    await community_detector.initialize()

    result = await community_detector.detect_communities(
        entities=entities,
        relations=relations,
        max_levels=2,
        resolution=1.0,
    )

    logger.info(f"Detected {result.total_communities} communities across {result.levels} levels")
    logger.info(f"Modularity: {result.modularity:.4f}")
    logger.info(f"Execution time: {result.execution_time_ms:.1f}ms")

    for community in result.communities:
        logger.info(f"  - Community {community.id} (level {community.level}): {len(community.entity_ids)} entities")

    return result.communities


async def test_community_summarization(communities, entities, relations):
    """Test community summarization."""
    logger.info("=" * 60)
    logger.info("Testing Community Summarization")
    logger.info("=" * 60)

    if not communities:
        logger.warning("No communities to summarize")
        return

    from chatbot_graphrag.services.graph.community_summarizer import community_summarizer

    await community_summarizer.initialize()

    # Build entity/relation maps by community
    entities_by_community = {}
    relations_by_community = {}
    entity_id_set = {e.id for e in entities}

    for community in communities:
        community_entities = [e for e in entities if e.id in community.entity_ids]
        entities_by_community[community.id] = community_entities

        # Get relations within community
        community_relations = [
            r for r in relations
            if r.source_id in community.entity_ids and r.target_id in community.entity_ids
        ]
        relations_by_community[community.id] = community_relations

    result = await community_summarizer.summarize_communities(
        communities=communities,
        entities_by_community=entities_by_community,
        relations_by_community=relations_by_community,
        concurrency=2,
    )

    logger.info(f"Generated {result.total_summarized} summaries")
    logger.info(f"Total tokens: {result.total_tokens}")

    for report in result.reports:
        logger.info(f"  - {report.title}")
        logger.info(f"    Summary: {report.summary[:100]}...")
        logger.info(f"    Themes: {', '.join(report.themes)}")


async def test_graph_batch_loader():
    """Test full batch loading (requires NebulaGraph)."""
    logger.info("=" * 60)
    logger.info("Testing Graph Batch Loader")
    logger.info("=" * 60)

    from chatbot_graphrag.models.pydantic.ingestion import Chunk, ChunkMetadata
    from chatbot_graphrag.core.constants import ChunkType
    from chatbot_graphrag.services.graph.batch_loader import graph_batch_loader

    # Create test chunks
    chunks = [
        Chunk(
            id="chunk_001",
            doc_id="doc_001",
            doc_version="1.0",
            content=SAMPLE_TEXT,
            contextual_content="Hospital services information",
            metadata=ChunkMetadata(
                chunk_type=ChunkType.PARAGRAPH,
                section_title="心臟血管科",
                position_in_doc=0,
                char_start=0,
                char_end=len(SAMPLE_TEXT),
            ),
        ),
    ]

    await graph_batch_loader.initialize()

    result = await graph_batch_loader.process_chunks(
        chunks=chunks,
        doc_id="doc_001",
        concurrency=1,
        enable_relations=True,
    )

    logger.info(f"Batch load result:")
    logger.info(f"  - Chunks processed: {result.chunks_processed}")
    logger.info(f"  - Entities extracted: {result.entities_extracted}")
    logger.info(f"  - Relations extracted: {result.relations_extracted}")
    logger.info(f"  - Entities loaded: {result.entities_loaded}")
    logger.info(f"  - Relations loaded: {result.relations_loaded}")
    logger.info(f"  - Execution time: {result.execution_time_ms:.1f}ms")

    if result.errors:
        logger.warning(f"  - Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            logger.warning(f"    - {error}")


async def main():
    """Run all tests."""
    logger.info("GraphRAG Pipeline Test Suite")
    logger.info("=" * 60)

    try:
        # Test 1: Entity extraction
        entities = await test_entity_extraction()

        # Test 2: Relation extraction
        relations = await test_relation_extraction(entities)

        # Test 3: Community detection
        communities = await test_community_detection(entities, relations)

        # Test 4: Community summarization
        await test_community_summarization(communities, entities, relations)

        logger.info("=" * 60)
        logger.info("Basic tests completed!")

        # Test 5: Full batch loading (optional, requires NebulaGraph)
        run_batch_test = "--full" in sys.argv
        if run_batch_test:
            await test_graph_batch_loader()
        else:
            logger.info("Skipping batch loader test. Use --full to run it.")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
