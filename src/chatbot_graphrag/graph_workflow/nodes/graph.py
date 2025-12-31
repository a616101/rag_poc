"""
圖譜節點

圖譜種子抽取、遍歷和子圖處理。

主要節點：
- graph_seed_extract_node: 從 chunk 中抽取實體種子
- graph_traverse_node: 從實體種子遍歷知識圖譜
- subgraph_to_queries_node: 根據圖譜結果生成額外查詢
"""

import logging
from typing import Any

from chatbot_graphrag.graph_workflow.types import GraphRAGState

logger = logging.getLogger(__name__)


async def graph_seed_extract_node(state: GraphRAGState) -> dict[str, Any]:
    """
    圖譜種子抽取節點。

    從重排序的 chunk 中抽取實體種子用於圖譜遍歷。

    Returns:
        更新後的狀態，包含實體種子
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("graph_seed_extract", "START")

    start_time = time.time()
    reranked_chunks = state.get("reranked_chunks", [])
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    logger.debug(f"Extracting graph seeds from {len(reranked_chunks)} chunks...")

    # 從 chunk metadata 中抽取實體 ID
    entity_ids = set()
    for chunk in reranked_chunks:
        if hasattr(chunk, "metadata"):
            chunk_entity_ids = chunk.metadata.get("entity_ids", [])
            entity_ids.update(chunk_entity_ids)

    entity_list = list(entity_ids)
    logger.info(f"Extracted {len(entity_list)} entity seeds")

    emit_status("graph_seed_extract", "DONE")
    return {
        "graph_subgraph": {"entity_seeds": entity_list},
        "retrieval_path": retrieval_path + [f"graph_seed_extract:{len(entity_list)}"],
        "timing": {**timing, "graph_seed_extract_ms": (time.time() - start_time) * 1000},
    }


async def graph_traverse_node(state: GraphRAGState) -> dict[str, Any]:
    """
    圖譜遍歷節點。

    從實體種子遍歷知識圖譜。

    Returns:
        更新後的狀態，包含遍歷結果
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("graph_traverse", "START")

    start_time = time.time()
    graph_subgraph = state.get("graph_subgraph", {})
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))
    budget = state.get("budget")

    entity_seeds = graph_subgraph.get("entity_seeds", [])

    if not entity_seeds:
        logger.debug("No entity seeds - skipping traversal")
        return {
            "graph_subgraph": graph_subgraph,
            "retrieval_path": retrieval_path + ["graph_traverse:no_seeds"],
            "timing": {**timing, "graph_traverse_ms": (time.time() - start_time) * 1000},
        }

    logger.info(f"Traversing graph from {len(entity_seeds)} seeds...")

    try:
        # 延遲導入 NebulaGraph 客戶端
        from chatbot_graphrag.services.graph import nebula_client

        # chunk metadata 中的實體 ID 已經是 VID 格式 (e_{type}_{hash})
        # 由 entity_extractor._generate_entity_id() 設定
        seed_vids = entity_seeds[:10]  # Limit to top 10 seeds

        logger.debug(f"Graph traverse seed VIDs: {seed_vids[:3]}...")

        # 遍歷圖譜
        traversal_result = await nebula_client.traverse(
            seed_vids=seed_vids,
            max_hops=2,
            max_vertices=50,  # Total max vertices across all hops
        )

        # 使用遍歷結果更新子圖（GraphTraversalResult 是 Pydantic 模型）
        entities = traversal_result.subgraph.entities
        relations = traversal_result.subgraph.relations

        # 從實體中抽取社群 ID
        community_ids = set()
        for entity in entities:
            # Entity 是 Pydantic 模型，存取可能包含社群資訊的 doc_ids
            if hasattr(entity, 'doc_ids') and entity.doc_ids:
                community_ids.update(entity.doc_ids)

        graph_subgraph.update({
            "entities": [e.model_dump() for e in entities],
            "relations": [r.model_dump() for r in relations],
            "community_ids": list(community_ids),
        })

        # 更新預算
        if budget and hasattr(budget, "increment_queries"):
            budget.increment_queries()

        entity_count = len(graph_subgraph.get("entities", []))
        relation_count = len(graph_subgraph.get("relations", []))
        logger.info(f"Traversed: {entity_count} entities, {relation_count} relations")

        emit_status("graph_traverse", "DONE")
        return {
            "graph_subgraph": graph_subgraph,
            "retrieval_path": retrieval_path + [f"graph_traverse:{entity_count}e,{relation_count}r"],
            "timing": {**timing, "graph_traverse_ms": (time.time() - start_time) * 1000},
            "budget": budget,
        }

    except Exception as e:
        logger.error(f"Graph traversal error: {e}")
        return {
            "graph_subgraph": graph_subgraph,
            "error": str(e),
            "retrieval_path": retrieval_path + ["graph_traverse:error"],
            "timing": {**timing, "graph_traverse_ms": (time.time() - start_time) * 1000},
        }


async def subgraph_to_queries_node(state: GraphRAGState) -> dict[str, Any]:
    """
    子圖轉查詢節點。

    根據圖譜遍歷結果生成額外的查詢。

    Returns:
        更新後的狀態，包含額外的查詢
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("subgraph_to_queries", "START")

    start_time = time.time()
    question = state.get("normalized_question") or state.get("question", "")
    graph_subgraph = state.get("graph_subgraph", {})
    followup_queries = list(state.get("followup_queries", []))
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    entities = graph_subgraph.get("entities", [])

    logger.debug(f"Generating queries from {len(entities)} entities...")

    # 抽取實體名稱用於查詢擴展
    entity_names = set()
    for entity in entities[:10]:
        if isinstance(entity, dict):
            name = entity.get("name", "")
            if name:
                entity_names.add(name)

    # 生成基於實體的追蹤查詢
    new_queries = []
    for name in list(entity_names)[:3]:
        new_queries.append(f"{question} {name}")

    followup_queries.extend(new_queries)
    logger.info(f"Added {len(new_queries)} entity-based queries")

    emit_status("subgraph_to_queries", "DONE")
    return {
        "followup_queries": followup_queries,
        "retrieval_path": retrieval_path + [f"subgraph_to_queries:{len(new_queries)}"],
        "timing": {**timing, "subgraph_to_queries_ms": (time.time() - start_time) * 1000},
    }
