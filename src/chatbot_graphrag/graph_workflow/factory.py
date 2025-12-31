"""
GraphRAG LangGraph 工作流程工廠

建構 GraphRAG 問答系統的計算圖。
支援 HITL（人機協作）工作流程的持久化檢查點。

主要工廠函數：
- build_graphrag_workflow: 建構完整工作流程
- build_graphrag_workflow_with_memory: 建構帶檢查點的工作流程
- build_simple_workflow: 建構簡化測試工作流程
"""

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from chatbot_graphrag.graph_workflow.types import GraphRAGState
from chatbot_graphrag.graph_workflow.routing import (
    route_after_guard,
    route_after_cache,
    route_by_intent,
    route_after_groundedness,
    route_by_budget,
    route_after_retrieval,
    route_hitl,
)

logger = logging.getLogger(__name__)

# 單例 checkpointer 實例（跨工作流程建構重用）
_persistent_checkpointer: Optional[BaseCheckpointSaver] = None


def get_checkpointer(use_persistence: Optional[bool] = None) -> BaseCheckpointSaver:
    """
    取得用於工作流程狀態持久化的 checkpointer。

    第 3 階段：支援 PostgreSQL 持久化 HITL 工作流程。

    Args:
        use_persistence: 覆蓋配置設定。若為 None，使用 settings.hitl_persistence_enabled

    Returns:
        BaseCheckpointSaver: PostgresSaver（持久化）或 MemorySaver（記憶體內）
    """
    global _persistent_checkpointer

    from chatbot_graphrag.core.config import settings

    # Determine if we should use persistence
    if use_persistence is None:
        use_persistence = settings.hitl_persistence_enabled

    if not use_persistence:
        logger.info("Using in-memory checkpointer (non-persistent)")
        return MemorySaver()

    # Return cached instance if available
    if _persistent_checkpointer is not None:
        return _persistent_checkpointer

    try:
        # Try to use PostgreSQL checkpointer (LangGraph >= 0.2.0)
        from langgraph.checkpoint.postgres import PostgresSaver
        import psycopg2

        # Parse connection URL and create PostgresSaver
        conn_string = settings.postgres_url
        logger.info(f"Initializing PostgreSQL checkpointer...")

        # Create connection pool
        connection = psycopg2.connect(conn_string)
        _persistent_checkpointer = PostgresSaver(connection)

        # Setup tables if needed (idempotent)
        _persistent_checkpointer.setup()

        logger.info("PostgreSQL checkpointer initialized successfully")
        return _persistent_checkpointer

    except ImportError as e:
        logger.warning(
            f"PostgreSQL checkpointer not available: {e}. "
            "Install with: pip install langgraph-checkpoint-postgres"
        )
        return MemorySaver()

    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}")
        logger.info("Falling back to in-memory checkpointer")
        return MemorySaver()


async def get_async_checkpointer(use_persistence: Optional[bool] = None) -> BaseCheckpointSaver:
    """
    取得用於工作流程狀態持久化的非同步 checkpointer。

    第 3 階段：支援非同步 PostgreSQL 以提供高效能 HITL 工作流程。

    Args:
        use_persistence: 覆蓋配置設定。若為 None，使用 settings.hitl_persistence_enabled

    Returns:
        BaseCheckpointSaver: AsyncPostgresSaver（持久化）或 MemorySaver（記憶體內）
    """
    global _persistent_checkpointer

    from chatbot_graphrag.core.config import settings

    if use_persistence is None:
        use_persistence = settings.hitl_persistence_enabled

    if not use_persistence:
        logger.info("Using in-memory checkpointer (non-persistent)")
        return MemorySaver()

    if _persistent_checkpointer is not None:
        return _persistent_checkpointer

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        conn_string = settings.postgres_url
        logger.info("Initializing async PostgreSQL checkpointer...")

        _persistent_checkpointer = await AsyncPostgresSaver.from_conn_string(conn_string)
        await _persistent_checkpointer.setup()

        logger.info("Async PostgreSQL checkpointer initialized successfully")
        return _persistent_checkpointer

    except ImportError as e:
        logger.warning(
            f"Async PostgreSQL checkpointer not available: {e}. "
            "Install with: pip install langgraph-checkpoint-postgres"
        )
        return MemorySaver()

    except Exception as e:
        logger.error(f"Failed to initialize async PostgreSQL checkpointer: {e}")
        return MemorySaver()


def build_graphrag_workflow(
    checkpointer: Optional[Any] = None,
    interrupt_before: Optional[list[str]] = None,
) -> StateGraph:
    """
    Build the GraphRAG LangGraph workflow.

    The workflow implements a 22-node graph:

    START → guard → acl → normalize → cache_lookup
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼ [hit]                                   ▼ [miss]
              cache_response                           intent_router
                    │                                         │
                    │         ┌───────────────────────────────┼───────────────────────────────┐
                    │         ▼ [direct]                      ▼ [local]                       ▼ [global/drift]
                    │    direct_answer               hybrid_seed              community_reports → followups
                    │         │                           │                            │
                    │         │                           ▼                            ▼
                    │         │                      rrf_merge ←───────────────── rrf_merge
                    │         │                           │
                    │         │                           ▼
                    │         │                    rerank_40_to_12
                    │         │                           │
                    │         │                           ▼
                    │         │                   graph_seed_extract → graph_traverse
                    │         │                                              │
                    │         │                                              ▼
                    │         │                                      subgraph_to_queries
                    │         │                                              │
                    │         │                                              ▼
                    │         │                                         hop_hybrid → rerank_hop
                    │         │                                                          │
                    │         │                                                          ▼
                    │         │                                                   chunk_expander
                    │         │                                                          │
                    │         │                                                          ▼
                    │         │                                                   context_packer
                    │         │                                                          │
                    │         │                                                          ▼
                    │         │                                                   evidence_table
                    │         │                                                          │
                    │         │                                                          ▼
                    │         │                                                   groundedness
                    │         │                                    ┌───────────────────┼───────────────────┐
                    │         │                                    ▼ [pass]            ▼ [retry]          ▼ [needs_review]
                    │         │                             final_answer ←── targeted_retry    interrupt_hitl
                    │         │                                    │                                  │
                    │         │                                    │                                  ▼
                    │         └────────────────────────────────────┼─────────────────────────→ final_answer
                    │                                              │
                    │                                              ▼
                    └──────────────────────────────────────→ cache_store → telemetry → END

    Args:
        checkpointer: Optional checkpointer for state persistence
        interrupt_before: List of nodes to interrupt before (for HITL)

    Returns:
        Compiled StateGraph
    """
    # Import nodes lazily to avoid circular imports
    from chatbot_graphrag.graph_workflow.nodes import (
        guard_node,
        acl_node,
        normalize_node,
        cache_lookup_node,
        cache_response_node,
        intent_router_node,
        direct_answer_node,
        hybrid_seed_node,
        community_reports_node,
        followups_node,
        rrf_merge_node,
        rerank_node,
        graph_seed_extract_node,
        graph_traverse_node,
        subgraph_to_queries_node,
        hop_hybrid_node,
        chunk_expander_node,
        context_packer_node,
        evidence_table_node,
        groundedness_node,
        targeted_retry_node,
        interrupt_hitl_node,
        final_answer_node,
        cache_store_node,
        telemetry_node,
    )

    # Create the graph
    workflow = StateGraph(GraphRAGState)

    # === Add Nodes ===

    # Input processing
    workflow.add_node("guard", guard_node)
    workflow.add_node("acl", acl_node)
    workflow.add_node("normalize", normalize_node)
    workflow.add_node("cache_lookup", cache_lookup_node)
    workflow.add_node("cache_response", cache_response_node)

    # Intent routing
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("direct_answer", direct_answer_node)

    # Retrieval - Local mode
    workflow.add_node("hybrid_seed", hybrid_seed_node)
    workflow.add_node("rrf_merge", rrf_merge_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("graph_seed_extract", graph_seed_extract_node)
    workflow.add_node("graph_traverse", graph_traverse_node)
    workflow.add_node("subgraph_to_queries", subgraph_to_queries_node)
    workflow.add_node("hop_hybrid", hop_hybrid_node)

    # Retrieval - Global/DRIFT mode
    workflow.add_node("community_reports", community_reports_node)
    workflow.add_node("followups", followups_node)

    # Context building
    workflow.add_node("chunk_expander", chunk_expander_node)
    workflow.add_node("context_packer", context_packer_node)
    workflow.add_node("evidence_table", evidence_table_node)

    # Quality & output
    workflow.add_node("groundedness", groundedness_node)
    workflow.add_node("targeted_retry", targeted_retry_node)
    workflow.add_node("interrupt_hitl", interrupt_hitl_node)
    workflow.add_node("final_answer", final_answer_node)
    workflow.add_node("cache_store", cache_store_node)
    workflow.add_node("telemetry", telemetry_node)

    # === Add Edges ===

    # Start -> Guard
    workflow.add_edge(START, "guard")

    # Guard -> ACL or END (if blocked)
    workflow.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "blocked": "final_answer",  # Return error response
            "continue": "acl",
        },
    )

    # ACL -> Normalize
    workflow.add_conditional_edges(
        "acl",
        route_after_guard,  # Reuse guard routing for ACL
        {
            "blocked": "final_answer",
            "continue": "normalize",
        },
    )

    # Normalize -> Cache Lookup
    workflow.add_edge("normalize", "cache_lookup")

    # Cache Lookup -> Cache Response or Intent Router
    workflow.add_conditional_edges(
        "cache_lookup",
        route_after_cache,
        {
            "hit": "cache_response",
            "miss": "intent_router",
        },
    )

    # Cache Response -> Final Answer (for streaming cached content)
    # Note: final_answer_node will stream the cached response via SSE
    # and cache_store_node will skip re-caching due to cache_hit=True
    workflow.add_edge("cache_response", "final_answer")

    # Intent Router -> Direct/Local/Global/Drift
    workflow.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "direct": "direct_answer",
            "local": "hybrid_seed",
            "global": "community_reports",
            "drift": "community_reports",  # DRIFT starts with community reports
        },
    )

    # Direct Answer -> Cache Store
    workflow.add_edge("direct_answer", "cache_store")

    # === Local Mode Path ===
    workflow.add_edge("hybrid_seed", "rrf_merge")
    workflow.add_edge("rrf_merge", "rerank")
    workflow.add_edge("rerank", "graph_seed_extract")
    workflow.add_edge("graph_seed_extract", "graph_traverse")
    workflow.add_edge("graph_traverse", "subgraph_to_queries")
    workflow.add_edge("subgraph_to_queries", "hop_hybrid")
    workflow.add_edge("hop_hybrid", "chunk_expander")

    # === Global/DRIFT Mode Path ===
    workflow.add_edge("community_reports", "followups")
    workflow.add_edge("followups", "hybrid_seed")  # Join with local path

    # === Context Building Path ===
    workflow.add_edge("chunk_expander", "context_packer")
    workflow.add_edge("context_packer", "evidence_table")
    workflow.add_edge("evidence_table", "groundedness")

    # Groundedness -> Final Answer / Retry / HITL
    workflow.add_conditional_edges(
        "groundedness",
        route_after_groundedness,
        {
            "pass": "final_answer",
            "retry": "targeted_retry",
            "needs_review": "interrupt_hitl",
        },
    )

    # Targeted Retry -> Hop Hybrid (loop back)
    workflow.add_conditional_edges(
        "targeted_retry",
        route_by_budget,
        {
            "continue": "hop_hybrid",
            "exhausted": "final_answer",
        },
    )

    # HITL -> Final Answer / Wait / Timeout
    # Phase 3: Added timeout route for graceful degradation
    workflow.add_conditional_edges(
        "interrupt_hitl",
        route_hitl,
        {
            "wait": END,  # Interrupt and wait for human input
            "continue": "final_answer",  # Human review completed
            "timeout": "final_answer",  # Timeout - use fallback (handled in final_answer)
        },
    )

    # Final Answer -> Cache Store
    workflow.add_edge("final_answer", "cache_store")

    # Cache Store -> Telemetry
    workflow.add_edge("cache_store", "telemetry")

    # Telemetry -> END
    workflow.add_edge("telemetry", END)

    # Compile the graph
    if checkpointer:
        compiled = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before or ["interrupt_hitl"],
        )
    else:
        compiled = workflow.compile()

    logger.info("GraphRAG workflow compiled successfully")
    return compiled


def build_graphrag_workflow_with_memory(use_persistence: Optional[bool] = None) -> StateGraph:
    """
    建構帶有檢查點的工作流程，支援 HITL。

    第 3 階段：配置時使用持久化 PostgreSQL checkpointer。

    Args:
        use_persistence: 覆蓋持久化配置設定。
            若為 None，使用 settings.hitl_persistence_enabled

    Returns:
        帶有檢查點的已編譯 StateGraph
    """
    checkpointer = get_checkpointer(use_persistence)
    return build_graphrag_workflow(
        checkpointer=checkpointer,
        interrupt_before=["interrupt_hitl"],
    )


async def build_graphrag_workflow_with_memory_async(
    use_persistence: Optional[bool] = None,
) -> StateGraph:
    """
    建構帶有非同步檢查點的工作流程，支援 HITL。

    第 3 階段：使用非同步 PostgreSQL checkpointer 以提供高效能場景。

    Args:
        use_persistence: 覆蓋持久化配置設定。

    Returns:
        帶有非同步檢查點的已編譯 StateGraph
    """
    checkpointer = await get_async_checkpointer(use_persistence)
    return build_graphrag_workflow(
        checkpointer=checkpointer,
        interrupt_before=["interrupt_hitl"],
    )


def build_simple_workflow() -> StateGraph:
    """
    建構簡化的測試工作流程。

    只包含基本節點：guard -> normalize -> hybrid_seed -> final_answer
    """
    from chatbot_graphrag.graph_workflow.nodes import (
        guard_node,
        normalize_node,
        hybrid_seed_node,
        final_answer_node,
        telemetry_node,
    )

    workflow = StateGraph(GraphRAGState)

    workflow.add_node("guard", guard_node)
    workflow.add_node("normalize", normalize_node)
    workflow.add_node("hybrid_seed", hybrid_seed_node)
    workflow.add_node("final_answer", final_answer_node)
    workflow.add_node("telemetry", telemetry_node)

    workflow.add_edge(START, "guard")
    workflow.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "blocked": "final_answer",
            "continue": "normalize",
        },
    )
    workflow.add_edge("normalize", "hybrid_seed")
    workflow.add_edge("hybrid_seed", "final_answer")
    workflow.add_edge("final_answer", "telemetry")
    workflow.add_edge("telemetry", END)

    return workflow.compile()
