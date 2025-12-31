"""
GraphRAG 的 LangGraph 工作流程

本模組提供 GraphRAG 問答系統的主要工作流程建構器和所有支援類型。

包含：
- 工作流程工廠函數（build_graphrag_workflow 等）
- 狀態類型定義（GraphRAGState）
- 預算管理（BudgetManager）
- 路由函數（route_after_guard, route_by_intent 等）
"""

from chatbot_graphrag.graph_workflow.types import (
    GraphRAGState,
    QueryMode,
    GroundednessStatus,
    LoopBudget,
    EvidenceItem,
    RetrievalResult,
    create_initial_state,
)
from chatbot_graphrag.graph_workflow.budget import (
    BudgetManager,
    BudgetSnapshot,
    create_budget,
    estimate_tokens,
)
from chatbot_graphrag.graph_workflow.routing import (
    route_after_guard,
    route_after_cache,
    route_by_intent,
    route_after_groundedness,
    route_by_budget,
    route_after_retrieval,
    route_hitl,
)
from chatbot_graphrag.graph_workflow.factory import (
    build_graphrag_workflow,
    build_graphrag_workflow_with_memory,
    build_simple_workflow,
)

__all__ = [
    # ==================== 工廠函數 ====================
    "build_graphrag_workflow",
    "build_graphrag_workflow_with_memory",
    "build_simple_workflow",
    # ==================== 類型定義 ====================
    "GraphRAGState",
    "QueryMode",
    "GroundednessStatus",
    "LoopBudget",
    "EvidenceItem",
    "RetrievalResult",
    "create_initial_state",
    # ==================== 預算管理 ====================
    "BudgetManager",
    "BudgetSnapshot",
    "create_budget",
    "estimate_tokens",
    # ==================== 路由函數 ====================
    "route_after_guard",
    "route_after_cache",
    "route_by_intent",
    "route_after_groundedness",
    "route_by_budget",
    "route_after_retrieval",
    "route_hitl",
]
