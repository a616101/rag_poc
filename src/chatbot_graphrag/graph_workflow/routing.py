"""
GraphRAG 工作流程的條件路由

定義根據狀態決定下一個節點的路由函數。

路由函數類型：
- route_after_guard: 守衛後路由
- route_after_cache: 快取後路由
- route_by_intent: 意圖路由
- route_after_groundedness: 落地性後路由
- route_by_budget: 預算路由
- route_hitl: HITL 路由
"""

import logging
from typing import Literal

from chatbot_graphrag.graph_workflow.types import (
    GraphRAGState,
    GroundednessStatus,
    QueryMode,
)

logger = logging.getLogger(__name__)

# 路由返回的類型別名
GuardRoute = Literal["blocked", "continue"]  # 守衛路由
IntentRoute = Literal["direct", "local", "global", "drift"]  # 意圖路由
GroundednessRoute = Literal["pass", "retry", "needs_review"]  # 落地性路由
CacheRoute = Literal["hit", "miss"]  # 快取路由
BudgetRoute = Literal["continue", "exhausted"]  # 預算路由


def route_after_guard(state: GraphRAGState) -> GuardRoute:
    """
    守衛節點後的路由。

    Returns:
        "blocked" 如果輸入被阻擋
        "continue" 如果輸入通過安全檢查
    """
    if state.get("guard_blocked", False):
        logger.info(f"Guard blocked: {state.get('guard_reason', 'unknown')}")
        return "blocked"

    if state.get("acl_denied", False):
        logger.info("ACL denied access")
        return "blocked"

    return "continue"


def route_after_cache(state: GraphRAGState) -> CacheRoute:
    """
    快取查詢後的路由。

    Returns:
        "hit" 如果快取命中
        "miss" 如果快取未命中
    """
    # Check if we have a cached answer
    cached_answer = state.get("final_answer", "")
    cache_hit_flag = state.get("cache_hit", False)

    logger.debug(f"route_after_cache: cache_hit={cache_hit_flag}, has_answer={bool(cached_answer)}, answer_len={len(cached_answer) if cached_answer else 0}")

    if cached_answer and cache_hit_flag:
        logger.info(f"Cache hit - routing to cache_response (answer: {len(cached_answer)} chars)")
        return "hit"

    logger.debug("Cache miss - routing to intent_router")
    return "miss"


def route_by_intent(state: GraphRAGState) -> IntentRoute:
    """
    根據意圖分析路由。

    Returns:
        "direct" - 直接回答，無需檢索
        "local" - 使用本地模式（基於實體）
        "global" - 使用全局模式（基於社群）
        "drift" - 使用 DRIFT 模式（動態探索）
    """
    query_mode = state.get("query_mode", "")

    if query_mode == QueryMode.DIRECT.value:
        logger.info("Intent: direct answer")
        return "direct"
    elif query_mode == QueryMode.GLOBAL.value:
        logger.info("Intent: global mode (community)")
        return "global"
    elif query_mode == QueryMode.DRIFT.value:
        logger.info("Intent: drift mode (exploration)")
        return "drift"
    else:
        # Default to local mode
        logger.info("Intent: local mode (entity)")
        return "local"


def route_after_groundedness(state: GraphRAGState) -> GroundednessRoute:
    """
    根據落地性評估路由。

    Returns:
        "pass" - 證據支持回答
        "retry" - 需要更多證據（循環回去）
        "needs_review" - 需要人工審核（HITL）
    """
    status = state.get("groundedness_status", "")
    score = state.get("groundedness_score", 0.0)

    # Check budget before allowing retry
    budget = state.get("budget")
    budget_exhausted = False
    if budget:
        budget_exhausted = budget.is_exhausted() if hasattr(budget, "is_exhausted") else False

    if status == GroundednessStatus.PASS.value or score >= 0.8:
        logger.info(f"Groundedness passed (score={score:.2f})")
        return "pass"

    if status == GroundednessStatus.NEEDS_REVIEW.value:
        logger.info("Groundedness requires human review")
        return "needs_review"

    # Retry if budget allows
    if status == GroundednessStatus.RETRY.value and not budget_exhausted:
        logger.info(f"Groundedness retry (score={score:.2f})")
        return "retry"

    # Budget exhausted or low score but no explicit retry
    if budget_exhausted:
        logger.info("Budget exhausted - forcing pass")
        return "pass"

    if score < 0.4:
        logger.info(f"Low groundedness score ({score:.2f}) - needs review")
        return "needs_review"

    # Default to pass for medium scores
    return "pass"


def route_by_budget(state: GraphRAGState) -> BudgetRoute:
    """
    根據預算狀態路由。

    Returns:
        "continue" - 預算允許更多迭代
        "exhausted" - 預算已耗盡
    """
    budget = state.get("budget")

    if not budget:
        logger.warning("No budget in state - defaulting to continue")
        return "continue"

    if hasattr(budget, "is_exhausted") and budget.is_exhausted():
        logger.info("Budget exhausted")
        return "exhausted"

    if hasattr(budget, "can_continue") and not budget.can_continue():
        logger.info("Budget cannot continue")
        return "exhausted"

    return "continue"


def route_after_retrieval(state: GraphRAGState) -> Literal["has_results", "no_results"]:
    """
    根據檢索結果路由。

    Returns:
        "has_results" - 有一些結果
        "no_results" - 找不到結果
    """
    merged = state.get("merged_results")
    if merged and hasattr(merged, "chunks") and len(merged.chunks) > 0:
        return "has_results"

    seed = state.get("seed_results")
    if seed and hasattr(seed, "chunks") and len(seed.chunks) > 0:
        return "has_results"

    return "no_results"


HITLRoute = Literal["wait", "continue", "timeout"]  # HITL 路由


def route_hitl(state: GraphRAGState) -> HITLRoute:
    """
    HITL 中斷的路由。

    第 3 階段：支援逾時處理以優雅降級。

    Returns:
        "wait" - 等待人工輸入
        "continue" - 人工審核完成或不需要
        "timeout" - HITL 逾時，使用降級回應
    """
    # Check if HITL is required and not resolved
    if state.get("hitl_required", False) and not state.get("hitl_resolved", False):
        # Check for timeout
        from chatbot_graphrag.graph_workflow.nodes.quality import check_hitl_timeout

        if check_hitl_timeout(state):
            logger.warning("HITL timeout - proceeding with fallback response")
            return "timeout"

        logger.info("HITL required - waiting for human input")
        return "wait"

    # Check if HITL was approved/rejected
    if state.get("hitl_approved") is not None:
        if state.get("hitl_approved"):
            logger.info("HITL approved - continuing")
        else:
            logger.info("HITL rejected - using fallback")
        return "continue"

    return "continue"


def should_expand_chunks(state: GraphRAGState) -> bool:
    """檢查是否應該執行 chunk 擴展。"""
    reranked = state.get("reranked_chunks", [])
    return len(reranked) > 0 and len(reranked) <= 20  # 太多則不擴展


def should_build_evidence_table(state: GraphRAGState) -> bool:
    """檢查是否應該建構證據表。"""
    expanded = state.get("expanded_chunks", [])
    return len(expanded) > 0


def get_routing_decision_log(state: GraphRAGState) -> dict:
    """取得路由決策的摘要，用於日誌記錄。"""
    return {
        "guard_blocked": state.get("guard_blocked", False),
        "acl_denied": state.get("acl_denied", False),
        "query_mode": state.get("query_mode", ""),
        "groundedness_status": state.get("groundedness_status", ""),
        "groundedness_score": state.get("groundedness_score", 0.0),
        "hitl_required": state.get("hitl_required", False),
        "hitl_resolved": state.get("hitl_resolved", False),
        "has_results": route_after_retrieval(state) == "has_results",
    }
