"""
品質節點

落地性檢查和重試邏輯。
實現第 4 階段：Ragas 整合與 10% 抽樣。

主要節點：
- groundedness_node: 評估證據是否足以支持回答
- targeted_retry_node: 智能重試策略
- interrupt_hitl_node: HITL 中斷節點
"""

import logging
from typing import Any, Optional

from chatbot_graphrag.graph_workflow.types import (
    GraphRAGState,
    GroundednessStatus,
)

logger = logging.getLogger(__name__)


# 延遲載入的 Ragas 評估器
_ragas_evaluator: Optional[Any] = None


def _get_ragas_evaluator():
    """取得延遲載入的 Ragas 評估器實例。"""
    global _ragas_evaluator
    if _ragas_evaluator is None:
        try:
            from chatbot_graphrag.services.evaluation import get_ragas_evaluator
            _ragas_evaluator = get_ragas_evaluator()
        except ImportError:
            logger.warning("Ragas evaluator not available")
            _ragas_evaluator = False  # Mark as unavailable
    return _ragas_evaluator if _ragas_evaluator else None


async def groundedness_node(state: GraphRAGState) -> dict[str, Any]:
    """
    落地性節點。

    評估上下文是否提供足夠的基礎來支持回答。
    第 4 階段：整合 Ragas 評估與 10% 抽樣。

    Returns:
        更新後的狀態，包含 groundedness_status、groundedness_score 和 ragas_metrics
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("groundedness", "START")

    start_time = time.time()
    evidence_table = state.get("evidence_table", [])
    context_tokens = state.get("context_tokens", 0)
    context_text = state.get("context_text", "")
    question = state.get("normalized_question") or state.get("question", "")
    draft_answer = state.get("draft_answer", "")  # From response generation
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    logger.debug(f"Evaluating groundedness: {len(evidence_table)} evidence items...")

    # ==================== 啟發式評估 ====================
    # 計算平均相關性分數（快速，總是執行）
    if evidence_table:
        avg_score = sum(e.relevance_score for e in evidence_table) / len(evidence_table)
        max_score = max(e.relevance_score for e in evidence_table)
    else:
        avg_score = 0.0
        max_score = 0.0

    # 從啟發式方法決定落地性狀態
    if len(evidence_table) >= 3 and avg_score >= 0.5:
        status = GroundednessStatus.PASS
        score = min(1.0, (avg_score + max_score) / 2)
    elif len(evidence_table) >= 1 and max_score >= 0.4:
        status = GroundednessStatus.RETRY
        score = max_score
    else:
        if context_tokens > 0:
            status = GroundednessStatus.NEEDS_REVIEW
            score = avg_score
        else:
            status = GroundednessStatus.RETRY
            score = 0.0

    # ==================== Ragas 評估（10% 抽樣）====================
    ragas_metrics: dict[str, Any] = {}
    ragas_sampled = False

    evaluator = _get_ragas_evaluator()
    if evaluator and draft_answer and context_text:
        try:
            from chatbot_graphrag.services.evaluation import (
                EvaluationSample,
                should_sample_for_evaluation,
            )

            # 檢查此請求是否應該被抽樣
            if should_sample_for_evaluation():
                ragas_sampled = True

                # 從證據表準備上下文列表
                contexts = []
                for evidence in evidence_table[:10]:  # Limit to top 10
                    if hasattr(evidence, "content"):
                        contexts.append(evidence.content[:1000])  # Limit content length

                # 如果沒有證據表，使用原始上下文
                if not contexts and context_text:
                    contexts = [context_text[:3000]]

                sample = EvaluationSample(
                    question=question,
                    answer=draft_answer,
                    contexts=contexts,
                )

                result = await evaluator.evaluate(sample)

                if result.evaluated:
                    ragas_metrics = result.to_dict()

                    # 如果可用則使用 Ragas faithfulness 影響最終分數
                    if result.faithfulness is not None:
                        # 混合啟發式分數與 Ragas faithfulness
                        # 60% 啟發式，40% Ragas 以保持穩定性
                        blended_score = (score * 0.6) + (result.faithfulness * 0.4)
                        score = blended_score

                        # 如果 Ragas 指示問題則覆蓋狀態
                        if result.faithfulness < 0.3 and status == GroundednessStatus.PASS:
                            status = GroundednessStatus.NEEDS_REVIEW
                            logger.info(
                                f"Ragas faithfulness {result.faithfulness:.2f} triggered review"
                            )

                    logger.info(
                        f"Ragas evaluation: faithfulness={result.faithfulness}, "
                        f"relevancy={result.answer_relevancy}, "
                        f"precision={result.context_precision}"
                    )

        except Exception as e:
            logger.warning(f"Ragas evaluation failed: {e}")
            ragas_metrics = {"error": str(e)}

    logger.info(
        f"Groundedness: {status.value} (score={score:.2f}, "
        f"evidence={len(evidence_table)}, ragas_sampled={ragas_sampled})"
    )

    emit_status("groundedness", "DONE")
    return {
        "groundedness_status": status.value,
        "groundedness_score": score,
        "ragas_metrics": ragas_metrics,
        "ragas_sampled": ragas_sampled,
        "retrieval_path": retrieval_path + [f"groundedness:{status.value}:{score:.2f}"],
        "timing": {**timing, "groundedness_ms": (time.time() - start_time) * 1000},
    }


class RetryStrategy(str):
    """智能重試的重試策略類型。"""

    EXPAND_QUERY = "expand_query"  # 為查詢添加更多上下文/細節
    SIMPLIFY_QUERY = "simplify_query"  # 簡化複雜查詢
    KEYWORD_FOCUS = "keyword_focus"  # 專注於關鍵實體/術語
    BROADEN_SCOPE = "broaden_scope"  # 移除限制性過濾器
    ALTERNATIVE_PHRASING = "alternative_phrasing"  # 重新措辭問題


def _analyze_failure_type(state: GraphRAGState) -> tuple[str, str]:
    """
    分析導致低落地性的原因並決定重試策略。

    第 5 階段：智能失敗分析。

    Returns:
        (failure_type, retry_strategy) 元組
    """
    evidence_table = state.get("evidence_table", [])
    groundedness_score = state.get("groundedness_score", 0.0)
    context_tokens = state.get("context_tokens", 0)
    reranked_chunks = state.get("reranked_chunks", [])

    # 分析失敗模式
    if not evidence_table and not reranked_chunks:
        # 完全沒有結果 - 召回問題
        return "recall_low", RetryStrategy.BROADEN_SCOPE

    if evidence_table:
        avg_relevance = sum(e.relevance_score for e in evidence_table) / len(evidence_table)
        max_relevance = max(e.relevance_score for e in evidence_table)

        if max_relevance < 0.3:
            # 所有結果都是低品質 - 查詢不匹配
            return "query_mismatch", RetryStrategy.ALTERNATIVE_PHRASING

        if avg_relevance < 0.4 and max_relevance > 0.5:
            # 一些好的結果被噪音埋沒
            return "noise_high", RetryStrategy.KEYWORD_FOCUS

        if len(evidence_table) < 2 and avg_relevance > 0.5:
            # 結果太少但相關
            return "insufficient_coverage", RetryStrategy.EXPAND_QUERY

    if context_tokens > 8000 and groundedness_score < 0.5:
        # 大量上下文但低落地性 - 上下文有噪音
        return "context_diluted", RetryStrategy.KEYWORD_FOCUS

    # 預設為擴展查詢
    return "general_low_score", RetryStrategy.EXPAND_QUERY


def _apply_retry_strategy(
    question: str,
    strategy: str,
    user_language: str = "zh-TW",
) -> str:
    """
    應用重試策略來修改查詢。

    第 5 階段：策略特定的查詢修改。

    Args:
        question: 原始問題
        strategy: 要應用的重試策略
        user_language: 用於修改的使用者語言

    Returns:
        修改後的查詢
    """
    is_chinese = user_language.startswith("zh")

    if strategy == RetryStrategy.EXPAND_QUERY:
        # 添加詳細請求
        if is_chinese:
            return f"{question} 請提供詳細說明和相關資訊"
        return f"{question} Please provide detailed explanation and related information"

    elif strategy == RetryStrategy.SIMPLIFY_QUERY:
        # 抽取核心問題（簡單啟發式）
        # 移除括號內容和修飾詞
        import re
        simplified = re.sub(r"\([^)]*\)", "", question)
        simplified = re.sub(r"（[^）]*）", "", simplified)
        simplified = re.sub(r"(請問|可以|能不能|麻煩)", "", simplified)
        return simplified.strip()

    elif strategy == RetryStrategy.KEYWORD_FOCUS:
        # 專注於關鍵實體（抽取名詞/實體）
        if is_chinese:
            return f"關鍵詞：{question.replace('？', '').replace('?', '')}"
        return f"Key terms: {question.replace('?', '')}"

    elif strategy == RetryStrategy.BROADEN_SCOPE:
        # 請求更廣泛的搜尋
        if is_chinese:
            return f"關於 {question} 的所有相關資訊"
        return f"All information related to: {question}"

    elif strategy == RetryStrategy.ALTERNATIVE_PHRASING:
        # 請求替代措辭
        if is_chinese:
            return f"換個方式問：{question}"
        return f"In other words: {question}"

    # 預設
    return question


async def targeted_retry_node(state: GraphRAGState) -> dict[str, Any]:
    """
    針對性重試節點。

    第 5 階段：根據失敗分析實現智能重試策略。

    分析落地性低的原因並應用適當的重試策略：
    - recall_low: 擴大範圍以找到更多結果
    - noise_high: 專注於關鍵詞以過濾噪音
    - query_mismatch: 嘗試替代措辭
    - insufficient_coverage: 擴展查詢以獲取更多上下文
    - context_diluted: 專注於關鍵術語

    Returns:
        更新後的狀態，包含重試資訊和修改後的查詢
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("targeted_retry", "START")

    start_time = time.time()
    question = state.get("normalized_question") or state.get("question", "")
    user_language = state.get("user_language", "zh-TW")
    evidence_table = state.get("evidence_table", [])
    budget = state.get("budget")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    logger.info("Targeted retry triggered...")

    # 增加循環計數器
    if budget and hasattr(budget, "increment_loop"):
        budget.increment_loop()
    elif budget and hasattr(budget, "use_loop"):
        budget.use_loop()

    # 第 5 階段：分析失敗並選擇策略
    failure_type, retry_strategy = _analyze_failure_type(state)
    retry_reason = f"Failure type: {failure_type}, Strategy: {retry_strategy}"

    # 應用重試策略來修改查詢
    modified_query = _apply_retry_strategy(question, retry_strategy, user_language)

    logger.info(
        f"Retry analysis: {failure_type} -> {retry_strategy}, "
        f"query: '{question[:30]}...' -> '{modified_query[:30]}...'"
    )

    emit_status("targeted_retry", "DONE")
    return {
        "retry_reason": retry_reason,
        "retry_failure_type": failure_type,
        "retry_strategy": retry_strategy,
        "normalized_question": modified_query,
        "retrieval_path": retrieval_path + [f"targeted_retry:{retry_strategy}"],
        "timing": {**timing, "targeted_retry_ms": (time.time() - start_time) * 1000},
        "budget": budget,
    }


async def interrupt_hitl_node(state: GraphRAGState) -> dict[str, Any]:
    """
    人機協作（HITL）中斷節點。

    第 3 階段：標記狀態以進行人工審核，支援逾時處理。

    實現優雅的逾時處理：
    - 記錄 HITL 觸發時間
    - 根據設定設置逾時期限
    - 追蹤審核原因以供稽核

    Returns:
        更新後的狀態，包含 HITL 標誌和逾時資訊
    """
    import time
    from datetime import datetime, timezone

    from chatbot_graphrag.core.config import settings
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("interrupt_hitl", "START")

    start_time = time.time()
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))
    groundedness_score = state.get("groundedness_score", 0.0)

    # 決定 HITL 的原因
    needs_review_reason = "Low groundedness score"
    if groundedness_score < 0.3:
        needs_review_reason = f"Very low groundedness ({groundedness_score:.2f})"
    elif groundedness_score < 0.5:
        needs_review_reason = f"Insufficient evidence quality ({groundedness_score:.2f})"
    elif state.get("error"):
        needs_review_reason = f"Error during processing: {state.get('error')}"

    # 計算逾時期限
    now = datetime.now(timezone.utc)
    timeout_seconds = settings.hitl_timeout_seconds
    timeout_at = now.timestamp() + timeout_seconds

    logger.info(
        f"HITL interrupt triggered - awaiting human review. "
        f"Reason: {needs_review_reason}, Timeout in {timeout_seconds}s"
    )

    emit_status("interrupt_hitl", "DONE")
    return {
        "hitl_required": True,
        "hitl_resolved": False,
        "hitl_triggered_at": now.isoformat(),
        "hitl_timeout_at": timeout_at,
        "hitl_timeout_seconds": timeout_seconds,
        "needs_review_reason": needs_review_reason,
        "needs_human_review": True,
        "retrieval_path": retrieval_path + ["hitl:interrupt"],
        "timing": {**timing, "interrupt_hitl_ms": (time.time() - start_time) * 1000},
    }


def check_hitl_timeout(state: GraphRAGState) -> bool:
    """
    檢查 HITL 逾時是否已超過。

    第 3 階段：用於路由以決定是否應該繞過 HITL。

    Returns:
        如果逾時已超過則為 True，否則為 False
    """
    import time

    timeout_at = state.get("hitl_timeout_at")
    if not timeout_at:
        return False

    current_time = time.time()
    if current_time > timeout_at:
        logger.warning(
            f"HITL timeout exceeded: current={current_time:.0f}, deadline={timeout_at:.0f}"
        )
        return True

    return False


def get_hitl_fallback_response(state: GraphRAGState) -> str:
    """
    當 HITL 逾時時生成降級回應。

    第 3 階段：當人工審核不可用時提供優雅降級。

    Returns:
        降級回答文字
    """
    user_language = state.get("user_language", "zh-TW")

    if user_language.startswith("zh"):
        return (
            "很抱歉，由於系統繁忙，您的問題需要更多時間處理。"
            "請稍後再試，或聯繫客服人員獲取協助。"
        )
    else:
        return (
            "We apologize, but your question requires additional processing time. "
            "Please try again later or contact support for assistance."
        )
