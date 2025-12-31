"""
結果評估節點：檢查檢索結果並決定下一步行動。

功能：
1. 檢查檢索結果（來自 chunk_expander 的 expanded_docs）
2. 無結果時觸發 Smart Retry（使用 QueryVariationStrategy）
3. 有結果則設置 status="relevant" 進入回應生成

注意：
- Adaptive Chunk Expansion 已移至 chunk_expander 節點
- 此節點專注於「評估」和「決定是否重試」

路由：
- status="retry" → query_builder（重試檢索）
- status="relevant" → response_synth（進入回應生成）

前置節點：chunk_expander
後續節點：response_synth 或 query_builder（重試）
"""

from typing import Callable, List, cast

from langgraph.config import get_stream_writer
from loguru import logger

from chatbot_rag.llm import State
from chatbot_rag.services.query_variation import (
    QueryVariationStrategy,
    get_strategy_for_loop,
)
from ...types import StateDict
from ...constants import AskStreamStages
from ...events import emit_node_event


# 最大重試次數
MAX_RETRY_LOOPS = 2


def build_result_evaluator_node(
    base_llm_params: dict,
    *,
    agent_backend: str = "chat",
    prompt_service=None,
    domain_config=None,
    max_retry_loops: int = MAX_RETRY_LOOPS,
) -> Callable[[State], State]:
    """
    建立結果評估節點。

    主要功能：
    1. 檢查檢索結果是否存在（來自 chunk_expander 的 expanded_docs）
    2. 如果沒有結果，觸發 Smart Retry（使用查詢變異策略）
    3. 有結果則設置 status="relevant" 進入回應生成

    注意：Adaptive Chunk Expansion 已移至 chunk_expander 節點處理。

    Args:
        base_llm_params: LLM 參數（保留參數以維持介面一致性）
        agent_backend: Agent 後端類型（保留參數以維持介面一致性）
        prompt_service: Prompt 服務（保留參數以維持介面一致性）
        domain_config: 領域設定（保留參數以維持介面一致性）
        max_retry_loops: 最大重試次數

    Returns:
        結果評估節點函數
    """
    # 忽略未使用的參數（保留以維持介面一致性）
    _ = base_llm_params, agent_backend, prompt_service, domain_config

    async def result_evaluator_node(state: State) -> State:
        writer = get_stream_writer()
        emit_node_event(
            writer,
            node="result_evaluator",
            phase="evaluation",
            payload={
                "channel": "status",
                "stage": AskStreamStages.RESULT_EVALUATOR_START,
            },
        )

        state_dict = cast(StateDict, state)
        new_state = cast(State, dict(state))

        # 取得當前狀態
        retrieval_state = dict(state_dict.get("retrieval") or {})
        loop = retrieval_state.get("loop", 1)

        # 使用 chunk_expander 處理後的 expanded_docs，或 fallback 到 reranked_docs/raw_docs
        docs = (
            retrieval_state.get("expanded_docs")
            or retrieval_state.get("reranked_docs")
            or retrieval_state.get("raw_docs")
            or []
        )

        # ========== 檢查檢索結果是否存在 ==========

        if not docs:
            should_retry = loop < max_retry_loops
            stage = (
                AskStreamStages.RESULT_EVALUATOR_RETRY
                if should_retry
                else AskStreamStages.RESULT_EVALUATOR_NO_HITS
            )
            retrieval_state["status"] = "retry" if should_retry else "fallback"

            # Smart Retry: 根據下一輪 loop 選擇變異策略
            if should_retry:
                next_loop = loop + 1
                next_strategy = get_strategy_for_loop(next_loop)
                retrieval_state["next_strategy"] = next_strategy.value
                logger.info(
                    "[RESULT_EVALUATOR] No results, will retry with strategy={}",
                    next_strategy.value,
                )
            else:
                retrieval_state["next_strategy"] = QueryVariationStrategy.NONE.value
                logger.info("[RESULT_EVALUATOR] No results after {} loops, fallback", loop)

            new_state["retrieval"] = retrieval_state

            emit_node_event(
                writer,
                node="result_evaluator",
                phase="evaluation",
                payload={
                    "channel": "status",
                    "stage": stage,
                    "loop": loop,
                    "next_strategy": retrieval_state.get("next_strategy"),
                },
            )
            return new_state

        # ========== 有檢索結果，設置狀態進入回應生成 ==========

        # context 應該已經由 chunk_expander 設定
        # 這裡只需要更新 retrieval status
        retrieval_state["status"] = "relevant"
        new_state["retrieval"] = retrieval_state

        logger.info(
            "[RESULT_EVALUATOR] Found {} docs, proceeding to response",
            len(docs),
        )

        emit_node_event(
            writer,
            node="result_evaluator",
            phase="evaluation",
            payload={
                "channel": "status",
                "stage": AskStreamStages.RESULT_EVALUATOR_DONE,
                "documents_count": len(docs),
            },
        )
        return new_state

    return result_evaluator_node
