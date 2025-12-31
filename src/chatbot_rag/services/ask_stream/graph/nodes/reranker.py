"""
Reranker 節點：對檢索結果進行 Cross-Encoder 重新排序。

功能：
1. 對 tool_executor 取得的文件進行 Cross-Encoder reranking
2. 基於 settings.reranker_score_threshold 過濾低相關性結果
3. 將重新排序後的結果存入 retrieval.reranked_docs

前置節點：tool_executor
後續節點：result_evaluator

配置（透過 settings）：
- reranker_enabled: 是否啟用
- reranker_provider: 提供者 (cohere, voyageai, jina)
- reranker_model: 模型名稱
- reranker_score_threshold: 分數閾值
"""

from typing import Callable, List, Optional, cast

from langgraph.config import get_stream_writer
from langfuse import get_client
from loguru import logger

from chatbot_rag.core.config import settings
from chatbot_rag.llm import State
from chatbot_rag.services.reranker_service import reranker_service
from ...types import StateDict
from ...constants import AskStreamStages
from ...events import emit_node_event


def _get_langfuse_client():
    """安全取得 Langfuse client，避免初始化失敗影響主流程。"""
    try:
        return get_client()
    except Exception:
        return None


def build_reranker_node() -> Callable[[State], State]:
    """
    建立 Reranker 節點。

    此節點對 tool_executor 取得的文件進行 Cross-Encoder reranking，
    並基於分數閾值過濾低相關性結果。

    Returns:
        Callable: Reranker 節點函數
    """

    async def reranker_node(state: State) -> State:
        writer = get_stream_writer()
        emit_node_event(
            writer,
            node="reranker",
            phase="retrieval",
            payload={
                "channel": "status",
                "stage": AskStreamStages.RERANKER_START,
            },
        )

        state_dict = cast(StateDict, state)
        new_state = cast(State, dict(state))

        # 檢查是否啟用 reranker
        if not settings.reranker_enabled:
            logger.debug("[RERANKER_NODE] Reranker disabled, skipping")
            emit_node_event(
                writer,
                node="reranker",
                phase="retrieval",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.RERANKER_DONE,
                    "skipped": True,
                    "reason": "disabled",
                },
            )
            return new_state

        # 取得檢索狀態
        retrieval_state = dict(state_dict.get("retrieval") or {})
        raw_docs: List[dict] = retrieval_state.get("raw_docs") or []

        if not raw_docs:
            logger.debug("[RERANKER_NODE] No documents to rerank")
            emit_node_event(
                writer,
                node="reranker",
                phase="retrieval",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.RERANKER_DONE,
                    "skipped": True,
                    "reason": "no_documents",
                },
            )
            return new_state

        # 取得查詢文字
        query = retrieval_state.get("query") or state_dict.get("normalized_question") or ""
        if not query:
            logger.warning("[RERANKER_NODE] No query for reranking")
            emit_node_event(
                writer,
                node="reranker",
                phase="retrieval",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.RERANKER_DONE,
                    "skipped": True,
                    "reason": "no_query",
                },
            )
            return new_state

        # 執行 reranking（使用 Langfuse span 包裝）
        input_count = len(raw_docs)
        logger.info(
            "[RERANKER_NODE] Reranking {} documents with query: {}...",
            input_count,
            query[:50],
        )

        # 準備 Langfuse span 的 input
        langfuse = _get_langfuse_client()
        span_input = {
            "query": query[:200],
            "documents_count": input_count,
            "provider": settings.reranker_provider,
            "model": settings.reranker_model,
            "threshold": settings.reranker_score_threshold,
        }

        try:
            # 使用 Langfuse span 包裝 reranking 操作
            if langfuse:
                with langfuse.start_as_current_span(
                    name="reranker",
                    input=span_input,
                    metadata={
                        "provider": settings.reranker_provider,
                        "model": settings.reranker_model,
                    },
                ) as span:
                    reranked_docs = reranker_service.rerank(
                        query=query,
                        documents=raw_docs,
                        top_k=None,
                        score_threshold=settings.reranker_score_threshold,
                    )

                    output_count = len(reranked_docs)
                    filtered_count = input_count - output_count

                    # 計算分數統計
                    scores = [d.get("rerank_score", 0) for d in reranked_docs]
                    min_score = min(scores) if scores else 0
                    max_score = max(scores) if scores else 0

                    # 更新 span output
                    span.update(
                        output={
                            "output_count": output_count,
                            "filtered_count": filtered_count,
                            "min_score": round(min_score, 3),
                            "max_score": round(max_score, 3),
                            "scores_summary": [
                                {
                                    "filename": d.get("filename", "unknown")[:30],
                                    "score": round(d.get("rerank_score", 0), 3),
                                }
                                for d in reranked_docs[:10]
                            ],
                        },
                        metadata={
                            "output_count": output_count,
                            "filtered_count": filtered_count,
                        },
                    )
            else:
                # Langfuse 不可用時直接執行
                reranked_docs = reranker_service.rerank(
                    query=query,
                    documents=raw_docs,
                    top_k=None,
                    score_threshold=settings.reranker_score_threshold,
                )

                output_count = len(reranked_docs)
                filtered_count = input_count - output_count
                scores = [d.get("rerank_score", 0) for d in reranked_docs]
                min_score = min(scores) if scores else 0
                max_score = max(scores) if scores else 0

            logger.info(
                "[RERANKER_NODE] Reranked: {} -> {} docs (filtered={}, scores={:.3f}-{:.3f})",
                input_count,
                output_count,
                filtered_count,
                min_score,
                max_score,
            )

            # 更新 retrieval state
            retrieval_state["reranked_docs"] = reranked_docs
            retrieval_state["rerank_stats"] = {
                "input_count": input_count,
                "output_count": output_count,
                "filtered_count": filtered_count,
                "min_score": min_score,
                "max_score": max_score,
                "threshold": settings.reranker_score_threshold,
                "provider": settings.reranker_provider,
                "model": settings.reranker_model,
            }
            new_state["retrieval"] = retrieval_state

            emit_node_event(
                writer,
                node="reranker",
                phase="retrieval",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.RERANKER_DONE,
                    "input_count": input_count,
                    "output_count": output_count,
                    "filtered_count": filtered_count,
                    "min_score": round(min_score, 3),
                    "max_score": round(max_score, 3),
                    "threshold": settings.reranker_score_threshold,
                    "provider": settings.reranker_provider,
                    "model": settings.reranker_model,
                },
            )

            return new_state

        except Exception as exc:
            logger.error("[RERANKER_NODE] Reranking failed: {}", exc)
            emit_node_event(
                writer,
                node="reranker",
                phase="retrieval",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.RERANKER_ERROR,
                    "error": str(exc),
                },
            )
            # 失敗時保持原始文件順序
            retrieval_state["reranked_docs"] = raw_docs
            new_state["retrieval"] = retrieval_state
            return new_state

    return reranker_node
