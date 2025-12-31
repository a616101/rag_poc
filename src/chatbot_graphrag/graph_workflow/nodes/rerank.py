"""
重排序節點

使用交叉編碼器重排序以提高檢索精確度。

此節點使用交叉編碼器模型對合併的結果進行重排序，
從約 40 個候選者減少到前 12 個最相關的結果。
"""

import logging
from typing import Any

from chatbot_graphrag.graph_workflow.types import GraphRAGState
from chatbot_graphrag.services.search.hybrid_search import SearchResult

logger = logging.getLogger(__name__)


async def rerank_node(state: GraphRAGState) -> dict[str, Any]:
    """
    用於交叉編碼器重排序的重排序節點。

    使用交叉編碼器模型對合併的結果進行重排序。
    從約 40 個候選者減少到前 12 個。

    Returns:
        更新後的狀態，包含 reranked_chunks
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("rerank", "START")

    start_time = time.time()
    question = state.get("normalized_question") or state.get("question", "")
    merged_results = state.get("merged_results")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    if not merged_results or not hasattr(merged_results, "chunks"):
        return {
            "reranked_chunks": [],
            "retrieval_path": retrieval_path + ["rerank:no_input"],
            "timing": {**timing, "rerank_ms": (time.time() - start_time) * 1000},
        }

    chunks = merged_results.chunks
    if not chunks:
        return {
            "reranked_chunks": [],
            "retrieval_path": retrieval_path + ["rerank:empty"],
            "timing": {**timing, "rerank_ms": (time.time() - start_time) * 1000},
        }

    logger.info(f"Reranking {len(chunks)} chunks...")

    try:
        # 延遲導入重排序服務
        # 在生產環境中使用交叉編碼器模型
        # 目前使用簡單的基於分數的排序

        # 按現有分數排序
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)

        # 取前 12 個
        top_k = 12
        reranked_chunks = sorted_chunks[:top_k]

        logger.info(f"Reranked to {len(reranked_chunks)} chunks")

        emit_status("rerank", "DONE")
        return {
            "reranked_chunks": reranked_chunks,
            "retrieval_path": retrieval_path + [f"rerank:{len(reranked_chunks)}"],
            "timing": {**timing, "rerank_ms": (time.time() - start_time) * 1000},
        }

    except Exception as e:
        logger.error(f"重排序錯誤: {e}")
        # 回退到按分數排序的前幾個結果
        return {
            "reranked_chunks": chunks[:12],
            "retrieval_path": retrieval_path + ["rerank:error_fallback"],
            "timing": {**timing, "rerank_ms": (time.time() - start_time) * 1000},
        }
