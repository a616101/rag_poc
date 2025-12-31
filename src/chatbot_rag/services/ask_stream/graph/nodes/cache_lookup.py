"""
Cache Lookup 節點：查詢語意快取。

功能：
- 使用正規化問題檢查語意快取
- 找到相似度超過閾值的快取則設定 cache_hit=True

前置節點：language_normalizer
後續節點：cache_response（命中）| intent_analyzer（未命中）
"""

from typing import Callable, cast

from langgraph.config import get_stream_writer
from loguru import logger

from chatbot_rag.core.config import settings
from chatbot_rag.llm import State
from chatbot_rag.services.semantic_cache_service import semantic_cache_service
from ...constants import AskStreamStages
from ...events import emit_node_event


def build_cache_lookup_node() -> Callable[[State], State]:
    """
    建立 Cache Lookup 節點。

    此節點在 language_normalizer 之後執行，使用正規化問題檢查語意快取。
    這樣可以提高快取命中率，因為不同表達方式的問題會被正規化為相似形式。
    如果找到相似度超過閾值的快取，設定 cache_hit=True 並填入快取資料。
    """

    def cache_lookup_node(state: State) -> State:
        writer = get_stream_writer()

        emit_node_event(
            writer,
            node="cache_lookup",
            phase="cache",
            payload={
                "channel": "status",
                "stage": AskStreamStages.CACHE_LOOKUP_START,
            },
        )

        new_state = cast(State, dict(state))

        # 預設值
        new_state["cache_hit"] = False
        new_state["cache_id"] = None
        new_state["cache_score"] = None
        new_state["cached_answer"] = None
        new_state["cached_answer_meta"] = None

        # 檢查是否啟用快取
        if not settings.semantic_cache_enabled:
            logger.debug("[CacheLookup] Semantic cache is disabled")
            emit_node_event(
                writer,
                node="cache_lookup",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_MISS,
                    "reason": "disabled",
                },
            )
            emit_node_event(
                writer,
                node="cache_lookup",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_LOOKUP_DONE,
                    "cache_hit": False,
                },
            )
            return new_state

        # 取得問題（優先使用正規化問題，提高快取命中率）
        # 流程：guard → language_normalizer → cache_lookup
        # 此時 normalized_question 應該已經由 language_normalizer 設定
        question = state.get("normalized_question") or state.get("latest_question") or ""
        if not question:
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.type == "human":
                    content = msg.content
                    if isinstance(content, str):
                        question = content
                        break

        if not question:
            logger.debug("[CacheLookup] No question found in state")
            emit_node_event(
                writer,
                node="cache_lookup",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_MISS,
                    "reason": "no_question",
                },
            )
            emit_node_event(
                writer,
                node="cache_lookup",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_LOOKUP_DONE,
                    "cache_hit": False,
                },
            )
            return new_state

        # 查詢快取
        user_language = state.get("user_language")
        cache_result = semantic_cache_service.lookup(
            question=question,
            user_language=user_language,
        )

        if cache_result:
            # 快取命中
            new_state["cache_hit"] = True
            new_state["cache_id"] = cache_result["cache_id"]
            new_state["cache_score"] = cache_result["score"]
            new_state["cached_answer"] = cache_result["answer"]
            new_state["cached_answer_meta"] = cache_result["answer_meta"]
            new_state["skip_cache_store"] = True  # 不需要再儲存

            logger.info(
                f"[CacheLookup] Cache hit! score={cache_result['score']:.4f}, "
                f"cache_id={cache_result['cache_id']}"
            )

            emit_node_event(
                writer,
                node="cache_lookup",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_HIT,
                    "cache_id": cache_result["cache_id"],
                    "score": cache_result["score"],
                },
            )
        else:
            # 快取未命中
            logger.debug("[CacheLookup] Cache miss")
            emit_node_event(
                writer,
                node="cache_lookup",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_MISS,
                    "reason": "no_match",
                },
            )

        emit_node_event(
            writer,
            node="cache_lookup",
            phase="cache",
            payload={
                "channel": "status",
                "stage": AskStreamStages.CACHE_LOOKUP_DONE,
                "cache_hit": new_state["cache_hit"],
            },
        )

        return new_state

    return cache_lookup_node
