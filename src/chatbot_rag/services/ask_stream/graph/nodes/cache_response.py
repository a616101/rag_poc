"""
Cache Response 節點：串流返回快取的回答。

功能：
- 模擬 streaming 方式輸出快取回答（維持一致的用戶體驗）
- 輸出 SSE answer 和 meta 事件

前置節點：cache_lookup（快取命中時）
後續節點：telemetry
"""

import time
from typing import Any, Callable, Dict, List, cast

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from loguru import logger

from chatbot_rag.llm import State
from ...constants import AskStreamStages
from ...events import emit_node_event


# 模擬 streaming 的每次輸出字元數和延遲
STREAMING_CHUNK_SIZE = 10  # 每次輸出的字元數
STREAMING_DELAY_MS = 5  # 每次輸出的延遲（毫秒）


def build_cache_response_node() -> Callable[[State], State]:
    """
    建立 Cache Response 節點。

    此節點在快取命中時執行，模擬 streaming 方式輸出快取的回答，
    以維持與正常回答一致的用戶體驗。
    """

    def cache_response_node(state: State) -> State:
        writer = get_stream_writer()

        emit_node_event(
            writer,
            node="cache_response",
            phase="generation",
            payload={
                "channel": "status",
                "stage": AskStreamStages.CACHE_RESPONSE_START,
            },
        )

        cached_answer = state.get("cached_answer") or ""
        cached_meta = state.get("cached_answer_meta") or {}
        cache_id = state.get("cache_id")
        cache_score = state.get("cache_score")

        logger.info(
            f"[CacheResponse] Streaming cached answer, length={len(cached_answer)}"
        )

        # 模擬 streaming 輸出
        answer_tokens: List[str] = []
        start_time = time.monotonic()

        # 將答案分成小塊輸出
        for i in range(0, len(cached_answer), STREAMING_CHUNK_SIZE):
            chunk = cached_answer[i : i + STREAMING_CHUNK_SIZE]
            answer_tokens.append(chunk)

            writer(
                {
                    "source": "cache_response",
                    "node": "cache_response",
                    "phase": "generation",
                    "channel": "answer",
                    "delta": chunk,
                }
            )

            # 短暫延遲以模擬串流效果
            if STREAMING_DELAY_MS > 0:
                time.sleep(STREAMING_DELAY_MS / 1000.0)

        duration_ms = (time.monotonic() - start_time) * 1000.0

        # 構建 meta 資訊
        meta_payload: Dict[str, Any] = {
            "from_cache": True,
            "cache_id": cache_id,
            "cache_score": cache_score,
            "usage": None,  # 快取命中不消耗 token
            "channels": {
                "output_text": {
                    "text": cached_answer,
                    "char_count": len(cached_answer),
                    "duration_ms": duration_ms,
                }
            },
        }

        # 合併原始快取的 meta
        if cached_meta:
            meta_payload["original_meta"] = cached_meta

        writer(
            {
                "source": "cache_response",
                "node": "cache_response",
                "phase": "generation",
                "channel": "meta",
                "meta": meta_payload,
            }
        )

        emit_node_event(
            writer,
            node="cache_response",
            phase="generation",
            payload={
                "channel": "status",
                "stage": AskStreamStages.CACHE_RESPONSE_DONE,
                "from_cache": True,
                "cache_id": cache_id,
            },
        )

        # 更新 state
        new_state = cast(State, dict(state))

        # 建立 AIMessage
        final_msg = AIMessage(
            content=cached_answer,
            additional_kwargs={
                "from_cache": True,
                "cache_id": cache_id,
                "cache_score": cache_score,
                "responses_meta": meta_payload,
            },
        )

        new_state["messages"] = list(state.get("messages", [])) + [final_msg]
        new_state["final_answer"] = cached_answer
        new_state["response_meta"] = meta_payload
        new_state["skip_cache_store"] = True  # 確保不會重複儲存

        # 設定評估欄位
        question = state.get("latest_question") or state.get("normalized_question") or ""
        new_state["eval_question"] = question
        new_state["eval_answer"] = cached_answer
        new_state["eval_context"] = ""  # 快取命中時沒有檢索上下文

        return new_state

    return cache_response_node
