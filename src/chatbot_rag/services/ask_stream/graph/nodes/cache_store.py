"""
Cache Store 節點：儲存回答到語意快取。

功能：
- 將生成的回答儲存到語意快取
- 跳過快取命中、followup 對話、超出範圍等情況

前置節點：response_synth
後續節點：telemetry
"""

import re
from typing import Any, Callable, Dict, List, cast

from langgraph.config import get_stream_writer
from loguru import logger

from chatbot_rag.core.config import settings
from chatbot_rag.llm import State
from chatbot_rag.services.semantic_cache_service import semantic_cache_service
from ...constants import AskStreamStages
from ...events import emit_node_event


def build_cache_store_node() -> Callable[[State], State]:
    """
    建立 Cache Store 節點。

    此節點在 response_synth 之後執行，將生成的回答儲存到語意快取。
    會跳過以下情況：
    - 快取已停用
    - skip_cache_store 標記為 True（如快取命中或 followup 對話）
    - 是超出範圍的回答
    - 是 followup 類型的對話
    """

    def cache_store_node(state: State) -> State:
        writer = get_stream_writer()

        emit_node_event(
            writer,
            node="cache_store",
            phase="cache",
            payload={
                "channel": "status",
                "stage": AskStreamStages.CACHE_STORE_START,
            },
        )

        new_state = cast(State, dict(state))

        # 檢查是否應該跳過儲存
        skip_reason = _should_skip_store(state)
        if skip_reason:
            logger.debug(f"[CacheStore] Skipping cache store: {skip_reason}")
            emit_node_event(
                writer,
                node="cache_store",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_STORE_SKIP,
                    "reason": skip_reason,
                },
            )
            emit_node_event(
                writer,
                node="cache_store",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_STORE_DONE,
                    "stored": False,
                },
            )
            return new_state

        # 取得需要儲存的資料
        # 優先使用正規化問題作為主要快取鍵，同時保留原始問題用於雙向量儲存
        normalized_question = state.get("normalized_question") or ""
        original_question = state.get("latest_question") or ""
        question = normalized_question or original_question  # 主要快取問題
        answer = state.get("final_answer") or ""
        answer_meta = state.get("response_meta") or {}
        user_language = state.get("user_language") or "zh-hant"
        intent = state.get("intent") or ""

        if not question or not answer:
            logger.debug("[CacheStore] No question or answer to store")
            emit_node_event(
                writer,
                node="cache_store",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_STORE_SKIP,
                    "reason": "no_content",
                },
            )
            emit_node_event(
                writer,
                node="cache_store",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_STORE_DONE,
                    "stored": False,
                },
            )
            return new_state

        # 取得來源文件名稱（用於快取清除）
        source_filenames = _extract_source_filenames(state)

        # 儲存到快取（雙向量儲存：同時儲存正規化問題和原始問題）
        cache_id = semantic_cache_service.store(
            question=question,
            answer=answer,
            original_question=original_question if original_question != question else None,
            answer_meta=answer_meta,
            source_filenames=source_filenames,
            user_language=user_language,
            intent=intent,
        )

        if cache_id:
            logger.info(
                f"[CacheStore] Stored to cache: id={cache_id}, "
                f"sources={source_filenames}"
            )
            new_state["cache_id"] = cache_id
            new_state["source_filenames"] = source_filenames

            emit_node_event(
                writer,
                node="cache_store",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_STORE_DONE,
                    "stored": True,
                    "cache_id": cache_id,
                },
            )
        else:
            logger.warning("[CacheStore] Failed to store to cache")
            emit_node_event(
                writer,
                node="cache_store",
                phase="cache",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CACHE_STORE_DONE,
                    "stored": False,
                    "reason": "store_failed",
                },
            )

        return new_state

    return cache_store_node


def _should_skip_store(state: State) -> str:
    """
    檢查是否應該跳過快取儲存。

    Returns:
        str: 跳過原因，如果應該儲存則返回空字串
    """
    # 快取已停用
    if not settings.semantic_cache_enabled:
        return "disabled"

    # 明確標記跳過
    if state.get("skip_cache_store"):
        return "skip_flag"

    # 超出範圍的回答
    if state.get("is_out_of_scope"):
        return "out_of_scope"

    # Followup 類型的對話（依賴上下文，不適合快取）
    task_type = state.get("task_type") or ""
    if task_type == "conversation_followup":
        return "followup"

    # 快取已命中（不需要重複儲存）
    if state.get("cache_hit"):
        return "already_cached"

    return ""


def _extract_source_filenames(state: State) -> List[str]:
    """
    從 state 中提取來源文件名稱。

    支援兩種格式：
    1. raw_chunks 為 dict 列表（帶 payload.filename）
    2. raw_chunks 為 string 列表（從格式化文字中解析 "來源: xxx.md"）

    Returns:
        List[str]: 來源文件名稱列表
    """
    filenames: List[str] = []

    # 從 retrieval 狀態中提取
    retrieval: Dict[str, Any] = state.get("retrieval") or {}
    raw_chunks: List[Any] = retrieval.get("raw_chunks") or []

    for chunk in raw_chunks:
        if isinstance(chunk, dict):
            # 格式 1: dict 帶 payload
            payload = chunk.get("payload") or {}
            filename = payload.get("filename")
            if filename and filename not in filenames:
                filenames.append(filename)
        elif isinstance(chunk, str):
            # 格式 2: 格式化文字，解析 "(來源: xxx.md)" 或 "(來源: xxx)"
            # 匹配模式：(來源: filename) 或 (來源: filename 類型: xxx)
            matches = re.findall(r"\(來源:\s*([^)\s]+)", chunk)
            for match in matches:
                if match and match not in filenames:
                    filenames.append(match)

    # 也檢查 state 中是否已有 source_filenames
    existing = state.get("source_filenames") or []
    for fn in existing:
        if fn and fn not in filenames:
            filenames.append(fn)

    return filenames
