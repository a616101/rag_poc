from typing import TypedDict, List, Dict, Any, Optional, Callable
from contextvars import ContextVar

import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, SystemMessage, HumanMessage
from langgraph.config import get_stream_writer


# Context variable for passing retrieval parameters to tools
# 用於在 Agent 執行工具時傳遞檢索參數（如 top_k）
retrieval_top_k_var: ContextVar[int] = ContextVar("retrieval_top_k", default=3)


class State(TypedDict, total=False):
    """
    LangGraph 狀態結構（Unified Agent 版本）。

    核心欄位：
    - messages: 對話訊息列表（Human/AI/System），包含原始對話歷史
    - search_query: 查詢重寫後，用於檢索的關鍵查詢字串
    - guard_blocked: 是否被守門員節點攔截（True 則後續 RAG/LLM 不再執行）
    - intent: 問題意圖（例如 simple_faq / form_download / form_export / out_of_scope 等）
    - is_out_of_scope: 是否為超出服務範圍的問題

    Agent 輸出欄位：
    - tool_summary: Agent 工具協調後產生的摘要文字（舊架構用）
    - context: RAG 檢索到的原始文件內容
    - used_tools: Agent 使用過的工具列表
    - agent_loops: Agent 執行的迴圈次數

    多語言支援欄位：
    - user_language: 用戶原始提問的語言（"zh-hant" / "zh-hans" / "en" 等）
                     用於決定最終回答的語言

    檢索參數欄位：
    - top_k: 檢索時返回的文檔數量（由 API 端點傳入，預設 3）

    評估專用欄位（Langfuse Evaluation）：
    - eval_question: 用戶的原始問題（穩定的 JsonPath）
    - eval_context: 檢索到的上下文（可能為空）
    - eval_answer: 最終的回答文字
    - eval_query_rewrite: Query Builder 重寫後的查詢（可能為空）

    語意快取欄位：
    - cache_hit: 是否命中語意快取
    - cache_id: 快取項目 ID（如果命中）
    - cache_score: 快取命中的相似度分數
    - cached_answer: 快取的回答內容
    - cached_answer_meta: 快取的回答 metadata
    - skip_cache_store: 是否跳過快取儲存（例如 followup 對話）
    - source_filenames: 來源文件名稱列表（用於快取清除）

    其餘欄位：
    - retry_count: 查詢重試次數（用於 Agentic RAG 檢索重試）
    """

    messages: List[BaseMessage]
    search_query: str
    guard_blocked: bool
    guard_response: str  # 屏東基督教醫院：Guard 攔截時的預設回應
    guard_intent: str    # 屏東基督教醫院：Guard 檢測到的意圖（如 privacy_inquiry）
    intent: str
    is_out_of_scope: bool
    tool_summary: str
    context: str
    used_tools: List[str]
    agent_loops: int
    user_language: str
    top_k: int
    retry_count: int
    # 評估專用欄位
    eval_question: str
    eval_context: str
    eval_answer: str
    eval_query_rewrite: str
    task_type: str
    should_retrieve: bool
    followup_detected: bool
    normalized_question: str
    latest_question: str
    prev_answer_normalized: str
    plan: Dict[str, Any]
    followup_instruction: str
    followup_ready: bool
    active_tool_calls: List[Dict[str, Any]]
    summary_search_query: str
    retrieval: Dict[str, Any]
    final_answer: str
    response_meta: Dict[str, Any]
    conversation_summary: str
    conversation_summary_enabled: bool
    # 語意快取欄位
    cache_hit: bool
    cache_id: Optional[str]
    cache_score: Optional[float]
    cached_answer: Optional[str]
    cached_answer_meta: Optional[Dict[str, Any]]
    skip_cache_store: bool
    source_filenames: List[str]
    # Multi-Query 欄位
    decomposed_queries: List[str]  # LLM 分解後的多個查詢
    query_decompose_reason: str  # 查詢分解原因說明
    # Intent analyzer 輸出欄位
    intent_output: Dict[str, Any]  # Intent analyzer 的完整輸出
    routing_hint: str  # 路由提示（continue/direct_response/followup）


def _build_optimized_messages(state: State) -> List[BaseMessage]:
    """
    根據 state 建立優化後的 messages，減少 token 消耗。

    策略：
    1. 如果有 tool_summary 或 context（表示 Agent 已處理過），只傳：
       - 全域 system prompt（messages[0]）
       - Agent 摘要（作為 SystemMessage）
       - 最新的使用者問題
    2. 如果沒有（例如 out_of_scope 直接跳過 Agent），則傳完整 messages
    """
    tool_summary = state.get("tool_summary") or ""
    context = state.get("context") or ""
    messages = state.get("messages", [])

    # 如果沒有 Agent 摘要，使用原始 messages
    if not tool_summary and not context:
        return messages

    # 有 Agent 摘要：建立精簡版 messages
    optimized: List[BaseMessage] = []

    # 1. 保留全域 system prompt（通常是 messages[0]）
    if messages and isinstance(messages[0], SystemMessage):
        optimized.append(messages[0])

    # 2. 加入 Agent 摘要作為 SystemMessage
    summary_parts: List[str] = []
    if tool_summary:
        summary_parts.append(
            "以下是工具協調 Agent 根據使用者問題與知識庫內容，"
            "整合出的重點資訊與建議回答摘要。你在最終回答時可以完整參考，"
            "並用更自然的語氣為使用者整理說明：\n\n"
            + tool_summary
        )
    if context:
        # context 通常已經很長，只在沒有 tool_summary 時才加入
        if not tool_summary:
            summary_parts.append(
                "以下是 RAG 檢索到的原始文件內容：\n\n" + context
            )

    if summary_parts:
        optimized.append(SystemMessage(content="\n\n".join(summary_parts)))

    # 3. 只保留最新的使用者問題
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            optimized.append(msg)
            break

    return optimized


def create_llm_node(llm: BaseChatModel) -> Callable[[State], Dict[str, Any]]:
    """
    建立一個 LangGraph LLM node（搭配 ResponsesChatModel 使用）。

    - 使用傳入的 `llm.stream`，逐 event 轉成 ChatGenerationChunk / AIMessageChunk
    - 依照 channel 把 token 推到 get_stream_writer()
    - 最後回傳一個完整 AIMessage（含完整答案 + reasoning + meta）

    優化：如果 state 中有 tool_summary，只傳必要資訊而非整個對話歷史。
    """

    def llm_node(state: State) -> Dict[str, Any]:
        writer = get_stream_writer()

        answer_tokens: List[str] = []
        reasoning_tokens: List[str] = []
        final_meta: Optional[Dict[str, Any]] = None

        # 狀態 flag
        reasoning_started = False
        reasoning_finished = False
        answer_started = False
        answer_finished = False

        # 使用優化後的 messages，減少 token 消耗
        optimized_messages = _build_optimized_messages(state)

        for chunk in llm.stream(optimized_messages):
            # 原本程式直接從 chunk 取 additional_kwargs，維持相同行為
            add = getattr(chunk, "additional_kwargs", None) or {}
            channel = add.get("channel")
            delta = add.get("delta") or ""

            # ✅ 仍然把 raw_event 往外丟，保留完整細節（前端 SSE 再決定要不要用）
            raw_event = add.get("raw_event")
            raw_type = add.get("raw_type")
            if raw_event is not None:
                writer(
                    {
                        "source": "llm_node",
                        "node": "model",
                        "phase": "generation",
                        "channel": "raw_event",
                        "raw_channel": raw_type or channel,
                        "delta": delta,
                        "event": raw_event,
                    }
                )

            # ------- 高階狀態：開始 reasoning / answer -------

            if channel == "reasoning" and not reasoning_started:
                reasoning_started = True
                writer(
                    {
                        "source": "llm_node",
                        "node": "model",
                        "phase": "generation",
                        "channel": "status",
                        "stage": "reasoning_start",
                    }
                )

            if channel == "output_text" and not answer_started:
                answer_started = True
                writer(
                    {
                        "source": "llm_node",
                        "node": "model",
                        "phase": "generation",
                        "channel": "status",
                        "stage": "answer_start",
                    }
                )

            # ------- 真正內容 channel -------

            if channel == "output_text":
                answer_tokens.append(delta)
                writer(
                    {
                        "source": "llm_node",
                        "node": "model",
                        "phase": "generation",
                        "channel": "answer",
                        "delta": delta,
                    }
                )

            elif channel == "reasoning":
                reasoning_tokens.append(delta)
                writer(
                    {
                        "source": "llm_node",
                        "node": "model",
                        "phase": "generation",
                        "channel": "reasoning",
                        "delta": delta,
                    }
                )

            elif channel == "reasoning_summary":
                writer(
                    {
                        "source": "llm_node",
                        "node": "model",
                        "phase": "generation",
                        "channel": "reasoning_summary",
                        "delta": delta,
                    }
                )

            elif channel == "meta":
                final_meta = add.get("responses_meta")
                writer(
                    {
                        "source": "llm_node",
                        "node": "model",
                        "phase": "generation",
                        "channel": "meta",
                        "meta": final_meta,
                    }
                )

            # ✅ 這裡用 done_for 來判斷 end 狀態
            elif channel == "done":
                done_for = add.get("done_for")

                if done_for == "reasoning" and reasoning_started and not reasoning_finished:
                    reasoning_finished = True
                    writer(
                        {
                            "source": "llm_node",
                            "node": "model",
                            "phase": "generation",
                            "channel": "status",
                            "stage": "reasoning_end",
                        }
                    )

                elif done_for == "output_text" and answer_started and not answer_finished:
                    answer_finished = True
                    writer(
                        {
                            "source": "llm_node",
                            "node": "model",
                            "phase": "generation",
                            "channel": "status",
                            "stage": "answer_end",
                        }
                    )

            else:
                if delta:
                    writer(
                        {
                            "source": "llm_node",
                            "node": "model",
                            "phase": "generation",
                            "channel": channel or "other",
                            "delta": delta,
                        }
                    )

        final_msg = AIMessage(
            content="".join(answer_tokens),
            additional_kwargs={
                "reasoning_text": "".join(reasoning_tokens),
                "responses_meta": final_meta,
            },
        )

        return {"messages": state["messages"] + [final_msg]}

    return llm_node


def create_chat_llm_node(llm: BaseChatModel) -> Callable[[State], Dict[str, Any]]:
    """
    建立一個 LangGraph LLM node（搭配 Chat Completions / ChatOpenAI 使用）。

    設計目標：
    - 繼續支援 streaming
    - 不再模擬 Responses 版的 reasoning channel 或「兩階段推理」
    - 只輸出：
        - channel=status（answer_start / answer_end）
        - channel=answer（最終答案全文逐字串流）
        - channel=meta（簡化後的 usage / 時間統計）

    優化：如果 state 中有 tool_summary，只傳必要資訊而非整個對話歷史。
    """

    def llm_node(state: State) -> Dict[str, Any]:
        """
        Chat Completions 單階段：
        - 直接根據 state["messages"] 產生最終答案
        - 不做額外「推理階段」呼叫，也不輸出 reasoning channel
        """

        writer = get_stream_writer()
        # 使用優化後的 messages，減少 token 消耗
        base_messages = _build_optimized_messages(state)

        tokens: List[str] = []
        started = False

        # 統計資訊（用來組 meta）
        started_at = time.monotonic()
        first_token_at: Optional[float] = None
        last_token_at: Optional[float] = None

        # 最後一個 chunk 的 usage_metadata（若後端有提供 usage）
        usage_metadata: Optional[Dict[str, Any]] = None

        # 單次 Chat Completions 串流呼叫，啟用 stream_usage 以取得 usage
        for chunk in llm.stream(base_messages, stream_usage=True):
            msg_chunk: Optional[AIMessageChunk] = None
            if isinstance(chunk, AIMessageChunk):
                msg_chunk = chunk
            else:
                msg_chunk = getattr(chunk, "message", None)

            if msg_chunk is None:
                continue

            # 嘗試從 chunk 中抓取 usage_metadata（只會在最後一個 chunk 出現）
            try:
                um = getattr(msg_chunk, "usage_metadata", None)
                if um:
                    usage_metadata = dict(um)  # type: ignore[arg-type]
            except (AttributeError, TypeError):
                pass

            delta = msg_chunk.content or ""
            if not isinstance(delta, str) or not delta:
                continue

            now = time.monotonic()
            if first_token_at is None:
                first_token_at = now
            last_token_at = now

            tokens.append(delta)

            if not started:
                started = True
                writer(
                    {
                        "source": "llm_node",
                        "node": "model",
                        "phase": "generation",
                        "channel": "status",
                        "stage": "answer_start",
                    }
                )

            writer(
                {
                    "source": "llm_node",
                    "node": "model",
                    "phase": "generation",
                    "channel": "answer",
                    "delta": delta,
                }
            )

        # 結束事件
        writer(
            {
                "source": "llm_node",
                "node": "model",
                "phase": "generation",
                "channel": "status",
                "stage": "answer_end",
            }
        )

        # 組合簡化版 meta（僅統計答案 channel）
        channel_meta = {
            "text": "".join(tokens),
            "char_count": sum(len(t) for t in tokens),
            "duration_ms": (
                (last_token_at - first_token_at) * 1000.0
                if first_token_at is not None and last_token_at is not None
                else None
            ),
        }

        meta: Dict[str, Any] = {
            "response_id": None,
            "usage": usage_metadata,
            "channels": {
                "output_text": channel_meta,
            },
        }

        # 發出一筆 meta 事件，包含完整答案與統計
        writer(
            {
                "source": "llm_node",
                "node": "model",
                "phase": "generation",
                "channel": "meta",
                "meta": meta,
            }
        )

        # 建立最終 AIMessage，這裡不再提供 reasoning_text（留空字串）
        answer_text = "".join(tokens)
        final_msg = AIMessage(
            content=answer_text,
            additional_kwargs={
                "reasoning_text": "",
                "responses_meta": meta,
            },
        )

        return {"messages": state["messages"] + [final_msg]}

    return llm_node


