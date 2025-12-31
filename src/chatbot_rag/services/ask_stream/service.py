"""
Unified Agent 串流服務（/ask/stream 與 /ask/stream_chat 的核心邏輯）。

此模組實作 Agentic RAG 流程，整合以下節點：
- guard: 輸入驗證與安全檢查
- language_normalizer: 語言偵測與問題標準化
- cache_lookup / cache_response: 語意快取查詢
- intent_analyzer: LLM 意圖分析（決定是否需要檢索）
- query_builder: 多查詢分解與重寫
- tool_executor: 並行檢索執行
- reranker: Cross-encoder 重新排序
- result_evaluator: 自適應擴展與重試邏輯
- response_synth: LLM 回答生成（SSE 串流）
- cache_store: 語意快取儲存
- telemetry: Langfuse 追蹤更新

SSE 事件供 FastAPI 路由直接串流給前端。
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, List, Optional, cast
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from loguru import logger

from chatbot_rag.core.config import settings
from chatbot_rag.core.concurrency import (
    RequestContext,
    request_context_var,
    llm_concurrency,
)
from chatbot_rag.llm import State
from chatbot_rag.models.rag import LLMConfig, QuestionRequest
from chatbot_rag.services.language_utils import detect_preferred_language
from chatbot_rag.services.prompt_service import PromptService

from .types import StateDict
from .constants import AskStreamStages, LLM_STREAM_DEBUG
from .graph import build_ask_graph
from .tracing import create_trace_context


def resolve_llm_config(config: Optional[LLMConfig]) -> dict:
    """將 QuestionRequest.llm_config 解析為實際要使用的 LLM 參數。"""
    if not config:
        return {}

    resolved: dict = {}
    if config.model is not None:
        resolved["model"] = config.model
    if config.reasoning_effort is not None:
        resolved["reasoning_effort"] = config.reasoning_effort
    if config.reasoning_summary is not None:
        resolved["reasoning_summary"] = config.reasoning_summary
    return resolved


def build_initial_stream_state(request: QuestionRequest) -> State:
    """建立 Unified Agent 架構的初始狀態。"""
    init_messages: List[BaseMessage] = []
    if request.conversation_history:
        for msg in request.conversation_history:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)
            if not content:
                continue
            if role == "user":
                init_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                init_messages.append(AIMessage(content=content))
    init_messages.append(HumanMessage(content=request.question))
    detected_language = detect_preferred_language(request.question)
    summary_enabled = bool(getattr(request, "enable_conversation_summary", True))
    conversation_summary = (
        (request.conversation_summary or "").strip() if summary_enabled else ""
    )

    base_state: StateDict = {
        "messages": init_messages,
        "retry_count": 0,
        "user_language": detected_language,
        "top_k": request.top_k,
        "used_tools": [],
        "retrieval": {"loop": 0, "raw_chunks": [], "status": None},
        "intent": "",
        "followup_instruction": "",
        "conversation_summary": conversation_summary,
        "conversation_summary_enabled": summary_enabled,
    }
    return cast(State, base_state)


async def run_stream_graph(
    request: QuestionRequest,
    is_disconnected: Callable[[], Awaitable[bool]],
    *,
    agent_backend: str,
    extra_tags: Optional[list[str]] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> AsyncIterator[dict]:
    """執行 Unified Agent Graph 並串流事件。

    Args:
        request: 問題請求
        is_disconnected: 檢查客戶端是否斷線的函數
        agent_backend: 使用的 agent 後端 (chat/responses)
        extra_tags: 額外的 Langfuse 標籤（會與預設標籤合併）
        extra_metadata: 額外的 Langfuse metadata（會與預設 metadata 合併）
    """
    init_state: State = build_initial_stream_state(request)
    request_id = uuid.uuid4().hex[:8]
    conversation_summary_enabled = bool(
        getattr(request, "enable_conversation_summary", True)
    )

    # 建立並設定請求上下文（用於優先級排序）
    request_ctx = RequestContext(request_id=request_id)
    context_token = request_context_var.set(request_ctx)
    llm_concurrency.register_request(request_ctx)

    architecture = (
        "unified_agent__responses" if agent_backend == "responses" else "unified_agent_"
    )
    summary: dict[str, Any] = {
        "request_id": request_id,
        "question": request.question,
        "search_query": None,
        "guard_blocked": None,
        "is_out_of_scope": None,
        "intent": None,
        "agent_loops": None,
        "agent_used_tools": [],
        "architecture": architecture,
        "conversation_summary": (
            request.conversation_summary if conversation_summary_enabled else None
        ),
    }

    total_usage: dict[str, int] = {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    base_llm_params = resolve_llm_config(request.llm_config)

    # 初始化 PromptService（如果啟用）
    prompt_service: Optional[PromptService] = None
    if settings.langfuse_prompt_enabled:
        prompt_service = PromptService(
            default_label=settings.langfuse_prompt_label,
            cache_ttl_seconds=settings.langfuse_prompt_cache_ttl,
        )

    graph = build_ask_graph(
        base_llm_params=base_llm_params,
        agent_backend=agent_backend,
        prompt_service=prompt_service,
    )

    client_disconnected = False
    trace_id: str = ""

    # 合併標籤與 metadata
    base_tags = [architecture, f"request:{request_id}"]
    if extra_tags:
        base_tags.extend(extra_tags)

    base_metadata = {
        "top_k": request.top_k,
        "agent_backend": agent_backend,
    }
    if extra_metadata:
        base_metadata.update(extra_metadata)

    try:
        with create_trace_context(
            name="ask-stream-workflow",
            trace_id_seed=request_id,
            user_id=getattr(request, "user_id", None),
            session_id=getattr(request, "session_id", None),
            tags=base_tags,
            metadata=base_metadata,
            input={
                "question": request.question,
                "conversation_summary": request.conversation_summary if conversation_summary_enabled else None,
            },
        ) as trace_ctx:
            trace_id = trace_ctx.trace_id
            async for event in graph.astream(
                init_state,
                stream_mode="custom",
                config={"callbacks": [trace_ctx.handler]},
            ):
                if await is_disconnected():
                    client_disconnected = True
                    if LLM_STREAM_DEBUG:
                        logger.info(
                            "[ASK_STREAM] client disconnected (backend={})", agent_backend
                        )
                    break

                if isinstance(event, str):
                    if LLM_STREAM_DEBUG:
                        logger.info("[ASK_STREAM RAW STRING] {}", event)
                    continue
                if not isinstance(event, dict):
                    if LLM_STREAM_DEBUG:
                        logger.info(
                            "[ASK_STREAM NON-DICT] type={} value={}", type(event), event
                        )
                    continue

                channel = event.get("channel")
                event["request_id"] = request_id
                node = event.get("node")
                stage = event.get("stage")

                if (
                    node == "guard"
                    and channel == "status"
                    and stage == AskStreamStages.GUARD_END
                ):
                    if "blocked" in event:
                        summary["guard_blocked"] = bool(event.get("blocked"))
                elif (
                    node == "intent_analyzer"
                    and channel == "status"
                    and stage == AskStreamStages.PLANNER_DONE
                ):
                    summary["intent"] = event.get("intent")
                elif node == "query_builder" and channel == "status":
                    if stage == AskStreamStages.QUERY_BUILDER_DONE:
                        summary["search_query"] = event.get("query")
                elif node == "tool_executor" and channel == "status":
                    if stage == AskStreamStages.TOOL_EXECUTOR_DONE:
                        summary["agent_used_tools"] = event.get("used_tools") or []
                elif node == "response_synth" and channel == "status":
                    if stage == AskStreamStages.RESPONSE_DONE:
                        summary["is_out_of_scope"] = event.get("is_out_of_scope")
                        summary["agent_loops"] = event.get("loops")
                        used_tools = event.get("used_tools")
                        if used_tools:
                            summary["agent_used_tools"] = used_tools

                if channel == "meta":
                    meta = event.get("meta") or {}
                    usage = meta.get("usage") or {}
                    if isinstance(usage, dict):
                        total_usage["total_tokens"] += usage.get("total_tokens") or 0
                        total_usage["input_tokens"] += usage.get("input_tokens") or 0
                        total_usage["output_tokens"] += usage.get("output_tokens") or 0
                    if conversation_summary_enabled and node == "response_synth":
                        conv_summary = meta.get("conversation_summary")
                        if conv_summary is not None:
                            summary["conversation_summary"] = conv_summary

                if channel == "raw_event":
                    if LLM_STREAM_DEBUG:
                        logger.info("[ASK_STREAM RAW_EVENT DROPPED] {}", event)
                    continue

                if LLM_STREAM_DEBUG:
                    logger.info("[ASK_STREAM EVENT] channel={} payload={}", channel, event)

                yield event

        if not client_disconnected:
            summary["total_usage"] = total_usage
            summary["trace_id"] = trace_id

            summary_event = {
                "request_id": request_id,
                "trace_id": trace_id,
                "source": architecture,
                "node": architecture,
                "phase": "summary",
                "channel": "meta_summary",
                "summary": summary,
            }
            if LLM_STREAM_DEBUG:
                logger.info("[ASK_STREAM META_SUMMARY] {}", summary_event)
            yield summary_event
    finally:
        # 清理請求上下文
        llm_concurrency.unregister_request(request_id)
        request_context_var.reset(context_token)


async def stream_events(
    request: QuestionRequest,
    is_disconnected: Callable[[], Awaitable[bool]],
) -> AsyncIterator[dict]:
    """Responses backend（預設 ask/stream）統一入口。"""
    async for event in run_stream_graph(
        request,
        is_disconnected,
        agent_backend="responses",
    ):
        yield event


async def stream_events_chat(
    request: QuestionRequest,
    is_disconnected: Callable[[], Awaitable[bool]],
) -> AsyncIterator[dict]:
    """Chat backend 統一入口。"""
    async for event in run_stream_graph(
        request,
        is_disconnected,
        agent_backend="chat",
    ):
        yield event


class AskStreamService:
    """提供 Unified Agent 串流服務的封裝類別。"""

    async def stream_events(
        self,
        request: QuestionRequest,
        is_disconnected: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[dict]:
        async for event in stream_events(request, is_disconnected):
            yield event

    async def stream_events_chat(
        self,
        request: QuestionRequest,
        is_disconnected: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[dict]:
        async for event in stream_events_chat(request, is_disconnected):
            yield event

    async def stream_events_responses(
        self,
        request: QuestionRequest,
        is_disconnected: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[dict]:
        # 向後相容舊名稱
        async for event in stream_events(request, is_disconnected):
            yield event


ask_stream_service = AskStreamService()
