"""
================================================================================
Ask Stream Chat API Route - OpenAI Chat API 相容的串流問答端點
================================================================================

本模組實現 GraphRAG 系統的串流問答 API，使用 OpenAI Chat Completions API 相容格式。
此端點讓您可以使用任何支援 OpenAI API 的客戶端或函式庫來呼叫 GraphRAG 系統。

端點列表：
---------
- POST /api/v1/rag/ask/stream_chat   - 串流聊天完成（OpenAI 相容格式）

與 OpenAI API 的相容性：
----------------------
此端點相容 OpenAI Chat Completions API 的請求和回應格式：
- 請求格式：messages 陣列，每個訊息包含 role 和 content
- 回應格式：chat.completion 或 chat.completion.chunk

支援的角色：
-----------
- user: 使用者訊息
- assistant: 助理（AI）訊息
- system: 系統提示訊息

主要特性：
---------
1. OpenAI API 相容
   - 可直接使用 OpenAI Python SDK
   - 支援 LangChain、LlamaIndex 等框架整合
   - 串流格式與 OpenAI 完全一致

2. GraphRAG 工作流程整合
   - 支援 Local、Global、DRIFT 三種查詢模式
   - 知識圖譜增強檢索
   - 社群報告整合

3. 對話歷史支援
   - 支援多輪對話上下文
   - 自動轉換訊息格式為 LangChain 格式

使用範例：
---------
# 使用 OpenAI SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api/v1/rag",
    api_key="not-needed"  # GraphRAG 不需要 API key
)

response = client.chat.completions.create(
    model="graphrag",
    messages=[
        {"role": "system", "content": "你是一個醫療知識助理"},
        {"role": "user", "content": "什麼是糖尿病？"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")

# 使用 curl 測試串流
curl -X POST "http://localhost:8000/api/v1/rag/ask/stream_chat" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [
      {"role": "user", "content": "什麼是糖尿病？"}
    ],
    "stream": true
  }'

依賴關係：
---------
- chatbot_graphrag.graph_workflow: GraphRAG LangGraph 工作流程
- langchain_core.messages: LangChain 訊息類型
- fastapi: Web 框架
- pydantic: 資料驗證

作者：GraphRAG Team
版本：1.0.0
================================================================================
"""

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from chatbot_graphrag.graph_workflow import (
    build_graphrag_workflow,
    create_initial_state,
)
from chatbot_graphrag.graph_workflow.tracing import (
    create_trace_context,
    update_trace_with_result,
)

# ============================================================================
# 模組設定
# ============================================================================

logger = logging.getLogger(__name__)

# 建立路由器
# prefix: API 路徑前綴（與 ask_stream 共用）
# tags: OpenAPI 文件分類標籤（使用不同標籤區分）
router = APIRouter(prefix="/api/v1/rag", tags=["ask-chat"])


# ============================================================================
# 請求/回應模型 (OpenAI 相容格式)
# ============================================================================


class ChatMessage(BaseModel):
    """
    聊天訊息模型 (OpenAI 相容)。

    此模型遵循 OpenAI Chat Completions API 的訊息格式。

    屬性說明：
    ---------
    role : str
        訊息角色，支援以下值：
        - "user": 使用者訊息
        - "assistant": AI 助理訊息
        - "system": 系統提示（用於設定助理行為）

    content : str
        訊息內容文字。

    範例：
    -----
    用戶訊息：
    {"role": "user", "content": "什麼是糖尿病？"}

    系統訊息：
    {"role": "system", "content": "你是一個專業的醫療知識助理"}

    助理訊息：
    {"role": "assistant", "content": "糖尿病是一種慢性代謝疾病..."}
    """

    role: str = Field(
        ...,
        description="訊息角色：user（使用者）、assistant（助理）、system（系統）"
    )
    content: str = Field(
        ...,
        description="訊息內容"
    )


class ChatRequest(BaseModel):
    """
    聊天請求模型 (OpenAI 相容)。

    此模型遵循 OpenAI Chat Completions API 的請求格式，
    但針對 GraphRAG 系統加入了額外的參數。

    屬性說明：
    ---------
    messages : list[ChatMessage]
        對話訊息列表，必須至少包含一個訊息。
        訊息會按順序處理，最後一個 user 訊息會被視為當前問題。

    conversation_id : Optional[str]
        對話 ID，用於追蹤多輪對話。

    acl_groups : Optional[list[str]]
        存取控制群組，預設為 ["public"]。
        控制使用者可存取的文件範圍。

    tenant_id : str
        租戶識別碼，預設為 "default"。
        用於多租戶環境。

    stream : bool
        是否啟用串流回應，預設為 True。
        - True: 以 SSE 格式串流，相容 OpenAI 串流格式
        - False: 返回完整的 ChatCompletionResponse

    temperature : float
        模型溫度參數 (0.0-2.0)，預設為 0.1。
        較低的溫度產生更確定性的回答。

    max_tokens : Optional[int]
        最大生成 token 數（目前保留，未實際使用）。

    backend : str
        LLM API 後端類型，預設為 "responses"。
        - "responses": 使用 OpenAI Responses API
        - "chat": 使用 OpenAI Chat Completions API

    OpenAI 相容範例：
    ----------------
    {
        "messages": [
            {"role": "system", "content": "你是一個專業的醫療知識助理"},
            {"role": "user", "content": "糖尿病有哪些症狀？"}
        ],
        "stream": true,
        "temperature": 0.7
    }

    GraphRAG 擴展範例：
    ------------------
    {
        "messages": [
            {"role": "user", "content": "什麼是糖尿病？"}
        ],
        "acl_groups": ["public", "medical-staff"],
        "tenant_id": "hospital-a",
        "stream": true,
        "backend": "responses"
    }
    """

    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="對話訊息列表（必填，至少一個訊息）"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="對話 ID"
    )
    acl_groups: Optional[list[str]] = Field(
        default=["public"],
        description="存取控制群組列表"
    )
    tenant_id: str = Field(
        default="default",
        description="租戶識別碼"
    )
    stream: bool = Field(
        default=True,
        description="是否啟用串流回應"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="模型溫度參數 (0.0-2.0)"
    )
    max_tokens: Optional[int] = Field(
        None,
        description="最大生成 token 數"
    )
    backend: str = Field(
        default="responses",
        description="LLM API 後端: 'responses' (Responses API) 或 'chat' (Chat Completions API)"
    )
    include_sources: bool = Field(
        default=True,
        description="是否在回應中包含參考來源資訊"
    )


class ChatChoice(BaseModel):
    """
    聊天完成選項 (OpenAI 相容)。

    屬性說明：
    ---------
    index : int
        選項索引，通常為 0。

    message : Optional[ChatMessage]
        完整訊息（用於非串流回應）。

    delta : Optional[dict]
        增量內容（用於串流回應）。
        例如：{"content": "糖尿病"} 或 {"role": "assistant"}

    finish_reason : Optional[str]
        完成原因：
        - "stop": 正常完成
        - "length": 達到最大 token 限制
        - "error": 發生錯誤
        - None: 串流進行中
    """

    index: int = Field(default=0, description="選項索引")
    message: Optional[ChatMessage] = Field(None, description="完整訊息（非串流）")
    delta: Optional[dict] = Field(None, description="增量內容（串流）")
    finish_reason: Optional[str] = Field(None, description="完成原因")


class ChatCompletionResponse(BaseModel):
    """
    聊天完成回應 (OpenAI 相容)。

    此模型遵循 OpenAI Chat Completions API 的回應格式。

    屬性說明：
    ---------
    id : str
        唯一回應 ID，格式為 "chatcmpl-{uuid}"。

    object : str
        物件類型，固定為 "chat.completion"。

    created : int
        建立時間（Unix 時間戳）。

    model : str
        使用的模型名稱，固定為 "graphrag"。

    choices : list[ChatChoice]
        完成選項列表，通常只有一個選項。

    usage : Optional[dict]
        Token 使用統計：
        - prompt_tokens: 提示 token 數
        - completion_tokens: 完成 token 數
        - total_tokens: 總 token 數

    回應範例：
    ---------
    {
        "id": "chatcmpl-abc123def456",
        "object": "chat.completion",
        "created": 1703644800,
        "model": "graphrag",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "糖尿病是一種慢性代謝疾病..."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 120,
            "total_tokens": 135
        }
    }
    """

    id: str = Field(..., description="唯一回應 ID")
    object: str = Field(default="chat.completion", description="物件類型")
    created: int = Field(..., description="建立時間（Unix 時間戳）")
    model: str = Field(default="graphrag", description="模型名稱")
    choices: list[ChatChoice] = Field(..., description="完成選項列表")
    usage: Optional[dict] = Field(None, description="Token 使用統計")


# ============================================================================
# 輔助函式
# ============================================================================


def convert_messages(messages: list[ChatMessage]):
    """
    將 ChatMessage 列表轉換為 LangChain 訊息格式。

    此函式負責將 OpenAI 格式的訊息轉換為 LangChain 的訊息類型，
    以便於 GraphRAG 工作流程處理。

    參數說明：
    ---------
    messages : list[ChatMessage]
        OpenAI 格式的訊息列表。

    返回：
    -----
    list[BaseMessage]
        LangChain 格式的訊息列表。

    轉換對照：
    ---------
    - "user" → HumanMessage
    - "assistant" → AIMessage
    - "system" → SystemMessage

    範例：
    -----
    >>> messages = [
    ...     ChatMessage(role="system", content="你是助理"),
    ...     ChatMessage(role="user", content="你好")
    ... ]
    >>> lc_messages = convert_messages(messages)
    >>> print(type(lc_messages[0]))
    <class 'langchain_core.messages.SystemMessage'>
    """
    lc_messages = []
    for msg in messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            lc_messages.append(SystemMessage(content=msg.content))
    return lc_messages


# ============================================================================
# SSE 串流生成器 (OpenAI 相容格式)
# ============================================================================


async def generate_chat_stream(
    messages: list[ChatMessage],
    acl_groups: list[str],
    tenant_id: str,
    agent_backend: str = "responses",
    session_id: Optional[str] = None,
    include_sources: bool = True,
) -> AsyncGenerator[str, None]:
    """
    生成 OpenAI 相容的 SSE 串流回應（真正 LLM 串流版本）。

    此函式執行 GraphRAG 工作流程，並使用 LangGraph 的 stream_mode="custom"
    實現真正的 LLM token 串流。整合 Langfuse tracing 追蹤。

    參數說明：
    ---------
    messages : list[ChatMessage]
        對話訊息列表。

    acl_groups : list[str]
        存取控制群組列表。

    tenant_id : str
        租戶識別碼。

    agent_backend : str
        LLM API 後端：'responses' 或 'chat'

    session_id : Optional[str]
        會話 ID，用於 Langfuse 會話追蹤。

    include_sources : bool
        是否在回應中包含參考來源資訊，預設為 True。

    SSE 事件格式：
    -------------
    OpenAI 串流格式，每個事件以 "data: {JSON}\n\n" 發送：

    1. 角色事件（首個事件）
       data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}

    2. 內容事件（多個，真正的 LLM token 串流）
       data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"..."}}]}

    3. 完成事件
       data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

    4. 結束標記
       data: [DONE]
    """
    import time
    from typing import Any

    # 生成唯一完成 ID 和請求 ID
    request_id = uuid.uuid4().hex[:8]
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    role_sent = False
    full_answer = ""  # 用於 trace 輸出

    # ===== 步驟 1：提取使用者問題 =====
    question = ""
    for msg in reversed(messages):
        if msg.role == "user":
            question = msg.content
            break

    # 沒有使用者訊息時返回錯誤
    if not question:
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "graphrag",
            "choices": [{
                "index": 0,
                "delta": {"content": "錯誤：找不到使用者訊息"},
                "finish_reason": "error",
            }],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # ===== 步驟 2：建立 Langfuse trace context =====
    with create_trace_context(
        name="graphrag-stream-chat",
        trace_id_seed=request_id,
        session_id=session_id,
        tags=["graphrag", "stream-chat", "openai-compatible"],
        metadata={"tenant_id": tenant_id, "backend": agent_backend},
        input_data={"question": question, "message_count": len(messages)},
    ) as ctx:
        trace_id = ctx.trace_id if ctx else request_id

        try:
            # ===== 步驟 3：建立 GraphRAG 工作流程 =====
            workflow = build_graphrag_workflow()
            lc_messages = convert_messages(messages)

            initial_state = create_initial_state(
                question=question,
                messages=lc_messages,
                acl_groups=acl_groups,
                tenant_id=tenant_id,
                agent_backend=agent_backend,
            )
            initial_state["trace_id"] = trace_id

            # 執行配置
            config: dict[str, Any] = {"configurable": {"thread_id": trace_id}}
            if ctx and ctx.handler:
                config["callbacks"] = [ctx.handler]

            # ===== 步驟 4：使用真正串流執行工作流程 =====
            # Use combined stream mode to get both custom events and final state
            event_count = 0
            final_state = None

            async for stream_item in workflow.astream(initial_state, config, stream_mode=["custom", "values"]):
                event_count += 1

                # Handle tuple format from combined stream modes: (stream_mode, data)
                if isinstance(stream_item, tuple) and len(stream_item) == 2:
                    mode, data = stream_item
                    if mode == "values":
                        # This is the final state snapshot
                        final_state = data
                        logger.debug(f"SSE event #{event_count}: values snapshot (keys={list(data.keys())[:5]}...)")
                        continue
                    elif mode == "custom":
                        event = data
                    else:
                        logger.debug(f"SSE event #{event_count}: unknown mode={mode}")
                        continue
                else:
                    # Direct event (shouldn't happen with combined mode, but handle it)
                    event = stream_item

                # 從自定義事件中提取 channel 和內容
                channel = event.get("channel") if isinstance(event, dict) else None

                # Log received events for debugging
                if channel:
                    logger.debug(f"SSE event #{event_count}: channel={channel}, node={event.get('node', 'N/A')}")
                else:
                    logger.debug(f"SSE event #{event_count}: non-channel event: {type(event)}")

                # 首次收到 answer 事件時發送角色事件
                if not role_sent and channel == "answer":
                    role_sent = True
                    role_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "graphrag",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(role_chunk)}\n\n"

                # 串流 answer delta（真正的 LLM token）
                if channel == "answer":
                    delta = event.get("delta", "")
                    if delta:
                        full_answer += delta  # 累積完整回答
                        content_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "graphrag",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": delta},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(content_chunk)}\n\n"

                # 處理 status channel（用於前端顯示處理階段）
                elif channel == "status":
                    status_event = {
                        "type": "status",
                        "node": event.get("node", ""),
                        "stage": event.get("stage", ""),
                    }
                    yield f"data: {json.dumps(status_event)}\n\n"

                # 處理 sources channel（參考來源資訊）
                elif channel == "sources" and include_sources:
                    sources_event = {
                        "type": "sources",
                        "sources": event.get("sources", []),
                    }
                    yield f"data: {json.dumps(sources_event)}\n\n"

            # ===== 步驟 5：發送完成事件 =====
            logger.info(f"Workflow completed: received {event_count} events, full_answer_len={len(full_answer)}, role_sent={role_sent}")

            # Fallback: If no answer was streamed but final_state has answer, stream it manually
            if not full_answer and final_state:
                cached_answer = final_state.get("final_answer", "")
                if cached_answer:
                    logger.info(f"Fallback: streaming cached answer manually ({len(cached_answer)} chars)")

                    # Send role first
                    if not role_sent:
                        role_sent = True
                        role_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "graphrag",
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(role_chunk)}\n\n"

                    # Stream answer in chunks with delay for natural typing effect
                    # Match settings in output.py for consistent UX
                    chunk_size = 20  # Smaller chunks for smoother streaming
                    delay_per_chunk = 0.02  # 20ms delay between chunks
                    for i in range(0, len(cached_answer), chunk_size):
                        chunk = cached_answer[i:i + chunk_size]
                        full_answer += chunk
                        content_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "graphrag",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(content_chunk)}\n\n"
                        await asyncio.sleep(delay_per_chunk)

            # 如果從未發送過角色事件（可能是 blocked 或錯誤的情況），先發送
            if not role_sent:
                role_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "graphrag",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n"

            finish_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "graphrag",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"
            yield "data: [DONE]\n\n"

            # ===== 步驟 6：更新 Langfuse trace =====
            if ctx:
                update_trace_with_result(
                    output={
                        "answer": full_answer,
                        "answer_length": len(full_answer),
                    },
                    metadata={
                        "completion_id": completion_id,
                        "stream_mode": "custom",
                    },
                )

        except Exception as e:
            # ===== 錯誤處理 =====
            logger.error(f"聊天串流錯誤: {e}")
            error_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "graphrag",
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"錯誤: {str(e)}"},
                    "finish_reason": "error",
                }],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"


# ============================================================================
# API 端點
# ============================================================================


@router.post("/ask/stream_chat")
async def ask_stream_chat(request: ChatRequest):
    """
    OpenAI 相容的聊天完成端點。

    此端點遵循 OpenAI Chat Completions API 格式，讓您可以使用任何
    支援 OpenAI API 的客戶端或函式庫來呼叫 GraphRAG 系統。

    工作流程：
    ---------
    1. 接收 OpenAI 格式的聊天請求
    2. 將訊息轉換為 LangChain 格式
    3. 執行 GraphRAG 工作流程
    4. 以 OpenAI 相容格式返回回應

    串流模式 (stream=true)：
    ----------------------
    返回 SSE 串流，格式與 OpenAI 完全相容：

    ```
    data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}

    data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"糖尿病"}}]}

    data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"是一種"}}]}

    data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

    data: [DONE]
    ```

    非串流模式 (stream=false)：
    -------------------------
    返回完整的 ChatCompletionResponse JSON。

    與 OpenAI SDK 整合：
    -------------------
    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/api/v1/rag",
        api_key="not-needed"
    )

    # 非串流
    response = client.chat.completions.create(
        model="graphrag",
        messages=[{"role": "user", "content": "什麼是糖尿病？"}],
        stream=False
    )
    print(response.choices[0].message.content)

    # 串流
    stream = client.chat.completions.create(
        model="graphrag",
        messages=[{"role": "user", "content": "什麼是糖尿病？"}],
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
    ```

    使用 curl 測試：
    --------------
    # 串流模式
    curl -N -X POST "http://localhost:8000/api/v1/rag/ask/stream_chat" \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [
          {"role": "system", "content": "你是一個專業的醫療助理"},
          {"role": "user", "content": "糖尿病的症狀有哪些？"}
        ],
        "stream": true
      }'

    # 非串流模式
    curl -X POST "http://localhost:8000/api/v1/rag/ask/stream_chat" \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [
          {"role": "user", "content": "什麼是糖尿病？"}
        ],
        "stream": false
      }'

    多輪對話範例：
    -------------
    {
        "messages": [
            {"role": "system", "content": "你是一個醫療知識助理"},
            {"role": "user", "content": "什麼是糖尿病？"},
            {"role": "assistant", "content": "糖尿病是一種慢性代謝疾病..."},
            {"role": "user", "content": "那要如何預防？"}
        ]
    }

    HTTP 回應標頭（串流模式）：
    ------------------------
    - Content-Type: text/event-stream
    - Cache-Control: no-cache
    - Connection: keep-alive
    - X-Accel-Buffering: no

    參數：
    -----
    request : ChatRequest
        聊天請求，包含訊息列表和設定

    返回：
    -----
    StreamingResponse | ChatCompletionResponse
        串流模式返回 SSE 串流，非串流模式返回 JSON

    異常：
    -----
    HTTPException
        - 400: 請求無效（如找不到 user 訊息）
        - 500: 內部伺服器錯誤
    """
    logger.info(f"聊天串流請求: {len(request.messages)} 個訊息")

    # ===== 非串流模式處理 =====
    if not request.stream:
        try:
            import time
            from typing import Any

            request_id = uuid.uuid4().hex[:8]

            # 提取最後一個 user 訊息作為問題
            question = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    question = msg.content
                    break

            if not question:
                raise HTTPException(
                    status_code=400,
                    detail="請求必須包含至少一個 user 訊息"
                )

            # 建立 Langfuse trace context
            with create_trace_context(
                name="graphrag-stream-chat",
                trace_id_seed=request_id,
                session_id=request.conversation_id,
                tags=["graphrag", "stream-chat", "openai-compatible", "non-stream"],
                metadata={"tenant_id": request.tenant_id, "backend": request.backend},
                input_data={"question": question, "message_count": len(request.messages)},
            ) as ctx:
                trace_id = ctx.trace_id if ctx else request_id

                # 建立並執行工作流程
                workflow = build_graphrag_workflow()
                lc_messages = convert_messages(request.messages)

                initial_state = create_initial_state(
                    question=question,
                    messages=lc_messages,
                    acl_groups=request.acl_groups or ["public"],
                    tenant_id=request.tenant_id,
                    agent_backend=request.backend,
                )
                initial_state["trace_id"] = trace_id

                # 執行配置
                config: dict[str, Any] = {"configurable": {"thread_id": trace_id}}
                if ctx and ctx.handler:
                    config["callbacks"] = [ctx.handler]

                result = await workflow.ainvoke(initial_state, config)

                # 計算 token 使用量（簡化估算：字元數/4）
                prompt_tokens = sum(len(m.content) // 4 for m in request.messages)
                completion_content = result.get("final_answer", "")
                completion_tokens = len(completion_content) // 4

                # 更新 Langfuse trace
                if ctx:
                    update_trace_with_result(
                        output={
                            "answer": completion_content,
                            "answer_length": len(completion_content),
                        },
                        metadata={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    )

                # 返回完整回應
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    created=int(time.time()),
                    choices=[
                        ChatChoice(
                            index=0,
                            message=ChatMessage(
                                role="assistant",
                                content=completion_content,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"聊天錯誤: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== 串流模式處理 =====
    return StreamingResponse(
        generate_chat_stream(
            messages=request.messages,
            acl_groups=request.acl_groups or ["public"],
            tenant_id=request.tenant_id,
            agent_backend=request.backend,
            session_id=request.conversation_id,
            include_sources=request.include_sources,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # 禁用快取
            "Connection": "keep-alive",         # 保持連線
            "X-Accel-Buffering": "no",          # 禁用 nginx 緩衝
        },
    )
