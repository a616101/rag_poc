from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from openai import AsyncOpenAI, OpenAI

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    get_buffer_string,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun

# 關鍵：用官方 type 定義，跟 ResponseStreamEvent union 對齊
from openai.types.responses import (
    # 文字
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    # reasoning 文字
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    # reasoning summary
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    # 工具 arguments
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    # 音訊 / transcript
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    # 拒絕
    ResponseRefusalDeltaEvent,
    ResponseRefusalDoneEvent,
    # 狀態 / error
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseCompletedEvent,
    ResponseInProgressEvent,
    ResponseCreatedEvent,
    ResponseQueuedEvent,
    # output item / content part / annotation
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    # 搜尋類
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
    ResponseFileSearchCallCompletedEvent,
    ResponseFileSearchCallInProgressEvent,
    ResponseFileSearchCallSearchingEvent,
    # image gen
    ResponseImageGenCallCompletedEvent,
    ResponseImageGenCallGeneratingEvent,
    ResponseImageGenCallInProgressEvent,
    ResponseImageGenCallPartialImageEvent,
    # MCP / 自訂 tool
    ResponseMcpCallArgumentsDeltaEvent,
    ResponseMcpCallArgumentsDoneEvent,
    ResponseMcpCallCompletedEvent,
    ResponseMcpCallFailedEvent,
    ResponseMcpCallInProgressEvent,
    ResponseMcpListToolsCompletedEvent,
    ResponseMcpListToolsFailedEvent,
    ResponseMcpListToolsInProgressEvent,
    ResponseCustomToolCallInputDeltaEvent,
    ResponseCustomToolCallInputDoneEvent,
)

from chatbot_rag.llm.responses_accumulator import ResponsesAccumulator

try:  # LangChain tools 可能不存在於所有環境，故用 try-import 降低耦合
    from langchain_core.tools import BaseTool
except Exception:  # pragma: no cover
    BaseTool = Any  # type: ignore[misc]


class ResponsesChatModel(BaseChatModel):
    """LangChain ChatModel 封裝 /v1/responses（或相容實作）。

    - streaming=True → 真正用 SSE streaming + 分 channel
    - streaming=False → .stream() 退化成一次性完整訊息
    """

    # 基本設定
    model: str
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "EMPTY"

    # 決定要不要真的用 streaming（你要的參數）
    streaming: bool = False

    # reasoning 參數 → 對應 body.reasoning
    reasoning_effort: Optional[str] = None  # "low" | "medium" | "high" ...
    reasoning_summary: Optional[str] = None  # "concise" | "detailed" | "auto" ...

    # 生成參數
    temperature: Optional[float] = None  # 溫度參數，控制輸出的隨機性
    max_tokens: Optional[int] = None  # 最大輸出 token 數

    # 預設加進 request body 的參數（例如 temperature 等）
    default_extra_body: Optional[Dict[str, Any]] = None

    # 綁定的 LangChain tools（若有）
    _bound_tools: Optional[List[BaseTool]] = None
    _bound_tools_schema: Optional[List[Dict[str, Any]]] = None

    # -------- metadata --------

    @property
    def _llm_type(self) -> str:
        return "responses"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "streaming": self.streaming,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_summary": self.reasoning_summary,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    # -------- helpers --------

    def _get_client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _get_async_client(self) -> AsyncOpenAI:
        """取得非同步 OpenAI 客戶端。"""
        return AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    def _messages_to_input(self, messages: List[BaseMessage]) -> str:
        # 先用最簡單的 get_buffer_string，有需要再改成原生 Responses input 格式
        return get_buffer_string(messages)

    def _build_body(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "input": self._messages_to_input(messages),
        }

        # reasoning
        reasoning: Dict[str, Any] = {}
        if self.reasoning_effort is not None:
            reasoning["effort"] = self.reasoning_effort
        if self.reasoning_summary is not None:
            reasoning["summary"] = self.reasoning_summary
        if reasoning:
            body["reasoning"] = reasoning

        # 生成參數
        if self.temperature is not None:
            body["temperature"] = self.temperature
        # Responses API 使用 max_output_tokens 而非 max_tokens
        if self.max_tokens is not None:
            body["max_output_tokens"] = self.max_tokens

        if stop:
            body["stop"] = stop

        if self.default_extra_body:
            body.update(self.default_extra_body)

        # 若有經 bind_tools 綁定工具，將其 schema 加入 body.tools
        if self._bound_tools_schema:
            # 依 OpenAI Responses API 規格，tools 是一個 tools 列表
            body.setdefault("tools", self._bound_tools_schema)

        extra_body = kwargs.pop("extra_body", {})
        if extra_body:
            body.update(extra_body)

        body.update(kwargs)
        return body

    # -------- tools / function calling 支援 --------

    def bind_tools(self, tools: List[BaseTool]) -> "ResponsesChatModel":
        """
        參考 Chat Completions 的 bind_tools 介面，為 Responses API 綁定 LangChain tools。

        - 會將 tool 轉成 OpenAI 相容的工具描述（name / description / parameters）
        - 在 _build_body 時自動帶入 body["tools"]
        - 實際的 tool_calls 解析在 _generate 中進行
        """

        self._bound_tools = tools
        schemas: List[Dict[str, Any]] = []
        for tool in tools:
            # LangChain 的 tool 通常有 .name / .description / .args_schema
            name = getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
            description = getattr(tool, "description", "") or ""
            json_schema: Dict[str, Any] = {}
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None and hasattr(args_schema, "schema"):
                try:
                    json_schema = args_schema.schema()  # type: ignore[assignment]
                except Exception:  # noqa: BLE001
                    json_schema = {}

            # OpenAI Responses tools 結構：type / name / description / parameters
            tool_def: Dict[str, Any] = {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": json_schema or {"type": "object", "properties": {}},
            }
            schemas.append(tool_def)

        self._bound_tools_schema = schemas
        return self

    # -------- non-streaming: _generate / invoke --------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = self._get_client()
        body = self._build_body(messages, stop=stop, **kwargs)
        body["stream"] = False

        resp = client.responses.create(**body)

        raw = resp.model_dump(mode="python")

        # 官方有 resp.output_text shortcut（如果背後是 OpenAI 正版）
        text = getattr(resp, "output_text", None)
        if text is None:
            try:
                first_output = resp.output[0]
                first_part = first_output.content[0]
                if hasattr(first_part, "text"):
                    text = first_part.text
                else:
                    text = str(first_part)
            except Exception:
                text = ""

        reasoning_summary = None
        try:
            # 視你的模型 / 後端實作而定；OpenAI 官方會在 output.summary 裡存 reasoning summary
            reasoning_summary = getattr(resp, "reasoning", None)
        except Exception:
            pass

        # 嘗試從 raw response 中抽取 tool_calls（與 Chat Completions 相容的結構）
        tool_calls: List[Dict[str, Any]] = []

        # 依照官方 Responses API，目前 function calling 會出現在 output[*]，type="function_call"
        # LangChain 的 default_tool_parser 期待格式：
        # {"function": {"name": "...", "arguments": "..."}, "id": "..."}
        try:
            outputs = raw.get("output") or []
            if isinstance(outputs, list):
                for idx, item in enumerate(outputs):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "function_call":
                        continue
                    name = item.get("name")
                    if not name:
                        continue
                    args = item.get("arguments") or "{}"
                    call_id = item.get("call_id") or item.get("id") or f"call_{idx}"
                    # 使用 LangChain 期待的格式，讓 AIMessage 能自動解析到 .tool_calls
                    tool_calls.append(
                        {
                            "id": call_id,
                            "function": {
                                "name": name,
                                "arguments": args,
                            },
                        }
                    )
        except Exception:  # noqa: BLE001
            tool_calls = []

        additional_kwargs: Dict[str, Any] = {
            "reasoning_summary": reasoning_summary,
            "response_id": getattr(resp, "id", None),
        }
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls

        # 從 response 中提取 usage 資訊，設置 usage_metadata 供 Langfuse 追蹤
        usage_metadata: Optional[Dict[str, Any]] = None
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                # 轉換成 LangChain 標準的 usage_metadata 格式
                usage_dict = usage.model_dump(mode="python") if hasattr(usage, "model_dump") else dict(usage)
                usage_metadata = {
                    "input_tokens": usage_dict.get("input_tokens", 0),
                    "output_tokens": usage_dict.get("output_tokens", 0),
                    "total_tokens": usage_dict.get("total_tokens", 0),
                }
        except Exception:  # noqa: BLE001
            pass

        msg = AIMessage(
            content=text or "",
            additional_kwargs=additional_kwargs,
            response_metadata={"raw_response": raw},
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # -------- async non-streaming: _agenerate / ainvoke --------

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """非同步版本的 _generate，使用 AsyncOpenAI 客戶端。"""
        client = self._get_async_client()
        body = self._build_body(messages, stop=stop, **kwargs)
        body["stream"] = False

        resp = await client.responses.create(**body)

        raw = resp.model_dump(mode="python")

        # 官方有 resp.output_text shortcut（如果背後是 OpenAI 正版）
        text = getattr(resp, "output_text", None)
        if text is None:
            try:
                first_output = resp.output[0]
                first_part = first_output.content[0]
                if hasattr(first_part, "text"):
                    text = first_part.text
                else:
                    text = str(first_part)
            except Exception:
                text = ""

        reasoning_summary = None
        try:
            reasoning_summary = getattr(resp, "reasoning", None)
        except Exception:
            pass

        # 嘗試從 raw response 中抽取 tool_calls
        tool_calls: List[Dict[str, Any]] = []
        try:
            outputs = raw.get("output") or []
            if isinstance(outputs, list):
                for idx, item in enumerate(outputs):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "function_call":
                        continue
                    name = item.get("name")
                    if not name:
                        continue
                    args = item.get("arguments") or "{}"
                    call_id = item.get("call_id") or item.get("id") or f"call_{idx}"
                    tool_calls.append(
                        {
                            "id": call_id,
                            "function": {
                                "name": name,
                                "arguments": args,
                            },
                        }
                    )
        except Exception:  # noqa: BLE001
            tool_calls = []

        additional_kwargs: Dict[str, Any] = {
            "reasoning_summary": reasoning_summary,
            "response_id": getattr(resp, "id", None),
        }
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls

        # 從 response 中提取 usage 資訊
        usage_metadata: Optional[Dict[str, Any]] = None
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                usage_dict = (
                    usage.model_dump(mode="python")
                    if hasattr(usage, "model_dump")
                    else dict(usage)
                )
                usage_metadata = {
                    "input_tokens": usage_dict.get("input_tokens", 0),
                    "output_tokens": usage_dict.get("output_tokens", 0),
                    "total_tokens": usage_dict.get("total_tokens", 0),
                }
        except Exception:  # noqa: BLE001
            pass

        msg = AIMessage(
            content=text or "",
            additional_kwargs=additional_kwargs,
            response_metadata={"raw_response": raw},
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # -------- streaming event → channel → chunk --------

    def _event_to_chunk(
        self,
        event: Any,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Optional[ChatGenerationChunk]:
        """依照 openai.types.responses.* 的 event 類型分 channel。"""

        channel = "other"
        text_delta = ""
        done_for: Optional[str] = None

        # 把原始 event 也存起來，給 llm_node 使用
        raw_event = event.model_dump(mode="python") if hasattr(event, "model_dump") else None
        raw_type = getattr(event, "type", None)

        # 1) 一般文字輸出
        if isinstance(event, ResponseTextDeltaEvent):
            channel = "output_text"
            text_delta = event.delta

        # 2) reasoning 文字
        elif isinstance(event, ResponseReasoningTextDeltaEvent):
            channel = "reasoning"
            text_delta = event.delta

        # 3) reasoning summary
        elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
            channel = "reasoning_summary"
            text_delta = event.delta
        elif isinstance(event, ResponseReasoningSummaryPartAddedEvent):
            channel = "reasoning_summary"
            text_delta = event.part.text

        # 4) function call arguments streaming
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            channel = "tool_arguments"
            text_delta = event.delta

        # 5) audio / transcript
        elif isinstance(event, ResponseAudioTranscriptDeltaEvent):
            channel = "audio_transcript"
            text_delta = event.delta
        elif isinstance(event, ResponseAudioDeltaEvent):
            channel = "audio"

        # ✅ 這裡改成「分開」標記 done_for
        elif isinstance(event, ResponseTextDoneEvent):
            channel = "done"
            done_for = "output_text"
        elif isinstance(event, ResponseReasoningTextDoneEvent):
            channel = "done"
            done_for = "reasoning"
        elif isinstance(
            event,
            (
                ResponseReasoningSummaryTextDoneEvent,
                ResponseReasoningSummaryPartDoneEvent,
            ),
        ):
            channel = "done"
            done_for = "reasoning_summary"
        elif isinstance(
            event,
            (
                ResponseFunctionCallArgumentsDoneEvent,
                ResponseAudioDoneEvent,
                ResponseAudioTranscriptDoneEvent,
            ),
        ):
            channel = "done"

        # 6) 拒絕 / refusal
        elif isinstance(event, ResponseRefusalDeltaEvent):
            channel = "refusal"
            text_delta = event.delta
        elif isinstance(event, ResponseRefusalDoneEvent):
            channel = "refusal_done"

        # 7) 狀態 / error
        elif isinstance(
            event,
            (
                ResponseErrorEvent,
                ResponseFailedEvent,
                ResponseIncompleteEvent,
            ),
        ):
            channel = "error"
        elif isinstance(
            event,
            (
                ResponseCreatedEvent,
                ResponseInProgressEvent,
                ResponseCompletedEvent,
                ResponseQueuedEvent,
            ),
        ):
            channel = "status"

        # 8) 搜尋類 / image / MCP / custom tool
        elif isinstance(
            event,
            (
                ResponseWebSearchCallCompletedEvent,
                ResponseWebSearchCallInProgressEvent,
                ResponseWebSearchCallSearchingEvent,
                ResponseFileSearchCallCompletedEvent,
                ResponseFileSearchCallInProgressEvent,
                ResponseFileSearchCallSearchingEvent,
            ),
        ):
            channel = "search"
        elif isinstance(
            event,
            (
                ResponseImageGenCallCompletedEvent,
                ResponseImageGenCallGeneratingEvent,
                ResponseImageGenCallInProgressEvent,
                ResponseImageGenCallPartialImageEvent,
            ),
        ):
            channel = "image"
        elif isinstance(
            event,
            (
                ResponseMcpCallArgumentsDeltaEvent,
                ResponseMcpCallArgumentsDoneEvent,
                ResponseMcpCallCompletedEvent,
                ResponseMcpCallFailedEvent,
                ResponseMcpCallInProgressEvent,
                ResponseMcpListToolsCompletedEvent,
                ResponseMcpListToolsFailedEvent,
                ResponseMcpListToolsInProgressEvent,
            ),
        ):
            channel = "mcp"
        elif isinstance(
            event,
            (
                ResponseCustomToolCallInputDeltaEvent,
                ResponseCustomToolCallInputDoneEvent,
            ),
        ):
            channel = "custom_tool_input"
            if hasattr(event, "delta"):
                text_delta = event.delta

        # 9) output item / content part / annotation
        elif isinstance(
            event,
            (
                ResponseOutputItemAddedEvent,
                ResponseOutputItemDoneEvent,
                ResponseContentPartAddedEvent,
                ResponseContentPartDoneEvent,
                ResponseOutputTextAnnotationAddedEvent,
            ),
        ):
            channel = "output_item"

        # callback token
        if run_manager is not None and text_delta:
            run_manager.on_llm_new_token(text_delta)

        # 給使用者看的主文字
        content = text_delta if channel == "output_text" else ""

        additional_kwargs: Dict[str, Any] = {
            "channel": channel,
            "delta": text_delta,
        }
        if done_for is not None:
            additional_kwargs["done_for"] = done_for
        if raw_event is not None:
            additional_kwargs["raw_event"] = raw_event
            additional_kwargs["raw_type"] = raw_type

        msg_chunk = AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
        )
        return ChatGenerationChunk(message=msg_chunk)

    # -------- _stream：依 streaming flag 切換 --------

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # streaming=False → 舊邏輯，不動
        if not self.streaming:
            result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            full_msg: AIMessage = result.generations[0].message  # type: ignore[assignment]
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=full_msg.content,
                    additional_kwargs=full_msg.additional_kwargs,
                    response_metadata=full_msg.response_metadata,
                    usage_metadata=getattr(full_msg, "usage_metadata", None),  # type: ignore[arg-type]
                )
            )
            return

        # streaming=True → 真正用 Responses SSE
        client = self._get_client()
        body = self._build_body(messages, stop=stop, **kwargs)
        body["stream"] = True

        stream = client.responses.create(**body)

        acc = ResponsesAccumulator()

        for event in stream:
            # 先丟進 accumulator
            acc.apply_event(event)

            # 再照你原本邏輯轉成 chunk（這邊你已經寫好了）
            chunk = self._event_to_chunk(event, run_manager=run_manager)
            if chunk is not None:
                yield chunk

        # stream 結束時，補一個 meta chunk（包含 usage + 各 channel 完整輸出 / duration）
        if acc.has_content():
            meta = acc.build_meta()
            
            # 從 accumulator 的 usage 中提取 usage_metadata，供 Langfuse 追蹤
            usage_metadata: Optional[Dict[str, Any]] = None
            usage_dict = meta.get("usage")
            if usage_dict and isinstance(usage_dict, dict):
                usage_metadata = {
                    "input_tokens": usage_dict.get("input_tokens", 0),
                    "output_tokens": usage_dict.get("output_tokens", 0),
                    "total_tokens": usage_dict.get("total_tokens", 0),
                }
            
            meta_chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    additional_kwargs={
                        "channel": "meta",
                        "responses_meta": meta,
                    },
                    usage_metadata=usage_metadata,  # type: ignore[arg-type]
                )
            )
            yield meta_chunk

    # -------- _astream：非同步版本的 streaming --------

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """非同步版本的 _stream，使用 AsyncOpenAI 客戶端。"""
        # streaming=False → 使用 _agenerate 然後一次性回傳
        if not self.streaming:
            result = await self._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            full_msg: AIMessage = result.generations[0].message  # type: ignore[assignment]
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=full_msg.content,
                    additional_kwargs=full_msg.additional_kwargs,
                    response_metadata=full_msg.response_metadata,
                    usage_metadata=getattr(full_msg, "usage_metadata", None),  # type: ignore[arg-type]
                )
            )
            return

        # streaming=True → 真正用 Responses SSE (async)
        client = self._get_async_client()
        body = self._build_body(messages, stop=stop, **kwargs)
        body["stream"] = True

        stream = await client.responses.create(**body)

        acc = ResponsesAccumulator()

        async for event in stream:
            # 先丟進 accumulator
            acc.apply_event(event)

            # 再照原本邏輯轉成 chunk
            chunk = self._event_to_chunk(event, run_manager=None)
            if chunk is not None:
                # 非同步 callback
                if run_manager is not None:
                    text_delta = chunk.message.additional_kwargs.get("delta", "")
                    if text_delta:
                        await run_manager.on_llm_new_token(text_delta)
                yield chunk

        # stream 結束時，補一個 meta chunk
        if acc.has_content():
            meta = acc.build_meta()

            # 從 accumulator 的 usage 中提取 usage_metadata
            usage_metadata: Optional[Dict[str, Any]] = None
            usage_dict = meta.get("usage")
            if usage_dict and isinstance(usage_dict, dict):
                usage_metadata = {
                    "input_tokens": usage_dict.get("input_tokens", 0),
                    "output_tokens": usage_dict.get("output_tokens", 0),
                    "total_tokens": usage_dict.get("total_tokens", 0),
                }

            meta_chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    additional_kwargs={
                        "channel": "meta",
                        "responses_meta": meta,
                    },
                    usage_metadata=usage_metadata,  # type: ignore[arg-type]
                )
            )
            yield meta_chunk
