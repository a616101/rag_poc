"""
Langfuse trace 統一管理模組。

此模組提供統一的 trace context 管理，取代分散在各節點的 callback 配置。
使用此模組可確保：
1. 整個 workflow 共用同一個 trace_id（可預先產生）
2. 流程結束後可取得 trace_id 回傳前端
3. 支援 user feedback / custom scores API
4. Session view 可自動聚合同一用戶的 trace
"""

from contextlib import contextmanager
from typing import Optional, Generator
from dataclasses import dataclass

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler


@dataclass
class TraceContext:
    """Trace 執行結果，包含 trace_id 供後續 feedback 使用。"""

    trace_id: str
    handler: CallbackHandler


@contextmanager
def create_trace_context(
    *,
    name: str = "ask-stream-workflow",
    trace_id_seed: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    input: Optional[dict] = None,
) -> Generator[TraceContext, None, None]:
    """
    建立統一的 Langfuse trace context。

    使用方式：
        with create_trace_context(
            trace_id_seed="request_12345",
            user_id="user-123",
            input={"question": q}
        ) as ctx:
            result = graph.stream(state, config={"callbacks": [ctx.handler]})

        # 流程結束後可取得 trace_id 回傳前端
        return {"trace_id": ctx.trace_id}

    參數：
        name: trace 名稱，預設為 "ask-stream-workflow"
        trace_id_seed: 用於產生 deterministic trace_id 的種子（如 request_id）
        user_id: 用戶 ID，用於 Langfuse 用戶分析
        session_id: 會話 ID，用於聚合同一會話的 trace
        tags: 標籤列表，用於 Langfuse 過濾
        metadata: 額外元數據
        input: trace 輸入資料，會顯示在 Langfuse UI

    Yields：
        TraceContext: 包含 trace_id 和 handler 的上下文物件
    """
    langfuse = get_client()

    # 預先產生 deterministic trace_id
    if trace_id_seed:
        trace_id = langfuse.create_trace_id(seed=trace_id_seed)
    else:
        trace_id = langfuse.create_trace_id()

    # 建立 handler，綁定到此 trace
    handler = CallbackHandler()

    with langfuse.start_as_current_observation(
        as_type="span",
        name=name,
        metadata=metadata,
        trace_context={"trace_id": trace_id},
    ) as root_span:
        # 設定 trace 層級的 input 和 tags
        trace_updates: dict = {}
        if input is not None:
            trace_updates["input"] = input
        if tags:
            trace_updates["tags"] = tags
        if trace_updates:
            langfuse.update_current_trace(**trace_updates)

        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
        ):
            yield TraceContext(
                trace_id=trace_id,
                handler=handler,
            )
